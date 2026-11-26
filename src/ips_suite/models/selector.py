"""SelectorNet: common base of IPS and MS-IPS ips()     : exploration without gradient, by groups of I candidates. classify(): exploitation with gradient on the retained patches."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ips_suite.data.tiling import TileSpec, extract_patches
from ips_suite.models.attention import Transformer, TransformerSA
from ips_suite.models.encoder import PatchEncoder, ScaleEmbedding


class SelectorNet(nn.Module):

    def __init__(self, conf, device):
        super().__init__()
        self.device  = device
        self.conf    = conf
        self.M       = conf.M
        self.M_bag   = conf.M          # number of patches given to the classifier
        self.I       = conf.I
        self.D       = conf.D
        self.n_class = conf.n_class
        self.shuffle = bool(conf.get('shuffle', True))
        self.scales  = list(conf.scales)
        self.loading = conf.get('loading', 'eager')
        self.aggregator = conf.get('aggregator', 'cross')     # 'cross' | 'self'
        self.selection = conf.get('selection', 'attention')   # 'attention' | 'random'

        self.spec = TileSpec(conf.img_size, self.scales, device)

        enc_px = conf.get('encode_px', None)
        if enc_px is None:
            enc_px = [None] * len(self.scales)
        if len(enc_px) != len(self.scales):
            raise ValueError("encode_px must have one entry per scale")
        self.encode_px = [None if not e else int(e) for e in enc_px]
        pv = conf.get('pad_value', None) or [0.0] * conf.n_chan_in
        self.register_buffer('pad_val', torch.tensor(pv, dtype=torch.float32),
                             persistent=False)         # not in the weights file

        self.encoder = PatchEncoder(
            n_chan_in=conf.n_chan_in,
            pretrained=conf.pretrained,
            n_res_blocks=conf.n_res_blocks)

        self.projector = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim, eps=1e-5, elementwise_affine=False),
            nn.Linear(self.encoder.out_dim, conf.D),
            nn.BatchNorm1d(conf.D),
            nn.ReLU())

        self.use_scale_emb = (len(self.scales) > 1
                              and bool(conf.get('use_scale_emb', True)))
        self.scale_emb = (ScaleEmbedding(conf.D, len(self.scales))
                          if self.use_scale_emb else None)

        if self.aggregator == 'self':
            self.transf = TransformerSA(
                conf.H, conf.D, conf.D_k, conf.D_v,
                conf.D_inner, conf.attn_dropout, conf.dropout)
        else:
            self.transf = Transformer(
                conf.n_token, conf.H, conf.D, conf.D_k, conf.D_v,
                conf.D_inner, conf.attn_dropout, conf.dropout)

        self.head = nn.Linear(conf.D, conf.n_class)

        # Own generator for the shuffle of candidates: reseeded before each
        # evaluation pass, so that an evaluation is reproducible.
        self._gen = torch.Generator()
        self._gen.manual_seed(int(conf.seed))

    def reseed(self, seed):
        self._gen.manual_seed(int(seed))

    def encoded_size(self, sid):
        """Side of the patch that the encoder actually sees for scale sid."""
        return self.encode_px[sid] or self.spec.sizes[sid]

    # ------------------------------------------------------------ pixels
    def patches(self, img, coords2, sid):
        """(n,2) coordinates of scale sid -> (n,C,e,e) on the model device."""
        p = self.spec.sizes[sid]
        x = extract_patches(img, coords2, p, self.pad_val)
        if x.device != self.device:
            x = x.to(self.device, non_blocking=True)
        e = self.encode_px[sid]
        if e is not None and e != p:
            x = F.interpolate(x, size=(e, e), mode='bilinear',
                              align_corners=False)
        return x

    # ---------------------------------------------------------- encoding
    def embed_coords(self, img, coords):
        """(n,3) coordinates -> (n,D). One encoder call per scale present.

        The output order follows `coords`, so the random order that breaks
        ties between equal scores is preserved.
        """
        n = coords.shape[0]
        out = None
        for sid in range(len(self.spec.sizes)):
            mask = coords[:, 0] == sid
            if not bool(mask.any()):
                continue
            sub   = coords[mask]
            feats = self.projector(self.encoder(self.patches(img, sub[:, 1:], sid)))
            if self.scale_emb is not None:
                feats = self.scale_emb(feats, sub[:, 0])
            if out is None:
                out = torch.zeros(n, self.D, device=feats.device,
                                  dtype=feats.dtype)
            out = out.index_put((torch.nonzero(mask, as_tuple=True)[0],), feats)
        if out is None:
            raise RuntimeError("no candidate to encode")
        return out

    # ------------------------------------------------------- exploration
    @torch.no_grad()
    def select(self, img, candidates, M):
        """Top-M among `candidates`, by groups of I. Returns (M,3), best first."""
        N = int(candidates.shape[0])
        if M >= N:
            return candidates

        was_training = self.training
        if was_training:
            self.encoder.eval(); self.projector.eval(); self.transf.eval()
            if self.scale_emb is not None:
                self.scale_emb.eval()

        cand = candidates
        if self.shuffle:
            perm = torch.randperm(N, generator=self._gen).to(cand.device)
            cand = cand[perm]

        mem_coord = cand[:M]
        mem_emb   = self.embed_coords(img, mem_coord)

        n_iter = math.ceil((N - M) / self.I)
        for i in range(n_iter):
            s = i * self.I + M
            e = min(s + self.I, N)
            it_coord = cand[s:e]
            it_emb   = self.embed_coords(img, it_coord)

            all_emb   = torch.cat([mem_emb, it_emb], dim=0)
            all_coord = torch.cat([mem_coord, it_coord], dim=0)

            scores = self.transf.get_scores(all_emb.unsqueeze(0)).squeeze(0)
            top    = torch.topk(scores, M, dim=-1)[1]

            mem_emb   = all_emb[top]
            mem_coord = all_coord[top]

        if was_training:
            self.encoder.train(); self.projector.train(); self.transf.train()
            if self.scale_emb is not None:
                self.scale_emb.train()
        return mem_coord

    @torch.no_grad()
    def ips(self, img, return_stages=False):
        """Selection for one image (C,H,W)."""
        if self.selection == 'random':        # ablation: no learned selection
            keep = self.sample_baseline('uniform')
        else:
            keep = self.select(img, self.spec.candidates, self.M)
        return (keep, [keep]) if return_stages else keep

    @torch.no_grad()
    def score_map(self, img):
        """Attention logit of EVERY candidate (every size, every position), in
        the order of spec.candidates, for heatmaps. None when the aggregator
        gives no group-independent score (self-attention) or the selection is
        random."""
        if self.selection == 'random' or self.aggregator == 'self':
            return None
        was = self.training
        self.eval()
        C, out = self.spec.candidates, []
        for s in range(0, int(C.shape[0]), self.I):
            emb = self.embed_coords(img, C[s:s + self.I])
            out.append(self.transf.get_logits(emb.unsqueeze(0))[0])
        self.train(was)
        return torch.cat(out).float().cpu().numpy()

    # ------------------------------------------------ reference samplings
    @torch.no_grad()
    def sample_baseline(self, mode, coords_ref=None):
        """Selection without learning.

        'uniform'       : M candidates drawn uniformly among the N.
        'scale_matched' : same count per scale as `coords_ref`, random
                          positions inside each scale.
        uniform -> scale_matched measures the choice of scales,
        scale_matched -> top-M the choice of positions.
        """
        pool = self.spec.candidates
        N = int(pool.shape[0])
        if mode == 'uniform':
            idx = torch.randperm(N, generator=self._gen)[:self.M]
            return pool[idx.to(pool.device)]
        if mode == 'scale_matched':
            if coords_ref is None:
                raise ValueError("scale_matched needs coords_ref")
            out = []
            for sid in range(len(self.spec.sizes)):
                k = int((coords_ref[:, 0] == sid).sum())
                if k == 0:
                    continue
                sub = pool[pool[:, 0] == sid]
                pick = torch.randperm(int(sub.shape[0]), generator=self._gen)[:k]
                out.append(sub[pick.to(pool.device)])
            return torch.cat(out, dim=0)
        raise ValueError(f"unknown mode : {mode}")

    # ------------------------------------------------------ exploitation
    def classify(self, emb):
        """emb (B,M_bag,D) -> logits (B,n_class)."""
        image_emb = self.transf(emb)          # (B, n_token, D)
        return self.head(image_emb[:, 0])

    # ----------------------------------------------------------- costs
    @property
    def explore_pixels(self):
        """Pixels read from the image by one exploration pass."""
        return self.spec.pixels_per_pass

    @property
    def explore_encoded_pixels(self):
        """Pixels that enter the encoder during one exploration pass."""
        return sum(n * self.encoded_size(sid) ** 2
                   for sid, n in enumerate(self.spec.n_per_scale))

    @property
    def n_candidates(self):
        return self.spec.N
