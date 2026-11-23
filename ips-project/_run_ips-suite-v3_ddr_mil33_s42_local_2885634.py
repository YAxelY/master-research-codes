#!/usr/bin/env python
# coding: utf-8

# # IPS suite v3
# 
# Same four models as `ips-suite.ipynb` (v2): full-image **baseline**,
# single-scale **IPS**, flat multi-scale **MS-IPS**, coarse-to-fine
# **H-MS-IPS**. The model code is unchanged, so v2 weights load in v3.
# 
# What v3 adds is measurement, not a new model:
# 
# 1. **Held-out test set.** The epoch is chosen on validation, the score is
#    reported on test. DDR uses its own `test` split. RSNA is split
#    70 / 15 / 15 by patient.
# 2. **Seeds.** The seed is part of the run name. The data split never
#    depends on the seed.
# 3. **Everything is saved per image**: probabilities and selected patches on
#    validation and test, so any metric, threshold or confidence interval can
#    be computed later without retraining.
# 4. **Time and memory are split by phase**: data loading, selection (no
#    gradient), training step. Peak memory is recorded for each phase.
# 5. **Lesion overlap** of the selected patches against exact chance, on the
#    annotated test images (DDR masks, RSNA boxes).
# 6. **Controls and baselines**: IPS at every patch size used by a cascade,
#    values of `M`, lower-resolution images, attention MIL over all patches,
#    the selection module (cross-attention, self-attention, random), coarse
#    zones in the training bag, resampled patches, lazy loading, low-data
#    training. Two patch sizes at most per model.
# 7. **A data-free benchmark** (`IPS_MODE=bench`) of memory and latency
#    against `M`, `I`, patch size, number of candidates and loading mode.
# 
# Selection of what runs: environment variables, see the last cell.

# ## Settings: the only cell to edit on Kaggle
# 
# Nothing else in the notebook needs to be changed. The values below are a
# quick dry run: one configuration, one epoch, 200 images per split. On the
# cluster this cell does nothing, because `run-ips-suite-v3.sh` sets the
# variables itself.
# 
# | To do this | Change this |
# |---|---|
# | run another configuration | `IPS_RUNS` (a name, several names, or a group: `core`, `hires`, `mstudy`, ...) |
# | a real epoch on all images | `IPS_SMOKE` to `'0'` |
# | a full training | `IPS_EPOCHS` to `'50'` and `IPS_SMOKE` to `'0'` |
# | memory and speed test, no data | `IPS_MODE` to `'bench'` |
# | print the list of all configurations | `IPS_MODE` to `'list'` |

# In[ ]:


import os

ON_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if ON_KAGGLE:
    os.environ['IPS_MODE']    = 'train'       # train | bench | eval_v2 | list
    # dry run: one model of each kind (IPS, MIL, self-attention, cascade, RSNA)
    # dry run of the 18 November focus: padded sizes, MIL, cascades, both data sets
    os.environ['IPS_RUNS']    = ('ddr_mil33,ddr_ms66_33,ddr_h66_33,'
                                 'rsna_ips300,rsna_h300_150,rsna_mil300')
    os.environ['IPS_CACHE_DIR'] = '/gpfs01/berens/user/kdjoumessi/Workspace/Berens-lab/CIN/Projects/New/MS-IPS/master-research-codes/ips-project/outputs/cache'   # tests the image cache too
    os.environ['IPS_SEEDS']   = '42'
    os.environ['IPS_EPOCHS']  = '1'           # 50 for a full training
    os.environ['IPS_SMOKE']   = '200'         # images kept per split; '0' = all images
    os.environ['IPS_WORKERS'] = '3'           # Kaggle gives 4 CPU cores
    os.environ['IPS_DDR_GEOMETRY'] = 'retina'    # or 'stretch' (the v2 input)
print({k: v for k, v in os.environ.items() if k.startswith('IPS_')})

# In[ ]:


# ============================================================
# Imports, device, seeds, environment helpers
# ============================================================
import os, sys, math, json, time, random, hashlib, fcntl, contextlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score,
                             precision_score, recall_score, cohen_kappa_score,
                             average_precision_score, confusion_matrix,
                             balanced_accuracy_score, roc_curve)

SUITE_VERSION = 'v3'
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA = torch.cuda.is_available()


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def get(self, k, default=None):
        return self.__dict__.get(k, default)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _env(key, default=''):
    # an empty variable counts as unset, so the default applies
    return os.environ.get(key, '').strip() or default


def _env_int(key, default):
    v = _env(key)
    return int(v) if v else default


def _env_path(key, default):
    return Path(os.environ.get(key, str(default)))


print(f"IPS suite {SUITE_VERSION} | PyTorch {torch.__version__} | device {device}")
if CUDA:
    print(f"GPU {torch.cuda.get_device_name(0)} | "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# In[ ]:


# ============================================================
# Learning-rate schedule, identical to the original IPS code
# ============================================================
def adjust_learning_rate(n_epoch_warmup, n_epoch, max_lr, optimizer, n_step_epoch, step):
    max_steps    = int(n_epoch * n_step_epoch)
    warmup_steps = int(n_epoch_warmup * n_step_epoch)
    if step < warmup_steps:
        lr = max_lr * step / max(warmup_steps, 1)
    else:
        s = step - warmup_steps
        m = max(max_steps - warmup_steps, 1)
        q = 0.5 * (1 + math.cos(math.pi * s / m))
        end_lr = max_lr * 0.001
        lr = max_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr
    return lr

# In[ ]:


# ============================================================
# Geometry: exact tiling and vectorised extraction
#
# A candidate is a triple (scale id, row, column). The candidate table holds
# no pixel: N * 3 integers. Pixels are read only for the candidates asked for.
# ============================================================
class TileSpec:
    """Tiling of a square image by several patch sizes.

    A size that does not divide the image gets one more row and column of
    patches; they overflow the image and their missing part is filled with
    black (padding). 896 px: 66 px -> 14 x 14 patches over 924 px, 33 px ->
    28 x 28 over 924 px. 1024 px: 300 px -> 4 x 4 over 1200, 150 -> 7 x 7 over 1050.
    """

    def __init__(self, img_size, sizes, device):
        self.img_size = int(img_size)
        self.sizes    = [int(s) for s in sizes]
        rows = []
        for sid, p in enumerate(self.sizes):
            k = -(-self.img_size // p)                 # ceil: padded tiling
            r, c = np.divmod(np.arange(k * k), k)
            rows.append(np.stack([np.full(k * k, sid), r * p, c * p], 1))
        cand = np.concatenate(rows, 0).astype(np.int64)
        self.candidates = torch.from_numpy(cand).to(device)   # (N, 3)
        self.k_per_scale = [-(-self.img_size // p) for p in self.sizes]
        self.n_per_scale = [k * k for k in self.k_per_scale]
        self.N = int(self.candidates.shape[0])
        # pixels encoded by one full exploration pass: one (padded) image per scale
        self.pixels_per_pass = sum(n * p * p for n, p in zip(self.n_per_scale, self.sizes))
        self.padded = any(self.img_size % p for p in self.sizes)

    def __repr__(self):
        parts = ', '.join(f"{n}x{p}px" for n, p in
                          zip(self.n_per_scale, self.sizes))
        return (f"TileSpec({self.img_size}^2 : {parts} = {self.N} candidates, "
                f"{self.pixels_per_pass/1e6:.2f} M pixels/pass)")


def extract_patches(img, coords, p, pad=None):
    """img (C,H,W) -> (n,C,p,p). coords (n,2) = (row, column).

    A patch that overflows the image is completed with `pad` (one value per
    channel; zeros if None). The result lives on the device of `img`; with
    lazy loading the image is on the CPU and only the patches move to the GPU.
    """
    if coords.device != img.device:
        coords = coords.to(img.device)
    n = coords.shape[0]
    C, H, W = img.shape
    ar = torch.arange(p, device=img.device)
    rr = coords[:, 0].view(n, 1, 1) + ar.view(1, p, 1)
    cc = coords[:, 1].view(n, 1, 1) + ar.view(1, 1, p)
    if n == 0 or (int(coords[:, 0].max()) + p <= H and int(coords[:, 1].max()) + p <= W):
        return img[:, rr, cc].permute(1, 0, 2, 3).contiguous()
    inside = ((rr < H) & (cc < W)).unsqueeze(1)                       # (n,1,p,p)
    x = img[:, rr.clamp(max=H - 1), cc.clamp(max=W - 1)].permute(1, 0, 2, 3)
    fill = (torch.zeros(C) if pad is None else pad).to(img.device, img.dtype).view(1, C, 1, 1)
    return torch.where(inside, x, fill).contiguous()


def _selftest_geometry():
    spec = TileSpec(896, [128, 64, 32], torch.device('cpu'))
    assert spec.n_per_scale == [49, 196, 784] and spec.N == 1029, spec
    img = torch.rand(3, 896, 896)
    C = spec.candidates
    for sid, p in enumerate(spec.sizes):
        sub = C[C[:, 0] == sid][:7, 1:]
        fast = extract_patches(img, sub, p)
        slow = torch.stack([img[:, int(r):int(r) + p, int(c):int(c) + p]
                            for r, c in sub])
        assert torch.equal(fast, slow), f"extraction {p}px differs"
    # Proposition 1: every exactly tiled scale costs one image in pixels
    for p, n in zip(spec.sizes, spec.n_per_scale):
        assert n * p * p == 896 * 896
    # padded tiling: 66 px on 896 px, last patch completed with the pad value
    spec = TileSpec(896, [66, 33], torch.device('cpu'))
    assert spec.n_per_scale == [196, 784] and spec.padded, spec
    pad = torch.tensor([-1.0, -2.0, -3.0])
    last = spec.candidates[(spec.candidates[:, 0] == 0)][-1:, 1:]       # row 858, col 858
    x = extract_patches(img, last, 66, pad)[0]
    assert torch.equal(x[:, :38, :38], img[:, 858:, 858:])
    assert bool((x[0, 38:, :] == -1).all()) and bool((x[2, :, 38:] == -3).all())
    print("  [selftest] geometry, extraction, padding, proposition 1 : OK")

_selftest_geometry()

# In[ ]:


# ============================================================
# Patch encoder: truncated ResNet-18, as in the original ips_net.py
#
# n_res_blocks = 4 : cut after layer4, downsampling /32, output 512
# n_res_blocks = 2 : cut after layer2, downsampling /8,  output 128
# ============================================================
class PatchEncoder(nn.Module):

    def __init__(self, n_chan_in=3, pretrained=True, n_res_blocks=4):
        super().__init__()
        if n_res_blocks not in (2, 4):
            raise ValueError("n_res_blocks must be 2 or 4")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = resnet18(weights=weights)

        if n_chan_in != 3:
            old = net.conv1
            new = nn.Conv2d(n_chan_in, 64, kernel_size=7, stride=2,
                            padding=3, bias=False)
            if pretrained:
                with torch.no_grad():
                    mean_w = old.weight.mean(dim=1, keepdim=True)
                    new.weight.copy_(mean_w.repeat(1, n_chan_in, 1, 1))
            net.conv1 = new

        layers = [net.conv1, net.bn1, net.relu, net.maxpool,
                  net.layer1, net.layer2]
        self.out_dim = 128
        if n_res_blocks == 4:
            layers += [net.layer3, net.layer4]
            self.out_dim = 512
        layers.append(net.avgpool)
        self.net = nn.Sequential(*layers)
        self.downsample = 8 if n_res_blocks == 2 else 32

    def forward(self, x):
        return torch.flatten(self.net(x), 1)

    def feature_map_size(self, patch_size):
        """Exact side of the map before average pooling (1 = nothing to pool)."""
        h = (patch_size + 2 * 3 - 7) // 2 + 1      # conv1  k7 s2 p3
        h = (h + 2 * 1 - 3) // 2 + 1               # maxpool k3 s2 p1
        h = (h + 2 * 1 - 3) // 2 + 1               # layer2 s2 (layer1 s1)
        if self.downsample == 32:
            h = (h + 2 * 1 - 3) // 2 + 1           # layer3 s2
            h = (h + 2 * 1 - 3) // 2 + 1           # layer4 s2
        return max(1, h)


class ScaleEmbedding(nn.Module):
    """One learned vector per scale, added to the projected features."""

    def __init__(self, D, n_scales):
        super().__init__()
        self.emb = nn.Embedding(n_scales, D)
        nn.init.normal_(self.emb.weight, std=0.02)

    def forward(self, feats, scale_ids):
        return feats + self.emb(scale_ids)

# In[ ]:


# ============================================================
# Cross-attention transformer, from architecture/transformer.py
# (Bergner et al., ICLR 2023). Unchanged.
# ============================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def compute_attn(self, q, k):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))
        return self.dropout(torch.softmax(attn, dim=-1))

    def forward(self, q, k, v):
        return torch.matmul(self.compute_attn(q, k), v)


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_token, H, D, D_k, D_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.n_token, self.H, self.D_k, self.D_v = n_token, H, D_k, D_v
        self.q = nn.Parameter(torch.empty((1, n_token, D)))
        v = math.sqrt(1 / D_k)
        nn.init.uniform_(self.q, a=-v, b=v)
        self.q_w = nn.Linear(D, H * D_k, bias=False)
        self.k_w = nn.Linear(D, H * D_k, bias=False)
        self.v_w = nn.Linear(D, H * D_v, bias=False)
        self.fc  = nn.Linear(H * D_v, D, bias=False)
        self.attention  = ScaledDotProductAttention(D_k ** 0.5, attn_dropout)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)

    def get_attn(self, x):
        B, len_seq = x.shape[:2]
        q = self.q_w(self.q).view(1, self.n_token, self.H, self.D_k)
        k = self.k_w(x).view(B, len_seq, self.H, self.D_k)
        return self.attention.compute_attn(q.transpose(1, 2), k.transpose(1, 2))

    def get_logits(self, x):
        """Attention logits BEFORE the softmax, mean over heads and tokens: (B, L).

        The softmax weights of a patch depend on the other patches of its
        group; the logits do not, so they can be compared across groups and
        drawn as a heatmap over the whole image.
        """
        B, len_seq = x.shape[:2]
        q = self.q_w(self.q).view(1, self.n_token, self.H, self.D_k).transpose(1, 2)
        k = self.k_w(x).view(B, len_seq, self.H, self.D_k).transpose(1, 2)
        a = torch.matmul(q / self.attention.temperature, k.transpose(2, 3))
        return a.mean(dim=(1, 2))

    def forward(self, x):
        B, len_seq = x.shape[:2]
        q = self.q_w(self.q).view(1, self.n_token, self.H, self.D_k)
        k = self.k_w(x).view(B, len_seq, self.H, self.D_k)
        v = self.v_w(x).view(B, len_seq, self.H, self.D_v)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x = self.attention(q, k, v)
        x = x.transpose(1, 2).contiguous().view(B, self.n_token, -1)
        x = self.dropout(self.fc(x))
        x = x + self.q
        return self.layer_norm(x)


class MLP(nn.Module):
    def __init__(self, D, D_inner, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(D, D_inner)
        self.w_2 = nn.Linear(D_inner, D)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.dropout(self.w_2(torch.relu(self.w_1(x))))
        return self.layer_norm(x + residual)


class Transformer(nn.Module):
    def __init__(self, n_token, H, D, D_k, D_v, D_inner,
                 attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.crs_attn = MultiHeadCrossAttention(
            n_token, H, D, D_k, D_v, attn_dropout, dropout)
        self.mlp = MLP(D, D_inner, dropout)

    def get_scores(self, x):
        """Attention averaged over heads, then over tokens."""
        attn = self.crs_attn.get_attn(x)
        return attn.mean(dim=1).transpose(1, 2).mean(-1)

    def get_logits(self, x):
        return self.crs_attn.get_logits(x)

    def forward(self, x):
        return self.mlp(self.crs_attn(x))


# ------------------------------------------------------------------
# Self-attention aggregator, as in ips-self-attention.ipynb.
# Patches attend to each other; the score of a patch is the attention it
# RECEIVES (column sum of the attention matrix, mean over heads); the image
# token is the mean of the output tokens. Used only by the ablation runs.
# ------------------------------------------------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, H, D, D_k, D_v, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.H, self.D_k, self.D_v = H, D_k, D_v
        self.q_w = nn.Linear(D, H * D_k, bias=False)
        self.k_w = nn.Linear(D, H * D_k, bias=False)
        self.v_w = nn.Linear(D, H * D_v, bias=False)
        self.fc  = nn.Linear(H * D_v, D, bias=False)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(D, eps=1e-6)

    def forward(self, x):
        B, L, _ = x.shape
        q = self.q_w(x).view(B, L, self.H, self.D_k).transpose(1, 2)
        k = self.k_w(x).view(B, L, self.H, self.D_k).transpose(1, 2)
        v = self.v_w(x).view(B, L, self.H, self.D_v).transpose(1, 2)
        attn = torch.softmax(torch.matmul(q / self.D_k ** 0.5, k.transpose(2, 3)), dim=-1)
        out = torch.matmul(self.attn_dropout(attn), v)
        out = out.transpose(1, 2).contiguous().view(B, L, self.H * self.D_v)
        out = self.layer_norm(self.dropout(self.fc(out)) + x)
        return out, attn                                  # attn: (B, H, L, L)


class TransformerSA(nn.Module):
    def __init__(self, H, D, D_k, D_v, D_inner, attn_dropout=0.1, dropout=0.1):
        super().__init__()
        self.sa  = MultiHeadSelfAttention(H, D, D_k, D_v, attn_dropout, dropout)
        self.mlp = MLP(D, D_inner, dropout)

    def get_scores(self, x):
        _, attn = self.sa(x)
        return attn.mean(dim=1).sum(dim=1)                # (B, L)

    def get_logits(self, x):
        return None                                       # depends on the group

    def forward(self, x):
        out, _ = self.sa(x)
        return self.mlp(out).mean(dim=1, keepdim=True)    # (B, 1, D)

# ## Models
# 
# `SelectorNet` carries the shared logic. IPS is the case of one scale,
# MS-IPS of two or three, H-MS-IPS adds coarse stages before the fine one.
# 
# Three switches are new in v3. All default to the v2 behaviour.
# 
# * `loading = 'lazy'` keeps the image on the CPU: only the patches of the
#   current group reach the GPU.
# * `encode_px = [..]` resizes the patches of a scale before the encoder
#   (`None` = native size). Used for the resampling control and for a
#   low-resolution coarse stage.
# * `train_coarse = k` (H-MS-IPS) appends the `k` best coarse zones to the
#   training bag, so the coarse stage receives gradients.

# In[ ]:


# ============================================================
# SelectorNet: common base of IPS and MS-IPS
#
# ips()     : exploration without gradient, by groups of I candidates.
# classify(): exploitation with gradient on the retained patches.
# ============================================================
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

# In[ ]:


# ============================================================
# H-MS-IPS: cascade of stages, from the coarsest to the finest
#
#   conf.cascade      = [224, 56]   sizes, coarse to fine
#   conf.M_keep       = [4]         how many to keep at each stage but the last
#   conf.M            = 10          final budget
#   conf.train_coarse = 0           v2 behaviour: only the final patches train
#                     = k           the k best coarse zones join the training bag
#
# The retained zones have a fixed area, so cutting them finer multiplies the
# pool of the next stage without changing the number of pixels encoded.
# ============================================================
class HMSIPSNet(SelectorNet):

    def __init__(self, conf, device):
        cascade = [int(s) for s in conf.cascade]
        M_keep = [int(m) for m in conf.M_keep]
        if len(cascade) < 2:
            raise ValueError("cascade needs at least two stages")
        if len(M_keep) != len(cascade) - 1:
            raise ValueError(f"M_keep needs {len(cascade)-1} values, "
                             f"got {len(M_keep)}")
        for a, b in zip(cascade, cascade[1:]):
            if a % b != 0 or b >= a:
                raise ValueError(f"stage {a} -> {b} : ratio must be an integer > 1")

        inner = Struct(**{**conf.__dict__, 'scales': cascade})
        super().__init__(inner, device)

        self.cascade = cascade
        self.M_keep = M_keep
        self.n_stage = len(cascade)
        self.train_coarse = int(conf.get('train_coarse', 0) or 0)
        if self.train_coarse > M_keep[0]:
            raise ValueError("train_coarse cannot exceed M_keep[0]")
        self.M_bag = self.M + self.train_coarse

        # stage 0: full tiling of the image by cascade[0]
        k0 = -(-conf.img_size // cascade[0])         # ceil: padded zones
        self.n_zones = k0 * k0
        zr, zc = np.divmod(np.arange(self.n_zones), k0)
        self.zone_coords = torch.from_numpy(
            np.stack([np.zeros(self.n_zones, np.int64),
                      zr * cascade[0], zc * cascade[0]], 1)).to(device)

        # offsets from a parent to its children, one set per transition
        self.offsets = []
        for a, b in zip(cascade, cascade[1:]):
            r = a // b
            fr, fc = np.divmod(np.arange(r * r), r)
            self.offsets.append(torch.from_numpy(
                np.stack([fr * b, fc * b], 1).astype(np.int64)).to(device))

        self.pool_sizes = [self.n_zones]
        for k, off in enumerate(self.offsets):
            self.pool_sizes.append(M_keep[k] * int(off.shape[0]))
        if self.M > self.pool_sizes[-1]:
            raise ValueError(f"M={self.M} exceeds the final pool "
                             f"({self.pool_sizes[-1]} candidates)")

    def _pick(self, img, cand, k):
        """Keep k candidates: learned (attention) or, for the ablation, at random."""
        if self.selection == 'random':
            if k >= int(cand.shape[0]):
                return cand
            idx = torch.randperm(int(cand.shape[0]), generator=self._gen)[:k]
            return cand[idx.to(cand.device)]
        return self.select(img, cand, k)

    def _subdivide(self, parents, stage):
        """parents (k,3) of `stage` -> children (k*r^2, 3) of stage+1."""
        off = self.offsets[stage]
        r2 = int(off.shape[0])
        base = parents[:, 1:].repeat_interleave(r2, dim=0)
        shift = off.repeat(int(parents.shape[0]), 1)
        sid = torch.full((base.shape[0], 1), stage + 1,
                         dtype=torch.int64, device=base.device)
        return torch.cat([sid, base + shift], dim=1)

    @torch.no_grad()
    def ips(self, img, return_stages=False):
        keep = self._pick(img, self.zone_coords, self.M_keep[0])
        stages = [keep]
        for stage in range(self.n_stage - 1):
            child = self._subdivide(keep, stage)
            budget = (self.M if stage == self.n_stage - 2
                      else self.M_keep[stage + 1])
            keep = self._pick(img, child, budget)
            stages.append(keep)
        out = keep
        if self.train_coarse > 0:
            out = torch.cat([keep, stages[0][:self.train_coarse]], dim=0)
        return (out, stages) if return_stages else out

    @property
    def explore_pixels(self):
        """Whole (padded) image at stage 0, then the retained zones."""
        px = self.n_zones * self.cascade[0] ** 2
        for k, m in enumerate(self.M_keep):
            px += m * self.cascade[k] ** 2
        return px

    @property
    def explore_encoded_pixels(self):
        px = self.n_zones * self.encoded_size(0) ** 2
        for k in range(1, self.n_stage):
            px += self.pool_sizes[k] * self.encoded_size(k) ** 2
        return px

    @property
    def n_candidates(self):
        return sum(self.pool_sizes)

# In[ ]:


# ============================================================
# Baseline: end-to-end ResNet-18 on the (possibly downsampled) image.
# Same backbone as the patch models: only the selection differs.
# ============================================================
class BaselineNet(nn.Module):

    def __init__(self, conf, device):
        super().__init__()
        self.device = device
        self.conf   = conf
        self.input_size = int(conf.get('baseline_size', conf.img_size))

        weights = ResNet18_Weights.IMAGENET1K_V1 if conf.pretrained else None
        net = resnet18(weights=weights)
        if conf.n_chan_in != 3:
            old = net.conv1
            new = nn.Conv2d(conf.n_chan_in, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                new.weight.copy_(
                    old.weight.mean(dim=1, keepdim=True).repeat(1, conf.n_chan_in, 1, 1))
            net.conv1 = new
        net.fc = nn.Linear(512, conf.n_class)
        self.net = net

    def forward(self, imgs):
        if imgs.shape[-1] != self.input_size:
            imgs = F.interpolate(imgs, size=(self.input_size, self.input_size),
                                 mode='bilinear', align_corners=False,
                                 antialias=True)
        return self.net(imgs)


# ============================================================
# Attention MIL (Ilse et al., 2018): the image is cut into ALL its patches of
# one size, every patch goes through the encoder WITH gradients, a gated
# attention pools them. No selection: this is what IPS saves memory against.
# Same encoder and projector as the patch models.
# ============================================================
class MILNet(nn.Module):
    is_mil = True

    def __init__(self, conf, device):
        super().__init__()
        self.device, self.conf = device, conf
        self.p = int(conf.scales[0])
        self.spec = TileSpec(conf.img_size, [self.p], device)
        self.M = self.M_bag = conf.M          # top-M by attention, for the analyses
        self.loading = 'eager'
        self.encoder = PatchEncoder(n_chan_in=conf.n_chan_in, pretrained=conf.pretrained,
                                    n_res_blocks=conf.n_res_blocks)
        self.projector = nn.Sequential(
            nn.LayerNorm(self.encoder.out_dim, eps=1e-5, elementwise_affine=False),
            nn.Linear(self.encoder.out_dim, conf.D),
            nn.BatchNorm1d(conf.D),
            nn.ReLU())
        Da = 128
        self.att_v = nn.Linear(conf.D, Da)
        self.att_u = nn.Linear(conf.D, Da)
        self.att_w = nn.Linear(Da, 1)
        self.head = nn.Linear(conf.D, conf.n_class)
        self.last_attn = self.last_logits = None
        pv = conf.get('pad_value', None) or [0.0] * conf.n_chan_in
        self.register_buffer('pad_val', torch.tensor(pv, dtype=torch.float32),
                             persistent=False)
        self._gen = torch.Generator()
        self._gen.manual_seed(int(conf.seed))

    def reseed(self, seed):
        self._gen.manual_seed(int(seed))

    def forward(self, imgs):
        B, C, S, _ = imgs.shape
        p, k = self.p, -(-S // self.p)
        if k * p != S:                                            # padded tiling
            canvas = self.pad_val.to(imgs.dtype).view(1, C, 1, 1).expand(
                B, C, k * p, k * p).clone()
            canvas[:, :, :S, :S] = imgs
            imgs = canvas
        x = imgs.unfold(2, p, p).unfold(3, p, p)                  # B,C,k,k,p,p
        x = x.permute(0, 2, 3, 1, 4, 5).reshape(B * k * k, C, p, p)
        h = self.projector(self.encoder(x)).view(B, k * k, -1)    # row-major = spec order
        a = self.att_w(torch.tanh(self.att_v(h)) * torch.sigmoid(self.att_u(h))).squeeze(-1)
        w = torch.softmax(a, dim=-1)
        self.last_logits, self.last_attn = a.detach(), w.detach()
        return self.head((w.unsqueeze(-1) * h).sum(dim=1))

    @torch.no_grad()
    def ips(self, img, return_stages=False):
        """Top-M patches by attention, so the lesion analysis applies to MIL too."""
        was = self.training
        self.eval()
        self.forward(img.unsqueeze(0))
        self.train(was)
        keep = self.spec.candidates[self.last_attn[0].topk(self.M).indices]
        return (keep, [keep]) if return_stages else keep


# In[ ]:


# ============================================================
# Exploitation by batch: one encoder call per scale, ONE projector call for
# the whole batch (the projector holds a BatchNorm1d).
# ============================================================
def _embed_from_store(net, store, B):
    """store[sid] = (list of patches, list of batch indices, list of slots)."""
    raws, bidx, sidx, sids = [], [], [], []
    for sid in range(len(net.spec.sizes)):
        pats, bb, ss = store[sid]
        if not pats:
            continue
        x = torch.cat(pats, dim=0)
        raws.append(net.encoder(x))
        bidx.append(torch.cat(bb))
        sidx.append(torch.cat(ss))
        sids.append(torch.full((x.shape[0],), sid, dtype=torch.long,
                               device=x.device))
    if not raws:
        raise RuntimeError("empty batch")
    feats = net.projector(torch.cat(raws, dim=0))
    if net.scale_emb is not None:
        feats = net.scale_emb(feats, torch.cat(sids))
    emb = torch.zeros(B, net.M_bag, net.D, device=feats.device, dtype=feats.dtype)
    return emb.index_put((torch.cat(bidx), torch.cat(sidx)), feats)


def _new_store(n_scales):
    return {sid: ([], [], []) for sid in range(n_scales)}


@torch.no_grad()
def _fill_store(net, store, img, coords, b):
    """Extract the retained patches and file them by scale. No gradient:
    these are data, the graph starts at the encoder."""
    for sid in range(len(net.spec.sizes)):
        mask = coords[:, 0] == sid
        if not bool(mask.any()):
            continue
        idx = torch.nonzero(mask, as_tuple=True)[0]
        sub = coords[idx]
        pats, bb, ss = store[sid]
        pats.append(net.patches(img, sub[:, 1:], sid))
        bb.append(torch.full((idx.numel(),), b, dtype=torch.long,
                             device=idx.device))
        ss.append(idx)


def _embed_single(net, img, coords):
    store = _new_store(len(net.spec.sizes))
    _fill_store(net, store, img, coords, 0)
    return _embed_from_store(net, store, 1)

# In[ ]:


# ============================================================
# Metrics: binary (RSNA) or ordinal with 5 grades (DDR)
#
# Every metric is computed at every epoch. The reported value of a run is
# always the one at the epoch that maximises the MAIN metric on validation,
# never the maximum of each metric taken separately.
# ============================================================
def _ece(p, y, n_bins=15):
    """Expected calibration error on the predicted class."""
    conf_, pred = p.max(1), p.argmax(1)
    hit = (pred == y).astype(float)
    e = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        m = (conf_ > lo) & (conf_ <= hi)
        if m.any():
            e += m.mean() * abs(hit[m].mean() - conf_[m].mean())
    return float(e)


def compute_metrics(y_true, y_pred, y_prob, task_type, threshold=None):
    """Every metric that can be computed from labels and probabilities.

    ordinal (DDR): quadratic and linear kappa, accuracy, balanced accuracy,
        macro and weighted F1, per-grade precision / recall / F1, mean grade
        error, macro one-vs-rest AUC, referable DR (grade >= 2): AUC,
        sensitivity, specificity.
    binary (RSNA): AUC, AUPRC, F1, precision, recall, specificity.
    both: confusion matrix, NLL, Brier score, expected calibration error.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=int)
    if task_type == 'binary' and threshold is not None:
        y_pred = (y_prob[:, 1] >= threshold).astype(int)
    K = y_prob.shape[1]
    labels = list(range(K))
    nan = float('nan')
    two = len(np.unique(y_true)) > 1
    out = {'accuracy': float(accuracy_score(y_true, y_pred)),
           'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)) if two else nan,
           'confusion': confusion_matrix(y_true, y_pred, labels=labels).tolist()}
    if len(y_true):
        pt = np.clip(y_prob[np.arange(len(y_true)), y_true], 1e-12, 1.0)
        out['nll'] = float(-np.log(pt).mean())
        out['brier'] = float(((y_prob - np.eye(K)[y_true]) ** 2).sum(1).mean())
        out['ece'] = _ece(y_prob, y_true)
    if task_type == 'ordinal':
        out['kappa'] = float(cohen_kappa_score(y_true, y_pred, weights='quadratic'))
        out['kappa_linear'] = float(cohen_kappa_score(y_true, y_pred, weights='linear'))
        out['f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
        out['f1_weighted'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
        out['mae_grade'] = float(np.abs(y_pred - y_true).mean())
        for nm, fn in (('precision', precision_score), ('recall', recall_score), ('f1', f1_score)):
            out[f'{nm}_per_class'] = fn(y_true, y_pred, labels=labels, average=None,
                                        zero_division=0).tolist()
        try:
            out['auc_macro_ovr'] = float(roc_auc_score(y_true, y_prob, multi_class='ovr',
                                                       average='macro', labels=labels))
        except ValueError:
            out['auc_macro_ovr'] = nan
        ref = (y_true >= 2).astype(int)                 # referable DR
        p_ref = y_prob[:, 2:].sum(1)
        try:
            out['auc_referable'] = float(roc_auc_score(ref, p_ref))
        except ValueError:
            out['auc_referable'] = nan
        pr = y_pred >= 2
        out['sens_referable'] = float(pr[ref == 1].mean()) if (ref == 1).any() else nan
        out['spec_referable'] = float((~pr)[ref == 0].mean()) if (ref == 0).any() else nan
    else:
        out['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
        out['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
        out['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
        neg = y_true == 0
        out['specificity'] = float((y_pred[neg] == 0).mean()) if neg.any() else nan
        try:
            out['auc'] = float(roc_auc_score(y_true, y_prob[:, 1]))
            out['auprc'] = float(average_precision_score(y_true, y_prob[:, 1]))
        except ValueError:
            out['auc'] = out['auprc'] = nan
    return out


def bootstrap_ci(y_true, y_prob, task_type, n=1000, seed=0):
    """95 % percentile intervals over images, for the headline metrics."""
    y, p = np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=np.float64)
    rng, vals = np.random.default_rng(seed), defaultdict(list)
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        yi, pi = y[i], p[i]
        vals['accuracy'].append(float((pi.argmax(1) == yi).mean()))
        try:
            if task_type == 'ordinal':
                vals['kappa'].append(cohen_kappa_score(yi, pi.argmax(1), weights='quadratic'))
                vals['auc_referable'].append(roc_auc_score((yi >= 2).astype(int), pi[:, 2:].sum(1)))
            else:
                vals['auc'].append(roc_auc_score(yi, pi[:, 1]))
        except ValueError:
            pass
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for k, v in vals.items() if len(v) > 10}


def youden_threshold(y_true, p_pos):
    """Threshold that maximises sensitivity + specificity - 1."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y_true, np.asarray(p_pos))
    k = int(np.argmax(tpr - fpr))
    t = float(thr[k])
    return 0.5 if not np.isfinite(t) else t


# ============================================================
# Loss: exactly the one of the original IPS code
# (softmax, log(p + eps), NLLLoss).
# ============================================================
class IPSLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.nll = nn.NLLLoss()

    def forward(self, logits, target):
        p = torch.softmax(logits, dim=-1)
        return self.nll(torch.log(p + self.eps), target)

# In[ ]:


# ============================================================
# Profiler: wall time and peak GPU memory, by phase
#
#   data   : waiting for the DataLoader (decoding, augmentation)
#   select : exploration without gradient, plus cutting the retained patches
#   step   : forward and backward on the retained patches (or on the image
#            for the baseline)
#
# The GPU is synchronised at each phase boundary, otherwise the time of an
# asynchronous kernel would be charged to the next phase. The peak counter
# is reset at the start of a phase, so each phase has its own peak; what is
# already allocated at that moment (weights, optimiser state, images that
# are on the GPU) is included in the peak, as it should be.
# ============================================================
class Profiler:
    def __init__(self):
        self.t = defaultdict(float)
        self.peak = defaultdict(float)
        self.n_img = 0

    @contextlib.contextmanager
    def phase(self, name):
        if CUDA:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if CUDA:
                torch.cuda.synchronize()
                self.peak[name] = max(self.peak[name],
                                      torch.cuda.max_memory_allocated())
            self.t[name] += time.perf_counter() - t0

    def summary(self, prefix=''):
        out = {f'{prefix}t_{k}_s': round(v, 3) for k, v in self.t.items()}
        out.update({f'{prefix}peak_{k}_mb': round(v / 1e6, 1)
                    for k, v in self.peak.items()})
        out[f'{prefix}peak_mb'] = round(max(self.peak.values(), default=0.0) / 1e6, 1)
        out[f'{prefix}n_img'] = self.n_img
        return out

# In[ ]:


# ============================================================
# One epoch
#
# The learning-rate step counts DataLoader iterations, as in
# training/iterative.py: max_steps = n_epoch * len(loader), with a loader of
# batch size B_seq. The optimiser itself steps every B images.
#
# Returns a dict: loss, y_true, y_pred, y_prob, ids, coords (if collect),
# prof (Profiler).
# ============================================================
IMAGE_MODELS = ('baseline', 'mil')    # take whole images, one optimiser step per batch


def run_epoch(net, loader, conf, device, optimizer=None, epoch=0,
              criterion=None, is_baseline=False, collect=False):
    """One pass over a loader. With collect=True (evaluation) also returns,
    per image: selected patches, their attention in the aggregator, the coarse
    zones of a cascade, and for MIL the attention logits of every patch."""
    train = optimizer is not None
    net.train(train)
    prof = Profiler()

    tot_loss, n_seen = 0.0, 0
    y_true, y_pred, y_prob, ids, coords_out = [], [], [], [], []
    attn_out, zones_out, maps_out = [], [], []
    n_step_epoch = max(1, len(loader))
    it = epoch * n_step_epoch
    lazy = (not is_baseline) and net.loading == 'lazy'
    is_mil = bool(getattr(net, 'is_mil', False))
    is_cascade = hasattr(net, 'cascade')
    dev_aug = bool(getattr(loader.dataset, 'aug_on_device', False))

    def stack(xs):
        return np.stack(xs) if xs else None

    t_mark = time.perf_counter()

    if is_baseline:
        for imgs, labels, idx in loader:
            prof.t['data'] += time.perf_counter() - t_mark
            if dev_aug:
                with prof.phase('augment'):
                    imgs = loader.dataset.device_augment(imgs.to(device))
            with prof.phase('step'):
                imgs, labels = imgs.to(device), labels.to(device)
                it += 1
                if train:
                    adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch,
                                         conf.lr, optimizer, n_step_epoch, it)
                    optimizer.zero_grad()
                with (torch.enable_grad() if train else torch.no_grad()):
                    logits = net(imgs)
                    loss = criterion(logits, labels)
                    if train:
                        loss.backward()
                        optimizer.step()
                bs = labels.shape[0]
                tot_loss += loss.item() * bs
                n_seen += bs
                prob = torch.softmax(logits.detach(), -1)
                y_true += labels.tolist()
                y_pred += prob.argmax(-1).tolist()
                y_prob += prob.cpu().numpy().tolist()
                ids += idx.tolist()
                if collect and is_mil:
                    top = net.last_attn.topk(min(net.M, net.last_attn.shape[1]), dim=1)
                    for b in range(bs):
                        coords_out.append(net.spec.candidates[top.indices[b]].cpu().numpy())
                        attn_out.append(top.values[b].float().cpu().numpy())
                        maps_out.append(net.last_logits[b].float().cpu().numpy())
            prof.n_img += bs
            t_mark = time.perf_counter()
        return dict(loss=tot_loss / max(n_seen, 1), y_true=y_true, y_pred=y_pred,
                    y_prob=y_prob, ids=ids, coords=stack(coords_out),
                    attn=stack(attn_out), zones=None, maps=stack(maps_out), prof=prof)

    store = _new_store(len(net.spec.sizes))
    labels_buf, n_prep = [], 0

    def flush(step_it):
        nonlocal store, labels_buf, n_prep, tot_loss, n_seen
        if n_prep == 0:
            return
        B = n_prep
        lab = torch.stack(labels_buf)
        if train:
            adjust_learning_rate(conf.n_epoch_warmup, conf.n_epoch,
                                 conf.lr, optimizer, n_step_epoch, step_it)
            optimizer.zero_grad()
        with (torch.enable_grad() if train else torch.no_grad()):
            emb = _embed_from_store(net, store, B)
            logits = net.classify(emb)
            loss = criterion(logits, lab)
            if train:
                loss.backward()
                optimizer.step()
            if collect:     # weight of each retained patch in the final decision
                attn_out.extend(net.transf.get_scores(emb).float().cpu().numpy())
        tot_loss += loss.item() * B
        n_seen += B
        prob = torch.softmax(logits.detach(), -1)
        y_true.extend(lab.tolist())
        y_pred.extend(prob.argmax(-1).tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
        store = _new_store(len(net.spec.sizes))
        labels_buf = []
        n_prep = 0

    for imgs, labels, idx in loader:
        prof.t['data'] += time.perf_counter() - t_mark
        labels = labels.to(device)
        if dev_aug:
            with prof.phase('augment'):
                imgs = loader.dataset.device_augment(imgs if lazy else imgs.to(device))
        if not lazy:
            with prof.phase('select'):
                imgs = imgs.to(device)
        it += 1
        for k in range(imgs.shape[0]):
            with prof.phase('select'):
                if collect and is_cascade:
                    coords, stages = net.ips(imgs[k], return_stages=True)
                else:
                    coords = net.ips(imgs[k])
                _fill_store(net, store, imgs[k], coords, n_prep)
            if collect:
                c = coords.cpu().numpy()
                if c.shape[0] < net.M_bag:          # pool smaller than budget
                    pad = -np.ones((net.M_bag - c.shape[0], 3), np.int64)
                    c = np.concatenate([c, pad], 0)
                coords_out.append(c)
                if is_cascade:
                    zones_out.append(stages[0].cpu().numpy())
            ids.append(int(idx[k]))
            labels_buf.append(labels[k])
            n_prep += 1
            prof.n_img += 1
            if n_prep == conf.B:
                with prof.phase('step'):
                    flush(it)
        t_mark = time.perf_counter()
    with prof.phase('step'):
        flush(it)
    return dict(loss=tot_loss / max(n_seen, 1), y_true=y_true, y_pred=y_pred,
                y_prob=y_prob, ids=ids, coords=stack(coords_out),
                attn=stack(attn_out), zones=stack(zones_out), maps=None, prof=prof)


# ## Data
# 
# * **DDR**: `train` to train, `valid` to choose the epoch, `test` to report.
# * **RSNA**: 70 % / 15 % / 15 % by patient. The split is a hash of the
#   patient identifier with a fixed constant, so it never depends on the seed
#   of a run.
# * **Cache**: the decoded and resized images of a split are written once to
#   a `uint8` memory-mapped file. Without it, each DDR image (about 4.7
#   megapixels) is decoded and resized again at every epoch.
# * `train_frac < 1` keeps a stratified fraction of the training set
#   (low-data regime).

# In[ ]:


SPLIT_SEED = 42            # constant: the split never follows the run seed
CACHE_DIR = _env('IPS_CACHE_DIR')            # '' = no cache
CACHE_SETS = [s for s in _env('IPS_CACHE', 'ddr').split(',') if s]
SMOKE_N = _env_int('IPS_SMOKE', 0)           # >0 : keep N images per split


def _build_cache(path, n, shape, load_one, n_thread=8):
    """Write an (n, *shape) uint8 array once. Safe with concurrent jobs."""
    path = Path(path)
    done = path.with_suffix('.done')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix('.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if done.exists():
                return
            t0 = time.time()
            arr = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                            shape=(n,) + tuple(shape))

            def work(i):
                arr[i] = load_one(i)

            with ThreadPoolExecutor(n_thread) as ex:
                list(ex.map(work, range(n)))
            arr.flush()
            del arr
            done.write_text(str(n))
            print(f"  [cache] {path.name} : {n} images in {time.time()-t0:.0f} s")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _build_cache_jpg(folder, n, load_one, n_thread=8):
    """One JPEG per image (quality 95, no chroma subsampling), for images too
    large to keep raw: DDR at 1792 px would take about 120 GB raw, about 15 GB
    this way. Safe with concurrent jobs."""
    folder = Path(folder)
    done = folder.with_suffix('.done')
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder.with_suffix('.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if done.exists():
                return
            t0 = time.time()

            def work(i):
                Image.fromarray(load_one(i)).save(folder / f'{i}.jpg', quality=95,
                                                  subsampling=0)

            with ThreadPoolExecutor(n_thread) as ex:
                list(ex.map(work, range(n)))
            done.write_text(str(n))
            print(f"  [cache] {folder.name} : {n} images (JPEG) in {time.time()-t0:.0f} s")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class _Cached:
    """Common part: list of items, optional uint8 cache, optional subsample."""

    name = 'base'

    def _finish(self, conf, split, shape):
        if SMOKE_N and len(self.items) > SMOKE_N:
            # evenly spaced, not the first N: the DDR label files are sorted
            # by grade, so the first images are all of grade 0
            pick = np.linspace(0, len(self.items) - 1, SMOKE_N).round().astype(int)
            self.items = [self.items[i] for i in pick]
        frac = float(conf.get('train_frac', 1.0))
        if split == 'train' and frac < 1.0:
            rng = np.random.default_rng(SPLIT_SEED)
            by_lab = defaultdict(list)
            for i, (_, y) in enumerate(self.items):
                by_lab[y].append(i)
            keep = []
            for y, ii in sorted(by_lab.items()):
                ii = np.array(ii)
                rng.shuffle(ii)
                keep += ii[:max(1, int(round(frac * len(ii))))].tolist()
            self.items = [self.items[i] for i in sorted(keep)]
        self._arr, self._cache_path, self._cache_kind = None, None, None
        if CACHE_DIR and self.name in CACHE_SETS:
            tag = hashlib.md5('|'.join(str(p) for p, _ in self.items)
                              .encode()).hexdigest()[:10]
            geo = conf.get('ddr_geometry', 'stretch') if self.name == 'ddr' else 'id'
            stem = f"{self.name}_{split}_{conf.img_size}_{geo}_{tag}"
            load = lambda i: self._decode(self.items[i][0])
            if conf.img_size > 1024:          # raw would be too large
                self._cache_kind = 'jpg'
                self._cache_path = Path(CACHE_DIR) / f"{stem}_jpg"
                _build_cache_jpg(self._cache_path, len(self.items), load)
            else:
                self._cache_kind = 'npy'
                self._cache_path = Path(CACHE_DIR) / f"{stem}.npy"
                _build_cache(self._cache_path, len(self.items), shape, load)

    def _raw(self, i):
        if self._cache_path is not None and self._cache_kind == 'jpg':
            return np.asarray(Image.open(self._cache_path / f'{i}.jpg'))
        if self._cache_path is not None:
            if self._arr is None:               # opened once per worker
                self._arr = np.load(self._cache_path, mmap_mode='r')
            return np.asarray(self._arr[i])
        return self._decode(self.items[i][0])

    def __len__(self):
        return len(self.items)


class DDRDataset(_Cached, Dataset):
    """DDR, 5 grades (0-4). Grade 5 (ungradable) is excluded.

    ddr_geometry = 'stretch' : resize the whole photograph to a square, as in
                               v2. Native ratios go from 0.98 to 1.76, so the
                               retina becomes an ellipse and about half of the
                               square is black margin (dataset probe).
                 = 'retina'  : crop to the retina, keep pixels square, resize.
                               Same field of view on every image, and the
                               retina fills the square.
                 = 'crop'    : centre crop to the short side. Cuts the retina
                               on about one image in eight: not recommended.
    """
    name = 'ddr'

    def __init__(self, conf, split):
        assert split in ('train', 'valid', 'test')
        self.img_size = conf.img_size
        self.geometry = conf.get('ddr_geometry', 'stretch')
        self.split = split
        self._frames = {}
        root = Path(conf.data_dir) / 'DR_grading'
        lab_file, img_dir = root / f'{split}.txt', root / split

        self.items = []
        with open(lab_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname, grade = line.split()[0], int(line.split()[1])
                if grade == 5:
                    continue
                p = img_dir / fname
                if p.exists():
                    self.items.append((p, grade))
        self._finish(conf, split, (conf.img_size, conf.img_size, 3))

        counts = [0] * 5
        for _, g in self.items:
            counts[g] += 1
        print(f"[DDR] {split:5s} | {len(self.items):5d} images | "
              + ', '.join(f"C{i}={counts[i]}" for i in range(5)))

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.aug = (transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05)])
            if split == 'train' else None)
        # Training images leave the loader as uint8 and are augmented on the
        # GPU (run_epoch): the rotation and colour jitter of 896 or 1792 px
        # images were the slowest part of an epoch on the CPU.
        self.aug_on_device = split == 'train'

    def device_augment(self, imgs):
        """uint8 (B, C, H, W) on any device -> augmented, normalised float.
        Same transforms as before, drawn independently for each image."""
        x = imgs.float().div_(255.0)
        x = torch.stack([self.aug(t) for t in x])
        return self.norm(x)

    def _frame(self, img_path):
        """Square crop (left, top, side) in native pixels, or None (stretch).

        'retina': the bounding box of the non-black region, read on a small
        copy, extended to a square around its centre. Where the square leaves
        the photograph it is filled with black, so pixels stay square even
        when the retina is cut by the frame.
        """
        if self.geometry == 'stretch':
            return None
        key = str(img_path)
        if key in self._frames:
            return self._frames[key]
        with Image.open(img_path) as im:
            w, h = im.size
            if self.geometry == 'crop':
                s = min(w, h)
                fr = ((w - s) // 2, (h - s) // 2, s)
            elif self.geometry == 'retina':
                im.draft('L', (512, 512))
                g = np.asarray(im.convert('L'))
                m = g > 12
                if m.any():
                    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
                    sy, sx = h / g.shape[0], w / g.shape[1]
                    l, r = cols[0] * sx, (cols[-1] + 1) * sx
                    t, b = rows[0] * sy, (rows[-1] + 1) * sy
                else:
                    l, r, t, b = 0, w, 0, h
                side = int(round(max(r - l, b - t)))
                fr = (int(round((l + r) / 2 - side / 2)),
                      int(round((t + b) / 2 - side / 2)), side)
            else:
                raise ValueError(f"ddr_geometry={self.geometry}")
        self._frames[key] = fr
        return fr

    def _open(self, path, resample, mode, frame_from=None):
        img = Image.open(path).convert(mode)
        fr = self._frame(frame_from or path)
        if fr is not None:
            l, t, s = fr
            img = img.crop((l, t, l + s, t + s))      # outside the image: black
        return img.resize((self.img_size, self.img_size), resample)

    def _decode(self, path):
        return np.asarray(self._open(path, Image.BILINEAR, 'RGB'), dtype=np.uint8)

    def load_mask(self, mask_path, img_path):
        """Lesion mask, cut with the frame of ITS image, nearest neighbour."""
        return np.asarray(self._open(mask_path, Image.NEAREST, 'L',
                                     frame_from=img_path)) > 0

    def __getitem__(self, i):
        if self.aug_on_device:              # augmented and normalised in run_epoch
            # np.array copies: the cache is read-only, and a tensor on read-only
            # memory makes PyTorch warn
            return torch.from_numpy(np.array(self._raw(i))).permute(2, 0, 1), \
                self.items[i][1], i
        t = torch.from_numpy(self._raw(i).astype(np.float32) / 255.0).permute(2, 0, 1)
        return self.norm(t), self.items[i][1], i


class RSNADataset(_Cached, Dataset):
    """RSNA pneumonia, binary, one image per patient."""
    name = 'rsna'

    @staticmethod
    def split_of(pid):
        # 70 / 15 / 15 by patient. The v3 training set is a subset of the v2
        # one (hash >= 25); 5 % of the v2 training patients are now in val or
        # test, so v2 weights cannot be scored cleanly on RSNA.
        h = int(hashlib.md5((str(pid) + str(SPLIT_SEED)).encode()).hexdigest(), 16) % 100
        if h >= 30:
            return 'train'
        return 'val' if h < 15 else 'test'

    def __init__(self, conf, split):
        assert split in ('train', 'val', 'test', 'v2val')
        self.img_size = conf.img_size
        self.split = split
        root = Path(conf.data_dir)
        img_dir = root / 'Training' / 'Images'

        raw = pd.read_csv(root / 'stage2_train_metadata.csv')
        df = raw.groupby('patientId')['Target'].max().reset_index()
        sp = df['patientId'].apply(self.split_of)
        keep = sp.isin(['val', 'test']) if split == 'v2val' else (sp == split)
        df = df[keep].reset_index(drop=True)

        self.boxes = {}
        if {'x', 'y', 'width', 'height'} <= set(raw.columns):
            pos = raw[raw['Target'] == 1].dropna(subset=['x', 'y', 'width', 'height'])
            self.boxes = {p: g[['x', 'y', 'width', 'height']].to_numpy(float)
                          for p, g in pos.groupby('patientId')}

        self.items, missing = [], 0
        for pid, tgt in zip(df['patientId'], df['Target']):
            p = img_dir / f"{pid}.png"
            if p.exists():
                self.items.append((p, int(tgt)))
            else:
                missing += 1
        self._finish(conf, split, (conf.img_size, conf.img_size))
        pos = sum(l for _, l in self.items)
        print(f"[RSNA] {split:5s} | {len(self.items):5d} images | "
              f"pos={pos} ({100*pos/max(len(self.items),1):.1f}%)"
              + (f" | {missing} missing" if missing else ""))

        self.aug = (transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(5)])
                    if split == 'train' else None)

    def _decode(self, path):
        img = Image.open(path).convert('L')
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def __getitem__(self, i):
        t = (torch.from_numpy(self._raw(i).astype(np.float32) / 255.0)
             .unsqueeze(0) - 0.5) / 0.5
        if self.aug is not None:
            t = self.aug(t)
        return t, self.items[i][1], i


def _worker_init(worker_id):
    s = torch.initial_seed() % (2 ** 31)
    np.random.seed(s)
    random.seed(s)


def build_dataset(conf, split):
    """split in 'train' | 'val' | 'test' (dataset-independent names)."""
    if conf.dataset == 'ddr':
        return DDRDataset(conf, {'train': 'train', 'val': 'valid',
                                 'test': 'test'}[split])
    if conf.dataset == 'rsna':
        return RSNADataset(conf, split)
    raise ValueError(conf.dataset)


def build_loader(conf, ds, train):
    # The baseline takes one optimiser step per DataLoader batch; the patch
    # models gather B_seq images at a time until they have B. Both therefore
    # train with the same batch of B = 16.
    bs = conf.B if conf.model in ('baseline', 'mil') else conf.B_seq
    g = torch.Generator()
    g.manual_seed(int(conf.seed))
    return DataLoader(ds, batch_size=bs, shuffle=train, generator=g,
                      num_workers=conf.n_worker, pin_memory=conf.pin_memory,
                      persistent_workers=False, worker_init_fn=_worker_init)

# ## Configurations
# 
# One base per data set. Everything that a run does not name is inherited,
# so the backbone, the optimiser and the schedule are identical everywhere.
# 
# Each run belongs to one or more **groups**. `IPS_RUNS` accepts run names
# and group names. The seed is appended to the run name (`ddr_ips128_s42`).
# Every model uses at most two patch sizes.
# 
# | group | question |
# |---|---|
# | `core` | the main comparison: full image, half-resolution image, attention MIL, IPS at every final patch size, MS-IPS and H-MS-IPS with two sizes. Three seeds |
# | `ablation` | selection module: cross-attention, self-attention, random selection; coarse zones of the cascade trained or not |
# | `mstudy` | values of `M`, and at an equal pixel budget `M * p^2` few large or many small patches |
# | `patch` | IPS and cascades from very large to very small patches |
# | `base` | the image at a quarter of its resolution |
# | `ms` | resampling control for MS-IPS |
# | `lowres` | coarse stage read at half resolution |
# | `hires` | DDR at 1792 px: more resolution at the same memory |
# | `lowdata` | 25 % of the training set |
# 

# In[ ]:


# ============================================================
# Paths, driven by environment variables (Kaggle defaults).
# ============================================================
WORK    = _env_path('IPS_WORK', '/gpfs01/berens/user/kdjoumessi/Workspace/Berens-lab/CIN/Projects/New/MS-IPS/master-research-codes/ips-project/outputs')
RESULTS = _env_path('IPS_RESULTS', WORK / 'results')
RESULTS.mkdir(parents=True, exist_ok=True)
# One folder per configuration, every seed inside it:
#   RESULTS/<configuration>/checkpoints   weights of the best epoch, resume state
#                          /experiments   <run>_result.json, lesion and Grad-CAM tables
#                          /history       <run>_history.json, codecarbon
#                          /predictions   per image: probabilities, patches, maps
#                          /saliency      per image: Grad-CAM maps
# use_run_dirs() points the OUT_* paths to the folder of the run being processed.
OUT_CKPT = OUT_HIST = OUT_EXPE = OUT_SAL = OUT_PRED = RESULTS


def use_run_dirs(name):
    global OUT_CKPT, OUT_HIST, OUT_EXPE, OUT_SAL, OUT_PRED
    base = RESULTS / name
    OUT_CKPT, OUT_HIST, OUT_EXPE = base / 'checkpoints', base / 'history', base / 'experiments'
    OUT_SAL, OUT_PRED = base / 'saliency', base / 'predictions'
    for d in (OUT_CKPT, OUT_HIST, OUT_EXPE, OUT_SAL, OUT_PRED):
        d.mkdir(parents=True, exist_ok=True)
    return base

DDR_DIR = os.environ.get(
    'IPS_DDR_DIR',
    '/kaggle/input/datasets/samriddhibagchi/ddr-dataset-credits-to-authors/DDR-dataset')
RSNA_DIR = os.environ.get(
    'IPS_RSNA_DIR',
    '/kaggle/input/datasets/iamtapendu/rsna-pneumonia-processed-dataset')

print(f"  work        : {WORK}")
print(f"  results     : {RESULTS}  (one folder per configuration)")
print(f"  DDR         : {DDR_DIR}")
print(f"  RSNA        : {RSNA_DIR}")
print(f"  cache       : {CACHE_DIR or '(none)'}  for {CACHE_SETS}")

BASE = dict(
    n_epoch=50, n_epoch_warmup=10, lr=0.0001, wd=0.1,
    B=16, B_seq=4, eps=1e-6,
    M=10, I=128, D=512, H=8, D_k=64, D_v=64, D_inner=2048,
    n_token=1, attn_dropout=0.1, dropout=0.1,
    pretrained=bool(_env_int('IPS_PRETRAINED', 1)),
    n_res_blocks=4, shuffle=True, use_scale_emb=True,
    # v2 used 2 workers while the job reserved 16 CPUs: decoding and
    # augmentation, not the model, set the duration of every DDR run.
    n_worker=_env_int('IPS_WORKERS', 8), pin_memory=True,
    loading='eager', train_frac=1.0, ddr_geometry=_env('IPS_DDR_GEOMETRY', 'retina'),
    aggregator='cross', selection='attention',
)

# pad_value: black after normalisation, so padding looks like the black
# background of a fundus photograph (DDR) or a black border (RSNA)
DDR_BASE = dict(BASE, dataset='ddr', data_dir=DDR_DIR, img_size=896,
                n_chan_in=3, n_class=5, task_type='ordinal',
                main_metric='kappa',
                pad_value=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])

RSNA_BASE = dict(BASE, dataset='rsna', data_dir=RSNA_DIR, img_size=1024,
                 n_chan_in=1, n_class=2, task_type='binary',
                 main_metric='auc', pad_value=[-1.0])

REGISTRY = {}       # name -> (dict of settings, groups, note)


def reg(name, base, model, groups, note, **kw):
    d = dict(base, name=name, model=model, **kw)
    if model == 'hierarchical' and 'scales' not in d:
        d['scales'] = list(d['cascade'])
    if model == 'baseline' and 'scales' not in d:
        d['scales'] = [128]
    REGISTRY[name] = (d, set(groups.split(',')), note)


def make_conf(name, seed):
    d, _, _ = REGISTRY[name]
    return Struct(**dict(d, seed=int(seed), run_name=f"{name}_s{seed}"))


# ---------------------------------------------------------------- DDR, 896
# core: what goes in the main table (three seeds)
reg('ddr_base896', DDR_BASE, 'baseline', 'core,base', 'full image', baseline_size=896)
reg('ddr_base448', DDR_BASE, 'baseline', 'core,base', 'image at half resolution', baseline_size=448)
reg('ddr_mil128',  DDR_BASE, 'mil', 'core', 'attention MIL, all 49 patches of 128', scales=[128])
reg('ddr_ips128', DDR_BASE, 'selector', 'core,patch,mstudy', 'IPS reference', scales=[128])
reg('ddr_ips112', DDR_BASE, 'selector', 'core,patch', 'control of h224_112', scales=[112])
reg('ddr_ips64',  DDR_BASE, 'selector', 'core,patch,mstudy', '196 candidates', scales=[64])
reg('ddr_ips56',  DDR_BASE, 'selector', 'core,patch', 'control of h224_56', scales=[56])
reg('ddr_ips32',  DDR_BASE, 'selector', 'core,patch,mstudy', '784 candidates, map 1x1', scales=[32])
reg('ddr_ms128_64', DDR_BASE, 'selector', 'core,ms', 'MS-IPS, flat, native', scales=[128, 64])
reg('ddr_ms64_32',  DDR_BASE, 'selector', 'core,ms', 'MS-IPS, flat, native', scales=[64, 32])
reg('ddr_h224_112', DDR_BASE, 'hierarchical', 'core', 'H-MS-IPS, final 112', cascade=[224, 112], M_keep=[4])
reg('ddr_h224_56',  DDR_BASE, 'hierarchical', 'core', 'H-MS-IPS, final 56', cascade=[224, 56], M_keep=[4])
# 64 then 32: keeps a quarter of the image, like every other cascade
# (49 zones of 64 px out of 196), then the best 10 of their 196 patches of 32
reg('ddr_h64_32',   DDR_BASE, 'hierarchical', 'core', 'H-MS-IPS, final 32', cascade=[64, 32], M_keep=[49])
# ablation: the selection module, and gradients to the coarse stage
reg('ddr_h224_112_tc', DDR_BASE, 'hierarchical', 'ablation', 'coarse zones trained',
    cascade=[224, 112], M_keep=[4], train_coarse=4)
reg('ddr_h224_56_tc',  DDR_BASE, 'hierarchical', 'ablation', 'coarse zones trained',
    cascade=[224, 56], M_keep=[4], train_coarse=4)
# mstudy: generated below for every family; extra: zones kept by the cascade
reg('ddr_h224_56_k8',  DDR_BASE, 'hierarchical', 'mstudy', '8 zones kept of 16', cascade=[224, 56], M_keep=[8])
reg('ddr_h224_56_k2',  DDR_BASE, 'hierarchical', 'mstudy', '2 zones kept of 16', cascade=[224, 56], M_keep=[2])
# patch: the rest of the single-size sweep
reg('ddr_ips224', DDR_BASE, 'selector', 'patch', '16 candidates', scales=[224])
reg('ddr_ips28',  DDR_BASE, 'selector', 'patch', 'control of h224_28', scales=[28])
reg('ddr_h224_28', DDR_BASE, 'hierarchical', 'patch', 'H-MS-IPS, final 28', cascade=[224, 28], M_keep=[4])
reg('ddr_h224_28_tc', DDR_BASE, 'hierarchical', 'patch', 'same, coarse zones trained',
    cascade=[224, 28], M_keep=[4], train_coarse=4)
# other controls
reg('ddr_base224', DDR_BASE, 'baseline', 'base', 'image at quarter resolution', baseline_size=224)
reg('ddr_ms128_64_rs', DDR_BASE, 'selector', 'ms', 'MS-IPS, 64 px upsampled to 128',
    scales=[128, 64], encode_px=[None, 128])
reg('ddr_h224lr_56_tc', DDR_BASE, 'hierarchical', 'lowres', 'coarse stage at 112 px',
    cascade=[224, 56], M_keep=[4], train_coarse=4, encode_px=[112, None])
for _n, _kw in [('ddr_ips128', dict(scales=[128])), ('ddr_ips64', dict(scales=[64])),
                ('ddr_ips32', dict(scales=[32]))]:
    reg(_n + '_f25', DDR_BASE, 'selector', 'lowdata', '25 % of train', train_frac=0.25, **_kw)
reg('ddr_h224_56_f25', DDR_BASE, 'hierarchical', 'lowdata', '25 % of train',
    cascade=[224, 56], M_keep=[4], train_frac=0.25)
reg('ddr_ips128_lazy', DDR_BASE, 'selector', 'misc', 'image stays on the CPU', scales=[128], loading='lazy')
reg('ddr_ips128_i16',  DDR_BASE, 'selector', 'misc', 'I = 16', scales=[128], I=16)

# hires: the same patch models on DDR at twice the resolution (1792 px).
# Memory depends on M and the patch size, not on the image, so these train
# at the memory of their 896 px twins while every lesion is twice as wide.
DDR_HI = dict(DDR_BASE, img_size=1792)
reg('ddr_r1792_ips128', DDR_HI, 'selector', 'hires', '196 candidates, memory of ddr_ips128', scales=[128])
reg('ddr_r1792_ips128_m20', DDR_HI, 'selector', 'hires', '', scales=[128], M=20)
reg('ddr_r1792_ips112', DDR_HI, 'selector', 'hires', 'control of r1792_h448_112', scales=[112])
reg('ddr_r1792_h448_112', DDR_HI, 'hierarchical', 'hires', 'H-MS-IPS at 1792',
    cascade=[448, 112], M_keep=[4])

# --------------------------------------------------------------- RSNA, 1024
# An opacity box measures 209 +/- 63 x 339 +/- 149 px (short x long side):
# 256 px is about one opacity, 128 px a part of one.
reg('rsna_base1024', RSNA_BASE, 'baseline', 'core,base', 'full image', baseline_size=1024)
reg('rsna_base512',  RSNA_BASE, 'baseline', 'core,base', 'image at half resolution', baseline_size=512)
reg('rsna_mil128',   RSNA_BASE, 'mil', 'core', 'attention MIL, all 64 patches of 128', scales=[128])
reg('rsna_mil256',   RSNA_BASE, 'mil', 'core', 'attention MIL, all 16 patches of 256', scales=[256])
reg('rsna_ips256', RSNA_BASE, 'selector', 'core,patch', 'about one opacity per patch', scales=[256])
reg('rsna_ips128', RSNA_BASE, 'selector', 'core,patch,mstudy', 'IPS reference', scales=[128])
reg('rsna_ips64',  RSNA_BASE, 'selector', 'core,patch,mstudy', 'control of h256_64', scales=[64])
reg('rsna_ms256_128', RSNA_BASE, 'selector', 'core,ms', 'MS-IPS, flat, native', scales=[256, 128])
reg('rsna_ms128_64',  RSNA_BASE, 'selector', 'core,ms', 'MS-IPS, flat, native', scales=[128, 64])
reg('rsna_h256_128', RSNA_BASE, 'hierarchical', 'core', 'H-MS-IPS, final 128', cascade=[256, 128], M_keep=[4])
reg('rsna_h256_64',  RSNA_BASE, 'hierarchical', 'core', 'H-MS-IPS, final 64', cascade=[256, 64], M_keep=[4])
reg('rsna_h256_128_tc', RSNA_BASE, 'hierarchical', 'ablation', 'coarse zones trained',
    cascade=[256, 128], M_keep=[4], train_coarse=4)
reg('rsna_h256_64_tc',  RSNA_BASE, 'hierarchical', 'ablation', 'coarse zones trained',
    cascade=[256, 64], M_keep=[4], train_coarse=4)
reg('rsna_ips32',  RSNA_BASE, 'selector', 'patch', '1024 candidates, map 1x1', scales=[32])
reg('rsna_h256_32', RSNA_BASE, 'hierarchical', 'patch', 'H-MS-IPS, final 32', cascade=[256, 32], M_keep=[4])
reg('rsna_base256',  RSNA_BASE, 'baseline', 'base', 'image at quarter resolution', baseline_size=256)
reg('rsna_ms128_64_rs', RSNA_BASE, 'selector', 'ms', 'MS-IPS, 64 px upsampled to 128',
    scales=[128, 64], encode_px=[None, 128])
reg('rsna_h256lr_64_tc', RSNA_BASE, 'hierarchical', 'lowres', 'coarse stage at 128 px',
    cascade=[256, 64], M_keep=[4], train_coarse=4, encode_px=[128, None])
reg('rsna_ips128_f25', RSNA_BASE, 'selector', 'lowdata', '25 % of train', scales=[128], train_frac=0.25)
reg('rsna_ips64_f25',  RSNA_BASE, 'selector', 'lowdata', '25 % of train', scales=[64], train_frac=0.25)


# ------------------------------------------------------------------ focus
# The selection of 18 November: DDR at 66 and 33 px, RSNA at 300 and 150 px,
# with the full image and MIL at the same sizes. These sizes do not divide
# the image: the last row and column of patches are completed with black.
reg('ddr_mil66',   DDR_BASE, 'mil', 'focus', 'attention MIL, all 196 patches of 66', scales=[66])
reg('ddr_mil33',   DDR_BASE, 'mil', 'focus', 'attention MIL, all 784 patches of 33', scales=[33])
reg('ddr_ips66',   DDR_BASE, 'selector', 'focus', '196 candidates', scales=[66])
reg('ddr_ips33',   DDR_BASE, 'selector', 'focus', '784 candidates', scales=[33])
reg('ddr_ms66_33', DDR_BASE, 'selector', 'focus', 'MS-IPS, flat', scales=[66, 33])
# 66 then 33 keeps a quarter of the zones, like every cascade (49 of 196)
reg('ddr_h66_33',  DDR_BASE, 'hierarchical', 'focus', 'H-MS-IPS, 49 of 196 zones kept',
    cascade=[66, 33], M_keep=[49])
reg('rsna_mil300',    RSNA_BASE, 'mil', 'focus', 'attention MIL, all 16 patches of 300', scales=[300])
reg('rsna_mil150',    RSNA_BASE, 'mil', 'focus', 'attention MIL, all 49 patches of 150', scales=[150])
reg('rsna_ips300',    RSNA_BASE, 'selector', 'focus', '16 candidates', scales=[300])
reg('rsna_ips150',    RSNA_BASE, 'selector', 'focus', '49 candidates', scales=[150])
reg('rsna_ms300_150', RSNA_BASE, 'selector', 'focus', 'MS-IPS, flat', scales=[300, 150])
reg('rsna_h300_150',  RSNA_BASE, 'hierarchical', 'focus', 'H-MS-IPS, 4 of 16 zones kept',
    cascade=[300, 150], M_keep=[4])
for _n in ('ddr_base896', 'rsna_base1024'):
    REGISTRY[_n][1].add('focus')
reg('ddr_h66_33_tc',    DDR_BASE, 'hierarchical', 'focus_abl,ablation', 'coarse zones trained',
    cascade=[66, 33], M_keep=[49], train_coarse=4)
reg('rsna_h300_150_tc', RSNA_BASE, 'hierarchical', 'focus_abl,ablation', 'coarse zones trained',
    cascade=[300, 150], M_keep=[4], train_coarse=4)


# ------------------------------------------------------------------ ablations
# Every selector of focus and of core (IPS, MS-IPS, H-MS-IPS, both data sets)
# is also trained with the self-attention aggregator and with random selection.
def _variant(src, suffix, groups, note, **kw):
    d = REGISTRY[src][0]
    base = DDR_BASE if d['dataset'] == 'ddr' else RSNA_BASE
    if d.get('img_size') != base['img_size']:
        base = dict(base, img_size=d['img_size'])
    keep = {k: d[k] for k in ('scales', 'cascade', 'M_keep', 'n_res_blocks', 'M')
            if k in d and d[k] != base.get(k)}
    keep.update(kw)
    reg(src + suffix, base, d['model'], groups, note, **keep)


def _selectors(group):
    return [n for n, (d, g, _) in list(REGISTRY.items())
            if group in g and d['model'] in ('selector', 'hierarchical')]


_FOCUS_SEL, _CORE_SEL = _selectors('focus'), _selectors('core')
for _n in _FOCUS_SEL + _CORE_SEL:
    _g = 'focus_abl,ablation' if _n in _FOCUS_SEL else 'ablation'
    _variant(_n, '_sa', _g, 'self-attention aggregator', aggregator='self')
    _variant(_n, '_rand', _g, 'random selection, trained', selection='random')

# ------------------------------------------------------------------ M study
# Every selector of focus and core, with every value of M below that is
# smaller than its pool of candidates (M = 10 is the run itself). The pool is
# the number of candidates of IPS and MS-IPS, and the number of fine
# candidates left by the coarse stage for H-MS-IPS.
M_VALUES = (2, 5, 20, 40, 80, 160)


def _pool(d):
    if d['model'] == 'hierarchical':
        c = d['cascade']
        return d['M_keep'][-1] * (c[-2] // c[-1]) ** 2
    return sum((-(-d['img_size'] // p)) ** 2 for p in d['scales'])


for _n in _FOCUS_SEL + _CORE_SEL:
    REGISTRY[_n][1].add('mstudy')
    _g = 'focus_m,mstudy' if _n in _FOCUS_SEL else 'mstudy'
    for _M in M_VALUES:
        if _M < _pool(REGISTRY[_n][0]):
            _variant(_n, f'_m{_M}', _g, f'M = {_M}', M=_M)

for _n, (_d, _g, _) in REGISTRY.items():          # two patch sizes at most
    assert len(_d['scales']) <= 2, _n

GROUPS = sorted({g for _, gs, _ in REGISTRY.values() for g in gs})

# The 21 runs of the v2 suite, so that their saved weights can be scored on
# the held-out images (IPS_MODE=eval_v2). Keys are the v2 run names.
V2 = {
    'ddr_baseline_v2': (DDR_BASE, 'baseline', dict(baseline_size=896)),
    'ddr_ips_v2': (DDR_BASE, 'selector', dict(scales=[128])),
    'ddr_msips2_native_v2': (DDR_BASE, 'selector', dict(scales=[128, 64])),
    'ddr_msips3_native_v2': (DDR_BASE, 'selector', dict(scales=[128, 64, 32], n_res_blocks=2)),
    'ddr_ips_b2_v2': (DDR_BASE, 'selector', dict(scales=[128], n_res_blocks=2)),
    'ddr_msips2_native_b2_v2': (DDR_BASE, 'selector', dict(scales=[128, 64], n_res_blocks=2)),
    'ddr_hmsips_d2_v2': (DDR_BASE, 'hierarchical', dict(cascade=[224, 112], M_keep=[4])),
    'ddr_hmsips_d4_v2': (DDR_BASE, 'hierarchical', dict(cascade=[224, 56], M_keep=[4])),
    'ddr_hmsips_d8_v2': (DDR_BASE, 'hierarchical', dict(cascade=[224, 28], M_keep=[4], n_res_blocks=2)),
    'ddr_hmsips_d16_v2': (DDR_BASE, 'hierarchical', dict(cascade=[224, 14], M_keep=[4], n_res_blocks=2)),
    'ddr_hmsips3_v2': (DDR_BASE, 'hierarchical', dict(cascade=[224, 56, 14], M_keep=[4, 8], n_res_blocks=2)),
    'ddr_msips3_clinique_v2': (DDR_BASE, 'selector', dict(scales=[128, 32, 16], n_res_blocks=2)),
    'rsna_baseline_v2': (RSNA_BASE, 'baseline', dict(baseline_size=1024)),
    'rsna_ips_v2': (RSNA_BASE, 'selector', dict(scales=[128])),
    'rsna_msips2_native_v2': (RSNA_BASE, 'selector', dict(scales=[128, 64])),
    'rsna_msips3_native_v2': (RSNA_BASE, 'selector', dict(scales=[128, 64, 32], n_res_blocks=2)),
    'rsna_msips2_clinique_v2': (RSNA_BASE, 'selector', dict(scales=[256, 128])),
    'rsna_hmsips_d2_v2': (RSNA_BASE, 'hierarchical', dict(cascade=[256, 128], M_keep=[4])),
    'rsna_hmsips_d4_v2': (RSNA_BASE, 'hierarchical', dict(cascade=[256, 64], M_keep=[4])),
    'rsna_hmsips_d8_v2': (RSNA_BASE, 'hierarchical', dict(cascade=[256, 32], M_keep=[4], n_res_blocks=2)),
    'rsna_hmsips3_v2': (RSNA_BASE, 'hierarchical', dict(cascade=[256, 128, 64], M_keep=[4, 6])),
}


def make_v2_conf(run_name):
    base, model, kw = V2[run_name]
    # v2 weights were trained on the stretched photograph: evaluate them on it
    d = dict(base, name=run_name, model=model, seed=42, run_name=run_name,
             ddr_geometry='stretch', **kw)
    if model == 'hierarchical':
        d['scales'] = list(d['cascade'])
    if model == 'baseline':
        d['scales'] = [128]
    return Struct(**d)

# In[ ]:


# ============================================================
# Model construction and the cost of a configuration, known before training
# ============================================================
def build_model(conf, device):
    if conf.model == 'baseline':
        return BaselineNet(conf, device).to(device)
    if conf.model == 'hierarchical':
        return HMSIPSNet(conf, device).to(device)
    if conf.model == 'selector':
        return SelectorNet(conf, device).to(device)
    if conf.model == 'mil':
        return MILNet(conf, device).to(device)
    raise ValueError(conf.model)


def cost_report(conf, net, verbose=True):
    """Pixel counts of one image, exact, without running anything.

    explore_px          pixels read from the image by the selection
    explore_encoded_px  pixels that enter the encoder during the selection
    exploit_px_max      pixels that enter the encoder WITH gradient (upper bound)
    patch_bytes         (M + I) patches of the largest encoded size, float32
    image_bytes         one image, float32
    """
    hw = conf.img_size ** 2
    c = {'img_px': hw, 'image_bytes': conf.n_chan_in * hw * 4}
    if conf.model == 'baseline':
        s = int(conf.get('baseline_size', conf.img_size)) ** 2
        c.update(n_candidates=0, explore_px=0, explore_encoded_px=0,
                 exploit_px_max=s, patch_bytes=0, feature_maps='')
    elif conf.model == 'mil':
        p = net.p
        c.update(n_candidates=int(net.spec.N), explore_px=0, explore_encoded_px=0,
                 exploit_px_max=hw, patch_bytes=0,
                 feature_maps=f"{p}px->{net.encoder.feature_map_size(p)}x{net.encoder.feature_map_size(p)}")
    else:
        sizes = [net.encoded_size(sid) for sid in range(len(net.spec.sizes))]
        if conf.model == 'hierarchical':
            exploit = conf.M * sizes[-1] ** 2 + net.train_coarse * sizes[0] ** 2
        else:
            exploit = conf.M * max(sizes) ** 2
        c.update(n_candidates=int(net.n_candidates),
                 explore_px=int(net.explore_pixels),
                 explore_encoded_px=int(net.explore_encoded_pixels),
                 exploit_px_max=int(exploit),
                 patch_bytes=(conf.M + conf.I) * conf.n_chan_in * max(sizes) ** 2 * 4,
                 feature_maps=', '.join(
                     f"{p}px->{net.encoder.feature_map_size(p)}x{net.encoder.feature_map_size(p)}"
                     for p in sizes))
    if verbose:
        if conf.model == 'mil':                 # no selection: every patch is used
            print(f"  patches (all used) : {c['n_candidates']}, no selection")
        else:
            print(f"  candidates         : {c['n_candidates']}")
        print(f"  pixels, selection  : {c['explore_px']/1e6:.2f} M read, "
              f"{c['explore_encoded_px']/1e6:.2f} M encoded "
              f"({c['explore_encoded_px']/hw:.2f} x image)")
        print(f"  pixels, training   : {c['exploit_px_max']/1e6:.3f} M "
              f"({100*c['exploit_px_max']/hw:.1f} % of the image)")
        if conf.model in ('selector', 'hierarchical'):
            print(f"  feature maps       : {c['feature_maps']}")
            print(f"  (M+I) patches      : {c['patch_bytes']/1e6:.1f} MB "
                  f"+ image {c['image_bytes']/1e6:.1f} MB")
    return c


def efficiency_profile(conf, net):
    """Parameters, and floating-point operations for ONE image at inference,
    counted by torch.utils.flop_counter. For the patch models the selection
    and the classification are counted apart. A training step costs about the
    selection plus three times the classification (forward and backward)."""
    out = {'params': int(sum(p.numel() for p in net.parameters())),
           'params_trainable': int(sum(p.numel() for p in net.parameters() if p.requires_grad))}
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception as exc:
        out['flops_error'] = repr(exc)
        return out
    was = net.training
    net.eval()
    img = torch.randn(conf.n_chan_in, conf.img_size, conf.img_size, device=device)
    try:
        with torch.no_grad():
            if conf.model in ('baseline', 'mil'):
                with FlopCounterMode(display=False) as fc:
                    net(img.unsqueeze(0))
                out['gflops_infer'] = fc.get_total_flops() / 1e9
            else:
                if net.loading == 'lazy':
                    img = img.cpu()
                with FlopCounterMode(display=False) as fc:
                    coords = net.ips(img)
                out['gflops_select'] = fc.get_total_flops() / 1e9
                with FlopCounterMode(display=False) as fc:
                    net.classify(_embed_single(net, img, coords))
                out['gflops_classify'] = fc.get_total_flops() / 1e9
                out['gflops_infer'] = out['gflops_select'] + out['gflops_classify']
    except Exception as exc:
        out['flops_error'] = repr(exc)
    finally:
        net.train(was)
        if hasattr(net, 'reseed'):
            net.reseed(conf.seed)
    return out


@torch.no_grad()
def score_maps(net, ds, conf):
    """Attention logit of every candidate, for every image of ds (heatmaps).
    Returns (ids, maps) or None when the model has no such score."""
    if not hasattr(net, 'score_map'):
        return None
    loader = build_loader(conf, ds, train=False)
    ids, maps = [], []
    for imgs, _, idx in loader:
        if net.loading != 'lazy':
            imgs = imgs.to(device)
        for k in range(imgs.shape[0]):
            m = net.score_map(imgs[k])
            if m is None:
                return None
            maps.append(m.astype(np.float16))
            ids.append(int(idx[k]))
    return np.asarray(ids), np.stack(maps)


# In[ ]:


# ============================================================
# Evaluation of one split with the weights of the best epoch.
# Saves, per image: index, path, label, probabilities, selected patches.
# ============================================================
def evaluate_split(net, ds, conf, split, criterion, save=True, tag=''):
    """Score one split with the weights of the best epoch, and save per image:
    index, file, label, probabilities, selected patches (coords), their weight
    in the final decision (attn), the coarse zones of a cascade (zones), and on
    the test split the logit of every candidate (score_maps, for heatmaps)."""
    is_img = conf.model in IMAGE_MODELS
    if hasattr(net, 'reseed'):
        net.reseed(conf.seed + 1000)
    loader = build_loader(conf, ds, train=False)
    out = run_epoch(net, loader, conf, device, None, 0, criterion, is_img,
                    collect=True)
    met = compute_metrics(out['y_true'], out['y_pred'], out['y_prob'],
                          conf.task_type)
    met['loss'] = out['loss']
    met['n'] = len(out['y_true'])
    prof = out['prof']
    n = max(prof.n_img, 1)
    met.update(prof.summary('eval_'))
    met['ms_per_img_select'] = round(1000 * prof.t.get('select', 0.0) / n, 3)
    met['ms_per_img_step'] = round(1000 * prof.t.get('step', 0.0) / n, 3)
    met['ms_per_img_model'] = round(met['ms_per_img_select'] + met['ms_per_img_step'], 3)
    met['img_per_s_model'] = round(1000 / max(met['ms_per_img_model'], 1e-9), 2)
    if split == 'test':
        met['ci95'] = bootstrap_ci(out['y_true'], out['y_prob'], conf.task_type)
    if out['coords'] is not None and hasattr(net, 'spec'):
        sid = out['coords'][:, :conf.M, 0]
        met['scale_counts'] = {str(p): int((sid == k).sum())
                               for k, p in enumerate(net.spec.sizes)}
    if out['attn'] is not None:
        a = out['attn'][:, :conf.M]
        met['attn_top1_share'] = float(np.mean(a.max(1) / np.maximum(a.sum(1), 1e-12)))
    maps = out['maps']
    if split == 'test' and maps is None and conf.model in ('selector', 'hierarchical'):
        t0 = time.perf_counter()
        sm = score_maps(net, ds, conf)
        if sm is not None:
            order = {i: j for j, i in enumerate(sm[0])}
            maps = sm[1][[order[i] for i in out['ids']]]
            met['s_score_maps'] = round(time.perf_counter() - t0, 1)
    if save:
        empty = np.zeros((0,), np.float32)
        np.savez_compressed(
            OUT_PRED / f"{conf.run_name}{tag}_{split}.npz",
            ids=np.asarray(out['ids']),
            paths=np.asarray([str(ds.items[i][0].name) for i in out['ids']]),
            y_true=np.asarray(out['y_true']),
            y_prob=np.asarray(out['y_prob'], dtype=np.float32),
            coords=(out['coords'] if out['coords'] is not None
                    else np.zeros((0, 0, 3), np.int64)),
            attn=(out['attn'].astype(np.float32) if out['attn'] is not None else empty),
            zones=(out['zones'] if out['zones'] is not None
                   else np.zeros((0, 0, 3), np.int64)),
            score_maps=(maps.astype(np.float16) if maps is not None else empty),
            candidates=(net.spec.candidates.cpu().numpy() if hasattr(net, 'spec')
                        else np.zeros((0, 3), np.int64)),
            sizes=np.asarray(conf.scales), img_size=np.asarray(conf.img_size))
    return met, out


# In[ ]:


# ============================================================
# What is the selection worth?
#
#   topm          : the learned selection
#   scale_matched : same count per scale, random positions
#   uniform       : everything at random among the N candidates
#
# uniform -> scale_matched : the choice of scales
# scale_matched -> topm    : the choice of positions
# ============================================================
@torch.no_grad()
def evaluate_selection(net, loader, conf, device, mode, criterion):
    net.eval()
    net.reseed(conf.seed + 1000)      # same draws as the scored evaluation
    store = _new_store(len(net.spec.sizes))
    labels_buf, n_prep = [], 0
    tot_loss, n_seen = 0.0, 0
    y_true, y_pred, y_prob = [], [], []
    scale_counts = np.zeros(len(net.spec.sizes), dtype=np.int64)
    lazy = net.loading == 'lazy'
    M0 = net.M_bag
    net.M_bag = net.M              # random draws never add coarse zones

    def flush():
        nonlocal store, labels_buf, n_prep, tot_loss, n_seen
        if n_prep == 0:
            return
        lab = torch.stack(labels_buf)
        logits = net.classify(_embed_from_store(net, store, n_prep))
        tot_loss += criterion(logits, lab).item() * n_prep
        n_seen += n_prep
        prob = torch.softmax(logits, -1)
        y_true.extend(lab.tolist())
        y_pred.extend(prob.argmax(-1).tolist())
        y_prob.extend(prob.cpu().numpy().tolist())
        store = _new_store(len(net.spec.sizes))
        labels_buf = []
        n_prep = 0

    try:
        for imgs, labels, _ in loader:
            labels = labels.to(device)
            if not lazy:
                imgs = imgs.to(device)
            for k in range(imgs.shape[0]):
                if mode == 'topm':
                    coords = net.ips(imgs[k])[:net.M]
                elif mode == 'scale_matched':
                    coords = net.sample_baseline('scale_matched',
                                                 net.ips(imgs[k])[:net.M])
                elif mode == 'uniform':
                    coords = net.sample_baseline('uniform')
                else:
                    raise ValueError(mode)
                scale_counts += np.bincount(coords[:, 0].cpu().numpy(),
                                            minlength=len(net.spec.sizes))
                _fill_store(net, store, imgs[k], coords, n_prep)
                labels_buf.append(labels[k])
                n_prep += 1
                if n_prep == conf.B:
                    flush()
        flush()
    finally:
        net.M_bag = M0

    met = compute_metrics(y_true, y_pred, y_prob, conf.task_type)
    met['loss'] = tot_loss / max(n_seen, 1)
    met['scale_counts'] = scale_counts.tolist()
    return met


def selection_study(net, ds, conf, device, criterion):
    key = conf.main_metric
    loader = build_loader(conf, ds, train=False)
    rows = {m: evaluate_selection(net, loader, conf, device, m, criterion)
            for m in ('uniform', 'scale_matched', 'topm')}
    u, s, t = (rows[m].get(key, rows[m]['accuracy'])
               for m in ('uniform', 'scale_matched', 'topm'))
    rows['gain_scales'] = s - u
    rows['gain_positions'] = t - s
    print(f"  selection study ({key}) : uniform {u:.4f} | scale-matched {s:.4f} "
          f"| top-M {t:.4f}  -> scales {s-u:+.4f}, positions {t-s:+.4f}")
    if conf.get('train_coarse', 0):
        rows['note'] = ('coarse zones are left out of the bag for the three '
                        'rows, so top-M here differs from the test score')
    return rows

# In[ ]:


# ============================================================
# Do the selected patches fall on lesions?
#
# For every annotated image of the split and every lesion type:
#   hit        share of the M selected patches that touch the lesion
#   chance     the same share for a patch drawn uniformly at the same scale,
#              computed EXACTLY by enumerating all the candidates
#   cover      share of the lesion pixels inside the selected patches
#   area       share of the image inside the selected patches
#              (cover / area is the enrichment in pixels)
# For a cascade, the same for the zones kept by the coarse stage: if the
# coarse stage drops the lesion, the fine stage cannot recover it.
# ============================================================
def _integral(mask):
    ii = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1), np.int64)
    ii[1:, 1:] = mask.cumsum(0).cumsum(1)
    return ii


def _box_sum(ii, r, c, p):
    """Sum of the mask over a p x p box; the part outside the image counts 0."""
    H, W = ii.shape[0] - 1, ii.shape[1] - 1
    r, c = np.asarray(r), np.asarray(c)
    r2, c2 = np.minimum(r + p, H), np.minimum(c + p, W)
    r, c = np.minimum(r, H), np.minimum(c, W)
    return ii[r2, c2] - ii[r, c2] - ii[r2, c] + ii[r, c]


def _ddr_lesion_index(conf):
    """stem -> {type: path}. Types are the folder names (EX, HE, MA, SE)."""
    seg = Path(conf.data_dir) / 'lesion_segmentation'
    out = defaultdict(dict)
    if not seg.exists():
        return out
    for sp in ('train', 'valid', 'test'):
        # the valid part of the data set names its folder 'segmentation label'
        for lab in ('label', 'segmentation label'):
            d = seg / sp / lab
            if not d.exists():
                continue
            for td in sorted(x for x in d.iterdir() if x.is_dir()):
                for mp in td.glob('*'):
                    out[mp.stem][td.name] = mp
    return out


def _lesion_masks(ds, conf, i, les_index):
    """Dict type -> boolean mask (S,S) for image i, or {} if not annotated."""
    S = conf.img_size
    path, label = ds.items[i]
    if conf.dataset == 'rsna':
        boxes = ds.boxes.get(path.stem)
        if boxes is None or label != 1:
            return {}
        m = np.zeros((S, S), bool)
        for x, y, w, h in boxes:
            m[max(0, int(y)):min(S, int(y + h)), max(0, int(x)):min(S, int(x + w))] = True
        return {'opacity': m}
    files = les_index.get(path.stem)
    if not files:
        return {}
    masks = {t: ds.load_mask(p, path) for t, p in files.items()}
    masks = {t: m for t, m in masks.items() if m.any()}
    if masks:
        masks['any'] = np.logical_or.reduce(list(masks.values()))
    return masks


@torch.no_grad()
def lesion_study(net, ds, conf, device, max_images=None, split='', given=None):
    net.eval()
    net.reseed(conf.seed + 3000)
    S = conf.img_size
    sizes = net.spec.sizes
    les_index = _ddr_lesion_index(conf) if conf.dataset == 'ddr' else {}
    grids = {}
    for p in set(sizes):
        k = -(-S // p)                                  # padded grid, as in TileSpec
        r, c = np.divmod(np.arange(k * k), k)
        grids[p] = (r * p, c * p)

    acc = defaultdict(lambda: defaultdict(float))
    rows, n_img = [], 0
    for i in range(len(ds)):
        masks = _lesion_masks(ds, conf, i, les_index)
        if not masks:
            continue
        if given is not None and i in given:
            # the patches selected during the scored evaluation of this split
            c_, zones = given[i]
            sel = c_[:net.M]
            sel = sel[sel[:, 0] >= 0]
        else:
            img = ds[i][0]
            if net.loading != 'lazy':
                img = img.to(device)
            coords, stages = net.ips(img, return_stages=True)
            sel = coords[:net.M].cpu().numpy()
            zones = stages[0].cpu().numpy() if len(stages) > 1 else None
        canvas = np.zeros((S, S), bool)
        for sid, r, c in sel:
            canvas[r:r + sizes[sid], c:c + sizes[sid]] = True
        zcanvas = None
        if zones is not None:
            zcanvas = np.zeros((S, S), bool)
            for sid, r, c in zones:
                zcanvas[r:r + sizes[sid], c:c + sizes[sid]] = True
        row = {'img': ds.items[i][0].name, 'label': int(ds.items[i][1])}
        for t, m in masks.items():
            ii = _integral(m)
            hit = np.array([_box_sum(ii, r, c, sizes[sid]) > 0 for sid, r, c in sel])
            ch = {p: float((_box_sum(ii, *grids[p], p) > 0).mean()) for p in set(sizes)}
            chance = np.array([ch[sizes[sid]] for sid, _, _ in sel])
            a = acc[t]
            a['n_img'] += 1
            a['n_patch'] += len(sel)
            a['hits'] += float(hit.sum())
            a['expected_hits'] += float(chance.sum())
            a['img_any_hit'] += float(hit.any())
            a['img_any_chance'] += float(1 - np.prod(1 - chance))
            a['cover'] += float((canvas & m).sum() / m.sum())
            a['area'] += float(canvas.mean())
            a['lesion_frac'] += float(m.mean())
            row[f'{t}_hit'] = float(hit.mean())
            row[f'{t}_chance'] = float(chance.mean())
            row[f'{t}_cover'] = float((canvas & m).sum() / m.sum())
            if zones is not None:
                zhit = np.array([_box_sum(ii, r, c, sizes[sid]) > 0 for sid, r, c in zones])
                zc = ch[sizes[0]]
                a['zone_hits'] += float(zhit.sum())
                a['zone_expected'] += float(zc * len(zones))
                a['zone_n'] += len(zones)
                a['zone_any'] += float(zhit.any())
                a['zone_cover'] += float((zcanvas & m).sum() / m.sum())
                a['zone_lost'] += float(not (zcanvas & m).any())
                row[f'{t}_zone_cover'] = float((zcanvas & m).sum() / m.sum())
        rows.append(row)
        n_img += 1
        if max_images and n_img >= max_images:
            break

    res = {}
    for t, a in acc.items():
        n = max(a['n_img'], 1)
        r = {'n_img': int(a['n_img']),
             'hit_rate': a['hits'] / max(a['n_patch'], 1),
             'chance_rate': a['expected_hits'] / max(a['n_patch'], 1),
             'enrichment': a['hits'] / a['expected_hits'] if a['expected_hits'] else float('nan'),
             'img_any_hit': a['img_any_hit'] / n,
             'img_any_chance': a['img_any_chance'] / n,
             'lesion_cover': a['cover'] / n,
             'selected_area': a['area'] / n,
             'lesion_frac': a['lesion_frac'] / n}
        if a['zone_n']:
            r.update(zone_hit_rate=a['zone_hits'] / a['zone_n'],
                     zone_chance_rate=a['zone_expected'] / a['zone_n'],
                     zone_any=a['zone_any'] / n,
                     zone_cover=a['zone_cover'] / n,
                     zone_lost=a['zone_lost'] / n)
        res[t] = r
    if rows:
        pd.DataFrame(rows).to_csv(
            OUT_EXPE / f"{conf.run_name}_lesions_{split or 'all'}.csv", index=False)
    for t, r in sorted(res.items(), key=lambda kv: (kv[0] not in ('any', 'opacity'), kv[0])):
        z = (f" | zones keep {100*r['zone_cover']:.0f} % of the lesion, "
             f"lose it entirely in {100*r['zone_lost']:.0f} % of images"
             if 'zone_cover' in r else '')
        print(f"  lesions {split:4s} [{t:7s}] n={r['n_img']:4d}  hit {100*r['hit_rate']:.1f} % "
              f"vs chance {100*r['chance_rate']:.1f} %  (x{r['enrichment']:.2f}){z}")
    return res


# ============================================================
# Grad-CAM (Selvaraju et al., 2017), on the test split, for EVERY model.
#
# Class explained: pneumonia on RSNA, the predicted grade on DDR.
# Baseline: last residual stage of the ResNet-18, over the whole image.
# MIL: the same stage inside every patch.
# Selectors: the same stage inside every SELECTED patch; zero elsewhere, so
#   the map combines where the selector looked and what it used inside.
# Each map is normalised to [0, 1] and saved at 112 x 112 for figures.
# On annotated images, for each lesion type:
#   energy_in_lesion  share of the map's mass that falls on the lesion
#   energy_ratio      the same divided by the lesion's share of the image
#                     (1 = no better than spreading the map uniformly)
#   pointing_game     the maximum of the map is within 15 px of a lesion
#   pixel_auroc       the map as a pixel-wise lesion detector (224 x 224 grid)
# ============================================================
def _cam_from(acts, grads):
    w = grads.mean(dim=(2, 3), keepdim=True)
    return torch.relu((w * acts).sum(1))                   # (n, h, w)


def gradcam_one(net, conf, img, coords=None):
    """(S, S) Grad-CAM in [0, 1] for one image, and the class explained."""
    S = conf.img_size
    net.eval()
    x = img if (conf.model not in IMAGE_MODELS and net.loading == 'lazy') else img.to(device)
    if conf.model not in IMAGE_MODELS:
        if coords is None:
            coords = net.ips(x)                            # no gradient, no hook
        else:                                              # the evaluated selection
            coords = torch.as_tensor(coords[coords[:, 0] >= 0], device=device)
    acts = []

    def hook(_m, _i, o):
        o.retain_grad()
        acts.append(o)

    layer = net.net.layer4 if conf.model == 'baseline' else net.encoder.net[-2]
    h = layer.register_forward_hook(hook)
    try:
        with torch.enable_grad():
            if conf.model in IMAGE_MODELS:
                logits = net(x.unsqueeze(0))
            else:
                logits = net.classify(_embed_single(net, x, coords))
            target = 1 if conf.task_type == 'binary' else int(logits[0].argmax())
            net.zero_grad(set_to_none=True)
            logits[0, target].backward()
    finally:
        h.remove()
    canvas = torch.zeros(S, S)
    with torch.no_grad():
        if conf.model == 'baseline':
            cam = _cam_from(acts[0], acts[0].grad)[0]
            canvas = F.interpolate(cam[None, None], size=(S, S), mode='bilinear',
                                   align_corners=False)[0, 0].cpu()
        elif conf.model == 'mil':
            cam = _cam_from(acts[0], acts[0].grad)
            p, k = net.p, -(-S // net.p)
            up = F.interpolate(cam[:, None], size=(p, p), mode='bilinear',
                               align_corners=False)[:, 0].cpu()
            canvas = up.view(k, k, p, p).permute(0, 2, 1, 3).reshape(k * p, k * p)[:S, :S]
        else:
            # one encoder call per size present, in size order; inside a size
            # the patches keep the order of coords
            present = [s for s in range(len(net.spec.sizes)) if bool((coords[:, 0] == s).any())]
            for a, s in zip(acts, present):
                sub_ = coords[coords[:, 0] == s].cpu().numpy()
                p = net.spec.sizes[s]
                up = F.interpolate(_cam_from(a, a.grad)[:, None], size=(p, p), mode='bilinear',
                                   align_corners=False)[:, 0].cpu()
                for (_, r, c), u in zip(sub_, up):
                    h_, w_ = min(p, S - r), min(p, S - c)        # padded patches: crop
                    if h_ > 0 and w_ > 0:
                        canvas[r:r + h_, c:c + w_] = torch.maximum(
                            canvas[r:r + h_, c:c + w_], u[:h_, :w_])
        m = float(canvas.max())
        if m > 0:
            canvas = canvas / m
    net.zero_grad(set_to_none=True)
    return canvas.numpy(), target


def gradcam_study(net, ds, conf, split='test', size=112, tol=15, given=None):
    les_index = _ddr_lesion_index(conf) if conf.dataset == 'ddr' else {}
    if hasattr(net, 'reseed'):
        net.reseed(conf.seed + 4000)
    t0 = time.perf_counter()
    ids, cams, targets, rows = [], [], [], []
    acc = defaultdict(lambda: defaultdict(float))
    for i in range(len(ds)):
        cam, tgt = gradcam_one(net, conf, ds[i][0],
                               coords=(given[i][0] if given is not None and i in given else None))
        ids.append(i)
        targets.append(tgt)
        cams.append(np.asarray(Image.fromarray(cam.astype(np.float32)).resize(
            (size, size), Image.BILINEAR), dtype=np.float16))
        masks = _lesion_masks(ds, conf, i, les_index)
        if not masks:
            continue
        tot = float(cam.sum())
        py, px = np.unravel_index(int(cam.argmax()), cam.shape)
        row = {'img': ds.items[i][0].name, 'label': int(ds.items[i][1]), 'target': tgt}
        for t, m in masks.items():
            e = float((cam * m).sum() / tot) if tot > 0 else 0.0
            near = bool(m[max(0, py - tol):py + tol + 1, max(0, px - tol):px + tol + 1].any())
            a = acc[t]
            a['n_img'] += 1
            a['energy'] += e
            a['area'] += float(m.mean())
            a['point'] += float(near)
            row[f'{t}_energy'] = e
            row[f'{t}_point'] = float(near)
            g = 224
            ms = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize((g, g), Image.BOX)) > 0
            if ms.any() and (~ms).any() and tot > 0:
                cs = np.asarray(Image.fromarray(cam.astype(np.float32)).resize((g, g), Image.BILINEAR))
                au = float(roc_auc_score(ms.ravel(), cs.ravel()))
                a['auroc'] += au
                a['n_auroc'] += 1
                row[f'{t}_auroc'] = au
        rows.append(row)
    np.savez_compressed(OUT_SAL / f"{conf.run_name}_gradcam_{split}.npz",
                        ids=np.asarray(ids), cams=np.stack(cams), target=np.asarray(targets),
                        size=np.asarray(size), img_size=np.asarray(conf.img_size))
    if rows:
        pd.DataFrame(rows).to_csv(OUT_EXPE / f"{conf.run_name}_gradcam_{split}.csv", index=False)
    res = {}
    for t, a in sorted(acc.items(), key=lambda kv: (kv[0] not in ('any', 'opacity'), kv[0])):
        n = max(a['n_img'], 1)
        r = {'n_img': int(a['n_img']), 'energy_in_lesion': a['energy'] / n,
             'lesion_area': a['area'] / n, 'pointing_game': a['point'] / n,
             'pixel_auroc': (a['auroc'] / a['n_auroc']) if a['n_auroc'] else float('nan')}
        r['energy_ratio'] = r['energy_in_lesion'] / r['lesion_area'] if r['lesion_area'] else float('nan')
        res[t] = r
        print(f"  grad-cam {split:4s} [{t:7s}] n={r['n_img']:4d}  energy on lesion "
              f"{100*r['energy_in_lesion']:.2f} % (x{r['energy_ratio']:.2f})  "
              f"pointing {100*r['pointing_game']:.0f} %  pixel AUROC {r['pixel_auroc']:.3f}")
    res['s_total'] = round(time.perf_counter() - t0, 1)
    return res


# In[ ]:


# ============================================================
# Data-free benchmark of memory and latency (IPS_MODE=bench)
#
# Random images, no training set needed, a few minutes on one GPU.
# One factor is varied at a time around the reference
# (patch 128, M = 10, I = 128, eager loading):
#   patch size, M, I, loading mode, number of candidates (image twice as
#   large), cascades, baseline resolution.
# It measures what proposition 3 predicts: the selection peak follows
# (M + I) and the patch size, not the number of candidates; with lazy
# loading it does not follow the image size either.
# ============================================================
def _bench_one(tag, conf, n_rep=6):
    conf = Struct(**dict(conf.__dict__, pretrained=False, seed=0,
                         run_name='bench'))
    net = build_model(conf, device)
    crit = IPSLoss(conf.eps)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-5, weight_decay=conf.wd)
    C, S = conf.n_chan_in, conf.img_size
    is_base = conf.model in IMAGE_MODELS
    cost = cost_report(conf, net, verbose=False)
    cost.update(efficiency_profile(conf, net))
    row = dict(tag=tag, dataset=conf.dataset, model=conf.model, img_size=S,
               scales='-'.join(map(str, conf.scales)) if not is_base
               else f"base{conf.get('baseline_size', S)}",
               M=conf.M, I=conf.I, loading=conf.get('loading', 'eager'),
               train_coarse=conf.get('train_coarse', 0), B=conf.B, **cost)
    if CUDA:
        torch.cuda.empty_cache()
        row['weights_mb'] = round(torch.cuda.memory_allocated() / 1e6, 1)
    prof = Profiler()
    labels = torch.randint(0, conf.n_class, (conf.B,), device=device)
    if is_base:
        imgs = torch.randn(conf.B, C, S, S, device=device)
        for _ in range(n_rep):
            net.train()
            with prof.phase('step'):
                opt.zero_grad()
                crit(net(imgs), labels).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            for _ in range(n_rep):
                with prof.phase('infer'):
                    net(imgs[:1])
        row['ms_step'] = round(1000 * prof.t['step'] / n_rep, 2)
        row['ms_infer_img'] = round(1000 * prof.t['infer'] / n_rep, 2)
    else:
        lazy = net.loading == 'lazy'
        imgs = torch.randn(conf.B_seq, C, S, S)
        if not lazy:
            imgs = imgs.to(device)
        net.train()
        net.ips(imgs[0])                                   # warm-up
        for _ in range(n_rep):
            store = _new_store(len(net.spec.sizes))
            for b in range(conf.B):
                img = imgs[b % conf.B_seq]
                with prof.phase('select'):
                    coords = net.ips(img)
                    _fill_store(net, store, img, coords, b)
            with prof.phase('step'):
                opt.zero_grad()
                crit(net.classify(_embed_from_store(net, store, conf.B)),
                     labels).backward()
                opt.step()
            del store
        net.eval()
        with torch.no_grad():
            for _ in range(n_rep):
                with prof.phase('infer'):
                    c = net.ips(imgs[0])
                    net.classify(_embed_single(net, imgs[0], c))
        row['ms_select_img'] = round(1000 * prof.t['select'] / (n_rep * conf.B), 2)
        row['ms_step'] = round(1000 * prof.t['step'] / n_rep, 2)
        row['ms_infer_img'] = round(1000 * prof.t['infer'] / n_rep, 2)
    for k, v in prof.peak.items():
        row[f'peak_{k}_mb'] = round(v / 1e6, 1)
    del net, opt, imgs
    if CUDA:
        torch.cuda.empty_cache()
    return row


def bench(quick=False):
    use_run_dirs('_bench')
    rows = []

    def go(tag, base, model, **kw):
        d = dict(base, name=tag, model=model, **kw)
        if model == 'hierarchical':
            d['scales'] = list(d['cascade'])
        if model == 'baseline':
            d['scales'] = [128]
        try:
            rows.append(_bench_one(tag, Struct(**d), n_rep=2 if quick else 6))
            r = rows[-1]
            print(f"  {tag:34s} select {r.get('peak_select_mb', 0):8.1f} MB  "
                  f"step {r.get('peak_step_mb', 0):8.1f} MB  "
                  f"infer {r.get('peak_infer_mb', 0):8.1f} MB")
        except Exception as exc:                       # e.g. out of memory
            print(f"  {tag:34s} FAILED {exc!r}")
            rows.append(dict(tag=tag, error=repr(exc)))
            if CUDA:
                torch.cuda.empty_cache()

    for nm, base, ps, cas, bs in [
            ('ddr', DDR_BASE, [224, 128, 112, 64, 56, 32, 28],
             [[224, 112], [224, 56], [224, 28]], [896, 448, 224]),
            ('rsna', RSNA_BASE, [256, 128, 64, 32],
             [[256, 128], [256, 64], [256, 32]], [1024, 512, 256])]:
        if quick:
            ps, cas, bs = ps[1:3], cas[:1], bs[1:]
        for p in ps:
            go(f'{nm}_ips{p}', base, 'selector', scales=[p])
        for M in ([5] if quick else [2, 5, 20, 40]):
            go(f'{nm}_ips128_M{M}', base, 'selector', scales=[128], M=M)
        for I in ([16] if quick else [8, 16, 32, 64, 256]):
            go(f'{nm}_ips128_I{I}', base, 'selector', scales=[128], I=I)
        go(f'{nm}_ips128_lazy', base, 'selector', scales=[128], loading='lazy')
        # four times the candidates, same M, I and patch size
        S2 = 2 * base['img_size']
        if not quick:
            go(f'{nm}_ips128_x2img_eager', dict(base, img_size=S2), 'selector', scales=[128])
            go(f'{nm}_ips128_x2img_lazy', dict(base, img_size=S2), 'selector',
               scales=[128], loading='lazy')
            go(f'{nm}_ms128_64', base, 'selector', scales=[128, 64])
        for c in cas:
            go(f'{nm}_h{c[0]}_{c[1]}', base, 'hierarchical', cascade=c, M_keep=[4])
            go(f'{nm}_h{c[0]}_{c[1]}_tc', base, 'hierarchical', cascade=c,
               M_keep=[4], train_coarse=4)
        for s in bs:
            go(f'{nm}_base{s}', base, 'baseline', baseline_size=s)
        go(f'{nm}_mil128', base, 'mil', scales=[128])
        go(f'{nm}_ips128_sa', base, 'selector', scales=[128], aggregator='self')
    df = pd.DataFrame(rows)
    df.to_csv(OUT_EXPE / 'bench.csv', index=False)
    print(f"  bench -> {OUT_EXPE / 'bench.csv'}  ({len(df)} rows)")
    return df

# In[ ]:


# ============================================================
# Runner
# ============================================================
FORCE = bool(_env_int('IPS_FORCE', 0))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def _conf_dict(conf):
    return {k: v for k, v in conf.__dict__.items() if k != 'data_dir'}


def post_training(net, conf, criterion, ds_val, ds_test, tag=''):
    """Everything measured with the weights of the best epoch."""
    is_sel = conf.model in ('selector', 'hierarchical')
    out = {}
    val_m, val_o = evaluate_split(net, ds_val, conf, 'val', criterion, tag=tag)
    test_m, test_o = evaluate_split(net, ds_test, conf, 'test', criterion, tag=tag)
    out['val'], out['test'] = val_m, test_m
    if conf.task_type == 'binary':
        thr = youden_threshold(val_o['y_true'], np.asarray(val_o['y_prob'])[:, 1])
        out['threshold_from_val'] = thr
        out['test_at_val_threshold'] = compute_metrics(
            test_o['y_true'], test_o['y_pred'], test_o['y_prob'], 'binary', thr)
    def _given(o):
        if o['coords'] is None:
            return None
        z = o['zones']
        return {i: (o['coords'][j], (z[j] if z is not None else None))
                for j, i in enumerate(o['ids'])}
    g_val, g_test = _given(val_o), _given(test_o)
    key = conf.main_metric
    print(f"  best epoch weights : val {key} {val_m.get(key, float('nan')):.4f} | "
          f"TEST {key} {test_m.get(key, float('nan')):.4f}  (n={test_m['n']})")
    jobs = []
    if is_sel and conf.get('selection', 'attention') == 'attention':
        jobs.append(('selection_test', lambda: selection_study(net, ds_test, conf, device, criterion)))
    if conf.model != 'baseline':          # selectors and MIL (its top-M by attention)
        jobs += [('lesions_test', lambda: lesion_study(net, ds_test, conf, device, split='test', given=g_test)),
                 ('lesions_val', lambda: lesion_study(net, ds_val, conf, device, split='val', given=g_val))]
    jobs.append(('gradcam_test', lambda: gradcam_study(net, ds_test, conf, split='test', given=g_test)))
    if jobs:
        for nm, fn in jobs:
            try:
                out[nm] = fn()
            except Exception as exc:        # the run itself is already saved
                import traceback
                traceback.print_exc()
                out[nm] = {'error': repr(exc)}
    return out


def run(name, seed, n_epoch=None):
    conf = make_conf(name, seed)
    if n_epoch is not None:
        conf.n_epoch = n_epoch
    use_run_dirs(name)
    res_path = OUT_EXPE / f"{conf.run_name}_result.json"
    if res_path.exists() and not FORCE:
        print(f"[skip] {conf.run_name} : result exists (IPS_FORCE=1 to redo)")
        return None
    # if the same configuration and seed is already running in another job,
    # skip it instead of writing the same files twice
    _lock = open(OUT_CKPT / f"{conf.run_name}.lock", 'w')
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[skip] {conf.run_name} : already running in another job")
        return None
    set_seed(conf.seed)
    print(f"\n{'='*72}\n{conf.run_name}  ({conf.model}, {conf.dataset})  "
          f"{REGISTRY[name][2]}\n{'='*72}")

    ds_tr = build_dataset(conf, 'train')
    ds_va = build_dataset(conf, 'val')
    ds_te = build_dataset(conf, 'test')
    train_loader = build_loader(conf, ds_tr, train=True)
    val_loader = build_loader(conf, ds_va, train=False)

    net = build_model(conf, device)
    cost = cost_report(conf, net)
    cost.update(efficiency_profile(conf, net))
    print(f"  parameters         : {cost['params']/1e6:.2f} M"
          + (f" | GFLOPs per image : {cost['gflops_infer']:.2f}" if 'gflops_infer' in cost else ''))
    criterion = IPSLoss(conf.eps)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0, weight_decay=conf.wd)
    is_base = conf.model in IMAGE_MODELS
    key = conf.main_metric

    w_path = OUT_CKPT / f"{conf.run_name}_weights.pt"
    s_path = OUT_CKPT / f"{conf.run_name}_state.pt"
    history, best_val, best_ep, start_ep, co2_prev = [], -float('inf'), 0, 0, 0.0
    if s_path.exists() and not FORCE:
        st = torch.load(s_path, map_location=device, weights_only=False)
        if st.get('n_epoch') == conf.n_epoch:
            net.load_state_dict(st['model'])
            optimizer.load_state_dict(st['optim'])
            history, best_val, best_ep = st['history'], st['best_val'], st['best_ep']
            start_ep, co2_prev = st['epoch'], st.get('co2', 0.0)
            print(f"  [resume] from epoch {start_ep}")

    tracker = None
    try:
        from codecarbon import EmissionsTracker
        _co2_dir = OUT_HIST / 'codecarbon'
        _co2_dir.mkdir(parents=True, exist_ok=True)
        tracker = EmissionsTracker(output_dir=str(_co2_dir), log_level='error')
        tracker.start()
    except Exception as exc:
        print(f"  [info] codecarbon unavailable ({exc})")

    for ep in range(start_ep, conf.n_epoch):
        tr = run_epoch(net, train_loader, conf, device, optimizer, ep,
                       criterion, is_base)
        if not is_base:
            net.reseed(conf.seed + 1000)        # same validation pass every epoch
        va = run_epoch(net, val_loader, conf, device, None, ep, criterion, is_base)

        tr_m = compute_metrics(tr['y_true'], tr['y_pred'], tr['y_prob'], conf.task_type)
        va_m = compute_metrics(va['y_true'], va['y_pred'], va['y_prob'], conf.task_type)
        tr_m['loss'], va_m['loss'] = tr['loss'], va['loss']
        history.append({'epoch': ep + 1, 'lr': optimizer.param_groups[0]['lr'],
                        'train': tr_m, 'val': va_m,
                        'prof': {**tr['prof'].summary('train_'),
                                 **va['prof'].summary('val_')}})

        score, shown = va_m.get(key, va_m['accuracy']), key
        if isinstance(score, float) and math.isnan(score):
            score, shown = va_m['accuracy'], key + '~acc'
        flag = ''
        if score > best_val:
            best_val, best_ep, flag = score, ep + 1, '  *'
            torch.save(net.state_dict(), w_path)
        p = history[-1]['prof']
        print(f"  ep {ep+1:3d}/{conf.n_epoch}  train {tr['loss']:.4f}  "
              f"val {va['loss']:.4f}  {shown} {score:.4f}{flag}   "
              f"[data {p.get('train_t_data_s', 0):.0f}s "
              f"aug {p.get('train_t_augment_s', 0):.0f}s "
              f"select {p.get('train_t_select_s', 0):.0f}s "
              f"step {p.get('train_t_step_s', 0):.0f}s | "
              f"peak {p.get('train_peak_mb', 0):.0f} MB]")
        torch.save({'epoch': ep + 1, 'n_epoch': conf.n_epoch,
                    'model': net.state_dict(), 'optim': optimizer.state_dict(),
                    'history': history, 'best_val': best_val,
                    'best_ep': best_ep, 'co2': co2_prev}, s_path)

    co2 = co2_prev
    energy = None
    if tracker is not None:
        try:
            co2 += float(tracker.stop() or 0.0)
            energy = float(tracker.final_emissions_data.energy_consumed)
        except Exception:
            pass

    def tot(k):
        return float(sum(h['prof'].get(k, 0.0) for h in history))

    def peak(k):
        return float(max((h['prof'].get(k, 0.0) for h in history), default=0.0))

    n_tr = max(sum(h['prof'].get('train_n_img', 0) for h in history), 1)
    timing = {
        'train_t_data_s': tot('train_t_data_s'),
        'train_t_augment_s': tot('train_t_augment_s'),
        'train_t_select_s': tot('train_t_select_s'),
        'train_t_step_s': tot('train_t_step_s'),
        'val_t_total_s': tot('val_t_data_s') + tot('val_t_select_s') + tot('val_t_step_s'),
        'train_model_ms_per_img': 1000 * (tot('train_t_select_s') + tot('train_t_step_s')) / n_tr,
        'train_peak_mb': peak('train_peak_mb'),
        'train_peak_select_mb': peak('train_peak_select_mb'),
        'train_peak_step_mb': peak('train_peak_step_mb'),
        'val_peak_mb': peak('val_peak_mb'),
        'total_co2_kg': co2, 'energy_kwh_last_session': energy,
        'gpu': torch.cuda.get_device_name(0) if CUDA else 'cpu',
    }

    net.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
    best_row = history[best_ep - 1]
    result = {
        'suite': SUITE_VERSION, 'run_name': conf.run_name, 'name': name,
        'seed': conf.seed, 'date': time.strftime('%Y-%m-%d %H:%M'),
        'note': REGISTRY[name][2], 'groups': sorted(REGISTRY[name][1]),
        'config': _conf_dict(conf), 'cost': cost,
        'n_train': len(ds_tr), 'n_val': len(ds_va), 'n_test': len(ds_te),
        'best_epoch': best_ep, 'main_metric': key,
        'val_at_best_epoch': best_row['val'],
        'train_at_best_epoch': best_row['train'],
        'final_epoch': {'train': history[-1]['train'], 'val': history[-1]['val']},
        'timing': timing, 'history': history,
    }
    with open(OUT_HIST / f"{conf.run_name}_history.json", 'w') as f:
        json.dump(_jsonable(history), f)
    with open(res_path.with_suffix('.partial.json'), 'w') as f:
        json.dump(_jsonable(result), f)
    result.update(post_training(net, conf, criterion, ds_va, ds_te))
    with open(res_path, 'w') as f:
        json.dump(_jsonable(result), f, indent=1)
    res_path.with_suffix('.partial.json').unlink(missing_ok=True)
    s_path.unlink(missing_ok=True)
    print(f"  result : {res_path}")
    return result


def eval_v2(run_name):
    """Score the saved weights of a v2 run on the held-out images."""
    conf = make_v2_conf(run_name)
    ckpt = _env_path('IPS_V2_CKPT', RESULTS) / f"{run_name}_weights.pt"
    use_run_dirs(run_name)
    res_path = OUT_EXPE / f"{run_name}_evalv3.json"
    if not ckpt.exists():
        print(f"[skip] {run_name} : no weights at {ckpt}")
        return None
    if res_path.exists() and not FORCE:
        print(f"[skip] {run_name} : already evaluated")
        return None
    print(f"\n{'='*72}\n{run_name}  (v2 weights, evaluation only)\n{'='*72}")
    set_seed(conf.seed)
    net = build_model(conf, device)
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    cost = cost_report(conf, net)
    cost.update(efficiency_profile(conf, net))
    ds_va = build_dataset(conf, 'val')
    ds_te = build_dataset(conf, 'test')
    result = {'suite': 'v2-weights', 'run_name': run_name, 'seed': conf.seed,
              'config': _conf_dict(conf), 'cost': cost,
              # v2 chose the epoch on DDR valid (test never seen: clean). On
              # RSNA, v2 chose its epoch on images that are now in val and
              # test, and trained on 5 % of the patients now in val or test.
              'test_is_clean': conf.dataset == 'ddr'}
    result.update(post_training(net, conf, IPSLoss(conf.eps), ds_va, ds_te,
                                tag='_evalv3'))
    with open(res_path, 'w') as f:
        json.dump(_jsonable(result), f, indent=1)
    print(f"  result : {res_path}")
    return result

# ## What to run
# 
# | variable | meaning | default |
# |---|---|---|
# | `IPS_MODE` | `train`, `bench`, `eval_v2`, `list` | `train` |
# | `IPS_RUNS` | run names and/or group names, comma separated | `core` |
# | `IPS_SEEDS` | seeds, comma separated | `42` |
# | `IPS_EPOCHS` | override the number of epochs | 50 |
# | `IPS_CACHE_DIR` | folder for the decoded-image cache | none |
# | `IPS_CACHE` | data sets to cache (`none` for no cache) | `ddr` |
# | `IPS_WORKERS` | DataLoader workers | 8 |
# | `IPS_V2_CKPT` | folder of the v2 weights (`eval_v2`) | checkpoints |
# | `IPS_FORCE` | `1` to redo a finished run | 0 |
# | `IPS_SMOKE` | keep N images per split (dry run) | 0 |
# | `IPS_DDR_GEOMETRY` | `retina`, `stretch` (the v2 input) or `crop` | `retina` |
# 
# A finished run is skipped; an interrupted run resumes at its last epoch.

# In[ ]:


MODE = _env('IPS_MODE', 'train')
_sel = [k for k in _env('IPS_RUNS', 'core').split(',') if k.strip()]
SEEDS = [int(s) for s in _env('IPS_SEEDS', '42').split(',') if s.strip()]
EPOCHS = _env_int('IPS_EPOCHS', 0) or None


def _expand(sel, table, groups=True):
    out = []
    for k in sel:
        k = k.strip()
        if k == 'all':
            out += list(table)
        elif groups and k in GROUPS:
            out += [n for n, (_, gs, _) in REGISTRY.items() if k in gs]
        elif k in table:
            out.append(k)
        else:
            raise ValueError(f"unknown run or group : {k}. "
                             f"Groups : {GROUPS}. Runs : {list(table)}")
    return list(dict.fromkeys(out))


if MODE == 'list':
    rows = []
    for n, (d, gs, note) in REGISTRY.items():
        c = make_conf(n, 0)
        c.pretrained = False
        net = build_model(c, torch.device('cpu'))
        k = cost_report(c, net, verbose=False)
        rows.append(dict(run=n, groups='+'.join(sorted(gs)), M=c.M,
                         candidates=k['n_candidates'],
                         explore_x_img=round(k['explore_encoded_px'] / k['img_px'], 3),
                         train_px=k['exploit_px_max'],
                         train_pct_img=round(100 * k['exploit_px_max'] / k['img_px'], 1),
                         note=note))
        del net
    _df = pd.DataFrame(rows)
    _df.to_csv(RESULTS / 'run_list.csv', index=False)
    print(_df.to_string(index=False))
elif MODE == 'bench':
    bench(quick=bool(_env_int('IPS_BENCH_QUICK', 0)))
elif MODE == 'eval_v2':
    _todo = _expand(_sel if _env('IPS_RUNS') else ['all'], V2, groups=False)
    for _k in _todo:
        try:
            eval_v2(_k)
        except Exception as _exc:
            import traceback
            print(f"\n[FAILED] {_k} : {_exc!r}")
            traceback.print_exc()
elif MODE == 'train':
    TO_RUN = _expand(_sel, REGISTRY)
    print(f"To run : {TO_RUN}  seeds {SEEDS}"
          + (f"  ({EPOCHS} epochs)" if EPOCHS else ""))
    _failed = []
    for _seed in SEEDS:
        for _key in TO_RUN:
            try:
                run(_key, _seed, n_epoch=EPOCHS)
            except Exception as _exc:
                import traceback
                _failed.append((_key, _seed, repr(_exc)))
                print(f"\n[FAILED] {_key} seed {_seed} : {_exc!r}")
                traceback.print_exc()
                print("  -> continuing with the next run\n")
    if _failed:
        print(f"\n{len(_failed)} run(s) failed :")
        for _k, _s, _e in _failed:
            print(f"  {_k} s{_s} : {_e}")
else:
    raise ValueError(f"IPS_MODE={MODE}")
