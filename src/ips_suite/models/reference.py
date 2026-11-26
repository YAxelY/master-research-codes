"""Baseline: end-to-end ResNet-18 on the (possibly downsampled) image. Same backbone as the patch models: only the selection differs."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

from ips_suite.data.tiling import TileSpec
from ips_suite.models.encoder import PatchEncoder


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
