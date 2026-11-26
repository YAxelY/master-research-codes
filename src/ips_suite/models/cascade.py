"""H-MS-IPS: cascade of stages, from the coarsest to the finest conf.cascade      = [224, 56]   sizes, coarse to fine conf.M_keep       = [4]         how many to keep at each stage but the last conf.M            = 10          final budget conf.train_coarse = 0           v2 behaviour: only the final patches train = k           the k best coarse zones join the training bag The retained zones have a fixed area, so cutting them finer multiplies the pool of the next stage without changing the number of pixels encoded."""

import numpy as np
import torch

from ips_suite.config.environment import Struct
from ips_suite.models.selector import SelectorNet


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
