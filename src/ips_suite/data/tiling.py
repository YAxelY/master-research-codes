"""Geometry: exact tiling and vectorised extraction A candidate is a triple (scale id, row, column). The candidate table holds no pixel: N * 3 integers. Pixels are read only for the candidates asked for."""

import numpy as np
import torch


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
