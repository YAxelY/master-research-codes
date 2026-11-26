"""Model construction and the cost of a configuration, known before training"""

import numpy as np
import torch

from ips_suite.data.datasets import build_loader
from ips_suite.models.batching import _embed_single
from ips_suite.models.cascade import HMSIPSNet
from ips_suite.models.reference import BaselineNet, MILNet
from ips_suite.models.selector import SelectorNet


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
