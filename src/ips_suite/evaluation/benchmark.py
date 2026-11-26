"""Data-free benchmark of memory and latency (IPS_MODE=bench) Random images, no training set needed, a few minutes on one GPU. One factor is varied at a time around the reference (patch 128, M = 10, I = 128, eager loading): patch size, M, I, loading mode, number of candidates (image twice as large), cascades, baseline resolution. It measures what proposition 3 predicts: the selection peak follows (M + I) and the patch size, not the number of candidates; with lazy loading it does not follow the image size either."""

import pandas as pd
import torch

from ips_suite.config.environment import CUDA, Struct, device
from ips_suite.config.experiments import DDR_BASE, RSNA_BASE, use_run_dirs
from ips_suite.evaluation.metrics import IPSLoss
from ips_suite.models.batching import _embed_from_store, _embed_single, _fill_store, _new_store
from ips_suite.models.factory import build_model, cost_report, efficiency_profile
from ips_suite.training.epoch import IMAGE_MODELS
from ips_suite.training.profiler import Profiler
from ips_suite.config import experiments


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
    df.to_csv(experiments.OUT_EXPE / 'bench.csv', index=False)
    print(f"  bench -> {experiments.OUT_EXPE / 'bench.csv'}  ({len(df)} rows)")
    return df
