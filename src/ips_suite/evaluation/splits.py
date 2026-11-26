"""Evaluation of one split with the weights of the best epoch. Saves, per image: index, path, label, probabilities, selected patches."""

import numpy as np
import time

from ips_suite.config.environment import device
from ips_suite.data.datasets import build_loader
from ips_suite.evaluation.metrics import bootstrap_ci, compute_metrics
from ips_suite.models.factory import score_maps
from ips_suite.training.epoch import IMAGE_MODELS, run_epoch
from ips_suite.config import experiments


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
            experiments.OUT_PRED / f"{conf.run_name}{tag}_{split}.npz",
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
