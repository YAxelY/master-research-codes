"""What is the selection worth? topm          : the learned selection scale_matched : same count per scale, random positions uniform       : everything at random among the N candidates uniform -> scale_matched : the choice of scales scale_matched -> topm    : the choice of positions"""

import numpy as np
import torch

from ips_suite.data.datasets import build_loader
from ips_suite.evaluation.metrics import compute_metrics
from ips_suite.models.batching import _embed_from_store, _fill_store, _new_store


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
