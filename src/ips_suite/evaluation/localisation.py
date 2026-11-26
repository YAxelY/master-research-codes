"""Do the selected patches fall on lesions? For every annotated image of the split and every lesion type: hit        share of the M selected patches that touch the lesion chance     the same share for a patch drawn uniformly at the same scale, computed EXACTLY by enumerating all the candidates cover      share of the lesion pixels inside the selected patches area       share of the image inside the selected patches (cover / area is the enrichment in pixels) For a cascade, the same for the zones kept by the coarse stage: if the coarse stage drops the lesion, the fine stage cannot recover it."""

import numpy as np
import pandas as pd
import time
import torch
import torch.nn.functional as F
from PIL import Image
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import roc_auc_score

from ips_suite.config.environment import device
from ips_suite.models.batching import _embed_single
from ips_suite.training.epoch import IMAGE_MODELS
from ips_suite.config import experiments


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
            experiments.OUT_EXPE / f"{conf.run_name}_lesions_{split or 'all'}.csv", index=False)
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
    np.savez_compressed(experiments.OUT_SAL / f"{conf.run_name}_gradcam_{split}.npz",
                        ids=np.asarray(ids), cams=np.stack(cams), target=np.asarray(targets),
                        size=np.asarray(size), img_size=np.asarray(conf.img_size))
    if rows:
        pd.DataFrame(rows).to_csv(experiments.OUT_EXPE / f"{conf.run_name}_gradcam_{split}.csv", index=False)
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
