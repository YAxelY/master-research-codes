# ============================================================
# lesion-maps.py : how much of the image the lesions cover, and where they are
#
# Paste this file in ONE Kaggle cell (CPU is enough) with the two data sets
# attached:
#   samriddhibagchi/ddr-dataset-credits-to-authors
#   iamtapendu/rsna-pneumonia-processed-dataset
# It trains nothing. Result: /kaggle/working/lesion_maps.zip (a few MB).
# Runtime: about 5 to 10 minutes.
#
# DDR  : every annotated image (masks of MA, HE, EX, SE). For each lesion type
#        and for all types together:
#          - share of the image and share of the retina covered, per image
#          - a 128 x 128 map of where the lesions are, in three frames:
#              stretch     the whole photograph squeezed to a square (v2 input)
#              retina      cropped to the retina, pixels square
#              retina_disc same, mirrored when needed so that the optic disc
#                          is always on the right (left and right eyes align)
#          - share of grid cells that contain a lesion, for each patch size
#          - distance of the lesion pixels to the centre of the retina
# RSNA : every positive patient (boxes). Same quantities, one frame.
# ============================================================
import os, json, time, hashlib, zipfile, traceback
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
INPUT = Path(os.environ.get('PROBE_INPUT', '/kaggle/input'))
OUT = Path(os.environ.get('PROBE_OUT', '/kaggle/working')) / 'lesion_maps'
OUT.mkdir(parents=True, exist_ok=True)
G = 128                                   # side of the maps
S_DDR, S_RSNA, SPLIT_SEED = 896, 1024, 42
DDR_PATCH = (224, 128, 112, 64, 56, 32, 28, 16, 14)     # at 896 px
RSNA_PATCH = (256, 128, 64, 32)                         # at 1024 px
T0 = time.time()
REPORT = {}


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def find_root(marker):
    for dp, dn, fn in os.walk(INPUT):
        if marker in dn or marker in fn:
            return Path(dp)
    return None


def cover(mask, frame, size):
    """Share of each cell of a size x size grid covered by the mask.

    frame = None (whole array) or (left, top, side) in native pixels; outside
    the array the crop is empty (zero).
    """
    im = Image.fromarray(mask.astype(np.uint8) * 255)
    if frame is not None:
        l, t, s = frame
        im = im.crop((l, t, l + s, t + s))
    return np.asarray(im.resize((size, size), Image.BOX), dtype=np.float32) / 255.0


def retina_frame(gray):
    """(left, top, side) of the bounding square of the non-black region."""
    m = gray > 12
    h, w = gray.shape
    if not m.any():
        return (0, 0, max(h, w)), m
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    l, r, t, b = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    side = int(max(r - l, b - t))
    return (int(round((l + r) / 2 - side / 2)), int(round((t + b) / 2 - side / 2)), side), m


# ================================================================== DDR
def ddr():
    g = find_root('DR_grading')
    if g is None:
        REPORT['ddr'] = {'error': 'DR_grading not found'}
        return
    gr, seg = g / 'DR_grading', g / 'lesion_segmentation'
    grade_of, path_of = {}, {}
    for sp in ('train', 'valid', 'test'):
        f = gr / f'{sp}.txt'
        if f.exists():
            for line in open(f):
                p = line.split()
                if len(p) >= 2:
                    grade_of[Path(p[0]).stem] = (sp, int(p[1]))
                    path_of[Path(p[0]).stem] = gr / sp / p[0]
    masks = defaultdict(dict)              # stem -> {type: path}
    seg_img, seg_split = {}, {}
    for sp in ('train', 'valid', 'test'):
        d = seg / sp / 'image'
        if d.exists():
            for p in d.iterdir():
                seg_img[p.stem] = p
        for lab in ('label', 'segmentation label'):
            d = seg / sp / lab
            if d.exists():
                for td in sorted(x for x in d.iterdir() if x.is_dir()):
                    for mp in td.glob('*'):
                        masks[mp.stem][td.name] = mp
                        seg_split[mp.stem] = sp
    types = sorted({t for d in masks.values() for t in d})
    log(f'DDR: {len(masks)} annotated images, types {types}')
    frames = ('stretch', 'retina', 'retina_disc')
    keys = types + ['any']
    cov_sum = {f: {t: np.zeros((G, G), np.float64) for t in keys} for f in frames}
    pres_sum = {f: {t: np.zeros((G, G), np.float64) for t in keys} for f in frames}
    n_with = {t: 0 for t in keys}
    mean_img = {f: np.zeros((G, G), np.float64) for f in frames}
    radial = {t: np.zeros(20, np.float64) for t in keys}       # r / R in 20 rings
    rows, n = [], 0
    for stem, files in sorted(masks.items()):
        ip = seg_img.get(stem) or path_of.get(stem)
        if ip is None or not Path(ip).exists():
            continue
        try:
            gray = np.asarray(Image.open(ip).convert('L'))
        except Exception:
            continue
        h, w = gray.shape
        fr, retina = retina_frame(gray)
        # optic disc: brightest region of a small smoothed copy, inside the retina
        small = np.asarray(Image.fromarray(gray).crop((fr[0], fr[1], fr[0] + fr[2], fr[1] + fr[2]))
                           .resize((32, 32), Image.BOX), dtype=np.float32)
        disc_col = int(np.unravel_index(np.argmax(small), small.shape)[1])
        flip = disc_col < 16                        # disc on the left -> mirror
        row = dict(stem=stem, seg_split=seg_split.get(stem), w=w, h=h,
                   grading_split=grade_of.get(stem, ('absent', -1))[0],
                   grade=grade_of.get(stem, ('absent', -1))[1],
                   retina_frac=float(retina.mean()), disc_col=disc_col, flipped=bool(flip))
        ms = {}
        for t, mp in files.items():
            try:
                m = np.asarray(Image.open(mp).convert('L')) > 0
            except Exception:
                continue
            if m.shape != gray.shape:
                m = np.asarray(Image.fromarray(m.astype(np.uint8) * 255)
                               .resize((w, h), Image.NEAREST)) > 0
            ms[t] = m
        if not ms:
            continue
        ms['any'] = np.logical_or.reduce(list(ms.values()))
        for f in frames:
            gi = np.asarray(Image.fromarray(gray).crop((fr[0], fr[1], fr[0] + fr[2], fr[1] + fr[2]))
                            .resize((G, G), Image.BOX) if f != 'stretch'
                            else Image.fromarray(gray).resize((G, G), Image.BOX), dtype=np.float64)
            if f == 'retina_disc' and flip:
                gi = gi[:, ::-1]
            mean_img[f] += gi
        cy, cx, R = fr[1] + fr[2] / 2, fr[0] + fr[2] / 2, fr[2] / 2
        for t in keys:
            m = ms.get(t)
            if m is None or not m.any():
                row[f'{t}_frac_img'] = 0.0
                row[f'{t}_frac_retina'] = 0.0
                continue
            n_with[t] += 1
            row[f'{t}_frac_img'] = float(m.mean())
            row[f'{t}_frac_retina'] = float(m.sum() / max(retina.sum(), 1))
            yy, xx = np.nonzero(m)
            row[f'{t}_cx'] = float((xx.mean() - fr[0]) / fr[2])     # 0..1 in the retina frame
            row[f'{t}_cy'] = float((yy.mean() - fr[1]) / fr[2])
            rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / R
            radial[t] += np.histogram(rr, bins=20, range=(0, 1.0))[0]
            for f in frames:
                c = cover(m, None if f == 'stretch' else fr, G)
                if f == 'retina_disc' and flip:
                    c = c[:, ::-1]
                cov_sum[f][t] += c
                pres_sum[f][t] += (c > 0)
            # share of patch-grid cells that contain a lesion (exact tiling of 896)
            for geo, frame in (('stretch', None), ('retina', fr)):
                for p in DDR_PATCH:
                    k = S_DDR // p
                    row[f'{t}_{geo}_cells{p}'] = float((cover(m, frame, k) > 0).mean())
        rows.append(row)
        n += 1
        if n % 100 == 0:
            log(f'DDR {n} images')
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'ddr_per_image.csv', index=False)
    arr = {}
    for f in frames:
        arr[f'mean_img_{f}'] = (mean_img[f] / max(n, 1)).astype(np.float32)
        for t in keys:
            arr[f'cov_{f}_{t}'] = (cov_sum[f][t] / max(n, 1)).astype(np.float32)      # mean coverage over ALL images
            arr[f'pres_{f}_{t}'] = (pres_sum[f][t] / max(n, 1)).astype(np.float32)    # share of images with a lesion there
    for t in keys:
        arr[f'radial_{t}'] = radial[t]
    np.savez_compressed(OUT / 'ddr_maps.npz', **arr)
    REPORT['ddr'] = dict(n_images=n, types=types, n_with_type=n_with,
                         by_grading_split=df.grading_split.value_counts().to_dict(),
                         flipped_share=float(df.flipped.mean()), grid=G,
                         patch_sizes=list(DDR_PATCH))
    log(f'DDR done: {n} images')


# ================================================================= RSNA
def split_of(pid):
    h = int(hashlib.md5((str(pid) + str(SPLIT_SEED)).encode()).hexdigest(), 16) % 100
    return 'train' if h >= 30 else ('val' if h < 15 else 'test')


def rsna():
    g = find_root('stage2_train_metadata.csv')
    if g is None:
        REPORT['rsna'] = {'error': 'stage2_train_metadata.csv not found'}
        return
    raw = pd.read_csv(g / 'stage2_train_metadata.csv')
    pos = raw[raw.Target == 1].dropna(subset=['x', 'y', 'width', 'height'])
    n_pat = int(raw.patientId.nunique())
    cov = np.zeros((G, G), np.float64)
    rows, boxes = [], []
    for pid, d in pos.groupby('patientId'):
        m = np.zeros((S_RSNA, S_RSNA), bool)
        for x, y, w, h in d[['x', 'y', 'width', 'height']].to_numpy():
            m[max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)] = True
            boxes.append(dict(cx=(x + w / 2) / S_RSNA, cy=(y + h / 2) / S_RSNA,
                              w=w / S_RSNA, h=h / S_RSNA))
        cov += cover(m, None, G)
        row = dict(split=split_of(pid), n_boxes=len(d), frac_img=float(m.mean()),
                   left_half=float(m[:, :S_RSNA // 2].mean() * 2 > 0),
                   right_half=float(m[:, S_RSNA // 2:].mean() * 2 > 0))
        for c in ('age', 'sex', 'position'):
            if c in d.columns:
                row[c] = d[c].iloc[0]
        for p in RSNA_PATCH:
            row[f'cells{p}'] = float((cover(m, None, S_RSNA // p) > 0).mean())
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / 'rsna_per_patient.csv', index=False)
    pd.DataFrame(boxes).to_csv(OUT / 'rsna_box_centres.csv', index=False)
    arr = {'pres_opacity': (cov / max(len(df), 1)).astype(np.float32)}
    # average radiograph, positives and negatives, as a background for the map
    img_dir = g / 'Training' / 'Images'
    if img_dir.exists():
        rng = np.random.default_rng(0)
        for name, ids in (('pos', pos.patientId.unique()),
                          ('neg', raw[raw.Target == 0].patientId.unique())):
            acc, k = np.zeros((G, G), np.float64), 0
            for pid in rng.choice(ids, min(400, len(ids)), replace=False):
                p = img_dir / f'{pid}.png'
                if p.exists():
                    acc += np.asarray(Image.open(p).convert('L').resize((G, G), Image.BOX), dtype=np.float64)
                    k += 1
            arr[f'mean_img_{name}'] = (acc / max(k, 1)).astype(np.float32)
    np.savez_compressed(OUT / 'rsna_maps.npz', **arr)
    REPORT['rsna'] = dict(n_patients=n_pat, n_positive=int(len(df)), n_boxes=int(len(pos)),
                          positive_share=float(len(df) / n_pat), grid=G,
                          patch_sizes=list(RSNA_PATCH),
                          by_split=df.split.value_counts().to_dict())
    log(f'RSNA done: {len(df)} positive patients')


for fn in (ddr, rsna):
    try:
        fn()
    except Exception as exc:
        traceback.print_exc()
        REPORT[fn.__name__ + '_error'] = {'error': repr(exc), 'trace': traceback.format_exc()}

json.dump(REPORT, open(OUT / 'report.json', 'w'), indent=1, default=str)
zpath = OUT.parent / 'lesion_maps.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(OUT.parent))
log(f"done -> {zpath}  ({zpath.stat().st_size/1e6:.1f} MB)")
