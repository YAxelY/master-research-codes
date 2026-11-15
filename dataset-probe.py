# ============================================================
# dataset-probe.py : facts about DDR and RSNA, exported as one zip
#
# Paste this file in ONE Kaggle cell (CPU is enough, no GPU needed) with the
# two data sets attached:
#   samriddhibagchi/ddr-dataset-credits-to-authors
#   iamtapendu/rsna-pneumonia-processed-dataset
# It trains nothing and writes nothing outside /kaggle/working.
# Result: /kaggle/working/dataset_probe.zip  (a few MB). Download it from
# the Output panel.
#
# What it collects
#   DDR  : folder tree, split sizes and grades, width / height / aspect
#          ratio of EVERY image, where the retina sits in the frame (black
#          margins), every lesion (type, area, diameter) from the masks,
#          which grading split each annotated image belongs to,
#          decoding time per image, a few thumbnails with lesions drawn.
#   RSNA : metadata columns, labels, boxes (size, count per patient),
#          image sizes, what the Training/Masks folder really contains,
#          the train / val / test counts of the v3 split, decoding time,
#          a few thumbnails with boxes drawn.
# Runtime: about 10 to 20 minutes.
# ============================================================
import os, json, time, hashlib, zipfile, traceback
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

try:
    from scipy import ndimage
except Exception:                                   # scipy is on Kaggle
    ndimage = None

Image.MAX_IMAGE_PIXELS = None
INPUT = Path(os.environ.get('PROBE_INPUT', '/kaggle/input'))
OUT = Path(os.environ.get('PROBE_OUT', '/kaggle/working')) / 'dataset_probe'
(OUT / 'thumbs').mkdir(parents=True, exist_ok=True)
S_DDR, S_RSNA, SPLIT_SEED = 896, 1024, 42
REPORT, T0 = {}, time.time()
rng = np.random.default_rng(0)


def log(*a):
    print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def find_root(marker):
    """First folder under INPUT that contains `marker` (file or folder)."""
    for dp, dn, fn in os.walk(INPUT):
        if marker in dn or marker in fn:
            return Path(dp)
    return None


def tree(root, depth=4, max_entries=40):
    """Folder tree with file counts and extensions, no file names."""
    lines = []
    root = Path(root)
    for dp, dn, fn in os.walk(root):
        rel = Path(dp).relative_to(root)
        if len(rel.parts) > depth:
            dn[:] = []
            continue
        dn.sort()
        ext = Counter(Path(f).suffix.lower() for f in fn)
        lines.append(f"{'  ' * len(rel.parts)}{rel.name or root.name}/  "
                     + (', '.join(f"{n} {e or '(none)'}" for e, n in ext.most_common(6))
                        if fn else ''))
        if len(lines) > 400:
            lines.append('... truncated')
            break
    return lines


def q(x, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return {}
    d = {f"p{k}": float(np.percentile(x, k)) for k in qs}
    d.update(mean=float(x.mean()), n=int(x.size))
    return d


def header_size(p):
    try:
        with Image.open(p) as im:
            return im.size + (im.mode, im.format)
    except Exception as e:
        return (0, 0, 'ERR', repr(e)[:40])


def retina_box(p, small=256):
    """Bounding box of the non-black region, as fractions of width / height."""
    with Image.open(p) as im:
        im.draft('RGB', (small, small))
        a = np.asarray(im.convert('L').resize((small, small), Image.BILINEAR))
    m = a > 12
    if not m.any():
        return None
    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
    return dict(left=cols[0] / small, right=(cols[-1] + 1) / small,
                top=rows[0] / small, bottom=(rows[-1] + 1) / small,
                fill=float(m.mean()),
                touches_top=bool(m[0].mean() > 0.2), touches_bottom=bool(m[-1].mean() > 0.2),
                touches_left=bool(m[:, 0].mean() > 0.2), touches_right=bool(m[:, -1].mean() > 0.2))


# ============================================================ DDR
def probe_ddr():
    g = find_root('DR_grading')
    if g is None:
        REPORT['ddr'] = {'error': 'DR_grading not found'}
        return
    R = {'root': str(g), 'tree': tree(g)}
    REPORT['ddr'] = R            # filled in place: a later failure keeps what is done
    log('DDR root', g)
    gr = g / 'DR_grading'

    # ---- splits, grades, every image size
    rows = []
    for sp in ('train', 'valid', 'test'):
        f = gr / f'{sp}.txt'
        if not f.exists():
            continue
        for line in open(f):
            parts = line.split()
            if len(parts) >= 2:
                rows.append(dict(split=sp, file=parts[0], grade=int(parts[1]),
                                 path=str(gr / sp / parts[0])))
    df = pd.DataFrame(rows)
    df['exists'] = [Path(p).exists() for p in df.path]
    with ThreadPoolExecutor(16) as ex:
        hs = list(ex.map(header_size, df.path))
    df['w'], df['h'], df['mode'], df['fmt'] = zip(*hs)
    df['ratio'] = df.w / df.h.replace(0, np.nan)
    df['mpx'] = df.w * df.h / 1e6
    df['bytes'] = [os.path.getsize(p) if e else 0 for p, e in zip(df.path, df.exists)]
    df.drop(columns='path').to_csv(OUT / 'ddr_images.csv', index=False)
    R['splits'] = {sp: dict(n=int(len(d)), missing=int((~d.exists).sum()),
                            grades={int(k): int(v) for k, v in d.grade.value_counts().sort_index().items()})
                   for sp, d in df.groupby('split')}
    ok = df[df.w > 0]
    R['width'], R['height'] = q(ok.w), q(ok.h)
    R['megapixels'], R['aspect_ratio'] = q(ok.mpx), q(ok.ratio)
    R['file_mb'] = q(ok.bytes / 1e6)
    R['distinct_sizes_top20'] = {f"{w}x{h}": int(n) for (w, h), n in
                                 Counter(zip(ok.w, ok.h)).most_common(20)}
    R['n_distinct_sizes'] = int(len(set(zip(ok.w, ok.h))))
    R['modes'] = dict(Counter(ok['mode']))
    R['formats'] = dict(Counter(ok.fmt))
    R['pixels_kept_at_896_pct'] = q(100 * S_DDR * S_DDR / (ok.w * ok.h))
    log('DDR sizes done', len(df))

    # ---- where is the retina in the frame (sample)
    samp = ok.sample(min(600, len(ok)), random_state=0)
    with ThreadPoolExecutor(8) as ex:
        boxes = list(ex.map(retina_box, samp.path))
    rb = pd.DataFrame([dict(file=f, w=w, h=h, **b) for f, w, h, b in
                       zip(samp.file, samp.w, samp.h, boxes) if b])
    rb.to_csv(OUT / 'ddr_retina_box.csv', index=False)
    if len(rb):
        rb['box_w_px'] = (rb.right - rb.left) * rb.w
        rb['box_h_px'] = (rb.bottom - rb.top) * rb.h
        R['retina'] = dict(
            n=int(len(rb)),
            box_ratio_w_over_h=q(rb.box_w_px / rb.box_h_px),
            box_width_frac=q(rb.right - rb.left), box_height_frac=q(rb.bottom - rb.top),
            left_margin_frac=q(rb.left), top_margin_frac=q(rb.top),
            fill_frac=q(rb.fill),
            cut_top_or_bottom_pct=float(100 * (rb.touches_top | rb.touches_bottom).mean()),
            cut_left_or_right_pct=float(100 * (rb.touches_left | rb.touches_right).mean()),
            # would a centred square crop of side min(w, h) cut the retina?
            centre_crop_cuts_retina_pct=float(100 * (
                (rb.box_w_px > np.minimum(rb.w, rb.h) * 1.02)).mean()))
    log('DDR retina boxes done')

    # ---- decoding time, as the training loader does it
    ts = []
    _tr = ok[ok.split == 'train'].path
    for p in _tr.sample(min(40, len(_tr)), random_state=1):
        t = time.perf_counter()
        np.asarray(Image.open(p).convert('RGB').resize((S_DDR, S_DDR), Image.BILINEAR))
        ts.append(1000 * (time.perf_counter() - t))
    R['decode_resize_ms'] = q(ts)

    # ---- lesions
    seg = g / 'lesion_segmentation'
    R['lesion_segmentation_exists'] = seg.exists()
    if seg.exists():
        R['lesion_tree'] = tree(seg, depth=3)
        grade_of = {Path(f).stem: (sp, gd) for sp, f, gd in zip(df.split, df.file, df.grade)}
        size_of = {Path(f).stem: (w, h) for f, w, h in zip(df.file, df.w, df.h)}
        path_of = {Path(f).stem: p for f, p in zip(df.file, df.path)}
        comp_rows, img_rows = [], defaultdict(dict)
        img_dirs = {}
        for sp in ('train', 'valid', 'test'):
            d = seg / sp / 'image'
            if d.exists():
                for p in d.iterdir():
                    img_dirs[p.stem] = p
            # the valid part names its folder 'segmentation label'
            lab = next((seg / sp / n for n in ('label', 'segmentation label')
                        if (seg / sp / n).exists()), None)
            if lab is None:
                continue
            for td in sorted(x for x in lab.iterdir() if x.is_dir()):
                files = sorted(td.glob('*'))
                log(f'DDR lesions {sp}/{td.name}: {len(files)} masks')
                for mp in files:
                    try:
                        m = np.asarray(Image.open(mp).convert('L')) > 0
                    except Exception:
                        continue
                    h, w = m.shape
                    r = img_rows[mp.stem]
                    r.update(stem=mp.stem, seg_split=sp, mask_w=w, mask_h=h,
                             grading_split=grade_of.get(mp.stem, ('absent', -1))[0],
                             grade=grade_of.get(mp.stem, ('absent', -1))[1],
                             same_size_as_grading_image=(size_of.get(mp.stem) == (w, h)))
                    r[f'{td.name}_area_frac'] = float(m.mean())
                    if ndimage is not None and m.any():
                        lab_arr, n = ndimage.label(m)
                        areas = ndimage.sum(m, lab_arr, range(1, n + 1))
                        sl = ndimage.find_objects(lab_arr)
                        r[f'{td.name}_n'] = int(n)
                        sx, sy = S_DDR / w, S_DDR / h
                        for a, s in zip(areas, sl):
                            bh, bw = s[0].stop - s[0].start, s[1].stop - s[1].start
                            comp_rows.append(dict(
                                stem=mp.stem, type=td.name, area_px=float(a),
                                diam_px=float(2 * np.sqrt(a / np.pi)),
                                bbox_w=bw, bbox_h=bh, img_w=w, img_h=h,
                                # after the stretch to 896 x 896 used in training
                                area_px896=float(a * sx * sy),
                                bbox_w896=float(bw * sx), bbox_h896=float(bh * sy),
                                # after a centred square crop then resize to 896
                                diam_px896_crop=float(2 * np.sqrt(a / np.pi) * S_DDR / min(w, h))))
                    else:
                        r[f'{td.name}_n'] = 0
        comp = pd.DataFrame(comp_rows)
        imgs = pd.DataFrame(list(img_rows.values()))
        comp.to_csv(OUT / 'ddr_lesion_components.csv', index=False)
        imgs.to_csv(OUT / 'ddr_lesion_images.csv', index=False)
        R['lesion_images'] = dict(
            n=int(len(imgs)),
            by_grading_split=dict(Counter(imgs.grading_split)) if len(imgs) else {},
            by_seg_split=dict(Counter(imgs.seg_split)) if len(imgs) else {},
            grades_in_grading_test=dict(Counter(
                int(x) for x in imgs[imgs.grading_split == 'test'].grade)) if len(imgs) else {},
            mask_matches_image_size_pct=float(100 * imgs.same_size_as_grading_image.mean())
            if len(imgs) else None,
            seg_image_folder_n=len(img_dirs))
        if len(comp):
            comp['diam_px896'] = 2 * np.sqrt(comp.area_px896 / np.pi)
            R['lesions_by_type'] = {
                t: dict(n_components=int(len(d)), n_images=int(d.stem.nunique()),
                        per_image=q(d.groupby('stem').size()),
                        diam_native_px=q(d.diam_px), diam_896_stretch_px=q(d.diam_px896),
                        diam_896_crop_px=q(d.diam_px896_crop),
                        bbox_long_side_896_px=q(np.maximum(d.bbox_w896, d.bbox_h896)),
                        area_896_px=q(d.area_px896),
                        # share of lesions that fit inside a patch of side p
                        fits_in_patch_pct={str(p): float(100 * (np.maximum(d.bbox_w896, d.bbox_h896) <= p).mean())
                                           for p in (14, 16, 28, 32, 56, 64, 112, 128, 224)})
                for t, d in comp.groupby('type')}
        cols = [c for c in imgs.columns if c.endswith('_area_frac')]
        if cols:
            imgs['total_area_frac'] = imgs[cols].fillna(0).sum(axis=1)
            R['lesion_area_frac_per_image'] = {
                'all': q(imgs.total_area_frac),
                'grading_test': q(imgs[imgs.grading_split == 'test'].total_area_frac),
                **{c[:-10]: q(imgs[c].dropna()) for c in cols}}
        # thumbnails with lesions drawn
        try:
            pick = imgs[imgs.grading_split == 'test'].stem.tolist()[:8] or imgs.stem.tolist()[:8]
            colors = {'EX': (255, 255, 0), 'HE': (255, 0, 0), 'MA': (0, 255, 0), 'SE': (0, 160, 255)}
            for stem in pick:
                ip = img_dirs.get(stem) or path_of.get(stem)
                if ip is None:
                    continue
                im = Image.open(ip).convert('RGB').resize((448, 448), Image.BILINEAR)
                a = np.asarray(im).copy()
                for sp, ln in [(x, y) for x in ('train', 'valid', 'test')
                               for y in ('label', 'segmentation label')]:
                    for t, col in colors.items():
                        for mp in (seg / sp / ln / t).glob(stem + '.*'):
                            m = np.asarray(Image.open(mp).convert('L').resize((448, 448), Image.NEAREST)) > 0
                            a[m] = col
                Image.fromarray(a).save(OUT / 'thumbs' / f'ddr_{stem}.jpg', quality=80)
        except Exception:
            traceback.print_exc()
    REPORT['ddr'] = R


# ============================================================ RSNA
def split_of(pid):
    h = int(hashlib.md5((str(pid) + str(SPLIT_SEED)).encode()).hexdigest(), 16) % 100
    return 'train' if h >= 25 else ('val' if h < 12 else 'test')


def probe_rsna():
    g = find_root('stage2_train_metadata.csv')
    if g is None:
        REPORT['rsna'] = {'error': 'stage2_train_metadata.csv not found'}
        return
    R = {'root': str(g), 'tree': tree(g, depth=3)}
    REPORT['rsna'] = R
    log('RSNA root', g)
    raw = pd.read_csv(g / 'stage2_train_metadata.csv')
    R['columns'] = {c: str(t) for c, t in raw.dtypes.items()}
    R['n_rows'], R['n_patients'] = int(len(raw)), int(raw.patientId.nunique())
    R['head'] = json.loads(raw.head(5).drop(columns=['patientId']).to_json(orient='records'))
    for c in raw.columns:
        if c in ('patientId', 'x', 'y', 'width', 'height'):
            continue
        if raw[c].nunique() <= 12:
            R.setdefault('categorical', {})[c] = {str(k): int(v) for k, v in raw[c].value_counts(dropna=False).items()}
        elif pd.api.types.is_numeric_dtype(raw[c]):
            R.setdefault('numeric', {})[c] = q(raw[c].dropna())
    other = [p.name for p in g.glob('*.csv')]
    R['csv_files'] = other
    for name in other:
        if name != 'stage2_train_metadata.csv':
            try:
                o = pd.read_csv(g / name, nrows=2000)
                R.setdefault('other_csv', {})[name] = dict(columns=list(o.columns), n_first_rows=len(o))
            except Exception:
                pass

    pat = raw.groupby('patientId')['Target'].max().reset_index()
    pat['split'] = pat.patientId.map(split_of)
    R['v3_split'] = {sp: dict(n=int(len(d)), positives=int(d.Target.sum()))
                     for sp, d in pat.groupby('split')}
    cls_col = next((c for c in raw.columns if c.lower() in ('class', 'label', 'detailed_class')), None)
    if cls_col:
        pc = raw.groupby('patientId')[cls_col].first()
        R['class_by_split'] = {sp: {str(k): int(v) for k, v in pc[d.patientId].value_counts().items()}
                               for sp, d in pat.groupby('split')}

    pos = raw[raw.Target == 1].dropna(subset=['x', 'y', 'width', 'height']).copy()
    pos['area_frac'] = pos.width * pos.height / S_RSNA ** 2
    per = pos.groupby('patientId')
    R['boxes'] = dict(n=int(len(pos)), patients=int(per.ngroups),
                      per_patient=dict(Counter(int(n) for n in per.size())),
                      width_px=q(pos.width), height_px=q(pos.height),
                      short_side_px=q(np.minimum(pos.width, pos.height)),
                      long_side_px=q(np.maximum(pos.width, pos.height)),
                      area_frac=q(pos.area_frac),
                      fits_in_patch_pct={str(p): float(100 * (np.maximum(pos.width, pos.height) <= p).mean())
                                         for p in (32, 64, 128, 256, 512)})
    union = []
    for pid, d in list(per)[:3000]:
        m = np.zeros((S_RSNA, S_RSNA), bool)
        for x, y, w, h in d[['x', 'y', 'width', 'height']].to_numpy():
            m[max(0, int(y)):int(y + h), max(0, int(x)):int(x + w)] = True
        union.append(m.mean())
    R['box_union_frac_per_patient'] = q(union)
    pos.drop(columns=['patientId']).to_csv(OUT / 'rsna_boxes.csv', index=False)

    img_dir = g / 'Training' / 'Images'
    R['images_dir_exists'] = img_dir.exists()
    if img_dir.exists():
        files = sorted(img_dir.glob('*'))
        R['n_image_files'] = len(files)
        R['patients_without_image'] = int(sum(not (img_dir / f'{p}.png').exists() for p in pat.patientId))
        samp = [files[i] for i in rng.choice(len(files), min(1500, len(files)), replace=False)]
        with ThreadPoolExecutor(16) as ex:
            hs = list(ex.map(header_size, samp))
        R['image_sizes'] = {f"{w}x{h} {m} {f}": int(n) for (w, h, m, f), n in Counter(hs).most_common(8)}
        ts, means, stds = [], [], []
        for p in samp[:60]:
            t = time.perf_counter()
            a = np.asarray(Image.open(p).convert('L'))
            ts.append(1000 * (time.perf_counter() - t))
            means.append(a.mean() / 255); stds.append(a.std() / 255)
        R['decode_ms'] = q(ts)
        R['pixel_mean'], R['pixel_std'] = float(np.mean(means)), float(np.mean(stds))

        # what is Training/Masks ?
        mdir = g / 'Training' / 'Masks'
        R['masks_dir_exists'] = mdir.exists()
        if mdir.exists():
            mfiles = sorted(mdir.glob('*'))
            R['n_mask_files'] = len(mfiles)
            rows = []
            pos_ids = set(pos.patientId)
            cand = [m for m in mfiles if m.stem in pos_ids][:40] + [m for m in mfiles if m.stem not in pos_ids][:20]
            for mp in cand:
                m = np.asarray(Image.open(mp))
                mb = (m if m.ndim == 2 else m[..., 0]) > 0
                box = np.zeros(mb.shape, bool)
                sc = mb.shape[0] / S_RSNA
                for x, y, w, h in pos[pos.patientId == mp.stem][['x', 'y', 'width', 'height']].to_numpy():
                    box[int(y * sc):int((y + h) * sc), int(x * sc):int((x + w) * sc)] = True
                inter, uni = (mb & box).sum(), (mb | box).sum()
                rows.append(dict(positive=mp.stem in pos_ids, shape=str(m.shape), dtype=str(m.dtype),
                                 n_unique=int(len(np.unique(m))), frac_on=float(mb.mean()),
                                 iou_with_boxes=float(inter / uni) if uni else None))
            mk = pd.DataFrame(rows)
            mk.to_csv(OUT / 'rsna_masks_sample.csv', index=False)
            R['masks'] = dict(
                shapes=dict(Counter(mk['shape'])), dtypes=dict(Counter(mk['dtype'])),
                frac_on_positive=q(mk[mk.positive].frac_on), frac_on_negative=q(mk[~mk.positive].frac_on),
                iou_with_boxes_positive=q(mk[mk.positive].iou_with_boxes.dropna()),
                reading=('IoU near 1 on positives and empty on negatives: the masks are the boxes. '
                         'Large area on negatives too: they are lung masks.'))
        # thumbnails with boxes
        try:
            for pid in list(per.groups)[:6]:
                im = Image.open(img_dir / f'{pid}.png').convert('RGB').resize((384, 384))
                dr = ImageDraw.Draw(im)
                for x, y, w, h in pos[pos.patientId == pid][['x', 'y', 'width', 'height']].to_numpy():
                    s = 384 / S_RSNA
                    dr.rectangle([x * s, y * s, (x + w) * s, (y + h) * s], outline=(255, 0, 0), width=2)
                im.save(OUT / 'thumbs' / f'rsna_pos_{len(list((OUT / "thumbs").glob("rsna_pos_*")))}.jpg', quality=80)
                if mdir.exists() and (mdir / f'{pid}.png').exists():
                    Image.open(mdir / f'{pid}.png').convert('L').resize((384, 384)).save(
                        OUT / 'thumbs' / f'rsna_mask_{len(list((OUT / "thumbs").glob("rsna_mask_*")))}.jpg', quality=80)
        except Exception:
            traceback.print_exc()
    REPORT['rsna'] = R


for fn in (probe_ddr, probe_rsna):
    try:
        fn()
    except Exception as exc:
        traceback.print_exc()
        REPORT[fn.__name__ + '_error'] = {'error': repr(exc), 'trace': traceback.format_exc()}

REPORT['env'] = dict(cpu=os.cpu_count(), seconds=round(time.time() - T0),
                     scipy=ndimage is not None)
with open(OUT / 'report.json', 'w') as f:
    json.dump(REPORT, f, indent=1, default=str)

zpath = OUT.parent / 'dataset_probe.zip'
with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():
            z.write(p, p.relative_to(OUT.parent))
log(f"done -> {zpath}  ({zpath.stat().st_size/1e6:.1f} MB)")
