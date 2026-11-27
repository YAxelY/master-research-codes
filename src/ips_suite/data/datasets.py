import fcntl
import hashlib
import numpy as np
import os
import pandas as pd
import random
import shutil
import time
import torch
from PIL import Image
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from ips_suite.config.environment import _env, _env_int


SPLIT_SEED = 42            # constant: the split never follows the run seed
CACHE_DIR = _env('IPS_CACHE_DIR')            # '' = no cache
CACHE_SETS = [s for s in _env('IPS_CACHE', 'ddr').split(',') if s]
SMOKE_N = _env_int('IPS_SMOKE', 0)           # >0 : keep N images per split


def _free_bytes(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(directory).free


def _cache_fits(path, n_bytes, free_bytes, margin=1.05):
    return Path(path).with_suffix('.done').exists() or free_bytes >= n_bytes * margin


def _build_cache(path, n, shape, load_one, n_thread=8):
    """Write an (n, *shape) uint8 array once. Safe with concurrent jobs."""
    path = Path(path)
    done = path.with_suffix('.done')
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix('.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if done.exists():
                return
            t0 = time.time()
            arr = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                            shape=(n,) + tuple(shape))

            def work(i):
                arr[i] = load_one(i)

            with ThreadPoolExecutor(n_thread) as ex:
                list(ex.map(work, range(n)))
            arr.flush()
            del arr
            done.write_text(str(n))
            print(f"  [cache] {path.name} : {n} images in {time.time()-t0:.0f} s")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _build_cache_jpg(folder, n, load_one, n_thread=8):
    """One JPEG per image (quality 95, no chroma subsampling), for images too
    large to keep raw: DDR at 1792 px would take about 120 GB raw, about 15 GB
    this way. Safe with concurrent jobs."""
    folder = Path(folder)
    done = folder.with_suffix('.done')
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder.with_suffix('.lock'), 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if done.exists():
                return
            t0 = time.time()

            def work(i):
                Image.fromarray(load_one(i)).save(folder / f'{i}.jpg', quality=95,
                                                  subsampling=0)

            with ThreadPoolExecutor(n_thread) as ex:
                list(ex.map(work, range(n)))
            done.write_text(str(n))
            print(f"  [cache] {folder.name} : {n} images (JPEG) in {time.time()-t0:.0f} s")
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


class _Cached:
    """Common part: list of items, optional uint8 cache, optional subsample."""

    name = 'base'

    def _finish(self, conf, split, shape):
        if SMOKE_N and len(self.items) > SMOKE_N:
            # evenly spaced, not the first N: the DDR label files are sorted
            # by grade, so the first images are all of grade 0
            pick = np.linspace(0, len(self.items) - 1, SMOKE_N).round().astype(int)
            self.items = [self.items[i] for i in pick]
        frac = float(conf.get('train_frac', 1.0))
        if split == 'train' and frac < 1.0:
            rng = np.random.default_rng(SPLIT_SEED)
            by_lab = defaultdict(list)
            for i, (_, y) in enumerate(self.items):
                by_lab[y].append(i)
            keep = []
            for y, ii in sorted(by_lab.items()):
                ii = np.array(ii)
                rng.shuffle(ii)
                keep += ii[:max(1, int(round(frac * len(ii))))].tolist()
            self.items = [self.items[i] for i in sorted(keep)]
        self._arr, self._cache_path, self._cache_kind = None, None, None
        if CACHE_DIR and self.name in CACHE_SETS:
            tag = hashlib.md5('|'.join(str(p) for p, _ in self.items)
                              .encode()).hexdigest()[:10]
            geo = conf.get('ddr_geometry', 'stretch') if self.name == 'ddr' else 'id'
            stem = f"{self.name}_{split}_{conf.img_size}_{geo}_{tag}"
            load = lambda i: self._decode(self.items[i][0])
            if conf.img_size > 1024:          # raw would be too large
                self._cache_kind = 'jpg'
                self._cache_path = Path(CACHE_DIR) / f"{stem}_jpg"
                _build_cache_jpg(self._cache_path, len(self.items), load)
            else:
                path = Path(CACHE_DIR) / f"{stem}.npy"
                needed = len(self.items) * int(np.prod(shape))
                free = _free_bytes(CACHE_DIR)
                if _cache_fits(path, needed, free):
                    self._cache_kind, self._cache_path = 'npy', path
                    _build_cache(path, len(self.items), shape, load)
                else:
                    print(f"  [cache] {stem}: needs {needed / 1e9:.1f} GB, {free / 1e9:.1f} GB "
                          f"free in {CACHE_DIR}: reading the original images instead")

    def _raw(self, i):
        if self._cache_path is not None and self._cache_kind == 'jpg':
            return np.asarray(Image.open(self._cache_path / f'{i}.jpg'))
        if self._cache_path is not None:
            if self._arr is None:               # opened once per worker
                self._arr = np.load(self._cache_path, mmap_mode='r')
            return np.asarray(self._arr[i])
        return self._decode(self.items[i][0])

    def __len__(self):
        return len(self.items)


class DDRDataset(_Cached, Dataset):
    """DDR, 5 grades (0-4). Grade 5 (ungradable) is excluded.

    ddr_geometry = 'stretch' : resize the whole photograph to a square, as in
                               v2. Native ratios go from 0.98 to 1.76, so the
                               retina becomes an ellipse and about half of the
                               square is black margin (dataset probe).
                 = 'retina'  : crop to the retina, keep pixels square, resize.
                               Same field of view on every image, and the
                               retina fills the square.
                 = 'crop'    : centre crop to the short side. Cuts the retina
                               on about one image in eight: not recommended.
    """
    name = 'ddr'

    def __init__(self, conf, split):
        assert split in ('train', 'valid', 'test')
        self.img_size = conf.img_size
        self.geometry = conf.get('ddr_geometry', 'stretch')
        self.split = split
        self._frames = {}
        root = Path(conf.data_dir) / 'DR_grading'
        lab_file, img_dir = root / f'{split}.txt', root / split

        self.items = []
        with open(lab_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                fname, grade = line.split()[0], int(line.split()[1])
                if grade == 5:
                    continue
                p = img_dir / fname
                if p.exists():
                    self.items.append((p, grade))
        self._finish(conf, split, (conf.img_size, conf.img_size, 3))

        counts = [0] * 5
        for _, g in self.items:
            counts[g] += 1
        print(f"[DDR] {split:5s} | {len(self.items):5d} images | "
              + ', '.join(f"C{i}={counts[i]}" for i in range(5)))

        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.aug = (transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.1, hue=0.05)])
            if split == 'train' else None)
        # Training images leave the loader as uint8 and are augmented on the
        # GPU (run_epoch): the rotation and colour jitter of 896 or 1792 px
        # images were the slowest part of an epoch on the CPU.
        self.aug_on_device = split == 'train'

    def device_augment(self, imgs):
        """uint8 (B, C, H, W) on any device -> augmented, normalised float.
        Same transforms as before, drawn independently for each image."""
        x = imgs.float().div_(255.0)
        x = torch.stack([self.aug(t) for t in x])
        return self.norm(x)

    def _frame(self, img_path):
        """Square crop (left, top, side) in native pixels, or None (stretch).

        'retina': the bounding box of the non-black region, read on a small
        copy, extended to a square around its centre. Where the square leaves
        the photograph it is filled with black, so pixels stay square even
        when the retina is cut by the frame.
        """
        if self.geometry == 'stretch':
            return None
        key = str(img_path)
        if key in self._frames:
            return self._frames[key]
        with Image.open(img_path) as im:
            w, h = im.size
            if self.geometry == 'crop':
                s = min(w, h)
                fr = ((w - s) // 2, (h - s) // 2, s)
            elif self.geometry == 'retina':
                im.draft('L', (512, 512))
                g = np.asarray(im.convert('L'))
                m = g > 12
                if m.any():
                    rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
                    sy, sx = h / g.shape[0], w / g.shape[1]
                    l, r = cols[0] * sx, (cols[-1] + 1) * sx
                    t, b = rows[0] * sy, (rows[-1] + 1) * sy
                else:
                    l, r, t, b = 0, w, 0, h
                side = int(round(max(r - l, b - t)))
                fr = (int(round((l + r) / 2 - side / 2)),
                      int(round((t + b) / 2 - side / 2)), side)
            else:
                raise ValueError(f"ddr_geometry={self.geometry}")
        self._frames[key] = fr
        return fr

    def _open(self, path, resample, mode, frame_from=None):
        img = Image.open(path).convert(mode)
        fr = self._frame(frame_from or path)
        if fr is not None:
            l, t, s = fr
            img = img.crop((l, t, l + s, t + s))      # outside the image: black
        return img.resize((self.img_size, self.img_size), resample)

    def _decode(self, path):
        return np.asarray(self._open(path, Image.BILINEAR, 'RGB'), dtype=np.uint8)

    def load_mask(self, mask_path, img_path):
        """Lesion mask, cut with the frame of ITS image, nearest neighbour."""
        return np.asarray(self._open(mask_path, Image.NEAREST, 'L',
                                     frame_from=img_path)) > 0

    def __getitem__(self, i):
        if self.aug_on_device:              # augmented and normalised in run_epoch
            # np.array copies: the cache is read-only, and a tensor on read-only
            # memory makes PyTorch warn
            return torch.from_numpy(np.array(self._raw(i))).permute(2, 0, 1), \
                self.items[i][1], i
        t = torch.from_numpy(self._raw(i).astype(np.float32) / 255.0).permute(2, 0, 1)
        return self.norm(t), self.items[i][1], i


class RSNADataset(_Cached, Dataset):
    """RSNA pneumonia, binary, one image per patient.

    Opacity (Target 1) against Normal (Target 0). The patients of the third
    class, 'No Lung Opacity / Not Normal' (another abnormality, no opacity),
    are left out of every split (conf.rsna_exclude, decided 23 November).
    """
    name = 'rsna'

    @staticmethod
    def split_of(pid):
        # 70 / 15 / 15 by patient. The v3 training set is a subset of the v2
        # one (hash >= 25); 5 % of the v2 training patients are now in val or
        # test, so v2 weights cannot be scored cleanly on RSNA.
        h = int(hashlib.md5((str(pid) + str(SPLIT_SEED)).encode()).hexdigest(), 16) % 100
        if h >= 30:
            return 'train'
        return 'val' if h < 15 else 'test'

    def __init__(self, conf, split):
        assert split in ('train', 'val', 'test', 'v2val')
        self.img_size = conf.img_size
        self.split = split
        root = Path(conf.data_dir)
        img_dir = root / 'Training' / 'Images'

        raw = pd.read_csv(root / 'stage2_train_metadata.csv')
        excl = list(conf.get('rsna_exclude') or [])
        if excl:
            if 'class' not in raw.columns:
                raise ValueError("rsna_exclude needs the 'class' column of "
                                 "stage2_train_metadata.csv")
            drop = set(raw.loc[raw['class'].isin(excl), 'patientId'])
            n_all = raw['patientId'].nunique()
            raw = raw[~raw['patientId'].isin(drop)]
            if split in ('train', 'v2val'):
                print(f"[RSNA] excluded {len(drop)} of {n_all} patients: "
                      + ', '.join(excl))
        df = raw.groupby('patientId')['Target'].max().reset_index()
        sp = df['patientId'].apply(self.split_of)
        keep = sp.isin(['val', 'test']) if split == 'v2val' else (sp == split)
        df = df[keep].reset_index(drop=True)

        self.boxes = {}
        if {'x', 'y', 'width', 'height'} <= set(raw.columns):
            pos = raw[raw['Target'] == 1].dropna(subset=['x', 'y', 'width', 'height'])
            self.boxes = {p: g[['x', 'y', 'width', 'height']].to_numpy(float)
                          for p, g in pos.groupby('patientId')}

        self.items, missing = [], 0
        for pid, tgt in zip(df['patientId'], df['Target']):
            p = img_dir / f"{pid}.png"
            if p.exists():
                self.items.append((p, int(tgt)))
            else:
                missing += 1
        self._finish(conf, split, (conf.img_size, conf.img_size))
        pos = sum(l for _, l in self.items)
        print(f"[RSNA] {split:5s} | {len(self.items):5d} images | "
              f"pos={pos} ({100*pos/max(len(self.items),1):.1f}%)"
              + (f" | {missing} missing" if missing else ""))

        self.aug = (transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(5)])
                    if split == 'train' else None)

    def _decode(self, path):
        img = Image.open(path).convert('L')
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        return np.asarray(img, dtype=np.uint8)

    def __getitem__(self, i):
        t = (torch.from_numpy(self._raw(i).astype(np.float32) / 255.0)
             .unsqueeze(0) - 0.5) / 0.5
        if self.aug is not None:
            t = self.aug(t)
        return t, self.items[i][1], i


def _worker_init(worker_id):
    s = torch.initial_seed() % (2 ** 31)
    np.random.seed(s)
    random.seed(s)


def build_dataset(conf, split):
    """split in 'train' | 'val' | 'test' (dataset-independent names)."""
    if conf.dataset == 'ddr':
        return DDRDataset(conf, {'train': 'train', 'val': 'valid',
                                 'test': 'test'}[split])
    if conf.dataset == 'rsna':
        return RSNADataset(conf, split)
    raise ValueError(conf.dataset)


def build_loader(conf, ds, train):
    # The baseline takes one optimiser step per DataLoader batch; the patch
    # models gather B_seq images at a time until they have B. Both therefore
    # train with the same batch of B = 16.
    bs = conf.B if conf.model in ('baseline', 'mil') else conf.B_seq
    g = torch.Generator()
    g.manual_seed(int(conf.seed))
    return DataLoader(ds, batch_size=bs, shuffle=train, generator=g,
                      num_workers=conf.n_worker, pin_memory=conf.pin_memory,
                      persistent_workers=False, worker_init_fn=_worker_init)


def split_hash(dataset):
    keys = sorted(f"{os.path.basename(str(path))}:{label}" for path, label in dataset.items)
    return hashlib.md5('\n'.join(keys).encode()).hexdigest()[:12]
