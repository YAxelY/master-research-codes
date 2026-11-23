#!/usr/bin/env python3
"""Small fake DDR and RSNA data sets, with the real folder layout, to test the
notebook on a laptop without the real data.

    python tools/make-fake-data.py /tmp/fake
    IPS_DDR_DIR=/tmp/fake/ddr IPS_RSNA_DIR=/tmp/fake/rsna IPS_PRETRAINED=0 \\
    IPS_SMOKE=16 IPS_EPOCHS=1 IPS_RUNS=ddr_ips66 python ips-suite-v3.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

rng = np.random.default_rng(0)
root = Path(sys.argv[1])
ddr = root / 'ddr' / 'DR_grading'
SIZES = [(700, 500), (600, 600), (880, 500)]
for sp, n in [('train', 30), ('valid', 15), ('test', 15)]:
    (ddr / sp).mkdir(parents=True, exist_ok=True)
    lines = []
    for i in range(n):
        w, h = SIZES[i % 3]
        a = rng.integers(0, 255, (h, w, 3), dtype=np.uint8)
        g = i % 5
        if g > 0:
            a[100:100 + 20 * g, 200:200 + 20 * g] = 255
        Image.fromarray(a).save(ddr / sp / f'{sp}{i:03d}.jpg', quality=90)
        lines.append(f'{sp}{i:03d}.jpg {g}')
    lines.append('ghost.jpg 5')
    (ddr / f'{sp}.txt').write_text('\n'.join(lines) + '\n')
    # the valid masks of the real data set are in 'segmentation label'
    lab = 'segmentation label' if sp == 'valid' else 'label'
    for t in ['EX', 'MA', 'HE']:
        d = root / 'ddr' / 'lesion_segmentation' / sp / lab / t
        d.mkdir(parents=True, exist_ok=True)
        for i in range(1, n, 2):
            w, h = SIZES[i % 3]
            m = np.zeros((h, w), np.uint8)
            g = i % 5
            if g > 0 and t == 'EX':
                m[100:100 + 20 * g, 200:200 + 20 * g] = 255
            if t == 'MA' and i % 4 == 1:
                m[300:306, 50:56] = 255
            if t == 'HE' and i % 3 == 0:
                m[50:70, 400:430] = 255
            Image.fromarray(m).save(d / f'{sp}{i:03d}.tif')
rs = root / 'rsna'
(rs / 'Training' / 'Images').mkdir(parents=True, exist_ok=True)
rows = []
for i in range(160):
    pid = f'p{i:04d}'
    a = rng.integers(0, 255, (1024, 1024), dtype=np.uint8)
    if i % 3 == 0:
        a[300:500, 200:400] = 255
        rows.append(dict(patientId=pid, x=200, y=300, width=200, height=200, Target=1,
                         **{'class': 'Lung Opacity'}))
        if i % 2 == 0:       # a box reaching the bottom-right border (padded patches)
            rows.append(dict(patientId=pid, x=900, y=880, width=124, height=144, Target=1,
                             **{'class': 'Lung Opacity'}))
    else:                    # negatives: Normal, or the excluded third class
        cls = 'No Lung Opacity / Not Normal' if i % 3 == 1 else 'Normal'
        rows.append(dict(patientId=pid, x=np.nan, y=np.nan, width=np.nan, height=np.nan, Target=0,
                         **{'class': cls}))
    Image.fromarray(a).save(rs / 'Training' / 'Images' / f'{pid}.png')
pd.DataFrame(rows).to_csv(rs / 'stage2_train_metadata.csv', index=False)
print('fake data in', root)
