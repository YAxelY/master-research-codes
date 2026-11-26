"""Paths, driven by environment variables (Kaggle defaults)."""

import os

from ips_suite.config.environment import Struct, _env, _env_int, _env_path
from ips_suite.data.datasets import CACHE_DIR, CACHE_SETS


WORK    = _env_path('IPS_WORK', '/kaggle/working')
RESULTS = _env_path('IPS_RESULTS', WORK / 'results')
RESULTS.mkdir(parents=True, exist_ok=True)
# Heavy files live apart from the results, so that the results folder can be
# shared (pushed) as it is: nothing in it is larger than a few hundred kB.
ARTIFACTS = _env_path('IPS_ARTIFACTS', WORK / 'artifacts')
ARTIFACTS.mkdir(parents=True, exist_ok=True)
# One folder per configuration, every seed inside it:
#   RESULTS/<configuration>/experiments    <run>_result.json, lesion and Grad-CAM tables
#                          /history        <run>_history.json, codecarbon
#                          /predictions    per image: probabilities (small)
#   ARTIFACTS/<configuration>/checkpoints  weights of the best epoch, resume state
#                            /saliency     per image: Grad-CAM maps (about 75 MB per run)
# use_run_dirs() points the OUT_* paths to the folders of the run being processed.
OUT_CKPT = OUT_HIST = OUT_EXPE = OUT_SAL = OUT_PRED = RESULTS


def use_run_dirs(name):
    global OUT_CKPT, OUT_HIST, OUT_EXPE, OUT_SAL, OUT_PRED
    base, heavy = RESULTS / name, ARTIFACTS / name
    OUT_HIST, OUT_EXPE, OUT_PRED = base / 'history', base / 'experiments', base / 'predictions'
    OUT_CKPT, OUT_SAL = heavy / 'checkpoints', heavy / 'saliency'
    for d in (OUT_CKPT, OUT_HIST, OUT_EXPE, OUT_SAL, OUT_PRED):
        d.mkdir(parents=True, exist_ok=True)
    return base

DDR_DIR = os.environ.get(
    'IPS_DDR_DIR',
    '/kaggle/input/datasets/samriddhibagchi/ddr-dataset-credits-to-authors/DDR-dataset')
RSNA_DIR = os.environ.get(
    'IPS_RSNA_DIR',
    '/kaggle/input/datasets/iamtapendu/rsna-pneumonia-processed-dataset')

print(f"  work        : {WORK}")
print(f"  results     : {RESULTS}  (one folder per configuration, light, to share)")
print(f"  artifacts   : {ARTIFACTS}  (weights and Grad-CAM maps, heavy, never shared)")
print(f"  DDR         : {DDR_DIR}")
print(f"  RSNA        : {RSNA_DIR}")
print(f"  cache       : {CACHE_DIR or '(none)'}  for {CACHE_SETS}")

BASE = dict(
    n_epoch=50, n_epoch_warmup=10, lr=0.0001, wd=0.1,
    B=16, B_seq=4, eps=1e-6,
    M=10, I=128, D=512, H=8, D_k=64, D_v=64, D_inner=2048,
    n_token=1, attn_dropout=0.1, dropout=0.1,
    pretrained=bool(_env_int('IPS_PRETRAINED', 1)),
    n_res_blocks=4, shuffle=True, use_scale_emb=True,
    # v2 used 2 workers while the job reserved 16 CPUs: decoding and
    # augmentation, not the model, set the duration of every DDR run.
    n_worker=_env_int('IPS_WORKERS', 8), pin_memory=True,
    loading='eager', train_frac=1.0, ddr_geometry=_env('IPS_DDR_GEOMETRY', 'retina'),
    aggregator='cross', selection='attention',
)

# pad_value: black after normalisation, so padding looks like the black
# background of a fundus photograph (DDR) or a black border (RSNA)
DDR_BASE = dict(BASE, dataset='ddr', data_dir=DDR_DIR, img_size=896,
                n_chan_in=3, n_class=5, task_type='ordinal',
                main_metric='kappa',
                pad_value=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225])

RSNA_BASE = dict(BASE, dataset='rsna', data_dir=RSNA_DIR, img_size=1024,
                 # opacity against normal: the third class is left out
                 rsna_exclude=['No Lung Opacity / Not Normal'],
                 n_chan_in=1, n_class=2, task_type='binary',
                 main_metric='auc', pad_value=[-1.0])

REGISTRY = {}       # name -> (dict of settings, groups, note)


def reg(name, base, model, groups, note, **kw):
    d = dict(base, name=name, model=model, **kw)
    if model == 'hierarchical' and 'scales' not in d:
        d['scales'] = list(d['cascade'])
    if model == 'baseline' and 'scales' not in d:
        d['scales'] = [128]
    REGISTRY[name] = (d, set(groups.split(',')), note)


def make_conf(name, seed):
    d, _, _ = REGISTRY[name]
    return Struct(**dict(d, seed=int(seed), run_name=f"{name}_s{seed}"))


# ---------------------------------------------------------------- DDR, 896
# core: what goes in the main table (three seeds)
reg('ddr_base896', DDR_BASE, 'baseline', 'core,base', 'full image', baseline_size=896)
reg('ddr_base448', DDR_BASE, 'baseline', 'core,base', 'image at half resolution', baseline_size=448)
reg('ddr_mil128',  DDR_BASE, 'mil', 'core', 'attention MIL, all 49 patches of 128', scales=[128])
reg('ddr_ips128', DDR_BASE, 'selector', 'core,patch,mstudy', 'IPS reference', scales=[128])
reg('ddr_ips112', DDR_BASE, 'selector', 'core,patch', 'control of h224_112', scales=[112])
reg('ddr_ips64',  DDR_BASE, 'selector', 'core,patch,mstudy', '196 candidates', scales=[64])
reg('ddr_ips56',  DDR_BASE, 'selector', 'core,patch', 'control of h224_56', scales=[56])
reg('ddr_ips32',  DDR_BASE, 'selector', 'core,patch,mstudy', '784 candidates, map 1x1', scales=[32])
reg('ddr_ms128_64', DDR_BASE, 'selector', 'core,ms', 'MS-IPS, flat, native', scales=[128, 64])
reg('ddr_ms64_32',  DDR_BASE, 'selector', 'core,ms', 'MS-IPS, flat, native', scales=[64, 32])
reg('ddr_h224_112', DDR_BASE, 'hierarchical', 'core', 'H-MS-IPS, final 112', cascade=[224, 112], M_keep=[4])
reg('ddr_h224_56',  DDR_BASE, 'hierarchical', 'core', 'H-MS-IPS, final 56', cascade=[224, 56], M_keep=[4])
# 64 then 32: keeps a quarter of the image, like every other cascade
# (49 zones of 64 px out of 196), then the best 10 of their 196 patches of 32
reg('ddr_h64_32',   DDR_BASE, 'hierarchical', 'core', 'H-MS-IPS, final 32', cascade=[64, 32], M_keep=[49])
# ablation: the selection module, and gradients to the coarse stage
reg('ddr_h224_112_tc', DDR_BASE, 'hierarchical', 'ablation', 'coarse zones trained',
    cascade=[224, 112], M_keep=[4], train_coarse=4)
reg('ddr_h224_56_tc',  DDR_BASE, 'hierarchical', 'ablation', 'coarse zones trained',
    cascade=[224, 56], M_keep=[4], train_coarse=4)
# mstudy: generated below for every family; extra: zones kept by the cascade
reg('ddr_h224_56_k8',  DDR_BASE, 'hierarchical', 'mstudy', '8 zones kept of 16', cascade=[224, 56], M_keep=[8])
reg('ddr_h224_56_k2',  DDR_BASE, 'hierarchical', 'mstudy', '2 zones kept of 16', cascade=[224, 56], M_keep=[2])
# patch: the rest of the single-size sweep
reg('ddr_ips224', DDR_BASE, 'selector', 'patch', '16 candidates', scales=[224])
reg('ddr_ips28',  DDR_BASE, 'selector', 'patch', 'control of h224_28', scales=[28])
reg('ddr_h224_28', DDR_BASE, 'hierarchical', 'patch', 'H-MS-IPS, final 28', cascade=[224, 28], M_keep=[4])
reg('ddr_h224_28_tc', DDR_BASE, 'hierarchical', 'patch', 'same, coarse zones trained',
    cascade=[224, 28], M_keep=[4], train_coarse=4)
# other controls
reg('ddr_base224', DDR_BASE, 'baseline', 'base', 'image at quarter resolution', baseline_size=224)
reg('ddr_ms128_64_rs', DDR_BASE, 'selector', 'ms', 'MS-IPS, 64 px upsampled to 128',
    scales=[128, 64], encode_px=[None, 128])
reg('ddr_h224lr_56_tc', DDR_BASE, 'hierarchical', 'lowres', 'coarse stage at 112 px',
    cascade=[224, 56], M_keep=[4], train_coarse=4, encode_px=[112, None])
for _n, _kw in [('ddr_ips128', dict(scales=[128])), ('ddr_ips64', dict(scales=[64])),
                ('ddr_ips32', dict(scales=[32]))]:
    reg(_n + '_f25', DDR_BASE, 'selector', 'lowdata', '25 % of train', train_frac=0.25, **_kw)
reg('ddr_h224_56_f25', DDR_BASE, 'hierarchical', 'lowdata', '25 % of train',
    cascade=[224, 56], M_keep=[4], train_frac=0.25)
reg('ddr_ips128_lazy', DDR_BASE, 'selector', 'misc', 'image stays on the CPU', scales=[128], loading='lazy')
reg('ddr_ips128_i16',  DDR_BASE, 'selector', 'misc', 'I = 16', scales=[128], I=16)

# hires: the same patch models on DDR at twice the resolution (1792 px).
# Memory depends on M and the patch size, not on the image, so these train
# at the memory of their 896 px twins while every lesion is twice as wide.
DDR_HI = dict(DDR_BASE, img_size=1792)
reg('ddr_r1792_ips128', DDR_HI, 'selector', 'hires', '196 candidates, memory of ddr_ips128', scales=[128])
reg('ddr_r1792_ips128_m20', DDR_HI, 'selector', 'hires', '', scales=[128], M=20)
reg('ddr_r1792_ips112', DDR_HI, 'selector', 'hires', 'control of r1792_h448_112', scales=[112])
reg('ddr_r1792_h448_112', DDR_HI, 'hierarchical', 'hires', 'H-MS-IPS at 1792',
    cascade=[448, 112], M_keep=[4])

# --------------------------------------------------------------- RSNA, 1024
# An opacity box measures 209 +/- 63 x 339 +/- 149 px (short x long side):
# 256 px is about one opacity, 128 px a part of one.
reg('rsna_base1024', RSNA_BASE, 'baseline', 'core,base', 'full image', baseline_size=1024)
reg('rsna_base512',  RSNA_BASE, 'baseline', 'core,base', 'image at half resolution', baseline_size=512)
reg('rsna_mil128',   RSNA_BASE, 'mil', 'core', 'attention MIL, all 64 patches of 128', scales=[128])
reg('rsna_mil256',   RSNA_BASE, 'mil', 'core', 'attention MIL, all 16 patches of 256', scales=[256])
reg('rsna_ips256', RSNA_BASE, 'selector', 'core,patch', 'about one opacity per patch', scales=[256])
reg('rsna_ips128', RSNA_BASE, 'selector', 'core,patch,mstudy', 'IPS reference', scales=[128])
reg('rsna_ips64',  RSNA_BASE, 'selector', 'core,patch,mstudy', 'control of h256_64', scales=[64])
reg('rsna_ms256_128', RSNA_BASE, 'selector', 'core,ms', 'MS-IPS, flat, native', scales=[256, 128])
reg('rsna_ms128_64',  RSNA_BASE, 'selector', 'core,ms', 'MS-IPS, flat, native', scales=[128, 64])
reg('rsna_h256_128', RSNA_BASE, 'hierarchical', 'core', 'H-MS-IPS, final 128', cascade=[256, 128], M_keep=[4])
reg('rsna_h256_64',  RSNA_BASE, 'hierarchical', 'core', 'H-MS-IPS, final 64', cascade=[256, 64], M_keep=[4])
reg('rsna_h256_128_tc', RSNA_BASE, 'hierarchical', 'ablation', 'coarse zones trained',
    cascade=[256, 128], M_keep=[4], train_coarse=4)
reg('rsna_h256_64_tc',  RSNA_BASE, 'hierarchical', 'ablation', 'coarse zones trained',
    cascade=[256, 64], M_keep=[4], train_coarse=4)
reg('rsna_ips32',  RSNA_BASE, 'selector', 'patch', '1024 candidates, map 1x1', scales=[32])
reg('rsna_h256_32', RSNA_BASE, 'hierarchical', 'patch', 'H-MS-IPS, final 32', cascade=[256, 32], M_keep=[4])
reg('rsna_base256',  RSNA_BASE, 'baseline', 'base', 'image at quarter resolution', baseline_size=256)
reg('rsna_ms128_64_rs', RSNA_BASE, 'selector', 'ms', 'MS-IPS, 64 px upsampled to 128',
    scales=[128, 64], encode_px=[None, 128])
reg('rsna_h256lr_64_tc', RSNA_BASE, 'hierarchical', 'lowres', 'coarse stage at 128 px',
    cascade=[256, 64], M_keep=[4], train_coarse=4, encode_px=[128, None])
reg('rsna_ips128_f25', RSNA_BASE, 'selector', 'lowdata', '25 % of train', scales=[128], train_frac=0.25)
reg('rsna_ips64_f25',  RSNA_BASE, 'selector', 'lowdata', '25 % of train', scales=[64], train_frac=0.25)


# ------------------------------------------------------------------ focus
# The selection of 18 November: DDR at 66 and 33 px, RSNA at 300 and 150 px,
# with the full image and MIL at the same sizes. These sizes do not divide
# the image: the last row and column of patches are completed with black.
reg('ddr_mil66',   DDR_BASE, 'mil', 'focus', 'attention MIL, all 196 patches of 66', scales=[66])
reg('ddr_mil33',   DDR_BASE, 'mil', 'focus', 'attention MIL, all 784 patches of 33', scales=[33])
reg('ddr_ips66',   DDR_BASE, 'selector', 'focus', '196 candidates', scales=[66])
reg('ddr_ips33',   DDR_BASE, 'selector', 'focus', '784 candidates', scales=[33])
reg('ddr_ms66_33', DDR_BASE, 'selector', 'focus', 'MS-IPS, flat', scales=[66, 33])
# 66 then 33 keeps a quarter of the zones, like every cascade (49 of 196)
reg('ddr_h66_33',  DDR_BASE, 'hierarchical', 'focus', 'H-MS-IPS, 49 of 196 zones kept',
    cascade=[66, 33], M_keep=[49])
reg('rsna_mil300',    RSNA_BASE, 'mil', 'focus', 'attention MIL, all 16 patches of 300', scales=[300])
reg('rsna_mil150',    RSNA_BASE, 'mil', 'focus', 'attention MIL, all 49 patches of 150', scales=[150])
reg('rsna_ips300',    RSNA_BASE, 'selector', 'focus', '16 candidates', scales=[300])
reg('rsna_ips150',    RSNA_BASE, 'selector', 'focus', '49 candidates', scales=[150])
reg('rsna_ms300_150', RSNA_BASE, 'selector', 'focus', 'MS-IPS, flat', scales=[300, 150])
reg('rsna_h300_150',  RSNA_BASE, 'hierarchical', 'focus', 'H-MS-IPS, 4 of 16 zones kept',
    cascade=[300, 150], M_keep=[4])
for _n in ('ddr_base896', 'rsna_base1024'):
    REGISTRY[_n][1].add('focus')
reg('ddr_h66_33_tc',    DDR_BASE, 'hierarchical', 'focus_abl,ablation', 'coarse zones trained',
    cascade=[66, 33], M_keep=[49], train_coarse=4)
reg('rsna_h300_150_tc', RSNA_BASE, 'hierarchical', 'focus_abl,ablation', 'coarse zones trained',
    cascade=[300, 150], M_keep=[4], train_coarse=4)


# ------------------------------------------------------------------ ablations
# Every selector of focus and of core (IPS, MS-IPS, H-MS-IPS, both data sets)
# is also trained with the self-attention aggregator and with random selection.
def _variant(src, suffix, groups, note, **kw):
    d = REGISTRY[src][0]
    base = DDR_BASE if d['dataset'] == 'ddr' else RSNA_BASE
    if d.get('img_size') != base['img_size']:
        base = dict(base, img_size=d['img_size'])
    keep = {k: d[k] for k in ('scales', 'cascade', 'M_keep', 'n_res_blocks', 'M')
            if k in d and d[k] != base.get(k)}
    keep.update(kw)
    reg(src + suffix, base, d['model'], groups, note, **keep)


def _selectors(group):
    return [n for n, (d, g, _) in list(REGISTRY.items())
            if group in g and d['model'] in ('selector', 'hierarchical')]


_FOCUS_SEL, _CORE_SEL = _selectors('focus'), _selectors('core')
for _n in _FOCUS_SEL + _CORE_SEL:
    _g = 'focus_abl,ablation' if _n in _FOCUS_SEL else 'ablation'
    _variant(_n, '_sa', _g, 'self-attention aggregator', aggregator='self')
    _variant(_n, '_rand', _g, 'random selection, trained', selection='random')

# ------------------------------------------------------------------ M study
# Every selector of focus and core, with every value of M below that is
# smaller than its pool of candidates (M = 10 is the run itself). The pool is
# the number of candidates of IPS and MS-IPS, and the number of fine
# candidates left by the coarse stage for H-MS-IPS.
M_VALUES = (2, 5, 20, 40, 80, 160)


def _pool(d):
    if d['model'] == 'hierarchical':
        c = d['cascade']
        return d['M_keep'][-1] * (c[-2] // c[-1]) ** 2
    return sum((-(-d['img_size'] // p)) ** 2 for p in d['scales'])


for _n in _FOCUS_SEL + _CORE_SEL:
    REGISTRY[_n][1].add('mstudy')
    _g = 'focus_m,mstudy' if _n in _FOCUS_SEL else 'mstudy'
    for _M in M_VALUES:
        if _M < _pool(REGISTRY[_n][0]):
            _variant(_n, f'_m{_M}', _g, f'M = {_M}', M=_M)

for _n, (_d, _g, _) in REGISTRY.items():          # two patch sizes at most
    assert len(_d['scales']) <= 2, _n

GROUPS = sorted({g for _, gs, _ in REGISTRY.values() for g in gs})

# The 21 runs of the v2 suite, so that their saved weights can be scored on
# the held-out images (IPS_MODE=eval_v2). Keys are the v2 run names.
V2 = {
    'ddr_baseline_v2': (DDR_BASE, 'baseline', dict(baseline_size=896)),
    'ddr_ips_v2': (DDR_BASE, 'selector', dict(scales=[128])),
    'ddr_msips2_native_v2': (DDR_BASE, 'selector', dict(scales=[128, 64])),
    'ddr_msips3_native_v2': (DDR_BASE, 'selector', dict(scales=[128, 64, 32], n_res_blocks=2)),
    'ddr_ips_b2_v2': (DDR_BASE, 'selector', dict(scales=[128], n_res_blocks=2)),
    'ddr_msips2_native_b2_v2': (DDR_BASE, 'selector', dict(scales=[128, 64], n_res_blocks=2)),
    'ddr_hmsips_d2_v2': (DDR_BASE, 'hierarchical', dict(cascade=[224, 112], M_keep=[4])),
    'ddr_hmsips_d4_v2': (DDR_BASE, 'hierarchical', dict(cascade=[224, 56], M_keep=[4])),
    'ddr_hmsips_d8_v2': (DDR_BASE, 'hierarchical', dict(cascade=[224, 28], M_keep=[4], n_res_blocks=2)),
    'ddr_hmsips_d16_v2': (DDR_BASE, 'hierarchical', dict(cascade=[224, 14], M_keep=[4], n_res_blocks=2)),
    'ddr_hmsips3_v2': (DDR_BASE, 'hierarchical', dict(cascade=[224, 56, 14], M_keep=[4, 8], n_res_blocks=2)),
    'ddr_msips3_clinique_v2': (DDR_BASE, 'selector', dict(scales=[128, 32, 16], n_res_blocks=2)),
    'rsna_baseline_v2': (RSNA_BASE, 'baseline', dict(baseline_size=1024)),
    'rsna_ips_v2': (RSNA_BASE, 'selector', dict(scales=[128])),
    'rsna_msips2_native_v2': (RSNA_BASE, 'selector', dict(scales=[128, 64])),
    'rsna_msips3_native_v2': (RSNA_BASE, 'selector', dict(scales=[128, 64, 32], n_res_blocks=2)),
    'rsna_msips2_clinique_v2': (RSNA_BASE, 'selector', dict(scales=[256, 128])),
    'rsna_hmsips_d2_v2': (RSNA_BASE, 'hierarchical', dict(cascade=[256, 128], M_keep=[4])),
    'rsna_hmsips_d4_v2': (RSNA_BASE, 'hierarchical', dict(cascade=[256, 64], M_keep=[4])),
    'rsna_hmsips_d8_v2': (RSNA_BASE, 'hierarchical', dict(cascade=[256, 32], M_keep=[4], n_res_blocks=2)),
    'rsna_hmsips3_v2': (RSNA_BASE, 'hierarchical', dict(cascade=[256, 128, 64], M_keep=[4, 6])),
}


def make_v2_conf(run_name):
    base, model, kw = V2[run_name]
    # v2 weights were trained on the stretched photograph: evaluate them on it
    d = dict(base, name=run_name, model=model, seed=42, run_name=run_name,
             ddr_geometry='stretch', **kw)
    if model == 'hierarchical':
        d['scales'] = list(d['cascade'])
    if model == 'baseline':
        d['scales'] = [128]
    return Struct(**d)
