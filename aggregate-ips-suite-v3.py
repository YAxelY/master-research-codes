#!/usr/bin/env python3
"""Aggregate the results of ips-suite-v3. CPU only, no data set needed.

    python aggregate-ips-suite-v3.py RESULTS_DIR [--boot 2000]

RESULTS_DIR is the results folder of the suite: one sub-folder per
configuration, each with `experiments/` (the *_result.json files) and
`predictions/` (the *.npz files).

Writes into RESULTS_DIR/aggregate/ :
  runs.csv        one row per run (one seed)
  configs.csv     mean, standard deviation and number of seeds per config
  pairs.csv       paired bootstrap of the test score, for the comparisons
                  the suite was built for (same test images, same seed)
  tables.md       the tables, ready to read
Units: memory in GB = MB / 1000. Scores are TEST scores at the epoch chosen
on validation, unless a column says `val`.

Experimental setup: every run records where it ran (`setup`, e.g.
kaggle/Tesla T4 or slurm/NVIDIA A40) and a hash of the images of each split.
Scores of runs on the same split hashes are comparable whatever the setup;
time and memory are comparable only within one GPU type. tables.md lists the
setups and warns when a table or a pair mixes them. Runs made before suite
v3.1 have no setup record and show `unrecorded`.
"""
import sys, re, json, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score

# (a, b, question): the difference reported is a - b
PAIRS = [
    # coarse stage, at the same final patch size
    ('ddr_h224_112', 'ddr_ips112', 'cascade against IPS, same final patch'),
    ('ddr_h224_56', 'ddr_ips56', 'cascade against IPS, same final patch'),
    ('ddr_h224_28', 'ddr_ips28', 'cascade against IPS, same final patch'),
    ('rsna_h256_128', 'rsna_ips128', 'cascade against IPS, same final patch'),
    ('rsna_h256_64', 'rsna_ips64', 'cascade against IPS, same final patch'),
    ('rsna_h256_32', 'rsna_ips32', 'cascade against IPS, same final patch'),
    # flat multi-scale against its two single-size parts
    ('ddr_ms128_64', 'ddr_ips128', 'MS-IPS against IPS at its larger size'),
    ('ddr_ms128_64', 'ddr_ips64', 'MS-IPS against IPS at its smaller size'),
    ('ddr_ms64_32', 'ddr_ips64', 'MS-IPS against IPS at its larger size'),
    ('ddr_ms64_32', 'ddr_ips32', 'MS-IPS against IPS at its smaller size'),
    ('rsna_ms256_128', 'rsna_ips256', 'MS-IPS against IPS at its larger size'),
    ('rsna_ms256_128', 'rsna_ips128', 'MS-IPS against IPS at its smaller size'),
    ('rsna_ms128_64', 'rsna_ips128', 'MS-IPS against IPS at its larger size'),
    ('rsna_ms128_64', 'rsna_ips64', 'MS-IPS against IPS at its smaller size'),
    # baselines
    ('ddr_ips128', 'ddr_base448', 'IPS against the image at half resolution'),
    ('ddr_ips128', 'ddr_mil128', 'IPS against attention MIL on all patches'),
    ('rsna_ips128', 'rsna_base512', 'IPS against the image at half resolution'),
    ('rsna_ips128', 'rsna_mil128', 'IPS against attention MIL on all patches'),
    # ablation of the selection module
    ('ddr_ips128', 'ddr_ips128_sa', 'cross-attention against self-attention'),
    ('ddr_ips128', 'ddr_ips128_rand', 'learned against random selection'),
    ('rsna_ips128', 'rsna_ips128_sa', 'cross-attention against self-attention'),
    ('rsna_ips128', 'rsna_ips128_rand', 'learned against random selection'),
    ('ddr_h224_112_tc', 'ddr_h224_112', 'coarse zones trained or not'),
    ('ddr_h224_56_tc', 'ddr_h224_56', 'coarse zones trained or not'),
    ('rsna_h256_128_tc', 'rsna_h256_128', 'coarse zones trained or not'),
    ('rsna_h256_64_tc', 'rsna_h256_64', 'coarse zones trained or not'),
    # equal trained pixels, resolution, resampling
    ('ddr_ips64_m40', 'ddr_ips128', 'equal pixels: 40 x 64 against 10 x 128'),
    ('ddr_ips32_m160', 'ddr_ips128', 'equal pixels: 160 x 32 against 10 x 128'),
    ('rsna_ips64_m40', 'rsna_ips128', 'equal pixels: 40 x 64 against 10 x 128'),
    ('ddr_r1792_ips128', 'ddr_ips128', 'twice the resolution, same memory'),
    ('ddr_r1792_ips128', 'ddr_base896', 'IPS at 1792 px against the full image at 896 px'),
    ('ddr_ms128_64_rs', 'ddr_ms128_64', 'resampled against native'),
]


def score(y, p, task):
    if task == 'ordinal':
        return cohen_kappa_score(y, p.argmax(1), weights='quadratic')
    return roc_auc_score(y, p[:, 1])


def boot(y, p, task, n, rng, q=None):
    """Bootstrap of one score (q is None) or of a paired difference."""
    out, N = [], len(y)
    for _ in range(n):
        i = rng.integers(0, N, N)
        try:
            s = score(y[i], p[i], task)
            if q is not None:
                s -= score(y[i], q[i], task)
            out.append(s)
        except ValueError:                  # a resample with one class only
            continue
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def load_pred(root, run, split='test'):
    """Per-image predictions of a run, wherever its configuration folder is."""
    hits = sorted(root.glob(f"*/predictions/{run}_{split}.npz"))
    if not hits:
        return None
    f = hits[0]
    z = np.load(f, allow_pickle=False)
    o = np.argsort(z['ids'])
    return z['y_true'][o], z['y_prob'][o].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--boot', type=int, default=2000)
    a = ap.parse_args()
    root = Path(a.root)
    out = root / '_aggregate'
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(0)

    rows = []
    for f in sorted(root.glob('*/experiments/*_result.json')):
        r = json.load(open(f))
        c, k, t, key = r['config'], r['cost'], r['timing'], r['main_metric']
        task = c['task_type']
        env = r.get('environment') or {}
        sh = (r.get('data') or {}).get('split_hash') or {}
        gpu = env.get('gpu') or t.get('gpu') or '?'
        row = dict(
            run=r['run_name'], config=r['name'], seed=r['seed'], dataset=c['dataset'],
            setup=f"{env.get('platform', 'unrecorded')}/{gpu}", gpu=gpu,
            code_commit=env.get('code_commit'), rng=env.get('rng', 'per-run seed (v3.0)'),
            split_test=sh.get('test'), split_val=sh.get('val'), split_train=sh.get('train'),
            model=c['model'],
            scales=(f"base{c.get('baseline_size')}" if c['model'] == 'baseline'
                    else '-'.join(map(str, c['scales']))),
            M=c['M'], I=c['I'], train_coarse=c.get('train_coarse', 0),
            train_frac=c.get('train_frac', 1.0), loading=c.get('loading', 'eager'),
            epochs=c['n_epoch'], best_epoch=r['best_epoch'], metric=key,
            candidates=k['n_candidates'],
            explore_x_img=k['explore_encoded_px'] / k['img_px'],
            train_px=k['exploit_px_max'],
            val=r['val'].get(key), test=r['test'].get(key),
            test_acc=r['test'].get('accuracy'), test_f1=r['test'].get('f1'),
            val_minus_test=r['val'].get(key, np.nan) - r['test'].get(key, np.nan),
            last_val=r['final_epoch']['val'].get(key),
            train_loss_best=r['train_at_best_epoch']['loss'],
            val_loss_best=r['val_at_best_epoch']['loss'],
            peak_gb=t['train_peak_mb'] / 1000,
            peak_select_gb=t['train_peak_select_mb'] / 1000,
            peak_step_gb=t['train_peak_step_mb'] / 1000,
            infer_peak_gb=r['test'].get('eval_peak_mb', 0) / 1000,
            infer_ms_img=r['test'].get('ms_per_img_select', 0) + r['test'].get('ms_per_img_step', 0),
            train_model_ms_img=t['train_model_ms_per_img'],
            train_data_h=t['train_t_data_s'] / 3600,
            train_model_h=(t['train_t_select_s'] + t['train_t_step_s']) / 3600,
            co2_kg=t['total_co2_kg'],
            params_m=k.get('params', np.nan) / 1e6,
            gflops_infer=k.get('gflops_infer'), gflops_select=k.get('gflops_select'),
            gflops_classify=k.get('gflops_classify'),
            test_ece=r['test'].get('ece'), test_brier=r['test'].get('brier'),
            test_kappa_linear=r['test'].get('kappa_linear'),
            test_auc_referable=r['test'].get('auc_referable'),
            test_auc_macro=r['test'].get('auc_macro_ovr'),
            attn_top1_share=r['test'].get('attn_top1_share'))
        gc = r.get('gradcam_test', {})
        G = gc.get('any') or gc.get('opacity')
        if G:
            row.update(gc_energy_ratio=G['energy_ratio'], gc_pointing=G['pointing_game'],
                       gc_auroc=G['pixel_auroc'], gc_n=G['n_img'])
        if 'test_at_val_threshold' in r:
            m = r['test_at_val_threshold']
            row.update(test_sens=m['recall'], test_spec=m['specificity'],
                       test_auprc=r['test'].get('auprc'))
        sel = r.get('selection_test', {})
        if 'gain_positions' in sel:
            row.update(gain_scales=sel['gain_scales'], gain_positions=sel['gain_positions'],
                       uniform=sel['uniform'].get(key))
        for split in ('test', 'val'):
            les = r.get(f'lesions_{split}', {})
            L = les.get('any') or les.get('opacity')
            if L:
                row.update({f'les_{split}_n': L['n_img'], f'les_{split}_hit': L['hit_rate'],
                            f'les_{split}_chance': L['chance_rate'],
                            f'les_{split}_x': L['enrichment'],
                            f'les_{split}_cover': L['lesion_cover'],
                            f'les_{split}_zone_cover': L.get('zone_cover'),
                            f'les_{split}_zone_lost': L.get('zone_lost')})
            if split == 'test':
                for typ in ('MA', 'HE', 'EX', 'SE'):
                    if typ in les:
                        row[f'les_{typ}_x'] = les[typ]['enrichment']
                        row[f'les_{typ}_hit'] = les[typ]['hit_rate']
        pr = load_pred(root, r['run_name'])
        if pr is not None and a.boot:
            row['ci_lo'], row['ci_hi'] = boot(pr[0], pr[1], task, a.boot, rng)
        rows.append(row)
    if not rows:
        sys.exit(f"no */experiments/*_result.json in {root}")
    runs = pd.DataFrame(rows)
    runs.to_csv(out / 'runs.csv', index=False)

    num = [c for c in runs.columns if runs[c].dtype.kind in 'fi'
           and c not in ('seed', 'M', 'I', 'train_coarse', 'epochs', 'candidates')]
    g = runs.groupby('config', sort=False)
    conf = g[['dataset', 'model', 'scales', 'M', 'train_coarse', 'candidates', 'metric']].first()
    conf['n_seeds'] = g.size()
    conf['setups'] = g['setup'].agg(lambda v: ', '.join(sorted(set(v))))
    for c in num:
        conf[c] = g[c].mean()
        if c in ('test', 'val', 'peak_gb', 'les_test_x', 'gain_positions',
                 'test_auc_referable', 'test_ece'):
            conf[c + '_sd'] = g[c].std(ddof=1)
    conf.reset_index().to_csv(out / 'configs.csv', index=False)

    # every ablation is also compared with the model it was derived from
    cfgs = set(runs.config)
    auto = []
    for c in sorted(cfgs):
        for suf, why in (('_sa', 'self-attention against cross-attention'),
                         ('_rand', 'random against learned selection'),
                         ('_tc', 'coarse zones trained against not')):
            if c.endswith(suf) and c[:-len(suf)] in cfgs:
                auto.append((c, c[:-len(suf)], why))
        mm = re.match(r'(.+)_m(\d+)$', c)
        if mm and mm.group(1) in cfgs:
            auto.append((c, mm.group(1), f'M = {mm.group(2)} against M = 10'))
    have = {frozenset((x, y)) for x, y, _ in PAIRS}
    all_pairs = PAIRS + [t for t in auto if frozenset(t[:2]) not in have]
    prow = []
    for x, y, why in all_pairs:
        for seed in sorted(set(runs[runs.config == x].seed) & set(runs[runs.config == y].seed)):
            px, py = load_pred(root, f"{x}_s{seed}"), load_pred(root, f"{y}_s{seed}")
            if px is None or py is None or len(px[0]) != len(py[0]):
                continue
            task = 'ordinal' if px[1].shape[1] > 2 else 'binary'
            d = score(px[0], px[1], task) - score(py[0], py[1], task)
            lo, hi = boot(px[0], px[1], task, a.boot, rng, py[1]) if a.boot else (np.nan, np.nan)
            rx = runs[(runs.config == x) & (runs.seed == seed)].iloc[0]
            ry = runs[(runs.config == y) & (runs.seed == seed)].iloc[0]
            same_split = (pd.isna(rx.split_test) or pd.isna(ry.split_test)
                          or rx.split_test == ry.split_test)
            prow.append(dict(a=x, b=y, question=why, seed=seed, diff=d, ci_lo=lo, ci_hi=hi,
                             excludes_zero=bool(lo > 0 or hi < 0),
                             same_setup=rx.setup == ry.setup, same_test_images=same_split))
    pairs = pd.DataFrame(prow)
    pairs.to_csv(out / 'pairs.csv', index=False)

    def md(df, cols, fmt):
        h = '| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n'
        body = ''
        for _, r in df.iterrows():
            body += '| ' + ' | '.join(
                '' if pd.isna(r.get(c)) else (fmt.get(c, '{}').format(r[c])) for c in cols) + ' |\n'
        return h + body

    f3 = '{:.3f}'
    fmt = dict(test=f3, test_sd=f3, val=f3, val_minus_test='{:+.3f}', peak_gb='{:.2f}',
               peak_step_gb='{:.2f}', peak_select_gb='{:.2f}', infer_peak_gb='{:.2f}',
               infer_ms_img='{:.1f}', train_px='{:.0f}', explore_x_img='{:.2f}',
               les_test_x='{:.2f}', les_test_zone_cover='{:.2f}', gain_positions='{:+.3f}',
               gain_scales='{:+.3f}', train_model_h='{:.1f}', train_data_h='{:.1f}',
               diff='{:+.4f}', ci_lo='{:+.4f}', ci_hi='{:+.4f}', train_loss_best=f3,
               gflops_infer='{:.1f}', params_m='{:.1f}', gc_energy_ratio='{:.2f}',
               gc_pointing='{:.2f}', gc_auroc='{:.3f}')
    cr = conf.reset_index()
    txt = "# ips-suite-v3 : aggregate\n\nTest scores at the epoch chosen on validation. " \
          "Memory in GB (MB / 1000). `_sd` is the standard deviation over seeds.\n"
    txt += "\n## Experimental setups\n\n" + md(
        runs.groupby('setup').agg(runs=('run', 'size'), configs=('config', 'nunique'),
                                  commits=('code_commit', lambda v: ', '.join(sorted({str(x) for x in v if x})) or '-'))
        .reset_index(), ['setup', 'runs', 'configs', 'commits'], {})
    warn = []
    for ds in runs.dataset.unique():
        d = runs[runs.dataset == ds]
        if d.setup.nunique() > 1:
            warn.append(f"{ds.upper()}: runs come from {d.setup.nunique()} setups "
                        f"({', '.join(sorted(d.setup.unique()))}); compare times and memory "
                        f"only within one GPU type.")
        for col in ('split_test', 'split_val'):
            h = d[col].dropna().unique()
            if len(h) > 1:
                warn.append(f"{ds.upper()}: {len(h)} different `{col}` hashes: some runs did "
                            f"not use the same images; their scores are NOT comparable.")
    if warn:
        txt += "\n**Warnings**\n\n" + "\n".join(f"- {w}" for w in warn) + "\n"
    for ds in cr.dataset.unique():
        d = cr[cr.dataset == ds].sort_values('test', ascending=False)
        txt += f"\n## {ds.upper()} : score and cost\n\n" + md(
            d, ['config', 'setups', 'n_seeds', 'M', 'candidates', 'train_px', 'test', 'test_sd', 'val',
                'val_minus_test', 'peak_gb', 'peak_select_gb', 'peak_step_gb',
                'infer_peak_gb', 'infer_ms_img', 'gflops_infer', 'params_m',
                'train_model_h', 'train_data_h'], fmt)
        txt += f"\n## {ds.upper()} : what the selection does, and Grad-CAM\n\n" + md(
            d,
            ['config', 'gain_scales', 'gain_positions', 'les_test_n', 'les_test_x',
             'les_test_zone_cover', 'gc_energy_ratio', 'gc_pointing', 'gc_auroc',
             'train_loss_best'], fmt)
    if len(pairs):
        pm = pairs.groupby(['a', 'b', 'question'], sort=False).agg(
            seeds=('seed', 'size'), diff=('diff', 'mean'), ci_lo=('ci_lo', 'min'),
            ci_hi=('ci_hi', 'max'), all_exclude_zero=('excludes_zero', 'all'),
            same_setup=('same_setup', 'all'), same_images=('same_test_images', 'all')).reset_index()
        txt += ("\n## Paired comparisons (a minus b, same test images)\n\n"
                "`diff` is the mean over seeds; the interval is the widest over seeds "
                "(bootstrap on test images, 95 %).\n\n"
                + md(pm, ['a', 'b', 'question', 'seeds', 'diff', 'ci_lo', 'ci_hi',
                          'all_exclude_zero', 'same_setup', 'same_images'], fmt))
    (out / 'tables.md').write_text(txt)
    print(txt)
    print(f"\n{len(runs)} runs, {len(conf)} configs, {len(pairs)} paired rows -> {out}")


if __name__ == '__main__':
    main()
