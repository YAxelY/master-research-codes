"""Tests of aggregate-ips-suite-v3.py on small synthetic results.

    python -m pytest tests
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import cohen_kappa_score

SCRIPT = Path(__file__).resolve().parents[1] / 'aggregate-ips-suite-v3.py'
N_IMAGES = 60


def predictions(seed, quality):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, N_IMAGES)
    logits = rng.normal(size=(N_IMAGES, 5))
    logits[np.arange(N_IMAGES), y] += quality
    prob = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
    return y, prob


def write_run(root, config, seed, y, prob, stored_test=None, setup=('kaggle', 'Tesla T4'),
              split_test='aaa'):
    kappa = cohen_kappa_score(y, prob.argmax(1), weights='quadratic')
    metrics = {'kappa': kappa if stored_test is None else stored_test, 'accuracy': 0.5,
               'loss': 1.0}
    result = {
        'run_name': f"{config}_s{seed}", 'name': config, 'seed': seed, 'main_metric': 'kappa',
        'best_epoch': 1,
        'config': {'task_type': 'ordinal', 'dataset': 'ddr', 'model': 'selector',
                   'scales': [128], 'M': 10, 'I': 128, 'n_epoch': 1},
        'cost': {'n_candidates': 49, 'explore_encoded_px': 1.0, 'img_px': 1.0, 'exploit_px_max': 1},
        'timing': {'train_peak_mb': 1000, 'train_peak_select_mb': 500, 'train_peak_step_mb': 1000,
                   'train_model_ms_per_img': 1.0, 'train_t_data_s': 1, 'train_t_select_s': 1,
                   'train_t_step_s': 1, 'total_co2_kg': 0.01},
        'val': dict(metrics), 'test': dict(metrics),
        'final_epoch': {'val': dict(metrics)},
        'train_at_best_epoch': {'loss': 1.0}, 'val_at_best_epoch': {'loss': 1.0},
        'environment': {'platform': setup[0], 'gpu': setup[1], 'code_commit': 'abc1234'},
        'data': {'split_hash': {'test': split_test, 'val': 'vvv', 'train': 'ttt'}},
    }
    folder = root / config
    (folder / 'experiments').mkdir(parents=True, exist_ok=True)
    (folder / 'predictions').mkdir(parents=True, exist_ok=True)
    (folder / 'experiments' / f"{config}_s{seed}_result.json").write_text(json.dumps(result))
    np.savez(folder / 'predictions' / f"{config}_s{seed}_test.npz",
             ids=np.arange(N_IMAGES), y_true=y, y_prob=prob)
    return kappa


def aggregate(root):
    out = subprocess.run([sys.executable, str(SCRIPT), str(root), '--boot', '50'],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    tables = (root / '_aggregate' / 'tables.md').read_text()
    return pd.read_csv(root / '_aggregate' / 'runs.csv'), tables


def test_scores_are_recomputed_from_predictions(tmp_path):
    y, prob = predictions(0, quality=2.0)
    kappa = write_run(tmp_path, 'ddr_ips128', 42, y, prob)
    runs, tables = aggregate(tmp_path)
    assert runs.loc[0, 'test_from_predictions'] == pytest.approx(kappa)
    assert 'Warnings' not in tables


def test_a_wrong_stored_score_is_flagged(tmp_path):
    y, prob = predictions(0, quality=2.0)
    write_run(tmp_path, 'ddr_ips128', 42, y, prob, stored_test=0.99)
    _, tables = aggregate(tmp_path)
    assert 'differs from the one recomputed from its predictions' in tables


def test_mixed_setups_and_splits_are_flagged(tmp_path):
    y, prob = predictions(0, quality=2.0)
    write_run(tmp_path, 'ddr_ips128', 42, y, prob)
    write_run(tmp_path, 'ddr_mil128', 42, *predictions(1, quality=1.0),
              setup=('slurm', 'NVIDIA A40'), split_test='bbb')
    runs, tables = aggregate(tmp_path)
    assert set(runs.setup) == {'kaggle/Tesla T4', 'slurm/NVIDIA A40'}
    assert 'setups' in tables and 'NOT comparable' in tables


def test_results_are_deterministic(tmp_path):
    y, prob = predictions(0, quality=2.0)
    write_run(tmp_path, 'ddr_ips128', 42, y, prob)
    first = aggregate(tmp_path)[0]
    second = aggregate(tmp_path)[0]
    pd.testing.assert_frame_equal(first, second)
