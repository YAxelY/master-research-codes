"""Runner"""

import fcntl
import json
import math
import numpy as np
import time
import torch
from pathlib import Path

from ips_suite.config.environment import CUDA, SUITE_VERSION, _env_int, _env_path, device, environment_info, epoch_seed, set_seed
from ips_suite.config.experiments import REGISTRY, RESULTS, make_conf, make_v2_conf, use_run_dirs
from ips_suite.data.datasets import build_dataset, build_loader, split_hash
from ips_suite.evaluation.localisation import gradcam_study, lesion_study
from ips_suite.evaluation.metrics import IPSLoss, compute_metrics, youden_threshold
from ips_suite.evaluation.selection import selection_study
from ips_suite.evaluation.splits import evaluate_split
from ips_suite.models.factory import build_model, cost_report, efficiency_profile
from ips_suite.training.epoch import IMAGE_MODELS, run_epoch
from ips_suite.training.hub import HUB, RunFiles
from ips_suite.config import experiments


FORCE = bool(_env_int('IPS_FORCE', 0))


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return o


def _conf_dict(conf):
    return {k: v for k, v in conf.__dict__.items() if k != 'data_dir'}


def post_training(net, conf, criterion, ds_val, ds_test, tag=''):
    """Everything measured with the weights of the best epoch."""
    is_sel = conf.model in ('selector', 'hierarchical')
    out = {}
    val_m, val_o = evaluate_split(net, ds_val, conf, 'val', criterion, tag=tag)
    test_m, test_o = evaluate_split(net, ds_test, conf, 'test', criterion, tag=tag)
    out['val'], out['test'] = val_m, test_m
    if conf.task_type == 'binary':
        thr = youden_threshold(val_o['y_true'], np.asarray(val_o['y_prob'])[:, 1])
        out['threshold_from_val'] = thr
        out['test_at_val_threshold'] = compute_metrics(
            test_o['y_true'], test_o['y_pred'], test_o['y_prob'], 'binary', thr)
    def _given(o):
        if o['coords'] is None:
            return None
        z = o['zones']
        return {i: (o['coords'][j], (z[j] if z is not None else None))
                for j, i in enumerate(o['ids'])}
    g_val, g_test = _given(val_o), _given(test_o)
    key = conf.main_metric
    print(f"  best epoch weights : val {key} {val_m.get(key, float('nan')):.4f} | "
          f"TEST {key} {test_m.get(key, float('nan')):.4f}  (n={test_m['n']})")
    jobs = []
    if is_sel and conf.get('selection', 'attention') == 'attention':
        jobs.append(('selection_test', lambda: selection_study(net, ds_test, conf, device, criterion)))
    if conf.model != 'baseline':          # selectors and MIL (its top-M by attention)
        jobs += [('lesions_test', lambda: lesion_study(net, ds_test, conf, device, split='test', given=g_test)),
                 ('lesions_val', lambda: lesion_study(net, ds_val, conf, device, split='val', given=g_val))]
    jobs.append(('gradcam_test', lambda: gradcam_study(net, ds_test, conf, split='test', given=g_test)))
    if jobs:
        for nm, fn in jobs:
            try:
                out[nm] = fn()
            except Exception as exc:        # the run itself is already saved
                import traceback
                traceback.print_exc()
                out[nm] = {'error': repr(exc)}
    return out


def run(name, seed, n_epoch=None):
    conf = make_conf(name, seed)
    if n_epoch is not None:
        conf.n_epoch = n_epoch
    use_run_dirs(name)
    files = RunFiles(name, conf.run_name)
    if not FORCE:
        HUB.pull(files.restorable)
    res_path = files.result
    if res_path.exists() and not FORCE:
        print(f"[skip] {conf.run_name} : result exists (IPS_FORCE=1 to redo)")
        return None
    # if the same configuration and seed is already running in another job,
    # skip it instead of writing the same files twice
    _lock = open(experiments.OUT_CKPT / f"{conf.run_name}.lock", 'w')
    try:
        fcntl.flock(_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print(f"[skip] {conf.run_name} : already running in another job")
        return None
    set_seed(conf.seed)
    print(f"\n{'='*72}\n{conf.run_name}  ({conf.model}, {conf.dataset})  "
          f"{REGISTRY[name][2]}\n{'='*72}")

    ds_tr = build_dataset(conf, 'train')
    ds_va = build_dataset(conf, 'val')
    ds_te = build_dataset(conf, 'test')
    train_loader = build_loader(conf, ds_tr, train=True)
    val_loader = build_loader(conf, ds_va, train=False)

    net = build_model(conf, device)
    cost = cost_report(conf, net)
    cost.update(efficiency_profile(conf, net))
    print(f"  parameters         : {cost['params']/1e6:.2f} M"
          + (f" | GFLOPs per image : {cost['gflops_infer']:.2f}" if 'gflops_infer' in cost else ''))
    criterion = IPSLoss(conf.eps)
    optimizer = torch.optim.AdamW(net.parameters(), lr=0, weight_decay=conf.wd)
    is_base = conf.model in IMAGE_MODELS
    key = conf.main_metric

    w_path, s_path = files.weights, files.state
    # runs interrupted before the move of checkpoints out of RESULTS resume
    # from their old place; new states are written to ARTIFACTS
    _old_state = RESULTS / name / 'checkpoints' / f"{conf.run_name}_state.pt"
    _load = s_path if s_path.exists() or not _old_state.exists() else _old_state
    history, best_val, best_ep, start_ep, co2_prev = [], -float('inf'), 0, 0, 0.0
    if _load.exists() and not FORCE:
        st = torch.load(_load, map_location=device, weights_only=False)
        if st.get('n_epoch') == conf.n_epoch:
            net.load_state_dict(st['model'])
            optimizer.load_state_dict(st['optim'])
            history, best_val, best_ep = st['history'], st['best_val'], st['best_ep']
            start_ep, co2_prev = st['epoch'], st.get('co2', 0.0)
            print(f"  [resume] from epoch {start_ep}")

    tracker = None
    try:
        from codecarbon import EmissionsTracker
        _co2_dir = experiments.OUT_HIST / 'codecarbon'
        _co2_dir.mkdir(parents=True, exist_ok=True)
        tracker = EmissionsTracker(output_dir=str(_co2_dir), log_level='error')
        tracker.start()
    except Exception as exc:
        print(f"  [info] codecarbon unavailable ({exc})")

    for ep in range(start_ep, conf.n_epoch):
        seed_of_epoch = epoch_seed(conf.seed, ep)
        set_seed(seed_of_epoch)
        train_loader.generator.manual_seed(seed_of_epoch)
        if hasattr(net, 'reseed'):
            net.reseed(seed_of_epoch)
        tr = run_epoch(net, train_loader, conf, device, optimizer, ep,
                       criterion, is_base)
        if not is_base:
            net.reseed(conf.seed + 1000)        # same validation pass every epoch
        va = run_epoch(net, val_loader, conf, device, None, ep, criterion, is_base)

        tr_m = compute_metrics(tr['y_true'], tr['y_pred'], tr['y_prob'], conf.task_type)
        va_m = compute_metrics(va['y_true'], va['y_pred'], va['y_prob'], conf.task_type)
        tr_m['loss'], va_m['loss'] = tr['loss'], va['loss']
        history.append({'epoch': ep + 1, 'lr': optimizer.param_groups[0]['lr'],
                        'train': tr_m, 'val': va_m,
                        'prof': {**tr['prof'].summary('train_'),
                                 **va['prof'].summary('val_')}})

        score, shown = va_m.get(key, va_m['accuracy']), key
        if isinstance(score, float) and math.isnan(score):
            score, shown = va_m['accuracy'], key + '~acc'
        flag = ''
        if score > best_val:
            best_val, best_ep, flag = score, ep + 1, '  *'
            torch.save(net.state_dict(), w_path)
        p = history[-1]['prof']
        print(f"  ep {ep+1:3d}/{conf.n_epoch}  train {tr['loss']:.4f}  "
              f"val {va['loss']:.4f}  {shown} {score:.4f}{flag}   "
              f"[data {p.get('train_t_data_s', 0):.0f}s "
              f"aug {p.get('train_t_augment_s', 0):.0f}s "
              f"select {p.get('train_t_select_s', 0):.0f}s "
              f"step {p.get('train_t_step_s', 0):.0f}s | "
              f"peak {p.get('train_peak_mb', 0):.0f} MB]")
        torch.save({'epoch': ep + 1, 'n_epoch': conf.n_epoch,
                    'model': net.state_dict(), 'optim': optimizer.state_dict(),
                    'history': history, 'best_val': best_val,
                    'best_ep': best_ep, 'co2': co2_prev}, s_path)
        if HUB.is_due(ep + 1, conf.n_epoch):
            HUB.push(files.resume_point, f"{conf.run_name}: epoch {ep + 1}")

    co2 = co2_prev
    energy = None
    if tracker is not None:
        try:
            co2 += float(tracker.stop() or 0.0)
            energy = float(tracker.final_emissions_data.energy_consumed)
        except Exception:
            pass

    def tot(k):
        return float(sum(h['prof'].get(k, 0.0) for h in history))

    def peak(k):
        return float(max((h['prof'].get(k, 0.0) for h in history), default=0.0))

    n_tr = max(sum(h['prof'].get('train_n_img', 0) for h in history), 1)
    timing = {
        'train_t_data_s': tot('train_t_data_s'),
        'train_t_augment_s': tot('train_t_augment_s'),
        'train_t_select_s': tot('train_t_select_s'),
        'train_t_step_s': tot('train_t_step_s'),
        'val_t_total_s': tot('val_t_data_s') + tot('val_t_select_s') + tot('val_t_step_s'),
        'train_model_ms_per_img': 1000 * (tot('train_t_select_s') + tot('train_t_step_s')) / n_tr,
        'train_peak_mb': peak('train_peak_mb'),
        'train_peak_select_mb': peak('train_peak_select_mb'),
        'train_peak_step_mb': peak('train_peak_step_mb'),
        'val_peak_mb': peak('val_peak_mb'),
        'total_co2_kg': co2, 'energy_kwh_last_session': energy,
        'gpu': torch.cuda.get_device_name(0) if CUDA else 'cpu',
    }

    net.load_state_dict(torch.load(w_path, map_location=device, weights_only=True))
    best_row = history[best_ep - 1]
    result = {
        'suite': SUITE_VERSION, 'run_name': conf.run_name, 'name': name,
        'seed': conf.seed, 'date': time.strftime('%Y-%m-%d %H:%M'),
        'note': REGISTRY[name][2], 'groups': sorted(REGISTRY[name][1]),
        'config': _conf_dict(conf), 'cost': cost,
        'n_train': len(ds_tr), 'n_val': len(ds_va), 'n_test': len(ds_te),
        'best_epoch': best_ep, 'main_metric': key,
        'val_at_best_epoch': best_row['val'],
        'train_at_best_epoch': best_row['train'],
        'final_epoch': {'train': history[-1]['train'], 'val': history[-1]['val']},
        'timing': timing, 'history': history,
        'environment': environment_info(),
        'data': {'split_hash': {'train': split_hash(ds_tr), 'val': split_hash(ds_va),
                                'test': split_hash(ds_te)}},
    }
    with open(experiments.OUT_HIST / f"{conf.run_name}_history.json", 'w') as f:
        json.dump(_jsonable(history), f)
    with open(res_path.with_suffix('.partial.json'), 'w') as f:
        json.dump(_jsonable(result), f)
    result.update(post_training(net, conf, criterion, ds_va, ds_te))
    with open(res_path, 'w') as f:
        json.dump(_jsonable(result), f, indent=1)
    res_path.with_suffix('.partial.json').unlink(missing_ok=True)
    s_path.unlink(missing_ok=True)
    print(f"  result : {res_path}")
    if HUB.push(files.final(), f"{conf.run_name}: final results"):
        HUB.delete(files.state, f"{conf.run_name}: resume state no longer needed")
    return result


def eval_v2(run_name):
    """Score the saved weights of a v2 run on the held-out images."""
    conf = make_v2_conf(run_name)
    ckpt = _env_path('IPS_V2_CKPT', RESULTS) / f"{run_name}_weights.pt"
    use_run_dirs(run_name)
    res_path = experiments.OUT_EXPE / f"{run_name}_evalv3.json"
    if not ckpt.exists():
        print(f"[skip] {run_name} : no weights at {ckpt}")
        return None
    if res_path.exists() and not FORCE:
        print(f"[skip] {run_name} : already evaluated")
        return None
    print(f"\n{'='*72}\n{run_name}  (v2 weights, evaluation only)\n{'='*72}")
    set_seed(conf.seed)
    net = build_model(conf, device)
    net.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    cost = cost_report(conf, net)
    cost.update(efficiency_profile(conf, net))
    ds_va = build_dataset(conf, 'val')
    ds_te = build_dataset(conf, 'test')
    result = {'suite': 'v2-weights', 'run_name': run_name, 'seed': conf.seed,
              'config': _conf_dict(conf), 'cost': cost,
              # v2 chose the epoch on DDR valid (test never seen: clean). On
              # RSNA, v2 chose its epoch on images that are now in val and
              # test, and trained on 5 % of the patients now in val or test.
              'test_is_clean': conf.dataset == 'ddr'}
    result.update(post_training(net, conf, IPSLoss(conf.eps), ds_va, ds_te,
                                tag='_evalv3'))
    with open(res_path, 'w') as f:
        json.dump(_jsonable(result), f, indent=1)
    print(f"  result : {res_path}")
    return result
