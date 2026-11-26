"""Entry point: python -m ips_suite (IPS_MODE = train | bench | eval_v2 | list)."""
import pandas as pd
import torch

from ips_suite.config.environment import _env, _env_int
from ips_suite.config.experiments import GROUPS, REGISTRY, RESULTS, V2, make_conf
from ips_suite.evaluation.benchmark import bench
from ips_suite.models.factory import build_model, cost_report
from ips_suite.training.runner import eval_v2, run


def _expand(sel, table, groups=True):
    out = []
    for k in sel:
        k = k.strip()
        if k == 'all':
            out += list(table)
        elif groups and k in GROUPS:
            out += [n for n, (_, gs, _) in REGISTRY.items() if k in gs]
        elif k in table:
            out.append(k)
        else:
            raise ValueError(f"unknown run or group : {k}. "
                             f"Groups : {GROUPS}. Runs : {list(table)}")
    return list(dict.fromkeys(out))



def main():
    MODE = _env('IPS_MODE', 'train')
    _sel = [k for k in _env('IPS_RUNS', 'core').split(',') if k.strip()]
    SEEDS = [int(s) for s in _env('IPS_SEEDS', '42').split(',') if s.strip()]
    EPOCHS = _env_int('IPS_EPOCHS', 0) or None
    if MODE == 'list':
        rows = []
        for n, (d, gs, note) in REGISTRY.items():
            c = make_conf(n, 0)
            c.pretrained = False
            net = build_model(c, torch.device('cpu'))
            k = cost_report(c, net, verbose=False)
            rows.append(dict(run=n, groups='+'.join(sorted(gs)), M=c.M,
                             candidates=k['n_candidates'],
                             explore_x_img=round(k['explore_encoded_px'] / k['img_px'], 3),
                             train_px=k['exploit_px_max'],
                             train_pct_img=round(100 * k['exploit_px_max'] / k['img_px'], 1),
                             note=note))
            del net
        _df = pd.DataFrame(rows)
        _df.to_csv(RESULTS / 'run_list.csv', index=False)
        print(_df.to_string(index=False))
    elif MODE == 'bench':
        bench(quick=bool(_env_int('IPS_BENCH_QUICK', 0)))
    elif MODE == 'eval_v2':
        _todo = _expand(_sel if _env('IPS_RUNS') else ['all'], V2, groups=False)
        for _k in _todo:
            try:
                eval_v2(_k)
            except Exception as _exc:
                import traceback
                print(f"\n[FAILED] {_k} : {_exc!r}")
                traceback.print_exc()
    elif MODE == 'train':
        TO_RUN = _expand(_sel, REGISTRY)
        print(f"To run : {TO_RUN}  seeds {SEEDS}"
              + (f"  ({EPOCHS} epochs)" if EPOCHS else ""))
        _failed = []
        for _seed in SEEDS:
            for _key in TO_RUN:
                try:
                    run(_key, _seed, n_epoch=EPOCHS)
                except Exception as _exc:
                    import traceback
                    _failed.append((_key, _seed, repr(_exc)))
                    print(f"\n[FAILED] {_key} seed {_seed} : {_exc!r}")
                    traceback.print_exc()
                    print("  -> continuing with the next run\n")
        if _failed:
            print(f"\n{len(_failed)} run(s) failed :")
            for _k, _s, _e in _failed:
                print(f"  {_k} s{_s} : {_e}")
    else:
        raise ValueError(f"IPS_MODE={MODE}")


if __name__ == '__main__':
    main()
