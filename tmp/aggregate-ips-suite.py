#!/usr/bin/env python3
"""Agrege les runs de ips-suite : table par jeu, frontiere memoire/qualite,
repartition des echelles retenues, et tirages de reference.

    python3 aggregate-ips-suite.py [dossier-resultats]

Lit le bloc ### RESUME de chaque *_experiment.csv et les *_selection.json.
Les chiffres faisant foi sont les champs best_val_*, choisis sur validation.
"""
import csv, json, sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1
            else 'ips-project/results/ips-suite/experiments')
KEY = {'ddr': 'kappa', 'rsna': 'auc'}


def resume(p):
    L = p.read_text().splitlines()
    i = L.index('### RESUME')
    return dict(zip(next(csv.reader([L[i + 1]])), next(csv.reader([L[i + 2]]))))


def main():
    rows = {p.name.replace('_experiment.csv', ''): resume(p)
            for p in sorted(ROOT.glob('*_experiment.csv'))}
    if not rows:
        sys.exit(f'aucun resultat dans {ROOT}')

    for ds in sorted({r['dataset'] for r in rows.values()}):
        k = KEY.get(ds, 'accuracy')
        sub = {n: r for n, r in rows.items() if r['dataset'] == ds}
        print(f"\n{'=' * 104}\n{ds.upper()}  ({len(sub)} runs, metrique : {k}, "
              f"{list(sub.values())[0]['n_epochs']} epoques)\n{'=' * 104}")
        h = (f"{'run':26s} {'modele':>12s} {'echelles':>12s} {'bl':>2s} "
             f"{'N':>5s} {'Mpx':>5s} {k:>7s} {'VRAM':>7s} {'h':>6s} {'CO2 g':>7s}")
        print(h + '\n' + '-' * len(h))
        for n, r in sorted(sub.items(), key=lambda t: -float(t[1][f'best_val_{k}'])):
            print(f"{n:26s} {r['model']:>12s} {r['scales']:>12s} "
                  f"{r['n_res_blocks']:>2s} {r['n_candidates']:>5s} "
                  f"{float(r['pixels_per_pass_M']):>5.2f} "
                  f"{float(r[f'best_val_{k}']):>7.4f} "
                  f"{float(r['train_vram_mb']) / 1024:>6.2f}G "
                  f"{float(r['train_time_s']) / 3600:>6.2f} "
                  f"{float(r['total_co2_kg']) * 1000:>7.1f}")

        pts = sorted(((float(r['train_vram_mb']) / 1024,
                       float(r[f'best_val_{k}']), n) for n, r in sub.items()))
        best, front = -1e9, []
        for v, q, n in pts:
            if q > best:
                front.append((n, v, q)); best = q
        print(f"\n  frontiere memoire / qualite :")
        for n, v, q in front:
            print(f"    {v:5.2f} G  {k} {q:.4f}   {n}")
        print(f"    dominees : {', '.join(sorted(set(sub) - {n for n, _, _ in front}))}")

    print(f"\n\n{'=' * 104}\nTIRAGES DE REFERENCE : ce que vaut le choix des "
          f"echelles, puis celui des positions\n{'=' * 104}")
    h = (f"{'run':26s} {'metr':>6s} {'uniform':>8s} {'matched':>8s} {'top-M':>8s} "
         f"{'echelles':>10s} {'positions':>10s}")
    print(h + '\n' + '-' * len(h))
    for p in sorted(ROOT.glob('*_selection.json')):
        d = json.loads(p.read_text())
        if not {'uniform', 'scale_matched', 'topm'} <= set(d):
            continue
        n = p.name.replace('_selection.json', '')
        k = KEY.get(rows[n]['dataset'], 'accuracy')
        u, m, t = (d[x]['metrics'][k] for x in ('uniform', 'scale_matched', 'topm'))
        print(f"{n:26s} {k:>6s} {u:>8.4f} {m:>8.4f} {t:>8.4f} "
              f"{m - u:>+10.4f} {t - m:>+10.4f}")

    print(f"\n\n{'=' * 104}\nECHELLES EFFECTIVEMENT RETENUES (top-M, du plus "
          f"grossier au plus fin)\n{'=' * 104}")
    for p in sorted(ROOT.glob('*_selection.json')):
        d = json.loads(p.read_text())
        sc = d.get('topm', {}).get('cost', {}).get('scale_counts')
        if not sc:
            continue
        n = p.name.replace('_selection.json', '')
        tot = sum(sc)
        print(f"{n:26s} {rows[n]['scales']:>12s}  "
              + '  '.join(f'{c / tot * 100:5.1f}%' for c in sc))


if __name__ == '__main__':
    main()
