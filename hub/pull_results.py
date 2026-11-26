#!/usr/bin/env python3
"""Bring the results stored on the Hugging Face Hub back to this machine.

    python hub/pull_results.py --hub <hf-user>/ips-suite-runs
    python hub/pull_results.py --hub <hf-user>/ips-suite-runs --artifacts
    python hub/pull_results.py --hub <hf-user>/ips-suite-runs --analysis all
    python hub/pull_results.py --hub <hf-user>/ips-suite-runs --squash

Results land in hub-results/ips-suite-v3/results/<configuration>/, the layout
that aggregate-ips-suite-v3.py reads. --analysis brings only the outputs of
the analyses run on Kaggle (one tag, or all), never the weights. --squash drops the upload history of
the repository (old resume states), to be used when no run is in progress.
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from envfile import read_env_file  # noqa: E402

SUITE = 'ips-suite-v3'


def patterns_for(with_artifacts, analysis):
    if analysis:
        return [f"{SUITE}/analysis/**" if analysis == 'all' else f"{SUITE}/analysis/{analysis}/**"]
    return [f"{SUITE}/results/**"] + ([f"{SUITE}/artifacts/**"] if with_artifacts else [])


def pull_analysis(repo, destination, analysis):
    from huggingface_hub import snapshot_download
    snapshot_download(repo, repo_type='dataset', allow_patterns=patterns_for(False, analysis),
                      local_dir=str(destination))
    folder = destination / SUITE / 'analysis'
    for tag in sorted(p.name for p in folder.iterdir()) if folder.is_dir() else []:
        print(f"analysis {tag}: {folder / tag}")


def pull(repo, destination, with_artifacts):
    from huggingface_hub import snapshot_download
    patterns = patterns_for(with_artifacts, None)
    snapshot_download(repo, repo_type='dataset', allow_patterns=patterns,
                      local_dir=str(destination))
    results = destination / SUITE / 'results'
    n = len(list(results.glob('*/experiments/*_result.json')))
    print(f"{n} finished runs in {results}")
    print(f"next: python aggregate-ips-suite-v3.py {results}")


def squash(repo):
    from huggingface_hub import HfApi
    HfApi().super_squash_history(repo, repo_type='dataset')
    print(f"history of {repo} squashed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--hub', required=True)
    ap.add_argument('--to', type=Path, default=Path('hub-results'))
    ap.add_argument('--artifacts', action='store_true', help="also weights and Grad-CAM maps")
    ap.add_argument('--analysis', help="tag of an analysis output, or 'all'")
    ap.add_argument('--squash', action='store_true')
    ap.add_argument('--env-file', help=".env with HF_TOKEN")
    args = ap.parse_args()
    if args.env_file:
        os.environ.setdefault('HF_TOKEN', read_env_file(args.env_file).get('HF_TOKEN', ''))
    if args.squash:
        squash(args.hub)
    elif args.analysis:
        pull_analysis(args.hub, args.to, args.analysis)
    else:
        pull(args.hub, args.to, args.artifacts)
    return 0


if __name__ == '__main__':
    sys.exit(main())
