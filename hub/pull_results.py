#!/usr/bin/env python3
"""Bring the results stored on the Hugging Face Hub back to this machine.

    python hub/pull_results.py --hub <hf-user>/ips-suite-runs
    python hub/pull_results.py --hub <hf-user>/ips-suite-runs --artifacts
    python hub/pull_results.py --hub <hf-user>/ips-suite-runs --squash

Results land in hub-results/ips-suite-v3/results/<configuration>/, the layout
that aggregate-ips-suite-v3.py reads. --squash drops the upload history of
the repository (old resume states), to be used when no run is in progress.
"""
import argparse
import sys
from pathlib import Path

SUITE = 'ips-suite-v3'


def pull(repo, destination, with_artifacts):
    from huggingface_hub import snapshot_download
    patterns = [f"{SUITE}/results/**"] + ([f"{SUITE}/artifacts/**"] if with_artifacts else [])
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
    ap.add_argument('--squash', action='store_true')
    args = ap.parse_args()
    if args.squash:
        squash(args.hub)
    else:
        pull(args.hub, args.to, args.artifacts)
    return 0


if __name__ == '__main__':
    sys.exit(main())
