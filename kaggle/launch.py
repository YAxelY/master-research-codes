#!/usr/bin/env python3
"""Launch a share of the suite as a Kaggle notebook run ("Save & Run All").

    python kaggle/launch.py --profile axel --runs ddr_ms66_33,ddr_h66_33 \
        --hub <hf-user>/ips-suite-runs --epochs 50
    python kaggle/launch.py --profile axel --status

A profile is a folder ~/.config/ips-suite/kaggle/<profile>/ holding the
kaggle.json of one person's Kaggle account. See docs/RUNNING.md.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / 'ips-suite-v3.ipynb'
PROFILES = Path.home() / '.config' / 'ips-suite' / 'kaggle'
DATASETS = ['samriddhibagchi/ddr-dataset-credits-to-authors',
            'iamtapendu/rsna-pneumonia-processed-dataset']
SETTINGS_MARKER = "ON_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ"


def profile_dir(profile):
    path = PROFILES / profile
    if not (path / 'kaggle.json').is_file():
        sys.exit(f"no Kaggle credentials at {path / 'kaggle.json'}")
    return path


def kaggle_user(profile):
    return json.loads((profile_dir(profile) / 'kaggle.json').read_text())['username']


def code_commit():
    def git(*args):
        return subprocess.run(['git', '-C', str(REPO), *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    commit = git('rev-parse', '--short', 'HEAD')
    dirty = bool(git('status', '--porcelain', '--', NOTEBOOK.name, 'src'))
    return commit, dirty


def notebook_is_current():
    check = subprocess.run([sys.executable, str(REPO / 'tools' / 'build_notebook.py'), '--check'],
                           capture_output=True, text=True)
    return check.returncode == 0


def kernel_slug(runs, seeds, smoke):
    digest = hashlib.sha1(f"{runs}|{seeds}|{smoke}".encode()).hexdigest()[:8]
    return f"ips-suite-v3-{digest}"


def settings_cell(args, commit):
    values = {
        'IPS_MODE': 'train', 'IPS_RUNS': args.runs, 'IPS_SEEDS': args.seeds,
        'IPS_EPOCHS': str(args.epochs), 'IPS_SMOKE': str(args.smoke), 'IPS_WORKERS': '3',
        'IPS_CACHE_DIR': '/kaggle/working/cache', 'IPS_DDR_GEOMETRY': 'retina',
        'IPS_HUB_REPO': args.hub, 'IPS_HUB_EVERY': str(args.hub_every),
        'IPS_CODE_COMMIT': commit,
    }
    lines = ['import os', '', SETTINGS_MARKER, 'if ON_KAGGLE:']
    lines += [f"    os.environ[{k!r}] = {v!r}" for k, v in values.items()]
    lines.append("print({k: v for k, v in os.environ.items() if k.startswith('IPS_')})")
    return '\n'.join(lines) + '\n'


def notebook_with_settings(cell_source):
    nb = json.loads(NOTEBOOK.read_text())
    targets = [c for c in nb['cells'] if SETTINGS_MARKER in ''.join(c['source'])]
    if len(targets) != 1:
        sys.exit(f"expected one settings cell in {NOTEBOOK.name}, found {len(targets)}")
    targets[0]['source'] = cell_source.splitlines(keepends=True)
    return nb


def kernel_metadata(user, slug):
    return {
        'id': f"{user}/{slug}", 'title': slug, 'code_file': NOTEBOOK.name,
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': True, 'enable_internet': True,
        'dataset_sources': DATASETS, 'competition_sources': [], 'kernel_sources': [],
    }


def kaggle(profile, *args):
    env = {**os.environ, 'KAGGLE_CONFIG_DIR': str(profile_dir(profile))}
    return subprocess.run(['kaggle', *args], env=env, text=True)


def launch(args):
    if not notebook_is_current():
        sys.exit("the notebook does not match src/ips_suite: run python tools/build_notebook.py")
    commit, dirty = code_commit()
    if dirty and not args.allow_dirty:
        sys.exit("src/ or the notebook has uncommitted changes: commit them, so that "
                 "the results can name the exact code (or pass --allow-dirty)")
    commit += '-dirty' if dirty else ''
    user, slug = kaggle_user(args.profile), kernel_slug(args.runs, args.seeds, args.smoke)
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        nb = notebook_with_settings(settings_cell(args, commit))
        (folder / NOTEBOOK.name).write_text(json.dumps(nb, indent=1))
        (folder / 'kernel-metadata.json').write_text(json.dumps(kernel_metadata(user, slug), indent=1))
        print(f"{user}/{slug}: runs {args.runs}, seeds {args.seeds}, "
              f"{args.epochs} epochs, code {commit}, hub {args.hub}"
              + (f", code check on {args.smoke} images per split" if args.smoke else ""))
        if args.dry_run:
            print((folder / 'kernel-metadata.json').read_text())
            return 0
        push = ['kernels', 'push', '-p', str(folder)]
        if args.accelerator:
            push += ['--accelerator', args.accelerator]
        return kaggle(args.profile, *push).returncode


def status(args):
    return kaggle(args.profile, 'kernels', 'list', '--mine', '--search', 'ips-suite-v3').returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', required=True)
    ap.add_argument('--runs', help="run or group names, comma separated")
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--smoke', type=int, default=0, help="images per split for a code check, 0 = all")
    ap.add_argument('--hub', help="Hugging Face dataset repo, <user>/<name>")
    ap.add_argument('--hub-every', type=int, default=2)
    ap.add_argument('--accelerator', help="passed to kaggle kernels push")
    ap.add_argument('--allow-dirty', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()
    if args.status:
        return status(args)
    if not args.runs or not args.hub:
        ap.error("--runs and --hub are required to launch")
    return launch(args)


if __name__ == '__main__':
    sys.exit(main())
