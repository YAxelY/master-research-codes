#!/usr/bin/env python3
"""Launch the suite on one or several Kaggle accounts ("Save & Run All").

    python kaggle/launch.py --env-file .env --runs ddr_ips66,rsna_ips300 --epochs 1 --smoke 200
    python kaggle/launch.py --accounts accounts --runs focus --epochs 50
    python kaggle/launch.py --accounts accounts --status

An account is a .env file with KAGGLE_USERNAME and KAGGLE_API_TOKEN (and
optionally IPS_HUB_REPO, the default for --hub). With several accounts, every
account gets the same list of runs and a different share of it (IPS_SHARD =
i/n, applied by the package), so no run is launched twice. Each account owns
one Kaggle notebook (--slug), relaunched as a new version. See docs/RUNNING.md.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
from envfile import read_env_file  # noqa: E402

NOTEBOOK = REPO / 'ips-suite-v3.ipynb'
PLAN = REPO / 'kaggle' / 'last-plan.json'
DATASETS = ['samriddhibagchi/ddr-dataset-credits-to-authors',
            'iamtapendu/rsna-pneumonia-processed-dataset']
SETTINGS_MARKER = "ON_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ"


class Account:

    def __init__(self, env_file):
        values = read_env_file(env_file)
        missing = [k for k in ('KAGGLE_USERNAME', 'KAGGLE_API_TOKEN') if not values.get(k)]
        if missing:
            sys.exit(f"{env_file} lacks {', '.join(missing)}")
        self.source = Path(env_file)
        self.username = values['KAGGLE_USERNAME']
        self.hub_repo = values.get('IPS_HUB_REPO', '')
        self._env = {'KAGGLE_API_TOKEN': values['KAGGLE_API_TOKEN']}

    def kaggle(self, *args):
        inherited = {k: v for k, v in os.environ.items()
                     if k not in ('KAGGLE_API_TOKEN', 'KAGGLE_CONFIG_DIR')}
        return subprocess.run(['kaggle', *args], env={**inherited, **self._env}, text=True)


def load_accounts(args):
    if args.env_file:
        return [Account(args.env_file)]
    files = sorted(Path(args.accounts).glob('*.env'))
    if not files:
        sys.exit(f"no *.env account file in {args.accounts}")
    return [Account(f) for f in files]


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


def settings_cell(args, hub, shard, commit):
    values = {
        'IPS_MODE': 'train', 'IPS_RUNS': args.runs, 'IPS_SHARD': shard,
        'IPS_SEEDS': args.seeds, 'IPS_EPOCHS': str(args.epochs), 'IPS_SMOKE': str(args.smoke),
        'IPS_WORKERS': '3', 'IPS_CACHE_DIR': '/kaggle/working/cache',
        'IPS_DDR_GEOMETRY': 'retina', 'IPS_HUB_REPO': hub,
        'IPS_HUB_EVERY': str(args.hub_every), 'IPS_CODE_COMMIT': commit,
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


def push(account, args, hub, shard, commit):
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        nb = notebook_with_settings(settings_cell(args, hub, shard, commit))
        (folder / NOTEBOOK.name).write_text(json.dumps(nb, indent=1))
        (folder / 'kernel-metadata.json').write_text(
            json.dumps(kernel_metadata(account.username, args.slug), indent=1))
        command = ['kernels', 'push', '-p', str(folder)]
        if args.accelerator:
            command += ['--accelerator', args.accelerator]
        return account.kaggle(*command).returncode


def launch(args):
    if not notebook_is_current():
        sys.exit("the notebook does not match src/ips_suite: run python tools/build_notebook.py")
    commit, dirty = code_commit()
    if dirty and not args.allow_dirty:
        sys.exit("src/ or the notebook has uncommitted changes: commit them, so that "
                 "the results can name the exact code (or pass --allow-dirty)")
    commit += '-dirty' if dirty else ''
    accounts = load_accounts(args)
    hub = args.hub or accounts[0].hub_repo
    if not hub:
        sys.exit("no Hub repository: pass --hub or set IPS_HUB_REPO in the .env file")
    plan = {'date': time.strftime('%Y-%m-%d %H:%M'), 'commit': commit, 'hub': hub,
            'runs': args.runs, 'seeds': args.seeds, 'epochs': args.epochs,
            'smoke': args.smoke, 'slug': args.slug, 'shares': {}}
    failures = 0
    for index, account in enumerate(accounts):
        shard = f"{index}/{len(accounts)}" if len(accounts) > 1 else ''
        plan['shares'][account.username] = shard or 'all'
        print(f"{account.username}/{args.slug}: runs {args.runs}"
              + (f" (share {shard})" if shard else '')
              + f", seeds {args.seeds}, {args.epochs} epochs"
              + (f", code check on {args.smoke} images per split" if args.smoke else '')
              + f", code {commit}, hub {hub}")
        if not args.dry_run:
            failures += push(account, args, hub, shard, commit) != 0
    if not args.dry_run:
        PLAN.write_text(json.dumps(plan, indent=1))
        print(f"plan written to {PLAN.relative_to(REPO)}")
    return 1 if failures else 0


def status(args):
    codes = [account.kaggle('kernels', 'status', f"{account.username}/{args.slug}").returncode
             for account in load_accounts(args)]
    return max(codes)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument('--env-file', help=".env of one account")
    who.add_argument('--accounts', help="folder of <name>.env files, one per account")
    ap.add_argument('--runs', help="run or group names, comma separated")
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--smoke', type=int, default=0, help="images per split for a code check, 0 = all")
    ap.add_argument('--hub', help="Hugging Face dataset repo, <user>/<name>")
    ap.add_argument('--hub-every', type=int, default=2)
    ap.add_argument('--slug', default='ips-suite-v3-runner',
                    help="Kaggle notebook name; one per account keeps its attached secrets")
    ap.add_argument('--accelerator', help="passed to kaggle kernels push")
    ap.add_argument('--allow-dirty', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()
    if args.status:
        return status(args)
    if not args.runs:
        ap.error("--runs is required to launch")
    return launch(args)


if __name__ == '__main__':
    sys.exit(main())
