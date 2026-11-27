#!/usr/bin/env python3
"""Launch the suite on one or several Kaggle accounts ("Save & Run All").

    python kaggle/launch.py --env-file .env --runs ddr_ips66,rsna_ips300 --epochs 1 --smoke 200
    python kaggle/launch.py --accounts accounts --runs focus --epochs 50
    python kaggle/launch.py --accounts accounts --status
    python kaggle/launch.py --env-file .env --analysis aggregate

An account is a .env file with KAGGLE_USERNAME, KAGGLE_API_TOKEN and HF_TOKEN
(and optionally IPS_HUB_REPO, the default for --hub). The Hugging Face token
reaches the run through a private Kaggle dataset of that account,
ips-suite-hub-token, created by this script when missing (--refresh-token to
replace it). With several accounts, every
account gets the same list of runs and a different share of it (IPS_SHARD =
i/n, applied by the package), so no run is launched twice. Each account owns
one Kaggle notebook (--slug), relaunched as a new version. --analysis runs one
analysis of analysis/registry.json in a second notebook, next to the results
on the Hub, and uploads only its outputs. See docs/RUNNING.md.
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
ANALYSIS_NOTEBOOK = REPO / 'ips-suite-v3-analysis.ipynb'
ANALYSES = REPO / 'analysis' / 'registry.json'
TRAINING_SLUG, ANALYSIS_SLUG = 'ips-suite-v3-runner', 'ips-suite-v3-analysis'
PLAN = REPO / 'kaggle' / 'last-plan.json'
DATASETS = ['samriddhibagchi/ddr-dataset-credits-to-authors',
            'iamtapendu/rsna-pneumonia-processed-dataset']
SETTINGS_MARKER = "ON_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ"
TOKEN_DATASET = 'ips-suite-hub-token'


class Account:

    def __init__(self, env_file):
        values = read_env_file(env_file)
        missing = [k for k in ('KAGGLE_USERNAME', 'KAGGLE_API_TOKEN') if not values.get(k)]
        if missing:
            sys.exit(f"{env_file} lacks {', '.join(missing)}")
        self.source = Path(env_file)
        self.username = values['KAGGLE_USERNAME']
        self.hub_repo = values.get('IPS_HUB_REPO', '')
        self.hf_token = values.get('HF_TOKEN', '')
        self._env = {'KAGGLE_API_TOKEN': values['KAGGLE_API_TOKEN']}

    def _environment(self):
        inherited = {k: v for k, v in os.environ.items()
                     if k not in ('KAGGLE_API_TOKEN', 'KAGGLE_CONFIG_DIR')}
        return {**inherited, **self._env}

    def kaggle(self, *args):
        return subprocess.run(['kaggle', *args], env=self._environment(), text=True)

    def kaggle_quiet(self, *args):
        return subprocess.run(['kaggle', *args], env=self._environment(), text=True,
                              capture_output=True)


def token_dataset_is_ready(account):
    check = account.kaggle_quiet('datasets', 'status', f"{account.username}/{TOKEN_DATASET}")
    return check.returncode == 0 and 'ready' in check.stdout


def publish_token_dataset(account, replace):
    if not account.hf_token:
        sys.exit(f"{account.source} lacks HF_TOKEN, needed by the run to reach the Hub")
    exists = token_dataset_is_ready(account)
    if exists and not replace:
        return
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        token_file = folder / 'hf_token'
        token_file.write_text(account.hf_token)
        token_file.chmod(0o600)
        (folder / 'dataset-metadata.json').write_text(json.dumps({
            'title': TOKEN_DATASET, 'id': f"{account.username}/{TOKEN_DATASET}",
            'licenses': [{'name': 'other'}]}))
        if exists:
            account.kaggle('datasets', 'version', '-p', str(folder), '-m', 'token replaced')
        else:
            account.kaggle('datasets', 'create', '-p', str(folder))
    for _ in range(30):
        if token_dataset_is_ready(account):
            return
        time.sleep(10)
    sys.exit(f"{account.username}/{TOKEN_DATASET} is not ready on Kaggle yet: launch again in a minute")


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
    dirty = bool(git('status', '--porcelain', '--', NOTEBOOK.name, ANALYSIS_NOTEBOOK.name,
                     'src', 'analysis', 'aggregate-ips-suite-v3.py'))
    return commit, dirty


def notebook_is_current():
    check = subprocess.run([sys.executable, str(REPO / 'tools' / 'build_notebook.py'), '--check'],
                           capture_output=True, text=True)
    return check.returncode == 0


def settings_source(values):
    lines = ['import os', '', SETTINGS_MARKER, 'if ON_KAGGLE:']
    lines += [f"    os.environ[{k!r}] = {v!r}" for k, v in values.items()]
    lines.append("print({k: v for k, v in os.environ.items() if k.startswith('IPS_')})")
    return '\n'.join(lines) + '\n'


def analysis_settings_cell(name, hub, tag, commit):
    return settings_source({'IPS_ANALYSIS': name, 'IPS_HUB_REPO': hub,
                            'IPS_ANALYSIS_TAG': tag, 'IPS_CODE_COMMIT': commit})


def settings_cell(args, hub, shard, commit):
    values = {
        'IPS_MODE': 'train', 'IPS_RUNS': args.runs, 'IPS_SHARD': shard,
        'IPS_SEEDS': args.seeds, 'IPS_EPOCHS': str(args.epochs), 'IPS_SMOKE': str(args.smoke),
        'IPS_WORKERS': '3', 'IPS_CACHE_DIR': '/tmp/ips-cache',
        'IPS_DDR_GEOMETRY': 'retina', 'IPS_HUB_REPO': hub,
        'IPS_HUB_EVERY': str(args.hub_every), 'IPS_CODE_COMMIT': commit,
    }
    return settings_source(values)


def notebook_with_settings(cell_source, path=NOTEBOOK):
    nb = json.loads(path.read_text())
    targets = [c for c in nb['cells'] if SETTINGS_MARKER in ''.join(c['source'])]
    if len(targets) != 1:
        sys.exit(f"expected one settings cell in {path.name}, found {len(targets)}")
    targets[0]['source'] = cell_source.splitlines(keepends=True)
    return nb


def kernel_metadata(user, slug, code_file=NOTEBOOK.name, gpu=True, datasets=True):
    return {
        'id': f"{user}/{slug}", 'title': slug, 'code_file': code_file,
        'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
        'enable_gpu': gpu, 'enable_internet': True,
        'dataset_sources': (DATASETS if datasets else []) + [f"{user}/{TOKEN_DATASET}"],
        'competition_sources': [], 'kernel_sources': [],
    }


def push(account, notebook, metadata, accelerator=None):
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        (folder / metadata['code_file']).write_text(json.dumps(notebook, indent=1))
        (folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=1))
        command = ['kernels', 'push', '-p', str(folder)]
        if accelerator:
            command += ['--accelerator', accelerator]
        return account.kaggle(*command).returncode


def checked_commit(args):
    if not notebook_is_current():
        sys.exit("a notebook does not match src/ or analysis/: run python tools/build_notebook.py")
    commit, dirty = code_commit()
    if dirty and not args.allow_dirty:
        sys.exit("the code has uncommitted changes: commit them, so that the results "
                 "can name the exact code (or pass --allow-dirty)")
    return commit + ('-dirty' if dirty else '')


def hub_repo(args, accounts):
    hub = args.hub or accounts[0].hub_repo
    if not hub:
        sys.exit("no Hub repository: pass --hub or set IPS_HUB_REPO in the .env file")
    return hub


def launch_analysis(args):
    analyses = json.loads(ANALYSES.read_text())
    if args.analysis not in analyses:
        sys.exit(f"unknown analysis {args.analysis}; known: {', '.join(analyses)}")
    spec = analyses[args.analysis]
    commit = checked_commit(args)
    account = load_accounts(args)[0]
    hub = hub_repo(args, [account])
    tag = time.strftime('%Y-%m-%d_%H%M') + f"_{args.analysis}"
    notebook = notebook_with_settings(analysis_settings_cell(args.analysis, hub, tag, commit),
                                      ANALYSIS_NOTEBOOK)
    metadata = kernel_metadata(account.username, ANALYSIS_SLUG, ANALYSIS_NOTEBOOK.name,
                               gpu=args.gpu or spec['gpu'], datasets=spec['datasets'])
    print(f"{account.username}/{ANALYSIS_SLUG}: analysis {args.analysis} ({spec['description']}), "
          f"outputs to {hub}: ips-suite-v3/analysis/{tag}/, code {commit}")
    if args.dry_run:
        return 0
    publish_token_dataset(account, replace=args.refresh_token)
    return push(account, notebook, metadata)


def launch(args):
    commit = checked_commit(args)
    accounts = load_accounts(args)
    hub = hub_repo(args, accounts)
    plan = {'date': time.strftime('%Y-%m-%d %H:%M'), 'commit': commit, 'hub': hub,
            'runs': args.runs, 'seeds': args.seeds, 'epochs': args.epochs,
            'smoke': args.smoke, 'slug': args.slug, 'shares': {}}
    failures = 0
    for index, account in enumerate(accounts):
        shard = f"{index}/{len(accounts)}" if len(accounts) > 1 else ''
        plan['shares'][account.username] = shard or 'all'
        if args.only and account.username not in args.only.split(','):
            continue
        print(f"{account.username}/{args.slug}: runs {args.runs}"
              + (f" (share {shard})" if shard else '')
              + f", seeds {args.seeds}, {args.epochs} epochs"
              + (f", code check on {args.smoke} images per split" if args.smoke else '')
              + f", code {commit}, hub {hub}")
        if not args.dry_run:
            publish_token_dataset(account, replace=args.refresh_token)
            notebook = notebook_with_settings(settings_cell(args, hub, shard, commit))
            metadata = kernel_metadata(account.username, args.slug)
            failures += push(account, notebook, metadata, args.accelerator) != 0
    if not args.dry_run:
        PLAN.write_text(json.dumps(plan, indent=1))
        print(f"plan written to {PLAN.relative_to(REPO)}")
    return 1 if failures else 0


def status(args):
    slug = ANALYSIS_SLUG if args.analysis else args.slug
    codes = [account.kaggle('kernels', 'status', f"{account.username}/{slug}").returncode
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
    ap.add_argument('--analysis', help="name of an analysis in analysis/registry.json")
    ap.add_argument('--gpu', action='store_true', help="give the analysis a GPU")
    ap.add_argument('--slug', default=TRAINING_SLUG,
                    help="Kaggle notebook name; one per account keeps its attached secrets")
    ap.add_argument('--accelerator', help="passed to kaggle kernels push")
    ap.add_argument('--refresh-token', action='store_true',
                    help="replace the Hugging Face token stored on Kaggle")
    ap.add_argument('--only', help="with --accounts: relaunch only these usernames, "
                                   "keeping the shares of the full account list")
    ap.add_argument('--allow-dirty', action='store_true')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--status', action='store_true')
    args = ap.parse_args()
    if args.status:
        return status(args)
    if args.analysis:
        return launch_analysis(args)
    if not args.runs:
        ap.error("--runs is required to launch")
    return launch(args)


if __name__ == '__main__':
    sys.exit(main())
