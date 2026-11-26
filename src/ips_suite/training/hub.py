import os
import shutil
import time
from pathlib import Path

from ips_suite.config.environment import SUITE_VERSION, _env, _env_int
from ips_suite.config.experiments import ARTIFACTS, RESULTS, WORK

TOKEN_DATASET = 'ips-suite-hub-token'
KAGGLE_INPUT = Path('/kaggle/input')


def _token_from_kaggle_dataset():
    candidates = [KAGGLE_INPUT / TOKEN_DATASET / 'hf_token',
                  *KAGGLE_INPUT.glob(f'datasets/*/{TOKEN_DATASET}/hf_token')]
    for path in candidates:
        if path.is_file():
            return path.read_text().strip()
    return ''


def _token_from_kaggle_secret():
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret('HF_TOKEN')
    except Exception:
        return ''


def _hf_token():
    token = (os.environ.get('HF_TOKEN', '').strip() or _token_from_kaggle_dataset()
             or _token_from_kaggle_secret())
    if not token:
        raise RuntimeError("IPS_HUB_REPO is set but no Hugging Face token was found "
                           "(HF_TOKEN variable, Kaggle dataset ips-suite-hub-token, "
                           "or Kaggle secret HF_TOKEN)")
    return token


class HubMirror:

    def __init__(self, repo, suite, roots, every, cache_dir):
        self.repo = repo
        self.suite = suite
        self.roots = roots
        self.every = max(1, every)
        self.cache_dir = cache_dir
        self._api = None

    @property
    def enabled(self):
        return bool(self.repo)

    @property
    def api(self):
        if self._api is None:
            from huggingface_hub import HfApi
            self._api = HfApi(token=_hf_token())
            self._api.create_repo(self.repo, repo_type='dataset', private=True,
                                  exist_ok=True)
        return self._api

    def connect(self):
        return self.api

    def remote_path(self, local):
        local = Path(local)
        for part, root in self.roots.items():
            if local.is_relative_to(root):
                return f"{self.suite}/{part}/{local.relative_to(root).as_posix()}"
        raise ValueError(f"{local} is outside {list(self.roots.values())}")

    def is_due(self, epoch, n_epoch):
        return self.enabled and (epoch % self.every == 0 or epoch == n_epoch)

    def pull(self, files):
        if not self.enabled:
            return
        from huggingface_hub import hf_hub_download
        remote = set(self.api.list_repo_files(self.repo, repo_type='dataset'))
        for local in files:
            path = self.remote_path(local)
            if local.exists() or path not in remote:
                continue
            cached = hf_hub_download(self.repo, path, repo_type='dataset',
                                     local_dir=str(self.cache_dir))
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, local)
            print(f"  [hub] fetched {path}")

    def push(self, files, message, attempts=3):
        if not self.enabled:
            return False
        from huggingface_hub import CommitOperationAdd
        operations = [CommitOperationAdd(self.remote_path(f), str(f))
                      for f in files if Path(f).is_file()]
        if not operations:
            return True
        for attempt in range(1, attempts + 1):
            try:
                self.api.create_commit(self.repo, operations, commit_message=message,
                                       repo_type='dataset')
                return True
            except Exception as exc:
                print(f"  [hub] upload failed ({attempt}/{attempts}): {exc!r}")
                time.sleep(30 * attempt)
        return False

    def delete(self, local, message):
        if not self.enabled:
            return
        try:
            self.api.delete_file(self.remote_path(local), self.repo,
                                 repo_type='dataset', commit_message=message)
        except Exception as exc:
            print(f"  [hub] could not delete {local.name}: {exc!r}")


class RunFiles:

    def __init__(self, name, run_name):
        self.name, self.run_name = name, run_name
        self.result = RESULTS / name / 'experiments' / f"{run_name}_result.json"
        self.weights = ARTIFACTS / name / 'checkpoints' / f"{run_name}_weights.pt"
        self.state = ARTIFACTS / name / 'checkpoints' / f"{run_name}_state.pt"

    @property
    def restorable(self):
        return [self.result, self.weights, self.state]

    @property
    def resume_point(self):
        return [self.state, self.weights]

    def final(self):
        light = [f for f in (RESULTS / self.name).rglob('*')
                 if f.is_file() and (self.run_name in f.name or 'codecarbon' in f.parts)]
        heavy = [f for f in (ARTIFACTS / self.name).rglob('*')
                 if f.is_file() and self.run_name in f.name
                 and f not in (self.state,) and f.suffix != '.lock']
        return light + heavy


HUB = HubMirror(repo=_env('IPS_HUB_REPO'), suite=f"ips-suite-{SUITE_VERSION}",
                roots={'results': RESULTS, 'artifacts': ARTIFACTS},
                every=_env_int('IPS_HUB_EVERY', 2), cache_dir=WORK / 'hub-cache')
if HUB.enabled:
    print(f"  hub         : {HUB.repo} (private dataset, sync every {HUB.every} epochs)")
