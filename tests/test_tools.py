"""Tests of the launcher, the run sharing and the Hub mirror, without Kaggle or the Hub.

    python -m pytest tests
"""
import ast
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_from(path, name):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {}
    exec(compile(ast.Module([node], []), str(path), 'exec'), namespace)
    return namespace[name]


def test_shares_are_disjoint_and_complete():
    share = function_from(REPO / 'src' / 'ips_suite' / '__main__.py', '_share')
    runs = [f"run{i}" for i in range(14)]
    parts = [share(runs, f"{i}/3") for i in range(3)]
    assert sorted(sum(parts, [])) == sorted(runs)
    assert all(not set(a) & set(b) for a in parts for b in parts if a is not b)
    assert share(runs, '') == runs
    with pytest.raises(ValueError):
        share(runs, '3/3')


def test_launcher_writes_settings_and_metadata(tmp_path):
    launch = load('launch', REPO / 'kaggle' / 'launch.py')
    env = tmp_path / 'a.env'
    env.write_text('KAGGLE_USERNAME=someone\nKAGGLE_API_TOKEN=KGAT_x\nHF_TOKEN=hf_x\n')
    account = launch.Account(env)
    args = types.SimpleNamespace(runs='ddr_ips66', seeds='42', epochs=1, smoke=200, hub_every=2)
    cell = launch.settings_cell(args, 'me/runs', '1/2', 'abc1234')
    assert "os.environ['IPS_SHARD'] = '1/2'" in cell
    assert "os.environ['IPS_HUB_REPO'] = 'me/runs'" in cell
    assert 'hf_x' not in cell and 'KGAT_x' not in cell
    nb = launch.notebook_with_settings(cell)
    assert sum(launch.SETTINGS_MARKER in ''.join(c['source']) for c in nb['cells']) == 1
    meta = launch.kernel_metadata(account.username, 'runner')
    assert meta['is_private'] and 'someone/ips-suite-hub-token' in meta['dataset_sources']


def test_account_requires_kaggle_credentials(tmp_path):
    launch = load('launch', REPO / 'kaggle' / 'launch.py')
    env = tmp_path / 'b.env'
    env.write_text('HF_TOKEN=hf_x\n')
    with pytest.raises(SystemExit):
        launch.Account(env)


@pytest.fixture
def hub_module(tmp_path, monkeypatch):
    store = tmp_path / 'hub'
    stub = types.ModuleType('huggingface_hub')

    class CommitOperationAdd:
        def __init__(self, path_in_repo, path_or_fileobj):
            self.path_in_repo, self.source = path_in_repo, path_or_fileobj

    class HfApi:
        failures = 0

        def __init__(self, token):
            self.token = token

        def create_repo(self, *args, **kwargs):
            pass

        def list_repo_files(self, repo, repo_type):
            return [p.relative_to(store).as_posix() for p in store.rglob('*') if p.is_file()]

        def create_commit(self, repo, operations, commit_message, repo_type):
            if HfApi.failures:
                HfApi.failures -= 1
                raise IOError('network down')
            for op in operations:
                target = store / op.path_in_repo
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(op.source, target)

        def delete_file(self, path, repo, repo_type, commit_message):
            (store / path).unlink()

    def hf_hub_download(repo, path, repo_type, local_dir):
        target = Path(local_dir) / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(store / path, target)
        return str(target)

    stub.HfApi, stub.CommitOperationAdd, stub.hf_hub_download = HfApi, CommitOperationAdd, hf_hub_download
    work = tmp_path / 'work'
    modules = {
        'huggingface_hub': stub,
        'ips_suite': types.ModuleType('ips_suite'),
        'ips_suite.config': types.ModuleType('ips_suite.config'),
        'ips_suite.config.environment': types.SimpleNamespace(
            SUITE_VERSION='v3', _env=lambda k, d='': {'IPS_HUB_REPO': 'me/runs'}.get(k, d),
            _env_int=lambda k, d: 2),
        'ips_suite.config.experiments': types.SimpleNamespace(
            ARTIFACTS=work / 'artifacts', RESULTS=work / 'results', WORK=work),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv('HF_TOKEN', 'hf_test')
    hub = load('hub_under_test', REPO / 'src' / 'ips_suite' / 'training' / 'hub.py')
    monkeypatch.setattr(hub.time, 'sleep', lambda seconds: None)
    return hub, HfApi, work


def test_hub_resume_and_final_upload(hub_module):
    hub, HfApi, work = hub_module
    files = hub.RunFiles('ddr_ips66', 'ddr_ips66_s42')
    assert hub.HUB.remote_path(files.state) == \
        'ips-suite-v3/artifacts/ddr_ips66/checkpoints/ddr_ips66_s42_state.pt'
    assert [hub.HUB.is_due(e, 5) for e in range(1, 6)] == [False, True, False, True, True]
    for path in files.resume_point:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(path.name)
    HfApi.failures = 1
    assert hub.HUB.push(files.resume_point, 'epoch 2')
    for path in files.resume_point:
        path.unlink()
    hub.HUB.pull(files.restorable)
    assert files.state.read_text() == 'ddr_ips66_s42_state.pt'
    files.result.parent.mkdir(parents=True, exist_ok=True)
    files.result.write_text('{}')
    other_seed = work / 'results' / 'ddr_ips66' / 'experiments' / 'ddr_ips66_s7_result.json'
    other_seed.write_text('{}')
    assert sorted(p.name for p in files.final()) == ['ddr_ips66_s42_result.json',
                                                      'ddr_ips66_s42_weights.pt']
    assert hub.HUB.push(files.final(), 'final')
    hub.HUB.delete(files.state, 'drop')
    HfApi.failures = 5
    assert hub.HUB.push([files.result], 'never raises') is False


def test_hub_token_lookup_order(hub_module, tmp_path, monkeypatch):
    hub, _, _ = hub_module
    monkeypatch.setattr(hub, 'KAGGLE_INPUT', tmp_path / 'input')
    monkeypatch.delenv('HF_TOKEN')
    with pytest.raises(RuntimeError):
        hub._hf_token()
    folder = tmp_path / 'input' / 'datasets' / 'someone' / 'ips-suite-hub-token'
    folder.mkdir(parents=True)
    (folder / 'hf_token').write_text('hf_dataset\n')
    assert hub._hf_token() == 'hf_dataset'
    monkeypatch.setenv('HF_TOKEN', 'hf_env')
    assert hub._hf_token() == 'hf_env'


def test_package_has_no_undefined_names():
    check = load('check_names', REPO / 'tools' / 'check_names.py')
    problems = [(p, v) for p in (REPO / 'src').rglob('*.py') if (v := check.check(p))]
    assert problems == []


def test_notebook_matches_package():
    sys.path.insert(0, str(REPO / 'tools'))
    build = load('build_notebook', REPO / 'tools' / 'build_notebook.py')
    current = json.loads((REPO / 'ips-suite-v3.ipynb').read_text())
    assert build.code_of(current) == build.code_of(build.notebook())
