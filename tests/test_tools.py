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
                if isinstance(op.source, bytes):
                    target.write_bytes(op.source)
                else:
                    shutil.copy2(op.source, target)

        def delete_file(self, path, repo, repo_type, commit_message):
            (store / path).unlink()

    class RepositoryNotFoundError(Exception):
        pass

    def hf_hub_download(repo, path, repo_type, local_dir, force_download=False, token=None):
        # as on Kaggle: the HF_TOKEN variable is not set there, so only a token
        # passed explicitly authenticates; a private repository answers 401
        if token != 'hf_test':
            raise RepositoryNotFoundError('401 Client Error (anonymous)')
        if not (store / path).exists():
            raise FileNotFoundError(path)
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
    return hub, HfApi, work, store


def test_hub_resume_and_final_upload(hub_module):
    hub, HfApi, work, _ = hub_module
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


def test_hub_downloads_use_the_token_and_claims_are_read(hub_module):
    hub, HfApi, work, store = hub_module
    (store / '.gitattributes').parent.mkdir(parents=True, exist_ok=True)
    (store / '.gitattributes').write_text('*.pt filter=lfs')
    hub.HUB.connect()                                 # the download check passes
    files = hub.RunFiles('ddr_mil66', 'ddr_mil66_s42')
    files.state.parent.mkdir(parents=True, exist_ok=True)
    files.state.write_bytes(b'state')
    assert hub.HUB.push(files.resume_point, 'epoch 18')
    files.state.unlink()
    hub.HUB.pull(files.restorable)                    # a new session resumes
    assert files.state.read_bytes() == b'state'
    a = hub.RunClaims(hub.HUB, stale_minutes=120, session='A')
    b = hub.RunClaims(hub.HUB, stale_minutes=120, session='B')
    assert a.claim('ddr_mil66_s42')
    assert not b.claim('ddr_mil66_s42')               # B reads A's claim and skips
    a.release('ddr_mil66_s42')
    assert b.claim('ddr_mil66_s42')


def test_hub_stops_when_downloads_fail(hub_module, monkeypatch):
    hub, _, _, store = hub_module
    (store / '.gitattributes').parent.mkdir(parents=True, exist_ok=True)
    (store / '.gitattributes').write_text('x')
    monkeypatch.setattr(hub.HUB.api, 'token', 'wrong')
    with pytest.raises(RuntimeError, match='download failed'):
        hub.HUB.connect()
    with pytest.raises(Exception, match='401'):      # never read as "no claim"
        hub.RunClaims(hub.HUB, stale_minutes=120, session='A').claim('x_s42')


def test_hub_token_lookup_order(hub_module, tmp_path, monkeypatch):
    hub, _, _, _ = hub_module
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


def test_notebooks_match_the_code():
    sys.path.insert(0, str(REPO / 'tools'))
    build = load('build_notebook', REPO / 'tools' / 'build_notebook.py')
    for path, make in build.NOTEBOOKS.items():
        assert build.code_of(json.loads(path.read_text())) == build.code_of(make()), path.name


def test_analysis_registry_commands():
    run_analysis = load('run_analysis', REPO / 'analysis' / 'run_analysis.py')
    registry = run_analysis.load_registry()
    assert 'aggregate' in registry
    paths = {'python': 'py', 'results': '/r', 'artifacts': '/a', 'out': '/o'}
    for name, spec in registry.items():
        assert set(spec['needs']) <= {'results', 'artifacts'}, name
        assert '{' not in ' '.join(run_analysis.command_for(spec, paths)), name
        assert '{' not in run_analysis.fill(spec['outputs'], paths), name
    assert run_analysis.hub_patterns(['results']) == ['ips-suite-v3/results/**']
    assert run_analysis.command_for(registry['aggregate'], paths)[:3] == \
        ['py', 'aggregate-ips-suite-v3.py', '/r']


def test_launcher_analysis_notebook_and_metadata():
    launch = load('launch', REPO / 'kaggle' / 'launch.py')
    cell = launch.analysis_settings_cell('aggregate', 'me/runs', '2026-11-26_1200_aggregate', 'abc')
    nb = launch.notebook_with_settings(cell, launch.ANALYSIS_NOTEBOOK)
    settings = [''.join(c['source']) for c in nb['cells'] if launch.SETTINGS_MARKER in ''.join(c['source'])]
    assert len(settings) == 1 and "'IPS_ANALYSIS'] = 'aggregate'" in settings[0]
    meta = launch.kernel_metadata('me', launch.ANALYSIS_SLUG, launch.ANALYSIS_NOTEBOOK.name,
                                  gpu=False, datasets=False)
    assert meta['dataset_sources'] == ['me/ips-suite-hub-token'] and meta['enable_gpu'] is False


def test_pull_patterns():
    pull = load('pull_results', REPO / 'hub' / 'pull_results.py')
    assert pull.patterns_for(False, None) == ['ips-suite-v3/results/**']
    assert pull.patterns_for(False, 'all') == ['ips-suite-v3/analysis/**']
    assert pull.patterns_for(True, 'x') == ['ips-suite-v3/analysis/x/**']


def test_cache_is_skipped_when_the_disk_lacks_room(tmp_path):
    fits = function_from(REPO / 'src' / 'ips_suite' / 'data' / 'datasets.py', '_cache_fits')
    fits.__globals__['Path'] = Path
    cache = tmp_path / 'ddr_train.npy'
    assert fits(cache, n_bytes=10e9, free_bytes=20e9)
    assert not fits(cache, n_bytes=30e9, free_bytes=20e9)
    cache.with_suffix('.done').write_text('6260')
    assert fits(cache, n_bytes=30e9, free_bytes=1e9)


def claims_for(hub, session, now):
    return hub.RunClaims(hub.HUB, stale_minutes=120, session=session, clock=lambda: now[0],
                         settle_seconds=0)


def test_a_run_is_trained_by_one_session_only(hub_module):
    hub, _, _, store = hub_module
    now = [1000.0]
    first, second = claims_for(hub, 'kaggle-a', now), claims_for(hub, 'kaggle-b', now)
    assert first.claim('ddr_ips66_s42')
    assert (store / 'ips-suite-v3' / 'claims' / 'ddr_ips66_s42.json').exists()
    assert not second.claim('ddr_ips66_s42')
    now[0] += 60 * 60
    first.refresh('ddr_ips66_s42')
    now[0] += 60 * 60
    assert not second.claim('ddr_ips66_s42')
    first.release('ddr_ips66_s42')
    assert not (store / 'ips-suite-v3' / 'claims' / 'ddr_ips66_s42.json').exists()
    assert second.claim('ddr_ips66_s42')


def test_an_abandoned_claim_is_taken_over(hub_module):
    hub, _, _, _ = hub_module
    now = [1000.0]
    stopped, relaunched = claims_for(hub, 'kaggle-old', now), claims_for(hub, 'kaggle-new', now)
    assert stopped.claim('rsna_base1024_s42')
    now[0] += 3 * 60 * 60
    assert relaunched.claim('rsna_base1024_s42')


def test_simultaneous_claims_leave_one_session(hub_module, monkeypatch):
    hub, _, _, _ = hub_module
    now = [1000.0]
    mine, other = claims_for(hub, 'kaggle-a', now), claims_for(hub, 'kaggle-b', now)
    monkeypatch.setattr(hub.time, 'sleep', lambda seconds: other._write('ddr_ips66_s42', 'race'))
    assert not mine.claim('ddr_ips66_s42')
