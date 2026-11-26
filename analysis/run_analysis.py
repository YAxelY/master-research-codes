#!/usr/bin/env python3
"""Run one analysis of the registry next to the results it needs (on Kaggle).

Reads IPS_ANALYSIS (a name of analysis/registry.json), IPS_HUB_REPO and
IPS_ANALYSIS_TAG; downloads from the Hub only what the analysis needs, runs
its command, and uploads its outputs to <suite>/analysis/<tag>/ on the Hub.
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = 'ips-suite-v3'


def load_registry():
    return json.loads((HERE / 'registry.json').read_text())


def hub_patterns(needs):
    return [f"{SUITE}/{part}/**" for part in needs]


def fill(template, paths):
    return template.format(**paths)


def command_for(spec, paths):
    return [fill(part, paths) for part in spec['command']]


def run(name, hub_repo, tag, work):
    spec = load_registry()[name]
    from huggingface_hub import snapshot_download
    from ips_suite.training.hub import HubMirror, _hf_token
    data = work / 'hub-data'
    snapshot_download(hub_repo, repo_type='dataset', allow_patterns=hub_patterns(spec['needs']),
                      local_dir=str(data), token=_hf_token())
    paths = {'python': sys.executable, 'results': str(data / SUITE / 'results'),
             'artifacts': str(data / SUITE / 'artifacts'), 'out': str(work / 'analysis-out')}
    Path(paths['out']).mkdir(parents=True, exist_ok=True)
    subprocess.run(command_for(spec, paths), check=True, cwd=str(HERE.parent))
    produced = Path(fill(spec['outputs'], paths))
    published = work / 'analysis' / tag
    shutil.copytree(produced, published, dirs_exist_ok=True)
    mirror = HubMirror(hub_repo, SUITE, {'analysis': work / 'analysis'}, every=1,
                       cache_dir=work / 'hub-cache')
    files = [f for f in published.rglob('*') if f.is_file()]
    if not mirror.push(files, f"analysis {tag}"):
        sys.exit("the analysis ran but its outputs could not be uploaded to the Hub")
    print(f"{len(files)} files uploaded to {hub_repo}: {SUITE}/analysis/{tag}/")


def main():
    name = os.environ['IPS_ANALYSIS']
    run(name, os.environ['IPS_HUB_REPO'], os.environ['IPS_ANALYSIS_TAG'],
        Path(os.environ.get('IPS_WORK', '/kaggle/working')))


if __name__ == '__main__':
    main()
