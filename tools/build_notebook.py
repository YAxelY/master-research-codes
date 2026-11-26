#!/usr/bin/env python3
"""Write the two Kaggle notebooks from the package in src/ips_suite/.

    python tools/build_notebook.py           # rewrite the notebooks
    python tools/build_notebook.py --check   # exit 1 if a notebook is out of date

ips-suite-v3.ipynb trains: one %%writefile cell per module of the package and
a cell that runs it. ips-suite-v3-analysis.ipynb runs one analysis of
analysis/registry.json on results stored on the Hugging Face Hub. The package
and the analysis scripts are the only source of the code.
"""
import argparse
import json
import sys
from pathlib import Path

from check_names import check as undefined_names

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / 'src' / 'ips_suite'
NOTEBOOK = REPO / 'ips-suite-v3.ipynb'
ANALYSIS_NOTEBOOK = REPO / 'ips-suite-v3-analysis.ipynb'
ANALYSIS_FILES = ['aggregate-ips-suite-v3.py', 'analysis/registry.json', 'analysis/run_analysis.py']
MODULE_ORDER = [
    '__init__',
    'config/__init__', 'config/environment', 'config/experiments',
    'data/__init__', 'data/tiling', 'data/datasets',
    'models/__init__', 'models/encoder', 'models/attention', 'models/selector',
    'models/cascade', 'models/reference', 'models/batching', 'models/factory',
    'training/__init__', 'training/schedule', 'training/profiler', 'training/epoch',
    'training/hub', 'training/runner',
    'evaluation/__init__', 'evaluation/metrics', 'evaluation/splits',
    'evaluation/selection', 'evaluation/localisation', 'evaluation/benchmark',
    '__main__',
]

INTRO = """# IPS suite v3

Patch selection for high-resolution medical image classification: IPS,
MS-IPS and H-MS-IPS against the full image and attention MIL, on DDR and RSNA.

This notebook is **generated** from the package `src/ips_suite/` by
`tools/build_notebook.py`. Edit the package, not the cells below: the
settings cell is the only one meant to change here. The architecture and the
design decisions are in `docs/ARCHITECTURE.md` and `docs/DECISIONS.md`."""

SETTINGS_DOC = """## Settings: the only cell to edit on Kaggle

| To do this | Change this |
|---|---|
| run another configuration | `IPS_RUNS` (a name, several names, or a group) |
| a real epoch on all images | `IPS_SMOKE` to `'0'` |
| a full training | `IPS_EPOCHS` to `'50'` and `IPS_SMOKE` to `'0'` |
| memory and speed test, no data | `IPS_MODE` to `'bench'` |
| print the list of all configurations | `IPS_MODE` to `'list'` |"""

SETTINGS = """import os

ON_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if ON_KAGGLE:
    os.environ['IPS_MODE'] = 'train'
    os.environ['IPS_RUNS'] = 'ddr_mil33,ddr_ms66_33,ddr_h66_33,rsna_ips300,rsna_h300_150,rsna_mil300'
    os.environ['IPS_CACHE_DIR'] = '/kaggle/working/cache'
    os.environ['IPS_SEEDS'] = '42'
    os.environ['IPS_EPOCHS'] = '1'
    os.environ['IPS_SMOKE'] = '200'
    os.environ['IPS_WORKERS'] = '3'
    os.environ['IPS_DDR_GEOMETRY'] = 'retina'
print({k: v for k, v in os.environ.items() if k.startswith('IPS_')})
"""

PACKAGE_DOC = """## The code: package `ips_suite`

Each cell writes one file of the package (generated, do not edit here)."""

SUBPACKAGES = ('config', 'data', 'models', 'training', 'evaluation')
INSTALL = "%pip install -q codecarbon\n"

MAKE_FOLDERS = ("import os\n\nfor folder in " + repr(SUBPACKAGES) + ":\n"
                "    os.makedirs(f'ips_suite/{folder}', exist_ok=True)\n")

RUN_DOC = """## Run

Runs what the settings cell asks for. After editing a file above, restart the
kernel before running this cell again."""

RUN = """import sys

sys.path.insert(0, os.getcwd())
from ips_suite.__main__ import main

main()
"""


ANALYSIS_INTRO = """# IPS suite v3: analysis

Runs one analysis of `analysis/registry.json` where the results are: it
downloads from the Hugging Face Hub only what the analysis needs, runs it, and
uploads its outputs to `ips-suite-v3/analysis/<tag>/`. **Generated** by
`tools/build_notebook.py`; only the settings cell is meant to change here."""

ANALYSIS_SETTINGS = """import os
import time

ON_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
if ON_KAGGLE:
    os.environ['IPS_ANALYSIS'] = 'aggregate'
    os.environ['IPS_HUB_REPO'] = '<hf-user>/ips-suite-runs'
    os.environ['IPS_ANALYSIS_TAG'] = time.strftime('%Y-%m-%d_%H%M') + '_aggregate'
print({k: v for k, v in os.environ.items() if k.startswith('IPS_')})
"""

ANALYSIS_FOLDERS = ("import os\n\nfor folder in "
                    + repr(tuple(f"ips_suite/{name}" for name in SUBPACKAGES) + ('analysis',))
                    + ":\n    os.makedirs(folder, exist_ok=True)\n")

ANALYSIS_RUN = """import runpy
import sys

sys.path.insert(0, os.getcwd())
runpy.run_path('analysis/run_analysis.py', run_name='__main__')
"""


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
            'source': text.splitlines(keepends=True)}


def with_ids(cells):
    return [{'id': f"cell-{n:02d}", **cell} for n, cell in enumerate(cells)]


def module_cell(name):
    source = (PACKAGE / f"{name}.py").read_text()
    if not source.strip():
        sys.exit(f"{name}.py is empty: %%writefile needs content (a docstring is enough)")
    return code(f"%%writefile ips_suite/{name}.py\n{source}")


def assert_no_undefined_names():
    problems = [(path, scope, name) for path in sorted(PACKAGE.rglob('*.py'))
                for scope, name in undefined_names(path)]
    for path, scope, name in problems:
        print(f"{path.relative_to(REPO)}: {scope} reads undefined name {name}")
    if problems:
        sys.exit("fix the undefined names before building the notebook")


def repository_file_cell(relative):
    return code(f"%%writefile {relative}\n{(REPO / relative).read_text()}")


def package_cells():
    assert_no_undefined_names()
    missing = sorted(set(p.relative_to(PACKAGE).with_suffix('').as_posix()
                         for p in PACKAGE.rglob('*.py')) - set(MODULE_ORDER))
    if missing:
        sys.exit(f"modules missing from MODULE_ORDER: {missing}")
    return [module_cell(name) for name in MODULE_ORDER]


def as_notebook(cells):
    return {'cells': with_ids(cells),
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                        'name': 'python3'},
                         'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}


def notebook():
    cells = [markdown(INTRO), markdown(SETTINGS_DOC), code(SETTINGS), code(INSTALL),
             markdown(PACKAGE_DOC), code(MAKE_FOLDERS)]
    cells += package_cells()
    cells += [markdown(RUN_DOC), code(RUN)]
    return as_notebook(cells)


def analysis_notebook():
    cells = [markdown(ANALYSIS_INTRO), code(ANALYSIS_SETTINGS), code(ANALYSIS_FOLDERS)]
    cells += package_cells()
    cells += [repository_file_cell(f) for f in ANALYSIS_FILES]
    cells += [code(ANALYSIS_RUN)]
    return as_notebook(cells)


NOTEBOOKS = {NOTEBOOK: notebook, ANALYSIS_NOTEBOOK: analysis_notebook}


def code_of(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'][1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    stale = []
    for path, build in NOTEBOOKS.items():
        built = build()
        if args.check:
            if not path.exists() or code_of(json.loads(path.read_text())) != code_of(built):
                stale.append(path.name)
            continue
        path.write_text(json.dumps(built, indent=1, ensure_ascii=False) + '\n')
        print(f"wrote {path.name}: {len(built['cells'])} cells")
    if args.check:
        for name in stale:
            print(f"{name} is out of date: run python tools/build_notebook.py")
        if not stale:
            print("the notebooks match src/ and analysis/")
        return 1 if stale else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
