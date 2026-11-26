#!/usr/bin/env python3
"""Write ips-suite-v3.ipynb from the package in src/ips_suite/.

    python tools/build_notebook.py           # rewrite the notebook
    python tools/build_notebook.py --check   # exit 1 if the notebook is out of date

The package is the only source of the code. The notebook holds one
%%writefile cell per module, so Kaggle shows the same files, and a last cell
that runs the package: both ways run the very same code.
"""
import argparse
import json
import sys
from pathlib import Path

from check_names import check as undefined_names

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / 'src' / 'ips_suite'
NOTEBOOK = REPO / 'ips-suite-v3.ipynb'
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


def notebook():
    assert_no_undefined_names()
    missing = sorted(set(p.relative_to(PACKAGE).with_suffix('').as_posix()
                         for p in PACKAGE.rglob('*.py')) - set(MODULE_ORDER))
    if missing:
        sys.exit(f"modules missing from MODULE_ORDER: {missing}")
    cells = [markdown(INTRO), markdown(SETTINGS_DOC), code(SETTINGS),
             markdown(PACKAGE_DOC), code(MAKE_FOLDERS)]
    cells += [module_cell(name) for name in MODULE_ORDER]
    cells += [markdown(RUN_DOC), code(RUN)]
    return {'cells': with_ids(cells),
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                        'name': 'python3'},
                         'language_info': {'name': 'python'}},
            'nbformat': 4, 'nbformat_minor': 5}


def code_of(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code'][1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    built = notebook()
    if args.check:
        current = json.loads(NOTEBOOK.read_text())
        if code_of(current) != code_of(built):
            print(f"{NOTEBOOK.name} is out of date: run python tools/build_notebook.py")
            return 1
        print(f"{NOTEBOOK.name} matches src/ips_suite")
        return 0
    NOTEBOOK.write_text(json.dumps(built, indent=1, ensure_ascii=False) + '\n')
    print(f"wrote {NOTEBOOK.name}: {len(built['cells'])} cells")
    return 0


if __name__ == '__main__':
    sys.exit(main())
