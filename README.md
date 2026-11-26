# IPS suite v3: DDR and RSNA

Patch selection for high-resolution medical image classification: IPS,
MS-IPS (two patch sizes competing for the same places) and H-MS-IPS (coarse
zones, then small patches inside the kept zones), against the full-image
ResNet-18 and attention MIL.

## What is in this repository

| File | Role |
|---|---|
| `src/ips_suite/` | **the code**, as a package: `config/` (environment, experiments), `data/` (datasets, tiling), `models/`, `training/` (epoch, runner, Hub sync), `evaluation/` (metrics, studies, benchmark), `__main__.py`; run with `python -m ips_suite` |
| `ips-suite-v3.ipynb` | the same code for Kaggle, **generated** from `src/` by `tools/build_notebook.py` (one `%%writefile` cell per module) |
| `run-ips-suite-v3.sh` | SLURM script: runs `python -m ips_suite` for one or more configurations |
| `kaggle/launch.py` | launches a share of the runs on a Kaggle account (results synchronised to the Hugging Face Hub) |
| `hub/pull_results.py` | brings the results stored on the Hub back to this machine |
| `submit-v3.sh` | submits one job per configuration of a group |
| `aggregate-ips-suite-v3.py` | tables, seeds, bootstrap, paired comparisons (on a laptop) |
| `kaggle.json` | Kaggle credentials, used to download the data sets |
| `docs/` | `ARCHITECTURE.md` (diagrams), `DECISIONS.md` (every design choice), `RUNNING.md` (Kaggle, cluster, reporting), `SUITE-V3-contents.md`, `SUITE-V3-design.md` |
| `tools/` | `build_notebook.py` writes the notebook from `src/`; `make-submit.py` regenerates `submit-v3.sh`; `make-fake-data.py` builds small fake data sets for tests |
| `tmp/` | older notebooks and scripts, kept for reference, not used by the suite |

## Running

From the folder of the repository, on the cluster:

```bash
bash submit-v3.sh focus 42        # the selection of 18 November, 14 jobs
bash submit-v3.sh focus_abl 42    # its ablations, 18 jobs
bash submit-v3.sh focus_m 42      # its values of M, 36 jobs
bash submit-v3.sh focus 43        # another seed
```

Without training: `IPS_MODE=bench sbatch run-ips-suite-v3.sh` (memory and
speed benchmark), `IPS_MODE=eval_v2 sbatch run-ips-suite-v3.sh` (last
week's 21 models on the DDR test set).

On Kaggle: `python kaggle/launch.py ...` (see `docs/RUNNING.md`), or open the
notebook, set the settings cell, run all.

After any change in `src/ips_suite/`: `python tools/build_notebook.py`, then
commit both. The launcher refuses a notebook that does not match `src/`.

## Results

`ips-project/results/ips-suite-v3/` (light, pushed) and
`ips-project/artifacts/ips-suite-v3/` (heavy, never pushed), **one folder per
configuration**, every seed inside it:

```
results/.../<configuration>/experiments/   <run>_result.json, lesion and Grad-CAM tables
                           /history/       <run>_history.json, codecarbon
                           /predictions/   per image: probabilities, patches, score maps
artifacts/.../<configuration>/checkpoints/ best weights, resume state
                             /saliency/    per image: Grad-CAM maps
```

Then `python aggregate-ips-suite-v3.py ips-project/results/ips-suite-v3`
writes the tables to `_aggregate/`.

## Patch sizes

DDR (896 px) uses 66 and 33 px, RSNA (1024 px) 300 and 150 px. These sizes
do not divide the image: the last row and column of patches overflow it and
are completed with black (padding).
