# IPS suite v3: DDR and RSNA

Patch selection for high-resolution medical image classification: IPS,
MS-IPS (two patch sizes competing for the same places) and H-MS-IPS (coarse
zones, then small patches inside the kept zones), against the full-image
ResNet-18 and attention MIL.

## What is in this repository

| File | Role |
|---|---|
| `ips-suite-v3.ipynb` | the whole suite: models, data, training, evaluation, analyses |
| `run-ips-suite-v3.sh` | SLURM script: runs the notebook for one or more configurations |
| `submit-v3.sh` | submits one job per configuration of a group |
| `aggregate-ips-suite-v3.py` | tables, seeds, bootstrap, paired comparisons (on a laptop) |
| `kaggle.json` | Kaggle credentials, used to download the data sets |
| `docs/` | `SUITE-V3-contents.md`: everything that is run and measured; `SUITE-V3-design.md`: design and reasons |
| `tools/` | `make-submit.py` regenerates `submit-v3.sh`; `make-fake-data.py` builds small fake data sets for tests |
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

On Kaggle: open the notebook, set the first cell, run all.

## Results

`ips-project/results/ips-suite-v3/`, **one folder per configuration**, every
seed of that configuration inside it:

```
<configuration>/checkpoints/   weights of the best epoch, resume state
                /experiments/   <run>_result.json, lesion and Grad-CAM tables
                /history/       <run>_history.json, codecarbon
                /predictions/   per image: probabilities, patches, score maps
                /saliency/      per image: Grad-CAM maps
```

Then `python aggregate-ips-suite-v3.py ips-project/results/ips-suite-v3`
writes the tables to `_aggregate/`.

## Patch sizes

DDR (896 px) uses 66 and 33 px, RSNA (1024 px) 300 and 150 px. These sizes
do not divide the image: the last row and column of patches overflow it and
are completed with black (padding).
