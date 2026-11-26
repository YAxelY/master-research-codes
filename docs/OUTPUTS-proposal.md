# Organising the outputs of the experiments: proposal for discussion

Draft of 25 November 2026, to discuss before the next experiments.

## 1. What a job writes, and where

| Kind                                                                     | Size                                                                          | Folder                                                | Pushed?                              |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------- | ----------------------------------------------------- | ------------------------------------ |
| Results: `_result.json`, history, codecarbon, lesion and Grad-CAM tables | a few hundred kB per run                                                      | `ips-project/results/ips-suite-v3/<configuration>/`   | **yes**                              |
| Per-image predictions (`predictions/*.npz`)                              | about 0.1 to 0.2 MB per run                                                   | same                                                  | yes                                  |
| Job logs (`.out`, `.err`)                                                | small                                                                         | `ips-project/logs/`                                   | yes                                  |
| Weights and resume state (`checkpoints/`)                                | about 45 MB of weights and 135 MB of resume state per run (ResNet-18 encoder) | `ips-project/artifacts/ips-suite-v3/<configuration>/` | no                                   |
| Grad-CAM maps (`saliency/*.npz`)                                         | about 75 MB per run                                                           | same                                                  | no                                   |
| Decoded-image cache (`*_retina_*.npy`)                                   | up to 15 GB per split, 33 GB in total for DDR                                 | `ips-project/cache/`                                  | no, delete when the last job is done |
| ImageNet weights (`backbone_cache/`)                                     | about 50 MB                                                                   | `ips-project/backbone_cache/`                         | no                                   |
| Generated `_run_*.py`, `outputs/`, `data/`                               |                                                                               | `ips-project/`                                        | no                                   |

Everything in the "no" rows is in `.gitignore`, so `git add ips-project/results ips-project/logs` is enough and cannot pick up a heavy file by mistake.

## 2. Questions for the session

1. **How results come back.** Proposal: one branch per batch of jobs, `results/<suite>/<date>` (for example `results/ips-suite-v3/2026-11-28`), containing only `ips-project/results/` and `ips-project/logs/`. Is a branch per batch convenient, or do you prefer pushing to one branch?
2. **Which checkpoints to keep, and where.** Only the best weights of the runs that appear in the paper? Kept on the cluster, or copied to a shared storage (institutional drive, private Hugging Face repository)? For how long?
3. **Grad-CAM maps.** They are needed to redo the interpretability figures. Keep them on the cluster only, or archive those of the main runs?
4. **The image cache.** Keep it between batches (it saves the decoding time of every DDR job) or delete it after each batch? It can also be moved to a scratch disk with `IPS_CACHE_DIR=/path/to/scratch`.
5. **Status of the runs.** A short table (configuration, seed, job id, status: queued, running, done, failed, outdated) updated with each push, so that a missing result is visible without asking.
6. **Outdated results.** When the protocol changes (as for the RSNA class exclusion), the old results are moved to `results/_outdated/<date>/` rather than deleted, and the change is written in the log of decisions.

## 3. Already done (branch `kaggle-notebooks`, after the Kaggle check)

- Weights and Grad-CAM maps are written to `ips-project/artifacts/` instead of `results/`.
- `.gitignore` excludes the artifacts, the caches, the generated scripts and the data.
- The backbone cache path is absolute. Before, the job wrote it to `ips-project/ips-project/backbone_cache/`.
- Interrupted runs still resume from their old checkpoint place.
