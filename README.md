# Multi-Scale Iterative Patch Selection for High-Resolution Medical Images

Code for the MSc thesis *Deep Learning for High-Resolution Medical Image Analysis:
Multi-Scale Iterative Patch Selection for Thoracic and Retinal Pathologies*
(University of Dschang, URIFIA research unit, defended 21 July 2026, graded 17/20).

High-resolution medical images do not fit in GPU memory. Downsampling destroys the fine
detail that carries the diagnosis; cropping into patches destroys the global anatomical
context. This work introduces two architectures that select and fuse informative patches
across several spatial scales under a strictly bounded memory budget, formulated as
multiple-instance learning, with ResNet and Vision Transformer encoders.

## Results

| Benchmark | Task | Metric | Result |
| --- | --- | --- | --- |
| RSNA | Pneumonia detection | AUC | **0.867**, best of models compared |
| NODE21 | Lung nodule detection | AUC | Perfect score |
| DDR | Diabetic retinopathy grading | Quadratic weighted kappa | **0.886**, best of models compared |
| LUNA16 | Lung CT, large-bag setting | —   | Large-bag feasibility experiment |

Peak training memory reduced three to five fold, to **1.5 GB**, placing native-resolution
training within reach of a single consumer GPU. Saliency maps are produced by the selection
mechanism itself rather than by post-hoc attribution.

## Layout

Notebooks were developed on Kaggle; each has a matching `run-*.sh` launcher.

### Methods

| Notebook | Contents |
| --- | --- |
| `ms-ips-2d.ipynb` | MS-IPS, the main multi-scale iterative patch selection model |
| `ms-ips-op.ipynb` | MS-IPS, optimised variant |
| `h-ms-ips.ipynb` | H-MS-IPS, the hierarchical variant |
| `ips.ipynb` | Single-scale iterative patch selection, the baseline this work extends |
| `ips-self-attention.ipynb` | IPS with a self-attention aggregation head |
| `ips-keras-style.ipynb` | Reimplementation used to cross-check the selection logic |
| `baseline-mps-gx.ipynb` | Baseline for comparison |
| `ips-gradcam.ipynb` | Saliency and interpretability analysis |

### Data and experiments

| Notebook | Dataset |
| --- | --- |
| `master-2-rsna-pneumonia-detection.ipynb` | RSNA pneumonia detection |
| `master-2-eda-node21.ipynb` | NODE21 lung nodules |
| `master-2-eda-ddr.ipynb` | DDR diabetic retinopathy grading |
| `master-2-eda-lungct.ipynb` | Lung CT ingestion and exploration |
| `master-2-eda-lungct-hms-ips-ms-ips.ipynb` | Lung CT, both architectures |
| `master2-qmix-standard-ce.ipynb` | *(describe this one in a line)* |

`20260614-ips-project/` and `20260628-ips-project/` are dated project snapshots.

## Reproducing

Each notebook is self-contained and expects the corresponding Kaggle dataset to be attached.
The `run-*.sh` scripts launch the matching notebook. Provide your own Kaggle credentials
through the environment or `~/.kaggle/kaggle.json`; **do not commit that file.**

## Author

MBOUMEU YOUMBI Axel Aubin — mboumeu581@gmail.com — [github.com/YAxelY](https://github.com/YAxelY)
