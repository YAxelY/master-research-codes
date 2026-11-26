# Architecture of ips-suite-v3

What the code does, drawn, so that every design choice is visible without
reading the implementation. Each choice and its reason is listed in
[DECISIONS.md](DECISIONS.md); how to run it is in [RUNNING.md](RUNNING.md).
Diagrams are Mermaid: they render on GitHub, in VS Code (Markdown preview
with a Mermaid extension) and in Obsidian.

Written on 2026-11-26 from the code at the commit that adds this file.
File names refer to the package `src/ips_suite/` (the notebook is generated from it).

## 1. The whole system

```mermaid
flowchart LR
  subgraph code[One code base: src/ips_suite]
    PKG[package modules] -->|tools/build_notebook.py| NB[ips-suite-v3.ipynb<br/>one writefile cell per module]
    S[settings cell]:::cfg --> NB
  end
  L[kaggle/launch.py<br/>one share of the runs per account] -->|pushes the notebook<br/>with its settings| K1[Kaggle run A]
  L --> K2[Kaggle run B]
  SH[run-ips-suite-v3.sh] -->|python -m ips_suite| CL[supervisor's cluster<br/>Slurm job]
  K1 -->|resume state every 2 epochs,<br/>results at the end| HUB[(Hugging Face Hub<br/>private dataset)]
  K2 --> HUB
  CL -.->|optional, IPS_HUB_REPO| HUB
  CL -->|git push of results/| GIT[(GitHub<br/>results branch)]
  HUB -->|weights, maps, results| AN[Kaggle analysis notebook<br/>analysis/registry.json]
  AN -->|outputs only| HUB
  HUB -->|hub/pull_results.py| LOC[local results folder]
  GIT --> LOC
  LOC --> AGG[aggregate-ips-suite-v3.py<br/>tables per setup]
  classDef cfg fill:#fdebd0
```

The same package runs everywhere (through the notebook on Kaggle, directly on the cluster); only environment variables change
(`IPS_RUNS`, `IPS_EPOCHS`, data folders, `IPS_HUB_REPO`). The cluster keeps its
own way of running (Slurm script, git push of results); the Hub is optional
there.

## 2. Data (`data/datasets.py`)

```mermaid
flowchart TB
  subgraph DDR[DDR, 5 grades]
    D1[label files] -->|drop grade 5<br/>ungradable| D2[train 6 260 / valid 2 503 / test 3 759]
    D2 --> D3[retina crop:<br/>grey > 12, bounding box,<br/>square around the centre, black fill]
    D3 --> D4[resize to 896 px, bilinear]
  end
  subgraph RSNA[RSNA pneumonia, opacity against normal]
    R1[stage2 train metadata] -->|drop class<br/>No Lung Opacity / Not Normal| R2[one label per patient]
    R2 -->|MD5 of patient id, seed 42| R3[train 70 / val 15 / test 15 %]
    R3 --> R4[PNG 1024 px]
  end
  D4 --> C[(uint8 cache on disk,<br/>built once, keyed by<br/>file list + size + geometry)]
  R4 --> C
  C --> A[augmentation on the GPU, per epoch:<br/>DDR flips, rotation 15, colour jitter<br/>RSNA horizontal flip, rotation 5]
  A --> N[normalisation]
```

The images of each split are identified by a hash (`split_hash`, stored in
every result): two runs with the same hash used exactly the same images.

## 3. The models (`models/`)

All patch models share one encoder (ResNet-18, all four blocks, ImageNet
weights, average pooling to 512 values), one projector (LayerNorm, linear to
D = 512, BatchNorm, ReLU) and one aggregator (cross-attention with one learned
query, then an MLP), followed by a linear head.

### IPS: select without gradients, train on the selection

```mermaid
flowchart LR
  I[image 896 px] --> T[tiling by 66 px<br/>196 candidates<br/>last row and column padded]
  T --> X
  subgraph X[exploration, no gradient]
    direction TB
    X1[memory = first M = 10 candidates] --> X2[encode the next I = 128]
    X2 --> X3[score memory + new ones<br/>with the aggregator's attention]
    X3 --> X4[keep the M best]
    X4 -->|until all candidates are seen| X2
  end
  X --> E[exploitation, with gradient:<br/>encode the M kept patches]
  E --> AGG[cross-attention aggregator] --> H[head] --> Y[grade]
```

Memory depends on M and I, not on the image size: only M patches carry
gradients.

### MS-IPS: two patch sizes compete for the same M places

```mermaid
flowchart LR
  I[image] --> T1[tiling by 66 px: 196]
  I --> T2[tiling by 33 px: 784]
  T1 --> P[one pool of 980 candidates<br/>+ a learned scale embedding]
  T2 --> P
  P --> X[same exploration as IPS] --> E[M = 10 patches of any size] --> AGG[aggregator] --> Y[grade]
```

The encoder's average pooling gives 512 values whatever the patch size, so
patches of both sizes are comparable after the projector.

### H-MS-IPS: coarse zones first, small patches inside them

```mermaid
flowchart LR
  I[image] --> Z[zones of 66 px: 196]
  Z --> S0[exploration: keep 49 zones]
  S0 --> C[cut each kept zone into 4 patches of 33 px: 196]
  C --> S1[exploration: keep M = 10]
  S1 --> E[exploitation] --> AGG[aggregator] --> Y[grade]
```

On RSNA: zones of 300 px (16), keep 4, cut into 150 px patches (16), keep 10.

### Reference models

| Model                      | What it does                                                | Why it is here                                   |
| -------------------------- | ----------------------------------------------------------- | ------------------------------------------------ |
| Full image (`BaselineNet`) | ResNet-18 on the whole image, end to end                    | the reference every patch model is compared with |
| Attention MIL (`MILNet`)   | every patch encoded with gradients, gated attention pooling | what selection saves memory against              |
| `_sa` variants             | self-attention aggregator instead of cross-attention        | ablation of the aggregator                       |
| `_rand` variants           | random selection instead of learned                         | ablation of the selection                        |

## 4. One training run (`training/epoch.py`, `training/runner.py`)

```mermaid
sequenceDiagram
  participant R as run()
  participant H as HubMirror
  participant E as run_epoch
  R->>H: pull result, best weights, resume state
  alt result exists
    R-->>R: skip
  else resume state exists
    R->>R: load model, optimiser, history
  end
  loop each remaining epoch
    R->>R: seeds from (seed, epoch)
    R->>E: train: B_seq images selected, then one step per B = 16
    R->>E: validation (fixed selection seed)
    R->>R: best validation score: save weights
    R->>R: save resume state
    R->>H: every 2 epochs: push state and weights
  end
  R->>R: reload best weights
  R->>R: evaluate val and test, threshold on val, lesion and Grad-CAM studies
  R->>R: write result.json (with environment and split hashes)
  R->>H: push results, delete resume state
```

Optimiser AdamW (peak lr 1e-4, weight decay 0.1), linear warm-up over 10
epochs then cosine decay to 1/1000 of the peak, 50 epochs (schedule of the
original IPS code, `training/schedule.py`); loss = negative log-likelihood of the softmax. The epoch
kept is the best on validation; the test set is used once, at the end.

## 5. Where the files go

| Folder                                                | Content                                         | Shared         |
| ----------------------------------------------------- | ----------------------------------------------- | -------------- |
| `results/ips-suite-v3/<configuration>/experiments/`   | `<run>_result.json`, lesion and Grad-CAM tables | yes (git, Hub) |
| `results/.../history/`                                | per-epoch history, CO2                          | yes            |
| `results/.../predictions/`                            | per-image probabilities, val and test           | yes            |
| `artifacts/ips-suite-v3/<configuration>/checkpoints/` | best weights, resume state                      | Hub only       |
| `artifacts/.../saliency/`                             | Grad-CAM maps                                   | Hub only       |
| `cache/`                                              | decoded images                                  | never          |

## 6. Reproducibility across machines

| Same everywhere                                   | Differs with the machine                           |
| ------------------------------------------------- | -------------------------------------------------- |
| code (commit stored in `environment.code_commit`) | time per epoch, total time                         |
| images of each split (`data.split_hash`)          | peak memory by a few percent (allocator)           |
| seeds, and every random draw per epoch            | the last digits of the scores (GPU kernels differ) |
| hyper-parameters (`config`)                       | CO2 estimate                                       |

`IPS_DETERMINISTIC=1` makes two runs on the same GPU type give the same
numbers, at some cost in speed. Across GPU types the scores stay within
run-to-run noise; the aggregation script lists the setup of every run and
warns when a table mixes setups or splits.
