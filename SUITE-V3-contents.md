# Suite v3: everything it runs, measures and allows

Written on 16 November 2026 from the code of `ips-suite-v3.ipynb`. Every item
below is something the code does; section 9 lists what it does not do.

## 1. Data

| | Train | Validation (chooses the epoch) | Test (the reported score) |
|---|---|---|---|
| DDR, 5 grades, ungradable excluded | 6 260 | 2 503 | 3 759 (official split) |
| RSNA, pneumonia yes / no | about 18 700 | about 4 000 | about 4 000 (70 / 15 / 15 by patient) |

- DDR images are cropped to the retina (square pixels, no black margin), then
  resized to 896 px. RSNA images are used at 1024 px.
- The split never depends on the seed.
- Annotations, used only for analysis, never for training: DDR lesion masks
  (four types) on 185 test and 143 validation images; RSNA opacity boxes on
  about 950 test and 860 validation patients.

## 2. The models

| Model | What it does |
|---|---|
| Baseline | ResNet-18 on the whole image, at full or reduced resolution |
| MIL | cuts the image into all its 128 px patches, trains on all of them, a gated attention (Ilse et al., 2018) pools them. No selection |
| IPS | one patch size; scores every patch without gradients, keeps the best M, trains on those M only |
| MS-IPS | two patch sizes compete for the same M places |
| H-MS-IPS | keeps the best large zones, cuts only those into small patches, keeps the best M |

Variants used by the ablations:

| Variant | Change |
|---|---|
| `_sa` | self-attention aggregator instead of cross-attention |
| `_rand` | patches chosen at random, during training and testing |
| `_tc` | the coarse zones of H-MS-IPS are also given to the classifier, so the coarse stage is trained |
| `_m<M>` | another value of M |
| `_k<K>` | another number of zones kept by the coarse stage |
| `_f25` | 25 % of the training set |
| `_rs` | small patches upsampled to the large size (resampling control) |
| `lr` | coarse zones read at half resolution |
| `_lazy` | image kept on the CPU, only patches sent to the GPU |
| `_i16` | 16 candidates scored at a time instead of 128 |
| `r1792` | DDR at 1792 px instead of 896 |

Shared by all: ImageNet ResNet-18, AdamW, learning rate 10⁻⁴ with 10 epochs of
warm-up then cosine decay, weight decay 0.1, 50 epochs, batch 16, M = 10 and
128 candidates scored at a time unless stated. Every model uses at most two
patch sizes.

## 3. The 74 configurations

### core (22): the main table, three seeds

| DDR (896 px) | RSNA (1024 px) |
|---|---|
| full image 896, full image 448 | full image 1024, full image 512 |
| MIL 128 | MIL 128 |
| IPS 128, 112, 64, 56, 32 | IPS 256, 128, 64 |
| MS-IPS 128/64, 64/32 | MS-IPS 256/128, 128/64 |
| H-MS-IPS 224 then 112, 224 then 56 | H-MS-IPS 256 then 128, 256 then 64 |

### ablation (8)

| DDR | RSNA |
|---|---|
| IPS 128 self-attention, IPS 128 random selection | same |
| H-MS-IPS 224/112 and 224/56, coarse zones trained | H-MS-IPS 256/128 and 256/64, coarse zones trained |

### mstudy (20): values of M

| | Values |
|---|---|
| DDR, IPS 128 | M = 2, 5, 10, 20, 40 |
| DDR, IPS 64 | M = 10, 20, 40 |
| DDR, IPS 32 | M = 10, 40, 80, 160 |
| DDR, H-MS-IPS 224/56 | M = 10, 20, 40; zones kept 2, 4, 8 |
| RSNA, IPS 128 | M = 2, 5, 10, 20, 40 |
| RSNA, IPS 64 | M = 10, 20, 40 |
| RSNA, H-MS-IPS 256/64 | M = 10, 40 |

Several of these train on the same number of pixels (M × patch²), for
example 10 × 128 = 40 × 64 = 160 × 32 on DDR.

### Other groups (24)

| Group | Configurations |
|---|---|
| patch (6) | DDR IPS 224 and 28, H-MS-IPS 224/28 with and without trained zones; RSNA IPS 32, H-MS-IPS 256/32 |
| base (2) | full image at a quarter of the resolution, DDR and RSNA |
| ms (2) | MS-IPS 128/64 with the 64 px patches upsampled, DDR and RSNA |
| lowres (2) | H-MS-IPS with the coarse stage at half resolution, DDR and RSNA |
| hires (4) | DDR at 1792 px: IPS 128 (M 10 and 20), IPS 112, H-MS-IPS 448/112 |
| lowdata (6) | 25 % of the training set: DDR IPS 128, 64, 32 and H-MS-IPS 224/56; RSNA IPS 128, 64 |
| misc (2) | DDR IPS 128 with lazy loading, and with 16 candidates at a time |

## 4. What every training run measures

### 4.1 Before training (exact counts, no data needed)

- number of candidate patches
- pixels read and pixels encoded by the selection, also as a multiple of the image
- pixels trained on per image (with gradients), also as a share of the image
- size of the feature map of each patch size after the encoder
- memory of the M + I patches and of the image
- number of parameters
- GFLOPs for one image at inference, split into selection and classification

### 4.2 At every epoch, on training and validation

- every metric of 4.3 (on training images these are computed with augmentation on)
- loss and learning rate
- time spent reading images, selecting patches, and in the training step
- peak GPU memory of the selection and of the training step

### 4.3 With the weights of the best epoch, on validation and test

**DDR (5 grades)**

- quadratic kappa (main metric), linear kappa
- accuracy, balanced accuracy
- F1 macro and weighted
- precision, recall and F1 for each grade
- mean error in grades
- AUC one-vs-rest, macro
- referable DR (grade 2 or more): AUC, sensitivity, specificity
- confusion matrix

**RSNA (pneumonia yes / no)**

- AUC (main metric), AUPRC
- accuracy, balanced accuracy, F1, precision, recall, specificity
- confusion matrix
- the same at the decision threshold chosen on validation (Youden index),
  applied to test

**Both**

- calibration: expected calibration error, Brier score, log loss
- 95 % confidence intervals on test (1 000 bootstrap resamples) for the
  main metric, accuracy and, on DDR, referable AUC
- the same metrics at the best epoch on training, and at the last epoch on
  training and validation (overfitting)

### 4.4 Efficiency

- peak GPU memory: training (whole, selection, training step), validation, test
- training time: reading, selection, training step, and model time per image
- inference: time per image (selection, classification, total) and images per second
- energy (kWh) and CO₂ (codecarbon, on the cluster)
- GPU name

### 4.5 Interpretability

**What the selection is worth** (selectors trained with attention, on test):
the same weights evaluated with
1. patches drawn at random,
2. random positions but the model's own mix of sizes,
3. the model's own choice.

The difference between 1 and 2 is the value of choosing the sizes; between
2 and 3, the value of choosing the positions.

**Do the patches fall on lesions** (all models except the baseline; MIL uses
its 10 patches of highest attention), on validation and test, for each
lesion type and all together:
- share of the selected patches that touch a lesion
- the same share for random patches of the same size, computed exactly over all patches
- ratio of the two (how many times better than chance)
- share of images where at least one selected patch touches a lesion, and the chance value
- share of the lesion inside the selected patches
- share of the image covered by the selected patches
- for H-MS-IPS: share of the lesion inside the kept zones, and share of images where the zones miss the lesion entirely

**Which sizes are chosen**: how many of the selected patches are of each size.

**How concentrated the decision is**: share of the weight given to the most
important patch.

### 4.6 Saved for every validation and test image

| Field | Content |
|---|---|
| ids, paths | which image |
| y_true, y_prob | label and predicted probabilities |
| coords | the selected patches (size, row, column), in order of selection |
| attn | weight of each selected patch in the final decision |
| zones | the zones kept by the coarse stage (H-MS-IPS) |
| score_maps | a score for every candidate patch of the image, for heatmaps: on test for the cross-attention models, on validation and test for MIL |
| candidates, sizes, img_size | to place every score on the image |

### 4.7 Files of one run

| File | Content |
|---|---|
| `checkpoints/<run>_weights.pt` | weights of the best epoch |
| `experiments/<run>_result.json` | configuration, costs, every metric of section 4, history of all epochs |
| `experiments/<run>_lesions_test.csv`, `_val.csv` | lesion results per image |
| `predictions/<run>_test.npz`, `_val.npz` | section 4.6 |
| `history/codecarbon/<run>/` | energy log |

## 5. Runs without training

| Mode | What it does |
|---|---|
| `bench` | random images, a few minutes. Peak memory (selection, training step, inference) and time, varying one factor at a time: patch size (224 to 28 on DDR, 256 to 32 on RSNA), M (2 to 40), candidates scored at a time (8 to 256), lazy against eager loading, an image twice as large, MS-IPS, every H-MS-IPS with and without trained zones, the full image at three resolutions, MIL, self-attention. Writes `bench.csv` |
| `eval_v2` | scores the 21 models of the previous suite on the test images, with sections 4.3 to 4.6. Clean on DDR; on RSNA those models had seen part of the new test set |
| `list` | prints the 74 configurations with their exact costs (section 4.1) |

## 6. Aggregation (`aggregate-ips-suite-v3.py`, on a laptop)

- `runs.csv`: one row per run with the main numbers
- `configs.csv`: mean and standard deviation over seeds
- `pairs.csv`: 32 paired comparisons on the same test images, with a
  bootstrap interval for the difference (cascade against IPS at the same
  final patch, MS-IPS against each of its two sizes, IPS against the
  half-resolution image and against MIL, cross-attention against
  self-attention, learned against random selection, coarse zones trained or
  not, equal-pixel M, 1792 against 896 px, resampled against native)
- `tables.md`: the tables, ready to read

## 7. Questions it answers

| Question | Configurations | Measures |
|---|---|---|
| Does a coarse stage help at the same memory? | H-MS-IPS against IPS at the same final patch | score, paired difference |
| Does mixing two sizes help? | MS-IPS against its two sizes | score, sizes chosen, value of choosing sizes |
| Is selection better than shrinking the image? | IPS against the half-resolution image | score at similar memory |
| What does selection save against using every patch? | IPS against MIL | score, memory, GFLOPs |
| Which selection module? | cross-attention, self-attention, random | score, lesion hits |
| Does training the coarse stage help? | `_tc` against without | score, share of the lesion kept by the zones |
| How many patches are needed? | mstudy | score and memory against M |
| Few large or many small patches? | equal M × patch² | score at equal memory |
| Does more resolution help? | hires | score at the same memory |
| Which patch size for which lesion size? | patch sizes 28 to 256 on two data sets | score, lesion hits |
| Does it hold with little data? | lowdata | score |
| Do the patches fall on lesions? | all except the baseline | lesion study |
| Is the cost known in advance? | bench and every run | predicted against measured memory |
| Are the models calibrated? | all | ECE, Brier |
| Is a difference real? | core with three seeds | standard deviation, paired bootstrap |

## 8. Safety of the runs

- An interrupted run resumes at its last epoch.
- A finished run is skipped.
- A run already running in another job is skipped.
- A failed run does not stop the others in the same job.

## 9. What it does not do

- **No figures.** Heatmaps and qualitative examples are drawn from the saved
  files (section 4.6) by a script still to write; no rerun is needed.
- **No heatmap** for the self-attention and random-selection models (their
  scores depend on the other patches of the group), nor for the baseline.
  No Grad-CAM.
- **Lesion analysis only on annotated images**: 185 DDR and about 950 RSNA
  test images.
- **One seed** outside `core`, unless more are launched.
- **One backbone** (ResNet-18), one training recipe, no hyperparameter search.
- **No external data set.**
- **The referable-DR threshold is not tuned**: a grade of 2 or more is
  predicted from the most probable grade.
- **Energy and CO₂ only where codecarbon is installed** (the cluster, not Kaggle).
- **Memory and time are those of the cluster GPU.**
- **The training-set metrics are computed with augmentation on**, so they
  slightly understate the fit.
- **DDR at 1792 px is not cached**: those four runs read images from disk.
