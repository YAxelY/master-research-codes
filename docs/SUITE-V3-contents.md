# Suite v3: everything it runs, measures and allows

Written on 16 November 2026 from the code of `ips-suite-v3.ipynb`. Every item
below is something the code does; section 9 lists what it does not do.

## 1. Data

| | Train | Validation (chooses the epoch) | Test (the reported score) |
|---|---|---|---|
| DDR, 5 grades, ungradable excluded | 6 260 | 2 503 | 3 759 (official split) |
| RSNA, pneumonia yes / no | about 18 700 | about 4 000 | about 4 000 (70 / 15 / 15 by patient) |

- DDR images are cropped to the retina (square pixels, no black margin), then
  resized to 896 px (1792 px for `hires`). RSNA images are used at 1024 px.
- Decoded images are cached once on disk and shared by all jobs.
- DDR training images are augmented on the GPU (flips, rotation up to 15°,
  colour jitter), drawn independently for each image.
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

## 3. The 232 configurations

### focus (14): the selection of 18 November, launched first

| DDR (896 px) | RSNA (1024 px) |
|---|---|
| full image 896 | full image 1024 |
| MIL 66, MIL 33 | MIL 300, MIL 150 |
| IPS 66, IPS 33 | IPS 300, IPS 150 |
| MS-IPS 66/33 | MS-IPS 300/150 |
| H-MS-IPS 66 then 33 (keeps 49 of 196 zones) | H-MS-IPS 300 then 150 (keeps 4 of 16 zones) |

These sizes do not divide the image. The tiling gets one more row and
column of patches, which overflow the image and are completed with black:

| Size | Patches per side | Tiled width | Overflow |
|---|---|---|---|
| 66 px on 896 | 14 | 924 px | 28 px |
| 33 px on 896 | 28 | 924 px | 28 px |
| 300 px on 1024 | 4 | 1200 px | 176 px |
| 150 px on 1024 | 7 | 1050 px | 26 px |

The black part is read by the encoder like the rest, so the selection cost
of IPS 300 on RSNA is 1.37 images instead of 1.00.

### focus_abl (18) and focus_m (36): ablations and M of the focus

- `focus_abl`: each of the 8 selectors of `focus` with the self-attention
  aggregator and with random selection, plus the 2 cascades with their
  coarse zones trained.
- `focus_m`: each of the 8 selectors with every M among 2, 5, 20, 40, 80, 160
  smaller than its pool of candidates (up to 160 on DDR; up to 5 for IPS 300
  and H-MS-IPS 300/150, which have 16 candidates; up to 40 for IPS 150 and
  MS-IPS 300/150).

The groups below are the earlier selection (sizes that divide the image),
kept and runnable with the same commands.

### core (24): the main table, three seeds

| DDR (896 px) | RSNA (1024 px) |
|---|---|
| full image 896, full image 448 | full image 1024, full image 512 |
| MIL 128 | MIL 128, MIL 256 |
| IPS 128, 112, 64, 56, 32 | IPS 256, 128, 64 |
| MS-IPS 128/64, 64/32 | MS-IPS 256/128, 128/64 |
| H-MS-IPS 224 then 112, 224 then 56, 64 then 32 | H-MS-IPS 256 then 128, 256 then 64 |

### ablation (38): the selection module, on every selector

Each of the 17 selectors of `core` (IPS, MS-IPS and H-MS-IPS, both data
sets) is trained twice more:

| Variant | Applied to |
|---|---|
| self-attention aggregator (`_sa`) | DDR: IPS 128, 112, 64, 56, 32; MS-IPS 128/64, 64/32; H-MS-IPS 224/112, 224/56, 64/32. RSNA: IPS 256, 128, 64; MS-IPS 256/128, 128/64; H-MS-IPS 256/128, 256/64 |
| random selection (`_rand`) | the same 17 |
| coarse zones trained (`_tc`) | the 4 H-MS-IPS of `core` |

The cross-attention version is the `core` run itself.

### mstudy (80): values of M, on every selector

Each of the 17 selectors of `core` is trained with every value of M among
2, 5, 20, 40, 80 and 160 that is smaller than its pool of candidates
(M = 10 is the `core` run itself), plus the cascade 224/56 keeping 2 or 8
zones instead of 4.

| Selector | Pool | Values of M |
|---|---|---|
| DDR IPS 128, IPS 112; RSNA IPS 128 | 49 to 64 | 2, 5, 10, 20, 40 |
| DDR IPS 64, 56, 32; RSNA IPS 64 | 196 to 784 | 2, 5, 10, 20, 40, 80, 160 |
| DDR MS-IPS 128/64, 64/32; RSNA MS-IPS 128/64 | 245 to 980 | 2, 5, 10, 20, 40, 80, 160 |
| RSNA MS-IPS 256/128 | 80 | 2, 5, 10, 20, 40 |
| DDR H-MS-IPS 64/32 | 196 | 2, 5, 10, 20, 40, 80, 160 |
| DDR H-MS-IPS 224/56; RSNA H-MS-IPS 256/64 | 64 | 2, 5, 10, 20, 40 |
| DDR H-MS-IPS 224/112; RSNA H-MS-IPS 256/128; RSNA IPS 256 | 16 | 2, 5, 10 |

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
- time spent reading images, augmenting them (on the GPU, DDR), selecting patches, and in the training step
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
- training time: reading, augmentation, selection, training step, and model time per image
- inference: time per image (selection, classification, total) and images per second
- energy (kWh) and CO₂ (codecarbon, on the cluster)
- GPU name

### 4.5 Interpretability

All the analyses below use the very patches that produced the test score:
the lesion study, Grad-CAM and the top-M row of the selection study do not
draw a new selection.

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

**Grad-CAM** (every model, test split): the map of the class explained
(pneumonia on RSNA, the predicted grade on DDR), from the last residual stage
of the ResNet-18: over the whole image for the baseline, inside every patch
for MIL, inside every selected patch for the selectors (zero elsewhere, so
the map shows both where the model looked and what it used inside). On the
annotated images, for each lesion type:
- share of the map's mass on the lesion, and that share divided by the
  lesion's share of the image (1 = no better than a uniform map)
- pointing game: the maximum of the map lies within 15 px of a lesion
- pixel AUROC: the map used as a pixel-wise lesion detector

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

One folder per configuration, `results/ips-suite-v3/<configuration>/`, every
seed of the configuration inside it (file names end with the seed, `_s42`):

| File | Content |
|---|---|
| `checkpoints/<run>_weights.pt` | weights of the best epoch |
| `experiments/<run>_result.json` | configuration, costs, every metric of section 4, history of all epochs |
| `experiments/<run>_lesions_test.csv`, `_val.csv` | lesion results per image |
| `experiments/<run>_gradcam_test.csv` | Grad-CAM lesion results per annotated image |
| `history/<run>_history.json` | metrics, time and memory of every epoch |
| `history/codecarbon/` | energy log |
| `predictions/<run>_test.npz`, `_val.npz` | section 4.6 |
| `saliency/<run>_gradcam_test.npz` | Grad-CAM of every test image (112 × 112) and the class explained |

The benchmark writes `_bench/experiments/bench.csv`, the aggregation
`_aggregate/`.

## 5. Runs without training

| Mode | What it does |
|---|---|
| `bench` | random images, a few minutes. Peak memory (selection, training step, inference) and time, varying one factor at a time: patch size (224 to 28 on DDR, 256 to 32 on RSNA), M (2 to 40), candidates scored at a time (8 to 256), lazy against eager loading, an image twice as large, MS-IPS, every H-MS-IPS with and without trained zones, the full image at three resolutions, MIL, self-attention. Writes `bench.csv` |
| `eval_v2` | scores the 21 models of the previous suite on the test images, with sections 4.3 to 4.6. Clean on DDR; on RSNA those models had seen part of the new test set |
| `list` | prints the 74 configurations with their exact costs (section 4.1) |

## 6. Aggregation (`aggregate-ips-suite-v3.py`, on a laptop)

- `runs.csv`: one row per run with the main numbers
- `configs.csv`: mean and standard deviation over seeds
- `pairs.csv`: 32 hand-picked paired comparisons, plus every ablation
  against the model it was derived from (`_sa`, `_rand`, `_tc`, each value
  of M against M = 10), on the same test images, with a
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
| Which selection module? | cross-attention, self-attention, random, on all 16 selectors | score, lesion hits, Grad-CAM |
| Does training the coarse stage help? | `_tc` against without | score, share of the lesion kept by the zones |
| How many patches are needed, for each family? | mstudy | score and memory against M |
| Few large or many small patches? | equal M × patch² | score at equal memory |
| Does more resolution help? | hires | score at the same memory |
| Which patch size for which lesion size? | patch sizes 28 to 256 on two data sets | score, lesion hits |
| Does it hold with little data? | lowdata | score |
| Do the patches fall on lesions? | all except the baseline | lesion study |
| Do the models look at the lesions? | all, baseline included | Grad-CAM energy, pointing game, pixel AUROC |
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
- **No attention heatmap** for the self-attention and random-selection
  models (their scores depend on the other patches of the group) nor for the
  baseline. Grad-CAM covers every model, these included.
- **Lesion analysis and Grad-CAM metrics only on annotated images**: 185 DDR
  and about 950 RSNA test images (the Grad-CAM maps themselves are saved for
  every test image).
- **Grad-CAM of a selector is zero outside its selected patches** by
  construction; compare it with the baseline knowing that.
- **One seed** outside `core`, unless more are launched.
- **One backbone** (ResNet-18), one training recipe, no hyperparameter search.
- **No external data set.**
- **The referable-DR threshold is not tuned**: a grade of 2 or more is
  predicted from the most probable grade.
- **Energy and CO₂ only where codecarbon is installed** (the cluster, not Kaggle).
- **Memory and time are those of the cluster GPU.**
- **The training-set metrics are computed with augmentation on**, so they
  slightly understate the fit.
- **DDR at 1792 px is cached as JPEG** (quality 95, no chroma subsampling,
  about 15 GB) instead of raw (about 120 GB). The 896 px cache is raw.
