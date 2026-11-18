# Suite v3: design, claims, runs, and what to expect

Written on 15 November 2026 for the authors (Axel and the supervisors).
Companion files, all in `master-research-codes/`:

| File | Role |
|---|---|
| `ips-suite-v3.ipynb` | the suite (66 configurations, 4 modes) |
| `run-ips-suite-v3.sh` | cluster script, same mechanics as `run-ips-suite.sh` |
| `aggregate-ips-suite-v3.py` | tables, seeds, bootstrap, paired comparisons (CPU) |
| `dataset-probe.py` | one Kaggle cell that exports the facts about DDR and RSNA |

**Read section 9 first if you are short on time.** It says what this can
and cannot give.

**Venue warning.** The MIDL 2027 author instructions allow language models
"only for minor language edits" and state that generating "research ideas,
study designs, analyses, results, or substantive text" with one is not
allowed (https://2027.midl.io/author-instructions). This document was
drafted with such a tool. Treat it as an internal working note: every
design choice below has to be examined, changed where you disagree, and
owned by the authors before any of it reaches a submission, and no
sentence of it should be pasted into the paper. See section 10.

> **Update, 16 November 2026** (supervisor's decisions of 15 and 16 November).
> This section supersedes the run lists further down; the notebook
> (`IPS_MODE=list`) and `submit-v3.sh` are the reference.
>
> - **Two patch sizes at most** per model: the three-stage cascade is removed.
> - **IPS at 32 and 64 px on DDR** are in the main group (`core`), with MS-IPS
>   64/32 so that it can be compared with its two single-size parts.
> - **RSNA**: opacity boxes measure 209 ± 63 by 339 ± 149 px, so IPS at
>   256 px (about one opacity) is in `core`, with MS-IPS 256/128.
> - **New baseline**: attention MIL (Ilse et al., 2018) over all patches of
>   128 px, every patch trained, no selection.
> - **New ablation group** (`ablation`): self-attention aggregator and random
>   selection on all 16 selectors of `core` (IPS, MS-IPS, H-MS-IPS), coarse
>   zones of the cascades trained.
> - **M study** on the three families, both data sets, M from 2 to 160.
> - **Grad-CAM** for every model on the test split, with lesion metrics.
> - **DDR at 1792 px is cached** (JPEG, quality 95).
> - **Recorded in addition**: kappa linear, per-grade metrics, referable DR
>   (grade ≥ 2) AUC, sensitivity and specificity, calibration (ECE, Brier,
>   NLL), confusion matrices, 95 % bootstrap intervals on test, parameters
>   and GFLOPs per image, the weight of each selected patch in the decision,
>   the coarse zones of the cascades, and on the test split a score for every
>   candidate patch (heatmaps without rerunning).
> - **Launch**: `bash submit-v3.sh <group> <seed>` submits one job per
>   configuration; a configuration already running in another job is skipped.
>
> Group sizes: core 22, ablation 36, mstudy 54, patch 6, base 2, ms 2,
> lowres 2, hires 4, lowdata 6, misc 2 (136 configurations). The complete
> list of what is measured is `SUITE-V3-contents.md`.

## 1. The question, in one sentence

> With a fixed number of pixels that the network is allowed to train on,
> what is the best way to choose them: shrink the whole image, take patches
> of one size, let several sizes compete, or go from coarse to fine?

The v2 suite could not answer it, because its models did not train on the
same number of pixels. H-MS-IPS 224/56 used a third of the memory of IPS,
but it also trained on a fifth of the pixels (10 patches of 56 against 10
patches of 128). Its lower memory came from the smaller final patch, not
from the cascade. v3 compares models **at the same final patch size and the
same M**, which is the only comparison where the cascade itself is the
variable.

## 2. Notation

| Symbol | Meaning | Value in the runs |
|---|---|---|
| S | side of the square image | 896 (DDR), 1024 (RSNA) |
| C | channels | 3 (DDR), 1 (RSNA) |
| p | side of a patch | 224 down to 28 |
| N_p = (S/p)² | candidates of size p, exact tiling | 16 to 1024 |
| M | patches kept and trained on | 2 to 160, reference 10 |
| I | candidates scored per group | 128 |
| B | images per optimiser step | 16 |
| p_0 > p_1 > ... | stages of a cascade | e.g. 224 then 56 |
| K_s | zones kept after stage s | 4 |
| P_train | pixels that receive gradients, per image | M·p² for IPS |

## 3. The four models

All four share the same ResNet-18 backbone (ImageNet weights), the same
cross-attention aggregator, optimiser, schedule and loss. Only the way the
trained pixels are chosen differs.

**Baseline.** The image, resized to side s, goes through the whole
ResNet-18. P_train = s².

**IPS** (Bergner et al., 2023). One patch size p.
1. Selection, no gradient: the N_p candidates are shuffled, scored I at a
   time by the encoder and the attention module, and the best M are kept.
2. Training step: the M kept patches go through the same encoder and
   attention module with gradients; the attention output is classified.
P_train = M·p².

**MS-IPS** (flat). The candidate list is the union of several sizes. A
learned scale embedding is added to each patch embedding. Every patch is
encoded at its native size; the encoder's average pooling makes the output
size independent of p. One selection, one budget M shared by all sizes.
P_train is between M·p_min² and M·p_max², depending on what is selected.

**H-MS-IPS** (cascade). Stage 0 selects K_0 zones among the N_{p_0}
candidates of the coarsest size. Each next stage cuts the kept zones into
patches of the next size and selects among those only. The last stage keeps
M patches, and only these are trained on. P_train = M·p_last².

Three switches are new in v3 (all off by default, so v2 weights load):

| Switch | Effect | Used for |
|---|---|---|
| `train_coarse = k` | the k best coarse zones are appended to the training bag | claim C5 and hypothesis H3 |
| `encode_px` | patches of a scale are resized before the encoder | resampling control, low-resolution coarse stage |
| `loading = 'lazy'` | the image stays on the CPU | claim C3 |

## 4. Claims: what is proved, what is measured

Each claim is labelled **exact** (follows from counting, checked by a
self-test in the notebook), **fact about the code** (follows from reading
it), or **empirical** (a model to be fitted, with the data that supports
it so far).

### C1. A patch size costs one image (exact)

*Statement.* If p divides S, encoding all candidates of size p reads
N_p·p² = S² pixels. Therefore one selection pass over a set of exactly
tiled sizes reads (number of sizes)·S² pixels, whatever the sizes are.

*Proof.* N_p·p² = (S/p)²·p² = S². Sum over the sizes. ∎

*Check.* 49·128² = 196·64² = 784·32² = 802 816 = 896². Asserted in the
notebook's self-test.

*Consequence.* Adding a size to MS-IPS doubles, then triples, the selection
compute. It does not change P_train. With `encode_px = e` the encoder sees
N_p·e² pixels instead: upsampling 64 px patches to 128 px makes that size
cost four images (the v2 defect, kept as the `_rs` control).

### C2. Cutting the kept zones finer adds candidates, not pixels (exact)

*Statement.* Stage s+1 of a cascade reads K_s·p_s² pixels, independent of
p_{s+1}. Its number of candidates is K_s·(p_s/p_{s+1})².

*Proof.* The K_s kept zones have area p_s² each and are tiled exactly by
patches of size p_{s+1}: the pixels read are those of the zones, and the
count is the area ratio. ∎

*Consequence.* Total selection pixels of a cascade:
S² + Σ_s K_s·p_s². For 224 then anything, with K = 4 on DDR:
802 816 + 200 704 = 1.25 images. **A cascade therefore reads more pixels
during selection than single-scale IPS at its final size (1.00 image).**
The cascade does not save selection compute. With a coarse stage read at
half resolution (`encode_px`), stage 0 costs a quarter of an image and the
total drops to 0.50 image (run `*_h224lr_56_tc`).

### C3. Selection memory does not depend on the number of candidates (fact about the code, to be measured)

*Statement.* During selection the GPU holds, beyond the weights:
the image if loading is eager (C·S²·4 bytes per image), at most M + I
patches, and the activations of at most max(M, I) patches without
gradients. None of these terms contains N.

*Why.* The candidate table holds coordinates only (3 integers each).
Pixels are cut for one group of I at a time and released when the best M
are kept.

*What it does not say.* It is not an equality with (M + I)·C·p²·4 bytes:
the activations of the group, which are several times larger than its
input, are part of the peak. The bound is "independent of N", not "equal
to the input size".

*Check.* `IPS_MODE=bench`: the selection peak at fixed (M, I, p) on an
image twice as large (four times the candidates) must be unchanged with
lazy loading, and larger by exactly the image bytes with eager loading.
The smoke test already confirmed that lazy and eager give identical scores.

### C4. Training memory is an affine function of the trained pixels (empirical)

*Statement.* Peak memory of a run ≈ a + b·B·P_train.

*Evidence so far (v2, full network, B = 16).* A line through IPS and the
full image on DDR gives a ≈ 0.34 GB and b ≈ 0.435 GB per million pixels.
It then predicts the other runs without being fitted on them:

| Run | B·P_train (Mpx) | Predicted | Measured |
|---|---|---|---|
| DDR H-MS-IPS 224/112 | 2.01 | 1.21 GB | 1.21 GB |
| DDR H-MS-IPS 224/56 | 0.50 | 0.56 GB | 0.53 GB |
| RSNA full image | 16.78 | 7.64 GB | 7.67 GB |
| RSNA IPS 128 | 2.62 | 1.48 GB | 1.41 GB |
| RSNA H-MS-IPS 256/64 | 0.66 | 0.62 GB | 0.56 GB |

*Consequences.*
1. Memory is known before training from M and p alone. This is the part of
   "cost known in advance" that the data already supports.
2. Two models with the same M·p² use the same memory. **The memory gap
   between a cascade and IPS in v2 is entirely this effect.**
3. Equal M·p² is the fair axis for comparing "few large" with "many
   small" patches. The `mstudy` group is built on three such budgets.

*Check.* The bench fits a and b on about 60 points; every training run adds
one more.

### C5. In the v2 cascade, the coarse stage is never trained (fact about the code)

*Statement.* In `HMSIPSNet` (v2), the loss gradient with respect to the
scale embedding of every stage but the last is zero, and the encoder never
receives a gradient from a patch of a coarse size.

*Proof.* The loss is a function of the embeddings in the training bag. The
bag contains only last-stage patches. The path from a coarse zone to the
loss goes through `select`, which runs under `torch.no_grad()` and returns
indices (top-k), which carry no gradient. ∎

*Consequence (a fact).* The coarse zones are ranked by a network trained
only on fine patches, plus an embedding that still has its random initial
value.

*Whether this hurts quality is hypothesis H3, not a fact.* Two v2
observations are compatible with it: quality falls steadily as the two
sizes move apart (κ 0.887, 0.836, 0.726, 0.604 for 224 then 112, 56, 28,
14), and the fine cascades cannot fit their training data (final training
loss 0.62 to 0.82 against 0.20). The half-depth encoder used below 56 px is
a competing explanation; v3 removes it by using the full network
everywhere.

The same mechanism exists in flat MS-IPS in a weaker form: a size that is
rarely selected rarely receives gradients. On RSNA the 128 px size took
7 976 of 8 000 selections.

## 5. Hypotheses, the runs that test them, and what each outcome means

| | Hypothesis | Test (a against b) | If true | If false |
|---|---|---|---|---|
| H1 | A coarse stage improves selection at a fixed final patch | `h224_56` vs `ips56`, `h224_112` vs `ips112`, `h256_64` vs `ips64` | the cascade has a reason to exist: better score at equal memory | the cascade costs 25 % more selection compute for nothing |
| H2 | Selecting patches beats shrinking the image at equal memory | `ips128` vs `base448` (DDR), vs `base512` (RSNA) | patch selection is justified on this data | the data set does not need high resolution; nothing built on IPS can win there |
| H3 | The cascade is held back because its coarse stage is untrained | `*_tc` vs the same without; zone lesion recall before and after | a real, explainable fix; the paper has a method contribution | C5 is true but harmless; drop the cascade |
| H4 | At equal trained pixels, many small patches beat few large ones when lesions are tiny (DDR), and the reverse when they are large (RSNA) | `ips64_m40` and `ips32_m160` vs `ips128` | patch size should follow lesion size, shown without any multi-scale model | the lesion-size argument of the draft has no support |
| H5 | The score saturates in M | `ips128_m2/5/10/20`, `ips64_m10/20/40` | M can be cut, with a known memory gain | more patches keep helping: report the slope |
| H6 | With little data, smaller patches generalise better (Riffi-Aslett and Fell, on synthetic data and traffic signs) | `*_f25` runs | a replication on medical images | their finding does not transfer |
| H7 | Resampling patches to a common size biases the selection | `ms128_64_rs` vs `ms128_64`: scale counts and score | the v2 correction was needed; one sentence in the paper | the correction was cosmetic |
| H8 | Lazy loading changes memory, not results | `ips128_lazy` vs `ips128`, and the bench | C3 measured | a bug |
| H9 | On DDR, IPS at 1792 px beats IPS at 896 px, and reaches or beats the full image at 896 px, at the memory of IPS at 896 px | `r1792_ips128` vs `ips128` and vs `base896` | the case for patch selection on medical images, shown directly: resolution the full image cannot afford | DDR grading does not need more than 896 px with this backbone |

H1, H2 and H9 decide what the paper is about. Everything else refines the
picture. H9 was added after the dataset probe (section 7.1).

## 6. What every run records

| Recorded | Why |
|---|---|
| All metrics on train and validation at every epoch | curves, overfitting |
| Best epoch on validation by the main metric; all metrics at that epoch | ends the "maximum of each metric" artefact |
| **Test** metrics with the weights of that epoch | the reported number |
| Binary: threshold fixed on validation, sensitivity and specificity on test | an honest recall |
| Per image, validation and test: probabilities and selected patches (`predictions/*.npz`) | any metric, threshold, calibration, bootstrap, figure, later, without a GPU |
| Time per epoch split into data, selection, training step, GPU synchronised | time that describes the model |
| Peak memory per phase (selection, step), training and inference | C3, C4, the frontier figure |
| Inference latency per image | deployment cost |
| Exact pixel counts (selection read and encoded, trained) | C1, C2 |
| Selection study on test: uniform, scale-matched, learned | worth of positions and of scales |
| Lesion overlap on test and validation, per lesion type, against exact chance; for cascades, share of the lesion inside the kept zones | interpretability; direct evidence for H3 |
| Scale counts over the whole test set | which size is used |
| Energy and CO2 (codecarbon) | only meaningful once data time is small |

Chance in the lesion study is exact: all candidates of the same size are
enumerated, nothing is sampled.

Reproducibility changes: the seed is in the run name; the RSNA split uses a
fixed constant, not the run seed (in v2 it used `conf.seed`, which would
have changed the split with the seed); the candidate shuffle has its own
generator, reseeded before each evaluation, so an evaluation is repeatable.

## 7. Data splits

| | Train | Validation (choose epoch) | Test (report) |
|---|---|---|---|
| DDR | `train` | `valid` | `test`, never used in v2 |
| RSNA | 70 % of patients | 15 % | 15 % |

RSNA is split 70 / 15 / 15 by a hash of the patient identifier (decided
on 15 November, for a standard split). The v3 training set is a subset of
the v2 one. v2 weights chose their epoch on images now in validation and
test, and were trained on 5 % of the patients now in validation or test,
so on RSNA only v3 runs have a clean test score; `eval_v2` output for RSNA
is flagged as not clean. On DDR, v2 weights can be scored cleanly on test
right now (`IPS_MODE=eval_v2`).

### 7.1 What the dataset probe showed (15 November, real data)

Source: `results-done-right-I-guess/dataset_probe.zip`, produced by
`dataset-probe.py` on Kaggle. All numbers below are measured.

**DDR images.**

| Quantity | Value |
|---|---|
| Images (train / valid / test, with grade 5) | 6 835 / 2 733 / 4 105 |
| Ungradable (grade 5), excluded | 575 / 230 / 346 |
| Width | 702 to 5 184 px, median 2 592 |
| Height | 706 to 3 456 px, median 1 824 |
| Megapixels | 0.5 to 17.9, median 4.5 |
| Distinct sizes | 132; most common 2592×1728 and 2736×1824 |
| Aspect ratio | 0.98 to 1.76, median 1.33 |
| Pixels kept at 896×896 | median 18 % |
| Share of the photograph that is retina (not black) | median 54 % |
| Retina bounding box, width over height | median 1.00 (a disc) |
| Decode and resize, one image, one core | median 112 ms |

Two consequences.

1. *Geometry.* The retina is a disc inside a wider black frame. Stretching
   the frame to a square turns the disc into an ellipse and spends about
   half of the 896×896 pixels on black. `IPS_DDR_GEOMETRY=retina` crops to
   the retina and keeps pixels square: on a synthetic check the retina
   fills 78 % of the square instead of 36 %, and a square lesion stays
   square. A centred crop to the short side would cut the retina on about
   13 % of images, so `crop` is not recommended. **Recommendation: run v3
   with `retina`.** It is not comparable with v2, but v3 reruns every
   reference model anyway.
2. *Speed.* 112 ms per image on one core, with 2 workers in v2, accounts
   for a large part of the 14 minutes per epoch. The cache removes it.

**DDR lesions** (608 annotated images read, 42 783 connected components;
sizes after the stretch to 896 px):

| Type | Images | Lesions per image, median | Diameter, median (p95) | Fit inside a 14 px patch | inside 28 px | inside 56 px |
|---|---|---|---|---|---|---|
| Micro-aneurysm | 438 | 9 | 3.2 px (6.7) | 99.3 % | 100 % | 100 % |
| Hard exudate | 416 | 25 | 3.4 px (13.6) | 89.3 % | 97.4 % | 99.5 % |
| Haemorrhage | 488 | 11 | 6.3 px (23.9) | 73.4 % | 91.9 % | 98.1 % |
| Soft exudate | 153 | 3 | 12.3 px (33.7) | 42.8 % | 70.3 % | 96.3 % |

Lesions cover 0.19 % of the image (median; mean 0.69 %; mean 0.85 % on the
67 annotated images of the grading test split, which matches the earlier
0.84 %).

Three consequences.

1. At 896 px a micro-aneurysm is about 3 pixels wide. The backbone
   downsamples by 32. The "ladder from lesion size" of the draft would ask
   for patches of 8 to 16 px, which v2 showed cannot be trained. The lesion
   scale and the trainable patch scale do not meet at this resolution.
2. At native resolution the same micro-aneurysm is 7.8 px wide (median).
   The information was removed by the resize, before any model. **The
   experiment that patch selection exists for was never run: more
   resolution at the same memory.** The `hires` group does it (DDR at
   1792 px, same M and patch size, hence the memory of the 896 px runs by
   C3 and C4).
3. The annotated validation images live in a folder named
   `segmentation label`, not `label`. Earlier analyses and this probe read
   608 annotated images; the data set has 757. v3 reads both folder names.
   How many of the extra 149 fall in the grading test split is not known
   yet.

Annotated images by grading split (of the 608): train 427, valid 112,
test 67, absent 2.

**RSNA.**

| Quantity | Value |
|---|---|
| Patients (one image each, all 1024×1024, 8 bit) | 26 684 |
| Classes | Lung Opacity 6 012 patients; Normal; No Lung Opacity / Not Normal |
| v3 split | 70 / 15 / 15 by patient: about 18 700 / 4 000 / 4 000; exact counts are printed at the start of every RSNA run (the probe was run with the earlier 75 / 12 / 13 rule: 20 005 / 3 127 / 3 552) |
| Boxes | 9 555; one box 2 614 patients, two 3 266, three 119, four 13 |
| Box short side | median 208 px (5 % under 109 px) |
| Box long side | median 304 px |
| Boxes that fit in a 128 px patch / 256 px patch | 2.1 % / 36.2 % |
| Union of boxes per positive image | median 7.9 % (first 3 000 patients) |
| `Training/Masks` | the boxes drawn as images (IoU 1.00 with the boxes, empty on negatives), not lung masks |

Consequence: on RSNA every patch size we use is smaller than the lesion,
and the lesion covers a large part of the image. This is the opposite
regime from DDR, which is useful for H4, and it is also why a downsampled
image is expected to do well there (H2).

## 8. The runs

`IPS_MODE=list` prints all 66 with their exact costs.

| Tier | Group | Configs | Seeds | Runs | Purpose |
|---|---|---|---|---|---|
| 0 | `bench` | (no data) | | | C3, C4: memory and latency, a few minutes |
| 0 | `eval_v2` | 21 v2 weights | | | clean DDR test scores and lesion overlap for the existing models, no training |
| 1 | `core` | 17 (9 DDR, 8 RSNA) | 3 | 51 | H1, H2, H3 |
| 2 | `patch` + `mstudy` + `base` + `ms` + `lowres` | 43 more | 1 | 43 | H4, H5, H7, the single-scale sweep |
| 1 | `hires` | 4 (DDR at 1792 px) | 1, then 3 if it helps | 4 | H9: more resolution at the same memory |
| 3 | `lowdata`, `misc` | 8 | 1 (3 for `lowdata` if H6 looks true) | 8 | H6, H8 |

Core runs (one line each; M = 10 everywhere):

| DDR | RSNA | Role |
|---|---|---|
| `ddr_base896` | `rsna_base1024` | full image |
| `ddr_base448` | `rsna_base512` | H2: image at half resolution, memory close to IPS |
| `ddr_ips128` | `rsna_ips128` | IPS reference |
| `ddr_ips112`, `ddr_ips56` | `rsna_ips64` | H1: IPS at the cascade's final size |
| `ddr_h224_112`, `ddr_h224_56` | `rsna_h256_128`, `rsna_h256_64` | H1: the cascades |
| `ddr_h224_112_tc`, `ddr_h224_56_tc` | `rsna_h256_128_tc`, `rsna_h256_64_tc` | H3: coarse zones trained |

Note on `_tc`: training on 4 zones of 224 px costs 200 704 pixels per
image, more than IPS itself. These runs test H3 cleanly; they are not a
memory-efficient model. If H3 holds, the efficient version is the
low-resolution coarse stage (`*lr_*_tc`, 10 % of the image).

**Compute.** In v2 a run took about 14 h on DDR and 11 to 14 h on RSNA;
runs with over 1 000 candidates took 36 to 46 h. The 21 runs cost about
350 GPU-hours and finished in about three days, so roughly five GPUs were
in use. At that speed:

| Tier | GPU-hours (upper bound) | Days on five GPUs |
|---|---|---|
| 1, core | about 660 | 5 to 6 |
| 2 | about 700 | 6 |
| 3 | about 80 | under 1 |

v3 should be faster on DDR: v2 used 2 data workers while each job reserved
16 CPUs, and decoded every 4.7-megapixel image again at every epoch. v3
uses 12 workers and a decoded-image cache. How much faster is unknown until
the first job prints its per-epoch line, which shows data, selection and
step time separately. **Launch one `IPS_EPOCHS=1` job first and read that
line.**

RSNA overfits (v2 best epochs 11 to 20 of 50, final AUC about 0.025 below
the best). Fewer epochs there would save a third of the compute, but it
changes the learning-rate schedule. Authors' call; the suite takes
`IPS_EPOCHS`.

**Order of launch.**
1. `bench`, `eval_v2`, and one `IPS_EPOCHS=1` dry run. Same day.
2. `core`, seed 42 (17 jobs). Look at H1 and H2 on one seed.
3. If H1 or H3 shows a gap above 0.01: seeds 43 and 44 for `core`.
   If not: seeds only for the IPS and baseline rows, and move the compute
   to `mstudy` and `patch`, which is where the paper will be.
4. Tier 2, then tier 3.

**Decision rule for "a real difference"** (proposed, to be agreed before
looking at results): the paired bootstrap interval on the test images
excludes zero for every seed, and the mean difference exceeds twice the
standard deviation over seeds. The aggregate script prints both.

## 9. What we can get out of this

These are judgements, not results. Probabilities are my rough estimates.

**What the existing evidence already says.**
- On RSNA, the full image, IPS and every reasonable variant land within
  0.003 AUC of each other. Pneumonia opacities are large (median box union
  8.7 % of the image). It is likely (I would say four chances in five) that
  the image at 512 px, or even 256 px, reaches the same AUC. If so, RSNA
  cannot show a benefit of any high-resolution method. It can only show
  that memory can be cut for free, and a downsampled image will show the
  same thing more simply.
- On DDR, resolution matters (the master thesis sweep, old code: κ 0.790 at
  100 px, 0.906 at 896 px), and IPS loses 0.02 κ against the full image for
  a quarter of the memory. DDR is where a method could show something.
- Prior work already reports that small patches in plain IPS cost less
  memory and can score as well or better (Bergner et al., Table 4;
  Riffi-Aslett and Fell, Tables 4 and 6).

**Scenarios after the core runs.**

| Scenario | My estimate | What it gives |
|---|---|---|
| A. The cascade (as is or with trained coarse zones) beats IPS at the same final patch on DDR by a margin clear of noise | about 1 in 4 | a method paper: "coarse-to-fine selection improves the memory and quality trade-off", with C5 as the explanation |
| B. Cascade and IPS tie at every matched size | about 1 in 2 | no method contribution. A controlled study: patch size and M set the cost, the cost is predictable (C1 to C4), multi-scale selection adds nothing on these data |
| C. IPS at small patches beats the cascade | about 1 in 4 | same study, with a clear negative result on cascades |

Separately, about 2 chances in 5 that `ddr_base448` matches `ddr_ips128`.
If it does, the honest headline on DDR is that half resolution is enough
for grading at this backbone size, and selection only pays below that
memory.

**Is it a waste of time?**
- **Yes, if the goal stays "show that MS-IPS or H-MS-IPS beats IPS".** The
  21 corrected runs do not point that way, and running more cascade
  variants without the matched controls would repeat the v2 situation.
- **No, if the goal is to answer section 1.** Every outcome of the core
  group is informative, the matched design is what was missing, and the
  measurement side (test set, seeds, per-image outputs, lesion overlap on
  the trained models) is needed for the master thesis and for any venue.
  Tiers 0 and 1 are worth running whatever happens next.
- The parts most likely to survive as contributions, whatever H1 says:
  the cost rules C1 to C4 with measurements, the equal-pixel study of patch
  size against M on two medical data sets with opposite lesion sizes (H4,
  H5), and the lesion-overlap analysis. This is an empirical study, not a
  new method.

**After the dataset probe.** The most promising run in the whole suite is
now `ddr_r1792_ips128` (H9). If IPS at 1792 px beats the full image at
896 px for a quarter of its memory, the paper has a clear and useful
message on a medical task, even though the method is IPS itself; the
cascade and the equal-pixel study then become the analysis of how to spend
that memory. If it does not, DDR grading with ResNet-18 is saturated at
896 px and no selection method will show a gain there. Either answer is
worth one run. I would put the chance of a gain at about one in two: the
master thesis sweep was still rising at 896 px, but grading rests on more
than the smallest lesions.

The `hires` runs read 1792 px images: the decoded cache would take about
120 GB, so run them with `IPS_CACHE=none` and 12 workers.

**What would make it stronger, not in v3.** A third data set where
resolution clearly matters and lesions are small (the abandoned LUNA16 and
NODE21 work was aimed at this). Without it, the DDR result stands alone.

## 10. Venue and calendar

- MIDL 2027 dates (https://2027.midl.io/important-dates): abstract
  registration 30 November 2026, full paper 4 December 2026, both anywhere
  on Earth. Today is 15 November.
- The core group alone needs 5 to 6 days of cluster time at v2 speed. It
  fits before 4 December only if it is launched within a few days and
  nothing has to be rerun. There is no room for tier 2 with seeds.
- MIDL 2027 has three tracks: main (methods, 10 pages), Clinical
  Translation and Validation (14 pages, novelty not required, negative
  results accepted only if grounded, significant and surprising), and
  short papers (3 pages, not in the proceedings, deadline not announced,
  open to negative and replication results).
- The policy on language models quoted at the top applies to all three. It
  is the authors' and the supervisors' decision how to proceed: write the
  study design and the text themselves and use tools only for language
  edits, or choose a venue whose policy is disclosure-based. I should not
  be the one writing the paper's text for MIDL.

## 11. How to run

```bash
# list, with exact costs (no GPU, no data)
IPS_MODE=list bash run-ips-suite-v3.sh

# tier 0
IPS_MODE=bench   sbatch -J v3_bench  run-ips-suite-v3.sh
IPS_MODE=eval_v2 sbatch -J v3_evalv2 run-ips-suite-v3.sh

# dry run, one epoch: read the "[data .. select .. step ..]" line
IPS_EPOCHS=1 sbatch -J v3_try run-ips-suite-v3.sh ddr_ips56

# tier 1, one job per configuration and seed (loop in the script header)
IPS_SEEDS=42 sbatch -J v3_ddr_ips56_s42 run-ips-suite-v3.sh ddr_ips56

# tables
python aggregate-ips-suite-v3.py ips-project/results/ips-suite-v3
```

A finished run is skipped and an interrupted run resumes, so resubmitting
is safe. On Kaggle the notebook runs as is; set the variables with
`os.environ[...]` in a first cell (default: `core`, seed 42).

## 12. What was tested, and what was not

Tested on this machine, CPU only, on small fake data sets with the real
folder layout: all four modes; training, resume after a kill, skip of a
finished run; cache; test evaluation; selection study; lesion study with
masks (DDR layout) and boxes (RSNA layout); cascades with and without
trained coarse zones; resized scales; lazy loading (identical scores to
eager); 25 % training fraction; the retina crop (image and mask stay
aligned); a 1792 px run; loading v2-style weights; the aggregate
script.

Not tested: anything on a GPU (memory figures were all zero here), the real
data sets, the pretrained weights, the cluster script end to end. The
dry-run job is there to catch what a CPU test cannot.
