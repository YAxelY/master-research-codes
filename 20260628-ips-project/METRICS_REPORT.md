# IPS Project: Experiment Metrics Report

---

## Single Overview Table (all runs)

| Dataset    | Method                   | Encoder  | Ep  | Acc        | **κ/AUC**  | F1         | Train VRAM (MB) | Train Time (s) | Val VRAM (MB) | Val Time (s) | Total CO₂ (g) | CO₂/ep (g) |
| ---------- | ------------------------ | -------- | --- | ---------- | ---------- | ---------- | --------------- | -------------- | ------------- | ------------ | ------------- | ---------- |
| **DDR**    | Baseline (full-img 896)  | resnet18 | 50  | **0.8937** | **0.9261** | **0.7270** | 7231.7          | 631.0          | 3212.4        | 73.3         | 85.18         | 1.704      |
| **DDR**    | IPS+self-attn            | resnet18 | 50  | 0.8602     | 0.8960     | 0.6807     | 2094.5          | 606.2          | 1135.3        | 72.8         | 76.92         | 1.538      |
| **DDR**    | MS-IPS                   | resnet18 | 50  | 0.8650     | 0.8889     | 0.6809ᵃ    | 2928.1          | 611.8          | 2922.3        | 85.8         | 104.89        | 2.098      |
| **DDR**    | IPS                      | resnet18 | 50  | 0.8510     | 0.8842     | 0.6875     | 2098.7          | 600.0          | **707.2**     | 73.8         | 76.65         | 1.533      |
| **DDR**    | H-MS-IPS                 | resnet18 | 50  | 0.8466     | 0.8731     | 0.6650     | **1537.9**      | 608.5          | 1535.5        | 73.0         | 74.85         | 1.497      |
| **DDR**    | CE                       | resnet50 | 50  | 0.8470     | 0.8705     | 0.6412     | 7560.1          | **203.0**      | 1316.3        | **54.5**     | **29.57**     | **0.591**  |
| **DDR**    | QMIX                     | resnet50 | 50  | 0.7551     | 0.6901     | 0.3445     | 8348.9          | 543.9          | 2105.1        | 74.6         | 166.13        | 3.323      |
|            |                          |          |     |            |            |            |                 |                |               |              |               |            |
| **NODE21** | Baseline (full-img 1024) | resnet18 | 50  | **1.0000** | **1.0000** | **1.0000** | 8698.9          | 224.1          | 3444.5        | 44.5         | 26.37         | 0.527      |
| **NODE21** | H-MS-IPS                 | resnet18 | 50  | **1.0000** | **1.0000** | **1.0000** | 1994.5          | 191.2          | 1573.9        | **37.7**     | **19.84**     | **0.397**  |
| **NODE21** | IPS+self-attn            | resnet18 | 50  | **1.0000** | **1.0000** | **1.0000** | **1965.4**      | 220.4          | **898.2**     | 43.2         | 22.13         | 0.443      |
| **NODE21** | MS-IPS                   | resnet18 | 50  | **1.0000** | **1.0000** | **1.0000** | 1980.4          | **191.6**      | 1660.9        | 41.6         | 47.90         | 0.958      |
| **NODE21** | IPS                      | resnet18 | 50  | 0.9990     | **1.0000** | 0.9977     | 1968.7          | 206.5          | 901.9         | 41.1         | 22.26         | 0.445      |
|            |                          |          |     |            |            |            |                 |                |               |              |               |            |
| **RSNA**   | IPS                      | resnet18 | 50  | 0.8149     | **0.8723** | 0.7488     | **2168.7**      | 334.0          | 902.1         | 85.1         | 81.60         | 1.632      |
| **RSNA**   | IPS+self-attn            | resnet18 | 50  | 0.8212     | 0.8713     | 0.7441     | 2065.4          | 334.0          | **898.2**     | 85.1         | 82.08         | 1.642      |
| **RSNA**   | MS-IPS                   | resnet18 | 50  | 0.8278     | 0.8893     | **0.7510** | 2432.4          | 859.3          | 1661.3        | 263.7        | 260.23        | 5.205      |
| **RSNA**   | H-MS-IPS                 | resnet18 | 50  | 0.8229     | 0.8691     | 0.6219     | 2000.9          | **250.3**      | 1577.7        | **66.6**     | **72.49**     | **1.450**  |
| **RSNA**   | Baseline (full-img 1024) | resnet18 | 50  | **0.8335** | 0.8655     | 0.6249     | 8881.8          | 339.3          | 3629.2        | **44.8**     | 91.39         | 1.828      |
|            |                          |          |     |            |            |            |                 |                |               |              |               |            |
| **LUNA16** | IPS                      | resnet50 | 50  | **0.8090** | **0.8080** | **0.8811** | 432.1           | **235.8**      | 442.9         | **22.0ᶜ**    | **41.63**     | **0.833**  |
| **LUNA16** | IPS+self-attn            | resnet50 | 50  | 0.7753     | 0.7110     | 0.8611     | **431.9**       | 496.2          | **442.8**     | 56.3         | 56.06         | 1.121      |

---

## 1. IPS (baseline IPS) — `outputs/ips/`

| Run              | Dataset | Encoder  | Best Val Acc | Best κ/AUC    | Best Val F1 | Train VRAM (MB) | Train Time (s) | Val VRAM (MB) | Val Time (s) | Total CO₂ (g) | CO₂/epoch (g) |
| ---------------- | ------- | -------- | ------------ | ------------- | ----------- | --------------- | -------------- | ------------- | ------------ | ------------- | ------------- |
| ips_ddr_v1       | DDR     | resnet18 | 0.8610       | 0.8842 (κ)ᵇ   | 0.6875      | 2098.7          | 600.0          | 707.2ᵃ        | 73.8         | 76.65         | 1.533         |
| ips_node21_v1    | NODE21  | resnet18 | 0.9990       | 1.0000 (AUC)  | 0.9977      | 1968.7          | 206.5          | 901.9         | 41.1         | 22.26         | 0.445         |
| ips_rsna_v1      | RSNA    | resnet18 | 0.8149ᵃ      | 0.8723 (AUC)ᵃ | 0.7488      | 1603.7ᵃ         | 334.0          | 902.1         | 85.1         | 81.60         | 1.632         |
| ips_luna16_fold0 | LUNA16  | resnet50 | 0.8090       | 0.8080 (AUC)  | 0.8811      | 432.1           | 235.8ᶜ         | 442.9         | 22.0ᶜ        | 41.63         | 0.833         |

*LUNA16 also: best val precision 0.9211, recall 0.9552.*

---

## 2. ResNet18 Baseline (full-image) — `results/baseline-mps-gx/`

| Run                      | Dataset | Img Size | Best Val Acc | Best κ/AUC   | Best Val F1 | Train VRAM (MB) | Train Time (s) | Val VRAM (MB) | Val Time (s) | Total CO₂ (g) | CO₂/epoch (g) |
| ------------------------ | ------- | -------- | ------------ | ------------ | ----------- | --------------- | -------------- | ------------- | ------------ | ------------- | ------------- |
| baseline_ddr_resnet18    | DDR     | 896      | 0.8937       | 0.9261 (κ)   | 0.7270      | 7231.7          | 631.0          | 3212.4        | 73.3         | 85.18         | 1.704         |
| baseline_node21_resnet18 | NODE21  | 1024     | 1.0000       | 1.0000 (AUC) | 1.0000      | 8698.9          | 224.1          | 3444.5        | 44.5         | 26.37         | 0.527         |
| baseline_rsna_resnet18   | RSNA    | 1024     | 0.8335       | 0.8655 (AUC) | 0.6249      | 8881.8          | 339.3          | 3629.2        | 44.8         | 91.39         | 1.828         |

*~3–4× higher VRAM than IPS, driven by full-resolution images.*

---

## 3. Baseline Patch-Size Sweep (DDR, resnet18) — `results/baseline-patch-size/`

> 25 epochs each. These CSVs have **no CO₂ columns**.

| Img Size | Best Val Acc | Best Val κ | Best Val F1 | Train VRAM (MB) | Train Time (s) | Val VRAM (MB) | Val Time (s) |
| -------- | ------------ | ---------- | ----------- | --------------- | -------------- | ------------- | ------------ |
| 100      | 0.7843       | 0.7903     | 0.5034      | 665.3           | 150.7          | 610.2         | 54.5         |
| 224      | 0.8278       | 0.8620     | 0.5879      | 747.2           | 199.4          | 480.8         | 61.8         |
| 448      | 0.8626       | 0.9023     | 0.6457      | 1828.6          | 314.7          | 840.3         | 71.5         |
| 896      | 0.8690       | 0.9063     | 0.6756      | 6185.2          | 845.1          | 2264.7        | 106.8        |

*Clear accuracy↑ / cost↑ trade-off with resolution. (img-448 had a duplicate identical CSV; img-896 also had a `(2)` copy — both ignored.)*

---

## 4. H-MS-IPS (Hierarchical Multi-Scale IPS) — `results/h-ms-ips/`

| Run              | Dataset | Coarse/Fine | Best Val Acc | Best κ/AUC   | Best Val F1 | Train VRAM (MB) | Train Time (s) | Val VRAM (MB) | Val Time (s) | Total CO₂ (g) | CO₂/epoch (g) |
| ---------------- | ------- | ----------- | ------------ | ------------ | ----------- | --------------- | -------------- | ------------- | ------------ | ------------- | ------------- |
| hmsips_ddr_v1    | DDR     | 224/112     | 0.8466       | 0.8731 (κ)   | 0.6650      | 1537.9          | 608.5          | 1535.5        | 73.0         | 74.85         | 1.497         |
| hmsips_node21_v1 | NODE21  | 256/128     | 1.0000       | 1.0000 (AUC) | 1.0000      | 1994.5          | 191.2          | 1573.9        | 37.7         | 19.84         | 0.397         |
| hmsips_rsna_v1   | RSNA    | 256/128     | 0.8229       | 0.8691 (AUC) | 0.6219      | 2000.9          | 250.3          | 1577.7        | 66.6         | 72.49         | 1.450         |

---

## 5. IPS + Self-Attention — `results/ips-self-attention/exp_001/`

| Run              | Dataset | Encoder  | Best Val Acc | Best κ/AUC   | Best Val F1 | Train VRAM (MB) | Train Time (s) | Val VRAM (MB) | Val Time (s) | Total CO₂ (g) | CO₂/epoch (g) |
| ---------------- | ------- | -------- | ------------ | ------------ | ----------- | --------------- | -------------- | ------------- | ------------ | ------------- | ------------- |
| ips_ddr_v1       | DDR     | resnet18 | 0.8602       | 0.8960 (κ)   | 0.6807      | 2094.5          | 606.2          | 1135.3        | 72.8         | 76.92         | 1.538         |
| ips_node21_v1    | NODE21  | resnet18 | 1.0000       | 1.0000 (AUC) | 1.0000      | 1965.4          | 220.4          | 898.2         | 43.2         | 22.13         | 0.443         |
| ips_rsna_v1      | RSNA    | resnet18 | 0.8212       | 0.8713 (AUC) | 0.7441      | 1965.4          | 334.0          | 898.2         | 85.1         | 82.08         | 1.642         |
| ips_luna16_fold0 | LUNA16  | resnet50 | 0.7753       | 0.7110 (AUC) | 0.8611      | 431.9           | 496.2          | 442.8         | 56.3         | 56.06         | 1.121         |

---

## 6. Master2 — QMIX vs Standard CE (DDR, resnet50) — `results/master2-qmix-standard-ce/`

| Run         | Mode | Best Val Acc | Best Val κ | Best Val F1 | Train VRAM (MB) | Train Time (s) | Val VRAM (MB) | Val Time (s) | Total CO₂ (g) | CO₂/epoch (g) |
| ----------- | ---- | ------------ | ---------- | ----------- | --------------- | -------------- | ------------- | ------------ | ------------- | ------------- |
| ce_ddr_v1   | ce   | 0.8470       | 0.8705     | 0.6412      | 7560.1          | 203.0          | 1316.3        | 54.5         | 29.57         | 0.591         |
| qmix_ddr_v1 | qmix | 0.7551       | 0.6901     | 0.3445      | 8348.9          | 543.9          | 2105.1        | 74.6         | 166.13        | 3.323         |

*QMIX underperformed CE and cost ~5.6× the CO₂ here (final-epoch metrics collapsed — likely training instability).*

---

## 7. MS-IPS (Multi-Scale IPS) — `results/ms-ips-op/`

| Run             | Dataset | Coarse/Fine | Best Val Acc | Best κ/AUC   | Best Val F1 | Train VRAM (MB) | Train Time (s) | Val VRAM (MB) | Val Time (s) | Total CO₂ (g) | CO₂/epoch (g) |
| --------------- | ------- | ----------- | ------------ | ------------ | ----------- | --------------- | -------------- | ------------- | ------------ | ------------- | ------------- |
| msips_ddr_v1    | DDR     | 128/64      | 0.8550       | 0.8889 (κ)   | 0.6809ᵃ     | 2928.1          | 611.8          | 2922.3        | 85.8         | 104.89        | 2.098         |
| msips_node21_v1 | NODE21  | 128/64      | 1.0000       | 1.0000 (AUC) | 1.0000      | 1980.4          | 191.6          | 1660.9        | 41.6         | 47.90         | 0.958         |
| msips_rsna_v1   | RSNA    | 128/64      | 0.8278       | 0.8693 (AUC) | 0.7510      | 1980.4          | 859.3          | 1661.3        | 263.7        | 260.23        | 5.205         |

---

## Cross-method summary (best per dataset)

### DDR (κ = headline metric)

| Method                  | Best Val Acc | Best Val κ | Train VRAM (MB) | Total CO₂ (g) |
| ----------------------- | ------------ | ---------- | --------------- | ------------- |
| Baseline resnet18 (896) | **0.8937**   | **0.9261** | 7231.7          | 85.18         |
| IPS + self-attn         | 0.8602       | 0.8960     | 2094.5          | 76.92         |
| MS-IPS                  | 0.8550       | 0.8889     | 2928.1          | 104.89        |
| IPS                     | 0.8610       | 0.8842ᵇ    | 2098.7          | 76.65         |
| H-MS-IPS                | 0.8466       | 0.8731     | 1537.9          | 74.85         |
| CE (resnet50)           | 0.8470       | 0.8705     | 7560.1          | 29.57         |
| QMIX (resnet50)         | 0.7551       | 0.6901     | 8348.9          | 166.13        |

### RSNA (AUC)

| Method                   | Best Val Acc | Best Val AUC | Total CO₂ (g) |
| ------------------------ | ------------ | ------------ | ------------- |
| IPS                      | 0.8149       | **0.8723**   | 81.60         |
| IPS + self-attn          | 0.8212       | 0.8713       | 82.08         |
| MS-IPS                   | 0.8278       | 0.8693       | 260.23        |
| H-MS-IPS                 | 0.8229       | 0.8691       | 72.49         |
| Baseline resnet18 (1024) | 0.8335       | 0.8655       | 91.39         |

### NODE21

All methods reach ≈1.0 accuracy/AUC (saturated, easy task). Cheapest = H-MS-IPS (19.84 g CO₂); baseline @ 1024px hit perfect 1.0 across the board.

---
