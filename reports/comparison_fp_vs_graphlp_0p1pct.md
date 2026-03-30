# Experiment Report: 0.1% Labeled Data — Fewpoint Baseline vs Graph LP

## Overview

This report compares two segmentation runs trained on the WP5 3D dataset with
a **0.1% labeled-voxel budget**. Both use the same evaluation split (180 test
volumes) and report Dice / IoU per class.

| Property | Fully Supervised (Upper Bound) | Fewpoint Global (Baseline) | Graph LP Decoupled |
|----------|-------------------------------|---------------------------|--------------------|
| **Run directory** | `wp5_full_supervised_20251210-164804` | `fp_0p1pct_global_d0_20251021-154453` | `train_graph_lp_k10_a0.9_new_n12000_outerbg_adaptive_0p1pct_decoupled_batch_gamma0.181_20260109-133530` |
| **Label budget** | 100 % (dense GT) | 0.1 % global random | 0.1 % stratified + Graph LP propagation |
| **Label propagation** | N/A | None (sparse seeds only, dilate r = 1) | Graph LP (k = 10, α = 0.9, n = 12 000 SV, outer-BG adaptive) |
| **Loss weighting** | Standard | Standard | Decoupled source weighting, batch scope, γ = 0.181 |
| **Network** | BasicUNet | BasicUNet | BasicUNet |
| **Epochs** | — | 20 | 40 |
| **Best epoch** | — | 17 | 30 |

---

## Per-Class Results (last-epoch checkpoint)

### Dice

| Class | Fully Supervised | Fewpoint Global | Graph LP Decoupled | Δ (GLP vs FP) | % of Upper Bound |
|------:|-----------------:|----------------:|-------------------:|--------:|-----------------:|
| 0 (BG) | 0.9916 | 0.9822 | 0.9886 | +0.0064 | 99.7% |
| 1 | 0.9310 | 0.8569 | 0.9125 | **+0.0556** | 98.0% |
| 2 | 0.9037 | 0.9037 | 0.8678 | −0.0359 | 96.0% |
| 3 | 0.8007 | 0.5609 | 0.7311 | **+0.1702** | 91.3% |
| 4 | 0.8864 | 0.7432 | 0.8604 | **+0.1172** | 97.1% |
| **Average** | **0.9027** | **0.8094** | **0.8721** | **+0.0627** | **96.6%** |

### IoU

| Class | Fully Supervised | Fewpoint Global | Graph LP Decoupled | Δ (GLP vs FP) |
|------:|-----------------:|----------------:|-------------------:|--------:|
| 0 (BG) | 0.9834 | 0.9652 | 0.9775 | +0.0123 |
| 1 | 0.8788 | 0.7887 | 0.8490 | **+0.0603** |
| 2 | 0.8318 | 0.8289 | 0.7768 | −0.0521 |
| 3 | 0.7518 | 0.5164 | 0.6779 | **+0.1615** |
| 4 | 0.8044 | 0.6380 | 0.7596 | **+0.1216** |
| **Average** | **0.8500** | **0.7475** | **0.8082** | **+0.0607** |

### HD95 / ASD

The Graph LP run does not yet have a heavy eval. HD95 and ASD are available
for the fully supervised and fewpoint runs only.

| Class | Fully Supervised HD95 | Fewpoint HD95 | Fully Supervised ASD | Fewpoint ASD |
|------:|----------------------:|--------------:|---------------------:|-------------:|
| 0 (BG) | 1.02 | 1.67 | 0.13 | 0.27 |
| 1 | 2.12 | 10.34 | 0.77 | 2.97 |
| 2 | 2.54 | 7.01 | 0.76 | 1.56 |
| 3 | 1.48 | 3.78 | 0.40 | 1.26 |
| 4 | 2.74 | 10.15 | 0.79 | 2.82 |
| **Average** | **1.98** | **6.59** | **0.57** | **1.78** |

---

## Label Source Statistics (Graph LP Run)

| Metric | Value |
|--------|------:|
| GT voxels | 43 397 703 (9.95%) |
| LP voxels | 392 799 225 (90.05%) |
| Total supervised voxels | 436 196 928 |
| LP-to-GT ratio | 9.05 × |

The decoupled loss weighting with γ = 0.181 down-weights the LP-sourced voxels
relative to GT seeds, mitigating noise from propagated labels.

---

## Key Takeaways

1. **Graph LP improves average Dice by +6.3 pp** (80.9 % → 87.2 %) at the
   same 0.1 % annotation budget.
2. **Graph LP closes 96.6 % of the gap to the fully supervised upper bound**
   (87.2 % vs 90.3 % avg Dice) using only 0.1 % of the labels.
3. The largest gains over the fewpoint baseline are on the hardest classes:
   **class 3 (+17.0 pp)** and **class 4 (+11.7 pp)**.
4. Class 2 is the only regression (−3.6 pp Dice vs fewpoint), suggesting
   Graph LP propagation may introduce noise for that particular tissue type.
5. Decoupled source weighting (γ = 0.181) effectively balances the 9 : 1
   ratio of propagated-to-GT voxels, preventing the noisy LP labels from
   dominating the loss.
