# GraphLP confidence-ensemble loss-weight sweep (sweep40)
- Generated: 2026-01-30 10:26:15
- Source dir: `/data3/MONAI_experiments/sweep_graphlp_conf_ens_lossw_sweep40`
- Driver script: `/home/peisheng/MONAI/scripts/sweep_graphlp_confidence_ensemble_loss_weights.sh`

## 1. Objective
Evaluate agreement-count loss reweighting (`--agree_weight_mode decoupled`) for GraphLP confidence-ensemble pseudo labels, sweeping `--agree_gamma_table` and `--agree_imbalance_scope` (dataset vs batch).

## 2. Setup
- Dataset: `/data3/wp5_4_Dec_data/3ddl-dataset` (split config: `/data3/wp5_4_Dec_data/3ddl-dataset/data/dataset_config.json`)
- Split (from logs): 613 train / 174 test (new WP5 dataset via `wp5_bump_dataset_loader`)
- Model: `basicunet` init=`scratch` (BasicUNet from scratch)
- Training: epochs=40, batch_size=4, lr=0.001, seed=42
- Patch ROI: (112, 112, 80) norm=`clip_zscore`
- Pseudo-label source: ensemble vote dirs `runs/graph_lp_ens_vote_{COMQ|CQ}_tieQ/{labels,agreement}`
- Metric: per-epoch *test* Dice averaged over classes 0..4 (ignore label=6); best epoch selected by max avg Dice

## 3. Results
- Configs expected: 40
- Configs started (found dirs): 38
- Runs complete: 36
- Runs incomplete (skipped): 2
  - Incomplete run dirs:
    - `CQ_dec_bs_g2_0p20_g1_0p02_20260130-102303`
    - `CQ_dec_ds_g2_0p20_g1_0p02_20260130-102124`
  - Missing configs (not started yet):
    - `CQ_dec_bs_g2_0p25_g1_0p02`
    - `CQ_dec_ds_g2_0p25_g1_0p02`

### Overall Best
- **0.862931** (best epoch 22) — `COMQ_dec_bs_g4_0p20_g3_0p15_g2_0p00_20260130-005424`
  - Ensemble: `COMQ`, imbalance scope: `batch`, gamma table: `4:0.20,3:0.15,2:0.00,1:0.00`
  - Dice class 3: 0.6999; class 4: 0.8458

### Best Per Group
- COMQ/batch: **0.862931** (ep 22) — gamma `4:0.20,3:0.15,2:0.00,1:0.00` — `COMQ_dec_bs_g4_0p20_g3_0p15_g2_0p00_20260130-005424`
- COMQ/dataset: **0.862589** (ep 22) — gamma `4:0.20,3:0.05,2:0.00,1:0.00` — `COMQ_dec_ds_g4_0p20_g3_0p05_g2_0p00_20260130-000602`
- CQ/dataset: **0.860329** (ep 24) — gamma `2:0.35,1:0.00` — `CQ_dec_ds_g2_0p35_g1_0p00_20260130-084736`
- CQ/batch: **0.857953** (ep 24) — gamma `2:0.20,1:0.01` — `CQ_dec_bs_g2_0p20_g1_0p01_20260130-093544`

### Aggregate Stats (Best-Epoch Avg Dice)
| Ensemble | Scope | n | mean | std | min | max |
|---|---|---:|---:|---:|---:|---:|
| COMQ | batch | 10 | 0.860314 | 0.002219 | 0.856546 | 0.862931 |
| COMQ | dataset | 10 | 0.860895 | 0.001399 | 0.857977 | 0.862589 |
| CQ | batch | 8 | 0.853580 | 0.002105 | 0.851431 | 0.857953 |
| CQ | dataset | 8 | 0.856860 | 0.003241 | 0.850543 | 0.860329 |

### Leaderboard (Top 12)
| Rank | Best Dice | Best Ep | Ens | Scope | Gamma table | Run dir |
|---:|---:|---:|---|---|---|---|
| 1 | 0.862931 | 22 | COMQ | batch | `4:0.20,3:0.15,2:0.00,1:0.00` | `COMQ_dec_bs_g4_0p20_g3_0p15_g2_0p00_20260130-005424` |
| 2 | 0.862654 | 24 | COMQ | batch | `4:0.20,3:0.10,2:0.02,1:0.00` | `COMQ_dec_bs_g4_0p20_g3_0p10_g2_0p02_20260130-022956` |
| 3 | 0.862589 | 22 | COMQ | dataset | `4:0.20,3:0.05,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p20_g3_0p05_g2_0p00_20260130-000602` |
| 4 | 0.862487 | 22 | COMQ | batch | `4:0.25,3:0.125,2:0.00,1:0.00` | `COMQ_dec_bs_g4_0p25_g3_0p125_g2_0p00_20260129-223125` |
| 5 | 0.862222 | 24 | COMQ | dataset | `4:0.10,3:0.05,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p10_g3_0p05_g2_0p00_20260129-200850` |
| 6 | 0.862209 | 24 | COMQ | batch | `4:0.20,3:0.10,2:0.01,1:0.00` | `COMQ_dec_bs_g4_0p20_g3_0p10_g2_0p01_20260130-014217` |
| 7 | 0.862156 | 24 | COMQ | dataset | `4:0.20,3:0.15,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p20_g3_0p15_g2_0p00_20260130-005344` |
| 8 | 0.861876 | 24 | COMQ | dataset | `4:0.20,3:0.10,2:0.01,1:0.00` | `COMQ_dec_ds_g4_0p20_g3_0p10_g2_0p01_20260130-014124` |
| 9 | 0.861372 | 26 | COMQ | dataset | `4:0.20,3:0.10,2:0.02,1:0.00` | `COMQ_dec_ds_g4_0p20_g3_0p10_g2_0p02_20260130-022857` |
| 10 | 0.861011 | 24 | COMQ | dataset | `4:0.30,3:0.15,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p30_g3_0p15_g2_0p00_20260129-231812` |
| 11 | 0.860769 | 21 | COMQ | batch | `4:0.30,3:0.15,2:0.00,1:0.00` | `COMQ_dec_bs_g4_0p30_g3_0p15_g2_0p00_20260129-231917` |
| 12 | 0.860623 | 18 | COMQ | dataset | `4:0.20,3:0.10,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p20_g3_0p10_g2_0p00_20260129-214311` |

### COMQ / dataset scope
| Best Dice | Best Ep | Gamma table | Run dir |
|---:|---:|---|---|
| 0.862589 | 22 | `4:0.20,3:0.05,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p20_g3_0p05_g2_0p00_20260130-000602` |
| 0.862222 | 24 | `4:0.10,3:0.05,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p10_g3_0p05_g2_0p00_20260129-200850` |
| 0.862156 | 24 | `4:0.20,3:0.15,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p20_g3_0p15_g2_0p00_20260130-005344` |
| 0.861876 | 24 | `4:0.20,3:0.10,2:0.01,1:0.00` | `COMQ_dec_ds_g4_0p20_g3_0p10_g2_0p01_20260130-014124` |
| 0.861372 | 26 | `4:0.20,3:0.10,2:0.02,1:0.00` | `COMQ_dec_ds_g4_0p20_g3_0p10_g2_0p02_20260130-022857` |
| 0.861011 | 24 | `4:0.30,3:0.15,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p30_g3_0p15_g2_0p00_20260129-231812` |
| 0.860623 | 18 | `4:0.20,3:0.10,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p20_g3_0p10_g2_0p00_20260129-214311` |
| 0.859571 | 22 | `4:0.25,3:0.125,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p25_g3_0p125_g2_0p00_20260129-223031` |
| 0.859551 | 22 | `4:0.15,3:0.075,2:0.00,1:0.00` | `COMQ_dec_ds_g4_0p15_g3_0p075_g2_0p00_20260129-205549` |
| 0.857977 | 25 | `4:0.30,3:0.15,2:0.02,1:0.00` | `COMQ_dec_ds_g4_0p30_g3_0p15_g2_0p02_20260130-031636` |

### COMQ / batch scope
| Best Dice | Best Ep | Gamma table | Run dir |
|---:|---:|---|---|
| 0.862931 | 22 | `4:0.20,3:0.15,2:0.00,1:0.00` | `COMQ_dec_bs_g4_0p20_g3_0p15_g2_0p00_20260130-005424` |
| 0.862654 | 24 | `4:0.20,3:0.10,2:0.02,1:0.00` | `COMQ_dec_bs_g4_0p20_g3_0p10_g2_0p02_20260130-022956` |
| 0.862487 | 22 | `4:0.25,3:0.125,2:0.00,1:0.00` | `COMQ_dec_bs_g4_0p25_g3_0p125_g2_0p00_20260129-223125` |
| 0.862209 | 24 | `4:0.20,3:0.10,2:0.01,1:0.00` | `COMQ_dec_bs_g4_0p20_g3_0p10_g2_0p01_20260130-014217` |
| 0.860769 | 21 | `4:0.30,3:0.15,2:0.00,1:0.00` | `COMQ_dec_bs_g4_0p30_g3_0p15_g2_0p00_20260129-231917` |
| 0.859693 | 22 | `4:0.20,3:0.05,2:0.00,1:0.00` | `COMQ_dec_bs_g4_0p20_g3_0p05_g2_0p00_20260130-000657` |
| 0.859602 | 22 | `4:0.20,3:0.10,2:0.00,1:0.00` | `COMQ_dec_bs_g4_0p20_g3_0p10_g2_0p00_20260129-214343` |
| 0.859483 | 24 | `4:0.15,3:0.075,2:0.00,1:0.00` | `COMQ_dec_bs_g4_0p15_g3_0p075_g2_0p00_20260129-205616` |
| 0.856765 | 25 | `4:0.30,3:0.15,2:0.02,1:0.00` | `COMQ_dec_bs_g4_0p30_g3_0p15_g2_0p02_20260130-031719` |
| 0.856546 | 22 | `4:0.10,3:0.05,2:0.00,1:0.00` | `COMQ_dec_bs_g4_0p10_g3_0p05_g2_0p00_20260129-200850` |

### CQ / dataset scope
| Best Dice | Best Ep | Gamma table | Run dir |
|---:|---:|---|---|
| 0.860329 | 24 | `2:0.35,1:0.00` | `CQ_dec_ds_g2_0p35_g1_0p00_20260130-084736` |
| 0.860316 | 22 | `2:0.20,1:0.01` | `CQ_dec_ds_g2_0p20_g1_0p01_20260130-093419` |
| 0.859242 | 22 | `2:0.25,1:0.00` | `CQ_dec_ds_g2_0p25_g1_0p00_20260130-071301` |
| 0.857280 | 24 | `2:0.15,1:0.00` | `CQ_dec_ds_g2_0p15_g1_0p00_20260130-053834` |
| 0.857104 | 22 | `2:0.20,1:0.00` | `CQ_dec_ds_g2_0p20_g1_0p00_20260130-062551` |
| 0.856982 | 24 | `2:0.10,1:0.00` | `CQ_dec_ds_g2_0p10_g1_0p00_20260130-045106` |
| 0.853087 | 24 | `2:0.30,1:0.00` | `CQ_dec_ds_g2_0p30_g1_0p00_20260130-080024` |
| 0.850543 | 22 | `2:0.05,1:0.00` | `CQ_dec_ds_g2_0p05_g1_0p00_20260130-040401` |

### CQ / batch scope
| Best Dice | Best Ep | Gamma table | Run dir |
|---:|---:|---|---|
| 0.857953 | 24 | `2:0.20,1:0.01` | `CQ_dec_bs_g2_0p20_g1_0p01_20260130-093544` |
| 0.854792 | 22 | `2:0.35,1:0.00` | `CQ_dec_bs_g2_0p35_g1_0p00_20260130-084842` |
| 0.854645 | 22 | `2:0.30,1:0.00` | `CQ_dec_bs_g2_0p30_g1_0p00_20260130-080123` |
| 0.853989 | 24 | `2:0.25,1:0.00` | `CQ_dec_bs_g2_0p25_g1_0p00_20260130-071358` |
| 0.852713 | 24 | `2:0.05,1:0.00` | `CQ_dec_bs_g2_0p05_g1_0p00_20260130-040439` |
| 0.851596 | 24 | `2:0.15,1:0.00` | `CQ_dec_bs_g2_0p15_g1_0p00_20260130-053927` |
| 0.851520 | 24 | `2:0.20,1:0.00` | `CQ_dec_bs_g2_0p20_g1_0p00_20260130-062643` |
| 0.851431 | 24 | `2:0.10,1:0.00` | `CQ_dec_bs_g2_0p10_g1_0p00_20260130-045204` |

### Best-Epoch Distribution
| Best epoch | count |
|---:|---:|
| 18 | 1 |
| 21 | 1 |
| 22 | 14 |
| 24 | 17 |
| 25 | 2 |
| 26 | 1 |

## 4. Key Findings
- COMQ runs dominate the top of the leaderboard (best COMQ=0.862931 vs best CQ=0.860329).
- For CQ, `agree_imbalance_scope=dataset` outperforms `batch` on average by +0.003281 Dice (mean-delta).
- Adding a small weight for count-1 voxels (gamma includes `1:0.01`) is competitive: best such CQ run is 0.860316 (`CQ_dec_ds_g2_0p20_g1_0p01_20260130-093419`).

## 5. Issues / Anomalies
- Some runs are still in progress; they were skipped in quantitative comparisons above.
- Not all expected configs have been started yet (see missing list in Results).

## 6. Conclusions
- Best observed config so far: COMQ + decoupled weighting + batch imbalance scope with gamma `4:0.20,3:0.15,2:0.00,1:0.00` (avg Dice 0.862931).
- For CQ, the current best is dataset-scope with gamma `2:0.35,1:0.00` (avg Dice 0.860329), with `2:0.20,1:0.01` essentially tied.

## 7. Suggested Next Steps
- Let the in-progress `g1_0p02` runs finish, then re-run this summary to confirm whether `1:0.02` helps or hurts.
- Start/complete the missing `CQ_dec_{ds,bs}_g2_0p25_g1_0p02` configs to finish the intended 40-run sweep.
- If time-constrained, consider early stopping around epoch ~24 (most runs peak at 22–24) to speed future sweeps.
