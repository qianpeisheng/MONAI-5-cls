# Investigation: Graph-LP pseudo labels (coords-only vs intensity-aware) — 2026-01-29

Goal: understand why `runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian` produces **much higher pseudo-label Dice** than `runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000`, and identify practical ways to turn higher pseudo-label quality into **better trained model performance**.

This report focuses on the **train-split GT overlap** (datalist `datalist_train_new.json`, 613 cases), since both pseudo-label runs are generated from *train* data/seeds.

## 1) The two pseudo-label sets

### Coords-only Graph LP (baseline)
- Run: `runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000`
- Graph LP uses **coords-only affinity** (`--descriptor_type none` in docs/scripts).
- Pseudo-label eval: `runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000/eval/metrics/summary.json`

### Intensity-aware Graph LP (same solver, different affinity `W`)
- Run group: `runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian`
- Uses the same Graph LP solver (Zhou diffusion), but modifies the affinity with an **intensity descriptor term** in addition to the centroid coords term (see `scripts/run_graph_lp_3_descriptors_then_eval.sh` and `GRAPH_LP_TRAINING_GUIDE.md`).
- Best-performing descriptors here are:
  - `moments`: `.../moments/eval/metrics/summary.json`
  - `quantiles16`: `.../quantiles16/eval/metrics/summary.json`

## 2) Pseudo-label quality summary (Dice 0..4, ignore 6)

Numbers below are reproduced by running:
`python3 scripts/analyze_graph_lp_pseudolabel_sets.py --preset wp5_graphlp_intensity_vs_coords`

### Overall (all voxels, GT!=6)
| Set | Avg Dice | Dice (0,1,2,3,4) | Voxel Acc |
|---|---:|---|---:|
| Coords-only `C` | 0.7019 | (0.9450, 0.7500, 0.6588, 0.5896, 0.5660) | 0.8925 |
| Outer-bg adaptive coords `O` | 0.7422 | (0.9288, 0.7517, 0.6998, 0.7124, 0.6186) | 0.8787 |
| Moments `M` | **0.7725** | (0.9774, 0.8471, 0.7140, 0.6238, 0.7001) | 0.9412 |
| Quantiles16 `Q` | 0.7712 | (0.9770, 0.8459, 0.7127, 0.6234, 0.6970) | 0.9405 |
| Hist32 `H` | 0.7320 | (0.9553, 0.7909, 0.7000, 0.6028, 0.6111) | 0.9096 |

Key point: `Q` (and `M`) improve over coords-only by ~**+0.069 Dice** on average.

## 3) Where the improvement comes from: it’s 100% on graph-only voxels

Graph LP uses a `source_mask` (`source_masks/<id>_source.npy`) marking voxels belonging to SVs that had at least one GT seed (“seeded SVs”) vs purely graph-propagated SVs (“graph-only”).

Empirically:
- Seeded voxels are **identical** between coords-only and intensity-aware runs (the labeled nodes are effectively clamped).
- All gains come from better labeling of **graph-only** SVs.

From the deep breakdown in `scripts/analyze_graph_lp_pseudolabel_sets.py --preset ...`:

### Graph-only region (GT!=6 and `source==0`)
- Coverage (mean over cases): **0.9346** (so ~93.5% of non-ignore voxels are graph-only)
- Coords-only `C`: avg Dice **0.6929**, voxel acc **0.8909**
- Quantiles16 `Q`: avg Dice **0.7687**, voxel acc **0.9422**
- Improvement on graph-only voxels: **+0.0758 Dice**

### Seeded region (GT!=6 and `source==1`)
- Coverage (mean over cases): **0.0654**
- `C` and `Q` are identical here: avg Dice **0.8654**, voxel acc **0.9310**

Interpretation:
- If your training still underperforms with better pseudo labels, the root cause is not “seeded SV purity” (that piece is unchanged), but how training handles the **graph-only** region (coverage ~90% at the dataset-voxel level).

## 4) Case-level behavior: big improvements but also real regressions

From `per_case.csv` (`Q - C`):
- Mean delta: **+0.0693**, median delta: **+0.0735**

Largest improvements (avg Dice per case):
- `SN155_I127`: +0.1851
- `SN48_I2`: +0.1744
- `SN13_I60`: +0.1730
- `SN61_I24`: +0.1661
- `SN155_I19`: +0.1630

Largest regressions:
- `SN72_I155`: −0.1049
- `SN76_I141`: −0.0903
- `SN19_I48`: −0.0829
- `SN75_I154`: −0.0765
- `SN74_I176`: −0.0687

Persistent hard case: `SN21_I33` is the worst across multiple variants.

## 5) Combining pseudo-label runs (ensembles) and confidence filtering

### 5.1 Simple ensemble vote improves pseudo labels (small but consistent)

We tested majority vote (ties broken by `Q`):

- Vote(`C,M,Q`) → avg Dice **0.7732**
- Vote(`C,O,M,Q`) → avg Dice **0.7752** (best we tried)

`vote(C,O,M,Q)` slightly improves class-4 and reduces regressions vs using `Q` alone.

If you want to actually *materialize* this ensemble as a new pseudo-label set on disk:
- Script: `scripts/build_graph_lp_ensemble_labels.py`

### 5.2 Agreement / confidence is a strong “noise detector”

Even without any model training, agreement between runs strongly predicts correctness:

#### Agreement of `C` vs `Q` on graph-only voxels
- They agree on ~**90.1%** of graph-only voxels (mean over cases).
- `Q` Dice on the **agree** region: mean **0.8079**
- `Q` Dice on the **disagree** region: mean **0.6302**

So “disagreement voxels” are disproportionately noisy.

#### Confidence levels for `vote(C,O,M,Q)` on graph-only voxels
Using `maxc` = max vote count among {C,O,M,Q}:

| maxc | Meaning | Coverage (mean, within graph-only) | Dice (mean, within that subset) |
|---:|---|---:|---:|
| 4 | all agree | 0.8261 | **0.8587** |
| 3 | 3-of-4 agree | 0.1144 | 0.7115 |
| 2 | 2-of-4 agree | 0.0594 | 0.5122 |
| 1 | all different (rare) | 0.0001 | ~0.09 |

Takeaway:
- “Bigger pseudo-label weights” can fail because it **amplifies the noisy tail** (maxc=1/2 regions).
- A better lever is to **filter or downweight low-confidence voxels** while keeping high-coverage, high-precision voxels (maxc=4 is already ~82.6% of graph-only voxels).

## 6) Practical recommendations to improve trained model performance

These are concrete next experiments that directly use the findings above.

### A) Train on ensemble labels + confidence-derived source masks
1) Build ensemble labels and a confidence-based `source_masks/`:
`python3 scripts/build_graph_lp_ensemble_labels.py --datalist datalist_train_new.json --out_dir runs/graph_lp_ensemble_vote_C_O_M_Q_tieQ_thr3 --label_dir C=runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000 --label_dir O=runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000_outerbg_adaptive --label_dir M=runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/moments --label_dir Q=runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/quantiles16 --tie_break Q --seed_source_mask_dir runs/graph_lp_3desc_eval/graph_lp_3desc_20260109-162446_k10_a0.9_sigPhimedian/quantiles16/source_masks --confidence_threshold 3 --write_source_masks --write_confidence_maps`

2) Train using:
- `--train_label_override_dir runs/graph_lp_ensemble_vote_C_O_M_Q_tieQ_thr3/labels`
- `--train_label_source_dir runs/graph_lp_ensemble_vote_C_O_M_Q_tieQ_thr3/source_masks`

This changes the weighting target from “seeded vs graph-only” to “seeded-or-high-confidence vs low-confidence”.

### B) Curriculum: start with maxc=4 only, then relax
Because maxc=4 voxels are very accurate (Dice ~0.86 on graph-only subset), you can:
- Start training using only maxc>=4 as “reliable” (plus seeds),
- Then switch to maxc>=3 after N epochs,
- Optionally include maxc=2 late (or keep it ignored).

This is often more effective than globally cranking pseudo-label weight.

### C) Investigate the regression cases
If you want to improve the intensity-aware method itself, inspect the worst regression IDs (e.g., `SN72_I155`, `SN76_I141`) by:
- visualizing `C` vs `Q` vs GT,
- checking if the intensity descriptor is collapsing (e.g., unusual histograms / saturation),
- trying `--use_cosine` for `quantiles16` or adjusting `sigma_phi` estimation.

## 7) Helper scripts added in this work

- Analysis:
  - `scripts/analyze_graph_lp_pseudolabel_sets.py` (use `--preset wp5_graphlp_intensity_vs_coords`)
- Ensemble builder:
  - `scripts/build_graph_lp_ensemble_labels.py` (writes a new pseudo-label set on disk)

