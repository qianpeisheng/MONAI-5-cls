# Plan: Dedicated count-based confidence support for pseudo-label ensembles (Graph-LP)

Goal: replace the current **binary** “source mask” weighting (seed-supported vs graph-only) with **explicit, count-based agreement confidence** that works for *any number of ensembled pseudo-label sets*.

Key message: the ensemble builder must **not** decide reliability. It should only store the **raw agree/disagree counts** so the **training code** can decide thresholds/weights later.

Instead of `{2=seed, 1=agreed, 0=disagreed}`, we will store a single per-voxel **agreement-count** map:
- **Pseudo-label voxels**: `agree_count in [1..K]` where `K = #voters` for the ensemble
  - `disagree_count = K - agree_count` (derived in training)
- **GT-supported voxels** (SV contains ≥1 GT seed): encode as a **very large agree_count sentinel** (default `255`)
  - training treats this as the highest-confidence “GT tier”

This supports ensembling 2, 3, 4, 5… pseudo-label sets without changing the representation.

---

## A. Data / artifact spec (what files we will standardize)

### A1) Keep labels as-is
- `labels/<id>_labels.npy` (int16, values 0–4; label 6 remains “ignore” if present)

### A2) Optional input: seed-support mask from Graph-LP runs (for GT-tier encoding)
- input format (already exists in Graph-LP outputs): `source_masks/<id>_source.npy` (uint8 or bool)
  - `1` = voxel belongs to a SV that had ≥1 GT seed (seed-supported / GT-anchored)
  - `0` = purely graph-labeled (graph-only)

We will use this mask to **encode GT tier** by setting `agree_count=255` on those voxels in the ensemble output.

### A3) Add a dedicated agreement-count map (new, core change)
- `agreement/<id>_agree_count.npy` (uint8)
  - value is the **max vote count** (how many voters agree on the final chosen label at this voxel)
  - for pseudo-label voxels: range is `1..K`, where `K = number of ensembled label sets`
  - for GT-supported voxels: value is a sentinel `255` (very large agree_count)

Notes:
- This is the “raw number of agree”.
- The “raw number of disagree” is computed as `disagree_count = K - agree_count` using `K` from metadata (no thresholding/bucketing in the builder). This applies only when `agree_count <= K`.
- It generalizes what we previously called `maxc`.
- If we also save `confidence/<id>_maxc.npy`, it should be treated as an alias of `agreement/<id>_agree_count.npy` (same content, one canonical name).
- The GT-tier sentinel (`255`) is not “votes”; it is an encoding for “GT-supported SV” (documented in metadata).

### A4) Metadata for provenance (required)
`propagation_summary.json` (or `ensemble_summary.json`) must record:
- the voter list (label sources; e.g., `C,O,M,Q`)
- `K` (number of voters)
- tie-break rule
- how `agree_count` is computed (it should always be “max votes for the chosen label”)
- the GT sentinel value and meaning (default `255` = “SV contains ≥1 GT seed voxel”)
- explicitly state that the builder performs **no reliability thresholding** (it only records counts)

---

## B. Training semantics (how agreement count affects loss)

We will compute a per-voxel weight map `w(x)` from:
- `agree_count(x)` from `agreement/` (raw #agree for pseudo voxels; GT-tier sentinel for GT-supported voxels).

### B1) Intended semantics
- If `agree_count==255` (GT tier): weight is **always** `w_gt = 1.0` (fixed).
- Else (pseudo voxels, `1..K`):
  - weight is a function of raw vote counts: `w = f(agree_count, K)` (or equivalently `f(agree_count, disagree_count)`)
  - any thresholding / “ignore low-confidence” is done **here** (training), e.g. `f(agree_count<=t)=0`

Additionally:
- voxels with `label==6` are ignored (weight forced to 0; CE ignore + Dice mask).

### B2) Weighting modes (clear + tunable)

**Mode 1: table (most explicit / controllable)**
User provides a mapping from count → weight:
- `--agree_weight_mode table`
- `--agree_weight_table "1:0,2:0.02,3:0.1,4:0.2"` (example for K=4)

Rules:
- `w_gt` is always 1.0 for `agree_count==255` (not in the table).
- Counts not listed default to **0** (safest; avoids accidentally weighting a tier you forgot to specify).
- Works for any K, as long as the user provides weights for the counts they care about.

**Mode 2: decoupled-by-count (mass-ratio friendly, scale-invariant)**
Generalize the existing “decoupled gamma” idea to multiple pseudo-label groups, one per count value.

Treat `agree_count==255` as the GT tier. Let `N_gt` be #voxels with `agree_count==255`,
and `N_c` be #voxels with `agree_count==c` for `c in [1..K]`.

Keep `w_gt=1.0` and define for each count level `c`:
- `w_c = (N_gt / N_c) * gamma_c`

So weight-mass ratios are directly interpretable:
- `(w_c * N_c) / (w_gt * N_gt) ≈ gamma_c`

CLI:
- `--agree_weight_mode decoupled`
- `--agree_gamma_table "1:0,2:0.01,3:0.05,4:0.1"`
- `--agree_imbalance_scope dataset|batch` (default `dataset`)

This is the cleanest way to tune “how much total learning signal” comes from each confidence tier.

### B3) Logging / saved stats (required for debugging)
At run start compute and save:
- `runs/<exp>/metrics/agreement_stats.json`
  - `K`
  - `n_gt_vox`
  - `n_by_count: {1:...,2:..., ...}`
  - effective weights per tier (`w_by_count`)
  - implied mass ratios per tier (`q_by_count`)

---

## C. Code changes (minimal surface area, backward compatible)

### C1) `train_finetune_wp5.py` (training support)

1) CLI additions:
   - `--train_label_agreement_dir <dir>` (loads `agreement/<id>_agree_count.npy`)
   - `--agree_gt_sentinel 255` (default 255; used for the GT tier)
   - `--agree_weight_mode table|decoupled`
   - `--agree_weight_table ...` / `--agree_gamma_table ...`
   - `--agree_imbalance_scope dataset|batch`
   - keep existing `--train_label_source_dir` working unchanged

2) Data loading:
   - extend dataset dict/transforms to load a new key, e.g. `agree_count`
   - ensure it undergoes the same spatial transforms/cropping as `label` and `label_source`

3) Loss:
   - build a unified `voxel_weight_map`:
     - `agree_count==255` voxels get 1.0
     - graph-only voxels get `w_by_count[agree_count]`
   - apply to:
     - CE (via `cross_entropy_with_voxel_weights`)
     - Dice (via `dice_loss_masked_weighted`)

4) Backward compatibility:
   - If `--train_label_agreement_dir` is provided:
     - use agreement-count weighting (and still optionally incorporate seed support via `--train_label_source_dir`)
   - Else:
     - keep existing binary behavior (no changes to old experiments)

### C2) Ensemble tooling: update / formalize pseudo-label ensembling

We must update the ensembling code so it produces the new standardized artifacts.

1) Update `scripts/build_graph_lp_ensemble_labels.py` to:
   - accept **any number of** `--label_dir NAME=PATH`
   - write:
     - `labels/<id>_labels.npy` (majority vote, tie-break configurable)
     - `agreement/<id>_agree_count.npy` (the per-voxel max vote count, 1..K, with GT-tier sentinel=255 if provided)
     - (optionally) keep writing `confidence/<id>_maxc.npy` as an alias, but document `agreement/` as canonical
   - optionally take a seed-support source mask dir and encode GT tier:
     - input: `--seed_source_mask_dir <dir>` with `<id>_source.npy`
     - behavior: set `agree_count=255` where `source==1`
   - write `ensemble_summary.json` with voter list and `K`
   - explicitly: **no filtering / thresholding** in the builder (no `label=6` rewriting, no dropping voxels); only record raw counts

2) (Optional) Add a small convenience wrapper:
   - build multiple ensembles (different voter sets) into separate output dirs
   - avoids manual repeated commands

### C3) Evaluation tooling (optional but recommended)
We should prefer weight maps (ignore-by-weight) over rewriting labels to 6. If we still need label=6 rewriting for some workflows, optionally extend `scripts/eval_sv_voted_wp5.py` with `--pred_ignore_class 6`.

---

## D. Testing plan (unit-first, fast)

Unit tests (CPU, tiny tensors) to validate:

1) **Agreement-count weighting**
   - table mode: mapping correctness for arbitrary K (e.g., K=2,4,6)
   - decoupled-by-count: verify mass ratios per tier match specified `gamma_c`
   - precedence: seed-supported voxels override agree_count mapping (always weight 1)

2) **Transform / loading**
   - `agree_count` loads and is cropped/padded consistently with `label`

3) **Micro forward+loss smoke test**
   - ensure CE/Dice accept the weight map and gradients flow

4) **Ensemble builder correctness (toy volumes)**
   - majority vote produces expected `labels`
   - `agree_count` equals the number of voters for the chosen label (1..K), and uses 255 on GT-tier voxels if mask provided
   - tie-break is deterministic and recorded in `ensemble_summary.json`

---

## E. Rollout plan (phased, low risk)

1) Implement agreement-count loading + weighting in training (keep old flags working)
2) Update ensemble builder to write `agreement/` + `ensemble_summary.json`
3) Add unit tests
4) Update docs: new artifact format + new CLI flags + examples for K=2 and K=4
5) Run 1–2 short sanity trainings (few epochs) to verify:
   - weights logged as expected
   - agreement tiers behave as intended

---

## F. Post-implementation experiment grid (2 GPUs) — what we’ll run after approval

Once code support exists, the first tuning axes are:
1) **Which voters** to ensemble (defines K and the meaning of agreement levels)
2) **How to weight each agree_count tier** (table or decoupled gamma table)

Suggested initial setup (K=4 voters: `C,O,M,Q`):

**Agreement tiers**
- count 4 (unanimous): highest pseudo weight
- count 3: medium pseudo weight
- count 2: very low pseudo weight (often 0 or near-0)
- count 1: 0

**Decoupled gamma table examples to start**
- `gamma_table_A`: `"4:0.10,3:0.05,2:0.00,1:0.00"`
- `gamma_table_B`: `"4:0.20,3:0.10,2:0.02,1:0.00"`

Scheduling on 2 GPUs:
- GPU0: `gamma_table_A` vs `gamma_table_B` for the same ensemble
- GPU1: repeat the same grid but with a different voter set (e.g., `C,M,Q` (K=3)) or a different tie-break

We’ll generate a dedicated `.sh` runner after implementation (matching the final CLI and artifact paths).

---

## Defaults (locked in for implementation)

1) `agree_count + K` only (no runnerup / vote histogram for now)
2) GT tier encoded as `agree_count=255` (documented in `ensemble_summary.json`)
3) Canonical dir name: `agreement/` (keep `confidence/` as optional legacy alias output if needed)
