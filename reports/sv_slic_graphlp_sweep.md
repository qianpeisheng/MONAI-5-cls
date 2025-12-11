## Supervoxel + Graph LP Sweep (New WP5 dataset)

### Common Settings
- Dataset: `/data3/wp5_4_Dec_data/3ddl-dataset` (`datalist_train_new.json`)
- SLIC: mode `slic`, `compactness=0.05`, `sigma=1.0`
- Ignore label: `6`
- Seeds: `runs/strategic_sparse_0p1pct_new/strategic_seeds` (0.1% budget, class weights 0.1,1,1,2,2)
- Graph LP: `k=10`, `alpha=0.9`, `num_classes=5`
- Workers: 16 for gen/eval/LP
- Outputs per n:
  - SV (full-GT vote): `runs/sv_fullgt_slic_n${N}_new_ras` (+ `_eval`)
  - Graph LP: `runs/graph_lp_prop_0p1pct_k10_a0.9_new_n${N}` (+ `eval/`)

### Results Summary (Dice averages, classes 0–4, ignore 6)
| n_segments | SV vote Dice | Graph LP Dice |
|------------|--------------|---------------|
| 6000  | 0.8890 | 0.2800 |
| 8000  | 0.8987 | 0.4046 |
| 10000 | 0.9051 | 0.5593 |
| 12000 | 0.9128 | **0.7019** |
| 14000 | 0.9167 | 0.6327 |
| 16000 | 0.9183 | 0.5403 |
| 18000 | 0.9185 | 0.5273 |
| 20000 | **0.9197** | 0.5001 |

- Best SV-voted dense labels: n=20000 (Dice 0.9197).
- Best Graph LP pseudo-labels: n=12000 (Dice 0.7019).

### Per-class Dice (best Graph LP: n=12000)
- class 0: 0.9450
- class 1: 0.7500
- class 2: 0.6588
- class 3: 0.5896
- class 4: 0.5660

### Paths to key files
- SV eval metrics: `runs/sv_fullgt_slic_n${N}_new_ras_eval/metrics/summary.json`
- Graph LP eval metrics: `runs/graph_lp_prop_0p1pct_k10_a0.9_new_n${N}/eval/metrics/summary.json`
- Graph LP labels for training (best): `runs/graph_lp_prop_0p1pct_k10_a0.9_new_n12000/labels/`
- SV IDs (n=12000): `runs/sv_fullgt_slic_n12000_new_ras/`

### Recommendation
- For highest-quality dense pseudo-labels: use SV full-GT voting at n=20000.
- For Graph LP at 0.1% seeds: use n=12000 SLIC supervoxels with k=10, α=0.9.
