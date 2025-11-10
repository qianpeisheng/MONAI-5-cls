# AGENTS.md — Using MONAI here for WP5‑style 3D Segmentation

This guide orients you (and any agents) to use this MONAI checkout for medical image segmentation on datasets like WP5. It assumes you have data in NIfTI pairs (image/label) similar to `/data3/wp5/wp5-code/dataloaders/wp5-dataset/data` and a working notebook `mednist_tutorial.ipynb` in this folder.

Key references in this repo:
- Notebook: `mednist_tutorial.ipynb` (classification tutorial — reuse its env, transforms, and DataLoader patterns)
- Data spec for WP5: `WP5_Segmentation_Data_Guide.md` (labels, splits, transforms, pitfalls)

Scope: All code under this directory. Keep changes minimal, adopt MONAI idioms, avoid leaking domain‑specific assumptions unless documented here.

## Command Presentation (for agents)
- Always show runnable Python and shell commands in a single line when responding. This avoids terminals treating only the first line as the command and silently dropping subsequent flags when users paste multi‑line snippets.
- Prefer explicit flags and full paths where helpful for reproducibility.

## Environment
- Python: 3.8+ recommended.
- Install MONAI (if not using this repo in editable mode): `pip install -e .` from this folder or `pip install monai[all]` in a fresh venv.
- GPU: CUDA recommended for 3D models; CPU works for development.
- Use the conda recipe `environment-dev.yml` or `requirements*.txt` as needed.

## Data Shape and Labels (WP5‑like)
- 3D NIfTI volumes, float32 images, uint16 labels.
- Variable spatial shapes; keep spacing as is unless you standardize later.
- Label values observed: `{0,1,2,3,4,6}` where `0` is background; ignore class `6` by default for training and metrics (see policy below).
- Default split (WP5): train 380, test 180 via predefined serial‑number config.

## Default Policies (important!)
- Split: Use the predefined serial split if applicable (train=380, test=180 on WP5).
- Classes and ignore: compute metrics over classes `0..4` and ignore label `6`.
  - CrossEntropy: set `ignore_index=6` (or remap `6→4` to get 5 classes).
  - Dice: mask voxels where `label==6` before computing per‑class Dice.
- Patch size: `(112,112,80)` and sliding window inference are good defaults for variable sizes.
- Normalization: robust per‑sample (clip to [p1,p99] then z‑score) to handle heavy tails.

## Data Lists (recommended structure)
Create JSON lists for train/test with entries like `{ "image": "/path/.._image.nii", "label": "/path/.._label.nii", "id": "..." }`. Examples are in `WP5_Segmentation_Data_Guide.md`.

Quick generator for WP5 split (adjust paths as needed):
```python
import os, json, re
from pathlib import Path

root = Path('/data3/wp5/wp5-code/dataloaders/wp5-dataset')
data = root / 'data'
cfg = json.loads(Path(root/'3ddl_split_config_20250801.json').read_text())
test_serials = set(cfg['test_serial_numbers'])
pat = re.compile(r'^SN(\d+)B(\d+)_I(\d+).*_(image|label)\.nii$')

def pairs(dirp):
    out = {}
    for n in os.listdir(dirp):
        if not n.endswith('_image.nii'): continue
        b = n[:-10]
        m = pat.match(n); serial = int(m.group(1)) if m else None
        img = str(dirp/ f'{b}_image.nii')
        lbl = str(dirp/ f'{b}_label.nii')
        if os.path.exists(lbl):
            out[b] = (img,lbl,serial)
    return out

allp = pairs(data)
train, test = [], []
for k,(img,lbl,serial) in allp.items():
    rec = {"image": img, "label": lbl, "id": k}
    (test if serial in test_serials else train).append(rec)

Path('datalist_train.json').write_text(json.dumps(train, indent=2))
Path('datalist_test.json').write_text(json.dumps(test, indent=2))
print(len(train), len(test))
```

## Transforms (3D segmentation)
Use MONAI dict transforms. Start from the mednist tutorial’s patterns and adapt to 3D:
```python
from monai.transforms import (
    LoadImaged, EnsureChannelFirstd, Orientationd,
    ScaleIntensityRanged, RandSpatialCropd, RandFlipd, Compose
)

roi = (112,112,80)
train_transforms = Compose([
    LoadImaged(keys=['image','label']),
    EnsureChannelFirstd(keys=['image','label']),
    Orientationd(keys=['image','label'], axcodes='RAS'),
    # robust scaling proxy: clip range learned from WP5 stats
    ScaleIntensityRanged(keys=['image'], a_min=-3, a_max=8.5, b_min=0.0, b_max=1.0, clip=True),
    RandFlipd(keys=['image','label'], spatial_axis=0, prob=0.5),
    RandFlipd(keys=['image','label'], spatial_axis=1, prob=0.5),
    RandFlipd(keys=['image','label'], spatial_axis=2, prob=0.5),
    RandSpatialCropd(keys=['image','label'], roi_size=roi, random_size=False),
])

val_transforms = Compose([
    LoadImaged(keys=['image','label']),
    EnsureChannelFirstd(keys=['image','label']),
    Orientationd(keys=['image','label'], axcodes='RAS'),
    ScaleIntensityRanged(keys=['image'], a_min=-3, a_max=8.5, b_min=0.0, b_max=1.0, clip=True),
])
```

Mask/ignore class 6 before Dice metrics:
```python
import torch
def mask_ignore_class6(batch):
    # batch['label']: (B,1,X,Y,Z) integer labels
    lbl = batch['label']
    return (lbl != 6)
```

## Networks & Inference
Good starting choices for 3D:
- `monai.networks.nets.UNet`
- `monai.networks.nets.DynUNet`
- `monai.networks.nets.SegResNet`
- `monai.networks.nets.SwinUNETR` (transformer; higher memory)

Sliding window inference for variable sizes:
```python
from monai.inferers import sliding_window_inference
pred = sliding_window_inference(image, sw_roi_size=(112,112,80), sw_batch_size=1, predictor=net)
```

## MONAI Model Zoo / Bundles (pretrained weights)
MONAI pre-trained models are distributed as Bundles. Typical workflow:
1) Download a bundle (outside the restricted environment if needed).
2) Load the network and weights; adapt input channels/classes.
3) Run inference on WP5, then fine‑tune.

Pointers:
- Look for 3D segmentation bundles (UNet/DynUNet/SegResNet/SwinUNETR for CT/MRI). Ensure `in_channels=1` and `out_channels=5` (if merging `6→4`) or `6` if keeping a separate class without ignore.
- If the bundle expects multi‑class with different classes, replace the last layer and load weights with non‑strict state dict.

Example (conceptual):
```python
from monai.networks.nets import UNet
import torch

net = UNet(
    spatial_dims=3, in_channels=1, out_channels=5,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)
)
# load state dict from a compatible UNet bundle checkpoint (strict=False)
sd = torch.load('path/to/pretrained_unet_ckpt.pt', map_location='cpu')
net.load_state_dict(sd.get('state_dict', sd), strict=False)
```

If you can enable the Bundle runner:
```bash
python -m monai.bundle run \
  --meta_file configs/metadata.json \
  --config_file configs/inference.json \
  --ckpt_file models/model.pt
```

## Training Loops (sketch)
Use standard PyTorch/MONAI training, with CE (ignore_index=6) + Dice over 0..4:
```python
import torch, monai
from monai.data import Dataset, DataLoader
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.transforms import Compose
import json

train = json.load(open('datalist_train.json'))
val = json.load(open('datalist_test.json'))  # or a held‑out val split

ds_train = Dataset(train, transform=train_transforms)
dl_train = DataLoader(ds_train, batch_size=2, shuffle=True, num_workers=4)
ds_val = Dataset(val, transform=val_transforms)
dl_val = DataLoader(ds_val, batch_size=1)

net = UNet(spatial_dims=3, in_channels=1, out_channels=5,
           channels=(16,32,64,128,256), strides=(2,2,2,2)).cuda()

ce = torch.nn.CrossEntropyLoss(ignore_index=6)
dice = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
metric = DiceMetric(include_background=True, reduction="mean")

for epoch in range(50):
    net.train()
    for batch in dl_train:
        img = batch['image'].cuda()
        lbl = batch['label'].long().cuda()  # shape (B,1,X,Y,Z)
        out = net(img)
        # CE expects class dim: (B,C,...) and target (B,...)
        ce_loss = ce(out, lbl.squeeze(1))
        dice_loss = dice(out, lbl)
        loss = 0.5*ce_loss + 0.5*dice_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()

    # val (compute Dice over 0..4 only)
    net.eval();
    with torch.no_grad():
        for batch in dl_val:
            img = batch['image'].cuda(); lbl = batch['label'].long().cuda()
            mask = (lbl != 6)
            out = net(img)
            # convert predictions and labels to one‑hot for classes 0..4
            pred5 = torch.softmax(out[:, :5], dim=1)
            lbl5 = torch.clamp(lbl, 0, 4)
            metric(y_pred=pred5, y=monai.networks.one_hot(lbl5, num_classes=5), mask=mask)
        print('val dice:', metric.aggregate().item()); metric.reset()
```

## Plans for Your 3 Tasks

1) Directly use pretrained model zoo models on WP5
- Pick a 3D bundle compatible with single‑channel input.
- Replace final layer to match WP5 classes (5 if merging `6→4`), load pretrained weights with `strict=False`.
- Run sliding‑window inference; report metrics over classes `0..4` (ignore 6).
- Risk: domain shift from CT/MRI to industrial X‑ray; expect low baseline but useful for comparison.

2) Fine‑tune pretrained models on full WP5 train (380)
- Initialize from the same pretrained backbone; use learning rate 1e‑4 to 3e‑4, patch size `(112,112,80)`.
- Augment with flips; consider RandGaussianNoise/Intensity if helpful.
- Loss: CE(ignore 6) + Dice(0..4). Train 100–300 epochs or 20k–50k iterations depending on batch.
- Validate on the 180 test set; save the best checkpoint by Dice.

3) Few‑shot fine‑tune (≈1% of train ≈ 4 samples)
- Sample 1% from the 380 train (stratify over serial numbers if possible).
- Freeze early layers (encoder) for first 50–100 epochs, then unfreeze.
- Heavier regularization (weight decay), stronger data aug.
- Consider pseudo‑labeling unlabeled train volumes after initial fine‑tune.

Sampling snippet:
```python
import json, random
train = json.load(open('datalist_train.json'))
random.seed(42)
few = sorted(random.sample(range(len(train)), max(4, len(train)//100)))
json.dump([train[i] for i in few], open('datalist_train_1pct.json','w'), indent=2)
```

## Notebook usage
- Use `mednist_tutorial.ipynb` as a template for data dicts, Compose transforms, and DataLoader patterns; adapt to 3D transforms above.
- For experiments, create new notebooks copying its structure; keep logging to `runs/`.

## Reproducibility & Logging
- Set seeds: `torch`, `numpy`, `monai.utils.set_determinism(seed=42)`.
- Log with TensorBoard (`SummaryWriter`) or MONAI handlers; store checkpoints with epoch, Dice score, and config snapshot.

### Run/Log Folder Policy (local convention)
- Training stdout/stderr should be written inside the run’s own folder (the same folder that contains checkpoints and metrics), typically as `train.log`.
- Evaluation/inference stdout/stderr should be written inside the corresponding `*_eval` folder (the same folder that contains `metrics/summary.json` and optional `preds/`), typically as `eval.log`.
- For re-evaluations, reuse the original run folder name and append `_eval` (do not append timestamps). Overwrite `metrics/summary.json` when re-running.

## Pitfalls
- Ignoring class 6: ensure consistency across loss and metrics.
- Memory: 3D transformers need larger GPUs; reduce patch size or use UNet/DynUNet if constrained.
- Spacing: NIfTI headers often are 1.0mm; re-spacing not required unless you want standardization.

## Training with Supervoxel-Voted Labels

You can train segmentation models using supervoxel-voted labels instead of original ground truth labels. This is useful for:
- Training with pseudo-labels generated from supervoxel voting
- Exploring the upper bound of supervoxel-based segmentation quality
- Creating baseline comparisons for weakly-supervised approaches

### How It Works

The training script (`train_finetune_wp5.py`) supports an **optional** `--train_label_override_dir` flag that replaces GT labels with alternative labels (e.g., supervoxel-voted labels).

**IMPORTANT**: This is completely optional. If you don't specify this flag, training uses the original GT labels as normal.

### Label Format Requirements

Override labels can be in either format:
- **`.npy` format** (NumPy arrays) - e.g., supervoxel-voted labels: `<case_id>_labels.npy`
- **`.nii` format** (NIfTI) - e.g., GT-like format: `<case_id>_label.nii`

MONAI's `LoadImaged` transform automatically handles both formats. No conversion is needed.

**Technical details**:
- Override labels must match original GT dimensions exactly
- Values should be in the same range as GT (0-4 for WP5)
- Orientation should be RAS (same as training pipeline)
- The override function matches cases using the `"id"` field from datalist

### Usage Example

Using the best supervoxel configuration from parameter sweep (slic mode, 20k segments):

```bash
# Activate environment
. /home/peisheng/MONAI/venv/bin/activate

# Train with supervoxel-voted labels (fully supervised with SV labels)
python3 train_finetune_wp5.py \
  --mode train \
  --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
  --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
  --train_label_override_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n20000_c0.05_s1.0_ras2_voted \
  --output_dir runs/train_sv_best_slic_n20000 \
  --epochs 20 \
  --batch_size 2 \
  --num_workers 4 \
  --init scratch \
  --net basicunet \
  --norm clip_zscore \
  --roi_x 112 --roi_y 112 --roi_z 80 \
  --log_to_file
```

**Without the flag** (standard GT training):
```bash
# Same command but WITHOUT --train_label_override_dir uses original GT labels
python3 train_finetune_wp5.py \
  --mode train \
  --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset \
  --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json \
  --output_dir runs/train_baseline_gt \
  --epochs 20 \
  --batch_size 2 \
  --num_workers 4 \
  --init scratch \
  --net basicunet
```

### Available Supervoxel Label Directories

From the parameter sweep (see `runs/sv_sweep_ras2_summary.md` for full results):
- **Best overall**: `/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n20000_c0.05_s1.0_ras2_voted` (Dice: 0.9149)
- **Balanced**: `/data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted` (Dice: 0.9089)
- **Legacy 5k**: `/home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted` (documented earlier)

All contain 380 train cases with voted labels in `.npy` format.

### Validation

The script validates that:
1. All training cases have corresponding override label files
2. Files exist at the expected paths (handles naming variations automatically)
3. If any files are missing, training aborts with a clear error message

You'll see confirmation output when override is active:
```
============================================================
LABEL OVERRIDE ENABLED
============================================================
Overriding 380 training labels from:
  /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n20000_c0.05_s1.0_ras2_voted
✓ Successfully mapped all 380 cases to override labels
============================================================
```

### Key Difference from GT Training

| Aspect | GT Training (default) | SV Override Training |
|--------|----------------------|---------------------|
| **Label source** | Original manual annotations | Supervoxel-voted labels |
| **Label path** | `<case_id>_label.nii` | `<case_id>_labels.npy` |
| **Label quality** | Ground truth | Pseudo-labels (noisy) |
| **Command flag** | (none, default behavior) | `--train_label_override_dir <path>` |
| **Use case** | Standard supervised learning | Upper bound for SV quality, baseline comparisons |

### Batch Training with All SV Configurations

To train models using all 30 supervoxel configurations from the parameter sweep (distributed across 2 GPUs):

```bash
# Activate environment
. /home/peisheng/MONAI/venv/bin/activate

# Run all 30 training jobs (15 per GPU, sequential within each GPU)
bash scripts/train_all_sv_configs.sh

# Optional: Preview commands without executing
bash scripts/train_all_sv_configs.sh --dry-run

# Optional: Resume training (skip already completed runs)
bash scripts/train_all_sv_configs.sh --resume
```

**What it does:**
- Finds all 30 SV configurations in `/data3/wp5/monai-sv-sweeps/`
- Splits jobs: 15 on GPU 0, 15 on GPU 1
- Runs sequentially within each GPU (parallel across GPUs)
- Each run trains for 20 epochs with batch_size=2
- Saves to unique directories: `runs/train_sv_<config>_e20/`
- Each run logs to its own `train.log` file

**Estimated time:** ~15-20 hours per GPU (depends on hardware)

**Output structure:**
```
runs/
├── train_sv_slic_n2000_c0.05_s1.0_e20/
├── train_sv_slic_n4000_c0.05_s1.0_e20/
├── train_sv_slic-grad-mag_n2000_c0.05_s1.0_e20/
└── ... (30 total directories)
```

**To monitor progress:**
```bash
# Watch GPU 0 jobs
tail -f runs/train_sv_slic_n*_e20/train.log

# Check how many completed
ls -d runs/train_sv_*_e20/ | wc -l
```

### Summarizing Training Results

After all 30 training runs complete, use the summarization script to generate a comprehensive analysis:

```bash
# Activate environment
. /home/peisheng/MONAI/venv/bin/activate

# Generate summary report and plots
python3 scripts/summarize_sv_training_results.py
```

**What it produces:**
- **Report**: `runs/sv_training_results_summary.md` (self-contained markdown with embedded plots)
- **Plots**: `runs/sv_training_plots/` (5 visualization figures)
  - Dice/IoU vs n_segments trends
  - Per-class performance breakdown
  - Mode comparison (slic, slic-grad-mag, slic-grad-vec)
  - Training convergence samples

**Features:**
- Automatically loads correct baseline values from experiment directories:
  - 100% GT: `runs/grid_clip_zscore/scratch_subset_100/eval_20251021-120429/`
  - 10% GT: `runs/fewshot_grid_clip_zscore/points_10_d1_proportional/`
  - 1% GT: `runs/fp_1pct_global_d0_20251021-153502/`
- Ranks all 30 configurations by performance
- Compares SV-trained models vs. GT baselines
- Generates publication-ready plots with seaborn styling
- Creates self-contained report with relative image paths (view in any markdown viewer)

**Example findings:**
- Best: slic n=18000 → 0.8656 Dice (99.3% of 100% GT baseline)
- Standard SLIC outperforms geometry-aware variants
- All configs achieve 95-99% of fully-supervised performance

## Strategic Sparse Supervoxel Labeling (0.1% Budget)

A weakly-supervised approach that combines sparse annotation (0.1% of voxels) with supervoxel structure for label propagation.

### Overview

**Goal**: Train segmentation models using minimal annotations (0.1% of voxels ≈ 1,100 voxels/case) by:
1. **Strategic sampling**: Distribute labeled voxels among supervoxels (max 1 per SV)
2. **Label propagation**: Use k-NN to propagate labels from labeled SVs to unlabeled SVs

**Key Features**:
- Max 1 labeled voxel per supervoxel (no voting, direct assignment)
- Prioritizes foreground borders (high gradient)
- Prioritizes rare classes (3, 4 get 2x weight)
- Multi-k experiments (k = 1,3,5,7,10,15,20,25,30,50)

**Status**: ✅ Implementation complete, tests passing (12/12), ready to run experiments

**Recent Fixes** (2025-11-10):
- Fixed split config parsing to handle `test_serial_numbers` format
- Added support for `data/` subdirectory in dataset structure
- Fixed class 0 (background) exclusion bug - now includes all classes
- All 380 training cases now load correctly with proper class distribution

### Complete Pipeline (Recommended)

**Single command that runs everything: sampling → propagation → training**

```bash
bash scripts/run_strategic_sparse_complete.sh
```

**What it does**:
1. Strategic seed sampling (~1,100 seeds per case, stratified by class)
2. Multi-k label propagation (k=1,3,5,7,10,15,20,25,30,50)
3. Parallel training across 2 GPUs for all 10 k variants
   - 20 epochs, batch_size=2, BasicUNet
   - Loss: 0.5 × CrossEntropy + 0.5 × Dice (ignore class 6)
4. Estimated time: 12-20 hours total (pipeline ~3h + training ~10-15h)

**Options**:
```bash
# Use custom output directory
bash scripts/run_strategic_sparse_complete.sh --output_dir runs/my_experiment

# Only run pipeline, skip training
bash scripts/run_strategic_sparse_complete.sh --skip-training

# Only run training (if pipeline already done)
bash scripts/run_strategic_sparse_complete.sh --skip-pipeline

# Show help
bash scripts/run_strategic_sparse_complete.sh --help
```

### Manual Steps (Alternative)

If you prefer to run steps separately or need more control:

**Step 1: Strategic Seed Sampling**

```bash
python3 scripts/sample_strategic_sv_seeds.py --sv_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json --budget_ratio 0.001 --class_weights 0.1,1,1,2,2 --output_dir runs/strategic_seeds_0p1pct --seed 42
```

**Outputs**:
- `<case_id>_strategic_seeds.npy` - Binary mask of sampled voxels
- `<case_id>_sv_labels_sparse.json` - Sparse SV labels (1:1 mapping, no voting)
- `<case_id>_seeds_meta.json` - Statistics (includes class distribution)
- `summary_stats.json` - Overall statistics

**Note**: Class weights are `0.1,1,1,2,2` for classes 0,1,2,3,4 respectively. Class 0 (background) gets lower weight (0.1) due to its large proportion and ease of learning.

**Step 2: Multi-k Label Propagation**

```bash
python3 scripts/propagate_sv_labels_multi_k.py --sv_dir /data3/wp5/monai-sv-sweeps/sv_fullgt_slic_n12000_c0.05_s1.0_ras2_voted --seeds_dir runs/strategic_seeds_0p1pct --k_values 1,3,5,7,10,15,20,25,30,50 --output_dir runs/sv_sparse_prop_0p1pct_strategic --seed 42
```

**Outputs**:
- `cases/<case_id>/propagated_k01_labels.npy` - Dense labels for k=1
- `cases/<case_id>/propagated_k03_labels.npy` - Dense labels for k=3
- ... (one per k value)
- `k_variants/k01/<case_id>_labels.npy` - Symlinks for training
- `propagation_summary.json` - Statistics

**Step 3: Training on All K Variants (Parallel)**

**Note**: This step is included in the complete pipeline script above. Only run manually if needed.

Train models for all 10 k values across 2 GPUs:

```bash
bash scripts/train_all_k_variants.sh runs/strategic_sparse_0p1pct_k_multi/k_variants
```

**What it does**:
- Distributes 10 k values across 2 GPUs (5 per GPU)
- Sequential execution within each GPU
- Parallel across GPUs
- Each run: 20 epochs, batch_size=2, BasicUNet
- Loss: 0.5 × CrossEntropy + 0.5 × Dice (ignore class 6)
- Saves to `runs/train_sv_sparse_k{01,03,05,...}/`

**Monitor progress**:
```bash
# Watch GPU 0 jobs
tail -f runs/train_sv_sparse_k01/train.log

# Check completion
ls -d runs/train_sv_sparse_k*/ | wc -l
```

**Optional flags**:
```bash
# Preview commands without executing
bash scripts/train_all_k_variants.sh <k_variants_dir> --dry-run

# Resume training (skip completed runs)
bash scripts/train_all_k_variants.sh <k_variants_dir> --resume
```

### Directory Structure

```
runs/sv_sparse_prop_0p1pct_strategic/
├── strategic_seeds/              # Step 1 outputs
│   ├── SN13B0_..._strategic_seeds.npy
│   ├── SN13B0_..._sv_labels_sparse.json
│   ├── SN13B0_..._seeds_meta.json
│   └── summary_stats.json
├── cases/                         # Step 2 outputs
│   ├── SN13B0_I17_3D_B1_1B250409/
│   │   ├── sparse_sv_labels.json
│   │   ├── propagated_k01_labels.npy
│   │   ├── propagated_k03_labels.npy
│   │   ├── ...
│   │   ├── propagated_k50_labels.npy
│   │   └── propagation_meta.json
│   └── ... (380 cases)
├── k_variants/                    # Training directories (symlinks)
│   ├── k01/
│   │   ├── SN13B0_..._labels.npy -> ../../cases/SN13B0.../propagated_k01_labels.npy
│   │   └── ... (380 cases)
│   ├── k03/
│   ├── k05/
│   ├── ...
│   └── k50/
└── propagation_summary.json

runs/train_sv_sparse_k01/          # Training outputs
    ├── best.ckpt
    ├── last.ckpt
    ├── train.log
    └── metrics/
runs/train_sv_sparse_k03/
...
runs/train_sv_sparse_k50/
```

### Expected Performance

| Method | Training Labels | Test Dice (Expected) | % of 100% GT |
|--------|----------------|----------------------|--------------|
| 100% GT | 380 dense labels | 0.8718 | 100% |
| 12k SV full GT | 380 SV-voted | 0.9089 | 104.3% |
| 1% sparse points | ~1,100 points × 380 | 0.8310 | 95.3% |
| **0.1% → SV prop (k=5)** | **~1,100 points × 380** | **~0.78-0.82 (?)** | **~90-94% (?)** |

**Hypothesis**: Optimal k should be 5-10 (balances locality vs robustness).

### Algorithm Details

**Strategic Sampling**:
```python
# For each foreground supervoxel:
#   1. Score voxels by:
#      - Class weight: {1:1, 2:1, 3:2, 4:2} (prioritize rare)
#      - Gradient magnitude (borders)
#      - Distance to centroid (representativeness)
#   2. Select top 1 voxel per SV
# Rank all SVs globally by score, select top N within budget
```

**k-NN Propagation**:
```python
# For each unlabeled SV:
#   1. Find k nearest labeled SVs (by centroid distance)
#   2. Weighted vote: weight = 1 / (distance + ε)
#   3. Assign majority label
```

### Testing

Run comprehensive tests:
```bash
pytest tests/test_strategic_sparse_sv.py -v
```

**Test coverage**:
- Strategic sampling (max 1 per SV, budget, FG priority, rare class priority, gradient)
- Multi-k propagation (all k values, sparse preservation, voting, distance weighting)
- Helper functions (centroids, gradients, SV-to-dense conversion)

### Troubleshooting

**"Processing 0 cases from train split"**

Fixed in commits c784aad3 and f14ce60e. The issue was that:
1. Split config uses `{"test_serial_numbers": [9, 12, 15, ...]}` format
2. Dataset files are in `data_root/data/` subdirectory
3. Script now searches both `data_root/` and `data_root/data/` for image files
4. Case IDs are extracted from serial numbers (SN13B0_... → serial 13)

Verify the fix:
```bash
python3 -c "import os; from pathlib import Path; print(f'Cases: {len([n for n in os.listdir(Path(\"/data3/wp5/wp5-code/dataloaders/wp5-dataset/data\")) if n.endswith(\"_image.nii\")])}')"
```

Expected: `Cases: 560` (380 train + 180 test)

**"No class 0 (background) in sampled seeds"**

Fixed in commit [TBD]. The root cause was that even background-dominant SVs were contributing foreground seeds because:
1. Per-SV selection scored all voxels together
2. FG voxels had 10-20x higher class weights (1.0-2.0 vs 0.1)
3. FG voxels always won within mixed SVs

**Solution**:
1. Determine each SV's dominant class (via majority vote)
2. Only sample voxels of that dominant class from each SV
3. Use stratified sampling to allocate budget proportionally to GT class frequency
4. Result: Seed distribution matches GT distribution (~64% class 0, ~18% class 1, etc.)

Verify class distribution in a sample output:
```bash
python3 -c "import json; f='runs/strategic_sparse_0p1pct_k_multi/strategic_seeds/summary_stats.json'; print(json.dumps(json.load(open(f))['total_class_distribution'], indent=2))"
```

Expected: Class 0 should have ~64% of seeds

## Where to look in this repo
- `mednist_tutorial.ipynb` — reference for IO/transform patterns.
- `WP5_Segmentation_Data_Guide.md` — authoritative WP5 data details, label policy, split.

## Visualization (2D/3D viewers)

We include Streamlit apps to visually inspect GT labels, model predictions, and supervoxel outputs. These assume NIfTI images/labels are canonicalized to RAS (as in training via `Orientationd(axcodes='RAS')`).

Install (one line):
- Python deps: `pip install streamlit matplotlib scikit-image nibabel pyvista streamlit-pyvista plotly`

Tools
- GT vs Supervoxel‑Labeled (voted) — side‑by‑side 2D and 3D pair view
  - Script: `scripts/vis_gt_vs_sv_streamlit.py`
  - SV input: a folder with `<id>_labels.npy` produced from supervoxel voting, e.g. `/home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted`
  - Datalist: for case selection; use `datalist_train.json` for the above folder
  - Launch (with train split): `python3 -m streamlit run scripts/vis_gt_vs_sv_streamlit.py -- --sv-dir /home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted --datalist datalist_train.json`
  - Notes:
    - 2D overlays use class colors (1 red, 2 green, 3 blue, 4 yellow; 6 gray), with optional mismatch highlight.
    - 3D offers Matplotlib (static) and PyVista (interactive). PyVista volume/surface axes are aligned to (X,Y,Z); spacing uses NIfTI affine norms (dx,dy,dz).
    - Metrics shown per case: Dice and IoU for classes 0..4, ignoring label 6.

- Fixed eval comparisons (legacy) — GT vs up to 4 prediction folders
  - Script: `scripts/vis_wp5_streamlit.py`
  - Launch: `python3 -m streamlit run scripts/vis_wp5_streamlit.py`
  - Configure the fixed `RUNS` mapping inside the script to your prediction folders.

- Supervoxel ID viewer — inspect unlabeled `sv_ids` and optional fully labeled SV volumes
  - Script: `scripts/vis_sv_ids_streamlit.py`
  - Launch (example): `python3 -m streamlit run scripts/vis_sv_ids_streamlit.py -- --sv-dir /home/peisheng/MONAI/runs/sv_fill_5k_nofill_ras2 --data-root /data3/wp5/wp5-code/dataloaders/wp5-dataset --datalist datalist_train.json`
  - Features: 3D boundary meshes by ID, sampled SV surfaces, and 2D colorized ID slices.

Split matching and paths
- The folder `/home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted` contains voted voxel labels for the train split (≈380). Use `datalist_train.json` to see all cases.
- If you use `datalist_test.json`, ensure you have a corresponding voted folder for the test cases; otherwise no overlap will be listed.

Troubleshooting
- “No cases match SV‑labeled availability”: switch to `datalist_train.json` or point the app to a matching voted folder.
- PyVista/Matplotlib mismatch: the PyVista grid now uses dimensions `(X+1,Y+1,Z+1)` and spacing `(dx,dy,dz)` to match Matplotlib’s (X,Y,Z) meshing.
- Empty surfaces: try lowering decimation, disabling volume overlay, and reducing 3D downsample to 1.

## Supervoxel Voted Label Evaluation
- Script: `scripts/eval_sv_voted_wp5.py`
- Purpose: evaluate per-voxel supervoxel-voted labels (`<id>_labels.npy`) against GT with WP5 policies, and compute voting diagnostics.
- Inputs:
  - `--sv-dir`: folder with `<id>_labels.npy` (e.g., `/home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted`).
  - `--sv-ids-dir`: folder with `<id>_sv_ids.npy` to enable entropy/purity diagnostics (e.g., `/home/peisheng/MONAI/runs/sv_fullgt_5k_ras2`).
  - `--datalist`: `datalist_train.json` (train split recommended for this folder).
  - Policy: evaluate classes 0..4; ignore voxels with `gt==6`; both‑empty=1.0; optional HD95 with `--heavy --hd_percentile 95`.
- Parallel run (train split):
  - `python3 scripts/eval_sv_voted_wp5.py --sv-dir /home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted --sv-ids-dir /home/peisheng/MONAI/runs/sv_fullgt_5k_ras2 --datalist datalist_train.json --output_dir /home/peisheng/MONAI/runs/sv_fullgt_5k_ras2_voted_eval --ignore-class 6 --num_workers 8 --progress --log_to_file`
- Outputs:
  - `metrics/per_case.csv` — per-case Dice/IoU (and optional HD/ASD) for classes 0..4 + voxel accuracy.
  - `metrics/summary.json` — dataset averages and (if `--sv-ids-dir` provided) SV diagnostics: mean and voxel‑weighted mean normalized entropy, purity, and entropy fractions at 0.3/0.5.
  - `eval.log` — tee of stdout/stderr inside the eval folder.
  - Console also prints the dataset‑level entropy means after Dice/IoU.

## Evaluation & Metrics (official)
- Use `scripts/eval_wp5.py` for all evaluations. In‑script trainer evaluation has been removed.
- Classes to evaluate: 0–4 (background included), ignore voxels with label 6.
- When both prediction and GT are empty for a class in a volume, count the score as 1.0 (both‑empty=1.0).
- Report per‑class and the unweighted average across classes 0–4.
- Metrics to compute:
  - Dice coefficient (per class and mean)
  - Jaccard Coefficient / IoU (per class and mean)
  - Hausdorff Distance (HD). Default: use HD95 for robustness; to match prior baselines, set percentile=100 for full HD.
  - Average Surface Distance (ASD)
- MONAI helpers:
  - Dice: `monai.metrics.DiceMetric`
  - IoU: `monai.metrics.MeanIoU` (on one‑hot or discrete preds)
  - HD: `monai.metrics.HausdorffDistanceMetric(percentile=100.0)` for HD; `percentile=95.0` for HD95
  - ASD: `monai.metrics.SurfaceDistanceMetric`

Masking/selection notes
- Convert predictions to 5 classes (0..4) via argmax on `out[:, :5]` if your head has >5 channels.
- Clamp ground truth to 0..4 for IoU/Dice; for HD/ASD compute per‑class on binarized masks; mask out voxels where `gt==6`.

Baselines (corrected from actual experiments) — evaluate over classes 0..4, ignore class 6

**Source paths:**
- 100% GT: `runs/grid_clip_zscore/scratch_subset_100/eval_20251021-120429/metrics/summary.json`
- 10% GT: `runs/fewshot_grid_clip_zscore/points_10_d1_proportional/metrics/summary.json`
- 1% GT: `runs/fp_1pct_global_d0_20251021-153502/metrics/summary.json`

100% (full fine‑tune from scratch)

| Class | Dice | IoU | HD95 | ASD |
|---:|---:|---:|---:|---:|
| 0 | 0.9875 | 0.9754 | 1.6118 | 0.2869 |
| 1 | 0.8656 | 0.8038 | 17.4005 | 2.7269 |
| 2 | 0.9148 | 0.8484 | 6.4946 | 1.3603 |
| 3 | 0.8440 | 0.7989 | 1.1987 | 0.3054 |
| 4 | 0.7471 | 0.6548 | 7.4858 | 2.6580 |
| **average** | **0.8718** | **0.8163** | **6.8383** | **1.4675** |

10% GT (few‑shot, proportional sampling)

| Class | Dice | IoU |
|---:|---:|---:|
| 0 | 0.9943 | 0.9887 |
| 1 | 0.8760 | 0.8208 |
| 2 | 0.9163 | 0.8517 |
| 3 | 0.7935 | 0.7478 |
| 4 | 0.7759 | 0.6878 |
| **average** | **0.8712** | **0.8193** |

1% GT (few‑shot, global sampling, no dilation)

| Class | Dice | IoU |
|---:|---:|---:|
| 0 | 0.9824 | 0.9656 |
| 1 | 0.8571 | 0.7886 |
| 2 | 0.9036 | 0.8291 |
| 3 | 0.6642 | 0.6184 |
| 4 | 0.7475 | 0.6446 |
| **average** | **0.8310** | **0.7693** |

Replicating baselines
- Use the same split (train=380, test=180), same class policy (0..4 evaluated, 6 ignored), and the same metric definitions
- The 100% GT baseline uses HD95 (percentile=95.0); 10% and 1% GT do not compute HD/ASD
- Both‑empty=1.0 policy is always applied
- The `summarize_sv_training_results.py` script automatically reads these baseline values from the source paths above for accurate comparisons

External precompute (1% points, no dilation)
- Precompute: `. /home/peisheng/MONAI/venv/bin/activate && python3 scripts/precompute_sup_masks.py --mode few_points_global --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json --subset_ratio 1.0 --ratio 0.01 --dilate_radius 0 --balance proportional --seed 42 --fp_sample_mode uniform_all --out_dir runs/sup_masks_1pct_uniform_all`
- Train: `. /home/peisheng/MONAI/venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python3 -u train_finetune_wp5.py --mode train --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json --output_dir runs/fewpoints_01pct_static_from_dir --epochs 20 --batch_size 2 --num_workers 4 --init scratch --net basicunet --subset_ratio 1.0 --seed 42 --fewshot_mode few_points --fewshot_ratio 0.01 --fewshot_static --sup_masks_dir runs/sup_masks_1pct_uniform_all --pseudo_weight 0.0 --fg_crop_prob 0.0 --coverage_mode seeds --norm clip_zscore --roi_x 112 --roi_y 112 --roi_z 80 --log_to_file`
