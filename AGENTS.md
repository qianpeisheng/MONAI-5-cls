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

Baselines (as provided) — evaluate over classes 0..4, ignore class 6

100% (full fine‑tune)

| Class | Dice | IoU | HD | ASD |
|---:|---:|---:|---:|---:|
| 0 | 0.98275321 | 0.96620361 | 1.24574252 | 0.27684943 |
| 1 | 0.78495372 | 0.70042153 | 10.54281509 | 4.32888065 |
| 2 | 0.90701377 | 0.83129394 | 3.85318096 | 1.34848441 |
| 3 | 0.82481699 | 0.78822708 | 1.42007550 | 0.36657448 |
| 4 | 0.56270956 | 0.45448813 | 20.66707425 | 7.81830500 |
| average | 0.81244945 | 0.74812686 | 7.54577767 | 2.82781879 |

10% (few‑shot baseline)

| Class | Dice | IoU | HD | ASD |
|---:|---:|---:|---:|---:|
| 0 | 0.84598023 | 0.73461127 | 14.91666670 | 0.00387064 |
| 1 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 0 | 0 |
| 3 | 0.6 | 0.6 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 |
| average | 0.28919605 | 0.26692826 | 2.98333333 | 0.00077413 |

1% (few‑shot baseline)

| Class | Dice | IoU | HD | ASD |
|---:|---:|---:|---:|---:|
| 0 | 0.24273103 | 0.16986430 | 45.86634000 | 1.16762257 |
| 1 | 0.18725551 | 0.10789209 | 44.13901038 | 25.32536418 |
| 2 | 0 | 0 | 0 | 0 |
| 3 | 0.6 | 0.6 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 |
| average | 0.20599731 | 0.17555128 | 18.00107008 | 5.29859735 |

Replicating baselines
- Use the same split (train=380, test=180), same class policy (0..4 evaluated, 6 ignored), and the same metric definitions. If you suspect HD vs HD95 discrepancy, set HD percentile=100 to match these values. Both‑empty=1.0 policy is always applied.

External precompute (1% points, no dilation)
- Precompute: `. /home/peisheng/MONAI/venv/bin/activate && python3 scripts/precompute_sup_masks.py --mode few_points_global --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json --subset_ratio 1.0 --ratio 0.01 --dilate_radius 0 --balance proportional --seed 42 --fp_sample_mode uniform_all --out_dir runs/sup_masks_1pct_uniform_all`
- Train: `. /home/peisheng/MONAI/venv/bin/activate && CUDA_VISIBLE_DEVICES=1 python3 -u train_finetune_wp5.py --mode train --data_root /data3/wp5/wp5-code/dataloaders/wp5-dataset --split_cfg /data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json --output_dir runs/fewpoints_01pct_static_from_dir --epochs 20 --batch_size 2 --num_workers 4 --init scratch --net basicunet --subset_ratio 1.0 --seed 42 --fewshot_mode few_points --fewshot_ratio 0.01 --fewshot_static --sup_masks_dir runs/sup_masks_1pct_uniform_all --pseudo_weight 0.0 --fg_crop_prob 0.0 --coverage_mode seeds --norm clip_zscore --roi_x 112 --roi_y 112 --roi_z 80 --log_to_file`
