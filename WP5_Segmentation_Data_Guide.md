**WP5 Segmentation Data Guide**

- Location: `/data3/wp5/wp5-code/dataloaders/wp5-dataset/data`
- Purpose: Clear, correct spec for training segmentation models in external codebases (e.g., MONAI, PyTorch, others).
- Snapshot validated: 560 image–label pairs (1120 `.nii` files) present at analysis time.

**Dataset Overview**
- Total files: 1120 NIfTI (`.nii`) files.
- Pairs: 560 image–label pairs; each image has a matching label with identical shape.
- Dimensions: Mixed “3D” and “2.5D” products indicated in filenames.
  - Image counts by dimension: `3D` = 314, `2.5D` = 246.
- Design styles in filenames (images): `A1,A2,A3,B1,C1,C2,D1,D2,D3` with varying frequencies.
- Predefined split config: `/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json`.
 - Default split for experiments (from config): train 380, test 180 (sum 560).

**File Naming Convention**
- Standard (with design attributes): `SN{serial}B{bump}_I{image}_{dimension}_{style}_{batch}B{date}_{type}.nii`
- Legacy (no design attributes): `SN{serial}B{bump}_I{image}_{batch}B{date}_{type}.nii`
- Test legacy: `SN{serial}B{bump}_I{image}_T{date}_{type}.nii`
- Components:
  - `SN{serial}`: product unit ID (e.g., 2–88 observed)
  - `B{bump}`: bump index (zero-based)
  - `I{image}`: image index within the product
  - `{dimension}`: `3D` or `2.5D`
  - `{style}`: design variant (`A1,B1,C1,...`)
  - `{batch}B{date}` or `T{date}`: annotation batch or test format + YYMMDD date
  - `{type}`: `image` or `label`

**Data Format (Validated From Headers + Samples)**
- Container: NIfTI-1 single-file (`magic` = `n+1\0`, `vox_offset` ≈ 352.0).
- Image dtype: float32 (`datatype_code` 16).
- Label dtype: unsigned 16-bit integer (`datatype_code` 512, stored as integer masks).
- Pixel spacing (`pixdim[1:4]`): isotropic 1.0,1.0,1.0 across sampled headers.
- Shapes: variable; examples (X,Y,Z) include `(112,101,95)`, `(113,103,97)`, `(64,80,61)`, many others.
  - Global ranges observed over all images: X ∈ [20,137], Y ∈ [45,191], Z ∈ [38,152].
- Intensity stats (50 evenly spaced samples):
  - mean: median ≈ 1.05 (min 0.71, max 28990.7)
  - std: median ≈ 1.90 (min 1.35, max 10486.8)
  - min intensity: median ≈ -3.05 (min -5.52, max 4968.0)
  - max intensity: median ≈ 8.40 (min 6.19, max 62907.0)
- Implication: intensity scale varies across scans; recommend robust per-sample normalization (see “Normalization”).

**Segmentation Labels**
- Verified unique label values across the full dataset: `{0, 1, 2, 3, 4, 6}`.
  - Background: `0` (explicitly, class 0 is background).
  - Foreground classes present in most volumes: `1, 2, 4`.
  - Class `3` present in many 3D volumes (333/560 volumes include value `3`). Internal test code suggests class 3 corresponds to “void”.
  - Class `6` appears in a minority of cases (49/560 volumes include value `6`). Semantics not documented in repo.
- Per-volume examples:
  - 3D samples often contain `{1,2,3,4}`.
  - 2.5D samples often contain `{1,2,4}` (class `3` frequently absent).

Important: Because value `6` is used but undocumented, choose one of the following strategies explicitly in your pipeline:
- Ignore label 6 during loss/metrics (preferred when your framework supports `ignore_index`).
- Map `6 → 4` if domain owners confirm equivalence to an existing class.
- As a fallback, map `6 → 0` (background), but this discards annotated structure and is not recommended without confirmation.

Example mappers:
- Ignore 6 (PyTorch CE loss): use `ignore_index=6` and keep num_classes ≥ 7 if using CE directly with logits, or remap to dense indices and set mask.
- Remap to dense 0..K-1 labels:
  - Option A (merge): `{0→0, 1→1, 2→2, 3→3, 4→4, 6→4}` then `num_classes=5`.
  - Option B (keep 6 separate): `{0→0, 1→1, 2→2, 3→3, 4→4, 6→5}` then `num_classes=6`.

**Default Policy (Training and Evaluation)**
- Split: Use the predefined serial-number split in `/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json` resulting in train=380 and test=180 samples.
- Loss: Treat class `6` as ignored by default.
  - CrossEntropy: `ignore_index=6` (or remap 6→4 if using 5-class training).
  - Dice-based losses: mask out voxels where `label == 6` when computing per-class Dice.
- Metrics: Compute metrics over classes `0,1,2,3,4` and ignore class `6` by default.
  - This keeps continuity with prior experiments where the “last class” was ignored.
  - If using MONAI metrics, either drop channel 6 from predictions or mask GT voxels with value 6 before metric computation.

**3D vs 2.5D**
- Both stored as 3D NIfTI volumes with varying shapes.
- Treat both uniformly as 3D volumes at training time; or train separate models per dimension/design if performance differs.
- Counts (images): `3D`=314, `2.5D`=246.

**Pairs and Integrity**
- Each `*_image.nii` has a corresponding `*_label.nii` with identical `(X,Y,Z)`.
- All tested headers valid (`sizeof_hdr=348`, standard fields present).
- Use the helper script in “Verification & Utilities” to re-check on your machine.

**Recommended Normalization**
- Given heavy-tailed intensity distribution and outliers, use robust per-sample normalization:
  - Clip to [p1, p99] of the volume, then Z-score: `(x - mean) / std`.
  - This matches the dataset loader’s `clip_zscore` transform used in `train_mmt.py`.
- If training a cross-scan model, prefer per-sample over global normalization due to varying scales.

**Using This Data in Other Codebases**

- Data list generation (Python):
  ```python
  import os
  data_root = '/data3/wp5/wp5-code/dataloaders/wp5-dataset/data'
  pairs = []
  for name in os.listdir(data_root):
      if name.endswith('_image.nii'):
          base = name[:-10]  # remove '_image.nii'
          img = os.path.join(data_root, f'{base}_image.nii')
          lbl = os.path.join(data_root, f'{base}_label.nii')
          if os.path.exists(lbl):
              pairs.append({'image': img, 'label': lbl, 'id': base})
  ```

- Label mapping utility (choose one strategy):
  ```python
  import numpy as np

  def remap_labels(lbl, strategy='ignore6'):
      # lbl is a numpy array of integer labels
      if strategy == 'ignore6':
          # produce a mask where lbl==6 and use ignore_index=6 in your loss
          return lbl
      elif strategy == 'merge6to4':
          out = lbl.copy()
          out[out == 6] = 4
          return out
      elif strategy == 'six_classes':
          # keep 6 as a distinct class by reindexing to contiguous [0..5]
          table = np.zeros(7, dtype=np.int32)
          table[1] = 1
          table[2] = 2
          table[3] = 3
          table[4] = 4
          table[6] = 5
          # unknowns map to 0
          flat = lbl.reshape(-1)
          flat = table[np.clip(flat, 0, 6)]
          return flat.reshape(lbl.shape)
      else:
          raise ValueError('Unknown strategy')
  ```

- MONAI example (3D training with padding/cropping):
  ```python
  from monai.transforms import (LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
                                ScaleIntensityRanged, RandSpatialCropd, RandFlipd, ToTensord, Compose)
  from monai.data import Dataset, DataLoader

  train_files = pairs  # from above or your split

  # Note: NIfTI headers have pixdim=(1,1,1) already; keep spacing unless you want to standardize.
  # Use robust intensity normalization; here we show percentile scaling as a proxy.
  transforms = Compose([
      LoadImaged(keys=['image', 'label']),
      EnsureChannelFirstd(keys=['image', 'label']),
      Orientationd(keys=['image', 'label'], axcodes='RAS'),
      # Optional Spacingd(keys=['image','label'], pixdim=(1.0,1.0,1.0), mode=('bilinear','nearest')),
      ScaleIntensityRanged(keys=['image'], a_min=-3, a_max=8.5, b_min=0.0, b_max=1.0, clip=True),
      RandFlipd(keys=['image', 'label'], spatial_axis=0, prob=0.5),
      RandFlipd(keys=['image', 'label'], spatial_axis=1, prob=0.5),
      RandFlipd(keys=['image', 'label'], spatial_axis=2, prob=0.5),
      RandSpatialCropd(keys=['image','label'], roi_size=(112,112,80), random_size=False),
      ToTensord(keys=['image', 'label'])
  ])

  ds = Dataset(data=train_files, transform=transforms)
  loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4)
  ```

- PyTorch loss setup with ignore_index (if using cross-entropy):
  ```python
  import torch.nn as nn
  criterion = nn.CrossEntropyLoss(ignore_index=6)  # only if you keep 6 in labels
  # For Dice-based losses, first remap or mask voxels labeled 6.
  ```

- Metric masking example (conceptual):
  ```python
  import torch
  # pred_logits: (B, C, X, Y, Z), gt: (B, 1, X, Y, Z) integer labels
  mask = (gt != 6)  # ignore voxels with class 6
  # restrict to classes 0..4 in both pred and gt for metric computation
  pred_cls = torch.argmax(pred_logits[:, :5], dim=1, keepdim=True)
  gt_clipped = torch.clamp(gt, 0, 4)
  # Apply mask before metric (example for Dice-by-class)
  # ... compute per-class metrics on masked voxels only ...
  ```

- Patch size and padding:
  - Training code in this repo uses patches around `(112,112,80)` and pads volumes if smaller.
  - If your framework requires fixed size, pad to target size before cropping.

- Splitting:
  - Use predefined serial-number-based split for reproducibility: `/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json`.
  - Or stratify by `scan_serial_number` to minimize data leakage across products.

**Verification & Utilities**

- Verify pairs and label set using nibabel (if available):
  ```python
  import os, nibabel as nib, numpy as np
  root = '/data3/wp5/wp5-code/dataloaders/wp5-dataset/data'
  uniq = set()
  for name in os.listdir(root):
      if name.endswith('_label.nii'):
          arr = nib.load(os.path.join(root, name)).get_fdata().astype(np.uint16)
          uniq |= set(np.unique(arr).tolist())
  print('Unique label values:', sorted(uniq))
  ```

- Quick header reader without nibabel (pure Python; used for this report):
  ```python
  import struct, numpy as np, os
  def read_hdr(path):
      with open(path, 'rb') as f: hdr=f.read(348)
      endian = '<' if struct.unpack('<i', hdr[:4])[0]==348 else '>'
      dims = struct.unpack(endian+'8h', hdr[40:56]); shape = tuple(dims[1:4])
      datatype = struct.unpack(endian+'h', hdr[70:72])[0]
      vox_offset = struct.unpack(endian+'f', hdr[108:112])[0]
      return endian, shape, datatype, int(vox_offset)
  ```

**Practical Recommendations**
- Choose one label policy and keep it consistent across training, validation, and metrics:
  - If you cannot confirm class 6 semantics, use ignore-mask during training and evaluation.
  - If you must reduce to 5 classes, merge 6→4 (document this mapping in your experiment log).
- Normalize per-sample with clipping to handle outliers.
- Consider separate experiments for `3D` vs `2.5D` and/or per design style if distribution shift is suspected.
- Use serial-number-stratified splits to prevent leakage across the same product unit.

**Known Inconsistencies in Repo vs Data**
- Some earlier docs/code mention 558 pairs; current dataset has 560 pairs. This guide reflects the current on-disk count.
- Training code in `train_mmt.py` uses `num_classes=5` without handling label `6`. If your run includes class `6`, adopt one of the mapping/ignore strategies above.

**References In-Repo**
- Dataset docs: `/data3/wp5/wp5-code/dataloaders/wp5-dataset/README.md`
- Loader (alt version): `/data3/wp5/wp5-code/dataloaders/wp5-dataset/dataset_loader.py`
- Predefined split: `/data3/wp5/wp5-code/dataloaders/wp5-dataset/3ddl_split_config_20250801.json`
- Example training using this dataset: `/data3/wp5/model_wp5/mmt_wp5_normalized/code/train_mmt.py`

**Contact/Follow-Up**
- To fully resolve class `6` semantics, consult the dataset owners. Until then, prefer `ignore_index` or explicit remap with clear documentation.
