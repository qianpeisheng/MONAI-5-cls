from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import nibabel as nib
import numpy as np


@dataclass
class NiftiVolume:
    data: np.ndarray  # 3D array (X,Y,Z)
    affine: np.ndarray
    header: nib.Nifti1Header


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_nifti(path: str | Path) -> NiftiVolume:
    img = nib.load(str(path))
    arr = img.get_fdata()
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI at {path}, got shape {arr.shape}")
    return NiftiVolume(data=arr, affine=img.affine, header=img.header)


def load_label(path: str | Path) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    vol = load_nifti(path)
    # labels expected int types; cast preserving values
    lbl = vol.data.astype(np.int32)
    return lbl, vol.affine, vol.header


def save_nifti_like(
    out_path: str | Path,
    data: np.ndarray,
    affine: np.ndarray,
    header: nib.Nifti1Header | None = None,
    dtype: np.dtype | None = None,
) -> None:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    arr = data.astype(dtype or np.float32)
    img = nib.Nifti1Image(arr, affine, header=header)
    nib.save(img, str(out_path))


def get_spacing_from_header(header: nib.Nifti1Header) -> Tuple[float, float, float]:
    try:
        zooms = header.get_zooms()
        return float(zooms[0]), float(zooms[1]), float(zooms[2])
    except Exception:
        return 1.0, 1.0, 1.0


def parse_datalist(path: str | Path) -> List[Dict[str, str]]:
    data = json.loads(Path(path).read_text())
    # Normalize schema: expect list of {image,label,id}
    norm = []
    for rec in data:
        r = {k: rec.get(k) for k in ("image", "label", "id")}
        if r["label"] is None:
            raise ValueError("Each record must have a 'label' path")
        norm.append(r)
    return norm


def mirror_out_path(base_dir: str | Path, label_path: str | Path) -> Path:
    base_dir = Path(base_dir)
    label_path = Path(label_path)
    # Use only filename to avoid reproducing full absolute roots under base_dir
    return base_dir / label_path.name


def read_json(path: str | Path):
    return json.loads(Path(path).read_text())


def write_json(path: str | Path, obj) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2))

