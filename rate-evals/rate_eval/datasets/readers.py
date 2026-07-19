"""Source-loader functions: NIfTI/NPZ bytes -> `(D, H, W)` float32 (D = axial slices). The NIfTI reader reorients to canonical RAS first so axial indexing is source-independent."""

from __future__ import annotations

import os
from pathlib import Path

import nibabel as nib
import numpy as np


def read_nifti(path: Path) -> np.ndarray:
    """NIfTI -> `(D, H, W)` float32 HU in canonical RAS. transpose(2,0,1) moves S-I to the leading axis so slices are axial."""
    nii = nib.as_closest_canonical(nib.load(str(path)))
    return nii.get_fdata().astype(np.float32).transpose(2, 0, 1)


def read_npz_ct(path: Path, key: str = "ct") -> np.ndarray:
    """NPZ -> `(D, H, W)` float32. RAD-ChestCT NPZs store in-plane H<->W swapped and axial reversed vs CT-RATE RAS; env var ``RADCHESTCT_ORIENT`` selects the fix: ``identity`` (raw, default), ``transHW`` (swap in-plane), ``flipD_transHW`` (swap + axial flip -> RAS-consistent, the CT-CLIP correction). Both student-eval and teacher-extraction share this reader, keeping them same-estimand."""
    with np.load(str(path)) as f:
        arr = f[key].astype(np.float32)
    orient = os.environ.get("RADCHESTCT_ORIENT", "identity")
    if orient == "identity":
        return arr
    if orient == "transHW":
        return np.ascontiguousarray(np.transpose(arr, (0, 2, 1)))
    if orient == "flipD_transHW":
        return np.ascontiguousarray(np.flip(np.transpose(arr, (0, 2, 1)), axis=0))
    raise ValueError(f"unknown RADCHESTCT_ORIENT={orient!r} (identity|transHW|flipD_transHW)")


__all__ = ["read_nifti", "read_npz_ct"]
