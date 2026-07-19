"""Reorient any CT to CT-RATE's native LPS voxel order for the CT-CLIP loaders (CT-CLIP keys off on-disk voxel order)."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import nibabel as nib

from .affine_repair import safe_canonical


def _write_lps_from_ras(ras_img: "nib.Nifti1Image") -> Tuple[str, float, float]:
    """RAS-canonical image -> temp NIfTI in LPS voxel order; returns (tmp_path, xy_spacing, z_spacing) (caller unlinks)."""
    ras = ras_img.get_fdata(dtype=np.float32)
    lps = np.ascontiguousarray(ras[::-1, ::-1, :])  # reverse R->L, A->P => LPS
    zooms = [float(z) for z in ras_img.header.get_zooms()[:3]]
    # upstream ignores affine; spacing-only diag + slope=1/inter=0 preserves lps/HU on re-read
    out = nib.Nifti1Image(lps, np.diag([zooms[0], zooms[1], zooms[2], 1.0]))
    out.header.set_slope_inter(1.0, 0.0)
    tmp = tempfile.NamedTemporaryFile(suffix=".nii", delete=False)
    tmp.close()
    nib.save(out, tmp.name)
    return tmp.name, zooms[0], zooms[2]


def lps_nifti_path(src_path: str) -> Tuple[str, float, float]:
    """Reorient any nib.load-able path to LPS voxel order via temp NIfTI; returns (tmp_path, xy_spacing, z_spacing) (caller unlinks)."""
    if not Path(src_path).exists():
        from ..core.errors import DatasetError
        raise DatasetError(f"NIfTI not found: {src_path}")
    return _write_lps_from_ras(safe_canonical(str(src_path)))
