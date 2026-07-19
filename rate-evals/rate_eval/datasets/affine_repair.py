"""Degenerate-affine repair for CT-RATE V2: some train_fixed NIfTIs ship a NaN affine that crashes ``nib.as_closest_canonical`` (SVD did not converge), silently dropping intact volumes; ``safe_canonical`` rebuilds a clean diagonal affine from header zooms so the voxels are recovered."""

from __future__ import annotations

import numpy as np
import nibabel as nib


def affine_is_degenerate(affine: np.ndarray) -> bool:
    a = np.asarray(affine, dtype=np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        return True
    try:
        if abs(np.linalg.det(a[:3, :3])) < 1e-12:
            return True
        np.linalg.svd(a[:3, :3])
    except np.linalg.LinAlgError:
        return True
    return False


def _clean_affine(affine: np.ndarray, zooms) -> np.ndarray:
    """Diagonal affine from header zooms (NaN/Inf -> 1.0), preserving finite axis signs."""
    a = np.asarray(affine, dtype=np.float64)
    z = [float(v) if np.isfinite(v) and v != 0 else 1.0 for v in (list(zooms) + [1, 1, 1])[:3]]
    out = np.eye(4, dtype=np.float64)
    for i in range(3):
        sign = np.sign(a[i, i]) if (a.shape == (4, 4) and np.isfinite(a[i, i]) and a[i, i] != 0) else 1.0
        out[i, i] = sign * abs(z[i])
    return out


def safe_canonical(path: str) -> "nib.Nifti1Image":
    """``as_closest_canonical(load(path))`` that never raises on a degenerate affine."""
    img = nib.load(str(path))
    if affine_is_degenerate(img.affine):
        img = nib.Nifti1Image(img.dataobj, _clean_affine(img.affine, img.header.get_zooms()), img.header)
    return nib.as_closest_canonical(img)
