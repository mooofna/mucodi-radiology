"""RAS-canonical readers for STOIC2021 MetaImage (.mha) volumes."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from ...core.errors import DatasetError


def load_canonical_dhw(image_path: str):
    """Return (data_DHW, spacing_dhw) for a .mha in RAS-canonical orientation."""
    import SimpleITK as sitk  # noqa: PLC0415
    import nibabel as nib  # noqa: PLC0415

    path = Path(image_path)
    if not path.exists():
        raise DatasetError(f"MHA not found: {path}")
    img = sitk.ReadImage(str(path))
    tmp = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    tmp.close()
    try:
        sitk.WriteImage(img, tmp.name)
        can = nib.as_closest_canonical(nib.load(tmp.name))
        data = np.asarray(can.dataobj, dtype=np.float32)          # (X, Y, Z) RAS
        zooms = [float(z) for z in can.header.get_zooms()[:3]]    # (x, y, z) mm
    finally:
        os.unlink(tmp.name)
    dhw = np.ascontiguousarray(np.transpose(data, (2, 0, 1)))     # (Z, X, Y) = (D, H, W)
    spacing_dhw = (zooms[2], zooms[0], zooms[1])                  # (z, x, y)
    return dhw, spacing_dhw


def canonical_nifti_path(image_path: str) -> str:
    """Transcode a .mha to an RAS-canonical NIfTI temp file; caller must os.unlink it."""
    import SimpleITK as sitk  # noqa: PLC0415
    import nibabel as nib  # noqa: PLC0415

    path = Path(image_path)
    if not path.exists():
        raise DatasetError(f"MHA not found: {path}")
    img = sitk.ReadImage(str(path))
    raw = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
    raw.close()
    try:
        sitk.WriteImage(img, raw.name)
        can = nib.as_closest_canonical(nib.load(raw.name))
        out = tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False)
        out.close()
        nib.save(can, out.name)
    finally:
        os.unlink(raw.name)
    return out.name
