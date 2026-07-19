"""Shared per-wrapper NIfTI loaders: one canonical recipe each for CT-CLIP, TANGERINE, Pillar0, Curia."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from ..core.errors import DatasetError
from ..core.logging import get_logger
from ..config import get_config_value
from ._core.lidc_base import LIDCBaseDataset
from ._core.pillar0_chest_resample import resample_chest_to_pillar0


logger = get_logger(__name__)


class NiftiForCTClip(LIDCBaseDataset):
    """Wraps upstream `CTReportDatasetinfer.nii_img_to_tensor` (inference path) for one chest-CT NIfTI."""

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        transforms: Optional[Any] = None,
        model_preprocess: Optional[Any] = None,
        name_override: Optional[str] = None,
    ) -> None:
        super().__init__(config, split, transforms, model_preprocess, name_override)
        upstream_dir = Path(get_config_value(config, "ctclip_upstream_dir") or "")
        scripts = upstream_dir / "scripts"
        # subprocess workers re-import; path must persist
        from ..models.upstream_paths import add_upstream_to_path
        try:
            add_upstream_to_path(scripts)
        except FileNotFoundError as exc:
            raise DatasetError(str(exc)) from exc
        from data_inference_nii import CTReportDatasetinfer  # type: ignore[import-not-found]
        # body never references self -> callable unbound on None
        self._nii_img_to_tensor = CTReportDatasetinfer.nii_img_to_tensor

    def _tensor_from_lps(self, tmp: str, xy: float, z: float) -> torch.Tensor:
        df = pd.DataFrame([{
            "VolumeName": Path(tmp).name,
            "RescaleSlope": 1.0,
            "RescaleIntercept": 0.0,
            "XYSpacing": f"[{xy}, {xy}]",
            "ZSpacing": z,
        }])
        return self._nii_img_to_tensor(None, tmp, df)  # (1, 240, 480, 480)

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        from .ctclip_orient import lps_nifti_path  # noqa: PLC0415
        tmp, xy, z = lps_nifti_path(str(image_path))
        try:
            return self._tensor_from_lps(tmp, xy, z)
        finally:
            Path(tmp).unlink()


class NiftiForTangerine(LIDCBaseDataset):
    """Wraps upstream `Custom3DDataset` + `resample_ct_volume` for one chest-CT NIfTI."""

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        transforms: Optional[Any] = None,
        model_preprocess: Optional[Any] = None,
        name_override: Optional[str] = None,
    ) -> None:
        super().__init__(config, split, transforms, model_preprocess, name_override)
        upstream_dir = Path(get_config_value(config, "tangerine_upstream_dir") or "")
        from ..models.upstream_paths import add_upstream_to_path
        try:
            add_upstream_to_path(upstream_dir)
        except FileNotFoundError as exc:
            raise DatasetError(str(exc)) from exc
        from datasets_three_d_fine import Custom3DDataset, resample_ct_volume  # type: ignore[import-not-found]
        self._Custom3DDataset = Custom3DDataset
        self._resample_ct_volume = resample_ct_volume

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise DatasetError(f"NIfTI not found: {path}")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            resampled = tmp_dir / "resampled.nii.gz"
            self._resample_ct_volume(str(path), str(resampled), new_size=(256, 256, 256))
            csv = tmp_dir / "one.csv"
            csv.write_text(f"Path,Label\n{resampled},0\n")
            ds = self._Custom3DDataset(str(csv))
            vol, _ = ds[0]  # (1, 256, 256, 256) float32 in [0, 1]
        return vol


class NiftiForPillar0(LIDCBaseDataset):
    """Chest-CT NIfTI -> (1, 256, 256, 256) raw HU at 1.25 mm iso for Pillar0-ChestCT."""

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise DatasetError(f"NIfTI not found: {path}")
        from rate_eval.datasets.affine_repair import safe_canonical
        nii = safe_canonical(str(path))                          # RAS
        data = np.asarray(nii.get_fdata(), dtype=np.float32)     # HU
        zooms = [float(z) for z in nii.header.get_zooms()[:3]]   # (x, y, z) mm in RAS
        data = np.transpose(data, (2, 0, 1))                     # (D=axial, H=x, W=y)
        src_spacing_dhw = (zooms[2], zooms[0], zooms[1])         # (z, x, y) -> (D, H, W)
        vol = torch.from_numpy(np.ascontiguousarray(data))
        return resample_chest_to_pillar0(vol, src_spacing_dhw, pad_value=-1024.0)


class NiftiForCuria(LIDCBaseDataset):
    """Chest-CT NIfTI -> (1, D, H, W) HU canonical RAS for the Curia wrappers (both curia1 and curia2)."""

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise DatasetError(f"NIfTI not found: {path}")
        nii = nib.as_closest_canonical(nib.load(str(path)))
        data = nii.get_fdata().astype(np.float32).transpose(2, 0, 1)
        return torch.from_numpy(data).unsqueeze(0)
