"""RAD-ChestCT NPZ chest-CT for TANGERINE-vit: sitk BSpline resample to 256^3 -> HU clip [-1200,800] -> min-max [0,1], on Draelos NPZs (HU, [-1000,1000], 0.8 mm iso), flipD_transHW-oriented."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...core.errors import DatasetError
from ...core.logging import get_logger
from .._core.lidc_base import LIDCBaseDataset


logger = get_logger(__name__)

_HU_MIN, _HU_MAX = -1200.0, 800.0
TARGET_SHAPE = (256, 256, 256)


class RADChestCTForTangerine(LIDCBaseDataset):
    """RAD-ChestCT NPZ -> (1, 256, 256, 256) [0, 1] float32 for TANGERINE (BSpline, flipD_transHW)."""

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        from rate_eval.datasets.preprocess import _interpolate  # noqa: PLC0415
        path = Path(image_path)
        if not path.exists():
            raise DatasetError(f"NPZ not found: {path}")
        with np.load(str(path)) as f:
            data = np.asarray(f["ct"], dtype=np.float32)          # (D, H, W) HU, 0.8 mm iso
        t = torch.from_numpy(np.ascontiguousarray(data))
        # flipD_transHW: H<->W transpose + axial flip -> CT-RATE voxel order
        t = torch.flip(t.permute(0, 2, 1), dims=[0]).contiguous()
        vol_t = t.unsqueeze(0).unsqueeze(0)                        # (1, 1, D, H, W)
        # TANGERINE requires SimpleITK BSpline (== upstream resample_ct_volume), NOT trilinear.
        vol_t = _interpolate(vol_t, TARGET_SHAPE, "bspline", align_corners=False)
        vol_t = torch.clamp(vol_t, _HU_MIN, _HU_MAX)
        vol_t = (vol_t - _HU_MIN) / (_HU_MAX - _HU_MIN)
        return vol_t.squeeze(0)                                   # (1, 256, 256, 256)
