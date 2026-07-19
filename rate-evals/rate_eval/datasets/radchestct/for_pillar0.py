"""RAD-ChestCT NPZ -> raw HU at 1.25 mm iso 256^3 for Pillar0-ChestCT (flipD_transHW-oriented)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...core.errors import DatasetError
from ...core.logging import get_logger
from .._core.lidc_base import LIDCBaseDataset
from .._core.pillar0_chest_resample import resample_chest_to_pillar0

logger = get_logger(__name__)

_SRC_SPACING = 0.8  # mm iso (Draelos NPZs)


class RADChestCTForPillar0(LIDCBaseDataset):
    """RAD-ChestCT NPZ -> ``(1, 256, 256, 256)`` raw HU at 1.25 mm iso, flipD_transHW-oriented."""

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise DatasetError(f"NPZ not found: {path}")
        with np.load(str(path)) as f:
            data = np.clip(np.asarray(f["ct"], dtype=np.float32), -1000.0, 1000.0)  # (D, H, W) HU, 0.8 mm iso
        t = torch.from_numpy(np.ascontiguousarray(data))
        # flipD_transHW: H<->W transpose + axial flip -> CT-RATE voxel order
        t = torch.flip(t.permute(0, 2, 1), dims=[0]).contiguous()  # (D, W, H) axial-reversed
        src_spacing_dhw = (_SRC_SPACING, _SRC_SPACING, _SRC_SPACING)
        return resample_chest_to_pillar0(t, src_spacing_dhw, pad_value=-1000.0)
