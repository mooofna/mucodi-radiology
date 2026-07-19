"""STOIC2021 (.mha-format chest CT) dataset for the MuCoDi 3D student."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ...core.logging import get_logger
from .._core.lidc_base import LIDCBaseDataset


logger = get_logger(__name__)


class STOIC2021ForMuCoDiStudent(LIDCBaseDataset):
    """STOIC2021 .mha -> (1, 128, 256, 256) raw HU for the MuCoDi 3D student."""

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        from ._mha_read import load_canonical_dhw
        # spacing-blind trilinear resize to [128,256,256], RAS-canonical to match training
        data, _ = load_canonical_dhw(image_path)  # (D,H,W)
        vol_t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        vol_t = F.interpolate(vol_t, size=(128, 256, 256), mode="trilinear", align_corners=False)
        return vol_t.squeeze(0)  # (1, 128, 256, 256)
