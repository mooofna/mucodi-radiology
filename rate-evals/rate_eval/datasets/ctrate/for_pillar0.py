"""CT-RATE chest CT for Pillar0-ChestCT: NIfTI -> (1,256,256,256) real HU via spacing-blind resize to 256^3 (not 1.25 mm iso)."""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn.functional as F

from ._base import CTRateVolumeBase


class CTRateForPillar0(CTRateVolumeBase):
    """CT-RATE NIfTI -> (1, 256, 256, 256) real HU for Pillar0."""

    TARGET_SHAPE = (256, 256, 256)  # (D, H, W)

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        data = self._hu_ras_dhw(image_path)
        # optional L-R mirror (env-gated, off by default)
        if os.environ.get("CTRATE_LR_FLIP") == "1":
            data = np.ascontiguousarray(data[:, ::-1, :])
        vol_t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        vol_t = F.interpolate(vol_t, size=self.TARGET_SHAPE, mode="trilinear", align_corners=False)
        return vol_t.squeeze(0)  # (1, 256, 256, 256)
