"""CT-RATE chest CT for the Curia wrappers: NIfTI in canonical RAS HU -> (1, D, H, W)."""

from __future__ import annotations

import os

import numpy as np
import torch

from ._base import CTRateVolumeBase


class CTRateForCuria(CTRateVolumeBase):
    """CT-RATE NIfTI -> (1, D, H, W) real HU in RAS for the Curia wrappers (both curia1 and curia2)."""

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        data = self._hu_ras_dhw(image_path)
        # optional L-R mirror (env-gated, off by default)
        if os.environ.get("CTRATE_LR_FLIP") == "1":
            data = np.ascontiguousarray(data[:, ::-1, :])
        return torch.from_numpy(data).unsqueeze(0)  # (1, D, H, W)
