"""RAD-ChestCT NPZ for CT-CLIP: upstream CTReportDataset recipe -> (1,240,480,480) in [-1,1]."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...core.errors import DatasetError
from ...core.logging import get_logger
from .._core.lidc_base import LIDCBaseDataset

_SRC_SPACING = 0.8  # Draelos NPZs, 0.8 mm iso


class RADChestCTForCTClip(LIDCBaseDataset):
    """Returns the volume as `(1, 240, 480, 480)` float32 in [-1, 1] -- paper-faithful + oriented."""

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise DatasetError(f"NPZ not found: {path}")
        with np.load(str(path)) as f:
            data = np.clip(f["ct"].astype(np.float32), -1000.0, 1000.0)   # (D,H,W) HU, 0.8mm iso
        cur = (_SRC_SPACING, _SRC_SPACING, _SRC_SPACING)
        tgt = (1.5, 0.75, 0.75)
        new = [max(1, int(round(data.shape[i] * cur[i] / tgt[i]))) for i in range(3)]
        t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=tuple(new), mode="trilinear", align_corners=False)[0, 0] / 1000.0
        # center crop/pad to (H=480, W=480, D=240) in (H,W,D) layout
        t = t.permute(1, 2, 0)
        for dim, target in enumerate((480, 480, 240)):
            n = t.shape[dim]
            if n >= target:
                s = (n - target) // 2
                t = t.index_select(dim, torch.arange(s, s + target))
            else:
                pb = (target - n) // 2
                pads = [0, 0, 0, 0, 0, 0]
                pads[(2 - dim) * 2] = pb
                pads[(2 - dim) * 2 + 1] = target - n - pb
                t = F.pad(t, pads, value=-1.0)
        t = t.permute(2, 0, 1)                                           # (D=240, H=480, W=480)
        # orientation fix: in-plane transpose + axial flip
        t = torch.flip(t.permute(0, 2, 1), dims=[0])
        return t.unsqueeze(0).contiguous()                              # (1, 240, 480, 480)
