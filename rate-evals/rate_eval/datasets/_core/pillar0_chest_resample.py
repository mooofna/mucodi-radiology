"""Spacing-aware chest resample for Pillar0-ChestCT (1.25 mm iso -> 256^3 crop/pad)."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F

PILLAR0_CHEST_SPACING = (1.25, 1.25, 1.25)  # mm iso; ct_chest.yaml, not ct_brain
PILLAR0_CHEST_SHAPE = (256, 256, 256)       # (D, H, W)


def resample_chest_to_pillar0(
    vol_dhw: torch.Tensor,
    src_spacing_dhw: Sequence[float],
    pad_value: float = -1024.0,
) -> torch.Tensor:
    """``(D,H,W)`` raw-HU volume + mm spacing -> ``(1,256,256,256)`` at 1.25 mm iso."""
    src = tuple(float(s) for s in src_spacing_dhw)
    if len(src) != 3:
        raise ValueError(f"src_spacing_dhw must have 3 entries, got {src}")
    new = [max(1, int(round(int(vol_dhw.shape[i]) * src[i] / PILLAR0_CHEST_SPACING[i]))) for i in range(3)]
    t = vol_dhw.to(torch.float32)[None, None]
    t = F.interpolate(t, size=tuple(new), mode="trilinear", align_corners=False)[0, 0]
    for dim, target in enumerate(PILLAR0_CHEST_SHAPE):
        n = int(t.shape[dim])
        if n >= target:
            s = (n - target) // 2
            t = t.index_select(dim, torch.arange(s, s + target, device=t.device))
        else:
            pb = (target - n) // 2
            pads = [0, 0, 0, 0, 0, 0]  # F.pad order is (W_lo, W_hi, H_lo, H_hi, D_lo, D_hi)
            pads[(2 - dim) * 2] = pb
            pads[(2 - dim) * 2 + 1] = target - n - pb
            t = F.pad(t, pads, value=pad_value)
    return t.unsqueeze(0).contiguous()
