"""CT-RATE chest CT for TANGERINE-vit: NIfTI -> sitk BSpline resample to 256^3 -> HU clip [-1200,800] -> min-max [0,1] -> (1,256,256,256)."""

from __future__ import annotations

import torch

from ._base import CTRateVolumeBase

_HU_MIN, _HU_MAX = -1200.0, 800.0


class CTRateForTangerine(CTRateVolumeBase):
    """CT-RATE NIfTI -> (1, 256, 256, 256) float32 in [0, 1] for TANGERINE."""

    TARGET_SHAPE = (256, 256, 256)

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        data = self._hu_ras_dhw(image_path)
        vol_t = torch.from_numpy(data).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        # TANGERINE requires SimpleITK BSpline (== upstream resample_ct_volume), NOT trilinear.
        from rate_eval.datasets.preprocess import _interpolate
        vol_t = _interpolate(vol_t, self.TARGET_SHAPE, "bspline", align_corners=False)
        vol_t = torch.clamp(vol_t, _HU_MIN, _HU_MAX)
        vol_t = (vol_t - _HU_MIN) / (_HU_MAX - _HU_MIN)
        return vol_t.squeeze(0)  # (1, 256, 256, 256)
