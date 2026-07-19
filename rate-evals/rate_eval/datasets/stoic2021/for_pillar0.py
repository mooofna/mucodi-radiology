"""STOIC2021 .mha -> (1, 256, 256, 256) raw HU at 1.25 mm iso for Pillar0 (SimpleITK read)."""
from __future__ import annotations

import torch

from ..nifti_recipes import NiftiForPillar0


class STOIC2021ForPillar0(NiftiForPillar0):
    """STOIC2021 .mha -> (1, 256, 256, 256) raw HU for Pillar0."""

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        from .._core.pillar0_chest_resample import resample_chest_to_pillar0
        from ._mha_read import load_canonical_dhw
        data, src_spacing_dhw = load_canonical_dhw(image_path)  # RAS-canonical (D,H,W) + (z,x,y) mm
        return resample_chest_to_pillar0(torch.from_numpy(data), src_spacing_dhw, pad_value=-1024.0)
