"""STOIC2021 .mha -> (1, D, H, W) raw HU for the Curia wrappers (both curia1 and curia2)."""
from __future__ import annotations

import torch

from .._core.lidc_base import LIDCBaseDataset


class STOIC2021ForCuria(LIDCBaseDataset):
    """STOIC2021 .mha -> (1, D, H, W) raw HU for Curia (wrapper z-scores in extract_features)."""

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        from ._mha_read import load_canonical_dhw
        data, _ = load_canonical_dhw(image_path)  # RAS-canonical (D,H,W) raw HU
        return torch.from_numpy(data).unsqueeze(0)
