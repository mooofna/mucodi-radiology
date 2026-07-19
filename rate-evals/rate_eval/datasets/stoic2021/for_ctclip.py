"""STOIC2021 .mha chest CT for CT-CLIP (zero-shot + vocab-fine; one class serves both)."""
from __future__ import annotations

import os

import torch

from ..nifti_recipes import NiftiForCTClip


class STOIC2021ForCTClip(NiftiForCTClip):
    """STOIC2021 .mha -> (1, 240, 480, 480) via the upstream CT-CLIP inference recipe."""

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        from ._mha_read import canonical_nifti_path
        from ..ctclip_orient import lps_nifti_path
        # .mha -> RAS-canonical NIfTI, then reorient to CT-RATE's LPS voxel order
        ras_tmp = canonical_nifti_path(image_path)
        try:
            tmp, xy, z = lps_nifti_path(ras_tmp)
            try:
                return self._tensor_from_lps(tmp, xy, z)
            finally:
                os.unlink(tmp)
        finally:
            os.unlink(ras_tmp)
