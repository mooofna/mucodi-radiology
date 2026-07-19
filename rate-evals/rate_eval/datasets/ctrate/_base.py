"""Shared CT-RATE loader base: metadata-CSV resolution + real-HU RAS head (per-wrapper tails differ)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from ...core.errors import DatasetError
from ...core.logging import get_logger
from ...config import get_config_value
from ..ctrate_metadata import lookup_meta
from .._core.lidc_base import LIDCBaseDataset


logger = get_logger(__name__)


class CTRateVolumeBase(LIDCBaseDataset):
    """CT-RATE loaders sharing the metadata-CSV rescale head (Pillar0 / TANGERINE / Curia); tails differ per wrapper."""

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        transforms: Optional[Any] = None,
        model_preprocess: Optional[Any] = None,
        name_override: Optional[str] = None,
    ) -> None:
        super().__init__(config, split, transforms, model_preprocess, name_override)
        self._metadata_csv = Path(
            get_config_value(config, "metadata_csv")
            or os.path.expandvars("$DATA_ROOT/radiology/ctrate/dataset/metadata/validation_metadata.csv")
        )
        if not self._metadata_csv.exists():
            raise DatasetError(
                f"CT-RATE metadata CSV not found: {self._metadata_csv}. "
                f"Re-stage CT-RATE V2 (see the parent repo's dataprep stager) to fetch it.",
            )

    def _hu_ras_dhw(self, image_path: str) -> np.ndarray:
        """NIfTI -> real-HU (D, H, W) in RAS: safe_canonical + CSV slope/intercept (get_fdata is raw, scl_slope=NaN)."""
        path = Path(image_path)
        if not path.exists():
            raise DatasetError(f"NIfTI not found: {path}")
        meta = lookup_meta(self._metadata_csv, path.name)
        from rate_eval.datasets.affine_repair import safe_canonical
        nii = safe_canonical(str(path))  # RAS (rebuilds from header zooms if affine NaN/singular)
        data = nii.get_fdata().astype(np.float32) * meta.slope + meta.intercept
        return np.transpose(data, (2, 0, 1))  # (D=axial, H, W)
