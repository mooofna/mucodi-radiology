"""CT-RATE NIfTI -> (1,240,480,480) in [-1,1] for CT-CLIP (upstream CTReportDatasetinfer inference recipe)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch

from ...core.errors import DatasetError
from ...core.logging import get_logger
from ...config import get_config_value
from ..ctrate_metadata import lookup_meta
from .._core.lidc_base import LIDCBaseDataset


logger = get_logger(__name__)


class CTRateForCTClip(LIDCBaseDataset):
    """CT-RATE NIfTI -> (1, 240, 480, 480) via upstream `CTReportDatasetinfer.nii_img_to_tensor`."""

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        transforms: Optional[Any] = None,
        model_preprocess: Optional[Any] = None,
        name_override: Optional[str] = None,
    ) -> None:
        super().__init__(config, split, transforms, model_preprocess, name_override)
        upstream_dir = Path(get_config_value(config, "ctclip_upstream_dir") or "")
        scripts = upstream_dir / "scripts"
        # subprocess workers re-import this module; path must persist
        from ...models.upstream_paths import add_upstream_to_path
        try:
            add_upstream_to_path(scripts)
        except FileNotFoundError as exc:
            raise DatasetError(str(exc)) from exc
        # inference loader clips HU BEFORE resample (training data.py clips AFTER); do not swap
        from data_inference_nii import CTReportDatasetinfer  # type: ignore[import-not-found]
        self._nii_img_to_tensor = CTReportDatasetinfer.nii_img_to_tensor

        self._metadata_csv = Path(
            get_config_value(config, "metadata_csv")
            or os.path.expandvars("$DATA_ROOT/radiology/ctrate/dataset/metadata/validation_metadata.csv")
        )
        if not self._metadata_csv.exists():
            raise DatasetError(
                f"CT-RATE metadata CSV not found: {self._metadata_csv}. "
                f"Re-stage CT-RATE V2 (see the parent repo's dataprep stager) to fetch it.",
            )

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise DatasetError(f"NIfTI not found: {path}")
        meta = lookup_meta(self._metadata_csv, path.name)
        df = pd.DataFrame([{
            "VolumeName": path.name,
            "RescaleSlope": meta.slope,
            "RescaleIntercept": meta.intercept,
            "XYSpacing": f"[{meta.xy_spacing}, {meta.xy_spacing}]",
            "ZSpacing": meta.z_spacing,
        }])
        return self._nii_img_to_tensor(None, str(path), df)
