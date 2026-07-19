"""Generic NIfTI/NPZ CT dataset driven by the `loader.preprocess:` YAML block."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch

from ..core.errors import DatasetError
from ..core.logging import get_logger
from ..config import get_config_value
from ._core.lidc_base import LIDCBaseDataset
from .preprocess import PreprocessConfig, apply_preprocess
from .readers import read_nifti, read_npz_ct

logger = get_logger(__name__)


def _build_preprocess_cfg(config: Dict[str, Any]) -> PreprocessConfig:
    """Read `loader.preprocess:` and build a `PreprocessConfig`."""
    raw = get_config_value(config, "loader.preprocess")
    if raw is None:
        raise DatasetError(
            "NiftiCTDataset requires `loader.preprocess:` block. "
            "See rate_eval/datasets/nifti.py docstring for the schema."
        )

    target_shape = raw.get("target_shape")
    if target_shape is not None:
        target_shape = tuple(int(v) for v in target_shape)
        if len(target_shape) != 3:
            raise DatasetError(f"preprocess.target_shape must have 3 entries, got {target_shape}")

    hu_clip = raw.get("hu_clip")
    if hu_clip is not None:
        if len(hu_clip) != 2:
            raise DatasetError(f"preprocess.hu_clip must be [lo, hi], got {hu_clip}")
        hu_clip = (float(hu_clip[0]), float(hu_clip[1]))

    return PreprocessConfig(
        pipeline=raw["pipeline"],
        target_shape=target_shape,  # type: ignore[arg-type]
        interpolate_mode=raw.get("interpolate_mode", "trilinear"),
        interpolate_align_corners=bool(raw.get("interpolate_align_corners", False)),
        hu_clip=hu_clip,  # type: ignore[arg-type]
        output_norm=raw.get("output_norm"),
    )


_LOADERS = {
    "nifti": (read_nifti, lambda raw: {}),
    "npz": (
        read_npz_ct,
        lambda raw: {"key": raw.get("npz_key", "ct")} if raw.get("npz_key") else {},
    ),
}


def _resolve_source_loader(config: Dict[str, Any]):
    """Pick (reader_fn, fixed_kwargs) for the configured source format."""
    source = get_config_value(config, "loader.source")
    if source is None:
        raise DatasetError(
            "NiftiCTDataset requires `loader.source: nifti|npz`. "
            "See rate_eval/datasets/nifti.py docstring for the schema."
        )
    if source not in _LOADERS:
        raise DatasetError(
            f"unknown source {source!r}; supported: {sorted(_LOADERS)}"
        )
    reader, kwargs_factory = _LOADERS[source]
    raw_preprocess = get_config_value(config, "loader.preprocess") or {}
    return reader, kwargs_factory(raw_preprocess)


class NiftiCTDataset(LIDCBaseDataset):
    """Generic NIfTI/NPZ CT dataset parametrized by a `PreprocessConfig`."""

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        transforms: Optional[Any] = None,
        model_preprocess: Optional[Any] = None,
        name_override: Optional[str] = None,
    ) -> None:
        super().__init__(config, split, transforms, model_preprocess, name_override)
        self._preprocess_cfg: PreprocessConfig = _build_preprocess_cfg(config)
        self._reader, self._reader_kwargs = _resolve_source_loader(config)

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        path = Path(image_path)
        if not path.exists():
            raise DatasetError(f"image file not found: {path}")
        try:
            data = self._reader(path, **self._reader_kwargs)
        except Exception as exc:
            raise DatasetError(f"failed to read {path}: {exc}") from exc
        vol = torch.from_numpy(data)
        return apply_preprocess(vol, self._preprocess_cfg)


__all__ = ["NiftiCTDataset"]
