"""Pydantic schema for `configs/dataset/*.yaml` files; validation is opt-in via `from_yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


SourceKind = Literal["nifti", "npz"]


class LoaderSpec(BaseModel):
    """`loader:` block -- dataset class + source format + preprocess recipe."""

    model_config = ConfigDict(extra="allow")

    class_: str = Field(alias="class")
    source: Optional[SourceKind] = None
    preprocess: Optional["PreprocessSpec"] = None
    config: Optional[str] = None  # legacy
    config_file: Optional[str] = None  # legacy alias
    init_args: Optional[Dict[str, Any]] = None
    npz_key: Optional[str] = None  # for NPZ source


Pipeline = Literal["interp", "passthrough", "clip_interp_norm", "interp_clip_norm"]
InterpolateMode = Literal["trilinear", "nearest", "bspline"]
OutputNorm = Literal["halfrange_centered", "minmax"]


class PreprocessSpec(BaseModel):
    """Schema for `loader.preprocess:` blocks."""

    model_config = ConfigDict(extra="allow")

    # NiftiCTDataset recipe fields (use `pipeline=`)
    pipeline: Optional[Pipeline] = None
    target_shape: Optional[List[int]] = None  # [D, H, W]
    interpolate_mode: InterpolateMode = "trilinear"
    interpolate_align_corners: bool = False
    hu_clip: Optional[List[float]] = None  # [lo, hi]
    output_norm: Optional[OutputNorm] = None


class DataPaths(BaseModel):
    """`data:` block -- per-split JSONL manifest paths."""

    model_config = ConfigDict(extra="allow")

    train_json: Optional[str] = None
    valid_json: Optional[str] = None
    dev_json: Optional[str] = None
    test_json: Optional[str] = None
    root_dir: Optional[str] = None
    nii_root_dir: Optional[str] = None
    cache_dir: Optional[str] = None


class DatasetConfig(BaseModel):
    """Top-level dataset config."""

    model_config = ConfigDict(extra="allow")

    loader: LoaderSpec
    data: Optional[DataPaths] = None
    modality: Optional[str] = None
    img_paths_key: Optional[str] = None
    hf_dataset_id: Optional[str] = None
    hf_config_name: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "DatasetConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)


# resolve the PreprocessSpec forward reference
LoaderSpec.model_rebuild()


__all__ = [
    "DataPaths",
    "DatasetConfig",
    "InterpolateMode",
    "LoaderSpec",
    "OutputNorm",
    "Pipeline",
    "PreprocessSpec",
    "SourceKind",
]
