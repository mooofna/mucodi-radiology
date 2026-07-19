"""PreprocessConfig + apply_preprocess: shared 3D CT clip/interp/rescale core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


Pipeline = Literal["interp", "passthrough", "clip_interp_norm", "interp_clip_norm"]
InterpolateMode = Literal["trilinear", "nearest", "bspline"]
OutputNorm = Literal["halfrange_centered", "minmax"]


def _interpolate(x: torch.Tensor, target_shape, mode: str, align_corners: bool) -> torch.Tensor:
    """Resample (1,1,D,H,W) to target_shape (bspline via SimpleITK, matches upstream TANGERINE)."""
    if mode in {"trilinear", "nearest"}:
        return F.interpolate(x, size=target_shape, mode=mode, align_corners=align_corners if mode == "trilinear" else None)
    if mode == "bspline":
        import SimpleITK as sitk
        import numpy as np
        # SimpleITK reverses axes: array (D,H,W) <-> image (W,H,D)
        arr = x.squeeze(0).squeeze(0).cpu().numpy().astype(np.float32)
        sitk_img = sitk.GetImageFromArray(arr)
        original_size = sitk_img.GetSize()
        original_spacing = sitk_img.GetSpacing()
        new_size = (target_shape[2], target_shape[1], target_shape[0])
        new_spacing = [osz * osp / nsz for osz, osp, nsz in zip(original_size, original_spacing, new_size)]
        resample = sitk.ResampleImageFilter()
        resample.SetOutputSpacing(new_spacing)
        resample.SetSize(new_size)
        resample.SetOutputDirection(sitk_img.GetDirection())
        resample.SetOutputOrigin(sitk_img.GetOrigin())
        resample.SetTransform(sitk.Transform())
        resample.SetInterpolator(sitk.sitkBSpline)
        resampled = resample.Execute(sitk_img)
        out = sitk.GetArrayFromImage(resampled).astype(np.float32)
        return torch.from_numpy(out).unsqueeze(0).unsqueeze(0).to(x.device)
    raise ValueError(f"Unknown interpolate mode: {mode!r}")


@dataclass(frozen=True)
class PreprocessConfig:
    """Config to convert a 3D HU volume into a teacher-ready tensor; validated at construction."""

    pipeline: Pipeline
    target_shape: Optional[Tuple[int, int, int]] = None
    interpolate_mode: InterpolateMode = "trilinear"
    interpolate_align_corners: bool = False
    hu_clip: Optional[Tuple[float, float]] = None
    output_norm: Optional[OutputNorm] = None

    def __post_init__(self) -> None:
        if self.pipeline in {"interp", "clip_interp_norm", "interp_clip_norm"}:
            if self.target_shape is None:
                raise ValueError(f"pipeline={self.pipeline!r} requires target_shape")
        if self.pipeline in {"clip_interp_norm", "interp_clip_norm"}:
            if self.hu_clip is None:
                raise ValueError(f"pipeline={self.pipeline!r} requires hu_clip")
        if self.pipeline == "passthrough":
            if self.target_shape is not None or self.hu_clip is not None:
                raise ValueError(
                    "pipeline='passthrough' must not specify target_shape or hu_clip"
                )
        if self.output_norm == "halfrange_centered" and self.pipeline != "clip_interp_norm":
            raise ValueError(
                "output_norm='halfrange_centered' is only valid for pipeline='clip_interp_norm' "
                "(matches the CT-CLIP recipe: clip->interp->/halfrange)"
            )
        if self.output_norm == "minmax" and self.pipeline != "interp_clip_norm":
            raise ValueError(
                "output_norm='minmax' is only valid for pipeline='interp_clip_norm' "
                "(matches the TANGERINE recipe: interp->clip->minmax)"
            )


def apply_preprocess(vol_3d: torch.Tensor, cfg: PreprocessConfig) -> torch.Tensor:
    """Apply `cfg` to a 3D HU volume; `(D,H,W)` or `(1,D,H,W)` -> `(1,D',H',W')` float32."""
    if vol_3d.ndim == 3:
        x = vol_3d.unsqueeze(0).unsqueeze(0)
    elif vol_3d.ndim == 4 and vol_3d.shape[0] == 1:
        x = vol_3d.unsqueeze(0)
    else:
        raise ValueError(
            f"apply_preprocess expects (D,H,W) or (1,D,H,W); got {tuple(vol_3d.shape)}"
        )

    if cfg.pipeline == "passthrough":
        return x.squeeze(0)

    if cfg.pipeline == "interp":
        x = _interpolate(x, cfg.target_shape, cfg.interpolate_mode, cfg.interpolate_align_corners)
        return x.squeeze(0)

    if cfg.pipeline == "clip_interp_norm":
        # CT-CLIP: clamp BEFORE interp
        lo, hi = cfg.hu_clip
        x = x.clamp(min=lo, max=hi)
        x = _interpolate(x, cfg.target_shape, cfg.interpolate_mode, cfg.interpolate_align_corners)
        x = x.squeeze(0)
        if cfg.output_norm == "halfrange_centered":
            half = max(abs(lo), abs(hi))
            x = x / half
        return x

    if cfg.pipeline == "interp_clip_norm":
        # TANGERINE: interp BEFORE clamp
        x = _interpolate(x, cfg.target_shape, cfg.interpolate_mode, cfg.interpolate_align_corners)
        x = x.squeeze(0)
        lo, hi = cfg.hu_clip
        x = x.clamp(min=lo, max=hi)
        if cfg.output_norm == "minmax":
            x = (x - lo) / (hi - lo)
        return x

    raise ValueError(f"Unknown pipeline: {cfg.pipeline!r}")


__all__ = ["PreprocessConfig", "apply_preprocess", "Pipeline", "OutputNorm"]
