"""Unit tests for rate_eval.datasets.preprocess (each recipe vs a hand-computed expected tensor)."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from rate_eval.datasets.preprocess import PreprocessConfig, apply_preprocess


def _synth(shape=(4, 8, 8)) -> torch.Tensor:
    """Small synthetic HU volume -- non-trivial values so clip/interp matter."""
    return torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) * 10 - 1000


def test_passthrough_with_unused_args_raises():
    with pytest.raises(ValueError, match="passthrough"):
        PreprocessConfig(pipeline="passthrough", target_shape=(8, 8, 8))


def test_interp_without_target_shape_raises():
    with pytest.raises(ValueError, match="target_shape"):
        PreprocessConfig(pipeline="interp")


def test_clip_interp_norm_without_hu_clip_raises():
    with pytest.raises(ValueError, match="hu_clip"):
        PreprocessConfig(pipeline="clip_interp_norm", target_shape=(8, 8, 8))


def test_output_norm_halfrange_only_for_clip_interp_norm():
    with pytest.raises(ValueError, match="halfrange_centered"):
        PreprocessConfig(
            pipeline="interp_clip_norm",
            target_shape=(8, 8, 8),
            hu_clip=(-1000.0, 1000.0),
            output_norm="halfrange_centered",
        )


def test_output_norm_minmax_only_for_interp_clip_norm():
    with pytest.raises(ValueError, match="minmax"):
        PreprocessConfig(
            pipeline="clip_interp_norm",
            target_shape=(8, 8, 8),
            hu_clip=(-1000.0, 1000.0),
            output_norm="minmax",
        )


def test_passthrough_returns_unaltered_with_channel_added():
    """Pipeline 'passthrough' must not touch values; output is (1, D, H, W)."""
    vol = _synth((4, 6, 6))
    cfg = PreprocessConfig(pipeline="passthrough")
    out = apply_preprocess(vol, cfg)
    assert out.shape == (1, 4, 6, 6)
    assert torch.equal(out.squeeze(0), vol)


def test_interp_only_matches_legacy_pillar0_recipe():
    """Pipeline 'interp' = F.interpolate(...) -> squeeze(0)."""
    vol = _synth((4, 6, 6))
    cfg = PreprocessConfig(pipeline="interp", target_shape=(8, 12, 12))
    out = apply_preprocess(vol, cfg)

    expected = vol.unsqueeze(0).unsqueeze(0)
    expected = F.interpolate(expected, size=(8, 12, 12), mode="trilinear", align_corners=False)
    expected = expected.squeeze(0)
    assert torch.equal(out, expected)


def test_clip_interp_norm_matches_legacy_ctclip_recipe():
    """Pipeline 'clip_interp_norm' = clip -> interp -> /halfrange."""
    vol = _synth((4, 6, 6))
    cfg = PreprocessConfig(
        pipeline="clip_interp_norm",
        target_shape=(8, 12, 12),
        hu_clip=(-1000.0, 1000.0),
        output_norm="halfrange_centered",
    )
    out = apply_preprocess(vol, cfg)

    expected = vol.unsqueeze(0).unsqueeze(0)
    expected = expected.clamp(min=-1000.0, max=1000.0)
    expected = F.interpolate(expected, size=(8, 12, 12), mode="trilinear", align_corners=False)
    expected = expected.squeeze(0) / 1000.0
    assert torch.equal(out, expected)


def test_clip_interp_norm_no_output_norm_is_clamped_then_interpolated():
    """If output_norm is None, the clip+interp result is returned unscaled."""
    vol = _synth((4, 6, 6))
    cfg = PreprocessConfig(
        pipeline="clip_interp_norm",
        target_shape=(8, 12, 12),
        hu_clip=(-1000.0, 1000.0),
        output_norm=None,
    )
    out = apply_preprocess(vol, cfg)

    expected = vol.unsqueeze(0).unsqueeze(0)
    expected = expected.clamp(min=-1000.0, max=1000.0)
    expected = F.interpolate(expected, size=(8, 12, 12), mode="trilinear", align_corners=False)
    expected = expected.squeeze(0)
    assert torch.equal(out, expected)


def test_interp_clip_norm_matches_legacy_tangerine_recipe():
    """Pipeline 'interp_clip_norm' = interp -> clip -> minmax."""
    vol = _synth((4, 6, 6))
    cfg = PreprocessConfig(
        pipeline="interp_clip_norm",
        target_shape=(8, 12, 12),
        hu_clip=(-1200.0, 800.0),
        output_norm="minmax",
    )
    out = apply_preprocess(vol, cfg)

    expected = vol.unsqueeze(0).unsqueeze(0)
    expected = F.interpolate(expected, size=(8, 12, 12), mode="trilinear", align_corners=False)
    expected = expected.squeeze(0)
    expected = expected.clamp(min=-1200.0, max=800.0)
    expected = (expected - (-1200.0)) / (800.0 - (-1200.0))
    assert torch.equal(out, expected)


def test_invalid_input_shape_raises():
    with pytest.raises(ValueError, match=r"\(D,H,W\)"):
        apply_preprocess(torch.randn(2, 3), PreprocessConfig(pipeline="passthrough"))


def test_accepts_pre_channeled_input():
    """`(1, D, H, W)` input should be handled (caller may already have a channel)."""
    vol = _synth((4, 6, 6)).unsqueeze(0)
    cfg = PreprocessConfig(pipeline="passthrough")
    out = apply_preprocess(vol, cfg)
    assert out.shape == (1, 4, 6, 6)
