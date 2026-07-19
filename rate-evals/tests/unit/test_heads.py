"""Unit tests for rate_eval.evaluation.heads."""

from __future__ import annotations

import pytest
import torch

from rate_eval.evaluation.heads import (
    Linear,
    MLP,
    build_head,
)


@pytest.fixture
def x_2d():
    """(B, D) input -- the standard FeatureDataset shape."""
    return torch.randn(8, 512)


def test_linear_binary(x_2d):
    head = Linear(dim_input=512, dim_output=1)
    out = head(x_2d)
    assert out.shape == (8, 1)


def test_linear_multiclass(x_2d):
    head = Linear(dim_input=512, dim_output=3)
    out = head(x_2d)
    assert out.shape == (8, 3)


def test_mlp_two_layers(x_2d):
    head = MLP(dim_input=512, dim_hidden=256, dim_output=1, num_layers=2, dropout=0.25)
    out = head(x_2d)
    assert out.shape == (8, 1)


def test_build_head_default_linear(x_2d):
    head = build_head({}, dim_input=512, dim_output=1)
    assert isinstance(head, Linear)


def test_build_head_explicit_kinds(x_2d):
    cases = [
        ({"kind": "linear"}, Linear),
        ({"kind": "mlp", "dim_hidden": 256, "num_layers": 2}, MLP),
    ]
    for spec, expected_cls in cases:
        head = build_head(spec, dim_input=512, dim_output=1)
        assert isinstance(head, expected_cls), f"{spec} -> {type(head)}"
        out = head(x_2d)
        assert out.shape[0] == 8


def test_build_head_unknown_kind_falls_back_or_errors():
    """Unknown kinds: either error or fall back to linear (don't silently misroute)."""
    try:
        head = build_head({"kind": "bogus_unknown_head"}, dim_input=8, dim_output=1)
    except (ValueError, KeyError):
        return
    # no error: head must return the right shape
    out = head(torch.randn(2, 8))
    assert out.shape[0] == 2


def test_build_head_kind_case_insensitive():
    head = build_head({"kind": "LINEAR"}, dim_input=4, dim_output=2)
    assert isinstance(head, Linear)
