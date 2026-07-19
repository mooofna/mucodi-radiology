"""Fidelity + contract tests for the 3D EfficientNet student backbones (models/efficientnet3d.py)."""
from __future__ import annotations

import os

import pytest
import torch

from models.efficientnet3d import (
    _ARCH_REGISTRY,
    _DEFAULT_IN_CHANNELS,
    _EFFNET_BLOCKS_ARGS,
    _EfficientNet3DFeat,
    create_3d_backbone,
    probe_feat_dim,
)

# EfficientNet-B0 baseline network, Table 1 (Tan & Le 2019) -- the 7 MBConv stages.
_PAPER_B0_BLOCKS_ARGS = [
    "r1_k3_s11_e1_i32_o16_se0.25",
    "r2_k3_s22_e6_i16_o24_se0.25",
    "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s22_e6_i40_o80_se0.25",
    "r3_k5_s11_e6_i80_o112_se0.25",
    "r4_k5_s22_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]

# (width, depth, resolution) per tier, from the paper
_PAPER_COEFFS = {
    "efficientnet-b0": (1.0, 1.0, 224),
    "efficientnet-b1": (1.0, 1.1, 240),
    "efficientnet-b2": (1.1, 1.2, 260),
    "efficientnet-b3": (1.2, 1.4, 300),
    "efficientnet-b4": (1.4, 1.8, 380),
    "efficientnet-b5": (1.6, 2.2, 456),
    "efficientnet-b6": (1.8, 2.6, 528),
    "efficientnet-b7": (2.0, 3.1, 600),
}

# d_S (pooled width) + param count (M) at in_channels=4
_EXPECTED = {
    "efficientnet3d_tmin": (64, 0.0364),
    "efficientnet3d_t0": (104, 0.083),
    "efficientnet3d_t1": (176, 0.145),
    "efficientnet3d_t2": (256, 0.27),
    "efficientnet3d_t3": (384, 0.51),
    "efficientnet3d_t4": (512, 1.04),
    "efficientnet3d_t5": (768, 1.84),
    "efficientnet3d_t6": (1024, 3.11),
    "efficientnet3d_b0": (1280, 4.69),
    "efficientnet3d_b1": (1280, 7.45),
    "efficientnet3d_b2": (1408, 8.72),
    "efficientnet3d_b3": (1536, 12.06),
    "efficientnet3d_b4": (1792, 19.62),
    "efficientnet3d_b5": (2048, 31.05),
    "efficientnet3d_b6": (2304, 44.40),
}


def test_registry_is_the_locked_grid():
    """The grid is exactly the 15 documented rungs."""
    assert set(_ARCH_REGISTRY) == set(_EXPECTED)


def test_default_in_channels_is_four():
    assert _DEFAULT_IN_CHANNELS == 4


def test_blocks_args_verbatim_paper_b0():
    """The 7-stage MBConv spec is the EfficientNet-B0 baseline, verbatim."""
    assert _EFFNET_BLOCKS_ARGS == _PAPER_B0_BLOCKS_ARGS


def test_compound_scaling_coeffs_verbatim_paper():
    """MONAI's per-tier (width, depth, resolution) == Tan & Le 2019."""
    from monai.networks.nets.efficientnet import efficientnet_params

    for name, (w, d, res) in _PAPER_COEFFS.items():
        m_w, m_d, m_res = efficientnet_params[name][:3]
        assert (m_w, m_d, m_res) == (w, d, res), f"{name}: MONAI {(m_w, m_d, m_res)} != paper {(w, d, res)}"


@pytest.mark.parametrize("arch", sorted(_EXPECTED))
def test_build_at_4_channels(arch):
    exp_dS, exp_params_m = _EXPECTED[arch]
    bb = create_3d_backbone(arch, in_channels=4)

    first_conv = next(m for m in bb.modules() if isinstance(m, torch.nn.Conv3d))
    assert first_conv.in_channels == 4

    assert bb.feat_dim == exp_dS
    assert probe_feat_dim(bb, in_channels=4) == exp_dS

    # param count (encoder only) within 3% of reference
    n_m = sum(p.numel() for p in bb.parameters()) / 1e6
    assert n_m == pytest.approx(exp_params_m, rel=0.03), f"{arch}: {n_m:.3f}M vs {exp_params_m}M"


def test_create_backbone_in_channels_default_is_four():
    bb = create_3d_backbone("efficientnet3d_b0")
    first_conv = next(m for m in bb.modules() if isinstance(m, torch.nn.Conv3d))
    assert first_conv.in_channels == 4


def test_unknown_arch_raises():
    with pytest.raises(ValueError, match="Unknown 3D arch"):
        create_3d_backbone("resnet18_3d")


@pytest.mark.parametrize("arch", ["efficientnet3d_t0", "efficientnet3d_b0"])
def test_forward_maps_to_feat_dim(arch):
    """Forward at the live 128^3 geometry returns (B, d_S)."""
    bb = create_3d_backbone(arch, in_channels=4).eval()
    with torch.no_grad():
        y = bb(torch.randn(1, 4, 128, 128, 128))
    assert y.shape == (1, bb.feat_dim)


def test_silu_stripped_by_default(monkeypatch):
    monkeypatch.delenv("EFFNET3D_KEEP_SILU", raising=False)
    bb = _EfficientNet3DFeat(model_name="efficientnet-b0", in_channels=4)
    assert bb.keep_silu is False


def test_keep_silu_env_knob(monkeypatch):
    monkeypatch.setenv("EFFNET3D_KEEP_SILU", "1")
    bb = _EfficientNet3DFeat(model_name="efficientnet-b0", in_channels=4)
    assert bb.keep_silu is True


def test_silu_strip_changes_the_feature(monkeypatch):
    """Stripping the terminal SiLU must actually change the pooled feature."""
    x = torch.randn(1, 4, 128, 128, 128)

    monkeypatch.delenv("EFFNET3D_KEEP_SILU", raising=False)
    bb_stripped = _EfficientNet3DFeat(model_name="efficientnet-b0", in_channels=4).eval()

    monkeypatch.setenv("EFFNET3D_KEEP_SILU", "1")
    bb_kept = _EfficientNet3DFeat(model_name="efficientnet-b0", in_channels=4).eval()
    bb_kept.load_state_dict(bb_stripped.state_dict())  # identical weights

    with torch.no_grad():
        y_stripped = bb_stripped(x)
        y_kept = bb_kept(x)

    # differ in a relative sense
    assert not torch.allclose(y_stripped, y_kept, rtol=1e-3, atol=0.0)
    # y_kept ~ 0.5*y_stripped
    assert torch.allclose(y_kept, 0.5 * y_stripped, rtol=1e-5, atol=1e-20)
