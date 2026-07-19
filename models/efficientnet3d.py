"""3D EfficientNet (Tan & Le 2019) student backbones for radiology knowledge distillation."""
from __future__ import annotations

import os

import torch
import torch.nn as nn
from monai.networks.nets import EfficientNet, EfficientNetBN

# 4-window chest preset (lung, mediastinum, soft_tissue, bone) = 4 channels
_DEFAULT_IN_CHANNELS = 4

# EfficientNet-B0 7-stage MBConv block args (Tan & Le 2019, Table 1; verbatim)
_EFFNET_BLOCKS_ARGS = [
    "r1_k3_s11_e1_i32_o16_se0.25",
    "r2_k3_s22_e6_i16_o24_se0.25",
    "r2_k5_s22_e6_i24_o40_se0.25",
    "r3_k3_s22_e6_i40_o80_se0.25",
    "r3_k5_s11_e6_i80_o112_se0.25",
    "r4_k5_s22_e6_i112_o192_se0.25",
    "r1_k3_s11_e6_i192_o320_se0.25",
]


class _EfficientNet3DFeat(nn.Module):
    """MONAI 3D EfficientNetBN as a KD feature extractor; pooled head feature before the final SiLU."""

    def __init__(
        self,
        *,
        model_name: str | None = None,
        width: float | None = None,
        depth: float | None = None,
        in_channels: int,
    ) -> None:
        super().__init__()
        if model_name is not None:
            self.net = EfficientNetBN(
                model_name, spatial_dims=3, in_channels=in_channels, num_classes=2,
            )
        else:
            # image_size only sets static SAME-padding (recomputed at runtime)
            if width is None or depth is None:
                raise ValueError("pass either model_name or both (width, depth)")
            self.net = EfficientNet(
                blocks_args_str=list(_EFFNET_BLOCKS_ARGS),
                spatial_dims=3,
                in_channels=in_channels,
                num_classes=2,
                width_coefficient=width,
                depth_coefficient=depth,
                image_size=224,
            )
        # stochastic depth off: zero MONAI's per-block drop_connect
        for _blk in self.net._blocks:
            _blk.drop_connect_rate = 0.0
        self.feat_dim = int(self.net._fc.in_features)
        # drop unused classifier + dropout (DDP unused params)
        self.net._fc = nn.Identity()
        self.net._dropout = nn.Identity()
        # EFFNET3D_KEEP_SILU=1 keeps the terminal SiLU
        self.keep_silu = os.environ.get("EFFNET3D_KEEP_SILU", "0") == "1"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        net = self.net
        x = net._conv_stem(net._conv_stem_padding(x))
        x = net._swish(net._bn0(x))
        x = net._blocks(x)
        x = net._conv_head(net._conv_head_padding(x))
        x = net._bn1(x)
        if self.keep_silu:
            x = net._swish(x)
        x = net._avg_pooling(x)
        return x.flatten(start_dim=1)         # (B, feat_dim)


def _make_efficientnet_builder(model_name: str):
    def _builder(in_channels: int = _DEFAULT_IN_CHANNELS) -> nn.Module:
        return _EfficientNet3DFeat(model_name=model_name, in_channels=in_channels)

    _builder.__name__ = f"_build_{model_name.replace('-', '_')}"
    _builder.__doc__ = f"3D {model_name} (MONAI EfficientNetBN); pre-final-SiLU pooled feature."
    return _builder


def _make_efficientnet_coeff_builder(width: float, depth: float):
    """Coefficient-scaled 3D EfficientNet (sub-B0 rungs): same B0 block args, scaled width + depth."""

    def _builder(in_channels: int = _DEFAULT_IN_CHANNELS) -> nn.Module:
        return _EfficientNet3DFeat(width=width, depth=depth, in_channels=in_channels)

    _builder.__name__ = f"_build_efficientnet3d_w{width}_d{depth}".replace(".", "")
    _builder.__doc__ = (
        f"3D EfficientNet (coeff-scaled w={width}, d={depth}); pre-final-SiLU pooled feature."
    )
    return _builder


_ARCH_REGISTRY = {
    # paper tiers b0..b6 (named MONAI EfficientNetBN; coeffs = Tan & Le 2019)
    "efficientnet3d_b0": _make_efficientnet_builder("efficientnet-b0"),  # 4.69M / d_S 1280
    "efficientnet3d_b1": _make_efficientnet_builder("efficientnet-b1"),  # 7.45M / d_S 1280
    "efficientnet3d_b2": _make_efficientnet_builder("efficientnet-b2"),  # 8.72M / d_S 1408
    "efficientnet3d_b3": _make_efficientnet_builder("efficientnet-b3"),  # 12.06M / d_S 1536
    "efficientnet3d_b4": _make_efficientnet_builder("efficientnet-b4"),  # 19.62M / d_S 1792
    "efficientnet3d_b5": _make_efficientnet_builder("efficientnet-b5"),  # 31.05M / d_S 2048
    "efficientnet3d_b6": _make_efficientnet_builder("efficientnet-b6"),  # 44.40M / d_S 2304
    # sub-B0 coefficient variants (param-frontier low end, not paper tiers)
    "efficientnet3d_tmin": _make_efficientnet_coeff_builder(0.05, 0.25), # 0.0364M / d_S 64 (width+depth floor)
    "efficientnet3d_t0": _make_efficientnet_coeff_builder(0.08, 0.5),    # 0.083M / d_S 104
    "efficientnet3d_t1": _make_efficientnet_coeff_builder(0.14, 0.5),    # 0.145M / d_S 176
    "efficientnet3d_t2": _make_efficientnet_coeff_builder(0.20, 0.6),    # 0.27M  / d_S 256
    "efficientnet3d_t3": _make_efficientnet_coeff_builder(0.30, 0.7),    # 0.51M  / d_S 384
    "efficientnet3d_t4": _make_efficientnet_coeff_builder(0.40, 0.8),    # 1.04M  / d_S 512
    "efficientnet3d_t5": _make_efficientnet_coeff_builder(0.60, 1.0),    # 1.84M  / d_S 768
    "efficientnet3d_t6": _make_efficientnet_coeff_builder(0.80, 1.0),    # 3.11M  / d_S 1024
}


def create_3d_backbone(arch: str, in_channels: int = _DEFAULT_IN_CHANNELS) -> nn.Module:
    key = arch.lower()
    if key not in _ARCH_REGISTRY:
        raise ValueError(
            f"Unknown 3D arch '{arch}'. Available: {sorted(_ARCH_REGISTRY)}"
        )
    return _ARCH_REGISTRY[key](in_channels=in_channels)


@torch.no_grad()
def probe_feat_dim(backbone: nn.Module, in_channels: int, probe_shape=(8, 8, 8)) -> int:
    """Return the backbone's pooled output dim (d_S)."""
    feat_dim = getattr(backbone, "feat_dim", None)
    if feat_dim is not None:
        return int(feat_dim)
    was_training = backbone.training
    backbone.eval()
    try:
        dummy = torch.zeros(2, in_channels, *probe_shape)
        out = backbone(dummy)
        return int(out.shape[1])
    finally:
        backbone.train(was_training)
