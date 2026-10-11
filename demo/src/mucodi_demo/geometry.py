"""Where each feature-map cell of a student sits on the 128x256x256 input grid.

MONAI's EfficientNet pads every convolution with a static, TensorFlow-style SAME padding computed
for a cubic nominal image size (224 for the t-line, 224 to 528 for B0 to B6), not for the real
input. The centre of a feature cell is therefore shifted by a rung-specific offset that grows with
depth. This module derives the exact mapping ``input_index = stride * cell_index + offset`` from
the instantiated pad and conv modules, plus the local receptive field, so the web overlays can be
placed exactly.

The squeeze-excitation blocks pool over the whole feature map, so strictly every cell depends on
every input voxel. ``rf`` is the local receptive field of the convolution path only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from mucodi_demo.export import TAP_NAMES, Taps
from mucodi_demo.students import INPUT_SHAPE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from monai.networks.nets.efficientnet import EfficientNet, MBConvBlock


@dataclass(frozen=True)
class TapGeometry:
    name: str
    shape: tuple[int, int, int, int]  # channels, D, H, W
    stride: tuple[int, int, int]
    offset: tuple[float, float, float]  # input-voxel coordinate (D, H, W) of cell (0, 0, 0)
    rf: tuple[int, int, int]  # local receptive field in input voxels

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class _Walker:
    """Receptive-field arithmetic along the convolution path, per axis (D, H, W)."""

    def __init__(self) -> None:
        self.jump = [1, 1, 1]
        self.start = [0.0, 0.0, 0.0]
        self.rf = [1, 1, 1]

    def conv(self, pad: nn.Module, conv: nn.Module) -> None:
        if isinstance(pad, nn.Identity):
            before = [0, 0, 0]
        else:
            p = tuple(pad.padding)  # type: ignore[arg-type]
            # ConstantPad3d order is (W_left, W_right, H_left, H_right, D_left, D_right).
            before = [int(p[4]), int(p[2]), int(p[0])]
        if any(int(x) != 0 for x in conv.padding):  # type: ignore[union-attr]
            raise ValueError("expected padding to live in the separate pad module")
        k = [int(x) for x in conv.kernel_size]  # type: ignore[union-attr]
        s = [int(x) for x in conv.stride]  # type: ignore[union-attr]
        d = [int(x) for x in conv.dilation]  # type: ignore[union-attr]
        for a in range(3):
            span = (k[a] - 1) * d[a]
            self.start[a] += (span / 2 - before[a]) * self.jump[a]
            self.rf[a] += span * self.jump[a]
            self.jump[a] *= s[a]

    def snapshot(self, name: str, shape: tuple[int, ...]) -> TapGeometry:
        c, dd, hh, ww = (int(x) for x in shape[1:])
        return TapGeometry(
            name=name,
            shape=(c, dd, hh, ww),
            stride=(self.jump[0], self.jump[1], self.jump[2]),
            offset=(self.start[0], self.start[1], self.start[2]),
            rf=(self.rf[0], self.rf[1], self.rf[2]),
        )


def tap_geometry(backbone: nn.Module) -> list[TapGeometry]:
    """Geometry of every tap of :class:`mucodi_demo.export.Taps`, in output order."""
    net = cast("EfficientNet", backbone.net)
    with torch.no_grad():
        outputs = Taps(backbone).eval()(torch.zeros(INPUT_SHAPE))
    shapes = [tuple(t.shape) for t in outputs[1:]]

    walker = _Walker()
    walker.conv(net._conv_stem_padding, net._conv_stem)
    geoms = [walker.snapshot(TAP_NAMES[0], shapes[0])]
    for i, stage in enumerate(net._blocks):
        for block in cast("Iterable[MBConvBlock]", stage):
            if block.expand_ratio != 1:
                walker.conv(block._expand_conv_padding, block._expand_conv)
            walker.conv(block._depthwise_conv_padding, block._depthwise_conv)
            walker.conv(block._project_conv_padding, block._project_conv)
        geoms.append(walker.snapshot(TAP_NAMES[1 + i], shapes[1 + i]))
    walker.conv(net._conv_head_padding, net._conv_head)
    geoms.append(walker.snapshot(TAP_NAMES[-1], shapes[-1]))

    for g in geoms:
        expected = tuple((INPUT_SHAPE[2 + a] + g.stride[a] - 1) // g.stride[a] for a in range(3))
        if g.shape[1:] != expected:
            raise AssertionError(
                f"{g.name}: shape {g.shape[1:]} != ceil(input / stride) {expected}"
            )
    return geoms


def impulse_centre(
    backbone: nn.Module, tap_index: int, cell: tuple[int, int, int], seed: int = 0
) -> tuple[float, float, float]:
    """Centre of mass of |d cell / d input| inside the cell's local receptive field.

    The gradient outside the local field comes only from the global squeeze-excitation
    context and is excluded so the test measures the convolutional centre.
    """
    geom = tap_geometry(backbone)[tap_index]
    gen = torch.Generator().manual_seed(seed)
    x = torch.rand(INPUT_SHAPE, generator=gen).requires_grad_(True)
    outputs = Taps(backbone).eval()(x)
    target = outputs[1 + tap_index][0, :, cell[0], cell[1], cell[2]].sum()
    (grad,) = torch.autograd.grad(target, x)
    g = grad.abs().sum(dim=1)[0]  # (D, H, W)
    lo, hi = [], []
    for a in range(3):
        centre = geom.stride[a] * cell[a] + geom.offset[a]
        half = (geom.rf[a] - 1) / 2
        lo.append(max(0, round(centre - half)))
        hi.append(min(INPUT_SHAPE[2 + a], round(centre + half) + 1))
    box = g[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]
    total = box.sum()
    coords = []
    for a in range(3):
        idx = torch.arange(box.shape[a], dtype=torch.float64) + lo[a]
        marginal = box.sum(dim=tuple(b for b in range(3) if b != a)).double()
        coords.append(float((idx * marginal).sum() / total))
    return (coords[0], coords[1], coords[2])
