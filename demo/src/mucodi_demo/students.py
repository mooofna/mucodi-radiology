"""Load the trained 3D-EfficientNet students from their training checkpoints.

The backbone definition is imported from the repository itself (``models/efficientnet3d.py``),
so the demo always builds exactly the network the thesis trained. Only the backbone is kept:
the per-teacher projector heads are discarded after training, as in the thesis's inference path.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]

# Order follows the thesis ladder (Table 4.1); B1 was never trained.
RUNGS: tuple[str, ...] = (
    "tmin",
    "t0",
    "t1",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "b0",
    "b2",
    "b3",
    "b4",
    "b5",
    "b6",
)
IN_CHANNELS = 4
INPUT_SHAPE: tuple[int, int, int, int, int] = (1, IN_CHANNELS, 128, 256, 256)


def _import_backbone_factory() -> object:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from models.efficientnet3d import create_3d_backbone

    return create_3d_backbone


@dataclass(frozen=True)
class LoadedStudent:
    rung: str
    backbone: nn.Module
    feat_dim: int
    n_params: int
    step: int
    monai_version: str


def build_backbone(rung: str) -> nn.Module:
    """Build an untrained backbone for ``rung`` with the export-safe settings applied."""
    if os.environ.get("EFFNET3D_KEEP_SILU", "0") == "1":
        raise RuntimeError("EFFNET3D_KEEP_SILU=1 would change the network; unset it for export")
    create_3d_backbone = _import_backbone_factory()
    backbone: nn.Module = create_3d_backbone(f"efficientnet3d_{rung}", in_channels=IN_CHANNELS)  # type: ignore[operator]
    # MONAI's memory-efficient swish is a custom autograd function; the plain version exports.
    backbone.net.set_swish(memory_efficient=False)  # type: ignore[union-attr, operator]
    return backbone.eval()


def load_student(rung: str, checkpoint: Path) -> LoadedStudent:
    """Load the backbone weights of ``rung`` from a ``step_*.pth.tar`` training checkpoint."""
    import monai

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = ckpt.get("arch")
    if arch != f"efficientnet3d_{rung}":
        raise ValueError(f"{checkpoint} holds {arch!r}, expected efficientnet3d_{rung}")
    stamp = str(ckpt.get("monai_version", ""))
    if stamp and stamp != monai.__version__:
        raise RuntimeError(
            f"checkpoint was trained with MONAI {stamp}, environment has {monai.__version__}"
        )
    prefix = "backbone."
    state = {k[len(prefix) :]: v for k, v in ckpt["state_dict"].items() if k.startswith(prefix)}
    backbone = build_backbone(rung)
    backbone.load_state_dict(state, strict=True)
    n_params = sum(p.numel() for p in backbone.parameters())
    return LoadedStudent(
        rung=rung,
        backbone=backbone,
        feat_dim=int(backbone.feat_dim),  # type: ignore[arg-type]
        n_params=n_params,
        step=int(ckpt.get("step", -1)),
        monai_version=stamp,
    )
