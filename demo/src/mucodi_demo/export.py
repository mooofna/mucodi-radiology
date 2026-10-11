"""Export student backbones to ONNX and prove parity with PyTorch.

Two graphs per rung:

* ``core``: input ``volume`` (1, 4, 128, 256, 256) -> ``embedding`` (1, d_S). Used for every timing
  in the demo, so browser numbers measure the same forward pass as thesis Table 6.5.
* ``taps``: the same network with the stem, the seven stage outputs and the pre-pool head map as
  extra outputs, for the network view.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

from mucodi_demo.students import INPUT_SHAPE, LoadedStudent

OPSET = 18
TAP_NAMES: tuple[str, ...] = (
    "stem",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "stage7",
    "head",
)


class Core(nn.Module):
    """The thesis forward pass: pooled embedding before the final SiLU."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, volume: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.backbone(volume)
        return out


class Taps(nn.Module):
    """Replicates ``_EfficientNet3DFeat.forward`` and also returns the intermediate maps."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.net = backbone.net  # type: ignore[union-attr]

    def forward(self, volume: torch.Tensor) -> tuple[torch.Tensor, ...]:
        net = self.net
        x = net._conv_stem(net._conv_stem_padding(volume))
        x = net._swish(net._bn0(x))
        maps = [x]
        for stage in net._blocks:
            x = stage(x)
            maps.append(x)
        x = net._conv_head(net._conv_head_padding(x))
        head = net._bn1(x)
        maps.append(head)
        embedding = net._avg_pooling(head).flatten(start_dim=1)
        return (embedding, *maps)


def _export(module: nn.Module, path: Path, output_names: list[str]) -> None:
    dummy = torch.zeros(INPUT_SHAPE, dtype=torch.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            module,
            (dummy,),
            str(path),
            input_names=["volume"],
            output_names=output_names,
            opset_version=OPSET,
            dynamo=False,
            do_constant_folding=True,
        )
    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def export_core(student: LoadedStudent, out_dir: Path) -> Path:
    path = out_dir / f"{student.rung}.core.onnx"
    _export(Core(student.backbone).eval(), path, ["embedding"])
    return path


def export_taps(student: LoadedStudent, out_dir: Path) -> Path:
    path = out_dir / f"{student.rung}.taps.onnx"
    _export(Taps(student.backbone).eval(), path, ["embedding", *TAP_NAMES])
    return path


@dataclass(frozen=True)
class Parity:
    rung: str
    graph: str
    cosine: float
    max_abs: float
    max_rel: float


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    return float(a64 @ b64 / (np.linalg.norm(a64) * np.linalg.norm(b64)))


def check_parity(student: LoadedStudent, onnx_path: Path, volume: np.ndarray) -> Parity:
    """Compare the ONNX Runtime CPU embedding with PyTorch on ``volume`` (1, 4, D, H, W)."""
    with torch.no_grad():
        ref = student.backbone(torch.from_numpy(volume)).numpy()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 0
    sess = ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])
    got = sess.run(["embedding"], {"volume": volume})[0]
    diff = np.abs(got - ref)
    return Parity(
        rung=student.rung,
        graph=onnx_path.name,
        cosine=cosine(got, ref),
        max_abs=float(diff.max()),
        max_rel=float((diff / (np.abs(ref) + 1e-6)).max()),
    )


def write_report(rows: list[Parity], path: Path) -> None:
    path.write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n")
