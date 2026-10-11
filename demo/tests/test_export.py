"""ONNX export of a student against PyTorch, with seeded random weights."""

from __future__ import annotations

from pathlib import Path

import monai
import numpy as np
import onnxruntime as ort
from torch import nn

from mucodi_demo.export import TAP_NAMES, check_parity, export_core, export_taps
from mucodi_demo.geometry import tap_geometry
from mucodi_demo.students import INPUT_SHAPE, LoadedStudent


def student(backbone: nn.Module) -> LoadedStudent:
    n_params = sum(p.numel() for p in backbone.parameters())
    return LoadedStudent("tmin", backbone, 64, n_params, 0, monai.__version__)


def test_core_export_matches_pytorch(tmin: nn.Module, tmp_path: Path) -> None:
    path = export_core(student(tmin), tmp_path)
    volume = np.random.default_rng(0).random(INPUT_SHAPE, dtype=np.float32)
    parity = check_parity(student(tmin), path, volume)
    assert parity.cosine >= 0.99999
    assert parity.max_abs <= 1e-4


def test_taps_export_has_every_stage_on_its_grid(tmin: nn.Module, tmp_path: Path) -> None:
    path = export_taps(student(tmin), tmp_path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [o.name for o in session.get_outputs()] == ["embedding", *TAP_NAMES]
    outputs = session.run(None, {"volume": np.zeros(INPUT_SHAPE, dtype=np.float32)})
    for out, geom in zip(outputs[1:], tap_geometry(tmin), strict=True):
        assert out.shape[1:] == geom.shape
