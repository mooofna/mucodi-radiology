"""Build the ONNX model assets for every rung and describe them in ``models.json``."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx.external_data_helper import set_external_data, uses_external_data

from mucodi_demo.export import check_parity, export_core, export_taps
from mucodi_demo.geometry import tap_geometry
from mucodi_demo.students import INPUT_SHAPE, RUNGS, load_student

# Rungs whose stage maps the network view can show.
TAPS_RUNGS: tuple[str, ...] = ("tmin", "t2", "t6", "b0")
# GitHub Pages serves files up to 100 MB; larger graphs keep their weights in chunk files.
CHUNK_BYTES = 90 * 1024 * 1024
PARITY_MIN_COSINE = 0.99999


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def split_external(path: Path, chunk_bytes: int = CHUNK_BYTES) -> list[Path]:
    """Move initializers of an oversized ONNX file into chunk files below ``chunk_bytes``."""
    model = onnx.load(str(path))
    chunks: list[bytearray] = [bytearray()]
    for tensor in model.graph.initializer:
        if uses_external_data(tensor):
            raise ValueError(f"{path} already uses external data")
        raw = tensor.raw_data
        if len(raw) < 1024:
            continue
        if len(chunks[-1]) and len(chunks[-1]) + len(raw) > chunk_bytes:
            chunks.append(bytearray())
        index = len(chunks) - 1
        offset = len(chunks[index])
        chunks[index].extend(raw)
        set_external_data(
            tensor, location=f"{path.name}.data{index}", offset=offset, length=len(raw)
        )
        tensor.ClearField("raw_data")
    files = []
    for i, data in enumerate(chunks):
        target = path.parent / f"{path.name}.data{i}"
        target.write_bytes(bytes(data))
        files.append(target)
    onnx.save_model(model, str(path))
    return files


def build_models(
    checkpoints: Path, out_dir: Path, rungs: tuple[str, ...] = RUNGS
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    volume = np.random.default_rng(0).random(INPUT_SHAPE, dtype=np.float32)
    entries = []
    for rung in rungs:
        student = load_student(rung, checkpoints / f"{rung}.pth.tar")
        core = export_core(student, out_dir)
        parity = check_parity(student, core, volume)
        if parity.cosine < PARITY_MIN_COSINE:
            raise AssertionError(
                f"{rung}: ONNX/PyTorch cosine {parity.cosine} < {PARITY_MIN_COSINE}"
            )
        graphs = {"core": core}
        if rung in TAPS_RUNGS:
            graphs["taps"] = export_taps(student, out_dir)
        files: dict[str, list[dict[str, object]]] = {}
        for kind, path in graphs.items():
            parts = [path]
            if path.stat().st_size > CHUNK_BYTES:
                parts += split_external(path)
            files[kind] = [
                {"path": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)} for p in parts
            ]
        entries.append(
            {
                "rung": rung,
                "params": student.n_params,
                "feat_dim": student.feat_dim,
                "checkpoint_step": student.step,
                "monai_version": student.monai_version,
                "files": files,
                "taps": [g.as_dict() for g in tap_geometry(student.backbone)]
                if rung in TAPS_RUNGS
                else None,
                "parity": {"cosine": parity.cosine, "max_abs": parity.max_abs},
            }
        )
        print(f"{rung}: {student.n_params} params, cosine {parity.cosine:.9f}", flush=True)
    manifest: dict[str, object] = {
        "input": {"name": "volume", "shape": list(INPUT_SHAPE), "dtype": "float32"},
        "windows": [
            {"name": "lung", "center": -600, "width": 1500},
            {"name": "mediastinum", "center": 50, "width": 400},
            {"name": "soft_tissue", "center": 50, "width": 350},
            {"name": "bone", "center": 400, "width": 1800},
        ],
        "rungs": entries,
    }
    (out_dir / "models.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
