"""Golden embeddings for the bundled cases, from the thesis pipeline in PyTorch.

For each case and rung this stores the embedding of the full-precision reference input (native
NIfTI -> canonical -> resize -> windows) and checks that the int16 model-grid file the browser
loads gives the same embedding.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from mucodi_demo import reference
from mucodi_demo.export import cosine
from mucodi_demo.students import RUNGS, load_student

GRID_MIN_COSINE = 0.99999


def grid_input(grid_path: Path) -> np.ndarray:
    """The browser's instant path: dequantise the grid file and window it."""
    img = nib.load(str(grid_path))
    hu_ijk = np.asarray(img.get_fdata(), dtype=np.float32)  # (W, H, D); applies scl_slope = 0.1
    return reference.window(hu_ijk.transpose(2, 1, 0))


def build_goldens(checkpoints: Path, cases_dir: Path, downloads: Path) -> dict[str, object]:
    cases = json.loads((cases_dir / "cases.json").read_text())["cases"]
    inputs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for case in cases:
        native = next((downloads / case["id"] / "nifti").glob("native*.nii"))
        inputs[case["id"]] = (
            reference.model_input(native),
            grid_input(cases_dir / case["files"]["grid"]),
        )
    out: dict[str, dict[str, object]] = {cid: {} for cid in inputs}
    torch.set_num_threads(max(1, torch.get_num_threads()))
    for rung in RUNGS:
        student = load_student(rung, checkpoints / f"{rung}.pth.tar")
        for cid, (ref_in, grid_in) in inputs.items():
            with torch.no_grad():
                ref = student.backbone(torch.from_numpy(ref_in)).numpy()[0]
                grid = student.backbone(torch.from_numpy(grid_in)).numpy()[0]
            c = cosine(ref, grid)
            if c < GRID_MIN_COSINE:
                raise AssertionError(f"{cid}/{rung}: grid file cosine {c} < {GRID_MIN_COSINE}")
            out[cid][rung] = {"embedding": [float(v) for v in ref], "grid_cosine": c}
        print(
            rung,
            {cid: round(float(v[rung]["grid_cosine"]), 9) for cid, v in out.items()},
            flush=True,
        )  # type: ignore[index]
    result: dict[str, object] = {
        "source": "PyTorch fp32, thesis preprocessing (rate_eval readers, preprocess, windowing)",
        "cases": out,
    }
    (cases_dir / "goldens.json").write_text(json.dumps(result) + "\n")
    return result
