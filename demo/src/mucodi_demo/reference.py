"""Python reference of the student evaluation preprocessing, built from the thesis code itself.

The functions below load ``rate_eval``'s own modules by file path (``readers.py``,
``affine_repair.py``, ``preprocess.py``, ``models/common.py``), so the reference is the code that
produced the thesis embeddings, not a re-implementation. Importing the ``rate_eval`` package
normally would pull in the whole harness (Hydra, transformers, teacher wrappers), which the demo
does not need.

Pipeline (``configs/model/mucokd_student3d.yaml``): NIfTI -> HU -> closest canonical RAS ->
``(D, H, W)`` -> full field of view trilinear resize to 128x256x256 (``align_corners=False``) ->
four clamped HU windows, no normalisation.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from functools import cache
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

from mucodi_demo.students import REPO_ROOT

RATE_EVAL = REPO_ROOT / "rate-evals" / "rate_eval"
TARGET_SHAPE: tuple[int, int, int] = (128, 256, 256)
# rve.ANATOMICAL_WINDOWS["CT"] (YalaLab/rave, vision_engine/utils/windowing_utils.py), in the
# student's channel order from configs/model/mucokd_student3d.yaml.
WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("lung", -600.0, 1500.0),
    ("mediastinum", 50.0, 400.0),
    ("soft_tissue", 50.0, 350.0),
    ("bone", 400.0, 1800.0),
)


def _load(relative: str, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, RATE_EVAL / relative)
    if spec is None or spec.loader is None:
        raise ImportError(relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


@cache
def _modules() -> dict[str, ModuleType]:
    # models/common.py imports rve only for the window table; give it exactly that table.
    if "rve" not in sys.modules:
        rve = types.ModuleType("rve")
        rve.ANATOMICAL_WINDOWS = {  # type: ignore[attr-defined]
            "CT": {name: {"center": c, "width": w} for name, c, w in WINDOWS}
        }
        sys.modules["rve"] = rve
    return {
        "readers": _load("datasets/readers.py", "_rate_eval_readers"),
        "affine_repair": _load("datasets/affine_repair.py", "_rate_eval_affine_repair"),
        "preprocess": _load("datasets/preprocess.py", "_rate_eval_preprocess"),
        "common": _load("models/common.py", "_rate_eval_common"),
    }


def read_hu(path: Path) -> np.ndarray:
    """NIfTI -> float32 HU (D, H, W) in closest canonical RAS, as the eval loader reads it."""
    m = _modules()
    img = m["affine_repair"].safe_canonical(str(path))
    return np.asarray(img.get_fdata(), dtype=np.float32).transpose(2, 0, 1)


def canonical_affine(path: Path) -> np.ndarray:
    """Affine of the canonical image, i.e. voxel (x, y, z) of ``read_hu`` before the transpose."""
    m = _modules()
    return np.asarray(m["affine_repair"].safe_canonical(str(path)).affine, dtype=np.float64)


def resize(hu: np.ndarray, shape: tuple[int, int, int] = TARGET_SHAPE) -> np.ndarray:
    """Full field of view trilinear resize with the thesis code path."""
    m = _modules()
    x = torch.from_numpy(np.ascontiguousarray(hu))[None, None]
    with torch.no_grad():
        y = m["preprocess"]._interpolate(x, list(shape), "trilinear", False)
    return np.asarray(y[0, 0].numpy(), dtype=np.float32)


def window(resized: np.ndarray) -> np.ndarray:
    """(D, H, W) HU -> (1, 4, D, H, W) model input with the thesis windowing code."""
    m = _modules()
    x = torch.from_numpy(np.ascontiguousarray(resized))[None, None]
    with torch.no_grad():
        y = m["common"].batch_apply_ct_windowing(
            x, ct_window_type=[w[0] for w in WINDOWS], modality="CT", per_sample=True
        )
    return np.asarray(y.numpy(), dtype=np.float32)


def model_input(path: Path) -> np.ndarray:
    return window(resize(read_hu(path)))
