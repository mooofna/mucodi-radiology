"""Contract every teacher wrapper must satisfy."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch


@runtime_checkable
class TeacherWrapper(Protocol):
    """Method-level contract for the radiology FM wrappers in `rate_eval.models`."""

    def extract_features(self, inputs: torch.Tensor, modality: str) -> np.ndarray: ...


__all__ = ["TeacherWrapper"]
