"""Offline tests: nothing here needs trained checkpoints, scans or the thesis result files."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from mucodi_demo.students import build_backbone


@pytest.fixture(scope="session")
def tmin() -> nn.Module:
    """An untrained TMIN backbone with seeded weights: the smallest rung, fast on a CPU."""
    torch.manual_seed(0)
    return build_backbone("tmin")
