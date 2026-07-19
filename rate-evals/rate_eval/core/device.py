"""Torch device selection."""

from typing import Optional

import torch

from .logging import get_logger


def setup_device(device_spec: Optional[str] = None) -> torch.device:
    """Set up and validate a torch device. Defaults to CUDA when available, CPU otherwise."""
    if device_spec is None:
        device_spec = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device_spec)

    if device.type == "cuda" and not torch.cuda.is_available():
        logger = get_logger(__name__)
        logger.warning("CUDA requested but not available, falling back to CPU")
        device = torch.device("cpu")

    return device
