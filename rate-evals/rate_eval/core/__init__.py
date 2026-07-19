"""Pure primitives for rate_eval -- error classes, logging, device, seeding, cross-validation."""

from .errors import RATEEvalError, ModelError, DatasetError
from .logging import setup_logging, get_logger

# device/seed import torch -- kept out so get_logger stays torch-free

__all__ = [
    "RATEEvalError",
    "ModelError",
    "DatasetError",
    "setup_logging",
    "get_logger",
]
