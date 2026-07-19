"""Toolkit for evaluating radiology foundation models."""

from .config import get_config_value, load_config, setup_pipeline as setup_pipeline_new
from .components import create_dataset, create_model
from .core.logging import get_logger, setup_logging

# `setup_device` is resolved lazily via __getattr__ so bare `import rate_eval` stays torch-free.

__version__ = "2.1.0"
__author__ = "RATE Evaluation Team"

__all__ = [
    "load_config",
    "get_config_value",
    "create_model",
    "create_dataset",
    "setup_logging",
    "get_logger",
    "setup_device",  # resolved lazily via __getattr__
]


def __getattr__(name: str):
    if name == "setup_device":
        from .core.device import setup_device as _setup_device

        return _setup_device
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def setup_pipeline(config_path=None, log_level="INFO", **overrides):
    """Initialize the RATE evaluation pipeline with OmegaConf support."""
    config = setup_pipeline_new(config_path=config_path, **overrides)

    logging_config = getattr(config, "logging", {})
    setup_logging(
        level=getattr(logging_config, "level", log_level),
        format_str=getattr(logging_config, "format", None),
    )

    logger = get_logger(__name__)
    logger.info(f"RATE Evaluation Pipeline v{__version__} initialized with OmegaConf")

    return config
