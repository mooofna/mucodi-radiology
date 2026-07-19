"""Logging configuration with optional colorlog support."""

import logging
import os
import sys
import warnings
from typing import Optional

try:
    import colorlog

    HAS_COLORLOG = True
except ImportError:
    HAS_COLORLOG = False

# at import time, before setup_logging()
warnings.filterwarnings("ignore", category=UserWarning, module="scipy")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="runpy")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def setup_logging(
    level: str = "INFO",
    format_str: Optional[str] = None,
    colored: bool = True,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    debug: bool = False,
) -> None:
    """Configure pipeline logging; non-zero ranks are quieted to WARNING unless debug."""
    if world_size is not None and world_size > 1 and not debug:
        if rank == 0:
            actual_level = level if level != "INFO" else "INFO"
        else:
            actual_level = "WARNING"
    else:
        actual_level = level
    numeric_level = getattr(logging, actual_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {actual_level}")

    if colored and HAS_COLORLOG:
        if format_str is None:
            format_str = "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        formatter = colorlog.ColoredFormatter(
            format_str,
            datefmt="%H:%M:%S",
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
        )

        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)

        logging.basicConfig(level=numeric_level, handlers=[handler])
    else:
        if format_str is None:
            format_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        logging.basicConfig(
            level=numeric_level, format=format_str, handlers=[logging.StreamHandler(sys.stdout)]
        )

    # quiet noisy third-party loggers
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
