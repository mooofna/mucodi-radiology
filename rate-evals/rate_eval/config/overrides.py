"""Apply argparse-parsed CLI overrides to an OmegaConf DictConfig via a declarative mapping table."""

from __future__ import annotations

from argparse import Namespace
from typing import Any, Dict, Tuple

from omegaconf import DictConfig, OmegaConf

# (CLI attr, config dotted-key) pairs; applied when the CLI attr is not None.
_CLI_OVERRIDE_MAP: Tuple[Tuple[str, str], ...] = (
    ("batch_size", "hardware.batch_size_per_gpu"),
    ("device", "hardware.device"),
    ("num_workers", "hardware.num_workers_per_gpu"),
    ("model_repo_id", "model.repo_id"),
    ("model_revision", "model.revision"),
    ("ct_window_type", "model.preprocessing.ct.window_type"),
    ("pool_op", "model.extraction.pool_op"),
    ("modality", "dataset.modality"),
)


def apply_cli_overrides(config: DictConfig, args: Namespace) -> Dict[str, Any]:
    """Apply CLI override flags to config in place; return the applied overrides."""
    applied: Dict[str, Any] = {}
    for cli_attr, config_key in _CLI_OVERRIDE_MAP:
        value = getattr(args, cli_attr, None)
        if value is None:
            continue
        OmegaConf.update(config, config_key, value)
        applied[config_key] = value
    return applied


__all__ = ["apply_cli_overrides"]
