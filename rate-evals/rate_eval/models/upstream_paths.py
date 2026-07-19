"""Guarded `sys.path` mutation for loading upstream model code (CT-CLIP, TANGERINE)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union


def add_upstream_to_path(upstream_dir: Union[str, Path]) -> Path:
    """Idempotently prepend `upstream_dir` to `sys.path`; returns the resolved path."""
    upstream_dir = Path(upstream_dir).resolve()
    if not upstream_dir.exists():
        raise FileNotFoundError(
            f"upstream directory not found: {upstream_dir}. "
            "Verify scripts/setup_external_teachers.sh has been run.",
        )
    upstream_str = str(upstream_dir)
    if upstream_str not in sys.path:
        sys.path.insert(0, upstream_str)
    return upstream_dir


__all__ = ["add_upstream_to_path"]
