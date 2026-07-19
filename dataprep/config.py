"""Dataset configuration loader -- reads named profiles from datasets.yaml."""
from __future__ import annotations

import os
import yaml
from pathlib import Path

_YAML_PATH = Path(__file__).parent / "datasets.yaml"


def _expand_env(value):
    """Expand $VAR / ${VAR} in strings; recurse into dicts/lists."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def load_profile(name: str) -> dict:
    """Load a profile; string values are expandvars'd so YAML can reference $DATA_ROOT etc."""
    with open(_YAML_PATH) as f:
        profiles = yaml.safe_load(f)
    if name not in profiles:
        raise ValueError(f"Unknown dataset profile '{name}'. Available: {list(profiles)}")
    return _expand_env(profiles[name])


def get_teacher_dims_str(profile: dict) -> str:
    """Format teacher_dims as CLI string 'name:dim,name:dim,...'."""
    return ",".join(f"{k}:{v}" for k, v in profile["teacher_dims"].items())
