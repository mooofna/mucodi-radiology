"""Every dataset YAML's loader class resolves via `_load_dataset_class`."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from rate_eval.components import _load_dataset_class, create_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_CONFIG_DIR = REPO_ROOT / "configs" / "dataset"


def _list_dataset_yamls():
    return sorted(DATASET_CONFIG_DIR.rglob("*.yaml")) if DATASET_CONFIG_DIR.is_dir() else []


def _yaml_id(p):
    try:
        return str(p.relative_to(DATASET_CONFIG_DIR)).replace("/", "__")[:-5]
    except ValueError:
        return p.stem


@pytest.mark.parametrize("yaml_path", _list_dataset_yamls(), ids=_yaml_id)
def test_loader_class_resolves(yaml_path):
    """Every YAML's `loader.class` must be importable via the factory (catches config<->registry drift)."""
    raw = yaml.safe_load(yaml_path.read_text())
    if raw is None or "loader" not in raw:
        pytest.skip("not a dataset YAML")
    loader_spec = raw["loader"]
    name = yaml_path.stem
    try:
        cls = _load_dataset_class(name, loader_spec)
    except Exception as exc:
        # skip optional-dep datasets (RVE-backed) when rve missing
        msg = str(exc).lower()
        if "rve" in msg or "merlin" in msg or "rad_vision_engine" in msg:
            pytest.skip(f"optional dep missing: {exc}")
        raise
    assert cls is not None
    assert hasattr(cls, "__name__")
