"""Contract tests for the dataset registry + every registered DatasetSpec."""
from __future__ import annotations

from pathlib import Path

import pytest

from dataprep.datasets.spec import Access, DatasetSpec
from dataprep.datasets.registry import REGISTRY, get

_REPO_ROOT = Path(__file__).resolve().parents[2]

ALL = sorted(REGISTRY)


def test_registry_nonempty_and_keys_match_names():
    assert len(REGISTRY) >= 12
    for name, spec in REGISTRY.items():
        assert isinstance(spec, DatasetSpec)
        assert spec.name == name, f"registry key {name!r} != spec.name {spec.name!r}"


def test_names_are_lowercase_slugs():
    for name in ALL:
        assert name == name.lower()
        assert " " not in name


@pytest.mark.parametrize("name", ALL)
def test_spec_fields_well_typed(name):
    spec = get(name)
    assert isinstance(spec.access, Access)
    assert spec.modality and isinstance(spec.modality, str)
    assert spec.role in {"scored", "kd-corpus", "descriptive", "anchor", "candidate"}
    assert spec.token_env is None or isinstance(spec.token_env, str)
    assert spec.stage is None or callable(spec.stage)
    assert spec.build_labels is None or callable(spec.build_labels)
    assert isinstance(spec.committed_outputs, tuple)
    for rel in spec.committed_outputs:
        assert isinstance(rel, str) and not rel.startswith("/"), rel
        assert rel.startswith("data/"), f"{name}: committed output not under data/: {rel}"


@pytest.mark.parametrize("name", ALL)
def test_committed_outputs_exist(name):
    """Once staged, a spec's declared outputs must all be present; skip when none are (unshipped)."""
    spec = get(name)
    if not any((_REPO_ROOT / rel).is_file() for rel in spec.committed_outputs):
        pytest.skip(f"{name}: outputs not staged (run `python -m dataprep.datasets {name} --all`)")
    missing = [rel for rel in spec.committed_outputs if not (_REPO_ROOT / rel).is_file()]
    assert not missing, f"{name}: partially staged, missing {missing}"


def test_get_unknown_raises():
    with pytest.raises(KeyError):
        get("does_not_exist")


def test_scored_cells_present():
    scored = {n for n in ALL if get(n).role == "scored"}
    assert {"lidc", "stoic2021", "rspect", "radchestct"} <= scored
