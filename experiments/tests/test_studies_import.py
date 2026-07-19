"""Every study file should load cleanly and declare EXPERIMENTS."""

from pathlib import Path

import pytest

from experiments.config import Experiment
from experiments.launch import _load_study


EXPERIMENTS_DIR = Path(__file__).resolve().parents[1]
STUDIES_DIR = EXPERIMENTS_DIR / "studies"
FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"

def _is_study_entrypoint(path: Path) -> bool:
    """A study's launchable entrypoint: stem matches its package dir + declares EXPERIMENTS."""
    return path.stem == path.parent.name and "EXPERIMENTS" in path.read_text()


_STUDY_PATHS = sorted(
    [p for p in STUDIES_DIR.glob("*/*.py") if _is_study_entrypoint(p)]
    + [p for p in FIXTURES_DIR.glob("*.py") if not p.name.startswith("_")]
)


@pytest.mark.parametrize("study_path", _STUDY_PATHS, ids=lambda p: p.stem)
def test_study_loads_and_declares_experiments(study_path):
    try:
        name, experiments = _load_study(study_path)
    except (FileNotFoundError, SystemExit) as exc:
        # needs gitignored cluster-only data under data/evaluation/; skip on fresh checkout
        pytest.skip(f"{study_path.name} needs cluster-only data (not on this checkout): {exc}")
    assert isinstance(name, str) and name
    assert isinstance(experiments, list) and experiments
    for e in experiments:
        assert isinstance(e, Experiment)
        assert e.name
