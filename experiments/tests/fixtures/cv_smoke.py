"""cv_5fold mini-study fixture for the launcher dry-run test (not a production study)."""
from pathlib import Path

from experiments.config import Experiment, cv_5fold_eval

STUDY_NAME = "cv_smoke"

# Repo-relative: fixtures/ -> tests/ -> experiments/ -> repo root.
REPO = Path(__file__).resolve().parents[3]
CACHE_ROOT = REPO / "experiments" / "runs" / "_fixture_cache" / "curia2"
RUNS_OUT = REPO / "experiments" / "runs" / STUDY_NAME
DATA_EVAL = REPO / "data" / "evaluation"


EXPERIMENTS = [
    Experiment(
        name="curia2_rspect",
        study=STUDY_NAME,
        evaluations=[
            cv_5fold_eval(
                wrapper="curia2",
                dataset="rspect",
                cache_dir=CACHE_ROOT / "curia2_rspect",
                labels_json=DATA_EVAL / "rspect_labels.json",
                output_dir=RUNS_OUT / "curia2_rspect",
            ),
        ],
    ),
]
