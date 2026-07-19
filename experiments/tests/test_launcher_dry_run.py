"""Launcher dry-run tests: load a study, render scripts, assert wiring."""

from pathlib import Path

import pytest

from experiments.config import Experiment, cv_5fold_eval
from experiments.launch import _load_study, _submit_experiment
from experiments.phase_detector import RadiologyPhaseDetector
from experiments.pipeline import build_experiment_jobs


REPO_ROOT = Path(__file__).resolve().parents[2]
# fixtures live outside studies/ (that dir holds production-only studies)
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_cv_smoke_loads_one_cv_cell():
    name, exps = _load_study(FIXTURES / "cv_smoke.py")
    assert name == "cv_smoke"
    assert len(exps) == 1
    assert len(exps[0].evaluations) == 1
    assert exps[0].evaluations[0].protocol == "cv_5fold"


def test_kd_smoke_loads_training_only():
    name, exps = _load_study(FIXTURES / "kd_smoke.py")
    assert name == "kd_smoke"
    assert len(exps) == 1
    assert exps[0].training is not None
    assert exps[0].training.arch == "mobileone_mu1"
    assert exps[0].evaluations == []


def test_load_study_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        _load_study(tmp_path / "nonexistent.py")


def test_load_study_no_experiments_raises(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("STUDY_NAME = 'x'\n# no EXPERIMENTS\n")
    with pytest.raises(ValueError, match="EXPERIMENTS"):
        _load_study(bad)


def test_dry_run_renders_all_scripts(tmp_path, monkeypatch):
    """End-to-end dry-run: one cv_5fold cell writes 3 scripts (1 extract + 1 evaluate + 1 aggregate)."""
    from experiments import pipeline as p, config as c
    monkeypatch.setattr(p, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(c, "RUNS_DIR", tmp_path)

    run_dir = tmp_path / "cv_smoke" / "curia2_rspect"
    exp = Experiment(
        name="curia2_rspect", study="cv_smoke",
        evaluations=[
            cv_5fold_eval(
                wrapper="curia2", dataset="rspect",
                cache_dir=run_dir / "cache", labels_json=tmp_path / "l.json",
                output_dir=run_dir,  # == exp.run_dir() so the aggregate relative_to resolves
            ),
        ],
    )
    jobs = build_experiment_jobs(exp)
    _submit_experiment(
        exp, jobs,
        detector=RadiologyPhaseDetector(),
        skip_done=False,
        dry_run=True,
    )

    jobs_dir = exp.run_dir() / "jobs"
    scripts = sorted(jobs_dir.glob("*.sh"))
    assert len(scripts) == 3, f"Expected 3 scripts, got {[s.name for s in scripts]}"

    names = {s.name for s in scripts}
    assert any(n.startswith("02_") and "extract" in n and "curia2_rspect" in n for n in names)
    assert any(n.startswith("03_") for n in names)
    assert any(n.startswith("04_") and "aggregate" in n for n in names)


def test_dry_run_skips_completed_phases(tmp_path, monkeypatch):
    """If skip_done=True and a valid summary.json is present, evaluate should not be submitted."""
    import json
    from experiments import pipeline as p, config as c
    monkeypatch.setattr(p, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(c, "RUNS_DIR", tmp_path)

    run_dir = tmp_path / "t" / "single"
    out_dir = run_dir / "cell"
    out_dir.mkdir(parents=True)
    # a numeric pooled.auroc makes evaluate_done() return True
    (out_dir / "summary.json").write_text(json.dumps({"pooled": {"auroc": 0.75}}))

    exp = Experiment(
        name="single",
        study="t",
        evaluations=[
            cv_5fold_eval(
                wrapper="curia2", dataset="rspect",
                cache_dir=run_dir / "cache", labels_json=tmp_path / "l.json",
                output_dir=out_dir,
            ),
        ],
    )
    jobs = build_experiment_jobs(exp)
    submitted = _submit_experiment(
        exp, jobs,
        detector=RadiologyPhaseDetector(),
        skip_done=True,
        dry_run=True,
    )
    assert "evaluate:" not in str(submitted)
