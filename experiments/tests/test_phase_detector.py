"""Disk-state phase detection tests."""

import json
from pathlib import Path

import pytest

from experiments.config import (
    Experiment, TrainingConfig,
    cv_5fold_eval,
)
from experiments.phase_detector import RadiologyPhaseDetector, _legacy_extract_done


@pytest.fixture
def detector() -> RadiologyPhaseDetector:
    return RadiologyPhaseDetector()


def _write_valid_summary(out_dir: Path) -> None:
    """Write a minimal valid summary.json (rate-evaluate cv shape: numeric pooled.auroc)."""
    import json
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "cv_folds": 5,
        "cv_seed": 42,
        "feature_dim": 512,
        "n_samples": 10,
        "head_spec": {"kind": "linear"},
        "pooled": {"auroc": 0.75},
        "pooled_auroc_ci": [0.7, 0.8],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary))


def test_train_done_false_when_no_checkpoint(detector, tmp_path):
    exp = Experiment(
        name="t",
        training=TrainingConfig(
            train_steps=1000,
            save_dir=str(tmp_path / "ckpts"),
        ),
    )
    assert detector.train_done(exp) is False


def test_train_done_true_when_checkpoint_exists(detector, tmp_path):
    save_dir = tmp_path / "ckpts"
    save_dir.mkdir()
    (save_dir / "step_0001000.pth.tar").write_text("stub")
    exp = Experiment(
        name="t",
        training=TrainingConfig(
            train_steps=1000,
            save_dir=str(save_dir),
        ),
    )
    assert detector.train_done(exp) is True


def test_train_done_returns_false_when_no_training(detector):
    exp = Experiment(name="noop", training=None)
    assert detector.train_done(exp) is False


def test_extract_done_false_when_cache_empty(detector, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    exp = Experiment(name="x", evaluations=[])
    cfg = cv_5fold_eval(
        wrapper="ctclip_zero_shot", dataset="rspect",
        cache_dir=cache_dir, labels_json=tmp_path / "labels.json",
        output_dir=tmp_path / "out",
    )
    assert detector.extract_done(exp, cfg) is False


def test_extract_done_true_via_legacy_npz(detector, tmp_path):
    cache_dir = tmp_path / "cache"
    (cache_dir / "embeddings" / "test").mkdir(parents=True)
    (cache_dir / "embeddings" / "test" / "sample.npz").write_bytes(b"stub")
    exp = Experiment(name="x", evaluations=[])
    cfg = cv_5fold_eval(
        wrapper="ctclip_zero_shot", dataset="rspect",
        cache_dir=cache_dir, labels_json=tmp_path / "labels.json",
        output_dir=tmp_path / "out",
    )
    assert detector.extract_done(exp, cfg) is True


def test_extract_done_true_when_cache_meta_finished(detector, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    meta_yaml = """\
schema_version: "1.0"
wrapper:
  name: ctclip_zero_shot
dataset:
  name: rspect
extraction:
  started_utc: "2026-05-01T00:00:00Z"
  finished_utc: "2026-05-01T01:00:00Z"
preprocess: {}
"""
    (cache_dir / "cache_meta.yaml").write_text(meta_yaml)
    (cache_dir / "embeddings").mkdir()
    exp = Experiment(name="x", evaluations=[])
    cfg = cv_5fold_eval(
        wrapper="ctclip_zero_shot", dataset="rspect",
        cache_dir=cache_dir, labels_json=tmp_path / "labels.json",
        output_dir=tmp_path / "out",
    )
    assert detector.extract_done(exp, cfg) is True


def test_extract_done_false_when_cache_meta_wrong_wrapper(detector, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    meta_yaml = """\
schema_version: "1.0"
wrapper:
  name: tangerine_vit
dataset:
  name: rspect
extraction:
  started_utc: "2026-05-01T00:00:00Z"
  finished_utc: "2026-05-01T01:00:00Z"
preprocess: {}
"""
    (cache_dir / "cache_meta.yaml").write_text(meta_yaml)
    exp = Experiment(name="x", evaluations=[])
    cfg = cv_5fold_eval(
        wrapper="ctclip_zero_shot", dataset="rspect",
        cache_dir=cache_dir, labels_json=tmp_path / "labels.json",
        output_dir=tmp_path / "out",
    )
    assert detector.extract_done(exp, cfg) is False


def test_extract_done_true_when_no_cache_dir(detector, tmp_path):
    from experiments.config import EvalConfig
    exp = Experiment(name="x", evaluations=[])
    cfg = EvalConfig(
        wrapper="curia2", dataset="rspect",
        output_dir=str(tmp_path / "out"), protocol="cv_5fold",
    )
    assert detector.extract_done(exp, cfg) is True


def test_evaluate_done_false_when_no_summary(detector, tmp_path):
    exp = Experiment(name="x", evaluations=[])
    cfg = cv_5fold_eval(
        wrapper="x", dataset="y",
        cache_dir=tmp_path / "c", labels_json=tmp_path / "l.json",
        output_dir=tmp_path / "out",
    )
    assert detector.evaluate_done(exp, cfg) is False


def test_evaluate_done_true_when_valid_summary(detector, tmp_path):
    out = tmp_path / "out"
    _write_valid_summary(out)
    exp = Experiment(name="x", evaluations=[])
    cfg = cv_5fold_eval(
        wrapper="x", dataset="y",
        cache_dir=tmp_path / "c", labels_json=tmp_path / "l.json",
        output_dir=out,
    )
    assert detector.evaluate_done(exp, cfg) is True


def test_evaluate_done_false_when_summary_invalid(detector, tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "summary.json").write_text("{ not valid json ")
    exp = Experiment(name="x", evaluations=[])
    cfg = cv_5fold_eval(
        wrapper="x", dataset="y",
        cache_dir=tmp_path / "c", labels_json=tmp_path / "l.json",
        output_dir=out,
    )
    assert detector.evaluate_done(exp, cfg) is False


def test_aggregate_done_vacuously_true_with_no_evaluations(detector):
    exp = Experiment(name="empty", evaluations=[])
    assert detector.aggregate_done(exp) is True


def test_aggregate_done_false_when_cv_csv_missing(detector, tmp_path):
    exp = Experiment(
        name="x", study="s",
        evaluations=[
            cv_5fold_eval(
                wrapper="curia2", dataset="rspect",
                cache_dir=tmp_path / "c", labels_json=tmp_path / "l.json",
                output_dir=tmp_path / "cell",
            ),
        ],
    )
    assert detector.aggregate_done(exp) is False
