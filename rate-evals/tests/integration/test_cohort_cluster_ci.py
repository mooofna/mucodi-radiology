"""Integration test: --cohort clusters sibling volumes into one fold and adds a patient-cluster CI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rate_eval.cli.evaluate import _build_parser

QA_KEY = "Does this CT scan show an abnormality?"


def _write_cache(cache_dir: Path, n_patients: int = 30, dim: int = 8, seed: int = 0) -> int:
    """Write a synthetic CT-RATE-shaped cache + labels JSON. Returns the row count."""
    rng = np.random.default_rng(seed)
    emb_dir = cache_dir / "embeddings" / "test"
    emb_dir.mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict] = {}
    n_rows = 0
    for i in range(n_patients):
        y = i % 2
        # class-separable features so the head trains and AUROC is well-defined
        center = 1.0 if y == 1 else -1.0
        for study in ("a", "b"):
            acc = f"valid_{1000 + i}_{study}_1"
            vec = (rng.standard_normal(dim).astype(np.float32) * 0.3) + center
            np.savez(emb_dir / f"{acc}.npz", embedding=vec)
            labels[acc] = {
                "split": "test",
                "patient_id": 1000 + i,
                "qa_results": {"default_qa": [{QA_KEY: int(y)}]},
            }
            n_rows += 1
    (cache_dir / "labels.json").write_text(json.dumps(labels, indent=2))
    return n_rows


def _run(cache_dir: Path, out_dir: Path, cohort: str | None, extra: list[str] | None = None) -> None:
    argv = [
        "cv",
        "--checkpoint-dir", str(cache_dir),
        "--labels-json", str(cache_dir / "labels.json"),
        "--qa-key", QA_KEY,
        "--output-dir", str(out_dir),
        "--cv-folds", "5",
        "--max-epochs", "3",
        "--patience", "2",
        "--n-boot", "200",
        "--device", "cpu",
        "--log-level", "WARNING",
    ]
    if cohort:
        argv += ["--cohort", cohort]
    if extra:
        argv += extra
    args = _build_parser().parse_args(argv)
    args.func(args)


def test_cohort_emits_cluster_ci_and_clusters_siblings(tmp_path):
    cache_dir = tmp_path / "cache"
    n_rows = _write_cache(cache_dir, n_patients=30)
    out_dir = tmp_path / "out_cohort"
    _run(cache_dir, out_dir, cohort="ct_rate")

    summary = json.loads((out_dir / "summary.json").read_text())
    assert "pooled_auroc_ci_cluster" in summary
    assert len(summary["pooled_auroc_ci_cluster"]) == 2
    assert summary["ci_resample_unit"] == "patient_cluster"
    # 60 sibling rows collapse to 30 patients
    assert summary["n_patients"] == 30
    assert summary["n_patients"] < n_rows
    assert "pooled_auroc_ci" in summary and len(summary["pooled_auroc_ci"]) == 2

    # each patient's _a/_b siblings share one fold
    rows = json.loads((out_dir / "predictions_oof.json").read_text())
    fold_by_patient: dict[str, set[int]] = {}
    for r in rows:
        fold_by_patient.setdefault(r["patient_id"], set()).add(r["fold"])
    assert all(len(folds) == 1 for folds in fold_by_patient.values())
    assert len(fold_by_patient) == 30


def test_no_cohort_is_byte_identical_legacy_shape(tmp_path):
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, n_patients=30)
    out_dir = tmp_path / "out_plain"
    _run(cache_dir, out_dir, cohort=None)

    summary = json.loads((out_dir / "summary.json").read_text())
    assert "pooled_auroc_ci_cluster" not in summary
    assert "ci_resample_unit" not in summary
    assert "n_patients" not in summary
    # without --cohort each accession is its own patient -> 60 ids
    rows = json.loads((out_dir / "predictions_oof.json").read_text())
    assert len({r["patient_id"] for r in rows}) == 60


def test_l2_normalize_and_grid_record_provenance(tmp_path):
    cache_dir = tmp_path / "cache"
    _write_cache(cache_dir, n_patients=30)
    out_dir = tmp_path / "out_probe"
    # small cohort (<2000) -> grid falls back to fixed 1/d_t
    _run(cache_dir, out_dir, cohort="ct_rate", extra=["--l2-normalize", "--l2-grid"])
    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["feature_l2_normalized"] is True
    # fallback wd = base_wd(0.01) * 1/d_t(8) = 0.00125
    assert summary["weight_decay_effective"] == pytest.approx(0.01 / 8)
    assert len(summary["weight_decay_per_fold"]) == 5
