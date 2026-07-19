"""Integration test for `rate-aggregate cv-summaries`."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from rate_eval.cli.aggregate import CV_SUMMARY_FIELDS, collect_cv_summaries


def _write_summary(root: Path, teacher: str, auroc: float, auprc: float):
    out_dir = root / f"{teacher}_lidc"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "0.9",
        "protocol": "cv",
        "head_spec": {"kind": "linear", "dim_hidden": 256},
        "feature_dim": 768,
        "n_samples": 735,
        "cv_folds": 5,
        "cv_seed": 42,
        "pooled": {
            "auroc": auroc,
            "auprc": auprc,
            "brier": 0.18,
            "accuracy": 0.75,
            "f1": 0.62,
            "sens_at_95spec": 0.45,
        },
        "pooled_auroc_ci": [auroc - 0.03, auroc + 0.03],
        "pooled_auprc_ci": [auprc - 0.04, auprc + 0.04],
        "fold_summary": {"auroc": [auroc, 0.02]},
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))


def test_collect_cv_summaries_smoke(tmp_path):
    _write_summary(tmp_path, "ctclip_zero_shot", auroc=0.78, auprc=0.65)
    _write_summary(tmp_path, "tangerine_vit", auroc=0.82, auprc=0.70)

    out_csv = tmp_path / "out.csv"
    rows = collect_cv_summaries(
        results_root=tmp_path,
        teachers=["ctclip_zero_shot", "tangerine_vit"],
        suffix="_lidc",
        out_path=out_csv,
    )
    assert len(rows) == 2

    with out_csv.open() as f:
        reader = csv.DictReader(f)
        loaded = list(reader)
    assert reader.fieldnames == CV_SUMMARY_FIELDS

    assert loaded[0]["teacher"] == "ctclip_zero_shot"
    assert float(loaded[0]["auroc"]) == 0.78
    assert float(loaded[0]["auroc_lo"]) == 0.75
    assert float(loaded[0]["auroc_hi"]) == 0.81
    assert loaded[1]["teacher"] == "tangerine_vit"


def test_collect_cv_summaries_missing_teacher(tmp_path):
    """Teacher with no summary.json yields a `_status: missing` row, not an error."""
    _write_summary(tmp_path, "tangerine_vit", auroc=0.82, auprc=0.70)
    rows = collect_cv_summaries(
        results_root=tmp_path,
        teachers=["ctclip_zero_shot", "tangerine_vit"],
        suffix="_lidc",
    )
    assert len(rows) == 2
    assert rows[0]["teacher"] == "ctclip_zero_shot"
    assert rows[0]["_status"] == "missing"
    assert rows[0]["auroc"] is None
    assert rows[1]["teacher"] == "tangerine_vit"
    assert "_status" not in rows[1]
