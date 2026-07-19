"""Unit tests for rate_eval.io.result_schema.ResultSummary."""

from __future__ import annotations

import json
import math

import pytest

from rate_eval.io.result_schema import HeadSpec, MetricBlock, ResultSummary


def _make_cv_summary(**overrides):
    base = {
        "head_spec": {"kind": "linear", "dim_hidden": 256, "dropout": 0.5},
        "feature_dim": 768,
        "n_samples": 735,
        "cv_folds": 5,
        "cv_seed": 42,
        "pooled": {
            "auroc": 0.78,
            "auprc": 0.65,
            "brier": 0.18,
            "accuracy": 0.74,
            "f1": 0.62,
            "sens_at_95spec": 0.45,
            "n_pos": 200,
            "n_neg": 535,
        },
        "pooled_auroc_ci": [0.74, 0.82],
        "pooled_auprc_ci": [0.60, 0.70],
        "fold_summary": {"auroc": [0.78, 0.02], "auprc": [0.65, 0.03]},
        "per_fold": [{"fold": i, "auroc": 0.78 + 0.005 * i} for i in range(5)],
    }
    base.update(overrides)
    return base


def test_cv_summary_infers_protocol_when_missing():
    """Legacy CV summaries (no `protocol` key) get inferred as protocol=cv."""
    raw = _make_cv_summary()
    summary = ResultSummary.from_dict(raw)
    assert summary.protocol == "cv"
    assert summary.schema_version == "0.9"
    assert summary.pooled.auroc == 0.78
    assert summary.pooled_auroc_ci == [0.74, 0.82]


def test_explicit_protocol_passes_through():
    raw = _make_cv_summary(schema_version="1.0", protocol="cv")
    summary = ResultSummary.from_dict(raw)
    assert summary.schema_version == "1.0"
    assert summary.protocol == "cv"


def test_extra_fields_allowed():
    """`extra='allow'` accommodates wrapper-specific keys (model.config sha, etc.)."""
    raw = _make_cv_summary()
    raw["wrapper_repo_sha"] = "abc123"
    summary = ResultSummary.from_dict(raw)
    payload = summary.model_dump()
    assert payload["wrapper_repo_sha"] == "abc123"


def test_head_spec_zero_shot_fields():
    """zero_shot heads carry prompt fields; linear heads carry dim_hidden/dropout."""
    head = HeadSpec(kind="zero_shot", teacher="ctclip", prompt_pos="A", prompt_neg="B")
    assert head.kind == "zero_shot"
    assert head.dim_hidden is None


def test_roundtrip_json(tmp_path):
    """Write to disk and re-parse; numeric fields stable."""
    raw = _make_cv_summary()
    summary = ResultSummary.from_dict(raw)
    out_path = summary.write(tmp_path)
    reloaded = ResultSummary.from_json(out_path)
    assert reloaded.pooled.auroc == summary.pooled.auroc
    assert reloaded.pooled_auroc_ci == summary.pooled_auroc_ci
    assert reloaded.protocol == "cv"


def test_metric_block_handles_nan():
    """Brier/AUROC may be NaN for degenerate folds -- model parses without crashing."""
    block = MetricBlock(auroc=float("nan"), auprc=0.5)
    assert math.isnan(block.auroc)
    assert block.auprc == 0.5


def test_benchmark_provenance_fields_roundtrip(tmp_path):
    """The probe-provenance fields validate and survive a write->read cycle."""
    raw = _make_cv_summary(
        schema_version="1.0",
        protocol="cv",
        feature_l2_normalized=True,
        weight_decay_effective=1e-1 / 512,
        group_id="valid_1005",
    )
    summary = ResultSummary.from_dict(raw)
    out_path = summary.write(tmp_path)
    reloaded = ResultSummary.from_json(out_path)
    assert reloaded.feature_l2_normalized is True
    assert reloaded.weight_decay_effective == pytest.approx(1e-1 / 512)
    assert reloaded.group_id == "valid_1005"


def test_no_cv_keys_falls_back_to_single_split():
    raw = {
        "head_spec": {"kind": "zero_shot"},
        "feature_dim": 512,
        "auroc": 0.7,
    }
    summary = ResultSummary.from_dict(raw)
    assert summary.protocol == "single_split"
