"""Patient-cluster bootstrap + byte-stability tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rate_eval.evaluation import stats
from rate_eval.io.feature_loaders import resolve_patient_key_fn

import sys

sys.path.insert(0, str(Path(__file__).parent))
from _stats_fixtures import (  # noqa: E402
    make_binary_fixture,
    make_multiclass_fixture,
    make_clustered_binary_fixture,
)

_GOLDEN = json.loads((Path(__file__).parent / "_golden_stats.json").read_text())


def _close(a, b):
    return np.allclose(np.atleast_1d(a), np.atleast_1d(b), rtol=1e-14, atol=1e-15)


def test_groups_none_is_byte_identical_to_legacy():
    yb, sb = make_binary_fixture()
    ym, sm = make_multiclass_fixture()
    rng = np.random.default_rng(7)
    sb2 = np.clip(0.20 * yb + rng.normal(0.45, 0.25, size=yb.size), 0, 1).astype(float)

    assert _close(stats.bootstrap_ci_auroc(yb, sb, seed=42, use_bca=True), _GOLDEN["auroc_bca"])
    assert _close(stats.bootstrap_ci_auroc(yb, sb, seed=42, use_bca=False), _GOLDEN["auroc_pct"])
    assert _close(stats.bootstrap_ci_auprc(yb, sb, seed=42), _GOLDEN["auprc"])
    assert _close(stats.bootstrap_ci_balanced_accuracy(ym, sm.argmax(1), seed=42), _GOLDEN["bacc"])
    assert _close(stats.bootstrap_ci_multiclass_auroc(ym, sm, seed=42), _GOLDEN["mc_auroc"])
    obs, (lo, hi), p = stats.paired_bootstrap_diff(yb, sb, sb2, seed=42)
    assert _close([obs, lo, hi, p], _GOLDEN["paired"])

    half = yb.size // 2
    af = stats.aggregate_folds([yb[:half], yb[half:]], [sb[:half], sb[half:]], seed=42)
    assert _close(af.pooled_auroc_ci, _GOLDEN["aggregate_folds_auroc_ci"])
    assert _close(af.pooled_auprc_ci, _GOLDEN["aggregate_folds_auprc_ci"])

    hm = ym.size // 2
    mcaf = stats.multiclass_aggregate_folds([ym[:hm], ym[hm:]], [sm[:hm], sm[hm:]], seed=42)
    assert _close(mcaf.pooled_balanced_accuracy_ci, _GOLDEN["mc_aggregate_bacc_ci"])
    assert _close(mcaf.pooled_auroc_ovr_ci, _GOLDEN["mc_aggregate_auroc_ci"])


def test_groups_none_kwarg_matches_positional():
    """Passing `groups=None` explicitly must equal the default-arg call exactly."""
    yb, sb = make_binary_fixture()
    assert _close(
        stats.bootstrap_ci_auroc(yb, sb, seed=42),
        stats.bootstrap_ci_auroc(yb, sb, seed=42, groups=None),
    )


def test_cluster_ci_wider_when_clustering_strong():
    yc, sc, gc = make_clustered_binary_fixture()
    _, rlo, rhi = stats.bootstrap_ci_auroc(yc, sc, seed=42)            # row
    _, clo, chi = stats.bootstrap_ci_auroc(yc, sc, seed=42, groups=gc)  # cluster
    assert (chi - clo) > (rhi - rlo)
    # AUPRC path (percentile, no BCa)
    _, rlo2, rhi2 = stats.bootstrap_ci_auprc(yc, sc, seed=42)
    _, clo2, chi2 = stats.bootstrap_ci_auprc(yc, sc, seed=42, groups=gc)
    assert (chi2 - clo2) > (rhi2 - rlo2)


def test_cluster_point_estimate_unchanged():
    """Clustering only changes the CI -- the point estimate is identical."""
    yc, sc, gc = make_clustered_binary_fixture()
    p_row, _, _ = stats.bootstrap_ci_auroc(yc, sc, seed=42)
    p_clu, _, _ = stats.bootstrap_ci_auroc(yc, sc, seed=42, groups=gc)
    assert p_row == p_clu


def test_all_unique_groups_match_row_within_tolerance():
    yb, sb = make_binary_fixture(n=120)
    groups = np.array([f"P{i}" for i in range(yb.size)])  # one row per patient
    assert np.unique(groups).size == yb.size
    _, rlo, rhi = stats.bootstrap_ci_auroc(yb, sb, seed=42)
    _, clo, chi = stats.bootstrap_ci_auroc(yb, sb, seed=42, groups=groups)
    # no clustering -> widths close
    assert abs((chi - clo) - (rhi - rlo)) < 0.03


def test_patient_key_registry():
    ctrate = resolve_patient_key_fn("ct_rate")
    assert ctrate("valid_1005_a_2") == "valid_1005"
    assert ctrate("valid_137_b_1") == "valid_137"
    rspect = resolve_patient_key_fn("rspect")
    assert rspect("f734af05be__f734af05be207_1") == "f734af05be"
    # unknown/None cohort -> legacy __-split fallback
    fallback = resolve_patient_key_fn(None)
    assert fallback("series__file") == "series"
    assert fallback("LIDC-IDRI-0001") == "LIDC-IDRI-0001"


def test_bca_stable_with_heterogeneous_cluster_sizes():
    rng = np.random.default_rng(3)
    y, s, g = [], [], []
    for p in range(40):
        size = int(rng.integers(1, 6))  # 1..5 rows per patient
        lab = int(rng.integers(0, 2))
        for _ in range(size):
            y.append(lab)
            s.append(float(np.clip(0.3 * lab + rng.normal(0.45, 0.15), 0, 1)))
            g.append(f"P{p}")
    y = np.array(y); s = np.array(s); g = np.array(g)
    pt, lo, hi = stats.bootstrap_ci_auroc(y, s, seed=42, groups=g, use_bca=True)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert 0.0 <= lo <= pt <= hi <= 1.0


def test_nan_drop_warning_threshold(caplog):
    import logging

    with caplog.at_level(logging.WARNING, logger="rate_eval.evaluation.stats"):
        stats._warn_nan_drop(500, 1000, "balanced_accuracy")  # 50% < 70% -> warn
    assert any("reduced" in r.message for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rate_eval.evaluation.stats"):
        stats._warn_nan_drop(900, 1000, "balanced_accuracy")  # 90% -> no warn
    assert not any("reduced" in r.message for r in caplog.records)


def test_bh_fdr_correct():
    out = stats.bh_fdr_correct({"a": 0.01, "b": 0.04, "c": 0.5})
    assert _close(out["a"][0], 0.03) and out["a"][1] is True
    assert _close(out["b"][0], 0.06) and out["b"][1] is False
    assert _close(out["c"][0], 0.5) and out["c"][1] is False
