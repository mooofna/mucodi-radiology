"""Tests for the new cluster-aware F1 / Brier bootstrap CIs (rate_eval.evaluation.stats)."""
from __future__ import annotations

import numpy as np
import pytest

from rate_eval.evaluation.stats import (
    bootstrap_ci_auprc,
    bootstrap_ci_brier,
    bootstrap_ci_f1,
    fold_metrics,
)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    n_pat = 200
    y_pat = rng.integers(0, 2, size=n_pat)
    # signal-bearing per-patient scores so metrics are non-degenerate
    s_pat = np.clip(0.5 + 0.25 * (y_pat * 2 - 1) + rng.normal(0, 0.2, n_pat), 0, 1)
    # 2 identical volumes per patient -> max within-cluster correlation
    y = np.repeat(y_pat, 2)
    s = np.repeat(s_pat, 2)
    groups = np.repeat(np.arange(n_pat), 2)
    return y, s, groups


def test_f1_ci_point_matches_fold_metric(data):
    y, s, _ = data
    point, lo, hi = bootstrap_ci_f1(y, s)
    assert lo <= point <= hi
    assert abs(point - fold_metrics(y, s)["f1"]) < 1e-9  # same threshold=0.5 estimand
    assert 0.0 <= lo <= hi <= 1.0


def test_brier_ci_brackets_point(data):
    y, s, _ = data
    point, lo, hi = bootstrap_ci_brier(y, s)
    assert lo <= point <= hi
    assert abs(point - fold_metrics(y, s)["brier"]) < 1e-9


def test_cluster_ci_not_narrower_than_iid(data):
    y, s, g = data
    _, il, ih = bootstrap_ci_f1(y, s, groups=None)
    _, cl, ch = bootstrap_ci_f1(y, s, groups=g)
    assert (ch - cl) >= (ih - il) - 1e-6


def test_degenerate_single_class_returns_nan():
    y = np.zeros(20, dtype=int)
    s = np.linspace(0, 1, 20)
    for fn in (bootstrap_ci_f1, bootstrap_ci_brier, bootstrap_ci_auprc):
        point, lo, hi = fn(y, s)
        assert np.isnan(point) and np.isnan(lo) and np.isnan(hi)


def test_deterministic_seed(data):
    y, s, g = data
    assert bootstrap_ci_f1(y, s, groups=g, seed=42) == bootstrap_ci_f1(y, s, groups=g, seed=42)


def test_bca_path_runs(data):
    y, s, g = data
    point, lo, hi = bootstrap_ci_brier(y, s, groups=g, use_bca=True, n_boot=200)
    assert lo <= point <= hi
