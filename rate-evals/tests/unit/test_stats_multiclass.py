"""Unit tests for the multi-class stats path."""

from __future__ import annotations

import numpy as np
import pytest

from rate_eval.evaluation import stats


def test_balanced_accuracy_perfect():
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = y_true.copy()
    assert stats._safe_balanced_accuracy(y_true, y_pred) == pytest.approx(1.0)


def test_balanced_accuracy_majority():
    """Always-predict-majority on a balanced 3-class set gets bacc = 1/3."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_pred = np.zeros_like(y_true)
    assert stats._safe_balanced_accuracy(y_true, y_pred) == pytest.approx(1.0 / 3.0)


def test_balanced_accuracy_imbalanced_majority():
    """Always-predict-majority on an imbalanced 3-class set still gets 1/3."""
    y_true = np.array([0] * 90 + [1] * 5 + [2] * 5)
    y_pred = np.zeros_like(y_true)
    assert stats._safe_balanced_accuracy(y_true, y_pred) == pytest.approx(1.0 / 3.0)


def test_multiclass_auroc_perfect():
    y_true = np.array([0, 0, 1, 1, 2, 2])
    y_score = np.eye(3)[y_true].astype(float)
    assert stats._safe_multiclass_auroc_ovr(y_true, y_score) == pytest.approx(1.0)


def test_multiclass_auroc_uniform():
    """Uniform predictions yield AUROC = 0.5 (random)."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
    y_score = np.ones((9, 3)) / 3.0
    auc = stats._safe_multiclass_auroc_ovr(y_true, y_score)
    assert auc == pytest.approx(0.5, abs=1e-9)


def test_multiclass_fold_metrics_shape_contract():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=120)
    logits = rng.normal(size=(120, 3))
    proba = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    m = stats.multiclass_fold_metrics(y_true, proba)
    assert "balanced_accuracy" in m
    assert "auroc_ovr" in m
    assert "macro_f1" in m
    assert "macro_precision" in m
    assert "macro_recall" in m
    assert "n_class_0" in m
    assert "n_class_1" in m
    assert "n_class_2" in m
    assert 0.0 <= m["balanced_accuracy"] <= 1.0
    assert 0.0 <= m["auroc_ovr"] <= 1.0


def test_bootstrap_ci_balanced_accuracy_contains_point():
    """Point estimate should fall within CI bounds (basic sanity)."""
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=300)
    y_pred = y_true.copy()
    # inject noise so the CI is non-degenerate
    flip_idx = rng.choice(300, size=30, replace=False)
    y_pred[flip_idx] = (y_pred[flip_idx] + 1) % 3
    pt, lo, hi = stats.bootstrap_ci_balanced_accuracy(y_true, y_pred, n_boot=500, seed=42)
    assert lo <= pt <= hi
    assert 0.85 < pt < 0.95  # ~10% noise -> ~90% bacc


def test_multiclass_aggregate_folds_shape():
    rng = np.random.default_rng(0)
    folds_y = []
    folds_s = []
    for _ in range(5):
        n = 100
        y = rng.integers(0, 3, size=n)
        logits = rng.normal(size=(n, 3))
        s = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
        folds_y.append(y); folds_s.append(s)
    cv = stats.multiclass_aggregate_folds(folds_y, folds_s, n_boot=100, seed=42)
    assert isinstance(cv.pooled_metrics, dict)
    assert "balanced_accuracy" in cv.pooled_metrics
    assert len(cv.per_fold_metrics) == 5
    lo, hi = cv.pooled_balanced_accuracy_ci
    assert not np.isnan(lo)
    assert not np.isnan(hi)


def test_binary_fold_metrics_unchanged():
    """Binary fold_metrics produces the expected values on a fixed input."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_score = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9])
    m = stats.fold_metrics(y_true, y_score)

    # perfect ranking of pos vs neg
    assert m["auroc"] == pytest.approx(1.0)
    # all positives at top of score distribution
    assert m["auprc"] == pytest.approx(1.0)
    # TP=3, TN=3 at threshold 0.5
    assert m["accuracy"] == pytest.approx(1.0)
    assert m["n_pos"] == 3
    assert m["n_neg"] == 3


def test_binary_bootstrap_ci_auroc_unchanged():
    """Bootstrap CI structure unchanged for the binary path."""
    y_true = np.array([0, 0, 0, 1, 1, 1] * 20)
    y_score = np.array([0.1, 0.2, 0.4, 0.6, 0.8, 0.9] * 20)
    pt, lo, hi = stats.bootstrap_ci_auroc(y_true, y_score, n_boot=200, seed=42)
    assert pt == pytest.approx(1.0)
    assert lo <= pt <= hi


def test_binary_imports_unchanged():
    """The binary public API surface must still expose its original symbols."""
    assert hasattr(stats, "fold_metrics")
    assert hasattr(stats, "aggregate_folds")
    assert hasattr(stats, "bootstrap_ci_auroc")
    assert hasattr(stats, "bootstrap_ci_auprc")
    assert hasattr(stats, "paired_bootstrap_diff")
    assert hasattr(stats, "holm_correct")
    assert hasattr(stats, "sensitivity_at_specificity")
    assert hasattr(stats, "plot_roc_with_bootstrap")
    assert hasattr(stats, "plot_pr_with_bootstrap")
    assert hasattr(stats, "plot_calibration")
    assert hasattr(stats, "bh_fdr_correct")
    assert hasattr(stats, "_cluster_resample_idx")


def test_percentile_ci_empty_returns_nan():
    """Regression: _percentile_ci on an empty (all-NaN filtered) array returns (nan, nan), not an IndexError."""
    lo, hi = stats._percentile_ci(np.array([], dtype=float))
    assert np.isnan(lo) and np.isnan(hi)


@pytest.mark.parametrize("n_pos", [0, 1])
def test_plot_bootstrap_degenerate_class_no_crash(n_pos):
    """Regression: ROC/PR plotters must not crash on a degenerate class."""
    import matplotlib
    matplotlib.use("Agg")

    y_true = np.zeros(50, dtype=int)
    if n_pos:
        y_true[:n_pos] = 1
    y_score = np.linspace(0.0, 1.0, 50)
    fig_roc = stats.plot_roc_with_bootstrap(y_true, y_score, n_boot=50, seed=42)
    fig_pr = stats.plot_pr_with_bootstrap(y_true, y_score, n_boot=50, seed=42)
    assert fig_roc is not None
    assert fig_pr is not None
