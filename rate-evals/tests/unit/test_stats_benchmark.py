"""Unit tests for the four torch-free teacher-selection stats kernels in rate_eval.evaluation.stats."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from rate_eval.evaluation.stats import (
    bootstrap_ci_auroc,
    inverse_variance_aggregate,
    macro_auroc_cluster_ci,
    multiclass_paired_bootstrap_diff,
    paired_macro_auroc_cluster_diff,
)


def _binary_class(n_pat: int, vols_per_pat: int, seed: int, signal: float = 1.0):
    """One scored binary class with multi-volume patients."""
    rng = np.random.default_rng(seed)
    y, s, pids = [], [], []
    for i in range(n_pat):
        label = i % 2
        center = signal if label == 1 else -signal
        for _ in range(vols_per_pat):
            y.append(label)
            s.append(float(rng.standard_normal() * 0.5 + center))
            pids.append(f"P{i:03d}")
    return np.array(y), np.array(s, dtype=float), np.array(pids)


def test_single_class_macro_equals_binary_cluster_ci():
    """With one class the macro cluster CI reduces to bootstrap_ci_auroc(groups=pids)."""
    y, s, pids = _binary_class(n_pat=24, vols_per_pat=2, seed=1)
    m_point, m_lo, m_hi = macro_auroc_cluster_ci([(y, s, pids)], n_boot=300, seed=7, use_bca=True)
    b_point, b_lo, b_hi = bootstrap_ci_auroc(y, s, n_boot=300, seed=7, use_bca=True, groups=pids)
    assert m_point == pytest.approx(b_point, abs=1e-9)
    assert m_lo == pytest.approx(b_lo, abs=1e-6)
    assert m_hi == pytest.approx(b_hi, abs=1e-6)


def test_macro_point_is_mean_of_per_class_aurocs():
    """The macro point estimate is the plain mean of the per-class full-data AUROCs."""
    classes = [_binary_class(n_pat=20, vols_per_pat=1, seed=s) for s in (1, 2, 3)]
    per_class = [(y, sc, p) for (y, sc, p) in classes]
    point, _, _ = macro_auroc_cluster_ci(per_class, n_boot=100, seed=0)
    expected = np.mean([roc_auc_score(y, sc) for (y, sc, _p) in classes])
    assert point == pytest.approx(expected, abs=1e-9)


def test_macro_unique_patients_matches_iid_within_tolerance():
    """When every patient is unique (one row each), the cluster macro CI ~ the i.i.d. AUROC CI."""
    rng = np.random.default_rng(5)
    n = 300
    y = (np.arange(n) % 2)
    s = rng.standard_normal(n) * 0.5 + np.where(y == 1, 0.6, -0.6)
    pids = np.array([f"U{i}" for i in range(n)])
    m_point, m_lo, m_hi = macro_auroc_cluster_ci([(y, s, pids)], n_boot=400, seed=3)
    b_point, b_lo, b_hi = bootstrap_ci_auroc(y, s, n_boot=400, seed=3, groups=None)
    assert m_point == pytest.approx(b_point, abs=1e-9)
    assert m_lo == pytest.approx(b_lo, abs=0.03)
    assert m_hi == pytest.approx(b_hi, abs=0.03)


def test_macro_cluster_ci_wider_than_naive_unclustered():
    """Strong within-patient correlation -> clustered macro CI wider than treating each volume as its own patient."""
    # duplicate each patient's row many times (perfectly correlated siblings)
    base = _binary_class(n_pat=30, vols_per_pat=1, seed=2)
    y0, s0, _ = base
    reps = 6
    y = np.repeat(y0, reps)
    s = np.repeat(s0, reps)
    pid_clustered = np.repeat(np.array([f"P{i}" for i in range(y0.size)]), reps)
    pid_unique = np.array([f"R{i}" for i in range(y.size)])  # each row its own "patient"
    _, lo_c, hi_c = macro_auroc_cluster_ci([(y, s, pid_clustered)], n_boot=400, seed=9)
    _, lo_u, hi_u = macro_auroc_cluster_ci([(y, s, pid_unique)], n_boot=400, seed=9)
    assert (hi_c - lo_c) > (hi_u - lo_u)


def test_paired_macro_observed_matches_macro_difference():
    classes_a = [_binary_class(20, 2, seed=s, signal=1.2) for s in (1, 2)]
    # B = A with weaker signal (noisier scores), same patients/labels
    rng = np.random.default_rng(11)
    classes_b = [(y, sc + rng.standard_normal(sc.size) * 0.8, p) for (y, sc, p) in classes_a]
    per_a = [(y, sc, p) for (y, sc, p) in classes_a]
    per_b = [(y, sc, p) for (y, sc, p) in classes_b]
    observed, (lo, hi), p = paired_macro_auroc_cluster_diff(per_a, per_b, n_boot=300, seed=4)
    macro_a = np.mean([roc_auc_score(y, sc) for (y, sc, _p) in classes_a])
    macro_b = np.mean([roc_auc_score(y, sc) for (y, sc, _p) in classes_b])
    assert observed == pytest.approx(macro_a - macro_b, abs=1e-9)
    assert lo <= observed <= hi or np.isnan(lo)


def test_paired_macro_identical_teachers_zero_diff():
    per = [_binary_class(20, 2, seed=s) for s in (1, 2)]
    per = [(y, sc, p) for (y, sc, p) in per]
    observed, (lo, hi), p = paired_macro_auroc_cluster_diff(per, per, n_boot=200, seed=0)
    assert observed == pytest.approx(0.0, abs=1e-12)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(0.0, abs=1e-12)
    assert p == pytest.approx(1.0 / 200)  # symmetric p floored at 1/n_boot


def _softmax_fixture(n: int, n_classes: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, size=n)
    logits = rng.standard_normal((n, n_classes))
    logits[np.arange(n), y] += 1.5
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return y, e / e.sum(axis=1, keepdims=True)


def test_multiclass_paired_observed_and_degenerate():
    y, sa = _softmax_fixture(150, 3, seed=1)
    _, sb = _softmax_fixture(150, 3, seed=2)
    from rate_eval.evaluation.stats import _safe_multiclass_auroc_ovr
    observed, (lo, hi), p = multiclass_paired_bootstrap_diff(y, sa, sb, n_boot=300, seed=3)
    expected = _safe_multiclass_auroc_ovr(y, sa) - _safe_multiclass_auroc_ovr(y, sb)
    assert observed == pytest.approx(expected, abs=1e-9)
    # identical softmax -> zero diff, p floored
    obs0, (lo0, hi0), p0 = multiclass_paired_bootstrap_diff(y, sa, sa, n_boot=200, seed=3)
    assert obs0 == pytest.approx(0.0, abs=1e-12)
    assert p0 == pytest.approx(1.0 / 200)


def test_multiclass_paired_groups_none_is_deterministic():
    y, sa = _softmax_fixture(120, 3, seed=4)
    _, sb = _softmax_fixture(120, 3, seed=5)
    r1 = multiclass_paired_bootstrap_diff(y, sa, sb, n_boot=200, seed=42)
    r2 = multiclass_paired_bootstrap_diff(y, sa, sb, n_boot=200, seed=42)
    assert r1[0] == r2[0] and r1[1] == r2[1] and r1[2] == r2[2]


def test_inverse_variance_weights_narrow_ci_more():
    # narrow CI (large n) at 0.70 dominates a wide CI (small n) at 0.90
    points = [0.70, 0.90]
    cis = [(0.69, 0.71), (0.80, 1.00)]
    mean, lo, hi = inverse_variance_aggregate(points, cis)
    assert 0.70 <= mean < 0.74
    assert hi - lo < (0.71 - 0.69)
    assert lo < mean < hi


def test_inverse_variance_drops_degenerate_cells():
    points = [0.70, 0.80, float("nan")]
    cis = [(0.69, 0.71), (0.50, 0.50), (0.6, 0.7)]  # zero-width + nan-point dropped
    mean, lo, hi = inverse_variance_aggregate(points, cis)
    # only the first cell survives -> mean == its point
    assert mean == pytest.approx(0.70, abs=1e-9)


def test_inverse_variance_empty_returns_nan():
    mean, lo, hi = inverse_variance_aggregate([], [])
    assert np.isnan(mean) and np.isnan(lo) and np.isnan(hi)
