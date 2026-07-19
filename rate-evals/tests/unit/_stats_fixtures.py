"""Deterministic seeded numpy fixtures shared by the stats byte-stability + cluster-bootstrap tests."""

from __future__ import annotations

import numpy as np


def make_binary_fixture(n: int = 80, seed: int = 0):
    """A correlated binary (y_true, y_score) pair with both classes present."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    # score correlated with label -> AUROC above chance
    y_score = np.clip(0.30 * y_true + rng.normal(0.45, 0.20, size=n), 0.0, 1.0)
    return y_true.astype(np.int64), y_score.astype(float)


def make_multiclass_fixture(n: int = 90, n_classes: int = 3, seed: int = 1):
    """A 3-class (y_true, y_score softmax) pair, every class represented."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, n_classes, size=n)
    logits = rng.normal(0.0, 1.0, size=(n, n_classes))
    # nudge true-class logit up -> informative but imperfect
    logits[np.arange(n), y_true] += 1.1
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    y_score = e / e.sum(axis=1, keepdims=True)
    return y_true.astype(np.int64), y_score.astype(float)


def make_clustered_binary_fixture(
    n_patients: int = 30, rows_per_patient: int = 4, seed: int = 2,
):
    """Repeat-measures fixture with strong within-patient correlation; returns (y_true, y_score, groups)."""
    rng = np.random.default_rng(seed)
    y_true_list, y_score_list, groups = [], [], []
    for p in range(n_patients):
        patient_label = int(rng.integers(0, 2))
        latent = 0.30 * patient_label + rng.normal(0.45, 0.12)
        for _ in range(rows_per_patient):
            # tiny within-patient jitter -> high intra-cluster correlation
            y_true_list.append(patient_label)
            y_score_list.append(float(np.clip(latent + rng.normal(0.0, 0.02), 0.0, 1.0)))
            groups.append(f"P{p:03d}")
    return (
        np.asarray(y_true_list, dtype=np.int64),
        np.asarray(y_score_list, dtype=float),
        np.asarray(groups),
    )
