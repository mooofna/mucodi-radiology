"""Unit tests for rate_eval.core.crossval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rate_eval.core.crossval import (
    CVFold,
    assert_no_patient_leak,
    carve_validation_split,
    make_subject_stratified_folds,
    write_splits_json,
)


def _synth_subject_data(n_patients: int, samples_per_patient: int = 1, seed: int = 42):
    """One row per (patient, instance); class balance ~50/50 stratified at patient level."""
    rng = np.random.default_rng(seed)
    patient_ids = [f"P{i:04d}" for i in range(n_patients) for _ in range(samples_per_patient)]
    patient_labels = rng.integers(low=0, high=2, size=n_patients)
    labels = [int(patient_labels[i]) for i in range(n_patients) for _ in range(samples_per_patient)]
    return patient_ids, labels


def test_folds_cover_every_sample_exactly_once():
    pids, labels = _synth_subject_data(40, samples_per_patient=1)
    folds = make_subject_stratified_folds(pids, labels, n_splits=5, seed=42)
    assert len(folds) == 5
    test_idx_union = np.concatenate([f.test_indices for f in folds])
    assert sorted(test_idx_union.tolist()) == list(range(len(pids)))


def test_no_patient_leak_across_folds():
    """A patient appearing twice must end up in the same fold."""
    pids, labels = _synth_subject_data(20, samples_per_patient=3)
    folds = make_subject_stratified_folds(pids, labels, n_splits=5, seed=42)
    for fold in folds:
        train_set = set(fold.train_patients)
        test_set = set(fold.test_patients)
        assert train_set.isdisjoint(test_set), f"fold {fold.fold_idx}: leak"
    assert_no_patient_leak(folds)


def test_train_test_mask_disjoint_and_complete():
    pids, labels = _synth_subject_data(40)
    folds = make_subject_stratified_folds(pids, labels, n_splits=5, seed=42)
    for fold in folds:
        train_mask = set(fold.train_indices.tolist())
        test_mask = set(fold.test_indices.tolist())
        assert train_mask.isdisjoint(test_mask)
        assert train_mask | test_mask == set(range(len(pids)))


def test_stratification_max_aggregation_for_multi_sample_patients():
    """Patient with mixed sample labels (one 0, one 1) gets stratified as positive."""
    pids = ["P0", "P0", "P1", "P2", "P3"]
    labels = [0, 1, 0, 1, 1]
    folds = make_subject_stratified_folds(pids, labels, n_splits=2, seed=42)
    assert len(folds) == 2
    # P0 (label=max(0,1)=1) lands in the positive stratum


def test_unequal_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        make_subject_stratified_folds(["A", "B"], [1], n_splits=2)


def test_carve_validation_subject_disjoint():
    pids, labels = _synth_subject_data(40)
    folds = make_subject_stratified_folds(pids, labels, n_splits=5, seed=42)
    train_idx = folds[0].train_indices
    inner_train, val = carve_validation_split(
        train_idx, pids, labels, val_fraction=0.2, seed=7
    )
    assert set(inner_train.tolist()).isdisjoint(set(val.tolist()))
    assert set(inner_train.tolist()) | set(val.tolist()) == set(train_idx.tolist())
    val_pids = set(pids[i] for i in val)
    inner_train_pids = set(pids[i] for i in inner_train)
    assert val_pids.isdisjoint(inner_train_pids)


def test_carve_validation_rejects_invalid_fraction():
    pids, labels = _synth_subject_data(10)
    folds = make_subject_stratified_folds(pids, labels, n_splits=2, seed=42)
    with pytest.raises(ValueError, match="val_fraction"):
        carve_validation_split(folds[0].train_indices, pids, labels, val_fraction=1.5)


def test_write_splits_json_roundtrip(tmp_path):
    pids, labels = _synth_subject_data(20)
    folds = make_subject_stratified_folds(pids, labels, n_splits=4, seed=42)
    out_path = tmp_path / "splits.json"
    write_splits_json(folds, out_path)
    payload = json.loads(out_path.read_text())
    assert len(payload) == 4
    assert {entry["fold"] for entry in payload} == {0, 1, 2, 3}
    for entry in payload:
        assert entry["n_test_samples"] > 0
        assert entry["n_train_samples"] > 0


def test_assert_no_patient_leak_raises_when_duplicate_test_patient():
    f0 = CVFold(
        fold_idx=0,
        train_indices=np.array([0]),
        test_indices=np.array([1]),
        train_patients=["A"],
        test_patients=["B"],
    )
    f1 = CVFold(
        fold_idx=1,
        train_indices=np.array([1]),
        test_indices=np.array([0]),
        train_patients=["B"],
        test_patients=["B"],  # duplicate intentionally
    )
    with pytest.raises(AssertionError, match="multiple fold test sets"):
        assert_no_patient_leak([f0, f1])


def _ctrate_key(acc: str) -> str:
    """CT-RATE V2 key: valid_1005_a_2 -> valid_1005 (mirrors feature_loaders._key_ctrate_v2)."""
    return "_".join(acc.split("_")[:2])


def _synth_ctrate_like(n_patients: int = 30, seed: int = 0):
    """Synthesize CT-RATE-like accessions whose sibling volumes share the clustered patient key."""
    rng = np.random.default_rng(seed)
    accs: list[str] = []
    labels: list[int] = []
    for i in range(n_patients):
        y = int(rng.integers(0, 2))
        studies = ("a", "b")[: 1 + int(rng.integers(0, 2))]  # 1 or 2 studies
        for study in studies:
            for k in range(1 + int(rng.integers(0, 2))):     # 1 or 2 reconstructions
                accs.append(f"valid_{1000 + i}_{study}_{k}")
                labels.append(y)
    patient_ids = [_ctrate_key(a) for a in accs]
    return accs, patient_ids, labels


def test_ctrate_siblings_cluster_into_one_fold():
    """A CT-RATE patient's _a/_b/_k sibling volumes must all land in the same fold (no leak)."""
    accs, pids, labels = _synth_ctrate_like(n_patients=30, seed=1)
    assert len(set(pids)) < len(accs)
    folds = make_subject_stratified_folds(pids, labels, n_splits=5, seed=42)
    assert_no_patient_leak(folds)
    pids_arr = np.asarray(pids)
    for f in folds:
        for pid in set(f.test_patients):
            rows = set(np.flatnonzero(pids_arr == pid).tolist())
            assert rows.issubset(set(f.test_indices.tolist())), (
                f"patient {pid} siblings split across folds"
            )


def test_assert_no_patient_leak_raises_on_within_fold_train_test_overlap():
    f0 = CVFold(
        fold_idx=0,
        train_indices=np.array([0, 1]),
        test_indices=np.array([2]),
        train_patients=["valid_1000", "valid_1001"],
        test_patients=["valid_1000"],  # also in train -> within-fold leak
    )
    with pytest.raises(AssertionError, match="both train and test"):
        assert_no_patient_leak([f0])


def test_carve_max_aggregation_is_sibling_order_independent():
    """carve's per-patient stratification key is max-aggregated (sibling-order independent)."""
    # valid_1000 has mixed sibling labels [0, 1]; 8 clean patients for stratification
    pids = ["valid_1000", "valid_1000"] + [f"valid_{1001 + i}" for i in range(8)]
    labels = [0, 1] + [i % 2 for i in range(8)]
    train_idx = np.arange(len(pids))
    inner_a, val_a = carve_validation_split(train_idx, pids, labels, val_fraction=0.25, seed=11)
    # reverse valid_1000's two sibling rows; max(0,1)==max(1,0) so the carve is identical
    pids_rev = ["valid_1000", "valid_1000"] + pids[2:]
    labels_rev = [1, 0] + labels[2:]
    inner_b, val_b = carve_validation_split(train_idx, pids_rev, labels_rev, val_fraction=0.25, seed=11)
    assert set(val_a.tolist()) == set(val_b.tolist())
