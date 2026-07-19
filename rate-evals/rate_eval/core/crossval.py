"""Subject-level stratified k-fold split logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class CVFold:
    """One CV fold: indices into the original (per-sample) input arrays."""
    fold_idx: int
    train_indices: np.ndarray
    test_indices: np.ndarray
    train_patients: List[str] = field(default_factory=list)
    test_patients: List[str] = field(default_factory=list)


def make_subject_stratified_folds(
    patient_ids: Sequence[str],
    labels: Sequence[int],
    n_splits: int = 5,
    seed: int = 42,
) -> List[CVFold]:
    """Subject-level stratified k-fold over patient_ids (one entry per sample)."""
    patient_ids = np.asarray(patient_ids)
    labels = np.asarray(labels)
    if patient_ids.shape[0] != labels.shape[0]:
        raise ValueError(
            f"patient_ids ({patient_ids.shape[0]}) and labels ({labels.shape[0]}) must have the same length",
        )

    # per-patient stratification key: max-aggregate (any positive -> positive)
    unique_patients = np.unique(patient_ids)
    patient_labels = np.array([
        int(labels[patient_ids == pid].max()) for pid in unique_patients
    ], dtype=labels.dtype)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds: List[CVFold] = []
    for fold_idx, (train_pat_idx, test_pat_idx) in enumerate(skf.split(unique_patients, patient_labels)):
        train_patients = unique_patients[train_pat_idx]
        test_patients = unique_patients[test_pat_idx]
        # expand patient split to per-sample indices
        train_mask = np.isin(patient_ids, train_patients)
        test_mask = np.isin(patient_ids, test_patients)
        assert (train_mask & test_mask).sum() == 0, "fold leak: sample appears in both train and test"
        assert (train_mask | test_mask).sum() == patient_ids.shape[0], "fold gap: some samples unassigned"
        folds.append(
            CVFold(
                fold_idx=fold_idx,
                train_indices=np.flatnonzero(train_mask),
                test_indices=np.flatnonzero(test_mask),
                train_patients=train_patients.tolist(),
                test_patients=test_patients.tolist(),
            ),
        )
    return folds


def carve_validation_split(
    train_indices: np.ndarray,
    patient_ids: Sequence[str],
    labels: Sequence[int],
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Carve a subject-level stratified validation slice out of a fold's train indices."""
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    patient_ids = np.asarray(patient_ids)
    labels = np.asarray(labels)
    train_pids = patient_ids[train_indices]
    train_labels = labels[train_indices]

    # per-patient stratification label: max-aggregate (matches outer split)
    unique_patients = np.unique(train_pids)
    patient_labels = np.array(
        [int(train_labels[train_pids == pid].max()) for pid in unique_patients],
        dtype=train_labels.dtype,
    )

    n_val_folds = max(2, int(round(1.0 / val_fraction)))
    skf = StratifiedKFold(n_splits=n_val_folds, shuffle=True, random_state=seed)
    inner_train_pat_idx, val_pat_idx = next(skf.split(unique_patients, patient_labels))

    val_patients = unique_patients[val_pat_idx]
    val_mask = np.isin(train_pids, val_patients)
    inner_train_mask = ~val_mask
    return train_indices[inner_train_mask], train_indices[val_mask]


def write_splits_json(folds: Sequence[CVFold], out_path: Path) -> None:
    """Persist fold->patient assignments for reproducibility / leak audits."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "fold": f.fold_idx,
            "train_patients": list(f.train_patients),
            "test_patients": list(f.test_patients),
            "n_train_samples": int(f.train_indices.size),
            "n_test_samples": int(f.test_indices.size),
        }
        for f in folds
    ]
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=False))


def assert_no_patient_leak(folds: Sequence[CVFold]) -> None:
    """Validate fold integrity at the patient (cluster) level."""
    test_patients_seen: set[str] = set()
    for f in folds:
        # check first so a duplicate surfaces the multiple-test-sets error
        for pid in f.test_patients:
            if pid in test_patients_seen:
                raise AssertionError(f"patient {pid!r} appears in multiple fold test sets")
            test_patients_seen.add(pid)
        overlap = set(f.train_patients) & set(f.test_patients)
        if overlap:
            raise AssertionError(
                f"fold {f.fold_idx}: {len(overlap)} patient(s) in both train and test "
                f"(e.g. {sorted(overlap)[:3]})"
            )
