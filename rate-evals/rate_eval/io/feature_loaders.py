"""Cache walkers + label-to-feature joiners (pure numpy/json, no torch at import time)."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger(__name__)


def _maybe_warn_cache_drift(
    cache_dir: Path, dataset_config_path: Optional[Path]
) -> None:
    """Warn if `cache_meta.yaml` records a different dataset-config sha than the current one."""
    if dataset_config_path is None:
        return
    if os.environ.get("RATE_SKIP_CACHE_META_CHECK", "0") == "1":
        return
    try:
        from .cache_meta import check_dataset_config_drift
    except Exception:  # noqa: BLE001 -- never block loading
        return
    msg = check_dataset_config_drift(cache_dir, dataset_config_path)
    if msg:
        _logger.warning(msg)


_DEFAULT_QA_KEY = "Does this scan contain a malignant nodule (median radiologist rating > 3)?"

_PATIENT_ID_SEPARATOR = "__"


def _patient_id_from_accession(accession: str) -> str:
    """Default grouping key: prefix before the first `__` (series level), else the full accession."""
    return accession.split(_PATIENT_ID_SEPARATOR, 1)[0] if _PATIENT_ID_SEPARATOR in accession else accession


# per-cohort patient-grouping registry (one group id per row for the bootstrap)


def _key_ctrate_v2(acc: str) -> str:
    """CT-RATE: `valid_1005_a_2` -> `valid_1005` (a/b = same patient, must cluster)."""
    return "_".join(acc.split("_")[:2])


def _key_split_dunder(acc: str) -> str:
    """RSPECT StudyInstanceUID / MIDRC: prefix before the first `__`."""
    return acc.split(_PATIENT_ID_SEPARATOR, 1)[0]


def _key_identity(acc: str) -> str:
    """One scan = one patient (RAD-ChestCT, LIDC, COCA, MosMed, STOIC, DLCS24, ...)."""
    return acc


# Keyed by cohort slug (lowercased, matched on substring of the passed cohort name).
_PATIENT_KEY_FNS: Dict[str, Callable[[str], str]] = {
    "ct_rate": _key_ctrate_v2,
    "ctrate": _key_ctrate_v2,
    "rspect": _key_split_dunder,
    "midrc": _key_split_dunder,
}


def resolve_patient_key_fn(cohort: Optional[str]) -> Callable[[str], str]:
    """Return the patient-grouping fn for a cohort (substring match); default is the `__`-split."""
    if cohort:
        c = cohort.lower()
        for slug, fn in _PATIENT_KEY_FNS.items():
            if slug in c:
                return fn
    return _patient_id_from_accession


def _iter_embeddings(cache_dir: Path) -> Tuple[List[str], np.ndarray]:
    """Walk `embeddings/{train,valid,test}/*.npz` -> (accessions, features); falls back to all of `embeddings/`."""
    embeddings_dir = cache_dir / "embeddings"
    if not embeddings_dir.is_dir():
        raise FileNotFoundError(
            f"cache directory has no embeddings/ subdir: {embeddings_dir}",
        )

    candidate_dirs = [embeddings_dir / s for s in ("train", "valid", "dev", "test")]
    candidate_dirs = [d for d in candidate_dirs if d.is_dir()]
    if not candidate_dirs:
        candidate_dirs = [embeddings_dir]

    accessions: List[str] = []
    arrays: List[np.ndarray] = []
    for d in candidate_dirs:
        for npz_path in sorted(d.glob("*.npz")):
            try:
                payload = np.load(npz_path)
                arr = payload["embedding"]
            except KeyError:
                continue  # legacy files under a different key
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim != 1:
                raise ValueError(
                    f"unexpected embedding shape {arr.shape} in {npz_path}",
                )
            accessions.append(npz_path.stem)
            arrays.append(arr.astype(np.float32))

    if not arrays:
        raise FileNotFoundError(f"no .npz embeddings found under {embeddings_dir}")

    feats = np.stack(arrays, axis=0)
    return accessions, feats


def _load_lidc_labels(
    labels_json_path: Path,
    qa_key: str = _DEFAULT_QA_KEY,
) -> Dict[str, int]:
    """Read the LIDC malignancy labels JSON and return `{accession: label}` (int 0/1)."""
    raw = json.loads(Path(labels_json_path).read_text())
    out: Dict[str, int] = {}
    for accession, entry in raw.items():
        qa_results = entry.get("qa_results", {})  # {"default_qa": [{"<question>": 0|1}, ...]}
        for _qa_set, qa_list in qa_results.items():
            if not isinstance(qa_list, list):
                continue
            for qa_pair in qa_list:
                if not isinstance(qa_pair, dict):
                    continue
                if qa_key in qa_pair:
                    out[accession] = int(qa_pair[qa_key])
                    break
            if accession in out:
                break
    return out


def _iter_embeddings_per_split(
    cache_dir: Path,
) -> Dict[str, Tuple[List[str], np.ndarray]]:
    """Per-split `{split: (accessions, features)}`, preserving the rate-extract split boundary."""
    embeddings_dir = cache_dir / "embeddings"
    if not embeddings_dir.is_dir():
        raise FileNotFoundError(
            f"cache directory has no embeddings/ subdir: {embeddings_dir}",
        )

    out: Dict[str, Tuple[List[str], np.ndarray]] = {}
    for split in ("train", "valid", "test", "dev"):
        d = embeddings_dir / split
        if not d.is_dir():
            continue
        accessions: List[str] = []
        arrays: List[np.ndarray] = []
        for npz_path in sorted(d.glob("*.npz")):
            try:
                payload = np.load(npz_path)
                arr = payload["embedding"]
            except KeyError:
                continue
            if arr.ndim == 2 and arr.shape[0] == 1:
                arr = arr[0]
            elif arr.ndim != 1:
                raise ValueError(
                    f"unexpected embedding shape {arr.shape} in {npz_path}",
                )
            accessions.append(npz_path.stem)
            arrays.append(arr.astype(np.float32))
        if arrays:
            out[split] = (accessions, np.stack(arrays, axis=0))
    if not out:
        raise FileNotFoundError(
            f"no .npz embeddings under {embeddings_dir}/{{train,valid,test,dev}}",
        )
    return out


def load_features_from_cache_split(
    cache_dir: Path,
    labels_json_path: Path,
    qa_key: str = _DEFAULT_QA_KEY,
    dataset_config_path: Optional[Path] = None,
    cohort: Optional[str] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]]:
    """Per-split `{split: (features, labels, patient_ids)}` for fixed-split evaluation."""
    cache_dir = Path(cache_dir)
    labels_json_path = Path(labels_json_path)
    _maybe_warn_cache_drift(cache_dir, Path(dataset_config_path) if dataset_config_path else None)

    per_split = _iter_embeddings_per_split(cache_dir)
    label_map = _load_lidc_labels(labels_json_path, qa_key=qa_key)
    key_fn = resolve_patient_key_fn(cohort)

    out: Dict[str, Tuple[np.ndarray, np.ndarray, List[str]]] = {}
    for split, (accessions, feats) in per_split.items():
        keep_idx, keep_acc, keep_lbl = [], [], []
        for i, acc in enumerate(accessions):
            if acc in label_map:
                keep_idx.append(i)
                keep_acc.append(acc)
                keep_lbl.append(label_map[acc])
        if not keep_idx:
            continue
        keep_arr = np.array(keep_idx, dtype=np.int64)
        patient_ids = [key_fn(acc) for acc in keep_acc]
        out[split] = (
            feats[keep_arr],
            np.array(keep_lbl, dtype=np.int64),
            patient_ids,
        )
    if not out:
        raise RuntimeError(
            f"no overlap between cache splits and labels JSON -- check that "
            f"--labels-json points to the right file ({labels_json_path})",
        )
    return out


def load_features_from_cache(
    cache_dir: Path,
    labels_json_path: Path,
    qa_key: str = _DEFAULT_QA_KEY,
    dataset_config_path: Optional[Path] = None,
    cohort: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Features + labels for accessions in BOTH the cache and the labels JSON -> (features, labels, patient_ids)."""
    cache_dir = Path(cache_dir)
    labels_json_path = Path(labels_json_path)
    _maybe_warn_cache_drift(cache_dir, Path(dataset_config_path) if dataset_config_path else None)

    accessions, feats = _iter_embeddings(cache_dir)
    label_map = _load_lidc_labels(labels_json_path, qa_key=qa_key)

    keep_idx, keep_acc, keep_lbl = [], [], []
    missing = 0
    for i, acc in enumerate(accessions):
        if acc in label_map:
            keep_idx.append(i)
            keep_acc.append(acc)
            keep_lbl.append(label_map[acc])
        else:
            missing += 1
    if not keep_idx:
        raise RuntimeError(
            f"no overlap between cache ({len(accessions)} embeddings) and labels "
            f"({len(label_map)} accessions) -- check that --labels-json points to the right file",
        )

    keep_arr = np.array(keep_idx, dtype=np.int64)
    key_fn = resolve_patient_key_fn(cohort)
    patient_ids = [key_fn(acc) for acc in keep_acc]
    return (
        feats[keep_arr],
        np.array(keep_lbl, dtype=np.int64),
        patient_ids,
    )
