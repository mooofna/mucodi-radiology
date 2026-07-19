"""crossval -> deploy evaluator (cross-cohort external validation)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _extract_qa_key(labels_json_path: Path) -> str:
    """The single ``default_qa`` question string of a per-class label JSON."""
    d = json.loads(Path(labels_json_path).read_text())
    for rec in d.values():
        qa = (rec.get("qa_results") or {}).get("default_qa") or []
        if qa and isinstance(qa[0], dict) and qa[0]:
            return next(iter(qa[0].keys()))
    raise ValueError(f"no default_qa question found in {labels_json_path}")


def _l2_normalize_rows(features: np.ndarray) -> np.ndarray:
    """Per-vector unit-norm (zero-norm -> 1), matching the eval-time L2 in ``cli/evaluate.py``."""
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (features / norms).astype(np.float32)


def deploy_one_class(
    *,
    source_cache: Path,
    source_labels_json: Path,
    target_cache: Path,
    target_labels_json: Path,
    class_name: str,
    out_dir: Path,
    head_spec: Dict[str, Any],
    l2_normalize: bool = True,
    source_cohort: str = "ct_rate",
    target_cohort: str = "radchestct",
    cv_folds: int = 5,
    cv_seed: int = 42,
    seed: int = 42,
    val_fraction: float = 0.15,
    n_boot: int = 1000,
    alpha: float = 0.05,
    max_epochs: int = 32,
    patience: int = 16,
    batch_size: int = 64,
    max_lr: float = 1e-4,
    weight_decay: float = 0.01,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one class's crossval(source) + ensemble-deploy(target); write both per-class dirs."""
    import torch  # lazy -- keep module import torch-free

    from ..core.crossval import (
        assert_no_patient_leak,
        carve_validation_split,
        make_subject_stratified_folds,
        write_splits_json,
    )
    from ..core.seed import Seed
    from ..evaluation import (
        aggregate_folds,
        bootstrap_ci_auroc,
        build_head,
        fold_metrics,
        load_features_from_cache,
        train_one_fold,
    )
    from ..evaluation.stats import bootstrap_ci_auprc
    from ..evaluation.train import TrainConfig, predict_proba
    from ..io.feature_loaders import _patient_id_from_accession, resolve_patient_key_fn

    Seed.set(seed)

    # guard the source key only: CT-RATE must cluster on patient id, not the __-split fallback
    if resolve_patient_key_fn(source_cohort) is _patient_id_from_accession:
        raise SystemExit(
            f"source cohort {source_cohort!r} does not resolve to a known patient-grouping key "
            f"(fell back to the __-split default) -- pass a registered cohort (e.g. 'ct_rate')."
        )

    src_qa = _extract_qa_key(source_labels_json)
    tgt_qa = _extract_qa_key(target_labels_json)

    feats_s, y_s, pids_s = load_features_from_cache(
        cache_dir=Path(source_cache), labels_json_path=Path(source_labels_json),
        qa_key=src_qa, cohort=source_cohort,
    )
    feats_t, y_t, pids_t = load_features_from_cache(
        cache_dir=Path(target_cache), labels_json_path=Path(target_labels_json),
        qa_key=tgt_qa, cohort=target_cohort,
    )
    if l2_normalize:
        feats_s = _l2_normalize_rows(feats_s)
        feats_t = _l2_normalize_rows(feats_t)

    n_s, d = feats_s.shape
    if feats_t.shape[1] != d:
        raise SystemExit(
            f"feature-dim mismatch: source d={d} but target d={feats_t.shape[1]} "
            f"(class {class_name!r}) -- source and target must be the SAME encoder."
        )
    if n_s < cv_folds * 2:
        raise RuntimeError(f"too few source samples ({n_s}) for {cv_folds}-fold CV (class {class_name})")

    internal_dir = Path(out_dir) / "internal" / "per_class" / class_name
    external_dir = Path(out_dir) / "external" / "per_class" / class_name
    internal_dir.mkdir(parents=True, exist_ok=True)
    external_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = TrainConfig(
        max_epochs=max_epochs, patience=patience, batch_size=batch_size,
        max_lr=max_lr, weight_decay=weight_decay, device=device,
    )

    folds = make_subject_stratified_folds(
        patient_ids=pids_s, labels=y_s.tolist(), n_splits=cv_folds, seed=cv_seed,
    )
    assert_no_patient_leak(folds)
    write_splits_json(folds, internal_dir / "splits.json")

    feats_s_t = torch.from_numpy(feats_s)
    y_s_t = torch.from_numpy(y_s)

    fold_y_true: List[np.ndarray] = []
    fold_y_score: List[np.ndarray] = []
    fold_pids_test: List[List[str]] = []
    fold_train_stats: List[Dict[str, Any]] = []
    heads: List[Any] = []

    for fold in folds:
        inner_train_idx, val_idx = carve_validation_split(
            fold.train_indices, pids_s, y_s.tolist(),
            val_fraction=val_fraction, seed=seed + fold.fold_idx,
        )
        head = build_head(head_spec, dim_input=d, dim_output=1)
        head, stats = train_one_fold(
            head=head,
            train_features=feats_s_t[inner_train_idx],
            train_labels=y_s_t[inner_train_idx],
            val_features=feats_s_t[val_idx],
            val_labels=y_s_t[val_idx],
            config=train_cfg,
        )
        test_proba = predict_proba(head, feats_s_t[fold.test_indices], device=device)
        fold_y_true.append(y_s[fold.test_indices])
        fold_y_score.append(test_proba)
        fold_pids_test.append([pids_s[i] for i in fold.test_indices])
        fold_train_stats.append({"fold": fold.fold_idx, **stats})
        heads.append(head)

    cv = aggregate_folds(fold_y_true, fold_y_score, n_boot=n_boot, alpha=alpha, seed=seed)
    pooled_y = np.concatenate(fold_y_true)
    pooled_s = np.concatenate(fold_y_score)
    pooled_pids = np.concatenate([np.asarray(p) for p in fold_pids_test])
    _, i_lo, i_hi = bootstrap_ci_auroc(
        pooled_y, pooled_s, n_boot=n_boot, alpha=alpha, seed=seed, groups=pooled_pids,
    )

    internal_summary = {
        "head_spec": head_spec,
        "protocol": "crossval_deploy_internal",
        "cv_folds": cv_folds,
        "cv_seed": cv_seed,
        "n_samples": int(n_s),
        "feature_dim": int(d),
        "pooled": cv.pooled_metrics,
        "pooled_auroc_ci": list(cv.pooled_auroc_ci),
        "pooled_auprc_ci": list(cv.pooled_auprc_ci),
        "pooled_auroc_ci_cluster": [i_lo, i_hi],
        "ci_resample_unit": "patient_cluster",
        "n_patients": int(np.unique(pooled_pids).size),
        "cohort": source_cohort,
        "feature_l2_normalized": bool(l2_normalize),
        "fold_summary": {k: list(v) for k, v in cv.fold_summary.items()},
        "per_fold": cv.per_fold_metrics,
        "training_stats": fold_train_stats,
    }
    (internal_dir / "summary.json").write_text(json.dumps(internal_summary, indent=2, default=float))
    internal_rows = [
        {"fold": fi, "patient_id": pid, "label": int(yi), "score": float(si)}
        for fi, (pids, yy, ss) in enumerate(zip(fold_pids_test, fold_y_true, fold_y_score))
        for pid, yi, si in zip(pids, yy.tolist(), ss.tolist())
    ]
    (internal_dir / "predictions_oof.json").write_text(json.dumps(internal_rows, indent=2, default=float))

    feats_t_t = torch.from_numpy(feats_t)
    per_head = np.stack([predict_proba(h, feats_t_t, device=device) for h in heads], axis=0)
    ensemble = per_head.mean(axis=0)  # mean-of-model-probabilities

    ext = fold_metrics(y_t, ensemble)
    _, e_lo, e_hi = bootstrap_ci_auroc(y_t, ensemble, n_boot=n_boot, alpha=alpha, seed=seed, groups=pids_t)
    _, ea_lo, ea_hi = bootstrap_ci_auprc(y_t, ensemble, n_boot=n_boot, alpha=alpha, seed=seed, groups=pids_t)

    external_summary = {
        "head_spec": head_spec,
        "protocol": "crossval_deploy_external",
        "n_ensemble": len(heads),
        "n_samples_source": int(n_s),
        "n_samples": int(y_t.size),
        "feature_dim": int(d),
        "pooled": ext,
        "pooled_auroc_ci": list(cv.pooled_auroc_ci),  # placeholder row CI unused downstream
        "pooled_auroc_ci_cluster": [e_lo, e_hi],
        "pooled_auprc_ci": [ea_lo, ea_hi],
        "ci_resample_unit": "patient_cluster",
        "n_patients": int(np.unique(pids_t).size),
        "cohort": target_cohort,
        "source_cohort": source_cohort,
        "feature_l2_normalized": bool(l2_normalize),
    }
    (external_dir / "summary.json").write_text(json.dumps(external_summary, indent=2, default=float))
    external_rows = [
        {"fold": 0, "patient_id": pid, "label": int(yi), "score": float(si)}
        for pid, yi, si in zip(pids_t, y_t.tolist(), ensemble.tolist())
    ]
    (external_dir / "predictions_oof.json").write_text(json.dumps(external_rows, indent=2, default=float))

    return {
        "class_name": class_name,
        "internal_dir": str(internal_dir),
        "external_dir": str(external_dir),
        "internal_auroc": internal_summary["pooled"].get("auroc"),
        "external_auroc": ext.get("auroc"),
        "n_source": int(n_s),
        "n_target": int(y_t.size),
    }
