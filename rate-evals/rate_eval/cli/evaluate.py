"""CLI for evaluating cached embeddings: cv (subject-level k-fold CV) and compare (paired-bootstrap AUROC + Holm)."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ..core.logging import get_logger, setup_logging
from ..core.crossval import (
    CVFold,
    assert_no_patient_leak,
    carve_validation_split,
    make_subject_stratified_folds,
    write_splits_json,
)
from ..evaluation import (
    TrainConfig,
    aggregate_folds,
    bootstrap_ci_auroc,
    build_head,
    holm_correct,
    load_features_from_cache,
    paired_bootstrap_diff,
    plot_calibration,
    plot_pr_with_bootstrap,
    plot_roc_with_bootstrap,
    train_one_fold,
    train_one_fold_multiclass,
    predict_proba_multiclass,
    multiclass_aggregate_folds,
    bootstrap_ci_multiclass_auroc,
)
from ..evaluation.train import predict_proba
from ..core.seed import Seed


logger = get_logger(__name__)


def _build_train_config(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        max_lr=args.max_lr,
        weight_decay=args.weight_decay,
        device=args.device,
    )


def _l2_grid(d: int, n_patients: int, base_weight_decay: float) -> List[float]:
    """Inner-CV weight-decay candidates: {1e-2, 1e-1, 1} x (1/d); fixed fallback below ~2000 patients."""
    inv_d = 1.0 / float(d)
    if n_patients >= 2000:
        return [m * inv_d for m in (1e-2, 1e-1, 1.0)]
    return [base_weight_decay * inv_d]


def _train_binary_fold_grid(
    head_spec: Dict[str, Any],
    d: int,
    features_t: "torch.Tensor",
    labels_t: "torch.Tensor",
    inner_train_idx: np.ndarray,
    val_idx: np.ndarray,
    train_cfg: TrainConfig,
    wd_grid: List[float],
):
    """Train a binary head per weight_decay candidate, select best val AUROC."""
    from dataclasses import replace

    best = None
    for wd in wd_grid:
        cfg = replace(train_cfg, weight_decay=wd)
        head = build_head(head_spec, dim_input=d, dim_output=1)
        head, stats = train_one_fold(
            head=head,
            train_features=features_t[inner_train_idx],
            train_labels=labels_t[inner_train_idx],
            val_features=features_t[val_idx],
            val_labels=labels_t[val_idx],
            config=cfg,
        )
        val_auroc = float(stats.get("best_val_auroc", float("nan")))
        score = val_auroc if not np.isnan(val_auroc) else -1.0
        if best is None or score > best[3]:
            best = (head, stats, wd, score)
    head, stats, wd_sel, _ = best
    return head, {**stats, "weight_decay": wd_sel}, wd_sel


def _pooled_cluster_ci_fields(
    fold_y_true: List[np.ndarray],
    fold_y_score: List[np.ndarray],
    fold_pids_test: List[List[str]],
    *,
    cohort: Optional[str],
    n_boot: int,
    alpha: float,
    seed: int,
    multiclass: bool,
) -> Dict[str, Any]:
    """Patient-cluster BCa CI on the pooled OOF predictions; {} when no cohort is given."""
    if not cohort:
        return {}
    y_true_all = np.concatenate(fold_y_true)
    pooled_groups = np.concatenate([np.asarray(p) for p in fold_pids_test])
    if multiclass:
        y_score_all = np.concatenate(fold_y_score, axis=0)
        _, lo, hi = bootstrap_ci_multiclass_auroc(
            y_true_all, y_score_all, n_boot=n_boot, alpha=alpha, seed=seed, groups=pooled_groups,
        )
        ci_key = "pooled_auroc_ovr_ci_cluster"
    else:
        y_score_all = np.concatenate(fold_y_score)
        _, lo, hi = bootstrap_ci_auroc(
            y_true_all, y_score_all, n_boot=n_boot, alpha=alpha, seed=seed, groups=pooled_groups,
        )
        ci_key = "pooled_auroc_ci_cluster"
    return {
        ci_key: [lo, hi],
        "ci_resample_unit": "patient_cluster",
        "n_patients": int(np.unique(pooled_groups).size),
        "cohort": cohort,
    }


def _head_spec(args: argparse.Namespace) -> Dict[str, Any]:
    if getattr(args, "zero_shot", False):
        return {
            "kind": "zero_shot",
            "teacher": args.zero_shot_teacher,
            "prompt_pos": args.zero_shot_prompt_pos,
            "prompt_neg": args.zero_shot_prompt_neg,
        }
    spec: Dict[str, Any] = {"kind": args.head}
    if args.head_config_json:
        spec.update(json.loads(args.head_config_json))
    return spec


def _compute_zero_shot_scores(
    teacher: str,
    features: np.ndarray,
    prompt_pos: str,
    prompt_neg: str,
    device: Optional[str],
) -> np.ndarray:
    """Score cached image latents against a (positive, negative) prompt pair."""
    from omegaconf import OmegaConf

    from ..components import _resolve_model_class
    from ..config import load_model_config

    model_cls = _resolve_model_class(teacher)
    if not hasattr(model_cls, "score_from_latent"):
        raise SystemExit(
            f"--zero-shot only supports teachers exposing score_from_latent; "
            f"'{teacher}' ({model_cls.__name__}) does not.",
        )

    model_yaml = load_model_config(teacher)
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = OmegaConf.create({"model": model_yaml, "device": resolved_device})
    model = model_cls(cfg)

    feats_t = torch.from_numpy(features).float()
    return model.score_from_latent(feats_t, prompts=(prompt_pos, prompt_neg))


def _run_cv(args: argparse.Namespace) -> None:
    setup_logging(level=args.log_level)
    Seed.set(args.seed)
    if args.zero_shot:
        missing = [
            n for n, v in (
                ("--zero-shot-teacher", args.zero_shot_teacher),
                ("--zero-shot-prompt-pos", args.zero_shot_prompt_pos),
                ("--zero-shot-prompt-neg", args.zero_shot_prompt_neg),
            ) if not v
        ]
        if missing:
            raise SystemExit(f"--zero-shot requires {', '.join(missing)}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading features from cache: %s", args.checkpoint_dir)
    features, labels, patient_ids = load_features_from_cache(
        cache_dir=Path(args.checkpoint_dir),
        labels_json_path=Path(args.labels_json),
        qa_key=args.qa_key,
        cohort=getattr(args, "cohort", None),
    )
    # L2-normalize each feature at eval time (skip in zero-shot; wrapper normalizes there)
    feature_l2_normalized = bool(getattr(args, "l2_normalize", False)) and not args.zero_shot
    if feature_l2_normalized:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        features = (features / norms).astype(np.float32)
    n, d = features.shape
    n_patients = int(len(set(patient_ids)))
    logger.info("Loaded %d samples x %d-dim features (pos=%d, neg=%d)",
                n, d, int((labels == 1).sum()), int((labels == 0).sum()))
    if n < args.cv_folds * 2:
        raise RuntimeError(
            f"Too few samples ({n}) for {args.cv_folds}-fold CV; need >= {args.cv_folds * 2}",
        )

    # --num-classes is authoritative (default 2 = binary); auto-detect below is advisory
    distinct_labels = int(np.unique(labels).size)
    num_classes = int(getattr(args, "num_classes", 2) or 2)
    if num_classes < 2:
        raise SystemExit(f"--num-classes must be >= 2, got {num_classes}")
    if num_classes > 2 and distinct_labels < num_classes:
        logger.warning(
            "data has %d distinct labels but --num-classes=%d -- some classes are absent",
            distinct_labels, num_classes,
        )
    if num_classes == 2 and distinct_labels > 2:
        raise SystemExit(
            f"data has {distinct_labels} distinct labels but --num-classes=2 (default). "
            f"Pass --num-classes {distinct_labels} for multi-class evaluation.",
        )
    if num_classes > 2 and getattr(args, "zero_shot", False):
        raise SystemExit("--zero-shot is binary-only; use a head probe for multi-class")

    folds: List[CVFold] = make_subject_stratified_folds(
        patient_ids=patient_ids,
        labels=labels.tolist(),
        n_splits=args.cv_folds,
        seed=args.cv_seed,
    )
    assert_no_patient_leak(folds)
    write_splits_json(folds, out_dir / "splits.json")
    logger.info("Wrote subject-level fold assignments -> %s", out_dir / "splits.json")

    features_t = torch.from_numpy(features)
    labels_t = torch.from_numpy(labels)

    fold_y_true: List[np.ndarray] = []
    fold_y_score: List[np.ndarray] = []
    fold_train_stats: List[Dict[str, Any]] = []
    fold_pids_test: List[List[str]] = []

    head_spec = _head_spec(args)
    train_cfg = _build_train_config(args)

    if args.zero_shot:
        # score every latent once, then slice into per-fold OOF arrays
        logger.info("=== Zero-shot mode: teacher=%s pos=%r neg=%r ===",
                    args.zero_shot_teacher, args.zero_shot_prompt_pos, args.zero_shot_prompt_neg)
        scores = _compute_zero_shot_scores(
            teacher=args.zero_shot_teacher,
            features=features,
            prompt_pos=args.zero_shot_prompt_pos,
            prompt_neg=args.zero_shot_prompt_neg,
            device=args.device,
        )
        for fold in folds:
            fold_y_true.append(labels[fold.test_indices])
            fold_y_score.append(scores[fold.test_indices])
            fold_pids_test.append([patient_ids[i] for i in fold.test_indices])
            fold_train_stats.append({
                "fold": fold.fold_idx, "best_epoch": -1, "best_val_auroc": float("nan"),
            })
            logger.info("Fold %d: zero-shot, n_test=%d", fold.fold_idx, fold.test_indices.size)
    elif num_classes > 2:
        for fold in folds:
            logger.info("=== Fold %d/%d (train=%d, test=%d) [multi-class C=%d] ===",
                        fold.fold_idx + 1, args.cv_folds, fold.train_indices.size,
                        fold.test_indices.size, num_classes)

            inner_train_idx, val_idx = carve_validation_split(
                fold.train_indices, patient_ids, labels.tolist(),
                val_fraction=args.val_fraction, seed=args.seed + fold.fold_idx,
            )

            head = build_head(head_spec, dim_input=d, dim_output=num_classes)
            head, stats = train_one_fold_multiclass(
                head=head,
                train_features=features_t[inner_train_idx],
                train_labels=labels_t[inner_train_idx],
                val_features=features_t[val_idx],
                val_labels=labels_t[val_idx],
                num_classes=num_classes,
                config=train_cfg,
                use_class_weights=getattr(args, "use_class_weights", False),
            )

            test_proba = predict_proba_multiclass(
                head, features_t[fold.test_indices], device=args.device,
            )  # shape (N_test, C)
            test_y = labels[fold.test_indices]
            fold_y_true.append(test_y)
            fold_y_score.append(test_proba)
            fold_pids_test.append([patient_ids[i] for i in fold.test_indices])
            fold_train_stats.append({"fold": fold.fold_idx, **stats})

            logger.info("Fold %d: best_epoch=%d val_bacc=%.4f n_test=%d",
                        fold.fold_idx, stats["best_epoch"], stats["best_val_bacc"], test_y.size)
    else:
        use_grid = bool(getattr(args, "l2_grid", False))
        wd_grid = _l2_grid(d, n_patients, args.weight_decay) if use_grid else None
        selected_wds: List[float] = []
        if use_grid:
            logger.info("Inner-CV L2 grid %s (d=%d, n_patients=%d)", wd_grid, d, n_patients)
        for fold in folds:
            logger.info("=== Fold %d/%d (train=%d, test=%d) ===",
                        fold.fold_idx + 1, args.cv_folds, fold.train_indices.size, fold.test_indices.size)

            inner_train_idx, val_idx = carve_validation_split(
                fold.train_indices, patient_ids, labels.tolist(),
                val_fraction=args.val_fraction, seed=args.seed + fold.fold_idx,
            )

            if use_grid:
                head, stats, wd_sel = _train_binary_fold_grid(
                    head_spec, d, features_t, labels_t, inner_train_idx, val_idx, train_cfg, wd_grid,
                )
                selected_wds.append(wd_sel)
            else:
                head = build_head(head_spec, dim_input=d, dim_output=1)
                head, stats = train_one_fold(
                    head=head,
                    train_features=features_t[inner_train_idx],
                    train_labels=labels_t[inner_train_idx],
                    val_features=features_t[val_idx],
                    val_labels=labels_t[val_idx],
                    config=train_cfg,
                )

            test_proba = predict_proba(head, features_t[fold.test_indices], device=args.device)
            test_y = labels[fold.test_indices]
            fold_y_true.append(test_y)
            fold_y_score.append(test_proba)
            fold_pids_test.append([patient_ids[i] for i in fold.test_indices])
            fold_train_stats.append({"fold": fold.fold_idx, **stats})

            logger.info("Fold %d: best_epoch=%d val_auroc=%.4f n_test=%d",
                        fold.fold_idx, stats["best_epoch"], stats["best_val_auroc"], test_y.size)

    title_prefix = args.title_prefix or f"{Path(args.checkpoint_dir).name} | head={args.head}"

    if num_classes > 2:
        mc = multiclass_aggregate_folds(
            fold_y_true=fold_y_true,
            fold_y_score=fold_y_score,
            n_boot=args.n_boot,
            alpha=args.alpha,
            seed=args.seed,
        )
        summary = {
            "head_spec": head_spec,
            "num_classes": num_classes,
            "cv_folds": args.cv_folds,
            "cv_seed": args.cv_seed,
            "n_samples": int(n),
            "feature_dim": int(d),
            "pooled": mc.pooled_metrics,
            "pooled_balanced_accuracy_ci": list(mc.pooled_balanced_accuracy_ci),
            "pooled_auroc_ovr_ci": list(mc.pooled_auroc_ovr_ci),
            "fold_summary": {k: list(v) for k, v in mc.fold_summary.items()},
            "per_fold": mc.per_fold_metrics,
            "training_stats": fold_train_stats,
        }
        summary.update(_pooled_cluster_ci_fields(
            fold_y_true, fold_y_score, fold_pids_test,
            cohort=getattr(args, "cohort", None),
            n_boot=args.n_boot, alpha=args.alpha, seed=args.seed, multiclass=True,
        ))
        if feature_l2_normalized:
            summary["feature_l2_normalized"] = True
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

        rows = []
        for fold_idx, (pids, y, p) in enumerate(zip(fold_pids_test, fold_y_true, fold_y_score)):
            for pid, yi, pi in zip(pids, y.tolist(), p.tolist()):
                rows.append({
                    "fold": fold_idx,
                    "patient_id": pid,
                    "label": int(yi),
                    "score": [float(x) for x in pi],
                })
        (out_dir / "predictions_oof.json").write_text(json.dumps(rows, indent=2, default=float))

        bacc = mc.pooled_metrics["balanced_accuracy"]
        bacc_lo, bacc_hi = mc.pooled_balanced_accuracy_ci
        auroc = mc.pooled_metrics["auroc_ovr"]
        auroc_lo, auroc_hi = mc.pooled_auroc_ovr_ci
        print(f"\n=== CV results [multi-class, C={num_classes}]: {title_prefix} ===")
        print(f"  pooled balanced_accuracy = {bacc:.4f}  [{bacc_lo:.4f}, {bacc_hi:.4f}] (percentile 95% CI, n_boot={args.n_boot})")
        print(f"  pooled AUROC OvR (macro) = {auroc:.4f}  [{auroc_lo:.4f}, {auroc_hi:.4f}]")
        print(f"  pooled accuracy = {mc.pooled_metrics['accuracy']:.4f}")
        print(f"  pooled macro_f1 = {mc.pooled_metrics['macro_f1']:.4f}")
        if "balanced_accuracy" in mc.fold_summary:
            mean, std = mc.fold_summary["balanced_accuracy"]
            print(f"  per-fold bacc mean +/- std = {mean:.4f} +/- {std:.4f}")
        print(f"  artefacts -> {out_dir}")
        return

    cv = aggregate_folds(
        fold_y_true=fold_y_true,
        fold_y_score=fold_y_score,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
    )

    summary = {
        "head_spec": head_spec,
        "cv_folds": args.cv_folds,
        "cv_seed": args.cv_seed,
        "n_samples": int(n),
        "feature_dim": int(d),
        "pooled": cv.pooled_metrics,
        "pooled_auroc_ci": list(cv.pooled_auroc_ci),
        "pooled_auprc_ci": list(cv.pooled_auprc_ci),
        "fold_summary": {k: list(v) for k, v in cv.fold_summary.items()},
        "per_fold": cv.per_fold_metrics,
        "training_stats": fold_train_stats,
    }
    summary.update(_pooled_cluster_ci_fields(
        fold_y_true, fold_y_score, fold_pids_test,
        cohort=getattr(args, "cohort", None),
        n_boot=args.n_boot, alpha=args.alpha, seed=args.seed, multiclass=False,
    ))
    if feature_l2_normalized:
        summary["feature_l2_normalized"] = True
    if getattr(args, "l2_grid", False) and selected_wds:
        summary["weight_decay_effective"] = float(np.median(selected_wds))
        summary["weight_decay_per_fold"] = [float(w) for w in selected_wds]
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))

    rows = []
    for fold_idx, (pids, y, p) in enumerate(zip(fold_pids_test, fold_y_true, fold_y_score)):
        for pid, yi, pi in zip(pids, y.tolist(), p.tolist()):
            rows.append({"fold": fold_idx, "patient_id": pid, "label": int(yi), "score": float(pi)})
    (out_dir / "predictions_oof.json").write_text(json.dumps(rows, indent=2, default=float))

    y_true_all = np.concatenate(fold_y_true)
    y_score_all = np.concatenate(fold_y_score)
    plot_roc_with_bootstrap(y_true_all, y_score_all, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
                            title=f"{title_prefix} -- ROC (pooled OOF)").savefig(out_dir / "roc.png", dpi=160)
    plot_pr_with_bootstrap(y_true_all, y_score_all, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
                           title=f"{title_prefix} -- PR (pooled OOF)").savefig(out_dir / "pr.png", dpi=160)
    plot_calibration(y_true_all, y_score_all, n_bins=10,
                     title=f"{title_prefix} -- Calibration (pooled OOF)").savefig(out_dir / "calibration.png", dpi=160)

    auc, auc_lo, auc_hi = bootstrap_ci_auroc(y_true_all, y_score_all, n_boot=args.n_boot, seed=args.seed)
    print(f"\n=== CV results: {title_prefix} ===")
    print(f"  pooled AUROC = {auc:.4f}  [{auc_lo:.4f}, {auc_hi:.4f}] (BCa 95% CI, n_boot={args.n_boot})")
    print(f"  pooled AUPRC = {cv.pooled_metrics['auprc']:.4f}  "
          f"[{cv.pooled_auprc_ci[0]:.4f}, {cv.pooled_auprc_ci[1]:.4f}]")
    print(f"  pooled Brier = {cv.pooled_metrics['brier']:.4f}")
    print(f"  pooled sens@95%spec = {cv.pooled_metrics['sens_at_95spec']:.4f}")
    print(f"  per-fold AUROC mean +/- std = {cv.fold_summary['auroc'][0]:.4f} +/- {cv.fold_summary['auroc'][1]:.4f}")
    print(f"  artefacts -> {out_dir}")


def _run_compare(args: argparse.Namespace) -> None:
    """Compare >=2 models' pooled-OOF predictions via paired bootstrap + Holm correction."""
    setup_logging(level=args.log_level)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(args.predictions_oof) < 2:
        raise SystemExit("--predictions-oof needs at least 2 paths to compare")

    per_model: Dict[str, List[Dict[str, Any]]] = {}
    for path in args.predictions_oof:
        rows = json.loads(Path(path).read_text())
        name = Path(path).parent.name
        # multiclass guard: scalar paired-AUROC compare is undefined on list-valued scores
        if rows and isinstance(rows[0].get("score"), list):
            raise SystemExit(
                f"multiclass compare unsupported: '{name}' has list-valued (softmax) scores "
                f"(num_classes>2). Use the ranking.py macro path for diagnosis / true-softmax cells."
            )
        per_model[name] = rows
        logger.info("loaded %s: %d OOF rows", name, len(rows))

    # align models by position after asserting identical (patient_id, label) sequences
    names = list(per_model.keys())
    ref_rows = per_model[names[0]]
    n = len(ref_rows)
    if n == 0:
        raise SystemExit("predictions_oof is empty")
    ref_keys = [(str(r["patient_id"]), int(r["label"])) for r in ref_rows]
    for name in names[1:]:
        rows = per_model[name]
        if len(rows) != n:
            raise SystemExit(
                f"row-count mismatch: '{names[0]}' has {n} rows, '{name}' has {len(rows)} -- "
                f"compare requires the same cohort + cv_seed."
            )
        if [(str(r["patient_id"]), int(r["label"])) for r in rows] != ref_keys:
            raise SystemExit(
                f"OOF alignment mismatch between '{names[0]}' and '{name}': the (patient_id, label) "
                f"sequences differ -- compare requires the same cohort + cv_seed (no silent re-pair)."
            )

    y_true = np.array([int(r["label"]) for r in ref_rows], dtype=int)
    group_id = np.array([str(r["patient_id"]) for r in ref_rows])
    aligned: Dict[str, np.ndarray] = {
        name: np.array([float(r["score"]) for r in per_model[name]], dtype=float) for name in names
    }
    n_patients = int(np.unique(group_id).size)
    logger.info("comparing %d models over %d rows (%d patient clusters)", len(names), n, n_patients)

    pairs: Dict[str, Dict[str, Any]] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            diff, (lo, hi), p = paired_bootstrap_diff(
                y_true, aligned[a], aligned[b],
                n_boot=args.n_boot, alpha=args.alpha, seed=args.seed, groups=group_id,
            )
            pairs[f"{a}__vs__{b}"] = {
                "diff_auroc": diff,
                "ci": [lo, hi],
                "p_value": p,
            }

    holm_in = {k: v["p_value"] for k, v in pairs.items()}
    holm = holm_correct(holm_in, alpha=args.alpha)
    for k, (adj_p, reject) in holm.items():
        pairs[k]["p_holm"] = adj_p
        pairs[k]["reject_at_alpha"] = reject

    out = {
        "n_rows": n,
        "n_patients": n_patients,
        "ci_resample_unit": "patient_cluster",
        "alpha": args.alpha,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "pairs": pairs,
    }
    (out_dir / "pairwise.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\n=== Pairwise comparison (n_rows={n}, {n_patients} patient clusters) ===")
    for k, v in pairs.items():
        marker = "ok" if v["reject_at_alpha"] else " "
        print(f"  [{marker}] {k}: deltaAUROC = {v['diff_auroc']:+.4f} "
              f"[{v['ci'][0]:+.4f}, {v['ci'][1]:+.4f}]  p={v['p_value']:.4f}  Holm-p={v['p_holm']:.4f}")
    print(f"  -> {out_dir / 'pairwise.json'}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate frozen-feature radiology FM embeddings via subject-level k-fold CV.",
    )
    sub = p.add_subparsers(dest="command")

    cv = sub.add_parser("cv", help="Subject-level stratified k-fold CV")
    cv.add_argument("--checkpoint-dir", type=str, required=True,
                    help="Path to rate-extract output dir (contains embeddings/{train,valid,test}/*.npz)")
    cv.add_argument("--labels-json", type=str, required=True,
                    help="Path to labels JSON (LIDC malignancy format)")
    cv.add_argument("--qa-key", type=str,
                    default="Does this scan contain a malignant nodule (median radiologist rating > 3)?")
    cv.add_argument("--cohort", type=str, default=None,
                    help="Cohort name selecting the patient-grouping key (feature_loaders registry: "
                         "ct_rate / rspect / midrc; unknown/None falls back to the __-split "
                         "default). When set, a patient's multi-volume siblings cluster into the same "
                         "fold (closing the point-estimate leak) and the pooled OOF additionally gets a "
                         "patient-cluster BCa CI (pooled_auroc_ci_cluster + ci_resample_unit + n_patients). "
                         "The legacy row CI (pooled_auroc_ci) is unchanged.")
    cv.add_argument("--output-dir", type=str, required=True)
    cv.add_argument("--cv-folds", type=int, default=5)
    cv.add_argument("--cv-seed", type=int, default=42)
    cv.add_argument("--seed", type=int, default=42, help="Seed for trainer + bootstrap")
    cv.add_argument("--num-classes", type=int, default=2,
                    help="Number of classes. 2 (default) = binary path (BCE + AUROC). "
                         ">2 = multi-class path (CrossEntropy + balanced accuracy + AUROC OvR). "
                         "Required for COVIDx CT (3) and other multi-class tasks.")
    cv.add_argument("--use-class-weights", action="store_true",
                    help="Multi-class only: enable inverse-frequency CrossEntropy weighting. "
                         "Off by default (matches Curia paper recipe -- let the metric do balancing, "
                         "not the loss).")
    cv.add_argument("--head", type=str, default="linear",
                    choices=["linear", "mlp"])
    cv.add_argument("--head-config-json", type=str, default=None,
                    help="JSON dict with extra head kwargs (e.g. '{\"dim_hidden\":256}')")
    cv.add_argument("--max-epochs", type=int, default=32)
    cv.add_argument("--patience", type=int, default=16)
    cv.add_argument("--batch-size", type=int, default=64)
    cv.add_argument("--max-lr", type=float, default=1e-4)
    cv.add_argument("--weight-decay", type=float, default=0.01)
    cv.add_argument("--l2-normalize", action="store_true",
                    help="L2-normalize each feature vector at eval time, uniformly across teachers "
                         "(benchmark KD-suitability probe; mirrors KDInfoNCELoss). Records "
                         "feature_l2_normalized=true. Off by default.")
    cv.add_argument("--l2-grid", action="store_true",
                    help="Select weight_decay by inner-CV over the re-centered grid {1e-2,1e-1,1} x "
                         "(1/d_t) on the carved val slice (fixed 1/d_t fallback below ~2000 patients). "
                         "Records weight_decay_effective. Off by default (uses fixed --weight-decay).")
    cv.add_argument("--val-fraction", type=float, default=0.15)
    cv.add_argument("--n-boot", type=int, default=1000)
    cv.add_argument("--alpha", type=float, default=0.05)
    cv.add_argument("--device", type=str, default=None, help="cuda / cpu (default: auto)")
    cv.add_argument("--title-prefix", type=str, default=None)
    cv.add_argument("--log-level", type=str, default="INFO")
    cv.add_argument("--zero-shot", action="store_true",
                    help="Bypass head training; score cached features against text prompts via the teacher's text encoder. "
                         "Requires --zero-shot-teacher / --zero-shot-prompt-pos / --zero-shot-prompt-neg.")
    cv.add_argument("--zero-shot-teacher", type=str, default=None,
                    help="Teacher whose text encoder + score_from_latent path is used (e.g. ctclip_zero_shot).")
    cv.add_argument("--zero-shot-prompt-pos", type=str, default=None,
                    help='Positive prompt, e.g. "Lung nodule is present."')
    cv.add_argument("--zero-shot-prompt-neg", type=str, default=None,
                    help='Negative prompt, e.g. "Lung nodule is not present."')
    cv.set_defaults(func=_run_cv)

    cmp = sub.add_parser("compare", help="Paired-bootstrap pairwise AUROC comparison + Holm correction")
    cmp.add_argument("--predictions-oof", type=str, nargs="+", required=True,
                     help="Paths to predictions_oof.json from >=2 model runs")
    cmp.add_argument("--output-dir", type=str, required=True)
    cmp.add_argument("--n-boot", type=int, default=1000)
    cmp.add_argument("--alpha", type=float, default=0.05)
    cmp.add_argument("--seed", type=int, default=42)
    cmp.add_argument("--log-level", type=str, default="INFO")
    cmp.set_defaults(func=_run_compare)

    return p


def evaluate_embeddings_cli() -> None:
    """rate-evaluate console-script entry."""
    parser = _build_parser()
    args = parser.parse_args()
    if not getattr(args, "command", None):
        parser.print_help()
        raise SystemExit(2)
    args.func(args)


def main() -> None:
    evaluate_embeddings_cli()


if __name__ == "__main__":
    main()
