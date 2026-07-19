"""Per-fold + pooled-OOF metrics, bootstrap CIs, and plotting over numpy arrays."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

_log = logging.getLogger(__name__)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def sensitivity_at_specificity(
    y_true: np.ndarray, y_score: np.ndarray, target_specificity: float = 0.95,
) -> Tuple[float, float]:
    """Highest-sensitivity threshold with specificity >= target; (nan, nan) if none."""
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(y_true, y_score)
    spec = 1.0 - fpr
    valid = spec >= target_specificity
    if not np.any(valid):
        return float("nan"), float("nan")
    idx = np.argmax(tpr * valid - 1e9 * (~valid))
    return float(tpr[idx]), float(thr[idx])


def fold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Flat dict of all single-fold binary metrics (auroc, auprc, brier, specificity, sens@95spec, ...)."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    out: Dict[str, float] = {}
    out["auroc"] = _safe_auc(y_true, y_score)
    out["auprc"] = _safe_ap(y_true, y_score)
    out["brier"] = float(brier_score_loss(y_true, y_score)) if np.unique(y_true).size > 1 else float("nan")
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    out["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    out["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")

    sens, thr = sensitivity_at_specificity(y_true, y_score, target_specificity=0.95)
    out["sens_at_95spec"] = sens
    out["sens_at_95spec_threshold"] = thr

    out["n_pos"] = int((y_true == 1).sum())
    out["n_neg"] = int((y_true == 0).sum())
    return out


def _percentile_ci(values: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    # empty -> NaN CI
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    lo = float(np.quantile(values, alpha / 2))
    hi = float(np.quantile(values, 1 - alpha / 2))
    return lo, hi


def bca_ci(
    samples: np.ndarray,
    point_estimate: float,
    jackknife: np.ndarray,
    alpha: float = 0.05,
) -> Tuple[float, float]:
    """Bias-corrected and accelerated bootstrap confidence interval (Efron, 1987)."""
    samples = np.asarray(samples, dtype=float)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return float("nan"), float("nan")

    p_lt = float((samples < point_estimate).mean())
    if p_lt <= 0.0 or p_lt >= 1.0:
        # all samples on one side -> percentile fallback
        return _percentile_ci(samples, alpha)
    z0 = st.norm.ppf(p_lt)

    jk = np.asarray(jackknife, dtype=float)
    jk_mean = jk.mean()
    num = ((jk_mean - jk) ** 3).sum()
    den = 6.0 * (((jk_mean - jk) ** 2).sum() ** 1.5)
    a = num / den if den > 0 else 0.0

    z_alpha_lo = st.norm.ppf(alpha / 2)
    z_alpha_hi = st.norm.ppf(1 - alpha / 2)

    def _adjust(z_alpha: float) -> float:
        return float(st.norm.cdf(z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))))

    q_lo = np.clip(_adjust(z_alpha_lo), 0.0, 1.0)
    q_hi = np.clip(_adjust(z_alpha_hi), 0.0, 1.0)
    lo = float(np.quantile(samples, q_lo))
    hi = float(np.quantile(samples, q_hi))
    return lo, hi


def _bootstrap_resample_idx(n: int, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Stratified-friendly bootstrap indices (n_boot, n)."""
    return rng.integers(0, n, size=(n_boot, n), dtype=np.int64)


def _cluster_resample_idx(
    groups: np.ndarray, n_boot: int, rng: np.random.Generator,
) -> List[np.ndarray]:
    """Patient-cluster bootstrap indices: resample whole groups (ragged per-replicate list)."""
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    member_idx = {g: np.where(groups == g)[0] for g in uniq}
    out: List[np.ndarray] = []
    for _ in range(n_boot):
        chosen = uniq[rng.integers(0, uniq.size, size=uniq.size)]
        out.append(np.concatenate([member_idx[g] for g in chosen]))
    return out


def _warn_nan_drop(n_kept: int, n_boot: int, metric: str) -> None:
    """Warn when NaN-dropped replicates (e.g. a resample that loses a class) cut effective n_boot below ~70%."""
    if n_boot > 0 and n_kept < 0.70 * n_boot:
        _log.warning(
            "effective n_boot for %s reduced to %d/%d (%d%%) by NaN-dropped resamples "
            "(class lost in resampling); CI has higher Monte-Carlo variance.",
            metric, n_kept, n_boot, round(100 * n_kept / n_boot),
        )


def bootstrap_ci_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    use_bca: bool = True,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Bootstrap (BCa or percentile) CI for AUROC -> (point, lo, hi); groups -> patient-cluster resample."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = y_true.size
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan"), float("nan")

    point = _safe_auc(y_true, y_score)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    if groups is None:
        idx = _bootstrap_resample_idx(n, n_boot, rng)
        for b in range(n_boot):
            ii = idx[b]
            samples[b] = _safe_auc(y_true[ii], y_score[ii])
    else:
        idx_list = _cluster_resample_idx(groups, n_boot, rng)
        for b in range(n_boot):
            ii = idx_list[b]
            samples[b] = _safe_auc(y_true[ii], y_score[ii])
    samples = samples[~np.isnan(samples)]

    if not use_bca:
        lo, hi = _percentile_ci(samples, alpha)
        return point, lo, hi

    # BCa jackknife: leave one row out (ungrouped) or one patient out (grouped).
    if groups is None:
        jackknife = np.empty(n, dtype=float)
        full_mask = np.ones(n, dtype=bool)
        for i in range(n):
            full_mask[i] = False
            jackknife[i] = _safe_auc(y_true[full_mask], y_score[full_mask])
            full_mask[i] = True
    else:
        groups_arr = np.asarray(groups)
        uniq = np.unique(groups_arr)
        jackknife = np.empty(uniq.size, dtype=float)
        for j, gid in enumerate(uniq):
            keep = groups_arr != gid
            jackknife[j] = _safe_auc(y_true[keep], y_score[keep])
    jackknife = jackknife[~np.isnan(jackknife)]
    lo, hi = bca_ci(samples, point, jackknife, alpha=alpha)
    return point, lo, hi


def bootstrap_ci_auprc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Percentile-bootstrap CI for AUPRC; groups -> patient-cluster resample."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan"), float("nan")

    point = _safe_ap(y_true, y_score)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    if groups is None:
        idx = _bootstrap_resample_idx(y_true.size, n_boot, rng)
        for b in range(n_boot):
            ii = idx[b]
            samples[b] = _safe_ap(y_true[ii], y_score[ii])
    else:
        idx_list = _cluster_resample_idx(groups, n_boot, rng)
        for b in range(n_boot):
            ii = idx_list[b]
            samples[b] = _safe_ap(y_true[ii], y_score[ii])
    samples = samples[~np.isnan(samples)]
    lo, hi = _percentile_ci(samples, alpha)
    return point, lo, hi


def _bootstrap_ci_scalar(
    stat_fn,
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    use_bca: bool = False,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Generic bootstrap CI for a scalar binary stat_fn(y_true, y_score) -> (point, lo, hi)."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan"), float("nan")

    point = stat_fn(y_true, y_score)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    if groups is None:
        idx = _bootstrap_resample_idx(y_true.size, n_boot, rng)
        for b in range(n_boot):
            ii = idx[b]
            samples[b] = stat_fn(y_true[ii], y_score[ii])
    else:
        idx_list = _cluster_resample_idx(groups, n_boot, rng)
        for b in range(n_boot):
            ii = idx_list[b]
            samples[b] = stat_fn(y_true[ii], y_score[ii])
    samples = samples[~np.isnan(samples)]

    if not use_bca:
        lo, hi = _percentile_ci(samples, alpha)
        return point, lo, hi

    if groups is None:
        n = y_true.size
        jackknife = np.empty(n, dtype=float)
        mask = np.ones(n, dtype=bool)
        for i in range(n):
            mask[i] = False
            jackknife[i] = stat_fn(y_true[mask], y_score[mask])
            mask[i] = True
    else:
        ga = np.asarray(groups)
        uniq = np.unique(ga)
        jackknife = np.empty(uniq.size, dtype=float)
        for j, gid in enumerate(uniq):
            keep = ga != gid
            jackknife[j] = stat_fn(y_true[keep], y_score[keep])
    jackknife = jackknife[~np.isnan(jackknife)]
    if np.isnan(point) or jackknife.size == 0:
        lo, hi = _percentile_ci(samples, alpha)
        return point, lo, hi
    lo, hi = bca_ci(samples, point, jackknife, alpha=alpha)
    return point, lo, hi


def _f1_at(threshold: float):
    def _f(yt: np.ndarray, ys: np.ndarray) -> float:
        if np.unique(yt).size < 2:
            return float("nan")
        return float(f1_score(yt.astype(int), (ys >= threshold).astype(int), zero_division=0))
    return _f


def _brier_stat(yt: np.ndarray, ys: np.ndarray) -> float:
    if np.unique(yt).size < 2:
        return float("nan")
    return float(brier_score_loss(yt.astype(int), ys.astype(float)))


def bootstrap_ci_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
    use_bca: bool = False,
) -> Tuple[float, float, float]:
    """Cluster-aware percentile CI for F1 at a fixed threshold (default 0.5, matching `fold_metrics`)."""
    return _bootstrap_ci_scalar(_f1_at(threshold), y_true, y_score,
                                n_boot=n_boot, alpha=alpha, seed=seed, use_bca=use_bca, groups=groups)


def bootstrap_ci_brier(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
    use_bca: bool = False,
) -> Tuple[float, float, float]:
    """Cluster-aware percentile CI for the Brier score."""
    return _bootstrap_ci_scalar(_brier_stat, y_true, y_score,
                                n_boot=n_boot, alpha=alpha, seed=seed, use_bca=use_bca, groups=groups)


def _sens95_stat(yt: np.ndarray, ys: np.ndarray) -> float:
    if np.unique(yt).size < 2:
        return float("nan")
    sens, _ = sensitivity_at_specificity(yt.astype(int), ys.astype(float), target_specificity=0.95)
    return float(sens)


def bootstrap_ci_sens_at_95spec(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
    use_bca: bool = False,
) -> Tuple[float, float, float]:
    """Cluster-aware percentile CI for sensitivity at 95% specificity."""
    return _bootstrap_ci_scalar(_sens95_stat, y_true, y_score,
                                n_boot=n_boot, alpha=alpha, seed=seed, use_bca=use_bca, groups=groups)


@dataclass
class CVMetrics:
    """Pooled-out-of-fold + per-fold metric snapshot."""

    pooled_metrics: Dict[str, float] = field(default_factory=dict)
    pooled_auroc_ci: Tuple[float, float] = (float("nan"), float("nan"))
    pooled_auprc_ci: Tuple[float, float] = (float("nan"), float("nan"))
    per_fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    fold_summary: Dict[str, Tuple[float, float]] = field(default_factory=dict)  # metric -> (mean, std)


def aggregate_folds(
    fold_y_true: Sequence[np.ndarray],
    fold_y_score: Sequence[np.ndarray],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> CVMetrics:
    """Aggregate per-fold scores into pooled-OOF + per-fold-variance summary."""
    per_fold = [fold_metrics(yt, ys) for yt, ys in zip(fold_y_true, fold_y_score)]

    y_true_all = np.concatenate([np.asarray(y) for y in fold_y_true])
    y_score_all = np.concatenate([np.asarray(s) for s in fold_y_score])
    pooled = fold_metrics(y_true_all, y_score_all)

    auroc, auroc_lo, auroc_hi = bootstrap_ci_auroc(y_true_all, y_score_all, n_boot=n_boot, alpha=alpha, seed=seed, groups=groups)
    auprc, auprc_lo, auprc_hi = bootstrap_ci_auprc(y_true_all, y_score_all, n_boot=n_boot, alpha=alpha, seed=seed, groups=groups)
    pooled["auroc"] = auroc
    pooled["auprc"] = auprc

    skip_keys = {"n_pos", "n_neg", "sens_at_95spec_threshold"}
    fold_summary: Dict[str, Tuple[float, float]] = {}
    if per_fold:
        for k in per_fold[0].keys():
            if k in skip_keys:
                continue
            vals = np.array([m[k] for m in per_fold], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                fold_summary[k] = (float("nan"), float("nan"))
            else:
                fold_summary[k] = (float(vals.mean()), float(vals.std(ddof=1)) if vals.size > 1 else 0.0)

    return CVMetrics(
        pooled_metrics=pooled,
        pooled_auroc_ci=(auroc_lo, auroc_hi),
        pooled_auprc_ci=(auprc_lo, auprc_hi),
        per_fold_metrics=per_fold,
        fold_summary=fold_summary,
    )


def paired_bootstrap_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, Tuple[float, float], float]:
    """Paired bootstrap on AUROC(A) - AUROC(B) -> (observed_diff, (lo, hi), two_sided_p)."""
    y_true = np.asarray(y_true)
    y_score_a = np.asarray(y_score_a)
    y_score_b = np.asarray(y_score_b)
    n = y_true.size
    if not (n == y_score_a.size == y_score_b.size):
        raise ValueError("y_true, y_score_a, y_score_b must have the same length")

    auroc_a = _safe_auc(y_true, y_score_a)
    auroc_b = _safe_auc(y_true, y_score_b)
    observed = auroc_a - auroc_b

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    if groups is None:
        idx = _bootstrap_resample_idx(n, n_boot, rng)
        for b in range(n_boot):
            ii = idx[b]
            a = _safe_auc(y_true[ii], y_score_a[ii])
            bb = _safe_auc(y_true[ii], y_score_b[ii])
            diffs[b] = a - bb
    else:
        idx_list = _cluster_resample_idx(groups, n_boot, rng)
        for b in range(n_boot):
            ii = idx_list[b]
            a = _safe_auc(y_true[ii], y_score_a[ii])
            bb = _safe_auc(y_true[ii], y_score_b[ii])
            diffs[b] = a - bb
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return observed, (float("nan"), float("nan")), float("nan")

    lo, hi = _percentile_ci(diffs, alpha)
    p_lt = float((diffs < 0).mean())
    p_gt = float((diffs > 0).mean())
    p_two_sided = float(2.0 * min(p_lt, p_gt))
    p_two_sided = max(p_two_sided, 1.0 / n_boot)
    return observed, (lo, hi), p_two_sided


def holm_correct(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Tuple[float, bool]]:
    """Holm-Bonferroni correction -> {key: (adjusted_p, reject_at_alpha)}."""
    keys = list(p_values.keys())
    raw = np.array([p_values[k] for k in keys], dtype=float)
    order = np.argsort(raw)
    m = len(keys)
    adj_sorted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = raw[idx] * (m - rank)
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adj_sorted[idx] = running_max
    return {k: (float(adj_sorted[i]), bool(adj_sorted[i] <= alpha)) for i, k in enumerate(keys)}


def bh_fdr_correct(p_values: Dict[str, float], alpha: float = 0.05) -> Dict[str, Tuple[float, bool]]:
    """Benjamini-Hochberg FDR correction -> {key: (adjusted_p, reject_at_alpha)}."""
    keys = list(p_values.keys())
    raw = np.array([p_values[k] for k in keys], dtype=float)
    order = np.argsort(raw)
    m = len(keys)
    adj = np.empty(m, dtype=float)
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        val = min(raw[idx] * m / (rank + 1), 1.0)
        running_min = min(running_min, val)
        adj[idx] = running_min
    return {k: (float(adj[i]), bool(adj[i] <= alpha)) for i, k in enumerate(keys)}


def plot_roc_with_bootstrap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    title: str = "ROC",
) -> "plt.Figure":
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = _safe_auc(y_true, y_score)

    rng = np.random.default_rng(seed)
    base_fpr = np.linspace(0, 1, 101)
    interp_tprs = np.empty((n_boot, base_fpr.size), dtype=float)
    aurocs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ii = rng.integers(0, y_true.size, size=y_true.size)
        if np.unique(y_true[ii]).size < 2:
            interp_tprs[b] = np.nan
            aurocs[b] = np.nan
            continue
        f, t, _ = roc_curve(y_true[ii], y_score[ii])
        interp_tprs[b] = np.interp(base_fpr, f, t, left=0.0, right=1.0)
        aurocs[b] = _safe_auc(y_true[ii], y_score[ii])
    auroc_lo, auroc_hi = _percentile_ci(aurocs[~np.isnan(aurocs)], alpha)

    mean_tpr = np.nanmean(interp_tprs, axis=0)
    lo_tpr = np.nanquantile(interp_tprs, alpha / 2, axis=0)
    hi_tpr = np.nanquantile(interp_tprs, 1 - alpha / 2, axis=0)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.fill_between(base_fpr, lo_tpr, hi_tpr, alpha=0.25, label=f"95% bootstrap CI")
    ax.plot(fpr, tpr, color="C0", linewidth=1.5, label=f"AUROC = {auc:.3f} [{auroc_lo:.3f}, {auroc_hi:.3f}]")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def plot_pr_with_bootstrap(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    title: str = "Precision-Recall",
) -> "plt.Figure":
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = _safe_ap(y_true, y_score)

    rng = np.random.default_rng(seed)
    base_recall = np.linspace(0, 1, 101)
    interp_precs = np.empty((n_boot, base_recall.size), dtype=float)
    aps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        ii = rng.integers(0, y_true.size, size=y_true.size)
        if np.unique(y_true[ii]).size < 2:
            interp_precs[b] = np.nan
            aps[b] = np.nan
            continue
        p, r, _ = precision_recall_curve(y_true[ii], y_score[ii])
        order = np.argsort(r)  # np.interp needs ascending recall
        interp_precs[b] = np.interp(base_recall, r[order], p[order], left=p[order][0], right=p[order][-1])
        aps[b] = _safe_ap(y_true[ii], y_score[ii])
    ap_lo, ap_hi = _percentile_ci(aps[~np.isnan(aps)], alpha)

    mean_prec = np.nanmean(interp_precs, axis=0)
    lo_prec = np.nanquantile(interp_precs, alpha / 2, axis=0)
    hi_prec = np.nanquantile(interp_precs, 1 - alpha / 2, axis=0)

    prevalence = float((y_true == 1).mean())
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axhline(prevalence, linestyle="--", color="gray", linewidth=1, label=f"baseline = {prevalence:.3f}")
    ax.fill_between(base_recall, lo_prec, hi_prec, alpha=0.25, label="95% bootstrap CI")
    ax.plot(recall, precision, color="C1", linewidth=1.5, label=f"AUPRC = {ap:.3f} [{ap_lo:.3f}, {ap_hi:.3f}]")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.005)
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    return fig


# multi-class (>2 class) metrics: balanced accuracy + one-vs-rest AUROC


def _safe_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean per-class recall; NaN if any class is absent."""
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(balanced_accuracy_score(y_true, y_pred))


def _safe_multiclass_auroc_ovr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """One-vs-rest macro-averaged AUROC; NaN if single-class."""
    if np.unique(y_true).size < 2:
        return float("nan")
    if y_score.ndim != 2:
        raise ValueError(f"y_score must be 2D (N, C), got {y_score.shape}")
    try:
        return float(roc_auc_score(y_true, y_score, multi_class="ovr", average="macro"))
    except ValueError:
        return float("nan")


def multiclass_fold_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """Per-fold multi-class metrics from `(N,)` int labels + `(N, C)` softmax scores."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    if y_score.ndim != 2:
        raise ValueError(f"y_score must be 2D (N, C), got {y_score.shape}")
    y_pred = np.argmax(y_score, axis=1).astype(int)

    out: Dict[str, float] = {}
    out["balanced_accuracy"] = _safe_balanced_accuracy(y_true, y_pred)
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["auroc_ovr"] = _safe_multiclass_auroc_ovr(y_true, y_score)
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    out["macro_precision"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    out["macro_recall"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    classes, counts = np.unique(y_true, return_counts=True)
    for c, n in zip(classes, counts):
        out[f"n_class_{int(c)}"] = int(n)
    return out


def bootstrap_ci_balanced_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Percentile-bootstrap CI for balanced accuracy -> (point, lo, hi); groups -> patient-cluster resample."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if y_true.size != y_pred.size:
        raise ValueError("y_true / y_pred length mismatch")

    point = _safe_balanced_accuracy(y_true, y_pred)
    if np.isnan(point):
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    if groups is None:
        idx = _bootstrap_resample_idx(y_true.size, n_boot, rng)
        for b in range(n_boot):
            ii = idx[b]
            samples[b] = _safe_balanced_accuracy(y_true[ii], y_pred[ii])
    else:
        idx_list = _cluster_resample_idx(groups, n_boot, rng)
        for b in range(n_boot):
            ii = idx_list[b]
            samples[b] = _safe_balanced_accuracy(y_true[ii], y_pred[ii])
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return point, float("nan"), float("nan")
    _warn_nan_drop(samples.size, n_boot, "balanced_accuracy")
    lo, hi = _percentile_ci(samples, alpha)
    return point, lo, hi


def bootstrap_ci_multiclass_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    """Percentile-bootstrap CI for OvR macro-averaged multi-class AUROC; groups -> patient-cluster resample."""
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    point = _safe_multiclass_auroc_ovr(y_true, y_score)
    if np.isnan(point):
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    if groups is None:
        idx = _bootstrap_resample_idx(y_true.size, n_boot, rng)
        for b in range(n_boot):
            ii = idx[b]
            samples[b] = _safe_multiclass_auroc_ovr(y_true[ii], y_score[ii])
    else:
        idx_list = _cluster_resample_idx(groups, n_boot, rng)
        for b in range(n_boot):
            ii = idx_list[b]
            samples[b] = _safe_multiclass_auroc_ovr(y_true[ii], y_score[ii])
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return point, float("nan"), float("nan")
    _warn_nan_drop(samples.size, n_boot, "multiclass_auroc_ovr")
    lo, hi = _percentile_ci(samples, alpha)
    return point, lo, hi


@dataclass
class MulticlassCVMetrics:
    """Pooled-out-of-fold + per-fold metric snapshot for multi-class classification."""

    pooled_metrics: Dict[str, float] = field(default_factory=dict)
    pooled_balanced_accuracy_ci: Tuple[float, float] = (float("nan"), float("nan"))
    pooled_auroc_ovr_ci: Tuple[float, float] = (float("nan"), float("nan"))
    per_fold_metrics: List[Dict[str, float]] = field(default_factory=list)
    fold_summary: Dict[str, Tuple[float, float]] = field(default_factory=dict)


def multiclass_aggregate_folds(
    fold_y_true: Sequence[np.ndarray],
    fold_y_score: Sequence[np.ndarray],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> MulticlassCVMetrics:
    """Aggregate per-fold multi-class scores into pooled-OOF + per-fold-variance summary."""
    per_fold = [multiclass_fold_metrics(yt, ys) for yt, ys in zip(fold_y_true, fold_y_score)]

    y_true_all = np.concatenate([np.asarray(y).astype(int) for y in fold_y_true])
    y_score_all = np.concatenate([np.asarray(s).astype(float) for s in fold_y_score], axis=0)
    pooled = multiclass_fold_metrics(y_true_all, y_score_all)

    y_pred_all = np.argmax(y_score_all, axis=1)
    bacc, bacc_lo, bacc_hi = bootstrap_ci_balanced_accuracy(
        y_true_all, y_pred_all, n_boot=n_boot, alpha=alpha, seed=seed, groups=groups,
    )
    auroc, auroc_lo, auroc_hi = bootstrap_ci_multiclass_auroc(
        y_true_all, y_score_all, n_boot=n_boot, alpha=alpha, seed=seed, groups=groups,
    )
    pooled["balanced_accuracy"] = bacc
    pooled["auroc_ovr"] = auroc

    skip_keys = {k for k in (per_fold[0].keys() if per_fold else []) if k.startswith("n_class_")}
    fold_summary: Dict[str, Tuple[float, float]] = {}
    if per_fold:
        for k in per_fold[0].keys():
            if k in skip_keys:
                continue
            vals = np.array([m[k] for m in per_fold], dtype=float)
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                fold_summary[k] = (float("nan"), float("nan"))
            else:
                fold_summary[k] = (
                    float(vals.mean()),
                    float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                )

    return MulticlassCVMetrics(
        pooled_metrics=pooled,
        pooled_balanced_accuracy_ci=(bacc_lo, bacc_hi),
        pooled_auroc_ovr_ci=(auroc_lo, auroc_hi),
        per_fold_metrics=per_fold,
        fold_summary=fold_summary,
    )


def plot_calibration(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration",
) -> "plt.Figure":
    y_true = np.asarray(y_true); y_score = np.asarray(y_score)
    if np.unique(y_true).size < 2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.text(0.5, 0.5, "single class -- calibration undefined", ha="center", va="center")
        return fig
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=n_bins, strategy="quantile")
    brier = float(brier_score_loss(y_true, y_score))

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1, label="perfect calibration")
    ax.plot(mean_pred, frac_pos, "o-", color="C2", label=f"empirical (Brier = {brier:.3f})")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Empirical positive rate")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


# torch-free kernels for diagnosis-macro ranking + cross-cohort aggregate

PerClassOOF = Tuple[np.ndarray, np.ndarray, np.ndarray]
"""One scored class: (y_true, y_score, patient_ids), aligned row-for-row."""


def _build_patient_code_map(
    per_class_lists: Sequence[Sequence[PerClassOOF]],
) -> Tuple[np.ndarray, Dict[object, int]]:
    """Sorted patient universe (union over every class of every list) + pid->int-code map."""
    pid_arrays = [
        np.asarray(pids)
        for per_class in per_class_lists
        for (_, _, pids) in per_class
    ]
    if not pid_arrays:
        return np.array([]), {}
    universe = np.unique(np.concatenate(pid_arrays))
    code_of = {pid: i for i, pid in enumerate(universe.tolist())}
    return universe, code_of


def _prepare_per_class(
    per_class: Sequence[PerClassOOF], code_of: Dict[object, int],
) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Map each class to (y_true, y_score, pid_codes) using the shared patient-code map."""
    prepared: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for (yt, ys, pids) in per_class:
        yt = np.asarray(yt)
        ys = np.asarray(ys, dtype=float)
        codes = np.array([code_of[p] for p in np.asarray(pids).tolist()], dtype=np.int64)
        prepared.append((yt, ys, codes))
    return prepared


def _macro_auroc_from_multiplicity(
    prepared: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], mult: np.ndarray,
    strict_classes: bool = False,
) -> float:
    """Macro AUROC under a per-patient resample multiplicity vector (mult[k] = times patient k drawn)."""
    aucs: List[float] = []
    for (yt, ys, codes) in prepared:
        row_mult = mult[codes]
        total = int(row_mult.sum())
        if total == 0:
            aucs.append(float("nan"))
            continue
        ii = np.repeat(np.arange(yt.size), row_mult)
        aucs.append(_safe_auc(yt[ii], ys[ii]))
    arr = np.asarray(aucs, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    if strict_classes and np.any(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def macro_auroc_cluster_ci(
    per_class: Sequence[PerClassOOF],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    use_bca: bool = True,
    strict_classes: bool = True,
) -> Tuple[float, float, float]:
    """Joint patient-clustered BCa CI for macro-AUROC (mean of per-class binary AUROCs) -> (point, lo, hi)."""
    if not per_class:
        return float("nan"), float("nan"), float("nan")
    universe, code_of = _build_patient_code_map([per_class])
    n_pat = universe.size
    if n_pat == 0:
        return float("nan"), float("nan"), float("nan")
    prepared = _prepare_per_class(per_class, code_of)

    point = _macro_auroc_from_multiplicity(prepared, np.ones(n_pat, dtype=np.int64), strict_classes=strict_classes)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, n_pat, size=n_pat), minlength=n_pat)
        samples[b] = _macro_auroc_from_multiplicity(prepared, mult, strict_classes=strict_classes)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return point, float("nan"), float("nan")
    _warn_nan_drop(samples.size, n_boot, "macro_auroc_cluster")

    if not use_bca:
        lo, hi = _percentile_ci(samples, alpha)
        return point, lo, hi

    # BCa jackknife: leave-one-patient-out over the shared universe.
    jackknife = np.empty(n_pat, dtype=float)
    base = np.ones(n_pat, dtype=np.int64)
    for k in range(n_pat):
        base[k] = 0
        jackknife[k] = _macro_auroc_from_multiplicity(prepared, base, strict_classes=strict_classes)
        base[k] = 1
    jackknife = jackknife[~np.isnan(jackknife)]
    if jackknife.size == 0:
        lo, hi = _percentile_ci(samples, alpha)
        return point, lo, hi
    lo, hi = bca_ci(samples, point, jackknife, alpha=alpha)
    return point, lo, hi


def _macro_stat_from_multiplicity(
    prepared: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray]], mult: np.ndarray,
    stat_fn, strict_classes: bool = False,
) -> float:
    """F1/AP analog of _macro_auroc_from_multiplicity (generic stat_fn)."""
    vals: List[float] = []
    for (yt, ys, codes) in prepared:
        row_mult = mult[codes]
        if int(row_mult.sum()) == 0:
            vals.append(float("nan"))
            continue
        ii = np.repeat(np.arange(yt.size), row_mult)
        vals.append(stat_fn(yt[ii], ys[ii]))
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")
    if strict_classes and np.any(np.isnan(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _macro_stat_cluster_ci(
    per_class: Sequence[PerClassOOF], stat_fn,
    n_boot: int = 1000, alpha: float = 0.05, seed: int = 42,
    use_bca: bool = True, strict_classes: bool = True,
) -> Tuple[float, float, float]:
    """F1/AUPRC analog of macro_auroc_cluster_ci: joint patient-clustered CI for a macro stat_fn."""
    if not per_class:
        return float("nan"), float("nan"), float("nan")
    universe, code_of = _build_patient_code_map([per_class])
    n_pat = universe.size
    if n_pat == 0:
        return float("nan"), float("nan"), float("nan")
    prepared = _prepare_per_class(per_class, code_of)
    point = _macro_stat_from_multiplicity(prepared, np.ones(n_pat, dtype=np.int64), stat_fn, strict_classes)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, n_pat, size=n_pat), minlength=n_pat)
        samples[b] = _macro_stat_from_multiplicity(prepared, mult, stat_fn, strict_classes)
    samples = samples[~np.isnan(samples)]
    if samples.size == 0:
        return point, float("nan"), float("nan")
    _warn_nan_drop(samples.size, n_boot, "macro_stat_cluster")

    if not use_bca:
        lo, hi = _percentile_ci(samples, alpha)
        return point, lo, hi

    jackknife = np.empty(n_pat, dtype=float)
    base = np.ones(n_pat, dtype=np.int64)
    for k in range(n_pat):
        base[k] = 0
        jackknife[k] = _macro_stat_from_multiplicity(prepared, base, stat_fn, strict_classes)
        base[k] = 1
    jackknife = jackknife[~np.isnan(jackknife)]
    if jackknife.size == 0:
        lo, hi = _percentile_ci(samples, alpha)
        return point, lo, hi
    lo, hi = bca_ci(samples, point, jackknife, alpha=alpha)
    return point, lo, hi


def macro_f1_cluster_ci(
    per_class: Sequence[PerClassOOF], n_boot: int = 1000, alpha: float = 0.05,
    seed: int = 42, threshold: float = 0.5, strict_classes: bool = True,
) -> Tuple[float, float, float]:
    """Joint patient-clustered percentile CI for macro-F1 at threshold."""
    return _macro_stat_cluster_ci(per_class, _f1_at(threshold), n_boot=n_boot, alpha=alpha,
                                  seed=seed, use_bca=False, strict_classes=strict_classes)


def macro_auprc_cluster_ci(
    per_class: Sequence[PerClassOOF], n_boot: int = 1000, alpha: float = 0.05,
    seed: int = 42, strict_classes: bool = True,
) -> Tuple[float, float, float]:
    """Joint patient-clustered percentile CI for macro-AUPRC."""
    return _macro_stat_cluster_ci(per_class, _safe_ap, n_boot=n_boot, alpha=alpha,
                                  seed=seed, use_bca=False, strict_classes=strict_classes)


def paired_macro_auroc_cluster_diff(
    per_class_a: Sequence[PerClassOOF],
    per_class_b: Sequence[PerClassOOF],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float], float]:
    """Paired patient-clustered bootstrap on (macroAUROC_A - macroAUROC_B) -> (diff, (lo, hi), two_sided_p)."""
    if not per_class_a or not per_class_b:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    universe, code_of = _build_patient_code_map([per_class_a, per_class_b])
    n_pat = universe.size
    if n_pat == 0:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    prep_a = _prepare_per_class(per_class_a, code_of)
    prep_b = _prepare_per_class(per_class_b, code_of)

    full = np.ones(n_pat, dtype=np.int64)
    observed = _macro_auroc_from_multiplicity(prep_a, full) - _macro_auroc_from_multiplicity(prep_b, full)

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        mult = np.bincount(rng.integers(0, n_pat, size=n_pat), minlength=n_pat)
        diffs[b] = (
            _macro_auroc_from_multiplicity(prep_a, mult)
            - _macro_auroc_from_multiplicity(prep_b, mult)
        )
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return observed, (float("nan"), float("nan")), float("nan")
    lo, hi = _percentile_ci(diffs, alpha)
    p_lt = float((diffs < 0).mean())
    p_gt = float((diffs > 0).mean())
    p_two_sided = max(2.0 * min(p_lt, p_gt), 1.0 / n_boot)
    return observed, (lo, hi), p_two_sided


def multiclass_paired_bootstrap_diff(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: Optional[np.ndarray] = None,
) -> Tuple[float, Tuple[float, float], float]:
    """Multiclass sibling of paired_bootstrap_diff: paired bootstrap on OvR-macro-AUROC(A) - (B) over (N, C) softmaxes."""
    y_true = np.asarray(y_true).astype(int)
    ya = np.asarray(y_score_a, dtype=float)
    yb = np.asarray(y_score_b, dtype=float)
    n = y_true.size
    if not (ya.shape[0] == yb.shape[0] == n):
        raise ValueError("y_true, y_score_a, y_score_b must share the row count")
    if ya.ndim != 2 or yb.ndim != 2:
        raise ValueError("y_score_a / y_score_b must be 2D (N, C) softmax matrices")

    observed = _safe_multiclass_auroc_ovr(y_true, ya) - _safe_multiclass_auroc_ovr(y_true, yb)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    if groups is None:
        idx = _bootstrap_resample_idx(n, n_boot, rng)
        for b in range(n_boot):
            ii = idx[b]
            diffs[b] = (
                _safe_multiclass_auroc_ovr(y_true[ii], ya[ii])
                - _safe_multiclass_auroc_ovr(y_true[ii], yb[ii])
            )
    else:
        idx_list = _cluster_resample_idx(groups, n_boot, rng)
        for b in range(n_boot):
            ii = idx_list[b]
            diffs[b] = (
                _safe_multiclass_auroc_ovr(y_true[ii], ya[ii])
                - _safe_multiclass_auroc_ovr(y_true[ii], yb[ii])
            )
    diffs = diffs[~np.isnan(diffs)]
    if diffs.size == 0:
        return observed, (float("nan"), float("nan")), float("nan")
    lo, hi = _percentile_ci(diffs, alpha)
    p_lt = float((diffs < 0).mean())
    p_gt = float((diffs > 0).mean())
    p_two_sided = max(2.0 * min(p_lt, p_gt), 1.0 / n_boot)
    return observed, (lo, hi), p_two_sided


def inverse_variance_aggregate(
    points: Sequence[float],
    cis: Sequence[Tuple[float, float]],
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Fixed-effect inverse-variance combine of independent per-cell AUROCs -> (weighted_mean, lo, hi)."""
    z = float(st.norm.ppf(1 - alpha / 2))
    xs: List[float] = []
    ws: List[float] = []
    for x, ci in zip(points, cis):
        lo_i, hi_i = ci
        if not (np.isfinite(x) and np.isfinite(lo_i) and np.isfinite(hi_i)):
            continue
        half = (float(hi_i) - float(lo_i)) / 2.0
        if half <= 0.0:
            continue
        var = (half / z) ** 2
        if var <= 0.0:
            continue
        xs.append(float(x))
        ws.append(1.0 / var)
    if not xs:
        return float("nan"), float("nan"), float("nan")
    xs_a = np.asarray(xs)
    ws_a = np.asarray(ws)
    mean = float((ws_a * xs_a).sum() / ws_a.sum())
    se = float(np.sqrt(1.0 / ws_a.sum()))
    lo = float(np.clip(mean - z * se, 0.0, 1.0))
    hi = float(np.clip(mean + z * se, 0.0, 1.0))
    return mean, lo, hi


def macro_mean_across_cohorts(
    points: Sequence[float],
    cis: Optional[Sequence[Optional[Tuple[float, float]]]] = None,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Unweighted macro-mean of per-cohort AUROCs with a between-cohort t-interval CI -> (mean, lo, hi)."""
    if cis is None:
        cis = [None] * len(list(points))
    pairs = [(float(x), c) for x, c in zip(points, cis) if np.isfinite(x)]
    if not pairs:
        return float("nan"), float("nan"), float("nan")
    xs = [p[0] for p in pairs]
    k = len(xs)
    mean = float(np.mean(xs))
    if k == 1:
        c = pairs[0][1]
        if c is not None and np.isfinite(c[0]) and np.isfinite(c[1]):
            lo, hi = float(c[0]), float(c[1])
        else:
            lo = hi = mean
        return mean, float(np.clip(lo, 0.0, 1.0)), float(np.clip(hi, 0.0, 1.0))
    s = float(np.std(xs, ddof=1))
    h = float(st.t.ppf(1.0 - alpha / 2.0, k - 1)) * s / np.sqrt(k)
    return mean, float(np.clip(mean - h, 0.0, 1.0)), float(np.clip(mean + h, 0.0, 1.0))
