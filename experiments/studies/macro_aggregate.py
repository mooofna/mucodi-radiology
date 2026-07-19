"""Patient-clustered macro-AUROC aggregator for cv_5fold_macro cells."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

# torch-free import (numpy/scipy/sklearn only)
from rate_eval.evaluation.stats import (
    macro_auprc_cluster_ci,
    macro_auroc_cluster_ci,
    macro_f1_cluster_ci,
)


def _load_class_oof(class_dir: Path):
    """Read one class's predictions_oof.json -> (y_true, y_score, patient_ids) in stable order."""
    f = class_dir / "predictions_oof.json"
    if not f.exists():
        return None
    rows = json.loads(f.read_text())
    if not rows:
        return None
    # stable sort -> deterministic patient universe
    rows.sort(key=lambda r: (str(r["patient_id"]), int(r.get("fold", 0))))
    y = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    s = np.array([float(r["score"]) for r in rows], dtype=float)
    pids = np.array([str(r["patient_id"]) for r in rows])
    return y, s, pids


def _class_point_and_ci(class_dir: Path):
    """Per-class point AUROC + its cluster CI (falls back to row CI) from summary.json."""
    f = class_dir / "summary.json"
    if not f.exists():
        return None, [None, None]
    d = json.loads(f.read_text())
    auroc = (d.get("pooled") or {}).get("auroc")
    ci = d.get("pooled_auroc_ci_cluster") or d.get("pooled_auroc_ci") or [None, None]
    return auroc, ci


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Patient-clustered macro-AUROC aggregator.")
    ap.add_argument("--per-class-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--wrapper", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--cohort", default=None)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--expected-classes", type=int, default=16,
                    help="A6: hard-fail unless exactly this many classes are aggregated (RAD-ChestCT macro = 16).")
    args = ap.parse_args(argv)

    per_class: list = []
    per_class_report: dict = {}
    for cd in sorted(args.per_class_dir.iterdir()):
        if not cd.is_dir():
            continue
        oof = _load_class_oof(cd)
        auroc, ci = _class_point_and_ci(cd)
        undefined = auroc is None or (isinstance(auroc, float) and math.isnan(auroc))
        if oof is None or undefined:
            per_class_report[cd.name] = {
                "auroc": None, "auroc_ci_cluster": [None, None],
                "skipped": "undefined AUROC (n_pos=0 / NaN)",
            }
            continue
        per_class.append(oof)
        per_class_report[cd.name] = {"auroc": auroc, "auroc_ci_cluster": ci}

    # hard-fail unless exactly expected_classes (glob-driven dir set)
    _n_found, _n_used = len(per_class_report), len(per_class)
    if _n_found != args.expected_classes or _n_used != args.expected_classes:
        _skipped = [c for c, r in per_class_report.items() if r.get("auroc") is None]
        raise SystemExit(
            f"macro_aggregate: expected exactly {args.expected_classes} classes, found "
            f"{_n_found} per-class dirs and used {_n_used} (skipped/undefined: {_skipped}). "
            f"Refusing to write a non-comparable macro. Dirs: {sorted(per_class_report)}"
        )

    if per_class:
        macro, lo, hi = macro_auroc_cluster_ci(
            per_class, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
        )
        # macro-F1 (thr 0.5) + macro-AUPRC over the same OOF + joint resample
        macro_f1, f1_lo, f1_hi = macro_f1_cluster_ci(
            per_class, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
        )
        macro_auprc, ap_lo, ap_hi = macro_auprc_cluster_ci(
            per_class, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
        )
        n_patients = int(np.unique(np.concatenate([p for (_, _, p) in per_class])).size)
    else:
        macro, lo, hi, n_patients = None, None, None, 0
        macro_f1, f1_lo, f1_hi = None, None, None
        macro_auprc, ap_lo, ap_hi = None, None, None

    out = {
        "wrapper": args.wrapper,
        "dataset": args.dataset,
        "cohort": args.cohort,
        "protocol": "cv_5fold_macro",
        "n_classes_used": len(per_class),
        "n_classes_total": len(per_class_report),
        "macro_auroc": macro,
        "macro_auroc_ci_cluster": [lo, hi],
        "macro_f1": macro_f1,
        "macro_f1_ci_cluster": [f1_lo, f1_hi],
        "macro_auprc": macro_auprc,
        "macro_auprc_ci_cluster": [ap_lo, ap_hi],
        "ci_resample_unit": "patient_cluster",
        "n_patients": n_patients,
        "per_class": per_class_report,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=float))
    if macro is not None:
        print(
            f"Macro AUROC ({len(per_class)}/{len(per_class_report)} classes): "
            f"{macro:.4f} [{lo:.4f} {hi:.4f}] (patient-cluster BCa, n_patients={n_patients})"
        )
    else:
        print("Macro AUROC: no usable classes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
