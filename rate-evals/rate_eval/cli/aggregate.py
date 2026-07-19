"""`rate-aggregate` CLI: project per-cell records to top-level CSV/JSON summaries."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..core.logging import get_logger
from ..io.result_schema import ResultSummary

logger = get_logger(__name__)


CV_SUMMARY_FIELDS: List[str] = [
    "teacher",
    "n_samples",
    "feature_dim",
    "head",
    "auroc",
    "auroc_lo",
    "auroc_hi",
    "auprc",
    "auprc_lo",
    "auprc_hi",
    "brier",
    "sens_at_95spec",
    "accuracy",
    "f1",
    "fold_mean_auroc",
    "fold_std_auroc",
]


def _row_from_summary(summary: ResultSummary, teacher: str) -> Dict[str, Any]:
    """Project a CV-mode `ResultSummary` into a flat row for `teacher_cv_summary__*.csv`."""
    pooled = summary.pooled
    fold_summary = summary.fold_summary or {}
    auroc_ci = summary.pooled_auroc_ci or [None, None]
    auprc_ci = summary.pooled_auprc_ci or [None, None]
    fold_auroc = fold_summary.get("auroc", [None, None])
    return {
        "teacher": teacher,
        "n_samples": summary.n_samples,
        "feature_dim": summary.feature_dim,
        "head": summary.head_spec.kind if summary.head_spec else None,
        "auroc": pooled.auroc if pooled else None,
        "auroc_lo": auroc_ci[0] if len(auroc_ci) >= 1 else None,
        "auroc_hi": auroc_ci[1] if len(auroc_ci) >= 2 else None,
        "auprc": pooled.auprc if pooled else None,
        "auprc_lo": auprc_ci[0] if len(auprc_ci) >= 1 else None,
        "auprc_hi": auprc_ci[1] if len(auprc_ci) >= 2 else None,
        "brier": pooled.brier if pooled else None,
        "sens_at_95spec": pooled.sens_at_95spec if pooled else None,
        "accuracy": pooled.accuracy if pooled else None,
        "f1": pooled.f1 if pooled else None,
        "fold_mean_auroc": fold_auroc[0] if isinstance(fold_auroc, list) and len(fold_auroc) >= 1 else None,
        "fold_std_auroc": fold_auroc[1] if isinstance(fold_auroc, list) and len(fold_auroc) >= 2 else None,
    }


def _missing_row(teacher: str) -> Dict[str, Any]:
    """Row for a teacher whose `summary.json` couldn't be read."""
    row: Dict[str, Any] = {field: None for field in CV_SUMMARY_FIELDS}
    row["teacher"] = teacher
    row["_status"] = "missing"
    return row


def collect_cv_summaries(
    results_root: Path,
    teachers: Iterable[str],
    suffix: str = "_lidc",
    out_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Read one `summary.json` per teacher under `<results_root>/<teacher><suffix>/` -> CSV-shaped rows (optionally written to `out_path`)."""
    rows: List[Dict[str, Any]] = []
    for teacher in teachers:
        summary_path = results_root / f"{teacher}{suffix}" / "summary.json"
        if not summary_path.exists():
            logger.warning(f"missing summary: {summary_path}")
            rows.append(_missing_row(teacher))
            continue
        try:
            summary = ResultSummary.from_json(summary_path)
        except Exception as exc:
            logger.warning(f"failed to parse {summary_path}: {exc}")
            rows.append(_missing_row(teacher))
            continue
        rows.append(_row_from_summary(summary, teacher))

    if not rows:
        raise SystemExit("no rows to write")

    if out_path is not None:
        _write_csv(rows, out_path)
        logger.info(f"wrote {out_path}")
    return rows


def _write_csv(rows: List[Dict[str, Any]], out_path: Path) -> None:
    """Write rows to CSV; field set = first row's keys."""
    fields = list(rows[0].keys())
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _add_cv_summaries_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "cv-summaries",
        help="Per-teacher pooled-OOF row from <root>/<teacher><suffix>/summary.json",
        description=(
            "Reads one summary.json per teacher under <results-root>/<teacher><suffix>/ "
            "and emits a wide CSV (one row per teacher)."
        ),
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--teachers", type=str, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--suffix",
        type=str,
        default="_lidc",
        help="Directory suffix appended to each teacher (e.g. '_lidc' or '_lidc__v2_max_ge4')",
    )
    parser.set_defaults(_func=_cmd_cv_summaries)


def _cmd_cv_summaries(args: argparse.Namespace) -> int:
    collect_cv_summaries(
        results_root=args.results_root,
        teachers=args.teachers,
        suffix=args.suffix,
        out_path=args.out,
    )
    return 0


def aggregate_cli() -> int:
    """Entry point for the `rate-aggregate` console script."""
    parser = argparse.ArgumentParser(
        prog="rate-aggregate",
        description="Project rate-eval ResultSummary records into top-level CSVs.",
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)
    _add_cv_summaries_parser(subparsers)
    args = parser.parse_args()
    return args._func(args)


def main() -> None:
    raise SystemExit(aggregate_cli())


if __name__ == "__main__":
    main()
