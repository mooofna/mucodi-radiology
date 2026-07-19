"""Disk-state detection for the launcher's --continue flow."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, TYPE_CHECKING

if TYPE_CHECKING:
    from experiments.config import EvalConfig, Experiment


class PhaseDetector(Protocol):
    """Decides whether each phase of an experiment is already complete on disk."""

    def train_done(self, exp: "Experiment") -> bool: ...
    def extract_done(self, exp: "Experiment", eval_cfg: "EvalConfig") -> bool: ...
    def evaluate_done(self, exp: "Experiment", eval_cfg: "EvalConfig") -> bool: ...
    def aggregate_done(self, exp: "Experiment") -> bool: ...


class RadiologyPhaseDetector:
    """Reads rate-evals canonical artefacts to decide per-phase completeness."""

    def train_done(self, exp: "Experiment") -> bool:
        ckpt = exp.checkpoint_file()
        return ckpt is not None and ckpt.exists()

    def extract_done(self, exp: "Experiment", eval_cfg: "EvalConfig") -> bool:
        """True iff cache_meta.yaml exists, has finished_utc, and wrapper/dataset match."""
        if not eval_cfg.cache_dir:
            return True   # no cache prerequisite
        cache_dir = Path(eval_cfg.cache_dir)
        try:
            from rate_eval.io.cache_meta import read_cache_meta
        except ImportError:
            return _legacy_extract_done(cache_dir)
        meta = read_cache_meta(cache_dir)
        if meta is None:
            return _legacy_extract_done(cache_dir)
        # accept both dataset-name forms so --continue reads old + new caches
        accepted_names = {
            eval_cfg.dataset,
            f"{eval_cfg.dataset}/{eval_cfg.wrapper}",
        }
        if meta.wrapper.name != eval_cfg.wrapper or meta.dataset.name not in accepted_names:
            return False
        finished = meta.extraction.model_dump().get("finished_utc")
        if finished is None:
            return _legacy_extract_done(cache_dir)
        return True

    def evaluate_done(self, exp: "Experiment", eval_cfg: "EvalConfig") -> bool:
        """True iff the per-protocol output artefact exists and carries a numeric AUROC."""
        import json
        import math

        def _numeric_macro_auroc(path: Path) -> bool:
            if not path.exists():
                return False
            try:
                v = json.loads(path.read_text()).get("macro_auroc")
                return isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v))
            except Exception:
                return False

        if eval_cfg.protocol == "cv_5fold_macro":
            return _numeric_macro_auroc(Path(eval_cfg.output_dir) / "macro_summary.json")
        if eval_cfg.protocol == "crossval_deploy":
            # done only when both internal + external macros present
            base = Path(eval_cfg.output_dir)
            return _numeric_macro_auroc(base / "internal" / "macro_summary.json") and _numeric_macro_auroc(
                base / "external" / "macro_summary.json"
            )
        summary = Path(eval_cfg.output_dir) / "summary.json"
        if not summary.exists():
            return False
        try:
            d = json.loads(summary.read_text())
            # accept pooled.auroc / pooled.auroc_ovr / top-level auroc
            pooled = d.get("pooled") if isinstance(d.get("pooled"), dict) else {}
            auroc = pooled.get("auroc") or pooled.get("auroc_ovr") or d.get("auroc")
            return (
                isinstance(auroc, (int, float))
                and not (isinstance(auroc, float) and math.isnan(auroc))
            )
        except Exception:
            return False

    def aggregate_done(self, exp: "Experiment") -> bool:
        """True iff the cv-summaries CSV exists, or there is nothing to aggregate."""
        cv_cells = [e for e in exp.evaluations if e.protocol == "cv_5fold"]
        if not cv_cells:
            return True
        return (exp.run_dir() / "results_cv.csv").exists()


def _legacy_extract_done(cache_dir: Path) -> bool:
    """Fallback for caches without cache_meta.yaml: does embeddings/ contain NPZs?"""
    emb_dir = cache_dir / "embeddings"
    if not emb_dir.is_dir():
        return False
    for split_dir in emb_dir.iterdir():
        if split_dir.is_dir() and any(split_dir.glob("*.npz")):
            return True
    return False
