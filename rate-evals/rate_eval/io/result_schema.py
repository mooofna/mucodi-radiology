"""Canonical Pydantic schema for rate-evaluate summary.json outputs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field


class HeadSpec(BaseModel):
    """Classifier head configuration recorded with the summary."""

    model_config = ConfigDict(extra="allow")

    kind: str  # "linear" | "mlp" | "zero_shot"
    dim_hidden: Optional[int] = None
    dropout: Optional[float] = None
    hidden_layers: Optional[List[int]] = None
    # zero-shot-only
    teacher: Optional[str] = None
    prompt_pos: Optional[str] = None
    prompt_neg: Optional[str] = None


class MetricBlock(BaseModel):
    """Single-cell metric record; mirrors CV mode's `pooled`."""

    model_config = ConfigDict(extra="allow")

    auroc: float
    auprc: Optional[float] = None
    brier: Optional[float] = None
    accuracy: Optional[float] = None
    f1: Optional[float] = None
    sens_at_95spec: Optional[float] = None
    sens_at_95spec_threshold: Optional[float] = None
    n_pos: Optional[int] = None
    n_neg: Optional[int] = None


class _CVFields(BaseModel):
    """Fields populated when `protocol == "cv"`."""

    model_config = ConfigDict(extra="allow")

    cv_folds: int
    cv_seed: int
    pooled: MetricBlock
    pooled_auroc_ci: List[float] = Field(min_length=2, max_length=2)
    pooled_auprc_ci: Optional[List[float]] = None
    fold_summary: Optional[Dict[str, List[float]]] = None  # metric -> [mean, std]
    per_fold: Optional[List[Dict[str, Any]]] = None
    training_stats: Optional[List[Dict[str, Any]]] = None


class ResultSummary(BaseModel):
    """Canonical record, one file per (teacher x dataset x label x protocol) cell."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "1.0"
    protocol: Literal["cv", "single_split"]

    head_spec: HeadSpec
    feature_dim: int

    teacher: Optional[str] = None
    dataset: Optional[str] = None
    rate_eval_version: Optional[str] = None
    timestamp: Optional[str] = None  # ISO-8601
    cache_meta: Optional[Dict[str, Any]] = None

    n_samples: Optional[int] = None

    # CV-mode fields (kept flat for back-compat)
    cv_folds: Optional[int] = None
    cv_seed: Optional[int] = None
    pooled: Optional[MetricBlock] = None
    pooled_auroc_ci: Optional[List[float]] = None
    pooled_auprc_ci: Optional[List[float]] = None
    fold_summary: Optional[Dict[str, List[float]]] = None
    per_fold: Optional[List[Dict[str, Any]]] = None
    training_stats: Optional[List[Dict[str, Any]]] = None

    # patient-cluster bootstrap CIs
    pooled_auroc_ci_cluster: Optional[List[float]] = None
    auroc_ci_cluster: Optional[List[float]] = None
    ci_resample_unit: Optional[Literal["row", "patient_cluster"]] = None
    n_patients: Optional[int] = None

    # probe provenance
    feature_l2_normalized: Optional[bool] = None
    weight_decay_effective: Optional[float] = None
    group_id: Optional[str] = None

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ResultSummary":
        """Parse a summary.json, inferring `schema_version`/`protocol` for legacy files."""
        path = Path(path)
        raw = json.loads(path.read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ResultSummary":
        """Parse from a dict, with legacy-shape inference for `schema_version`/`protocol`."""
        if "schema_version" not in raw:
            raw["schema_version"] = "0.9"
        if "protocol" not in raw:
            raw["protocol"] = _infer_protocol(raw)
        return cls.model_validate(raw)

    def write(self, out_dir: Union[str, Path], filename: str = "summary.json") -> Path:
        """Write the summary as indent=2 JSON for human diffability."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / filename
        payload = self.model_dump(exclude_none=True)
        payload.setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        path.write_text(json.dumps(payload, indent=2, sort_keys=False))
        return path


def _infer_protocol(raw: Dict[str, Any]) -> str:
    """Infer protocol from which keys are present."""
    has_cv = "cv_folds" in raw or "pooled" in raw or "per_fold" in raw
    return "cv" if has_cv else "single_split"


__all__ = ["HeadSpec", "MetricBlock", "ResultSummary"]
