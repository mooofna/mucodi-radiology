"""Per-cell `cache_meta.yaml` provenance sidecar for `cache/<wrapper>_<dataset>/`."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field


CACHE_META_FILENAME = "cache_meta.yaml"
SCHEMA_VERSION = "1.0"


class SplitInfo(BaseModel):
    model_config = ConfigDict(extra="allow")

    path: str
    sha256: Optional[str] = None
    n_rows: Optional[int] = None


class WrapperProvenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    config_yaml_path: Optional[str] = None
    config_yaml_sha256: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    upstream_git_sha: Optional[str] = None


class DatasetProvenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str
    config_yaml_path: Optional[str] = None
    config_yaml_sha256: Optional[str] = None
    splits_used: Dict[str, SplitInfo] = Field(default_factory=dict)


class PreprocessProvenance(BaseModel):
    """Mirrors the YAML `loader.preprocess:` block."""

    model_config = ConfigDict(extra="allow")

    pipeline: Optional[str] = None
    target_shape: Optional[List[int]] = None
    hu_clip: Optional[List[float]] = None
    output_norm: Optional[str] = None
    mask_aware: Optional[bool] = None
    legacy_class: Optional[str] = None


class ExtractionProvenance(BaseModel):
    model_config = ConfigDict(extra="allow")

    rate_eval_version: Optional[str] = None
    rate_eval_git_sha: Optional[str] = None
    started_utc: Optional[str] = None
    finished_utc: Optional[str] = None
    num_gpus: Optional[int] = None
    batch_size_per_gpu: Optional[int] = None
    num_samples: Dict[str, int] = Field(default_factory=dict)


class CacheMeta(BaseModel):
    """Top-level provenance record; one file per cache cell."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = SCHEMA_VERSION
    wrapper: WrapperProvenance
    dataset: DatasetProvenance
    preprocess: PreprocessProvenance = Field(default_factory=PreprocessProvenance)
    extraction: ExtractionProvenance = Field(default_factory=ExtractionProvenance)
    notes: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "CacheMeta":
        raw = yaml.safe_load(Path(path).read_text())
        return cls.model_validate(raw)

    def write(self, cache_dir: Union[str, Path]) -> Path:
        out_path = Path(cache_dir) / CACHE_META_FILENAME
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.model_dump(exclude_none=True)
        out_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))
        return out_path


def read_cache_meta(cache_dir: Union[str, Path]) -> Optional[CacheMeta]:
    """Return the parsed `cache_meta.yaml` from `cache_dir`, or None if absent."""
    path = Path(cache_dir) / CACHE_META_FILENAME
    if not path.exists():
        return None
    return CacheMeta.from_yaml(path)


def write_cache_meta_start(
    cache_dir: Union[str, Path],
    *,
    wrapper_name: str,
    dataset_name: str,
    wrapper_config_path: Optional[Union[str, Path]] = None,
    dataset_config_path: Optional[Union[str, Path]] = None,
    preprocess: Optional[Dict[str, Any]] = None,
    rate_eval_version: Optional[str] = None,
    num_gpus: Optional[int] = None,
    batch_size_per_gpu: Optional[int] = None,
    notes: Optional[str] = None,
) -> CacheMeta:
    """Write the initial cache_meta.yaml at extraction start."""
    wrapper_provenance = WrapperProvenance(
        name=wrapper_name,
        config_yaml_path=str(wrapper_config_path) if wrapper_config_path else None,
        config_yaml_sha256=_sha256_of_file(wrapper_config_path),
    )
    dataset_provenance = DatasetProvenance(
        name=dataset_name,
        config_yaml_path=str(dataset_config_path) if dataset_config_path else None,
        config_yaml_sha256=_sha256_of_file(dataset_config_path),
    )
    preprocess_provenance = PreprocessProvenance(**(preprocess or {}))
    extraction_provenance = ExtractionProvenance(
        rate_eval_version=rate_eval_version,
        rate_eval_git_sha=_git_sha_short(),
        started_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        num_gpus=num_gpus,
        batch_size_per_gpu=batch_size_per_gpu,
    )
    meta = CacheMeta(
        wrapper=wrapper_provenance,
        dataset=dataset_provenance,
        preprocess=preprocess_provenance,
        extraction=extraction_provenance,
        notes=notes,
    )
    meta.write(cache_dir)
    return meta


def update_cache_meta_finish(
    cache_dir: Union[str, Path],
    *,
    n_samples_per_split: Dict[str, int],
) -> Optional[CacheMeta]:
    """Append per-split sample counts + finished_utc to an existing cache_meta.yaml (no-op if absent)."""
    meta = read_cache_meta(cache_dir)
    if meta is None:
        return None
    meta.extraction.finished_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # merge so mid-run per-split counts aren't clobbered
    merged = dict(meta.extraction.num_samples)
    merged.update(n_samples_per_split)
    meta.extraction.num_samples = merged
    meta.write(cache_dir)
    return meta


def check_dataset_config_drift(
    cache_dir: Union[str, Path], current_dataset_config_path: Union[str, Path]
) -> Optional[str]:
    """Warn if the cache's recorded dataset-config sha differs from the current YAML's; None when matched or absent."""
    meta = read_cache_meta(cache_dir)
    if meta is None:
        return None
    recorded = meta.dataset.config_yaml_sha256
    current = _sha256_of_file(current_dataset_config_path)
    if not recorded or not current or recorded == current:
        return None
    return (
        f"cache_meta drift: cache was extracted from {meta.dataset.config_yaml_path!r} "
        f"sha={recorded[:12]}; current {current_dataset_config_path!r} sha={current[:12]}. "
        "Re-extract or override the warning with RATE_SKIP_CACHE_META_CHECK=1."
    )


def _sha256_of_file(path: Optional[Union[str, Path]]) -> Optional[str]:
    """Stream-hash a file. Returns None if the path is missing or unreadable."""
    if path is None:
        return None
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while True:
            chunk = fh.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_sha_short() -> Optional[str]:
    """Return the short SHA of the rate-evals repo HEAD, or None if git is unavailable."""
    try:
        import subprocess

        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:  # noqa: BLE001
        return None


__all__ = [
    "CACHE_META_FILENAME",
    "CacheMeta",
    "DatasetProvenance",
    "ExtractionProvenance",
    "PreprocessProvenance",
    "SCHEMA_VERSION",
    "SplitInfo",
    "WrapperProvenance",
    "check_dataset_config_drift",
    "read_cache_meta",
    "update_cache_meta_finish",
    "write_cache_meta_start",
]
