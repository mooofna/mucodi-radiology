"""Pure-function extraction orchestrator: `ExtractRequest` -> `gpu_fanout.run_fanout` per split."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from ..core.logging import get_logger
from ..io.cache import SimpleCheckpointManager
from ..io.cache_meta import update_cache_meta_finish, write_cache_meta_start

logger = get_logger(__name__)


_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _resolve_yaml_path(subdir: str, name: str) -> Optional[Path]:
    """Look up a YAML in configs/<subdir>/; None if absent."""
    candidate = _PROJECT_ROOT / "configs" / subdir / f"{name}.yaml"
    return candidate if candidate.exists() else None


def _extract_preprocess_provenance(config: DictConfig) -> Dict[str, Any]:
    """Flat preprocess-provenance dict from the live config."""
    out: Dict[str, Any] = {}
    loader = OmegaConf.select(config, "loader", default=None)
    if loader is None:
        return out
    preprocess = OmegaConf.select(loader, "preprocess", default=None)
    if preprocess is not None:
        # OmegaConf -> primitives so Pydantic + yaml.safe_dump can roundtrip.
        out.update(OmegaConf.to_container(preprocess, resolve=True))
    legacy_class = OmegaConf.select(loader, "class", default=None)
    if legacy_class is not None and preprocess is None:
        out["legacy_class"] = legacy_class
    return out


def _rate_eval_version() -> str:
    try:
        from .. import __version__

        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


@dataclass
class ExtractRequest:
    """Typed input for `pipelines.extract.run`."""

    model: str
    dataset: str
    splits: List[str]
    output_dir: Path
    config: DictConfig

    max_samples: Optional[int] = None
    skip_existing_cache: bool = False
    continue_on_error: bool = False
    check_nan: bool = False
    debug: bool = False

    num_gpus: Optional[int] = None
    batch_size: Optional[int] = None
    num_workers: Optional[int] = None
    device: Optional[str] = None


@dataclass
class ExtractResult:
    """Typed output from `pipelines.extract.run`."""

    per_split: Dict[str, "SplitResult"] = field(default_factory=dict)


@dataclass
class SplitResult:
    """Per-split extraction outcome."""

    split: str
    n_processed: int
    cache_dir: Path
    elapsed_seconds: float
    throughput_samples_per_second: float


def run(request: ExtractRequest) -> ExtractResult:
    """Run extraction for every split in `request.splits` -> one `SplitResult` each."""
    # lazy import to avoid a circular import
    from . import gpu_fanout

    result = ExtractResult()

    # provenance sidecar; non-fatal on failure
    try:
        write_cache_meta_start(
            request.output_dir,
            wrapper_name=request.model,
            dataset_name=request.dataset,
            wrapper_config_path=_resolve_yaml_path("model", request.model),
            dataset_config_path=_resolve_yaml_path("dataset", request.dataset),
            preprocess=_extract_preprocess_provenance(request.config),
            rate_eval_version=_rate_eval_version(),
            num_gpus=request.num_gpus,
            batch_size_per_gpu=request.batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"cache_meta sidecar write failed (non-fatal): {exc}")

    failed_splits: list[str] = []
    for split in request.splits:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'=' * 60}")

        checkpoint_manager = SimpleCheckpointManager(
            model_name=request.model,
            dataset_name=request.dataset,
            split=split,
            cache_dir=str(request.output_dir),
            skip_existing_cache=request.skip_existing_cache,
        )

        stats = checkpoint_manager.get_stats()
        logger.info(f"Resume stats for {split}: {stats}")
        if stats["has_processed_samples"]:
            logger.info(
                f"Resuming {split} extraction: {stats['total_processed']} samples already processed"
            )
        elif stats.get("ignored_processed"):
            logger.info(
                "Skip-existing-cache enabled: ignoring %d cached samples for this run",
                stats["ignored_processed"],
            )
        else:
            logger.info(
                f"No previous processed samples found for {split}, starting fresh extraction"
            )

        start_time = time.time()
        try:
            embeddings, accessions = gpu_fanout.run_fanout(
                request=request,
                split=split,
                checkpoint_manager=checkpoint_manager,
            )
        except Exception as exc:
            logger.error(f"Failed to extract embeddings for split {split}: {exc}")
            result.per_split[split] = SplitResult(
                split=split,
                n_processed=0,
                cache_dir=request.output_dir,
                elapsed_seconds=time.time() - start_time,
                throughput_samples_per_second=0.0,
            )
            failed_splits.append(split)
            continue
        elapsed = time.time() - start_time

        # workers wrote processed.csv independently; resync
        checkpoint_manager.refresh_from_disk()
        stats = checkpoint_manager.get_stats()

        throughput = stats["total_processed"] / elapsed if elapsed > 0 else 0.0

        logger.info(f"Extraction completed for {split} in {elapsed:.2f} seconds")
        logger.info(f"Total samples processed: {stats['total_processed']}")
        logger.info(f"Embeddings saved to checkpoint directory: {request.output_dir}")
        if elapsed > 0:
            logger.info(f"Throughput: {throughput:.2f} samples/sec")
        if stats["total_processed"] == 0:
            logger.error(f"No embeddings were extracted for split {split}!")
            failed_splits.append(split)

        result.per_split[split] = SplitResult(
            split=split,
            n_processed=stats["total_processed"],
            cache_dir=request.output_dir,
            elapsed_seconds=elapsed,
            throughput_samples_per_second=throughput,
        )

        # drop arrays; the .npz files are canonical
        del embeddings, accessions  # noqa: F841

    try:
        update_cache_meta_finish(
            request.output_dir,
            n_samples_per_split={
                split: split_result.n_processed
                for split, split_result in result.per_split.items()
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"cache_meta sidecar update failed (non-fatal): {exc}")

    # hard-fail if any split errored or wrote 0 embeddings
    if failed_splits:
        raise RuntimeError(
            f"extraction FAILED for split(s) {failed_splits}: errors above / 0 embeddings written. "
            f"Refusing to exit COMPLETED -- a downstream eval on incomplete features would be silently wrong."
        )

    logger.info(f"\n{'=' * 60}")
    logger.info("All splits processing completed!")
    logger.info(f"{'=' * 60}")

    return result


__all__ = ["ExtractRequest", "ExtractResult", "SplitResult", "run"]
