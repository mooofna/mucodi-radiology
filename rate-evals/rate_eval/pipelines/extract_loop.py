"""Per-GPU batched extraction loop (continue-on-error, NaN gating, subset slicing, per-sample checkpointing)."""

from __future__ import annotations

import time
import traceback
from contextlib import contextmanager
from typing import List, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from ..components import create_dataset, create_model
from ..core.logging import get_logger, setup_logging
from ..io.cache import SimpleCheckpointManager, SimpleResumableDataset
from ..models.preprocess_adapter import ModelPreprocessor
from .extract import ExtractRequest

logger = get_logger(__name__)


@contextmanager
def _timed(rank: int, batch_idx: Optional[int], label: str):
    """Tiny DEBUG-level timing helper."""
    start = time.time()
    try:
        yield
    finally:
        elapsed = time.time() - start
        if batch_idx is None:
            logger.debug(f"GPU {rank}: {label} took {elapsed:.2f}s")
        else:
            logger.debug(f"GPU {rank} Batch {batch_idx}: {label} took {elapsed:.2f}s")


def _validate_volume_shape(
    volumes: torch.Tensor, rank: int, batch_idx: int, continue_on_error: bool
) -> bool:
    """True to process, False to skip, raise on hard fail. 4D/5D/6D accepted; else rejected (or skipped under `continue_on_error`)."""
    if volumes.shape[0] == 0:
        logger.warning(f"GPU {rank}: Batch {batch_idx} - Empty batch received, skipping")
        return False

    logger.debug(f"GPU {rank}: Batch {batch_idx} - Raw volume shape: {volumes.shape}")
    if len(volumes.shape) < 4:
        logger.error(
            f"GPU {rank}: Batch {batch_idx} - Invalid volume shape {volumes.shape}, expected at least 4D"
        )
        if continue_on_error:
            logger.warning("!  CONTINUING after invalid shape (--continue-on-error enabled)")
            return False
        raise ValueError(f"Invalid volume shape: {volumes.shape}")

    if len(volumes.shape) == 4:
        logger.debug(f"GPU {rank}: Batch {batch_idx} - 2D input format")
    elif len(volumes.shape) == 5:
        logger.debug(f"GPU {rank}: Batch {batch_idx} - 3D volume format")
    elif len(volumes.shape) == 6:
        logger.debug(f"GPU {rank}: Batch {batch_idx} - Multiple 3D volumes format")
    else:
        logger.error(
            f"GPU {rank}: Batch {batch_idx} - Unexpected volume shape: {volumes.shape}"
        )
        if continue_on_error:
            logger.warning("!  CONTINUING after unexpected shape (--continue-on-error enabled)")
            return False
        raise ValueError(f"Unexpected volume shape: {volumes.shape}")
    return True


def _check_input_nan(
    volumes: torch.Tensor,
    rank: int,
    batch_idx: int,
    accessions: List[str],
    continue_on_error: bool,
) -> None:
    """Detect NaNs in the input tensor; raise (or warn) per `continue_on_error`."""
    if volumes.numel() == 0:
        return
    nan_count = torch.isnan(volumes).sum().item()
    logger.debug(
        f"GPU {rank}: Batch {batch_idx} - Input NaN check: {nan_count} NaN values found"
    )
    if nan_count == 0:
        return

    logger.error(
        f" INPUT NaN DETECTED: {nan_count} NaN values found in input batch {batch_idx} on GPU {rank}"
    )
    logger.error(f"Input batch shape: {volumes.shape}")
    logger.error(f"Problematic samples in this batch: {accessions}")
    if continue_on_error:
        logger.warning("!  CONTINUING despite INPUT NaN (--continue-on-error enabled)")
        logger.warning("This may result in corrupted embeddings being saved!")
        return
    raise RuntimeError(
        f"INPUT NaN DETECTED: Found {nan_count} NaN values in input data at batch {batch_idx}. "
        f"Problematic samples: {accessions}"
    )


def _check_output_nan(
    embeddings: np.ndarray,
    rank: int,
    batch_idx: int,
    accessions: List[str],
    continue_on_error: bool,
) -> None:
    """Detect NaNs in model output; raise (or warn) per `continue_on_error`."""
    nan_count = np.isnan(embeddings).sum()
    logger.debug(
        f"GPU {rank}: Batch {batch_idx} - NaN check: {nan_count} out of {embeddings.size} elements are NaN"
    )
    if nan_count == 0:
        return

    logger.error(
        f" MODEL OUTPUT NaN DETECTED: {nan_count} NaN values in embeddings at batch {batch_idx} on GPU {rank}"
    )
    logger.error("! INPUT WAS CLEAN - NaN originated from MODEL PROCESSING")
    logger.error(f"Output batch shape: {embeddings.shape}")
    logger.error(f"Affected samples in this batch: {accessions}")
    if continue_on_error:
        logger.warning("!  CONTINUING despite MODEL NaN (--continue-on-error enabled)")
        logger.warning("This may result in corrupted embeddings being saved!")
        return
    raise RuntimeError(
        f"MODEL NaN DETECTED: Model generated {nan_count} NaN values in embeddings. "
        f"Input was clean - issue is in model processing."
    )


def _compute_subset_indices(
    dataset_len: int,
    rank: int,
    world_size: int,
    max_samples: Optional[int],
) -> List[int]:
    """Subset slicing for this rank: contiguous per-rank slices, remainder absorbed by the last rank."""
    if world_size <= 1:
        indices = list(range(dataset_len))
        if max_samples is not None and max_samples < dataset_len:
            indices = indices[:max_samples]
        return indices

    if max_samples is not None:
        total = min(dataset_len, max_samples)
    else:
        total = dataset_len
    samples_per_gpu = total // world_size
    start_idx = rank * samples_per_gpu
    end_idx = total if rank == world_size - 1 else start_idx + samples_per_gpu
    if start_idx >= total:
        return []
    return list(range(start_idx, min(end_idx, dataset_len)))


def run_single_gpu(
    rank: int,
    world_size: int,
    request: ExtractRequest,
    split: str,
    checkpoint_manager: Optional[SimpleCheckpointManager] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Run extraction on one GPU rank for one split; saves per-sample to checkpoint_manager."""
    try:
        # mp.spawn loses the parent's log handlers
        if world_size > 1:
            log_level = "DEBUG" if request.debug else "INFO"
            setup_logging(level=log_level, rank=rank, world_size=world_size, debug=request.debug)

        if world_size > 1:
            device = f"cuda:{rank}"
            torch.cuda.set_device(rank)
        else:
            device = request.device or getattr(
                getattr(request.config, "hardware", None), "device", "cuda:0"
            )

        local_config = OmegaConf.create(OmegaConf.to_container(request.config, resolve=True))
        OmegaConf.update(local_config, "hardware.device", device)

        logger.info(f"GPU {rank}: Loading model {request.model} on {device}")

        with _timed(rank, None, "Model loading"):
            model = create_model(request.model, local_config)

        with _timed(rank, None, "Dataset creation"):
            if hasattr(model, "preprocess_single") and hasattr(model, "model_config"):
                model_preprocess = ModelPreprocessor(model.__class__, model.model_config)
            else:
                model_preprocess = None
            original_dataset = create_dataset(
                request.dataset,
                local_config,
                split,
                model_preprocess=model_preprocess,
            )

        modality = getattr(
            getattr(request.config, "dataset", None),
            "modality",
            original_dataset.modality,
        )
        if (
            hasattr(request.config, "dataset")
            and hasattr(request.config.dataset, "modality")
            and request.config.dataset.modality != original_dataset.modality
        ):
            logger.info(
                f"GPU {rank}: Overriding dataset modality "
                f"'{original_dataset.modality}' with '{modality}'"
            )
        original_dataset.modality = modality

        if checkpoint_manager is not None:
            dataset = SimpleResumableDataset(original_dataset, checkpoint_manager)
            logger.info(f"GPU {rank}: Resume mode: {len(dataset)} unprocessed samples remaining")
        else:
            dataset = original_dataset

        subset_indices = _compute_subset_indices(
            dataset_len=len(dataset),
            rank=rank,
            world_size=world_size,
            max_samples=request.max_samples,
        )
        if not subset_indices:
            logger.info(f"GPU {rank}: No samples to process")
            return np.array([]), []
        subset = Subset(dataset, subset_indices) if world_size > 1 or (
            request.max_samples is not None and len(subset_indices) < len(dataset)
        ) else dataset

        if world_size > 1:
            logger.info(
                f"GPU {rank}: Processing {len(subset)} samples "
                f"(indices {subset_indices[0]}-{subset_indices[-1]})"
            )
        else:
            logger.info(f"Processing {len(subset)} samples on {device}")

        batch_size = getattr(getattr(request.config, "hardware", None), "batch_size_per_gpu", 4)
        num_workers = min(
            getattr(getattr(request.config, "hardware", None), "num_workers_per_gpu", 4),
            8,
        )

        data_loader = DataLoader(
            subset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
            pin_memory=True,
        )

        embeddings: List[np.ndarray] = []
        accessions: List[str] = []
        if hasattr(model, "eval"):
            model.eval()

        if world_size > 1:
            pbar = tqdm(data_loader, desc=f"GPU {rank}", position=rank, leave=True)
        else:
            pbar = tqdm(data_loader, desc="Extracting embeddings")

        with torch.no_grad():
            for batch_idx, volumes in enumerate(pbar):
                batch_start = time.time()
                try:
                    # unpack [volume, extra_infos] datasets
                    if isinstance(volumes, list):
                        extra_infos = volumes[1]
                        volumes = volumes[0]
                    else:
                        extra_infos = None

                    logger.debug(
                        f"GPU {rank}: Starting batch {batch_idx}, received volume shape: "
                        f"{volumes.shape}, dtype: {volumes.dtype}"
                    )

                    if not _validate_volume_shape(
                        volumes, rank, batch_idx, request.continue_on_error
                    ):
                        continue

                    batch_accessions: List[str] = []
                    for i in range(len(volumes)):
                        if world_size > 1:
                            if hasattr(dataset, "get_accession"):
                                original_idx = subset_indices[batch_idx * batch_size + i]
                                batch_accessions.append(dataset.get_accession(original_idx))
                            else:
                                original_idx = batch_idx * batch_size + i
                                if original_idx >= len(dataset):
                                    break
                                batch_accessions.append(dataset.get_accession(original_idx))
                        else:
                            original_idx = batch_idx * batch_size + i
                            if original_idx >= len(dataset):
                                break
                            batch_accessions.append(dataset.get_accession(original_idx))

                    logger.debug(
                        f"GPU {rank}: Batch {batch_idx} - Processing {len(batch_accessions)} "
                        f"samples: {batch_accessions}"
                    )

                    try:
                        with _timed(rank, batch_idx, "GPU transfer"):
                            volumes = volumes.to(device)
                        logger.debug(f"GPU {rank}: Batch {batch_idx} - Moved to device {device}")

                        if volumes.numel() == 0:
                            logger.error(
                                f"GPU {rank}: Batch {batch_idx} - Empty tensor after moving to device"
                            )
                            if request.continue_on_error:
                                logger.warning(
                                    "!  CONTINUING after empty tensor (--continue-on-error enabled)"
                                )
                                continue
                            raise RuntimeError("Empty tensor after moving to device")

                        volumes = volumes.float()

                        if volumes.numel() > 0:
                            logger.debug(
                                f"GPU {rank}: Batch {batch_idx} - Input stats: "
                                f"min={volumes.min():.4f}, max={volumes.max():.4f}, "
                                f"mean={volumes.mean():.4f}"
                            )
                        else:
                            logger.error(
                                f"GPU {rank}: Batch {batch_idx} - Tensor is empty, cannot compute stats"
                            )
                            continue
                    except Exception as tensor_error:
                        logger.error(
                            f"GPU {rank}: Batch {batch_idx} - Error processing tensor: {tensor_error}"
                        )
                        if request.continue_on_error:
                            logger.warning(
                                "!  CONTINUING after tensor error (--continue-on-error enabled)"
                            )
                            continue
                        raise

                    if request.check_nan:
                        _check_input_nan(
                            volumes,
                            rank,
                            batch_idx,
                            batch_accessions,
                            request.continue_on_error,
                        )

                    with _timed(rank, batch_idx, "Model inference"):
                        if extra_infos is not None:
                            batch_embeddings = model.extract_features(
                                volumes, modality=modality, extra_infos=extra_infos
                            )
                        else:
                            batch_embeddings = model.extract_features(volumes, modality=modality)
                    logger.debug(
                        f"GPU {rank}: Batch {batch_idx} - Extracted embeddings shape: "
                        f"{batch_embeddings.shape}, dtype: {batch_embeddings.dtype}"
                    )

                    if batch_embeddings.size == 0:
                        logger.error(
                            f"GPU {rank}: Batch {batch_idx} - Model returned empty embeddings"
                        )
                        if request.continue_on_error:
                            logger.warning(
                                "!  CONTINUING after empty embeddings (--continue-on-error enabled)"
                            )
                            continue
                        raise RuntimeError("Model returned empty embeddings")

                    if batch_embeddings.size > 0:
                        logger.debug(
                            f"GPU {rank}: Batch {batch_idx} - Embedding stats: "
                            f"min={batch_embeddings.min():.4f}, max={batch_embeddings.max():.4f}, "
                            f"mean={batch_embeddings.mean():.4f}"
                        )
                        _check_output_nan(
                            batch_embeddings,
                            rank,
                            batch_idx,
                            batch_accessions,
                            request.continue_on_error,
                        )

                    embeddings.append(batch_embeddings)
                    accessions.extend(batch_accessions)

                    # per-rank .npz; processed.csv is fcntl-locked
                    if checkpoint_manager is not None:
                        with _timed(rank, batch_idx, "Checkpoint saving"):
                            for i, accession in enumerate(batch_accessions):
                                if not checkpoint_manager.is_sample_processed(accession):
                                    sample_embedding = batch_embeddings[i : i + 1]
                                    checkpoint_manager.save_sample_embedding(
                                        accession,
                                        sample_embedding,
                                        continue_on_error=request.continue_on_error,
                                    )

                    logger.debug(
                        f"GPU {rank} Batch {batch_idx}: Total batch time "
                        f"{time.time() - batch_start:.2f}s"
                    )
                except Exception as batch_error:
                    logger.error(f"GPU {rank}: Error processing batch {batch_idx}: {batch_error}")
                    logger.debug(
                        f"GPU {rank}: Batch {batch_idx} traceback: "
                        f"{''.join(traceback.format_tb(batch_error.__traceback__))}"
                    )
                    if request.continue_on_error:
                        logger.warning(
                            "!  CONTINUING after batch error (--continue-on-error enabled)"
                        )
                        logger.warning(
                            f"Batch {batch_idx} will be skipped, but extraction continues"
                        )
                        continue
                    raise

        if embeddings:
            gpu_embeddings = np.concatenate(embeddings, axis=0)
            logger.info(f"GPU {rank}: Extracted {gpu_embeddings.shape[0]} embeddings")
            return gpu_embeddings, accessions
        logger.error(f"GPU {rank}: No embeddings extracted")
        return np.array([]), []

    except Exception as e:
        logger.error(f"GPU {rank}: Error during extraction: {e}")
        traceback.print_exc()
        return np.array([]), []


__all__ = ["run_single_gpu"]
