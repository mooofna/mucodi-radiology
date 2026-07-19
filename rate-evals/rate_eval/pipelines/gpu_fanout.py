"""Multi-GPU fan-out: single-GPU runs inline; multi-GPU spawns one `mp.Process` per GPU, results via `mp.Queue`."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp

from ..components import create_dataset
from ..core.logging import get_logger
from ..io.cache import SimpleCheckpointManager
from .extract import ExtractRequest
from .extract_loop import run_single_gpu

logger = get_logger(__name__)


def _worker_process(
    rank: int,
    num_gpus: int,
    request: ExtractRequest,
    split: str,
    checkpoint_manager: Optional[SimpleCheckpointManager],
    results_queue: "mp.Queue",
) -> None:
    """`mp.Process` target -- runs `run_single_gpu` and posts the result to the queue."""
    result = run_single_gpu(
        rank=rank,
        world_size=num_gpus,
        request=request,
        split=split,
        checkpoint_manager=checkpoint_manager,
    )
    results_queue.put((rank, *result))


def _determine_num_gpus(requested: Optional[int]) -> int:
    """Resolve `request.num_gpus` against availability. Returns 1 for CPU-only."""
    if not torch.cuda.is_available():
        logger.warning("No GPUs available, using CPU")
        return 1
    available = torch.cuda.device_count()
    if requested:
        return min(requested, available)
    return available


def run_fanout(
    request: ExtractRequest,
    split: str,
    checkpoint_manager: Optional[SimpleCheckpointManager] = None,
) -> Tuple[np.ndarray, List[str]]:
    """Dispatch extraction across available GPUs -> (embeddings, accessions)."""
    num_gpus = _determine_num_gpus(request.num_gpus)
    if num_gpus > 1:
        gpu_ids = list(range(num_gpus))
        logger.info(f"Using {num_gpus} GPUs: {gpu_ids}")
    else:
        logger.info(f"Using {num_gpus} {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # pre-load once just to size logging (workers re-create their own)
    dataset = create_dataset(request.dataset, request.config, split)
    dataset_size = len(dataset)
    if request.max_samples:
        dataset_size = min(dataset_size, request.max_samples)
    logger.info(f"Total samples to process: {dataset_size}")
    del dataset

    if num_gpus == 1:
        return run_single_gpu(
            rank=0,
            world_size=1,
            request=request,
            split=split,
            checkpoint_manager=checkpoint_manager,
        )

    if checkpoint_manager is not None:
        logger.info("Multi-GPU checkpoint mode: All GPUs will save their processed embeddings")

    mp.set_start_method("spawn", force=True)

    processes: List[mp.Process] = []
    results_queue: "mp.Queue" = mp.Queue()

    for rank in range(num_gpus):
        p = mp.Process(
            target=_worker_process,
            args=(rank, num_gpus, request, split, checkpoint_manager, results_queue),
        )
        p.start()
        processes.append(p)

    all_embeddings: List[np.ndarray] = []
    all_accessions: List[str] = []
    for _ in range(num_gpus):
        _rank, embeddings, accessions = results_queue.get()
        if len(embeddings) > 0:
            all_embeddings.append(embeddings)
            all_accessions.extend(accessions)

    for p in processes:
        p.join()

    if all_embeddings:
        combined = np.concatenate(all_embeddings, axis=0)
        logger.info(f"Combined embeddings from all GPUs: {combined.shape}")
        return combined, all_accessions
    return np.array([]), []


__all__ = ["run_fanout"]
