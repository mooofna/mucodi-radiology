"""Per-sample embedding cache and resumable dataset wrapper."""

import fcntl
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..core.logging import get_logger


class SimpleCheckpointManager:
    """Simple checkpoint manager that stores individual .npz files per sample."""

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        split: str,
        cache_dir: str = "cache",
        skip_existing_cache: bool = False,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.split = split
        self.skip_existing_cache = skip_existing_cache

        self.cache_root = Path(cache_dir)
        self.embeddings_dir = self.cache_root / "embeddings" / split
        self.processed_csv = self.cache_root / "processed.csv"

        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

        logger = get_logger(__name__)

        self._initial_processed_samples: Set[str] = set()
        if self.skip_existing_cache:
            self._initial_processed_samples = self._load_processed_samples()
            if self._initial_processed_samples:
                logger.info(
                    "Ignoring %d cached samples for %s/%s %s",
                    len(self._initial_processed_samples),
                    self.model_name,
                    self.dataset_name,
                    self.split,
                )
            self.processed_samples: Set[str] = set()
        else:
            self.processed_samples = self._load_processed_samples()

        logger.info(
            "Checkpoint manager initialized: %d samples already processed",
            len(self.processed_samples),
        )

    def _load_processed_samples(self) -> set:
        """Load processed samples from CSV into a set for fast lookup."""
        if not self.processed_csv.exists():
            return set()

        try:
            df = pd.read_csv(self.processed_csv)

            filtered_df = df[
                (df["model_name"] == self.model_name)
                & (df["dataset_name"] == self.dataset_name)
                & (df["split"] == self.split)
            ]

            processed_set = set(str(acc) for acc in filtered_df["accession"].tolist())
            logger = get_logger(__name__)
            logger.info(f"Found {len(processed_set)} previously processed samples")
            return processed_set
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to load processed.csv: {e}")
            return set()

    def is_sample_processed(self, accession: str) -> bool:
        """Check if a sample has already been processed."""
        return str(accession) in self.processed_samples

    def save_sample_embedding(
        self, accession: str, embedding: np.ndarray, continue_on_error: bool = False
    ) -> None:
        """Save a single sample's embedding to .npz file and update processed.csv."""
        logger = get_logger(__name__)

        accession_str = str(accession)

        logger.debug(
            f"Checkpoint: Attempting to save embedding for {accession_str}, shape: {embedding.shape}, dtype: {embedding.dtype}"
        )
        logger.debug(
            f"Checkpoint: Embedding stats for {accession_str}: min={embedding.min():.4f}, max={embedding.max():.4f}, mean={embedding.mean():.4f}"
        )

        nan_count = np.isnan(embedding).sum()
        logger.debug(
            f"Checkpoint: NaN check for {accession_str}: {nan_count} out of {embedding.size} elements are NaN"
        )

        if nan_count > 0:
            logger.error(f"NaN detected in embedding for sample {accession_str}")
            logger.error(f"Embedding shape: {embedding.shape}")
            logger.error(f"NaN locations: {nan_count} out of {embedding.size} elements")

            logger.debug(f"Checkpoint: NaN pattern analysis for {accession_str}:")
            nan_mask = np.isnan(embedding)
            for batch_idx in range(embedding.shape[0]):
                batch_nan_count = nan_mask[batch_idx].sum()
                if batch_nan_count > 0:
                    logger.debug(f"  Batch {batch_idx}: {batch_nan_count} NaN values")

            if continue_on_error:
                logger.warning(
                    f"!  SAVING NaN embedding for {accession_str} (continue-on-error enabled)"
                )
                logger.warning(f"This embedding contains corrupted data!")
            else:
                raise RuntimeError(
                    f"NaN values detected in embedding for sample {accession_str}. Refusing to save corrupted data."
                )

        embedding_file = self.embeddings_dir / f"{accession_str}.npz"
        logger.debug(f"Checkpoint: Saving embedding to {embedding_file}")
        np.savez_compressed(embedding_file, embedding=embedding)
        logger.debug(f"Checkpoint: Successfully saved embedding for {accession_str}")

        if self.skip_existing_cache and accession_str in self._initial_processed_samples:
            self._initial_processed_samples.remove(accession_str)

        self.processed_samples.add(accession_str)

        self._append_to_processed_csv(accession_str)
        logger.debug(f"Checkpoint: Added {accession_str} to processed samples")

    def _append_to_processed_csv(self, accession: str) -> None:
        """Append a processed sample to the CSV, locked for multi-GPU write safety."""
        new_row = {
            "accession": str(accession),
            "model_name": self.model_name,
            "dataset_name": self.dataset_name,
            "split": self.split,
            "timestamp": datetime.now().isoformat(),
        }

        try:
            file_exists = self.processed_csv.exists()

            with open(self.processed_csv, "a", newline="") as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)

                write_header = not file_exists or f.tell() == 0

                df = pd.DataFrame([new_row])
                df.to_csv(f, header=write_header, index=False)
                # lock releases on close
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to append to processed.csv: {e}")

    def load_sample_embedding(self, accession: str) -> Optional[np.ndarray]:
        """Load a sample's embedding from .npz file."""
        embedding_file = self.embeddings_dir / f"{str(accession)}.npz"
        if not embedding_file.exists():
            return None

        try:
            data = np.load(embedding_file)
            return data["embedding"]
        except Exception as e:
            logger = get_logger(__name__)
            logger.warning(f"Failed to load embedding for {accession}: {e}")
            return None

    def get_all_embeddings(self) -> Tuple[np.ndarray, List[str]]:
        """Load all embeddings for this model/dataset/split combination."""
        embeddings = []
        accessions = []

        for accession in self.processed_samples:
            embedding = self.load_sample_embedding(accession)
            if embedding is not None:
                embeddings.append(embedding)
                accessions.append(accession)

        if embeddings:
            return np.stack(embeddings), accessions
        else:
            return np.array([]), []

    def get_all_embeddings_from_directory(self) -> Tuple[np.ndarray, List[str]]:
        """Load ALL embeddings from the directory, ignoring model/dataset filtering."""
        embeddings = []
        accessions = []

        logger = get_logger(__name__)

        for embedding_file in self.embeddings_dir.glob("*.npz"):
            accession = embedding_file.stem
            try:
                data = np.load(embedding_file)
                embedding = data["embedding"]

                if len(embeddings) < 5:
                    logger.debug(f"Loading {accession}: shape {embedding.shape}")

                embeddings.append(embedding)
                accessions.append(accession)
            except Exception as e:
                logger.warning(f"Failed to load embedding from {embedding_file}: {e}")
                continue

        if embeddings:
            shapes = [emb.shape for emb in embeddings]
            unique_shapes = set(shapes)
            if len(unique_shapes) > 1:
                logger.error(f"Shape mismatch detected! Found shapes: {unique_shapes}")
                logger.error(f"First few shapes: {shapes[:10]}")
                # reconcile by squeezing a leading singleton dim
                fixed_embeddings = []
                for emb in embeddings:
                    if emb.ndim == 3 and emb.shape[0] == 1:
                        fixed_embeddings.append(emb.squeeze(0))
                    else:
                        fixed_embeddings.append(emb)
                embeddings = fixed_embeddings

            return np.stack(embeddings), accessions
        else:
            return np.array([]), []

    def refresh_from_disk(self) -> None:
        """Reload the processed set from disk (multi-GPU workers write embeddings the main process hasn't seen)."""
        logger = get_logger(__name__)

        if self.processed_csv.exists():
            try:
                df = pd.read_csv(self.processed_csv)
                mask = (
                    (df["model_name"] == self.model_name)
                    & (df["dataset_name"] == self.dataset_name)
                    & (df["split"] == self.split)
                )
                filtered_df = df[mask]

                if self.skip_existing_cache and not filtered_df.empty:
                    filtered_df = filtered_df[
                        ~filtered_df["accession"].astype(str).isin(self._initial_processed_samples)
                    ]

                old_count = len(self.processed_samples)
                self.processed_samples = set(filtered_df["accession"].astype(str))
                new_count = len(self.processed_samples)

                logger.debug(
                    f"Checkpoint: Refreshed from disk - {old_count} -> {new_count} processed samples"
                )

            except Exception as e:
                logger.warning(f"Failed to refresh checkpoint from disk: {e}")
        else:
            logger.debug("Checkpoint: No processed.csv found, no refresh needed")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about processed samples."""
        return {
            "total_processed": len(self.processed_samples),
            "embeddings_dir": str(self.embeddings_dir),
            "processed_csv": str(self.processed_csv),
            "has_processed_samples": len(self.processed_samples) > 0,
            "ignored_processed": (
                len(self._initial_processed_samples) if self.skip_existing_cache else 0
            ),
            "skip_existing_cache": self.skip_existing_cache,
        }


class SimpleResumableDataset(Dataset):
    """Simple wrapper for datasets that supports resumable iteration."""

    def __init__(self, dataset, checkpoint_manager: SimpleCheckpointManager):
        self.dataset = dataset
        self.checkpoint_manager = checkpoint_manager

        logger = get_logger(__name__)
        logger.info("Filtering dataset for unprocessed samples...")

        self.unprocessed_indices: List[int] = []
        self.sample_id_map: Dict[int, str] = {}  # filtered index -> original accession

        total_samples = len(dataset)

        logger.info("Loading all accessions for efficient filtering...")
        all_accessions = dataset.get_all_accessions()

        for idx, accession in enumerate(all_accessions):
            if not checkpoint_manager.is_sample_processed(accession):
                filtered_idx = len(self.unprocessed_indices)
                self.unprocessed_indices.append(idx)
                self.sample_id_map[filtered_idx] = accession

        self._remaining = len(self.unprocessed_indices)

        already_processed = total_samples - self._remaining
        logger.info(
            "Dataset filtering complete: %d unprocessed / %d total samples (%d already processed)",
            self._remaining,
            total_samples,
            already_processed,
        )

    def __len__(self):
        return self._remaining

    def __getitem__(self, idx):
        original_idx = self.unprocessed_indices[idx]
        return self.dataset[original_idx]

    def get_accession(self, idx):
        return self.sample_id_map[idx]

    def get_sample_info(self, idx):
        original_idx = self.unprocessed_indices[idx]
        return self.dataset.get_sample_info(original_idx)
