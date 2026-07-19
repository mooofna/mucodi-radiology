"""In-memory dataset for frozen-feature CV evaluation."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset

# back-compat re-exports
from ..io.feature_loaders import (  # noqa: F401
    _DEFAULT_QA_KEY,
    _iter_embeddings,
    _iter_embeddings_per_split,
    _load_lidc_labels,
    _patient_id_from_accession,
    load_features_from_cache,
    load_features_from_cache_split,
)


class FeatureDataset(Dataset):
    """`(feature, label)` dataset over an in-memory `(N, D)` tensor + `(N,)` labels."""

    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        if features.ndim != 2:
            raise ValueError(f"features must be 2D (N, D), got {tuple(features.shape)}")
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D (N,), got {tuple(labels.shape)}")
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"features (N={features.shape[0]}) and labels (N={labels.shape[0]}) length mismatch",
            )
        self.features = features.to(torch.float32)
        self.labels = labels.to(torch.long)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]
