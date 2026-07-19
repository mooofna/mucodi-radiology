"""Shared LIDC chest-CT base: reads per-teacher JSONL manifests, dispatches to _preprocess_volume."""

from __future__ import annotations

import os

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from ...core.errors import DatasetError
from ...core.logging import get_logger
from ...config import get_config_value


logger = get_logger(__name__)


class LIDCBaseDataset:
    """Boilerplate: load the per-split JSONL, expose accessions, dispatch to `_preprocess_volume`."""

    SPLIT_TO_CONFIG_KEY = {
        "train": "data.train_json",
        "valid": "data.valid_json",
        "dev": "data.valid_json",
        "test": "data.test_json",
    }

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        transforms: Optional[Any] = None,
        model_preprocess: Optional[Any] = None,
        name_override: Optional[str] = None,
    ) -> None:
        self.config = config
        self.split = split
        self.modality = get_config_value(self.config, "modality")
        self.image_key = get_config_value(self.config, "img_paths_key") or "nii_path"
        self.additional_transforms = transforms
        self.model_preprocess = model_preprocess
        self.samples: List[Dict[str, Any]] = []
        self._load_manifest()
        logger.info(
            "Initialized %s (split=%s) with %d samples",
            self.__class__.__name__, split, len(self.samples),
        )

    def _load_manifest(self) -> None:
        json_key = self.SPLIT_TO_CONFIG_KEY.get(self.split)
        if json_key is None:
            raise DatasetError(f"unknown split '{self.split}' for {self.__class__.__name__}")
        json_path = get_config_value(self.config, json_key)
        if json_path is None:
            raise DatasetError(f"split JSONL path not configured: {json_key}")
        path = Path(json_path)
        if not path.exists():
            raise DatasetError(f"JSONL not found for split '{self.split}': {path}")
        with path.open() as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                row = json.loads(stripped)
                if "sample_name" not in row or self.image_key not in row:
                    raise DatasetError(f"JSONL row missing required keys: {row}")
                self.samples.append({"sample_name": row["sample_name"], "image_path": os.path.expandvars(row[self.image_key])})

    def __len__(self) -> int:
        return len(self.samples)

    def _preprocess_volume(self, image_path: str) -> torch.Tensor:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> torch.Tensor:
        sample = self.samples[idx]
        image = self._preprocess_volume(sample["image_path"])
        if self.additional_transforms is not None:
            image = self.additional_transforms(image)
        if self.model_preprocess is not None:
            image = self.model_preprocess(image, modality=self.modality)
        return image

    def get_sample_info(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        return {
            "sample_name": s["sample_name"],
            "split": self.split,
            "dataset_class": self.__class__.__name__,
            "accession": s["sample_name"],
            "sample_id": f"{self.split}_{s['sample_name']}",
        }

    def get_accession(self, idx: int) -> str:
        return self.samples[idx]["sample_name"]

    def get_all_accessions(self) -> List[str]:
        return [s["sample_name"] for s in self.samples]

    def get_accessions_batch(self, indices: List[int]) -> List[str]:
        return [self.samples[i]["sample_name"] for i in indices]
