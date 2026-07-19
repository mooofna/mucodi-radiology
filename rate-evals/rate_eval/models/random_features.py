"""Random-init 3D backbone -- the architecture-only floor for cross_cohort_benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..core.logging import get_logger
from ..core.device import setup_device
from ..config import get_config_value
from .common import batch_apply_ct_windowing


logger = get_logger(__name__)


class RandomFeatures3D:
    """A fresh random-init 3D backbone exposed as a rate-evals teacher."""

    def __init__(self, config: dict):
        self.config = config
        self.model_config = config.model
        self.device = setup_device(get_config_value(config, "device"))

        self.arch = get_config_value(self.model_config, "arch") or "efficientnet3d_b0"
        self.in_channels = int(get_config_value(self.model_config, "in_channels") or 4)
        self.ct_window_type = get_config_value(self.model_config, "ct_window_type") or [
            "lung",
            "mediastinum",
            "soft_tissue",
            "bone",
        ]
        self.seed = int(get_config_value(self.model_config, "seed") or 0)

        self._setup_model()

        logger.info(
            "Initialized RandomFeatures3D arch=%s in_channels=%d seed=%d on %s (NO checkpoint -- random-init floor)",
            self.arch,
            self.in_channels,
            self.seed,
            self.device,
        )

    def _setup_model(self) -> None:
        """Build a fresh random-init backbone -- no state_dict is ever loaded."""
        # parents[3] = repo root (same resolution as student.py)
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from models.efficientnet3d import create_3d_backbone

        # seed before construction for reproducible per-(arch,seed) init
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        self.backbone = create_3d_backbone(self.arch, in_channels=self.in_channels)
        self.backbone.to(self.device).eval()

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """Apply the configured multi-window CT stack to (B, C, D, H, W) HU volumes."""
        if modality not in ("chest_ct", "abdomen_ct", "brain_ct"):
            raise ValueError(
                f"RandomFeatures3D only supports CT modalities; got {modality!r}"
            )
        return batch_apply_ct_windowing(
            volumes,
            ct_window_type=self.ct_window_type,
            modality="CT",
            per_sample=True,
        )

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor, modality: str = "chest_ct") -> np.ndarray:
        """Extract d_S random-init backbone features for a batch of CT volumes."""
        inputs = inputs.to(self.device)
        windowed = self.preprocess(inputs, modality)
        feats = self.backbone(windowed)
        return feats.detach().cpu().numpy().astype(np.float32)

    def eval(self) -> "RandomFeatures3D":
        self.backbone.eval()
        return self
