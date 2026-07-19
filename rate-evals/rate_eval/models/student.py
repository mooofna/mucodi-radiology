"""MuCoDi 3D student wrapper for the rate-evals harness."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..core.logging import get_logger
from ..core.device import setup_device
from ..config import get_config_value
from .common import batch_apply_ct_windowing


logger = get_logger(__name__)


class MuCoDiStudent:
    """Wrap a MuCoDi-trained 3D student as a rate-evals teacher."""

    def __init__(self, config: dict):
        self.config = config
        self.model_config = config.model
        self.device = setup_device(get_config_value(config, "device"))

        self.checkpoint_path = get_config_value(self.model_config, "checkpoint_path")
        if not self.checkpoint_path or not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(
                f"MuCoDiStudent requires checkpoint_path (got {self.checkpoint_path!r})"
            )

        self.arch = get_config_value(self.model_config, "arch") or "efficientnet3d_b0"
        self.in_channels = int(get_config_value(self.model_config, "in_channels") or 4)
        self.ct_window_type = get_config_value(self.model_config, "ct_window_type") or ["lung", "mediastinum", "soft_tissue", "bone"]

        self._setup_model()

        logger.info(
            "Initialized MuCoDiStudent arch=%s in_channels=%d checkpoint=%s on %s",
            self.arch,
            self.in_channels,
            self.checkpoint_path,
            self.device,
        )

    def _setup_model(self) -> None:
        """Load student backbone + state_dict from checkpoint."""
        # parents[3] = repo root
        import sys

        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        from models.efficientnet3d import create_3d_backbone

        self.backbone = create_3d_backbone(self.arch, in_channels=self.in_channels)

        # keep only backbone.* keys; fall back to raw dict
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict") or checkpoint

        backbone_state = {}
        prefix = "backbone."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                backbone_state[k[len(prefix):]] = v
        if not backbone_state:
            raise RuntimeError(
                f"Checkpoint {self.checkpoint_path} contained no 'backbone.*' weights. "
                f"Available top-level prefixes: {sorted({k.split('.')[0] for k in state_dict})[:10]}"
            )

        missing, unexpected = self.backbone.load_state_dict(backbone_state, strict=False)
        # fail loud: a partial load silently depresses AUROC
        if missing or unexpected:
            raise RuntimeError(
                f"MuCoDiStudent: refusing to eval on a partial backbone load from "
                f"{self.checkpoint_path} (arch={self.arch}): {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys. missing[:5]={missing[:5]} "
                f"unexpected[:5]={unexpected[:5]}. Likely a MONAI-version/arch mismatch "
                f"(see the B5 monai_version stamp)."
            )

        self.backbone.to(self.device).eval()

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """Apply the configured multi-window CT stack to (B, C, D, H, W) HU volumes."""
        if modality not in ("chest_ct", "abdomen_ct", "brain_ct"):
            raise ValueError(
                f"MuCoDiStudent only supports CT modalities; got {modality!r}"
            )
        return batch_apply_ct_windowing(
            volumes,
            ct_window_type=self.ct_window_type,
            modality="CT",
            per_sample=True,
        )

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor, modality: str = "chest_ct") -> np.ndarray:
        """Extract 512-d backbone features for a batch of CT volumes -> (B, 512) float32."""
        inputs = inputs.to(self.device)
        windowed = self.preprocess(inputs, modality)
        feats = self.backbone(windowed)
        return feats.detach().cpu().numpy().astype(np.float32)

    def eval(self) -> "MuCoDiStudent":
        self.backbone.eval()
        return self
