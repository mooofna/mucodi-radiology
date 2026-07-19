"""Pillar0 multimodal medical image model implementation."""

import types

import torch
import numpy as np
from einops import rearrange

from ..core.errors import ModelError
from ..core.logging import get_logger
from ..core.device import setup_device
from ..io.download_lock import ModelDownloadLock
from ..config import load_model_config, get_config_value, merge_configs
from .common import batch_apply_ct_windowing, batch_apply_mr_windowing, batch_apply_normalization
from transformers import AutoModel


logger = get_logger(__name__)


class Pillar0:
    """Pillar0 multimodal medical image analysis."""

    def __init__(self, config: dict):
        self.config = config

        self.model_config = config.model
        self.device = setup_device(get_config_value(config, "device"))

        self.model_repo_id = get_config_value(self.model_config, "repo_id")
        self.model_revision = get_config_value(self.model_config, "revision")

        self.setup_model()

        logger.info(
            "Initialized Pillar0 %s@%s on device: %s",
            self.model_repo_id,
            self.model_revision,
            self.device,
        )

    def setup_model(self) -> None:
        """Initialize the model architecture and load pretrained weights."""
        download_lock = ModelDownloadLock(
            model_repo_id=self.model_repo_id, revision=self.model_revision
        )

        with download_lock.acquire_download_lock(timeout=600):
            logger.info(
                "Loading Pillar0 model %s@%s",
                self.model_repo_id,
                self.model_revision,
            )
            self.model = AutoModel.from_pretrained(
                self.model_repo_id, revision=self.model_revision, trust_remote_code=True
            )

        self.model.to(self.device)

        # upstream hard-codes normalize=True (collapses features, kills probe signal); force normalize=False
        def _extract_vision_feats_unnormalized(self, image=None, info=None, batch=None):
            return self.model.encode_image(image=image, normalize=False, info=info)
        self.model.extract_vision_feats = types.MethodType(
            _extract_vision_feats_unnormalized, self.model
        )
        logger.info("Pillar0: monkey-patched extract_vision_feats to normalize=False")

        self.model.eval()

        logger.info(f"Model loaded successfully on device: {self.device}")

    @staticmethod
    def preprocess_single(image, model_config, metadata=None, modality=None):
        """Preprocess a single exam for Pillar0 (no normalization here)."""
        return image

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """Preprocess input volumes before extraction."""
        apply_normalization = get_config_value(self.model_config, "apply_normalization")
        if modality in ["chest_ct", "abdomen_ct", "brain_ct"]:
            ct_window_type = get_config_value(self.model_config, "ct_window_type")
            assert ct_window_type is not None, "CT window type is not set"
            assert volumes.dim() == 5, f"Volumes should be 5D, got {volumes.dim()}"

            normalize_mean = get_config_value(self.model_config, "ct_normalize_mean")
            normalize_std = get_config_value(self.model_config, "ct_normalize_std")

            per_sample = get_config_value(self.model_config, "per_sample_windowing")
            volumes = batch_apply_ct_windowing(
                volumes,
                ct_window_type=ct_window_type,
                modality="CT",
                per_sample=per_sample,
            )
        elif modality in ["breast_mr"]:
            mr_window_type = get_config_value(self.model_config, "mr_window_type")
            assert mr_window_type is not None, "MR window type is not set"
            assert volumes.dim() == 5, f"Volumes should be 5D, got {volumes.dim()}"
            per_sample = get_config_value(self.model_config, "per_sample_windowing")
            volumes = batch_apply_mr_windowing(
                volumes, mr_window_type, modality="MR", per_sample=per_sample
            )
            normalize_mean = get_config_value(self.model_config, "mr_normalize_mean")
            normalize_std = get_config_value(self.model_config, "mr_normalize_std")
        else:
            assert "xray" in modality.lower(), f"Modality {modality} is not supported"
            normalize_mean = get_config_value(self.model_config, "xray_normalize_mean")
            normalize_std = get_config_value(self.model_config, "xray_normalize_std")

        if apply_normalization:
            volumes = batch_apply_normalization(volumes, normalize_mean, normalize_std)
        return volumes

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor, modality="chest_ct") -> np.ndarray:
        """Extract (B, feature_dim) features from (B, C, H, W) or (B, C, D, H, W) inputs."""
        inputs = inputs.to(self.device)
        inputs = self.preprocess(inputs, modality)

        if modality == "chest_xray_two_view":
            logger.debug(
                "Using Pillar0 two-view inference for modality: %s",
                modality,
            )

            inputs_as_dict = {modality: inputs}

            with torch.no_grad():
                features = self.model.extract_vision_feats(inputs_as_dict)
                features = features.cpu().numpy().astype(np.float32)

        elif modality == "chest_xray_single_view":
            logger.debug(f"Using MedGemma-style single-view inference for modality: {modality}")

            if len(inputs.shape) == 4:
                N, C, H, W = inputs.shape
                D_in = 1
                inputs = inputs.unsqueeze(2)
            elif len(inputs.shape) == 5:
                N, C, D_in, H, W = inputs.shape
            else:
                raise ValueError(f"Expected 4D or 5D input tensor, got shape {inputs.shape}")

            # MedGemma-style: process each view separately, then pool across views
            inputs_rearranged = rearrange(inputs, "n c d h w -> (n d) c 1 h w")

            inputs_as_dict = {modality: inputs_rearranged}

            with torch.no_grad():
                features = self.model.extract_vision_feats(inputs_as_dict)

                features_rearranged = rearrange(features, "(n d) f -> n f d", d=D_in)

                pool_op = get_config_value(self.model_config, "pool_op")
                if pool_op == "max":
                    features = features_rearranged.max(-1).values.cpu().numpy()
                elif pool_op == "mean":
                    features = features_rearranged.mean(-1).cpu().numpy()
                elif pool_op == "median":
                    features = features_rearranged.median(-1).values.cpu().numpy()
                elif pool_op == "middle":
                    middle_idx = D_in // 2
                    features = features_rearranged[:, :, middle_idx].cpu().numpy()
                else:
                    raise ValueError(f"Unsupported pooling operation: {pool_op}")

                features = features.astype(np.float32)

        else:
            logger.debug(f"Using default inference for modality: {modality}")
            inputs_as_dict = {modality: inputs}

            with torch.no_grad():
                features = self.model.extract_vision_feats(inputs_as_dict)
                features = features.cpu().numpy().astype(np.float32)

        return features

    def eval(self):
        if hasattr(self, "model"):
            self.model.eval()
        return self
