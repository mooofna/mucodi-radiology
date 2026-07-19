"""Curia-2 (raidium/curia-2, ViT-B DINOv2) rate-evals wrapper."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ..core.logging import get_logger
from ..core.device import setup_device
from ..config import get_config_value


logger = get_logger(__name__)


class Curia2:
    """Curia-2 axial-slice 2D ViT, mean-pooled to a single per-volume embedding."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_config = config.model
        self.device = setup_device(get_config_value(config, "device"))

        from transformers import AutoImageProcessor, AutoModel
        repo_id = get_config_value(self.model_config, "repo_id") or "raidium/curia-2"

        self.processor = AutoImageProcessor.from_pretrained(repo_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        self.model.eval().to(self.device)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Curia-2 model.config has no hidden_size -- cannot determine feature dim")
        self.hidden_size = int(hidden_size)
        logger.info("Curia-2 loaded (%s, hidden_size=%d)", repo_id, self.hidden_size)

    @staticmethod
    def preprocess_single(image: torch.Tensor, model_config: Any = None, metadata: Any = None, modality: str | None = None) -> torch.Tensor:
        return image

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        return volumes

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor, modality: str = "chest_ct", extra_infos: Any = None) -> np.ndarray:
        """CLS-mean (C=1) or mask-pooled patch tokens (C=2), per volume -> (B, hidden_size)."""
        if inputs.ndim != 5:
            raise ValueError(f"Curia-2 expects (B, C, D, H, W), got {tuple(inputs.shape)}")
        B, C, D, H, W = inputs.shape
        if C not in (1, 2):
            raise ValueError(f"Curia-2 expects C=1 (image-only) or C=2 (image+mask), got C={C}")

        has_mask = C == 2
        out = np.zeros((B, self.hidden_size), dtype=np.float32)
        for b in range(B):
            vol = inputs[b, 0].detach().to(torch.float32).cpu()  # (D, H, W)
            mask = inputs[b, 1].detach().to(torch.float32).cpu() if has_mask else None

            # per-slice resize + clip_min
            slices_proc = []
            for d in range(D):
                sl = vol[d].clone()
                sl = self.processor._resize(sl)
                sl = self.processor._clip_min(sl)
                slices_proc.append(sl)
            stack = torch.stack(slices_proc, dim=0).unsqueeze(1)   # (D, 1, crop, crop)

            # single global z-score across the stack (curia_image_processor.py:134)
            stack = self.processor._zscore_per_image(stack)

            # resize mask per slice with nearest
            if has_mask:
                crop = stack.shape[-1]
                mask_resized = []
                for d in range(D):
                    m = mask[d].unsqueeze(0).unsqueeze(0).float()
                    m = torch.nn.functional.interpolate(m, size=(crop, crop), mode="nearest")
                    mask_resized.append(m[0, 0])
                mask_stack = torch.stack(mask_resized, dim=0)

            slice_feats: list[np.ndarray] = []
            for d in range(D):
                pixel_values = stack[d:d + 1].to(self.device)
                outputs = self.model(pixel_values=pixel_values, output_hidden_states=False)
                if has_mask:
                    feat = self._mask_pool_patch_tokens(outputs, mask_stack[d])
                    if feat is not None:
                        slice_feats.append(feat.cpu().numpy())
                else:
                    feat = self._extract_feat(outputs)
                    slice_feats.append(feat.squeeze(0).cpu().numpy())

            if not slice_feats:
                logger.warning("Curia-2: mask was empty for all %d slices, falling back to CLS-mean", D)
                cls_feats = []
                for d in range(D):
                    pixel_values = stack[d:d + 1].to(self.device)
                    outputs = self.model(pixel_values=pixel_values)
                    cls_feats.append(self._extract_feat(outputs).squeeze(0).cpu().numpy())
                out[b] = np.stack(cls_feats, axis=0).mean(axis=0)
            else:
                out[b] = np.stack(slice_feats, axis=0).mean(axis=0)
        return out

    def _mask_pool_patch_tokens(self, outputs: Any, mask_2d: torch.Tensor) -> torch.Tensor | None:
        """Average patch tokens overlapping the mask; None if none in-mask (modeling_dinov2.py:163)."""
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise RuntimeError(
                "Curia-2 mask-aware path needs last_hidden_state; "
                f"got attrs {dir(outputs)}"
            )
        patch_tokens = last_hidden[:, 1:, :]              # (1, n_patches, hidden_size)
        n_patches = patch_tokens.shape[1]
        spatial_dim = int(round(n_patches ** 0.5))
        if spatial_dim * spatial_dim != n_patches:
            raise RuntimeError(f"non-square patch grid: n_patches={n_patches}")
        crop = mask_2d.shape[-1]
        stride = crop // spatial_dim
        m = mask_2d.unsqueeze(0).unsqueeze(0).float()
        m = torch.nn.functional.max_pool2d(m, kernel_size=stride, stride=stride)
        m_flat = (m.flatten() > 0)
        if not m_flat.any():
            return None
        m_flat = m_flat.to(patch_tokens.device)
        in_mask = patch_tokens[0, m_flat, :]
        return in_mask.mean(dim=0)

    @staticmethod
    def _extract_feat(outputs) -> torch.Tensor:
        """Pull the CLS / pooler / hidden-state tensor out of Curia-2's HF output."""
        if torch.is_tensor(outputs):
            return outputs
        for attr in ("pooler_output", "image_embeds", "last_hidden_state"):
            val = getattr(outputs, attr, None)
            if torch.is_tensor(val):
                if attr == "last_hidden_state":
                    return val[:, 0, :]                    # CLS at index 0
                return val
        raise RuntimeError(f"could not extract feature tensor from {type(outputs).__name__}")

    def eval(self) -> "Curia2":
        if hasattr(self, "model"):
            self.model.eval()
        return self
