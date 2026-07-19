"""TANGERINE (ViT-Large 3D-MAE) rate-evals wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from ..core.logging import get_logger
from ..core.device import setup_device
from ..config import get_config_value


logger = get_logger(__name__)


def _torch_load_compat(path: str | Path):
    """torch.load that survives PyTorch 2.6+ default `weights_only=True`."""
    try:
        return torch.load(str(path), map_location="cpu")
    except Exception:
        return torch.load(str(path), map_location="cpu", weights_only=False)


class TangerineVit:
    """TANGERINE 3D-MAE ViT-Large encoder (CLS-token head)."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_config = config.model
        self.device = setup_device(get_config_value(config, "device"))

        upstream_dir = Path(get_config_value(self.model_config, "upstream_dir")).resolve()
        ckpt_path = Path(get_config_value(self.model_config, "checkpoint_path")).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"TANGERINE checkpoint not found: {ckpt_path}")
        # path must persist across subprocess re-imports
        from .upstream_paths import add_upstream_to_path
        add_upstream_to_path(upstream_dir)

        from models_vit import vit_large_patch16_yo  # type: ignore[import-not-found]
        from util.pos_embed import interpolate_pos_embed  # type: ignore[import-not-found]

        # global_pool=True: forward_features -> concat(CLS, mean(patch_tokens)) = (B, 2048), un-normed
        model = vit_large_patch16_yo(num_classes=0, global_pool=True)
        ckpt = _torch_load_compat(ckpt_path)
        sd = ckpt.get("model") if isinstance(ckpt, dict) and "model" in ckpt else \
             ckpt.get("model_state") if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        if "pos_embed" in sd:
            interpolate_pos_embed(model, sd)
        sd = {
            k: v for k, v in sd.items()
            if k in model.state_dict() and model.state_dict()[k].shape == v.shape
        }
        msg = model.load_state_dict(sd, strict=False)
        logger.info(
            "TANGERINE vit loaded (loaded=%d, missing=%d, unexpected=%d)",
            len(sd), len(msg.missing_keys), len(msg.unexpected_keys),
        )

        model.eval().to(self.device)
        self.model = model

    @staticmethod
    def preprocess_single(image: torch.Tensor, model_config: Any = None, metadata: Any = None, modality: str | None = None) -> torch.Tensor:
        """No-op: dataset already produces the model-ready tensor."""
        return image

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """No-op at batch level; preprocessing baked into the dataset class."""
        return volumes

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor, modality: str = "chest_ct", extra_infos: Any = None) -> np.ndarray:
        """C=1 -> (B, 2048) CLS+patch-mean; C=2 -> (B, 1024) mask-pooled patch-mean."""
        if inputs.ndim != 5:
            raise ValueError(f"TangerineVit expects (B, C, D, H, W), got {tuple(inputs.shape)}")
        C = inputs.shape[1]
        if C not in (1, 2):
            raise ValueError(f"TangerineVit expects C=1 or C=2, got C={C}")
        inputs = inputs.to(self.device)

        if C == 1:
            feats = self.model.forward_features(inputs)
            return feats.cpu().numpy().astype(np.float32)

        # mask-aware path: replicate forward_features, then mask-pool patch tokens
        image = inputs[:, 0:1]                             # (B, 1, 256, 256, 256)
        mask = inputs[:, 1]                                # (B, 256, 256, 256)
        B = image.shape[0]
        x = self.model.patch_embed(image)                  # (B, 4096, 1024) D-major
        cls_tokens = self.model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)              # (B, 4097, 1024)
        x = x + self.model.pos_embed
        x = self.model.pos_drop(x)
        for blk in self.model.blocks:
            x = blk(x)
        # no norm/fc_norm, matching global_pool=True forward_features

        # non-contiguous after slice: reshape, not view
        patch_tokens = x[:, 1:, :].reshape(B, 16, 16, 16, 1024)  # D-major: idx = d*H*W + h*W + w
        mask_grid = F.max_pool3d(mask.unsqueeze(1), kernel_size=16, stride=16).squeeze(1)  # (B, 16, 16, 16)
        out = torch.zeros((B, 1024), dtype=patch_tokens.dtype, device=patch_tokens.device)
        for b in range(B):
            sel = mask_grid[b] > 0
            if sel.any():
                out[b] = patch_tokens[b][sel].mean(dim=0)
            else:
                logger.warning("TangerineVit: empty mask at batch index %d, falling back to mean(all_patch_tokens)", b)
                out[b] = patch_tokens[b].reshape(-1, 1024).mean(dim=0)
        return out.cpu().numpy().astype(np.float32)

    def eval(self) -> "TangerineVit":
        if hasattr(self, "model"):
            self.model.eval()
        return self
