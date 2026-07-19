"""Curia-1 (raidium/curia, ViT-B 86M DINOv2) rate-evals wrapper."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from typing import Any, Optional

import numpy as np
import torch

from ..core.logging import get_logger
from ..core.device import setup_device
from ..config import get_config_value


logger = get_logger(__name__)


class Curia1:
    """Curia-1 axial-slice 2D ViT (raidium/curia), mean-pooled to a per-volume embedding."""

    DEFAULT_REPO_ID = "raidium/curia"

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_config = config.model
        self.device = setup_device(get_config_value(config, "device"))

        from transformers import AutoImageProcessor, AutoModel

        repo_id = get_config_value(self.model_config, "repo_id") or self.DEFAULT_REPO_ID
        if repo_id != self.DEFAULT_REPO_ID:
            logger.warning(
                "Curia1 wrapper received non-default repo_id=%s; expected %s. "
                "If you intended Curia-2, use rate_eval.models.curia2.Curia2 instead.",
                repo_id, self.DEFAULT_REPO_ID,
            )
        self.repo_id = repo_id

        self.processor = AutoImageProcessor.from_pretrained(repo_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)
        self.model.eval().to(self.device)

        hidden_size = getattr(self.model.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError("Curia-1 model.config has no hidden_size -- cannot determine feature dim")
        self.hidden_size = int(hidden_size)
        # batch-independent (no cross-slice interaction); >1 for bulk
        self.slice_batch_size = max(1, int(os.environ.get("CURIA1_SLICE_BATCH", "1")))
        self.amp_bf16 = os.environ.get("CURIA1_AMP_BF16", "0") == "1"
        # evenly-spaced slice cap; z-score still over all slices (0 = every slice)
        self.slice_max = max(0, int(os.environ.get("CURIA1_MAX_SLICES", "0")))
        logger.info(
            "Curia-1 loaded (%s, hidden_size=%d, slice_batch_size=%d, amp_bf16=%s, slice_max=%d)",
            repo_id, self.hidden_size, self.slice_batch_size, self.amp_bf16, self.slice_max,
        )

    @staticmethod
    def preprocess_single(
        image: torch.Tensor,
        model_config: Any = None,
        metadata: Any = None,
        modality: Optional[str] = None,
    ) -> torch.Tensor:
        """No-op: preprocessing happens inside extract_features / predict_with_head."""
        return image

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        """No-op: see preprocess_single."""
        return volumes

    def eval(self) -> "Curia1":
        if hasattr(self, "model"):
            self.model.eval()
        return self

    @torch.no_grad()
    def extract_features(
        self,
        inputs: torch.Tensor,
        modality: str = "chest_ct",
        extra_infos: Any = None,
    ) -> np.ndarray:
        """Per-volume features via per-slice ViT forward + mean-pool (C=1 CLS, C=2 mask-aware)."""
        if inputs.ndim != 5:
            raise ValueError(f"Curia-1 expects (B, C, D, H, W), got {tuple(inputs.shape)}")
        B, C, D, H, W = inputs.shape
        if C not in (1, 2):
            raise ValueError(f"Curia-1 expects C=1 (image-only) or C=2 (image+mask), got C={C}")
        has_mask = C == 2

        out = np.zeros((B, self.hidden_size), dtype=np.float32)
        for b in range(B):
            vol = inputs[b, 0].detach().to(torch.float32).cpu()  # (D, H, W)
            mask = inputs[b, 1].detach().to(torch.float32).cpu() if has_mask else None

            sbs = self.slice_batch_size
            if sbs > 1 and not has_mask and self.processor.do_resize:
                # fast path: GPU-resize the whole stack in one interpolate
                crop = self.processor.crop_size
                vol_g = vol.to(self.device)
                stack = torch.nn.functional.interpolate(
                    vol_g.unsqueeze(1), size=(crop, crop),
                    mode="bicubic", align_corners=False, antialias=True,
                )  # (D, 1, crop, crop) on GPU
                stack = self.processor._clip_min(stack)
            else:
                slices_proc = []
                for d in range(D):
                    sl = vol[d].clone()
                    sl = self.processor._resize(sl)
                    sl = self.processor._clip_min(sl)
                    slices_proc.append(sl)
                stack = torch.stack(slices_proc, dim=0).unsqueeze(1)
            stack = self.processor._zscore_per_image(stack)

            mask_stack: Optional[torch.Tensor] = None
            if has_mask:
                crop = stack.shape[-1]
                mask_resized = []
                for d in range(D):
                    m = mask[d].unsqueeze(0).unsqueeze(0).float()
                    m = torch.nn.functional.interpolate(m, size=(crop, crop), mode="nearest")
                    mask_resized.append(m[0, 0])
                mask_stack = torch.stack(mask_resized, dim=0)

            slice_feats: list[np.ndarray] = []
            if has_mask:
                for d in range(D):
                    pixel_values = stack[d:d + 1].to(self.device)
                    outputs = self.model(pixel_values=pixel_values, output_hidden_states=False)
                    feat = self._mask_pool_patch_tokens(outputs, mask_stack[d])
                    if feat is not None:
                        slice_feats.append(feat.cpu().numpy())
            else:
                if self.slice_max and D > self.slice_max:
                    idx = torch.linspace(0, D - 1, self.slice_max).round().long()
                    sel = stack[idx]
                else:
                    sel = stack
                ctx = (torch.autocast(self.device.type, dtype=torch.bfloat16)
                       if self.amp_bf16 else nullcontext())
                with ctx:
                    for d0 in range(0, sel.shape[0], sbs):
                        outputs = self.model(pixel_values=sel[d0:d0 + sbs].to(self.device),
                                             output_hidden_states=False)
                        cls = self._extract_cls(outputs)          # (s, hidden)
                        slice_feats.extend(cls.float().detach().cpu().numpy())

            if not slice_feats:
                logger.warning("Curia-1: mask was empty for all %d slices; falling back to CLS-mean", D)
                cls_feats = []
                for d in range(D):
                    pixel_values = stack[d:d + 1].to(self.device)
                    outputs = self.model(pixel_values=pixel_values)
                    cls_feats.append(self._extract_cls(outputs).squeeze(0).cpu().numpy())
                out[b] = np.stack(cls_feats, axis=0).mean(axis=0)
            else:
                out[b] = np.stack(slice_feats, axis=0).mean(axis=0)
        return out

    def load_task_head(self, subfolder: str) -> Any:
        """Load a Curia-1 pretrained task head from raidium/curia/<subfolder>."""
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            self.repo_id, subfolder=subfolder, trust_remote_code=True,
        )

        # make the cached modeling_dinov2 module importable
        hf_home = os.environ.get(
            "HF_HOME",
            os.path.expandvars("$SCRATCH/cache/huggingface"),
        )
        modules_root = f"{hf_home}/modules/transformers_modules/raidium/curia"
        if os.path.isdir(modules_root):
            for snapshot in sorted(os.listdir(modules_root)):
                snap_path = os.path.join(modules_root, snapshot)
                if os.path.isdir(snap_path) and snap_path not in sys.path:
                    sys.path.insert(0, snap_path)

        import modeling_dinov2 as md  # type: ignore

        head_model = md.Dinov2ForImageClassification(config)
        head_st_path = hf_hub_download(self.repo_id, f"{subfolder}/model.safetensors")
        state_dict = load_file(head_st_path)
        missing, unexpected = head_model.load_state_dict(state_dict, strict=False)

        head_keys = [k for k in missing if k.startswith(("classifier.", "attention_module."))]
        if head_keys:
            raise RuntimeError(
                f"Curia-1 task head '{subfolder}' missing critical keys after load: "
                f"{head_keys[:10]} (total missing: {len(missing)})"
            )
        if unexpected:
            logger.warning(
                "Curia-1 task head '%s' had %d unexpected keys (ignored): %r",
                subfolder, len(unexpected), unexpected[:3],
            )

        head_model = head_model.eval().to(self.device)
        return head_model

    @torch.no_grad()
    def predict_with_head(
        self,
        head_model: Any,
        raw_image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """Direct inference: processor + head_model -> logits on self.device."""
        out = self.processor([raw_image])
        pv = out["pixel_values"]
        if not torch.is_tensor(pv):
            pv = torch.tensor(pv)
        pv = pv.to(self.device)

        if mask is not None:
            mask_t = torch.from_numpy(mask) if not torch.is_tensor(mask) else mask
            mask_t = mask_t.unsqueeze(0).float().to(self.device)
            output = head_model(pixel_values=pv, mask=mask_t)
        else:
            output = head_model(pixel_values=pv)

        if isinstance(output, dict):
            return output["logits"]
        return output.logits

    def _mask_pool_patch_tokens(
        self, outputs: Any, mask_2d: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Average patch tokens whose mask cell is non-zero; None if none are in-mask."""
        last_hidden = getattr(outputs, "last_hidden_state", None)
        if last_hidden is None:
            raise RuntimeError(
                "Curia-1 mask-aware path needs last_hidden_state; got attrs "
                f"{dir(outputs)}"
            )
        patch_tokens = last_hidden[:, 1:, :]  # (1, n_patches, hidden_size)
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
    def _extract_cls(outputs: Any) -> torch.Tensor:
        """Extract the CLS token from an HF model output (DINOv2 convention: index 0)."""
        if torch.is_tensor(outputs):
            return outputs
        for attr in ("pooler_output", "image_embeds", "last_hidden_state"):
            val = getattr(outputs, attr, None)
            if torch.is_tensor(val):
                if attr == "last_hidden_state":
                    return val[:, 0, :]
                return val
        raise RuntimeError(f"could not extract CLS feature from {type(outputs).__name__}")
