"""CT-CLIP zero-shot (CT-CLIP_v2.pt) rate-evals wrapper."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from ..core.logging import get_logger
from ..core.device import setup_device
from ..config import get_config_value


logger = get_logger(__name__)


def _torch_load_compat(path: str | Path):
    try:
        return torch.load(str(path), map_location="cpu")
    except Exception:
        return torch.load(str(path), map_location="cpu", weights_only=False)


class CTClipZeroShot:
    """CT-CLIP contrastive image latent + optional text-prompt zero-shot scoring."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.model_config = config.model
        self.device = setup_device(get_config_value(config, "device"))

        upstream_dir = Path(get_config_value(self.model_config, "upstream_dir")).resolve()
        ckpt_path = Path(get_config_value(self.model_config, "checkpoint_path")).resolve()
        if not ckpt_path.exists():
            raise FileNotFoundError(f"CT-CLIP checkpoint not found: {ckpt_path}")
        # persist path for subprocess re-import
        from .upstream_paths import add_upstream_to_path
        add_upstream_to_path(upstream_dir / "scripts")

        from transformer_maskgit import CTViT  # type: ignore[import-not-found]
        from transformers import BertModel, BertTokenizer
        from ct_clip import CTCLIP  # type: ignore[import-not-found]

        # verbatim upstream scripts/run_zero_shot.py hyperparams
        image_enc = CTViT(
            dim=512, codebook_size=8192, image_size=480, patch_size=20,
            temporal_patch_size=10, spatial_depth=4, temporal_depth=4,
            dim_head=32, heads=8,
        )
        text_enc = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
        clip = CTCLIP(
            image_encoder=image_enc, text_encoder=text_enc,
            dim_image=294912, dim_text=768, dim_latent=512,
            extra_latent_projection=False, use_mlm=False,
            downsample_image_embeds=False, use_all_token_embeds=False,
        )
        state = _torch_load_compat(ckpt_path)
        msg = clip.load_state_dict(state, strict=False)
        logger.info(
            "CT-CLIP loaded from %s (missing=%d, unexpected=%d)",
            ckpt_path.name, len(msg.missing_keys), len(msg.unexpected_keys),
        )

        clip.eval().to(self.device)
        self.model = clip

        # tokenizer built lazily
        self._tokenizer_cls = BertTokenizer
        self._tokenizer: Any = None
        self._text_latent_cache: dict[Tuple[str, ...], torch.Tensor] = {}

    @staticmethod
    def preprocess_single(image: torch.Tensor, model_config: Any = None, metadata: Any = None, modality: str | None = None) -> torch.Tensor:
        return image

    def preprocess(self, volumes: torch.Tensor, modality: str) -> torch.Tensor:
        return volumes

    @torch.no_grad()
    def _image_latent(self, inputs: torch.Tensor) -> torch.Tensor:
        """CLIP-aligned image latent (B, 512), unnormalized."""
        inputs = inputs.to(self.device)
        enc = self.model.visual_transformer(inputs, return_encoded_tokens=True)
        # mean over T, reshape to 294912 image_embeds
        enc = enc.mean(dim=1)                              # (B, 24, 24, 512)
        image_embeds = enc.reshape(enc.shape[0], -1)       # (B, 294912) == dim_image
        return self.model.to_visual_latent(image_embeds)   # (B, 512), unnormalized

    @torch.no_grad()
    def _image_latent_masked(self, inputs: torch.Tensor) -> torch.Tensor:
        """Mask-aware patch-mean (B, 512) over pre-VQ continuous transformer features."""
        image = inputs[:, 0:1].to(self.device)             # (B, 1, 240, 480, 480)
        mask = inputs[:, 1].to(self.device)                # (B, 240, 480, 480)
        vt = self.model.visual_transformer
        tokens = vt.to_patch_emb(image)                    # (B, T=24, H=24, W=24, 512)
        tokens = vt.encode(tokens)                         # continuous, same shape, pre-VQ
        # anisotropic max-pool: temporal stride 10, spatial stride 20
        mask_grid = F.max_pool3d(
            mask.unsqueeze(1), kernel_size=(10, 20, 20), stride=(10, 20, 20)
        ).squeeze(1)                                       # (B, 24, 24, 24) binary
        B = tokens.shape[0]
        out = torch.zeros((B, tokens.shape[-1]), dtype=tokens.dtype, device=tokens.device)
        for b in range(B):
            sel = mask_grid[b] > 0
            if sel.any():
                out[b] = tokens[b][sel].mean(dim=0)
            else:
                logger.warning(
                    "CTClipZeroShot: empty mask at batch index %d, falling back to mean(all_patch_tokens)", b,
                )
                out[b] = tokens[b].reshape(-1, tokens.shape[-1]).mean(dim=0)
        return out

    @torch.no_grad()
    def extract_features(self, inputs: torch.Tensor, modality: str = "chest_ct", extra_infos: Any = None) -> np.ndarray:
        """Frozen-feature representation for a downstream linear/MLP probe."""
        if inputs.ndim != 5:
            raise ValueError(f"CTClipZeroShot expects (B, C, D, H, W), got {tuple(inputs.shape)}")
        C = inputs.shape[1]
        if C not in (1, 2):
            raise ValueError(f"CTClipZeroShot expects C=1 or C=2, got C={C}")
        if C == 1:
            return self._image_latent(inputs).cpu().numpy().astype(np.float32)
        return self._image_latent_masked(inputs).cpu().numpy().astype(np.float32)

    def _ensure_tokenizer(self) -> None:
        if self._tokenizer is None:
            self._tokenizer = self._tokenizer_cls.from_pretrained(
                "microsoft/BiomedVLP-CXR-BERT-specialized", do_lower_case=True,
            )

    @torch.no_grad()
    def _encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        """Encode K prompts to L2-normalized text latents (K, 512), cached per prompt set."""
        key = tuple(prompts)
        cached = self._text_latent_cache.get(key)
        if cached is not None:
            return cached

        self._ensure_tokenizer()
        text_tokens = self._tokenizer(
            list(prompts), return_tensors="pt", padding="max_length",
            truncation=True, max_length=512,
        ).to(self.device)

        text_out = self.model.text_transformer(
            input_ids=text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
        )
        text_hidden = text_out.last_hidden_state if hasattr(text_out, "last_hidden_state") else text_out[0]
        text_embeds = text_hidden[:, 0, :]                # (K, 768) CLS
        text_latents = self.model.to_text_latent(text_embeds)  # (K, 512)
        text_latents = F.normalize(text_latents, dim=-1)
        self._text_latent_cache[key] = text_latents.detach()
        return text_latents

    @torch.no_grad()
    def score_from_latent(
        self,
        image_latents: torch.Tensor,
        prompts: Sequence[str],
    ) -> np.ndarray:
        """Score precomputed (B, 512) image latents against a (positive, negative) prompt pair."""
        if len(prompts) != 2:
            raise ValueError(
                f"score_from_latent expects 2 prompts (present, not_present); got {len(prompts)}",
            )
        image_latents = F.normalize(image_latents.to(self.device), dim=-1)
        text_latents = self._encode_prompts(prompts)                     # (2, 512), L2-normalized
        temp = self.model.temperature.exp()
        logits = image_latents @ text_latents.t() * temp                 # (B, 2)
        probs = torch.softmax(logits, dim=-1)
        return probs[:, 0].cpu().numpy().astype(np.float32)              # P("present")

    @torch.no_grad()
    def extract_zero_shot_score(
        self,
        inputs: torch.Tensor,
        prompts: Sequence[str] = ("Lung nodule is present.", "Lung nodule is not present."),
    ) -> np.ndarray:
        """Per-volume probability of prompts[0] (the "present" class)."""
        return self.score_from_latent(self._image_latent(inputs), prompts)

    def eval(self) -> "CTClipZeroShot":
        if hasattr(self, "model"):
            self.model.eval()
        return self
