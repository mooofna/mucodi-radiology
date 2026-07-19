# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.nn.functional import all_gather

class KDInfoNCELoss(nn.Module):
    """MoCo v3 InfoNCE for KD: student queries vs external teacher keys (no EMA)."""
    def __init__(self, temperature: float = 1.0):
        super().__init__()
        self.tau = temperature

    def gather_keys_multi(self, k_pos_dict: dict[str, torch.Tensor]):
        """Normalize per-teacher (before concat), then all-gather all teachers' keys in one call."""
        if not k_pos_dict:
            return {}
        teacher_names = list(k_pos_dict.keys())
        normalized = [F.normalize(k_pos_dict[t], dim=-1) for t in teacher_names]
        dims = [t.shape[-1] for t in normalized]
        k_cat = torch.cat(normalized, dim=-1)
        k_all_cat = torch.cat(list(all_gather(k_cat)), dim=0)
        k_all_per_teacher = torch.split(k_all_cat, dims, dim=-1)
        B = next(iter(k_pos_dict.values())).size(0)
        labels = torch.distributed.get_rank() * B + torch.arange(B, device=k_cat.device)
        return {t: (k_all_per_teacher[i], labels) for i, t in enumerate(teacher_names)}

    def forward_single(
        self,
        q: torch.Tensor,
        k_all: torch.Tensor,
        labels: torch.Tensor,
        logit_bias: torch.Tensor | None = None,
    ):
        """InfoNCE for one view; 2*tau scaling cancels tau in the CE gradient."""
        q = F.normalize(q, dim=-1)
        logits = (q @ k_all.T) / self.tau
        if logit_bias is not None:
            logits = logits + logit_bias
        return 2 * self.tau * F.cross_entropy(logits, labels)

    def forward_bank(
        self,
        q: torch.Tensor,
        pos_rows: torch.Tensor,
        neg_rows: torch.Tensor,
        logit_bias: torch.Tensor | None = None,
    ):
        """InfoNCE against a frozen feature bank; rows must be pre-L2-normalized."""
        k_all = torch.cat([pos_rows, neg_rows], dim=0).to(q.dtype)
        labels = torch.arange(q.size(0), device=q.device)
        return self.forward_single(q, k_all, labels, logit_bias=logit_bias)

