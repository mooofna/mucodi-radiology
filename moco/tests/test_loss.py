"""CPU-only tests for the contrastive KD loss core (moco/loss.py + feature_bank.py)."""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from moco.loss import KDInfoNCELoss


def test_forward_single_returns_2tau_ce():
    """forward_single == 2*tau*CE(normalize(q)@k.T / tau, labels), exactly."""
    torch.manual_seed(0)
    B, d, n_keys = 4, 8, 6
    q = torch.randn(B, d)
    k_all = torch.randn(n_keys, d)
    labels = torch.arange(B)
    tau = 0.3

    out = KDInfoNCELoss(temperature=tau).forward_single(q, k_all, labels)

    qn = F.normalize(q, dim=-1)
    expected = 2 * tau * F.cross_entropy((qn @ k_all.T) / tau, labels)
    assert torch.allclose(out, expected, atol=1e-6)


def test_2tau_cancels_inverse_tau_grad_blowup():
    """The 2*tau factor cancels the 1/tau in the CE gradient."""
    torch.manual_seed(0)
    B, d, n_keys = 8, 16, 16
    q0 = torch.randn(B, d)
    k_all = F.normalize(torch.randn(n_keys, d), dim=-1)
    labels = torch.arange(B)

    def grad_norm(tau: float, scaled: bool) -> float:
        q = q0.clone().requires_grad_(True)
        logits = (F.normalize(q, dim=-1) @ k_all.T) / tau
        ce = F.cross_entropy(logits, labels)
        loss = (2 * tau * ce) if scaled else ce
        loss.backward()
        return float(q.grad.norm())

    bare_ratio = grad_norm(0.05, scaled=False) / grad_norm(0.2, scaled=False)
    scaled_ratio = grad_norm(0.05, scaled=True) / grad_norm(0.2, scaled=True)

    assert bare_ratio > 2.5, f"bare CE grad should blow up ~1/tau, ratio={bare_ratio}"
    assert scaled_ratio < 2.0, f"2*tau*CE grad should stay bounded, ratio={scaled_ratio}"


def test_forward_bank_matches_forward_single_without_bias():
    """forward_bank reuses forward_single verbatim: pos+neg rows == an explicit k_all."""
    torch.manual_seed(0)
    B, d, M = 5, 8, 7
    q = torch.randn(B, d)
    pos_rows = F.normalize(torch.randn(B, d), dim=-1)
    neg_rows = F.normalize(torch.randn(M, d), dim=-1)
    loss = KDInfoNCELoss(temperature=0.2)

    bank_loss = loss.forward_bank(q, pos_rows, neg_rows)
    # explicit k_all: positives first, diag labels
    k_all = torch.cat([pos_rows, neg_rows], dim=0)
    single_loss = loss.forward_single(q, k_all, torch.arange(B))
    assert torch.allclose(bank_loss, single_loss, atol=1e-6)


@pytest.fixture(scope="module")
def gloo_pg():
    """Single-process gloo process group (world_size=1) for all_gather paths."""
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        yield
        return
    if not dist.is_available():
        pytest.skip("torch.distributed unavailable")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    try:
        dist.init_process_group("gloo", rank=0, world_size=1)
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"cannot init gloo process group: {exc}")
    yield
    dist.destroy_process_group()


def test_gather_keys_multi_normalizes_before_concat(gloo_pg):
    """Each teacher's keys are L2-normalized per teacher before concat (world_size=1)."""
    torch.manual_seed(0)
    B = 3
    k = {"t1": torch.randn(B, 4) * 5.0, "t2": torch.randn(B, 6) * 0.1}
    out = KDInfoNCELoss(temperature=0.2).gather_keys_multi(k)

    assert set(out) == {"t1", "t2"}
    for name, (k_all, labels) in out.items():
        # world_size=1 => all_gather is identity => k_all == per-teacher normalize(k).
        assert torch.allclose(k_all, F.normalize(k[name], dim=-1), atol=1e-6)
        # unit-norm per row
        assert torch.allclose(k_all.norm(dim=-1), torch.ones(B), atol=1e-5)
        assert torch.equal(labels, torch.arange(B))


def test_false_neg_bias_self_and_study_masking():
    """false_neg_bias masks self / same-study negatives, never the label diagonal."""
    from moco.feature_bank import FrozenTeacherBank

    rows = {"t": F.normalize(torch.randn(5, 4), dim=-1)}
    name_to_row = {f"s{i}": i for i in range(5)}
    # rows 0,1 -> study 0 ; rows 2,3 -> study 1 ; row 4 -> study 2
    study_codes = torch.tensor([0, 0, 1, 1, 2])
    bank = FrozenTeacherBank(rows, name_to_row, study_codes)

    pos_idx = torch.tensor([0, 2])      # B=2 queries (rows 0, 2)
    neg_idx = torch.tensor([1, 3, 4])   # M=3 negatives (rows 1, 3, 4)

    # off -> no bias at all.
    assert bank.false_neg_bias(pos_idx, neg_idx, "off", torch.float32) is None

    # self: negatives equal to a query's own row index
    b_self = bank.false_neg_bias(pos_idx, neg_idx, "self", torch.float32)
    assert b_self.shape == (2, 2 + 3)
    assert torch.isinf(b_self[:, 2:]).sum() == 0           # no self-collision among negs
    assert b_self[0, 0] == 0 and b_self[1, 1] == 0         # diagonal never masked

    # study: same-study negatives get masked
    b_study = bank.false_neg_bias(pos_idx, neg_idx, "study", torch.float32)
    neg = b_study[:, 2:]
    assert torch.isinf(neg[0, 0])           # q0 vs neg row1  (both study 0)
    assert torch.isinf(neg[1, 1])           # q1 vs neg row3  (both study 1)
    assert not torch.isinf(neg[0, 1])       # q0 (study 0) vs neg row3 (study 1) -> kept
    assert torch.isinf(b_study).sum() >= 2  # at least the two true collisions
