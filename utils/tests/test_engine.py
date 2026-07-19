"""CPU-only tests for the uniform multi-teacher backward (_dispatch_backward)."""
from __future__ import annotations

import torch

from utils.engine import _dispatch_backward


def test_dispatch_backward_uniform_sum_scaled_grad():
    """grad == scale * sum_t d(loss_t)/dx  (uniform sum, single scaled backward)."""
    x = torch.tensor(2.0, requires_grad=True)
    pairs = [("t1", x * 3.0), ("t2", x * 5.0)]
    scale = 0.25

    _dispatch_backward(pairs, scale)

    # grad = 0.25 * 8 = 2.0
    assert torch.isclose(x.grad, torch.tensor(2.0))


def test_dispatch_backward_uniform_weights_across_teacher_count():
    """Each teacher contributes with weight 1 (uniform); 3 teachers sum cleanly."""
    x = torch.tensor(1.0, requires_grad=True)
    pairs = [("a", x * 1.0), ("b", x * 1.0), ("c", x * 1.0)]
    _dispatch_backward(pairs, scale=1.0)
    # d/dx (x + x + x) = 3
    assert torch.isclose(x.grad, torch.tensor(3.0))


def test_dispatch_backward_empty_is_noop():
    """No teachers -> no backward, no grad populated, no error."""
    x = torch.tensor(1.0, requires_grad=True)
    _dispatch_backward([], scale=0.5)
    assert x.grad is None
