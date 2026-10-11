"""The Python reference preprocessing (rate_eval's own code) against its documented formulas."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from mucodi_demo import reference


def trilinear(x: npt.NDArray[np.float64], shape: tuple[int, int, int]) -> npt.NDArray[np.float64]:
    """Separable linear resize with align_corners=False: src = (dst + 0.5) * in / out - 0.5."""
    out = x
    for axis, n_out in enumerate(shape):
        n_in = out.shape[axis]
        src = np.maximum((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0.0)
        lo = np.minimum(np.floor(src).astype(int), n_in - 1)
        hi = np.minimum(lo + 1, n_in - 1)
        frac = src - lo
        a, b = np.take(out, lo, axis=axis), np.take(out, hi, axis=axis)
        view = [1, 1, 1]
        view[axis] = n_out
        out = a + (b - a) * frac.reshape(view)
    return out


def test_window_order_and_values() -> None:
    assert [name for name, _, _ in reference.WINDOWS] == [
        "lung",
        "mediastinum",
        "soft_tissue",
        "bone",
    ]
    assert [(c, w) for _, c, w in reference.WINDOWS] == [
        (-600.0, 1500.0),
        (50.0, 400.0),
        (50.0, 350.0),
        (400.0, 1800.0),
    ]


def test_windows_clamp_each_range_to_the_unit_interval() -> None:
    hu = np.random.default_rng(0).uniform(-1500, 2000, (6, 10, 12)).astype(np.float32)
    out = reference.window(hu)
    assert out.shape == (1, 4, 6, 10, 12)
    for c, (_, centre, width) in enumerate(reference.WINDOWS):
        expected = np.clip((hu - (centre - width / 2)) / (width + 1e-8), 0.0, 1.0)
        np.testing.assert_allclose(out[0, c], expected, atol=1e-6)


def test_resize_is_trilinear_without_aligned_corners() -> None:
    x = np.random.default_rng(1).normal(size=(5, 7, 9)).astype(np.float32)
    got = reference.resize(x, (3, 11, 4))
    np.testing.assert_allclose(got, trilinear(x.astype(np.float64), (3, 11, 4)), atol=1e-5)
