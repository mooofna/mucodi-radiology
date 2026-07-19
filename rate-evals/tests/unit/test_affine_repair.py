"""Unit tests for rate_eval.datasets.affine_repair (CT-RATE V2 degenerate-affine -> 100% coverage)."""

import numpy as np
import pytest

from rate_eval.datasets.affine_repair import affine_is_degenerate, _clean_affine

try:
    from nibabel.orientations import io_orientation
    _HAS_NIB = True
except Exception:  # pragma: no cover
    _HAS_NIB = False


def test_valid_affine_not_degenerate():
    assert not affine_is_degenerate(np.diag([0.75, 0.75, 1.5, 1.0]))


def test_nan_z_column_is_degenerate():
    a = np.diag([0.7, 0.7, 1.5, 1.0]).astype(float)
    a[2, 2] = np.nan  # NaN z -- real RAD/CT-RATE failure mode
    assert affine_is_degenerate(a)


def test_inf_and_singular_are_degenerate():
    inf = np.eye(4); inf[0, 0] = np.inf
    sing = np.zeros((4, 4)); sing[3, 3] = 1.0
    assert affine_is_degenerate(inf) and affine_is_degenerate(sing)


def test_clean_affine_from_zooms_is_finite_and_nonsingular():
    bad = np.diag([0.7, 0.7, 1.5, 1.0]).astype(float); bad[2, 2] = np.nan
    out = _clean_affine(bad, (0.7, 0.7, np.nan))   # NaN zoom -> 1.0 fallback
    assert np.isfinite(out).all()
    assert abs(np.linalg.det(out[:3, :3])) > 1e-9
    assert np.isclose(abs(out[2, 2]), 1.0)


@pytest.mark.skipif(not _HAS_NIB, reason="nibabel required")
def test_clean_affine_passes_io_orientation():
    bad = np.diag([0.7, 0.7, 1.5, 1.0]).astype(float); bad[2, 2] = np.nan
    _ = io_orientation(_clean_affine(bad, (0.7, 0.7, 0.7)))  # would raise LinAlgError on `bad`
