"""World placement of the bundled model grid (stored as W, H, D for the browser)."""

from __future__ import annotations

import numpy as np

from mucodi_demo.cases import grid_affine


def test_grid_voxels_sit_where_the_resize_samples() -> None:
    canonical = np.array(
        [[0.7, 0, 0, -180.0], [0, 0.68, 0, -170.0], [0, 0, 1.25, -300.0], [0, 0, 0, 1]]
    )
    shape_xyz = (512, 480, 300)
    affine = grid_affine(canonical, shape_xyz)
    rng = np.random.default_rng(0)
    for _ in range(50):
        w, h, d = int(rng.integers(0, 256)), int(rng.integers(0, 256)), int(rng.integers(0, 128))
        x = (h + 0.5) * shape_xyz[0] / 256 - 0.5
        y = (w + 0.5) * shape_xyz[1] / 256 - 0.5
        z = (d + 0.5) * shape_xyz[2] / 128 - 0.5
        np.testing.assert_allclose(affine @ [w, h, d, 1], canonical @ [x, y, z, 1], atol=1e-9)
