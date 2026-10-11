"""Feature-cell geometry of the network view, checked by back-propagating from single cells."""

from __future__ import annotations

from itertools import pairwise

import pytest
from torch import nn

from mucodi_demo.export import TAP_NAMES
from mucodi_demo.geometry import impulse_centre, tap_geometry
from mucodi_demo.students import INPUT_SHAPE


def test_taps_cover_the_network_in_order(tmin: nn.Module) -> None:
    geoms = tap_geometry(tmin)  # also asserts every shape is ceil(input / stride)
    assert [g.name for g in geoms] == list(TAP_NAMES)
    assert geoms[0].stride == (2, 2, 2)
    assert geoms[-1].stride == (32, 32, 32)
    assert all(a.rf <= b.rf for a, b in pairwise(geoms))


@pytest.mark.parametrize("tap", range(len(TAP_NAMES)), ids=list(TAP_NAMES))
def test_gradient_centre_is_the_cell_centre(tmin: nn.Module, tap: int) -> None:
    g = tap_geometry(tmin)[tap]
    cell = (g.shape[1] // 2, g.shape[2] // 2, g.shape[3] // 2)
    com = impulse_centre(tmin, tap, cell)
    checked = 0
    for a in range(3):
        centre = g.stride[a] * cell[a] + g.offset[a]
        half = (g.rf[a] - 1) / 2
        if centre - half < 0 or centre + half > INPUT_SHAPE[2 + a] - 1:
            continue  # a field clipped by the volume edge biases the centre of mass inwards
        assert abs(com[a] - centre) <= 0.1 * g.stride[a], (a, com[a], centre)
        checked += 1
    if checked == 0:
        pytest.skip("receptive field wider than the volume on every axis")
