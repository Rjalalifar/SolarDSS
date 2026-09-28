"""Horn (1981) slope and aspect on planar surfaces."""

from __future__ import annotations

import numpy as np
import pytest

from solardss.exceptions import InvalidGridError
from solardss.terrain import ElevationGrid, horn_slope_aspect


def test_plane_sloping_east_faces_west() -> None:
    dx = 10.0
    cols = np.arange(5)
    z = np.tile(0.1 * cols * dx, (5, 1))
    result = horn_slope_aspect(ElevationGrid(z=z, cell_size_x=dx, cell_size_y=dx))
    interior_slope = result.slope_deg.values[1:-1, 1:-1]
    interior_aspect = result.aspect_deg.values[1:-1, 1:-1]
    expected_slope = np.degrees(np.arctan(0.1))
    np.testing.assert_allclose(interior_slope, expected_slope, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(interior_aspect, 270.0, rtol=1e-12, atol=1e-10)
    assert np.isnan(result.slope_deg.values[0, 0])


def test_plane_higher_in_the_north_faces_south() -> None:
    dy = 5.0
    rows = np.arange(5)
    # Row 0 is north, so northing decreases as row increases.
    y_north = (4 - rows) * dy
    z = np.repeat((0.2 * y_north)[:, None], 5, axis=1)
    result = horn_slope_aspect(ElevationGrid(z=z, cell_size_x=dy, cell_size_y=dy))
    interior_slope = result.slope_deg.values[1:-1, 1:-1]
    interior_aspect = result.aspect_deg.values[1:-1, 1:-1]
    expected_slope = np.degrees(np.arctan(0.2))
    np.testing.assert_allclose(interior_slope, expected_slope, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(interior_aspect, 180.0, rtol=1e-12, atol=1e-10)


def test_flat_has_nan_aspect() -> None:
    z = np.zeros((4, 4))
    result = horn_slope_aspect(ElevationGrid(z=z, cell_size_x=1.0, cell_size_y=1.0))
    interior_slope = result.slope_deg.values[1:-1, 1:-1]
    interior_aspect = result.aspect_deg.values[1:-1, 1:-1]
    np.testing.assert_allclose(interior_slope, 0.0)
    assert np.all(np.isnan(interior_aspect))


def test_grid_too_small() -> None:
    with pytest.raises(InvalidGridError):
        horn_slope_aspect(ElevationGrid(z=np.zeros((2, 2)), cell_size_x=1.0, cell_size_y=1.0))


def test_geographic_crs_is_rejected() -> None:
    z = np.zeros((4, 4))
    with pytest.raises(InvalidGridError, match="reproject"):
        horn_slope_aspect(
            ElevationGrid(z=z, cell_size_x=0.0003, cell_size_y=0.0003, crs="EPSG:4326")
        )
