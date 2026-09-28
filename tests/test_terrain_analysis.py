"""Analytic terrain analysis: slope, aspect, TRI, zonal stats, CRS guards."""

from __future__ import annotations

import numpy as np
import pytest

from solardss.core.arrays import BoolArray, FloatArray
from solardss.exceptions import InvalidGridError, UnsupportedMethodError
from solardss.terrain.analysis import (
    analyze_terrain,
    circular_mean_deg,
    crs_is_geographic,
    horizon_obstruction,
    require_projected_grid,
    riley_tri,
    shading_potential,
    utm_epsg,
    zonal_statistics,
)
from solardss.terrain.contracts import ElevationGrid

DX = 10.0
EAST_GRADE = 0.1
NORTH_GRADE = 0.2


def _east_plane(rows: int = 5, cols: int = 5) -> FloatArray:
    return np.tile(EAST_GRADE * np.arange(cols) * DX, (rows, 1)).astype(np.float64)


def _north_plane(rows: int = 5, cols: int = 5) -> FloatArray:
    y_from_south = (rows - 1 - np.arange(rows)) * DX
    return np.repeat((NORTH_GRADE * y_from_south)[:, None], cols, axis=1).astype(np.float64)


def _interior_mask(shape: tuple[int, ...]) -> BoolArray:
    mask = np.zeros(shape, dtype=bool)
    mask[1:-1, 1:-1] = True
    return mask


def _stat(value: float | None) -> float:
    assert value is not None
    return value


def test_crs_classification_and_utm_zone() -> None:
    assert crs_is_geographic("EPSG:4326")
    assert crs_is_geographic("WGS 84")
    assert not crs_is_geographic("EPSG:32632")
    assert not crs_is_geographic(None)
    assert utm_epsg(9.0, 48.5) == "EPSG:32632"
    assert utm_epsg(18.0, -33.9) == "EPSG:32734"


def test_require_projected_grid_rejects_geographic() -> None:
    grid = ElevationGrid(z=_east_plane(), cell_size_x=DX, cell_size_y=DX, crs="EPSG:4326")
    with pytest.raises(InvalidGridError, match="reproject"):
        require_projected_grid(grid)


def test_analyze_east_plane_slope_aspect_elevation() -> None:
    z = _east_plane()
    grid = ElevationGrid(z=z, cell_size_x=DX, cell_size_y=DX, crs="EPSG:32632")
    result = analyze_terrain(grid, mask=_interior_mask(z.shape))
    expected_slope = float(np.degrees(np.arctan(EAST_GRADE)))
    np.testing.assert_allclose(
        _stat(result.slope_zonal.mean), expected_slope, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _stat(result.slope_zonal.max), expected_slope, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _stat(result.slope_zonal.median), expected_slope, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _stat(result.aspect_zonal.circular_mean), 270.0, rtol=1e-12, atol=1e-10
    )
    np.testing.assert_allclose(_stat(result.elevation_zonal.mean), 2.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_stat(result.elevation_zonal.median), 2.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_stat(result.elevation_zonal.min), 1.0)
    np.testing.assert_allclose(_stat(result.elevation_zonal.max), 3.0)
    np.testing.assert_allclose(_stat(result.elevation_zonal.p10), 1.0)
    np.testing.assert_allclose(_stat(result.elevation_zonal.p90), 3.0)
    assert result.slope_zonal.nodata_count == 0
    assert result.analysis_crs == "EPSG:32632"


def test_analyze_north_plane_faces_south() -> None:
    z = _north_plane()
    grid = ElevationGrid(z=z, cell_size_x=DX, cell_size_y=DX, crs="EPSG:32632")
    result = analyze_terrain(grid, mask=_interior_mask(z.shape))
    expected_slope = float(np.degrees(np.arctan(NORTH_GRADE)))
    np.testing.assert_allclose(
        _stat(result.slope_zonal.mean), expected_slope, rtol=1e-12, atol=1e-12
    )
    np.testing.assert_allclose(
        _stat(result.aspect_zonal.circular_mean), 180.0, rtol=1e-12, atol=1e-10
    )


def test_riley_tri_east_plane_is_analytic() -> None:
    z = _east_plane()
    grid = ElevationGrid(z=z, cell_size_x=DX, cell_size_y=DX, crs="EPSG:32632")
    tri = riley_tri(grid)
    expected = EAST_GRADE * DX * np.sqrt(6.0)
    np.testing.assert_allclose(tri.values[1:-1, 1:-1], expected, rtol=1e-12, atol=1e-12)
    assert np.isnan(tri.values[0, 0])
    result = analyze_terrain(grid, mask=_interior_mask(z.shape))
    np.testing.assert_allclose(
        _stat(result.ruggedness_zonal.mean), expected, rtol=1e-12, atol=1e-12
    )


def test_riley_tri_rejects_geographic_crs() -> None:
    grid = ElevationGrid(z=_east_plane(), cell_size_x=DX, cell_size_y=DX, crs="EPSG:4326")
    with pytest.raises(InvalidGridError, match="reproject"):
        riley_tri(grid)


def test_nodata_poisons_eight_neighborhood_and_is_reported() -> None:
    z = _east_plane()
    z[2, 2] = -9999.0
    grid = ElevationGrid(z=z, cell_size_x=DX, cell_size_y=DX, nodata=-9999.0, crs="EPSG:32632")
    result = analyze_terrain(grid, mask=_interior_mask(z.shape))
    assert result.elevation_zonal.nodata_count == 1
    assert result.elevation_zonal.finite_count == 8
    assert result.slope_zonal.finite_count == 0
    assert result.quality.nodata_fraction > 0.0
    assert any("nodata" in item or "finite" in item for item in result.quality.warnings)


def test_zonal_percentiles_and_circular_mean() -> None:
    values = np.asarray(np.arange(1.0, 11.0), dtype=np.float64)
    stats = zonal_statistics(values, statistic_used="mean")
    assert stats.min == 1.0
    assert stats.max == 10.0
    np.testing.assert_allclose(_stat(stats.mean), 5.5)
    np.testing.assert_allclose(_stat(stats.median), 5.5)
    np.testing.assert_allclose(_stat(stats.p10), np.percentile(values, 10.0))
    np.testing.assert_allclose(_stat(stats.p90), np.percentile(values, 90.0))
    wrapping = np.asarray([350.0, 10.0], dtype=np.float64)
    assert wrapping.mean() == 180.0
    circular = circular_mean_deg(wrapping)
    assert circular is not None
    assert min(circular, 360.0 - circular) < 1e-9
    empty = circular_mean_deg(np.asarray([np.nan, np.nan], dtype=np.float64))
    assert empty is None


def test_mask_mismatch_and_edge_border() -> None:
    z = _east_plane()
    grid = ElevationGrid(z=z, cell_size_x=DX, cell_size_y=DX, crs="EPSG:32632")
    with pytest.raises(InvalidGridError, match="mask"):
        analyze_terrain(grid, mask=np.ones((3, 3), dtype=bool))
    result = analyze_terrain(grid)
    assert result.quality.edge_cells_excluded is True
    assert result.slope_zonal.nodata_count == 16
    np.testing.assert_allclose(
        _stat(result.slope_zonal.mean),
        float(np.degrees(np.arctan(EAST_GRADE))),
        rtol=1e-12,
        atol=1e-12,
    )


def test_horizon_and_shading_are_explicitly_unimplemented() -> None:
    grid = ElevationGrid(z=_east_plane(), cell_size_x=DX, cell_size_y=DX, crs="EPSG:32632")
    with pytest.raises(UnsupportedMethodError, match="HORIZON"):
        horizon_obstruction(grid)
    with pytest.raises(UnsupportedMethodError, match="SHADING"):
        shading_potential(grid)
