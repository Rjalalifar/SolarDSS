"""Copernicus-compatible DEM provider: injected source, provenance, no download."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from solardss.core.arrays import FloatArray
from solardss.data.dem import (
    COPERNICUS_DEM_GLO30_ID,
    COPERNICUS_DEM_SOURCE_NAME,
    CopernicusDEMProvider,
    DEMRetriever,
    DEMTerrainProvider,
    MemoryDEMSource,
    project_window_for_analysis,
)
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.indicators.ids import IndicatorId
from solardss.provenance.observation import (
    IndicatorObservation,
    ObservationStatus,
    QualityFlag,
    ValueOrigin,
)
from solardss.providers.cache import MemoryCache
from solardss.providers.clock import FixedClock
from solardss.providers.contracts import AvailabilityStatus, HealthStatus
from solardss.providers.protocols import TerrainProvider
from solardss.providers.request import FetchRequest
from solardss.terrain.analysis import CALC_ASPECT_CIRCULAR, CALC_HORN, CALC_RILEY_TRI

T1 = datetime(2024, 1, 15, 12, 0, tzinfo=timezone.utc)
CRS = "EPSG:32632"
DX = 10.0
ORIGIN_X = 500_000.0
ORIGIN_Y = 5_800_000.0
EAST_GRADE = 0.1
EXPECTED_SLOPE = float(np.degrees(np.arctan(EAST_GRADE)))
EXPECTED_TRI = EAST_GRADE * DX * np.sqrt(6.0)
INTERIOR_BBOX = BoundingBox(
    min_lon=500_010.0,
    min_lat=5_799_960.0,
    max_lon=500_040.0,
    max_lat=5_799_990.0,
    crs=CRS,
)
CENTER = GeographicRef(longitude=500_025.0, latitude=5_799_975.0, crs=CRS, name="centre")


def _east_plane(rows: int = 5, cols: int = 5) -> FloatArray:
    return np.tile(EAST_GRADE * np.arange(cols) * DX, (rows, 1)).astype(np.float64)


def _source(*, nodata: float | None = None, z: FloatArray | None = None) -> MemoryDEMSource:
    elevation = _east_plane() if z is None else z
    return MemoryDEMSource(
        elevation=elevation,
        coordinate_system=CRS,
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
        cell_size_x=DX,
        cell_size_y=DX,
        nodata=nodata,
        source_name="synthetic Copernicus-compatible DEM",
        source_resolution="10m",
    )


def _provider(
    source: MemoryDEMSource | None = None,
    *,
    cache: MemoryCache | None = None,
    retriever: DEMRetriever | None = None,
) -> DEMTerrainProvider:
    return DEMTerrainProvider(
        source=_source() if source is None else source,
        cache=cache,
        retriever=retriever,
        clock=FixedClock(T1),
        environ={},
    )


def _request(indicator: IndicatorId, **kwargs: object) -> FetchRequest:
    payload: dict[str, object] = {
        "indicator_id": indicator.value,
        "bbox": INTERIOR_BBOX,
    }
    payload.update(kwargs)
    return FetchRequest.model_validate(payload)


def _observed(observation: IndicatorObservation) -> float:
    assert observation.status is ObservationStatus.OBSERVED
    assert isinstance(observation.value, float)
    return observation.value


def test_alias_and_protocol() -> None:
    provider = _provider()
    assert CopernicusDEMProvider is DEMTerrainProvider
    assert isinstance(provider, TerrainProvider)


def test_not_configured_without_source_or_path() -> None:
    provider = DEMTerrainProvider(clock=FixedClock(T1), environ={})
    assert provider.health().status is HealthStatus.NOT_CONFIGURED
    request = _request(IndicatorId.TERRAIN_SLOPE_MEAN)
    availability = provider.availability(request)
    assert availability.status is AvailabilityStatus.NOT_CONFIGURED
    observation = provider.fetch(request)
    assert observation.status is ObservationStatus.NOT_CONFIGURED
    assert observation.value is None


def test_injected_source_is_used_instead_of_retriever() -> None:
    def boom(_bbox: BoundingBox, _path: Path) -> Path:
        raise AssertionError("retriever must not run when a DEMRasterSource is injected")

    provider = _provider(retriever=boom)
    observation = provider.fetch(_request(IndicatorId.TERRAIN_SLOPE_MEAN))
    assert observation.status is ObservationStatus.OBSERVED
    np.testing.assert_allclose(_observed(observation), EXPECTED_SLOPE, rtol=1e-12, atol=1e-12)


def test_aoi_polygon_and_point_match_analytic_plane() -> None:
    provider = _provider()
    aoi = provider.fetch(_request(IndicatorId.TERRAIN_ELEVATION))
    assert aoi.status is ObservationStatus.OBSERVED
    np.testing.assert_allclose(_observed(aoi), 2.0, rtol=1e-12, atol=1e-12)
    assert aoi.unit == "m"
    assert "median=2.0" in aoi.assumptions
    assert "p10=" in ",".join(aoi.assumptions)
    slope = provider.fetch(_request(IndicatorId.TERRAIN_SLOPE_MEAN))
    slope_max = provider.fetch(_request(IndicatorId.TERRAIN_SLOPE_MAX))
    aspect = provider.fetch(_request(IndicatorId.TERRAIN_ASPECT))
    tri = provider.fetch(_request(IndicatorId.TERRAIN_RUGGEDNESS))
    np.testing.assert_allclose(_observed(slope), EXPECTED_SLOPE, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_observed(slope_max), EXPECTED_SLOPE, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(_observed(aspect), 270.0, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(_observed(tri), EXPECTED_TRI, rtol=1e-12, atol=1e-12)
    assert slope.provenance.calculation_method == CALC_HORN
    assert aspect.provenance.calculation_method == CALC_ASPECT_CIRCULAR
    assert tri.provenance.calculation_method == CALC_RILEY_TRI
    assert slope.quality_flags == (QualityFlag.NOMINAL.value,)

    point = provider.fetch(
        FetchRequest(indicator_id=IndicatorId.TERRAIN_SLOPE_MEAN.value, location=CENTER)
    )
    np.testing.assert_allclose(_observed(point), EXPECTED_SLOPE, rtol=1e-12, atol=1e-12)


def test_candidate_polygon_wkt_zonal_stats() -> None:
    pytest.importorskip("shapely")
    wkt = (
        "POLYGON((500010 5799960, 500040 5799960, 500040 5799990, 500010 5799990, 500010 5799960))"
    )
    provider = _provider()
    observation = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.TERRAIN_ELEVATION.value,
            location=GeographicRef(name="candidate"),
            parameters={"geometry_wkt": wkt, "geometry_crs": CRS},
        )
    )
    assert observation.status is ObservationStatus.OBSERVED
    np.testing.assert_allclose(_observed(observation), 2.0, rtol=1e-12, atol=1e-12)
    assert any(item.startswith("max=") for item in observation.assumptions)
    assert any(item.startswith("median=") for item in observation.assumptions)


def test_horizon_and_shading_are_unavailable() -> None:
    provider = _provider()
    for indicator in (
        IndicatorId.TERRAIN_HORIZON_OBSTRUCTION,
        IndicatorId.TERRAIN_SHADING_POTENTIAL,
    ):
        request = _request(indicator)
        assert provider.availability(request).status is AvailabilityStatus.UNAVAILABLE
        observation = provider.fetch(request)
        assert observation.status is ObservationStatus.UNAVAILABLE
        assert observation.value is None
        assert "not computed" in observation.assumptions[0]


def test_missing_coverage_and_nodata_are_not_hidden() -> None:
    provider = _provider()
    miss = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.TERRAIN_SLOPE_MEAN.value,
            bbox=BoundingBox(
                min_lon=1.0,
                min_lat=1.0,
                max_lon=2.0,
                max_lat=2.0,
                crs=CRS,
            ),
        )
    )
    assert miss.status is ObservationStatus.MISSING_COVERAGE
    assert miss.value is None

    z = _east_plane()
    z[2, 2] = -9999.0
    dirty = _provider(_source(nodata=-9999.0, z=z))
    slope = dirty.fetch(_request(IndicatorId.TERRAIN_SLOPE_MEAN))
    elevation = dirty.fetch(_request(IndicatorId.TERRAIN_ELEVATION))
    assert slope.status is ObservationStatus.UNAVAILABLE
    assert slope.value is None
    assert elevation.status is ObservationStatus.OBSERVED
    assert QualityFlag.UNKNOWN.value in elevation.quality_flags
    assert any("nodata" in item for item in elevation.assumptions)


def test_full_aoi_includes_edge_quality_flag() -> None:
    provider = _provider()
    full = BoundingBox(
        min_lon=ORIGIN_X,
        min_lat=ORIGIN_Y - 5 * DX,
        max_lon=ORIGIN_X + 5 * DX,
        max_lat=ORIGIN_Y,
        crs=CRS,
    )
    observation = provider.fetch(
        FetchRequest(indicator_id=IndicatorId.TERRAIN_SLOPE_MEAN.value, bbox=full)
    )
    assert observation.status is ObservationStatus.OBSERVED
    np.testing.assert_allclose(_observed(observation), EXPECTED_SLOPE, rtol=1e-12, atol=1e-12)
    assert QualityFlag.UNKNOWN.value in observation.quality_flags
    assert any("zonal_nodata_count=" in item for item in observation.assumptions)


def test_provenance_records_source_resolution_crs_and_method() -> None:
    provider = _provider()
    observation = provider.fetch(_request(IndicatorId.TERRAIN_SLOPE_MEAN))
    assert observation.provenance.source_name == COPERNICUS_DEM_SOURCE_NAME
    assert observation.provenance.source_version == COPERNICUS_DEM_GLO30_ID
    assert observation.provenance.value_origin is ValueOrigin.DERIVED
    assert observation.provenance.calculation_method == CALC_HORN
    assert observation.spatial.crs == CRS
    assert observation.spatial.resolution == "10m"
    assert observation.spatial.native_cell_size_m == DX
    notes = observation.provenance.dataset.notes if observation.provenance.dataset else ()
    joined = " ".join(notes)
    assert "native_crs=EPSG:32632" in joined
    assert "analysis_crs=EPSG:32632" in joined
    assert "source_resolution=10m" in joined
    assert any(item.startswith("processing=") for item in observation.assumptions)
    assert any(item.startswith("statistic=") for item in observation.assumptions)
    assert any("DEM source=" in item for item in observation.assumptions)


def test_fetch_terrain_returns_all_computed_indicators() -> None:
    provider = _provider()
    bundle = provider.fetch_terrain(_request(IndicatorId.TERRAIN_SLOPE_MEAN))
    ids = {item.indicator_id for item in bundle.observations}
    assert ids == {
        IndicatorId.TERRAIN_ELEVATION.value,
        IndicatorId.TERRAIN_SLOPE_MEAN.value,
        IndicatorId.TERRAIN_SLOPE_MAX.value,
        IndicatorId.TERRAIN_ASPECT.value,
        IndicatorId.TERRAIN_RUGGEDNESS.value,
    }
    assert bundle.analysis is not None
    assert bundle.analysis_crs == CRS


def test_cache_round_trip() -> None:
    cache = MemoryCache()
    provider = _provider(cache=cache)
    first = provider.fetch(_request(IndicatorId.TERRAIN_SLOPE_MEAN))
    second = provider.fetch(_request(IndicatorId.TERRAIN_SLOPE_MEAN))
    assert first.value == second.value
    assert first.provenance.request_hash == second.provenance.request_hash


def test_projected_window_is_not_warped() -> None:
    source = _source()
    window = source.read_window(INTERIOR_BBOX, pad_cells=2)
    grid = project_window_for_analysis(window)
    assert grid.crs == CRS
    np.testing.assert_allclose(grid.cell_size_x, DX)
    np.testing.assert_allclose(grid.cell_size_y, DX)


def test_geographic_window_is_warped_to_utm() -> None:
    pytest.importorskip("rasterio")
    z = _east_plane()
    source = MemoryDEMSource(
        elevation=z,
        coordinate_system="EPSG:4326",
        origin_x=9.0,
        origin_y=48.5015,
        cell_size_x=0.0003,
        cell_size_y=0.0003,
        source_resolution="~30m",
    )
    window = source.read_window(source.coverage_bbox() or INTERIOR_BBOX, pad_cells=0)
    grid = project_window_for_analysis(window)
    assert grid.crs is not None
    assert grid.crs.startswith("EPSG:326")
    assert grid.cell_size_x > 1.0
    assert grid.cell_size_y > 1.0
