"""ESA WorldCover provider: injected source, explicit profiles, no download."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from solardss.core.arrays import IntArray
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.data.worldcover import (
    ESA_WORLDCOVER_ID,
    ESA_WORLDCOVER_SOURCE_NAME,
    ESA_WORLDCOVER_VERSION,
    ESA_WORLDCOVER_YEAR,
    WORLDCOVER_BARE_SPARSE_ONLY,
    ESAWorldCoverProvider,
    LandCoverRasterProvider,
    LandCoverRetriever,
    MemoryLandCoverSource,
    WorldCoverLandCoverProvider,
)
from solardss.indicators.ids import IndicatorId
from solardss.indicators.land import CALC_AVAILABLE_AREA, CALC_CLASS_FRACTION, CALC_DOMINANT_CLASS
from solardss.provenance.observation import (
    IndicatorObservation,
    ObservationStatus,
    QualityFlag,
    ValueOrigin,
)
from solardss.providers.cache import MemoryCache
from solardss.providers.clock import FixedClock
from solardss.providers.contracts import HealthStatus
from solardss.providers.protocols import LandCoverProvider
from solardss.providers.request import FetchRequest

T1 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
CRS = "EPSG:32632"
DX = 10.0
ORIGIN_X = 500_000.0
ORIGIN_Y = 5_800_000.0
FULL_BBOX = BoundingBox(
    min_lon=ORIGIN_X,
    min_lat=ORIGIN_Y - 4 * DX,
    max_lon=ORIGIN_X + 4 * DX,
    max_lat=ORIGIN_Y,
    crs=CRS,
)


def _codes() -> IntArray:
    return np.asarray(
        [
            [40, 40, 40, 40],
            [40, 40, 40, 40],
            [10, 10, 10, 10],
            [80, 80, 50, 50],
        ],
        dtype=np.int16,
    )


def _source(codes: IntArray | None = None) -> MemoryLandCoverSource:
    return MemoryLandCoverSource(
        codes=_codes() if codes is None else codes,
        coordinate_system=CRS,
        origin_x=ORIGIN_X,
        origin_y=ORIGIN_Y,
        cell_size_x=DX,
        cell_size_y=DX,
        source_name="synthetic WorldCover",
        source_resolution="10m",
        source_version=ESA_WORLDCOVER_VERSION,
        source_year=ESA_WORLDCOVER_YEAR,
    )


def _provider(
    source: MemoryLandCoverSource | None = None,
    *,
    cache: MemoryCache | None = None,
    retriever: LandCoverRetriever | None = None,
) -> ESAWorldCoverProvider:
    return ESAWorldCoverProvider(
        source=_source() if source is None else source,
        cache=cache,
        retriever=retriever,
        clock=FixedClock(T1),
        environ={},
    )


def _request(indicator: IndicatorId, **kwargs: object) -> FetchRequest:
    payload: dict[str, object] = {"indicator_id": indicator.value, "bbox": FULL_BBOX}
    payload.update(kwargs)
    return FetchRequest.model_validate(payload)


def _observed(observation: IndicatorObservation) -> float | str:
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.value is not None
    return observation.value


def _observed_float(observation: IndicatorObservation) -> float:
    value = _observed(observation)
    assert isinstance(value, float)
    return value


def test_alias_and_protocol() -> None:
    provider = _provider()
    assert ESAWorldCoverProvider is WorldCoverLandCoverProvider
    assert isinstance(provider, LandCoverProvider)
    assert isinstance(provider, LandCoverRasterProvider)


def test_not_configured_without_source() -> None:
    provider = ESAWorldCoverProvider(clock=FixedClock(T1), environ={})
    assert provider.health().status is HealthStatus.NOT_CONFIGURED
    observation = provider.fetch(_request(IndicatorId.LAND_FOREST_FRACTION))
    assert observation.status is ObservationStatus.NOT_CONFIGURED
    assert observation.value is None


def test_injected_source_skips_retriever() -> None:
    def boom(_bbox: BoundingBox, _path: Path) -> Path:
        raise AssertionError("retriever must not run when a LandCoverRasterSource is injected")

    provider = _provider(retriever=boom)
    forest = provider.fetch(_request(IndicatorId.LAND_FOREST_FRACTION))
    np.testing.assert_allclose(_observed_float(forest), 0.25)


def test_known_proportions_and_dominant_class() -> None:
    provider = _provider()
    cover = provider.fetch(_request(IndicatorId.LAND_COVER))
    agri = provider.fetch(_request(IndicatorId.LAND_AGRICULTURAL_FRACTION))
    water = provider.fetch(_request(IndicatorId.LAND_WATER_FRACTION))
    forest = provider.fetch(_request(IndicatorId.LAND_FOREST_FRACTION))
    built = provider.fetch(_request(IndicatorId.LAND_BUILTUP_FRACTION))
    wetland = provider.fetch(_request(IndicatorId.LAND_WETLAND_FRACTION))
    bare = provider.fetch(_request(IndicatorId.LAND_BARE_LAND_FRACTION))
    assert _observed(cover) == "agricultural"
    np.testing.assert_allclose(_observed_float(agri), 0.5)
    np.testing.assert_allclose(_observed_float(water), 0.125)
    np.testing.assert_allclose(_observed_float(forest), 0.25)
    np.testing.assert_allclose(_observed_float(built), 0.125)
    np.testing.assert_allclose(_observed_float(wetland), 0.0)
    np.testing.assert_allclose(_observed_float(bare), 0.0)
    assert agri.provenance.calculation_method == CALC_CLASS_FRACTION
    assert cover.provenance.calculation_method == CALC_DOMINANT_CLASS
    assert cover.provenance.value_origin is ValueOrigin.DERIVED
    assert agri.quality_flags == (QualityFlag.NOMINAL.value,)
    assert any(item.startswith("agricultural:") for item in agri.assumptions)
    assert any("percent=" in item for item in agri.assumptions)


def test_available_area_requires_explicit_profile() -> None:
    provider = _provider()
    missing = provider.fetch(_request(IndicatorId.LAND_AVAILABLE_AREA))
    assert missing.status is ObservationStatus.NOT_CONFIGURED
    assert missing.value is None
    assert "non-water" in missing.assumptions[0]
    bare = provider.fetch(
        _request(
            IndicatorId.LAND_AVAILABLE_AREA,
            parameters={"available_land_profile": WORLDCOVER_BARE_SPARSE_ONLY.profile_id},
        )
    )
    np.testing.assert_allclose(_observed_float(bare), 0.0)
    crops = provider.fetch(
        _request(
            IndicatorId.LAND_AVAILABLE_AREA,
            parameters={
                "available_categories": ["agricultural"],
                "excluded_categories": ["water", "forest", "builtup"],
            },
        )
    )
    np.testing.assert_allclose(_observed_float(crops), 800.0)
    assert crops.provenance.calculation_method == CALC_AVAILABLE_AREA
    assert any("non-water is not automatically available" in item for item in crops.assumptions)


def test_point_site_returns_cell_class() -> None:
    provider = _provider()
    # Centre of the southern-west water cell (row 3, col 0).
    point = GeographicRef(
        longitude=ORIGIN_X + 0.5 * DX,
        latitude=ORIGIN_Y - 3.5 * DX,
        crs=CRS,
        name="water-cell",
    )
    cover = provider.fetch(FetchRequest(indicator_id=IndicatorId.LAND_COVER.value, location=point))
    assert _observed(cover) == "water"
    water = provider.fetch(
        FetchRequest(indicator_id=IndicatorId.LAND_WATER_FRACTION.value, location=point)
    )
    np.testing.assert_allclose(_observed_float(water), 1.0)


def test_candidate_polygon_wkt() -> None:
    pytest.importorskip("shapely")
    wkt = (
        "POLYGON((500000 5799980, 500040 5799980, 500040 5800000, 500000 5800000, 500000 5799980))"
    )
    provider = _provider()
    agri = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.LAND_AGRICULTURAL_FRACTION.value,
            location=GeographicRef(name="candidate"),
            parameters={"geometry_wkt": wkt, "geometry_crs": CRS},
        )
    )
    np.testing.assert_allclose(_observed_float(agri), 1.0)


def test_missing_coverage_and_provenance() -> None:
    provider = _provider()
    miss = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.LAND_COVER.value,
            bbox=BoundingBox(min_lon=1.0, min_lat=1.0, max_lon=2.0, max_lat=2.0, crs=CRS),
        )
    )
    assert miss.status is ObservationStatus.MISSING_COVERAGE
    cover = provider.fetch(_request(IndicatorId.LAND_COVER))
    assert cover.provenance.source_name == ESA_WORLDCOVER_SOURCE_NAME
    assert cover.provenance.source_version == f"{ESA_WORLDCOVER_VERSION}-{ESA_WORLDCOVER_YEAR}"
    assert cover.spatial.resolution == "10m"
    notes = cover.provenance.dataset.notes if cover.provenance.dataset else ()
    joined = " ".join(notes)
    assert ESA_WORLDCOVER_ID in joined
    assert f"source_year={ESA_WORLDCOVER_YEAR}" in joined
    assert f"source_version={ESA_WORLDCOVER_VERSION}" in joined
    assert "area_method=" in joined


def test_cache_and_bundle() -> None:
    cache = MemoryCache()
    provider = _provider(cache=cache)
    first = provider.fetch(_request(IndicatorId.LAND_FOREST_FRACTION))
    second = provider.fetch(_request(IndicatorId.LAND_FOREST_FRACTION))
    assert first.value == second.value
    bundle = provider.fetch_land_cover(_request(IndicatorId.LAND_COVER))
    ids = {item.indicator_id for item in bundle.observations}
    assert IndicatorId.LAND_COVER.value in ids
    assert IndicatorId.LAND_FOREST_FRACTION.value in ids
    assert IndicatorId.LAND_AVAILABLE_AREA.value in ids
    available = next(
        item
        for item in bundle.observations
        if item.indicator_id == IndicatorId.LAND_AVAILABLE_AREA.value
    )
    assert available.status is ObservationStatus.NOT_CONFIGURED
