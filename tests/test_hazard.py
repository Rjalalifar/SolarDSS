"""Hazard-map sampling with explicit legends. No default Low/Medium/High scale."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solardss.data.geo import BoundingBox, GeographicRef
from solardss.data.hazard import (
    FLOOD_PATH_ENV,
    FloodRiskProvider,
    HailRiskProvider,
    HazardCatalog,
    HazardFieldMap,
    LandslideRiskProvider,
    MemoryHazardRaster,
    RasterHazardProvider,
    VectorHazardProvider,
    WildfireRiskProvider,
    hazard_available,
    hazard_for_candidates,
    hazard_for_site,
    load_hazard_mapping,
)
from solardss.exceptions import InvalidParameterError
from solardss.indicators.hazard import (
    CALC_HAZARD_SAMPLE,
    HazardAggregation,
    HazardBin,
    HazardClass,
    HazardGrid,
    HazardKind,
    HazardValueMapping,
    aggregate_hazard_cells,
    map_hazard_value,
    mask_from_bbox,
)
from solardss.indicators.ids import IndicatorId
from solardss.provenance.observation import ObservationStatus, QualityFlag, ValueOrigin
from solardss.providers.clock import FixedClock
from solardss.providers.contracts import AvailabilityStatus, HealthStatus
from solardss.providers.protocols import HazardProvider
from solardss.providers.request import FetchRequest

T1 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "hazard"
FLOOD_GRID = FIXTURE_DIR / "flood_grid.json"
FLOOD_MAPPING = FIXTURE_DIR / "flood_mapping.json"
WILDFIRE_GEOJSON = FIXTURE_DIR / "wildfire.geojson"
WILDFIRE_MAPPING = FIXTURE_DIR / "wildfire_mapping.json"


def _flood_request(
    lon: float,
    lat: float,
    *,
    indicator: str = IndicatorId.HAZARD_FLOOD_RISK.value,
) -> FetchRequest:
    return FetchRequest(
        indicator_id=indicator,
        location=GeographicRef(longitude=lon, latitude=lat, name="site", crs="EPSG:4326"),
    )


def _flood_provider(*, mapping_path: Path | None = None) -> FloodRiskProvider:
    return FloodRiskProvider(
        path=FLOOD_GRID,
        mapping_path=mapping_path,
        clock=FixedClock(T1),
        environ={},
    )


def test_empty_mapping_does_not_invent_low_medium_high() -> None:
    category, score, warnings = map_hazard_value(2.0, None)
    assert category is None
    assert score is None
    assert any("Low/Medium/High" in item for item in warnings)

    empty = HazardValueMapping(mapping_id="empty", dataset_id="fixture")
    category, score, warnings = map_hazard_value(2.0, empty)
    assert category is None
    assert score is None
    assert any("not inferred" in item for item in warnings)


def test_unmapped_code_is_not_guessed() -> None:
    mapping = HazardValueMapping(
        mapping_id="partial",
        dataset_id="fixture",
        classes=(HazardClass(raw="1", category="Low", score=0.1),),
    )
    category, score, warnings = map_hazard_value(3.0, mapping)
    assert category is None
    assert score is None
    assert any("not guessed" in item for item in warnings)


def test_rejects_mixed_classes_and_bins_and_hallucinated_tertiles() -> None:
    with pytest.raises(InvalidParameterError, match="cannot mix"):
        HazardValueMapping(
            mapping_id="mixed",
            dataset_id="fixture",
            classes=(HazardClass(raw="1", category="Low"),),
            bins=(HazardBin(min_value=0.0, max_value=1.0, category="Low"),),
        )


def test_mean_refused_unless_legend_is_continuous() -> None:
    grid = HazardGrid(
        values=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        cell_size_x=1.0,
        cell_size_y=1.0,
        origin_x=0.0,
        origin_y=2.0,
    )
    mask = mask_from_bbox(grid, 0.0, 0.0, 2.0, 2.0)
    class_map = HazardValueMapping(
        mapping_id="classes",
        dataset_id="fixture",
        classes=(HazardClass(raw="1", category="Low"),),
    )
    with pytest.raises(InvalidParameterError, match="continuous=True"):
        aggregate_hazard_cells(grid, mask, aggregation=HazardAggregation.MEAN, mapping=class_map)
    continuous = HazardValueMapping(
        mapping_id="score",
        dataset_id="fixture",
        continuous=True,
        bins=(
            HazardBin(min_value=0.0, max_value=2.5, category="Lower", score=0.2),
            HazardBin(
                min_value=2.5,
                max_value=5.0,
                category="Upper",
                score=0.8,
                max_inclusive=True,
            ),
        ),
    )
    sample = aggregate_hazard_cells(
        grid, mask, aggregation=HazardAggregation.MEAN, mapping=continuous
    )
    np.testing.assert_allclose(float(sample.raw_value), 2.5)  # type: ignore[arg-type]


def test_json_grid_without_mapping_keeps_raw_value() -> None:
    provider = _flood_provider()
    assert isinstance(provider, HazardProvider)
    result = hazard_for_site(provider, _flood_request(0.5, 2.5))
    assert result.observation.status is ObservationStatus.OBSERVED
    assert result.record.indicator_id == IndicatorId.HAZARD_FLOOD_RISK.value
    np.testing.assert_allclose(float(result.record.raw_value), 1.0)  # type: ignore[arg-type]
    assert result.record.category is None
    assert result.record.score is None
    assert result.observation.value == 1.0
    assert result.observation.provenance.value_origin is ValueOrigin.RAW
    assert result.record.source_name == "fixture flood RP100"
    assert result.record.source_dataset == "fixture-flood-rp100"
    assert result.record.resolution == "1deg"
    assert result.record.return_period == "100-year"
    assert result.record.source_year == 2020
    assert result.record.source_version == "2020.1"
    assert result.record.uncertainty_notes == ("synthetic raster; not a published flood study",)
    joined = " ".join(result.observation.assumptions)
    assert "Low/Medium/High are not assigned" in joined
    assert "category=None" in joined


def test_explicit_flood_legend_maps_known_codes_only() -> None:
    provider = _flood_provider(mapping_path=FLOOD_MAPPING)
    low = hazard_for_site(provider, _flood_request(0.5, 2.5))
    medium = hazard_for_site(provider, _flood_request(1.5, 2.5))
    high = hazard_for_site(provider, _flood_request(2.5, 2.5))
    assert low.record.category == "Low"
    assert medium.record.category == "Medium"
    assert high.record.category == "High"
    np.testing.assert_allclose(float(low.record.score), 0.25)  # type: ignore[arg-type]
    assert low.observation.value == 0.25
    assert low.observation.provenance.value_origin is ValueOrigin.DERIVED
    assert low.observation.provenance.calculation_method == CALC_HAZARD_SAMPLE
    assert low.record.mapping_id == "fixture-flood-rp100-legend"
    assert low.record.quality_flags == (QualityFlag.NOMINAL.value,)

    mapping = load_hazard_mapping(FLOOD_MAPPING)
    unmapped = HazardValueMapping(
        mapping_id=mapping.mapping_id,
        dataset_id=mapping.dataset_id,
        classes=mapping.classes[:2],
    )
    partial = FloodRiskProvider(
        path=FLOOD_GRID,
        mapping=unmapped,
        clock=FixedClock(T1),
        environ={},
    )
    result = partial.hazard_for_site(_flood_request(2.5, 2.5))
    assert result.record.raw_value == 3.0
    assert result.record.category is None
    assert result.observation.quality_flags == (QualityFlag.UNKNOWN.value,)
    assert any("not guessed" in item for item in result.record.warnings)


def test_nodata_and_out_of_raster_stay_missing() -> None:
    provider = _flood_provider(mapping_path=FLOOD_MAPPING)
    nodata = provider.hazard_for_site(_flood_request(2.5, 1.5))
    assert nodata.observation.status is ObservationStatus.MISSING_COVERAGE
    assert nodata.record.missing is True
    assert nodata.record.category is None
    outside = provider.hazard_for_site(_flood_request(10.0, 10.0))
    assert outside.observation.status is ObservationStatus.MISSING_COVERAGE
    assert outside.record.missing is True


def test_wildfire_geojson_without_gis_extra() -> None:
    field_map = HazardFieldMap(
        value="value",
        category="class",
        return_period="return_period",
        site_id="id",
    )
    unmapped = WildfireRiskProvider(
        path=WILDFIRE_GEOJSON,
        field_map=field_map,
        clock=FixedClock(T1),
        environ={},
    )
    request = FetchRequest(
        indicator_id=IndicatorId.HAZARD_WILDFIRE_RISK.value,
        location=GeographicRef(longitude=13.05, latitude=52.05, name="inside"),
    )
    result = unmapped.hazard_for_site(request)
    assert result.observation.status is ObservationStatus.OBSERVED
    assert result.record.indicator_id == IndicatorId.HAZARD_WILDFIRE_RISK.value
    np.testing.assert_allclose(float(result.record.raw_value), 4.0)  # type: ignore[arg-type]
    assert result.record.category is None
    assert result.record.source_category == "extreme"
    assert result.record.return_period == "annual"

    mapped = WildfireRiskProvider(
        path=WILDFIRE_GEOJSON,
        mapping_path=WILDFIRE_MAPPING,
        field_map=field_map,
        clock=FixedClock(T1),
        environ={},
    )
    labelled = mapped.hazard_for_site(request)
    assert labelled.record.category == "Extreme"
    np.testing.assert_allclose(float(labelled.record.score), 0.9)  # type: ignore[arg-type]
    miss = mapped.hazard_for_site(
        FetchRequest(
            indicator_id=IndicatorId.HAZARD_WILDFIRE_RISK.value,
            location=GeographicRef(longitude=12.0, latitude=52.0, name="outside"),
        )
    )
    assert miss.observation.status is ObservationStatus.MISSING_COVERAGE


def test_flood_path_does_not_configure_wildfire() -> None:
    wildfire = WildfireRiskProvider(environ={FLOOD_PATH_ENV: str(FLOOD_GRID)})
    assert wildfire.health().status is HealthStatus.NOT_CONFIGURED
    availability = hazard_available(wildfire)
    assert availability.status is AvailabilityStatus.NOT_CONFIGURED
    assert "no global hazard dataset" in availability.reason
    flood = FloodRiskProvider(environ={FLOOD_PATH_ENV: str(FLOOD_GRID)})
    assert flood.health().status is HealthStatus.OK


def test_typed_providers_do_not_cross_indicators() -> None:
    flood = _flood_provider(mapping_path=FLOOD_MAPPING)
    wrong = flood.hazard_for_site(
        _flood_request(0.5, 2.5, indicator=IndicatorId.HAZARD_WILDFIRE_RISK.value)
    )
    assert wrong.observation.status is ObservationStatus.UNAVAILABLE
    assert "HAZARD_FLOOD_RISK" in wrong.observation.assumptions[0]


def test_catalog_missing_kind_is_not_configured() -> None:
    catalog = HazardCatalog([_flood_provider(mapping_path=FLOOD_MAPPING)])
    hail_request = FetchRequest(
        indicator_id=IndicatorId.HAZARD_HAIL_RISK.value,
        location=GeographicRef(longitude=0.5, latitude=2.5, name="site"),
    )
    availability = hazard_available(catalog, kind=HazardKind.HAIL, request=hail_request)
    assert availability.status is AvailabilityStatus.NOT_CONFIGURED
    assert "no global hazard dataset" in availability.reason
    result = hazard_for_site(catalog, hail_request, kind=HazardKind.HAIL)
    assert result.observation.status is ObservationStatus.NOT_CONFIGURED
    assert result.record.indicator_id == IndicatorId.HAZARD_HAIL_RISK.value
    flood = hazard_for_site(catalog, _flood_request(0.5, 2.5), kind=HazardKind.FLOOD)
    assert flood.record.category == "Low"


def test_landslide_and_hail_memory_rasters_are_isolated() -> None:
    mapping = HazardValueMapping(
        mapping_id="fixture-landslide-legend",
        dataset_id="fixture-landslide",
        classes=(
            HazardClass(raw="10", category="Low", score=0.2),
            HazardClass(raw="20", category="High", score=0.8),
        ),
    )
    raster = MemoryHazardRaster(
        values=np.asarray([[10.0, 20.0]], dtype=np.float64),
        coordinate_system="EPSG:4326",
        origin_x=0.0,
        origin_y=1.0,
        cell_size_x=1.0,
        cell_size_y=1.0,
        source_name="fixture landslide",
        source_dataset="fixture-landslide",
        source_version="v0",
        source_year=2018,
        resolution="1deg",
        uncertainty_notes=("synthetic susceptibility grid",),
    )
    landslide = LandslideRiskProvider(
        raster_source=raster,
        mapping=mapping,
        clock=FixedClock(T1),
        environ={},
    )
    hail = HailRiskProvider(environ={})
    low = landslide.hazard_for_site(
        FetchRequest(
            indicator_id=IndicatorId.HAZARD_LANDSLIDE_RISK.value,
            location=GeographicRef(longitude=0.5, latitude=0.5, name="west"),
        )
    )
    assert low.record.category == "Low"
    assert low.record.indicator_id == IndicatorId.HAZARD_LANDSLIDE_RISK.value
    assert hail.hazard_available().status is AvailabilityStatus.NOT_CONFIGURED


def test_hazard_for_candidates_prefixes_indicator_columns() -> None:
    provider = _flood_provider(mapping_path=FLOOD_MAPPING)
    table = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "longitude": [0.5, 1.5, 2.5],
            "latitude": [2.5, 2.5, 1.5],
        }
    )
    result = hazard_for_candidates(provider, table)
    frame = result.table.frame
    prefix = IndicatorId.HAZARD_FLOOD_RISK.value
    assert list(frame[f"{prefix}_category"]) == ["Low", "Medium", None]
    assert list(frame[f"{prefix}_status"]) == [
        ObservationStatus.OBSERVED.value,
        ObservationStatus.OBSERVED.value,
        ObservationStatus.MISSING_COVERAGE.value,
    ]
    assert result.records[0].candidate_id == "a"
    assert result.records[2].missing is True


def test_generic_adapters_require_kind() -> None:
    raster = MemoryHazardRaster(
        values=np.asarray([[1.0]], dtype=np.float64),
        coordinate_system="EPSG:4326",
        origin_x=0.0,
        origin_y=1.0,
        cell_size_x=1.0,
        cell_size_y=1.0,
    )
    with pytest.raises(InvalidParameterError, match="requires kind"):
        RasterHazardProvider(raster_source=raster, environ={})
    typed = RasterHazardProvider(
        kind=HazardKind.HAIL, raster_source=raster, clock=FixedClock(T1), environ={}
    )
    assert typed.indicator_id == IndicatorId.HAZARD_HAIL_RISK.value
    with pytest.raises(InvalidParameterError, match="requires kind"):
        VectorHazardProvider(environ={})


def test_unconfigured_provider_reports_no_global_dataset() -> None:
    provider = FloodRiskProvider(environ={})
    assert provider.health().status is HealthStatus.NOT_CONFIGURED
    result = provider.hazard_for_site(_flood_request(0.5, 2.5))
    assert result.observation.status is ObservationStatus.NOT_CONFIGURED
    assert "no global hazard dataset" in result.observation.assumptions[0]


def test_bbox_majority_uses_class_codes_not_mean() -> None:
    provider = _flood_provider(mapping_path=FLOOD_MAPPING)
    observation = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.HAZARD_FLOOD_RISK.value,
            bbox=BoundingBox(min_lon=0.0, min_lat=1.0, max_lon=3.0, max_lat=3.0),
        )
    )
    assert observation.status is ObservationStatus.OBSERVED
    # Valid cells are 1,2,3,1,2 - 1 and 2 are tied; unique order keeps 1.
    assert observation.value == 0.25
    mean = provider.hazard_for_site(
        FetchRequest(
            indicator_id=IndicatorId.HAZARD_FLOOD_RISK.value,
            bbox=BoundingBox(min_lon=0.0, min_lat=1.0, max_lon=3.0, max_lat=3.0),
            parameters={"hazard_aggregation": "mean"},
        )
    )
    assert mean.observation.status is ObservationStatus.UNAVAILABLE
    assert "continuous=True" in mean.observation.assumptions[0]
