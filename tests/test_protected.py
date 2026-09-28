"""Protected-area overlap science and pluggable cadastre providers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from solardss.data.geo import BoundingBox, GeographicRef
from solardss.data.protected import (
    GENERIC_FIELD_MAP,
    IUCN_IA_IB_HARD_EXCLUSION,
    MemoryProtectedAreaSource,
    RegionalProtectedAreaProvider,
    VectorProtectedAreaProvider,
)
from solardss.exceptions import InvalidParameterError
from solardss.indicators.ids import IndicatorId
from solardss.indicators.protected import (
    CALC_PROTECTED_OVERLAP,
    ConstraintDisposition,
    ProtectedAreaConstraintProfile,
    ProtectedAreaFeature,
    ProtectedAreaRule,
    analyze_protected_overlap,
    bbox_ring,
    clip_polygon,
    hard_exclusion_applies,
    resolve_constraint_profile,
)
from solardss.indicators.rooftop import projected_ring_area_m2
from solardss.provenance.observation import ObservationStatus, ValueOrigin
from solardss.providers.clock import FixedClock
from solardss.providers.contracts import HealthStatus
from solardss.providers.protocols import ProtectedAreaProvider
from solardss.providers.request import FetchRequest

T1 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
CRS = "EPSG:32632"
AOI = bbox_ring(0.0, 0.0, 10.0, 10.0)
AOI_BBOX = BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=10.0, max_lat=10.0, crs=CRS)
FIXTURE = Path(__file__).resolve().parent / "fixtures" / "protected" / "sample.geojson"


def _feature(
    site_id: str,
    ring: tuple[tuple[float, float], ...],
    *,
    category: str | None = None,
    designation: str | None = None,
    holes: tuple[tuple[tuple[float, float], ...], ...] = (),
    source_dataset: str = "test-cadastre",
) -> ProtectedAreaFeature:
    return ProtectedAreaFeature(
        site_id=site_id,
        name=site_id,
        category=category,
        designation=designation,
        coordinates=ring,
        holes=holes,
        source_dataset=source_dataset,
    )


def _provider(features: tuple[ProtectedAreaFeature, ...]) -> VectorProtectedAreaProvider:
    return VectorProtectedAreaProvider(
        source=MemoryProtectedAreaSource(
            features, crs=CRS, source_name="test cadastre", source_dataset="test-cadastre"
        ),
        clock=FixedClock(T1),
        environ={},
        crs=CRS,
    )


def _request(**kwargs: object) -> FetchRequest:
    payload: dict[str, object] = {
        "indicator_id": IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value,
        "bbox": AOI_BBOX,
    }
    payload.update(kwargs)
    return FetchRequest.model_validate(payload)


def test_clip_known_rectangles() -> None:
    subject = bbox_ring(0.0, 0.0, 10.0, 10.0)
    clipper = bbox_ring(5.0, 0.0, 15.0, 10.0)
    inter = clip_polygon(subject, clipper)
    np.testing.assert_allclose(projected_ring_area_m2(inter), 50.0)


def test_known_polygon_overlap_area_and_percent() -> None:
    park = _feature("park-east", bbox_ring(5.0, 0.0, 15.0, 10.0), category="V")
    result = analyze_protected_overlap(AOI, (park,), crs=CRS)
    np.testing.assert_allclose(result.summary.aoi_area_m2, 100.0)
    np.testing.assert_allclose(result.summary.intersection_area_m2, 50.0)
    np.testing.assert_allclose(result.summary.overlap_fraction, 0.5)
    np.testing.assert_allclose(result.summary.overlap_percent, 50.0)
    assert result.summary.site_ids == ("park-east",)
    assert result.summary.categories == ("V",)
    assert result.summary.records[0].source_dataset == "test-cadastre"
    assert result.summary.overall_disposition is ConstraintDisposition.INFORMATIONAL
    assert not hard_exclusion_applies(result.summary)


def test_missing_profile_is_not_a_hard_exclusion() -> None:
    park = _feature("any-park", bbox_ring(0.0, 0.0, 10.0, 10.0), category="II")
    result = analyze_protected_overlap(AOI, (park,), crs=CRS)
    assert result.summary.profile_id is None
    assert result.summary.hard_exclusion_site_ids == ()
    assert any("not a hard exclusion" in item for item in result.summary.warnings)


def test_iucn_ia_ib_profile_excludes_only_listed_classes() -> None:
    strict = _feature("core", bbox_ring(0.0, 0.0, 5.0, 10.0), category="Ia")
    landscape = _feature("scape", bbox_ring(5.0, 0.0, 10.0, 10.0), category="V")
    result = analyze_protected_overlap(
        AOI, (strict, landscape), crs=CRS, profile=IUCN_IA_IB_HARD_EXCLUSION
    )
    assert result.summary.hard_exclusion_site_ids == ("core",)
    assert "scape" not in result.summary.hard_exclusion_site_ids
    assert result.summary.overall_disposition is ConstraintDisposition.HARD_EXCLUSION
    assert hard_exclusion_applies(result.summary)
    records = {item.site_id: item.disposition for item in result.summary.records}
    assert records["scape"] is ConstraintDisposition.INFORMATIONAL


def test_warning_and_penalty_come_from_caller_lists() -> None:
    spa = _feature("spa", bbox_ring(0.0, 0.0, 4.0, 10.0), category="A")
    buffer_zone = _feature("buffer", bbox_ring(6.0, 0.0, 10.0, 10.0), category="buffer")
    profile = resolve_constraint_profile(
        warning_categories=("A",),
        penalty_categories=("buffer",),
        penalty_weight=0.25,
    )
    assert profile is not None
    result = analyze_protected_overlap(AOI, (spa, buffer_zone), crs=CRS, profile=profile)
    assert result.summary.warning_site_ids == ("spa",)
    assert result.summary.penalty_site_ids == ("buffer",)
    assert result.summary.hard_exclusion_site_ids == ()
    assert result.summary.overall_disposition is ConstraintDisposition.PENALTY


def test_rejects_universal_hard_exclusion() -> None:
    with pytest.raises(InvalidParameterError, match="universal hard exclusion"):
        ProtectedAreaConstraintProfile(
            profile_id="exclude-everything",
            unmatched_disposition=ConstraintDisposition.HARD_EXCLUSION,
        )
    with pytest.raises(InvalidParameterError, match="categories or designations"):
        ProtectedAreaRule(disposition=ConstraintDisposition.HARD_EXCLUSION)


def test_disjoint_union_is_exact_overlapping_is_capped() -> None:
    left = _feature("left", bbox_ring(0.0, 0.0, 4.0, 10.0), category="V")
    right = _feature("right", bbox_ring(6.0, 0.0, 10.0, 10.0), category="V")
    disjoint = analyze_protected_overlap(AOI, (left, right), crs=CRS)
    np.testing.assert_allclose(disjoint.summary.intersection_area_m2, 80.0)
    assert disjoint.summary.union_is_exact is True
    overlap_left = _feature("a", bbox_ring(0.0, 0.0, 8.0, 10.0), category="V")
    overlap_right = _feature("b", bbox_ring(2.0, 0.0, 10.0, 10.0), category="II")
    overlapped = analyze_protected_overlap(AOI, (overlap_left, overlap_right), crs=CRS)
    assert overlapped.summary.union_is_exact is False
    np.testing.assert_allclose(overlapped.summary.intersection_area_m2, 100.0)
    assert any(
        "overlapping protected-area designations" in item for item in overlapped.summary.warnings
    )


def test_hole_is_subtracted_from_intersection() -> None:
    park = _feature(
        "donut",
        bbox_ring(0.0, 0.0, 10.0, 10.0),
        category="IV",
        holes=(bbox_ring(2.0, 2.0, 8.0, 8.0),),
    )
    result = analyze_protected_overlap(AOI, (park,), crs=CRS)
    np.testing.assert_allclose(result.summary.intersection_area_m2, 64.0)
    np.testing.assert_allclose(result.summary.overlap_percent, 64.0)


def test_point_inside_has_fraction_one_and_zero_area() -> None:
    park = _feature("core", bbox_ring(0.0, 0.0, 10.0, 10.0), category="Ia")
    provider = _provider((park,))
    observation = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value,
            location=GeographicRef(longitude=3.0, latitude=3.0, crs=CRS, name="inside"),
        )
    )
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.value == 1.0
    assert any("intersection_area_m2=0" in item for item in observation.assumptions)
    assert any("intersection_percent=100" in item for item in observation.assumptions)


def test_vector_provider_known_polygons_and_provenance() -> None:
    park = _feature("park-east", bbox_ring(5.0, 0.0, 15.0, 10.0), category="V")
    provider = _provider((park,))
    assert isinstance(provider, ProtectedAreaProvider)
    observation = provider.fetch(_request())
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.unit == "1"
    np.testing.assert_allclose(float(observation.value), 0.5)  # type: ignore[arg-type]
    assert observation.provenance.calculation_method == CALC_PROTECTED_OVERLAP
    assert observation.provenance.value_origin is ValueOrigin.DERIVED
    assert observation.provenance.source_name == "test cadastre"
    joined = " ".join(observation.assumptions)
    assert "intersection_area_m2=50" in joined
    assert "intersection_percent=50" in joined
    assert "site_ids=park-east" in joined
    assert "categories=V" in joined
    assert "source_dataset=test-cadastre" in joined
    assert "overall_disposition=informational" in joined
    notes = observation.provenance.dataset.notes if observation.provenance.dataset else ()
    assert any("source_dataset=test-cadastre" in item for item in notes)


def test_geojson_file_without_gis_extra() -> None:
    provider = VectorProtectedAreaProvider(
        vector_path=FIXTURE,
        field_map=GENERIC_FIELD_MAP,
        clock=FixedClock(T1),
        environ={},
        source_name="fixture cadastre",
        source_dataset="sample.geojson",
    )
    aoi = BoundingBox(min_lon=13.0, min_lat=52.0, max_lon=13.1, max_lat=52.1, crs="EPSG:4326")
    observation = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value,
            bbox=aoi,
        )
    )
    assert observation.status is ObservationStatus.OBSERVED
    np.testing.assert_allclose(float(observation.value), 0.54, atol=0.02)  # type: ignore[arg-type]
    joined = " ".join(observation.assumptions)
    assert "park-east" in joined
    assert "strict-core" in joined
    with_profile = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value,
            bbox=aoi,
            parameters={"protected_area_profile": IUCN_IA_IB_HARD_EXCLUSION.profile_id},
        )
    )
    assert "strict-core" in " ".join(with_profile.assumptions)
    assert any("hard_exclusion" in item for item in with_profile.assumptions)


def test_empty_cadastre_is_zero_overlap_not_invented() -> None:
    provider = _provider(())
    observation = provider.fetch(_request())
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.value == 0.0
    assert "site_ids=none" in " ".join(observation.assumptions)


def test_not_configured_without_source() -> None:
    provider = VectorProtectedAreaProvider(clock=FixedClock(T1), environ={})
    assert provider.health().status is HealthStatus.NOT_CONFIGURED
    observation = provider.fetch(_request())
    assert observation.status is ObservationStatus.NOT_CONFIGURED
    assert observation.value is None


def test_regional_dispatcher_has_no_global_fallback() -> None:
    local = _provider((_feature("park-east", bbox_ring(5.0, 0.0, 15.0, 10.0), category="V"),))
    regional = RegionalProtectedAreaProvider({"DE": local}, clock=FixedClock(T1))
    assert isinstance(regional, ProtectedAreaProvider)
    missing_region = regional.fetch(_request(parameters={"region_id": "FR"}))
    assert missing_region.status is ObservationStatus.UNSUPPORTED_REGION
    assert missing_region.value is None
    assert (
        "no global" in missing_region.assumptions[0]
        or "no protected-area adapter" in (missing_region.assumptions[0])
    )
    no_region = regional.fetch(_request())
    assert no_region.status is ObservationStatus.UNAVAILABLE
    assert "region_id is required" in no_region.assumptions[0]
    hit = regional.fetch(_request(parameters={"region_id": "DE"}))
    assert hit.status is ObservationStatus.OBSERVED
    np.testing.assert_allclose(float(hit.value), 0.5)  # type: ignore[arg-type]
    empty = RegionalProtectedAreaProvider({}, clock=FixedClock(T1))
    assert empty.health().status is HealthStatus.NOT_CONFIGURED


def test_geopackage_and_shapefile_when_geopandas_available(tmp_path: Path) -> None:
    geopandas = pytest.importorskip("geopandas")
    shapely_geometry = pytest.importorskip("shapely.geometry")
    frame = geopandas.GeoDataFrame(
        {
            "id": ["park-east"],
            "name": ["East Park"],
            "category": ["V"],
            "designation": ["protected landscape"],
        },
        geometry=[
            shapely_geometry.Polygon(
                [(5.0, 0.0), (15.0, 0.0), (15.0, 10.0), (5.0, 10.0), (5.0, 0.0)]
            )
        ],
        crs=CRS,
    )
    gpkg = tmp_path / "parks.gpkg"
    shp = tmp_path / "parks.shp"
    try:
        frame.to_file(gpkg, layer="parks", driver="GPKG")
        frame.to_file(shp)
    except Exception as exc:  # pragma: no cover - driver availability
        pytest.skip(f"could not write vector files: {exc}")
    for path in (gpkg, shp):
        provider = VectorProtectedAreaProvider(
            vector_path=path,
            clock=FixedClock(T1),
            environ={},
            crs=CRS,
            source_dataset=path.name,
        )
        observation = provider.fetch(_request())
        assert observation.status is ObservationStatus.OBSERVED
        np.testing.assert_allclose(float(observation.value), 0.5)  # type: ignore[arg-type]
        assert path.name in " ".join(observation.assumptions)
