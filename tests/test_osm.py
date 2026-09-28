"""Bounded OSM / Overpass provider: mocked HTTP, fixtures, no live queries."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote

import httpx
import numpy as np
import pytest

from solardss.data.geo import BoundingBox, GeographicRef
from solardss.data.osm import (
    MAX_BBOX_SPAN_DEG,
    OSM_SOURCE_NAME,
    MemoryOSMSource,
    OSMBuildingProvider,
    OSMInfrastructureProvider,
    OverpassClient,
    OverpassQuerySpec,
    assert_bounded_overpass_query,
    build_overpass_query,
    classify_osm_tags,
    parse_overpass_json,
    require_bounded_bbox,
)
from solardss.indicators.ids import IndicatorId
from solardss.indicators.infrastructure import (
    FeatureKind,
    GeometryKind,
    InfraFeature,
    nearest_feature,
)
from solardss.indicators.rooftop import local_square_lonlat
from solardss.provenance.observation import ObservationStatus, ValueOrigin
from solardss.providers.cache import MemoryCache
from solardss.providers.clock import FixedClock
from solardss.providers.errors import OSMQueryBoundError
from solardss.providers.protocols import BuildingProvider, InfrastructureProvider
from solardss.providers.request import FetchRequest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "osm"
T1 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
SITE = GeographicRef(name="OSM sample", longitude=13.4, latitude=52.52)
ORIGIN = (13.4, 52.52)


def _load_fixture() -> dict[str, object]:
    raw: object = json.loads((FIXTURE_DIR / "overpass_sample.json").read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _transport(
    *,
    payload: dict[str, object] | None = None,
    status_code: int = 200,
    calls: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if status_code != 200:
            return httpx.Response(status_code, text="overpass error")
        assert payload is not None
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def _memory_source() -> MemoryOSMSource:
    ring = local_square_lonlat(13.39995, 52.51995, 12.0, lat0=52.52)
    return MemoryOSMSource(
        features=(
            InfraFeature(
                kind=FeatureKind.POWER_LINE,
                geometry_kind=GeometryKind.LINESTRING,
                coordinates=((13.401, 52.519), (13.401, 52.521)),
                osm_type="way",
                osm_id=101,
                tags=(("power", "line"),),
            ),
            InfraFeature(
                kind=FeatureKind.SUBSTATION,
                geometry_kind=GeometryKind.POINT,
                coordinates=((13.4, 52.521),),
                osm_type="node",
                osm_id=201,
                name="Test substation",
                tags=(("power", "substation"), ("name", "Test substation")),
            ),
            InfraFeature(
                kind=FeatureKind.ROAD,
                geometry_kind=GeometryKind.LINESTRING,
                coordinates=((13.399, 52.52), (13.401, 52.52)),
                osm_type="way",
                osm_id=301,
                tags=(("highway", "residential"),),
            ),
            InfraFeature(
                kind=FeatureKind.SETTLEMENT,
                geometry_kind=GeometryKind.POINT,
                coordinates=((13.4, 52.524),),
                osm_type="node",
                osm_id=401,
                tags=(("place", "village"),),
            ),
            InfraFeature(
                kind=FeatureKind.INDUSTRIAL,
                geometry_kind=GeometryKind.POLYGON,
                coordinates=(
                    (13.398, 52.518),
                    (13.399, 52.518),
                    (13.399, 52.519),
                    (13.398, 52.519),
                    (13.398, 52.518),
                ),
                osm_type="way",
                osm_id=501,
                tags=(("landuse", "industrial"),),
            ),
            InfraFeature(
                kind=FeatureKind.BUILDING,
                geometry_kind=GeometryKind.POLYGON,
                coordinates=ring,
                osm_type="way",
                osm_id=601,
                tags=(
                    ("building", "yes"),
                    ("building:levels", "2"),
                    ("roof:angle", "30"),
                    ("roof:direction", "S"),
                    ("roof:shape", "gabled"),
                ),
            ),
        )
    )


def _infra(**kwargs: object) -> OSMInfrastructureProvider:
    params: dict[str, object] = {
        "source": _memory_source(),
        "clock": FixedClock(T1),
        "environ": {},
    }
    params.update(kwargs)
    return OSMInfrastructureProvider(**params)  # type: ignore[arg-type]


def _request(indicator: IndicatorId, **kwargs: object) -> FetchRequest:
    payload: dict[str, object] = {"indicator_id": indicator.value, "location": SITE}
    payload.update(kwargs)
    return FetchRequest.model_validate(payload)


def test_protocol_and_source_name() -> None:
    provider = _infra()
    assert isinstance(provider, InfrastructureProvider)
    observation = provider.fetch(_request(IndicatorId.INFRA_DISTANCE_TO_POWER_LINE))
    assert observation.provenance.source_name == OSM_SOURCE_NAME
    assert observation.status is ObservationStatus.OBSERVED
    assert any(item.startswith("source=OpenStreetMap") for item in observation.assumptions)
    assert observation.provenance.source_version == "2024-06-01T12:00:00Z"


def test_distances_match_geometry_oracle() -> None:
    provider = _infra()
    source = _memory_source()
    for indicator, kind in (
        (IndicatorId.INFRA_DISTANCE_TO_POWER_LINE, FeatureKind.POWER_LINE),
        (IndicatorId.INFRA_DISTANCE_TO_SUBSTATION, FeatureKind.SUBSTATION),
        (IndicatorId.INFRA_DISTANCE_TO_ROAD, FeatureKind.ROAD),
        (IndicatorId.INFRA_DISTANCE_TO_SETTLEMENT, FeatureKind.SETTLEMENT),
    ):
        observation = provider.fetch(_request(indicator))
        matching = tuple(item for item in source.features if item.kind is kind)
        oracle = nearest_feature(ORIGIN[0], ORIGIN[1], matching, crs="EPSG:4326", kind=kind)
        assert oracle is not None
        assert observation.value == pytest.approx(oracle.distance_m)
        assert observation.unit == "m"
        assert observation.provenance.value_origin is ValueOrigin.DERIVED


def test_hosting_capacity_is_refused() -> None:
    provider = _infra()
    observation = provider.fetch(_request(IndicatorId.INFRA_GRID_HOSTING_CAPACITY))
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.value is None
    assert "hosting capacity" in observation.assumptions[0]


def test_demand_center_requires_declaration() -> None:
    provider = _infra()
    missing = provider.fetch(_request(IndicatorId.INFRA_DISTANCE_TO_DEMAND_CENTER))
    assert missing.status is ObservationStatus.NOT_CONFIGURED
    assert missing.value is None
    assert "settlements are not treated as demand centers" in missing.assumptions[0]
    declared = provider.fetch(
        _request(
            IndicatorId.INFRA_DISTANCE_TO_DEMAND_CENTER,
            parameters={"demand_center_lon": 13.4, "demand_center_lat": 52.52},
        )
    )
    assert declared.status is ObservationStatus.OBSERVED
    np.testing.assert_allclose(float(declared.value), 0.0)  # type: ignore[arg-type]
    assert any("caller-declared" in item for item in declared.assumptions)


def test_empty_extract_is_missing_coverage() -> None:
    provider = OSMInfrastructureProvider(
        source=MemoryOSMSource(features=()),
        clock=FixedClock(T1),
        environ={},
    )
    observation = provider.fetch(_request(IndicatorId.INFRA_DISTANCE_TO_ROAD))
    assert observation.status is ObservationStatus.MISSING_COVERAGE
    assert observation.value is None


def test_building_footprint_and_roof_tags() -> None:
    provider = OSMBuildingProvider(source=_memory_source(), clock=FixedClock(T1), environ={})
    assert isinstance(provider, BuildingProvider)
    area = provider.fetch(_request(IndicatorId.ROOF_BUILDING_FOOTPRINT_AREA))
    orientation = provider.fetch(_request(IndicatorId.ROOF_ORIENTATION))
    slope = provider.fetch(_request(IndicatorId.ROOF_SLOPE))
    assert area.status is ObservationStatus.OBSERVED
    assert float(area.value) > 0.0  # type: ignore[arg-type]
    np.testing.assert_allclose(float(orientation.value), 180.0)  # type: ignore[arg-type]
    np.testing.assert_allclose(float(slope.value), 30.0)  # type: ignore[arg-type]
    assert any("roof:direction=S" in item for item in orientation.assumptions)
    usable = provider.fetch(_request(IndicatorId.ROOF_USABLE_AREA))
    structural = provider.fetch(_request(IndicatorId.ROOF_STRUCTURAL_CAPACITY))
    assert usable.status is ObservationStatus.UNAVAILABLE
    assert structural.status is ObservationStatus.UNAVAILABLE
    assert "usable roof area" in usable.assumptions[0]
    assert "structural" in structural.assumptions[0]


def test_roof_orientation_unavailable_without_tags() -> None:
    ring = local_square_lonlat(13.39995, 52.51995, 12.0, lat0=52.52)
    source = MemoryOSMSource(
        features=(
            InfraFeature(
                kind=FeatureKind.BUILDING,
                geometry_kind=GeometryKind.POLYGON,
                coordinates=ring,
                osm_type="way",
                osm_id=1,
                tags=(("building", "yes"),),
            ),
        )
    )
    provider = OSMBuildingProvider(source=source, clock=FixedClock(T1), environ={})
    orientation = provider.fetch(_request(IndicatorId.ROOF_ORIENTATION))
    slope = provider.fetch(_request(IndicatorId.ROOF_SLOPE))
    area = provider.fetch(_request(IndicatorId.ROOF_BUILDING_FOOTPRINT_AREA))
    assert orientation.status is ObservationStatus.UNAVAILABLE
    assert slope.status is ObservationStatus.UNAVAILABLE
    assert area.status is ObservationStatus.OBSERVED


def test_overpass_query_is_always_bbox_bounded() -> None:
    bbox = BoundingBox(min_lon=13.39, min_lat=52.51, max_lon=13.41, max_lat=52.53)
    query = build_overpass_query(OverpassQuerySpec(bbox=bbox, kinds=(FeatureKind.POWER_LINE,)))
    assert "[bbox:" in query
    assert "out geom" in query
    assert 'way["power"="line"]' in query
    assert_bounded_overpass_query(query, bbox)
    with pytest.raises(OSMQueryBoundError, match="global"):
        require_bounded_bbox(
            BoundingBox(min_lon=-180.0, min_lat=-90.0, max_lon=180.0, max_lat=90.0)
        )
    with pytest.raises(OSMQueryBoundError, match="span"):
        require_bounded_bbox(
            BoundingBox(
                min_lon=0.0,
                min_lat=0.0,
                max_lon=MAX_BBOX_SPAN_DEG + 0.1,
                max_lat=0.2,
            )
        )
    with pytest.raises(OSMQueryBoundError, match="unfiltered"):
        assert_bounded_overpass_query("[bbox:52.0,13.0,52.1,13.1];\nnode;\nout;")


def test_classify_and_parse_fixture_geometry() -> None:
    assert classify_osm_tags({"power": "line"}) == (FeatureKind.POWER_LINE,)
    assert classify_osm_tags({"highway": "footway"}) == ()
    payload = _load_fixture()
    bbox = BoundingBox(min_lon=13.39, min_lat=52.51, max_lon=13.41, max_lat=52.53)
    snapshot = parse_overpass_json(
        payload,
        query=build_overpass_query(OverpassQuerySpec(bbox=bbox, kinds=(FeatureKind.POWER_LINE,))),
        bbox=bbox,
    )
    kinds = {item.kind for item in snapshot.features}
    assert FeatureKind.POWER_LINE in kinds
    assert FeatureKind.BUILDING in kinds
    assert FeatureKind.INDUSTRIAL in kinds
    building = next(item for item in snapshot.features if item.kind is FeatureKind.BUILDING)
    assert building.coordinates[0][0] == pytest.approx(13.39995)
    assert snapshot.timestamp_osm_base == "2024-06-01T12:00:00Z"


def test_overpass_client_posts_mocked_json() -> None:
    calls: list[httpx.Request] = []
    payload = _load_fixture()
    bbox = BoundingBox(min_lon=13.39, min_lat=52.51, max_lon=13.41, max_lat=52.53)
    client = OverpassClient(
        base_url="https://overpass.test/api/interpreter",
        transport=_transport(payload=payload, calls=calls),
        clock=FixedClock(T1),
        sleep=lambda _seconds: None,
        cache=MemoryCache(),
    )
    try:
        snapshot = client.query(
            OverpassQuerySpec(bbox=bbox, kinds=(FeatureKind.POWER_LINE, FeatureKind.BUILDING))
        )
        assert snapshot.timestamp_osm_base == "2024-06-01T12:00:00Z"
        assert calls and calls[0].method == "POST"
        body = unquote(calls[0].content.decode("utf-8")).replace("+", " ")
        assert "[bbox:" in body
        assert "out geom;" in body
        again = client.query(
            OverpassQuerySpec(bbox=bbox, kinds=(FeatureKind.POWER_LINE, FeatureKind.BUILDING))
        )
        assert again.timestamp_osm_base == snapshot.timestamp_osm_base
        assert len(calls) == 1
    finally:
        client.close()


def test_overpass_http_error_is_temporary_failure() -> None:
    provider = OSMInfrastructureProvider(
        transport=_transport(status_code=500),
        clock=FixedClock(T1),
        sleep=lambda _seconds: None,
        environ={},
        base_url="https://overpass.test/api/interpreter",
    )
    try:
        observation = provider.fetch(_request(IndicatorId.INFRA_DISTANCE_TO_POWER_LINE))
        assert observation.status is ObservationStatus.TEMPORARY_FAILURE
        assert observation.value is None
        assert any("Overpass HTTP 500" in item for item in observation.assumptions)
    finally:
        provider.close()
