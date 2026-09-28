"""Fake providers, provenance round-trips, and the local dataset cache."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from solardss.data.geo import BoundingBox, GeographicRef
from solardss.exceptions import InvalidParameterError
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.indicators.ids import IndicatorId
from solardss.provenance import (
    ObservationStatus,
    QualityFlag,
    SpatialCoverage,
    TemporalCoverage,
    ValueOrigin,
)
from solardss.providers import (
    CACHE_DIR_ENV,
    BuildingProvider,
    DirectoryCache,
    FakeBuildingProvider,
    FakeGridCapacityProvider,
    FakeInfrastructureProvider,
    FakeSeed,
    FakeSolarResourceProvider,
    FakeTerrainProvider,
    FetchRequest,
    FixedClock,
    GridCapacityProvider,
    MemoryCache,
    SequenceClock,
    SolarResourceProvider,
    cache_root_from_config,
    request_cache_key,
)
from solardss.providers.contracts import AvailabilityStatus, HealthStatus

T1 = datetime(2021, 6, 1, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2021, 6, 2, 12, 0, tzinfo=timezone.utc)

EUROPE = BoundingBox(min_lon=-10.0, min_lat=35.0, max_lon=30.0, max_lat=72.0)
GERMANY = BoundingBox(min_lon=5.0, min_lat=47.0, max_lon=16.0, max_lat=55.0)
HAMBURG_HOLE = BoundingBox(min_lon=9.7, min_lat=53.4, max_lon=10.3, max_lat=53.7)

GHI = IndicatorId.SOLAR_GHI_ANNUAL.value


def _berlin() -> GeographicRef:
    return GeographicRef(name="Berlin", place_id="berlin", longitude=13.4, latitude=52.52)


def _hamburg() -> GeographicRef:
    return GeographicRef(name="Hamburg", place_id="hamburg", longitude=10.0, latitude=53.55)


def _kansas() -> GeographicRef:
    return GeographicRef(name="Kansas", place_id="kansas", longitude=-98.0, latitude=38.5)


def _ghi_request(location: GeographicRef, **kwargs: object) -> FetchRequest:
    payload: dict[str, object] = {"indicator_id": GHI, "location": location}
    payload.update(kwargs)
    return FetchRequest.model_validate(payload)


def _ghi_seed() -> FakeSeed:
    return FakeSeed(
        value=1085.0,
        unit="kWh/m2/year",
        quality_flags=(QualityFlag.NOMINAL.value,),
        assumptions=("seeded fake catalog; not a satellite product",),
    )


def _solar(**kwargs: object) -> FakeSolarResourceProvider:
    payload: dict[str, object] = {
        "seeds": {GHI: _ghi_seed()},
        "operating_bbox": EUROPE,
        "coverage_bbox": GERMANY,
        "coverage_holes": (HAMBURG_HOLE,),
        "spatial": SpatialCoverage(resolution="0.25deg", crs="EPSG:4326"),
        "temporal": TemporalCoverage(resolution="annual", aggregation="mean"),
        "clock": FixedClock(T1),
        "source_version": "fake-1",
    }
    payload.update(kwargs)
    return FakeSolarResourceProvider(**payload)  # type: ignore[arg-type]


def test_fetch_request_requires_place() -> None:
    with pytest.raises((InvalidParameterError, ValidationError)):
        FetchRequest(indicator_id=GHI)


def test_request_hash_is_stable() -> None:
    left = _ghi_request(_berlin(), parameters={"ensemble": "mean"})
    right = _ghi_request(_berlin(), parameters={"ensemble": "mean"})
    assert request_cache_key("fake.solar_resource", left) == request_cache_key(
        "fake.solar_resource", right
    )
    assert len(request_cache_key("fake.solar_resource", left)) == 64


def test_same_request_same_observation_same_provenance() -> None:
    cache = MemoryCache()
    provider = _solar(cache=cache, clock=SequenceClock((T1, T2)))
    request = _ghi_request(_berlin())
    first = provider.fetch(request)
    second = provider.fetch(request)
    assert first.status is ObservationStatus.OBSERVED
    assert first.value == 1085.0
    assert first == second
    assert first.provenance.retrieved_at == T1
    assert first.provenance.request_hash == second.provenance.request_hash
    assert first.provenance.source_name == "Fake solar resource catalog"
    assert first.provenance.source_version == "fake-1"
    assert first.spatial.resolution == "0.25deg"
    assert first.temporal.resolution == "annual"
    assert first.quality_flags == (QualityFlag.NOMINAL.value,)
    assert first.assumptions == ("seeded fake catalog; not a satellite product",)
    assert first.provenance.value_origin is ValueOrigin.RAW
    assert first.provenance.calculation_method is None


def test_directory_cache_preserves_provenance_across_instances(tmp_path: Path) -> None:
    cache = DirectoryCache(tmp_path / "provider-cache")
    request = _ghi_request(_berlin())
    first = _solar(cache=cache, clock=SequenceClock((T1, T2))).fetch(request)
    second = _solar(cache=cache, clock=SequenceClock((T2,))).fetch(request)
    assert first == second
    assert first.provenance.retrieved_at == T1
    written = list((tmp_path / "provider-cache").rglob("*.bin"))
    assert written
    assert all(path.is_relative_to(tmp_path) for path in written)


def test_cache_root_is_configurable(tmp_path: Path) -> None:
    assert cache_root_from_config(environ={}) is None
    assert cache_root_from_config(environ={CACHE_DIR_ENV: ""}) is None
    configured = cache_root_from_config(environ={CACHE_DIR_ENV: str(tmp_path / "from-env")})
    assert configured == tmp_path / "from-env"
    explicit = cache_root_from_config(
        explicit=tmp_path / "explicit",
        environ={CACHE_DIR_ENV: "ignored"},
    )
    assert explicit == tmp_path / "explicit"
    assert DirectoryCache(explicit).root == tmp_path / "explicit"
    assert cache_root_from_config(environ={"HOME": str(tmp_path)}) is None


def test_unsupported_region_has_no_value() -> None:
    observation = _solar().fetch(_ghi_request(_kansas()))
    assert observation.status is ObservationStatus.UNSUPPORTED_REGION
    assert observation.value is None
    availability = _solar().availability(_ghi_request(_kansas()))
    assert availability.status is AvailabilityStatus.UNSUPPORTED_REGION
    assert availability.retryable is False


def test_missing_coverage_has_no_value() -> None:
    observation = _solar().fetch(_ghi_request(_hamburg()))
    assert observation.status is ObservationStatus.MISSING_COVERAGE
    assert observation.value is None


def test_unavailable_indicator_has_no_value() -> None:
    observation = _solar().fetch(
        FetchRequest(indicator_id=IndicatorId.CLIMATE_WIND_SPEED_MEAN.value, location=_berlin())
    )
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.value is None


def test_unseeded_but_covered_indicator_is_unavailable() -> None:
    observation = _solar().fetch(
        FetchRequest(indicator_id=IndicatorId.SOLAR_DNI_ANNUAL.value, location=_berlin())
    )
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.value is None


def test_temporary_failure_is_retryable_and_not_cached() -> None:
    cache = MemoryCache()
    request = _ghi_request(_berlin())
    failing = _solar(cache=cache, clock=SequenceClock((T1, T2)), temporary_failure=True)
    missed = failing.fetch(request)
    assert missed.status is ObservationStatus.TEMPORARY_FAILURE
    assert missed.value is None
    assert failing.availability(request).retryable is True
    recovered = _solar(cache=cache, clock=SequenceClock((T2,))).fetch(request)
    assert recovered.status is ObservationStatus.OBSERVED
    assert recovered.provenance.retrieved_at == T2
    assert recovered.value == 1085.0


def test_not_configured_provider() -> None:
    provider = _solar(configured=False, clock=FixedClock(T1))
    assert provider.health().status is HealthStatus.NOT_CONFIGURED
    observation = provider.fetch(_ghi_request(_berlin()))
    assert observation.status is ObservationStatus.NOT_CONFIGURED
    assert observation.value is None


def test_fake_solar_provider_matches_protocol() -> None:
    provider = _solar()
    assert isinstance(provider, SolarResourceProvider)
    health = provider.health()
    assert health.status is HealthStatus.OK
    coverage = provider.coverage(_ghi_request(_berlin()))
    assert coverage.status is AvailabilityStatus.AVAILABLE
    assert coverage.spatial is not None
    assert coverage.temporal is not None


def test_infrastructure_does_not_invent_hosting_capacity() -> None:
    provider = FakeInfrastructureProvider(
        seeds={
            IndicatorId.INFRA_DISTANCE_TO_POWER_LINE.value: FakeSeed(value=250.0, unit="m"),
        },
        clock=FixedClock(T1),
    )
    observation = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.INFRA_GRID_HOSTING_CAPACITY.value,
            location=_berlin(),
        )
    )
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.value is None
    assert "hosting capacity" in observation.assumptions[0]


def test_grid_capacity_empty_catalog_stays_empty() -> None:
    provider = FakeGridCapacityProvider(clock=FixedClock(T1))
    assert isinstance(provider, GridCapacityProvider)
    observation = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.INFRA_GRID_HOSTING_CAPACITY.value,
            location=_berlin(),
        )
    )
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.value is None


def test_building_provider_does_not_infer_structural_capacity() -> None:
    provider = FakeBuildingProvider(
        seeds={
            IndicatorId.ROOF_BUILDING_FOOTPRINT_AREA.value: FakeSeed(value=120.0, unit="m2"),
        },
        clock=FixedClock(T1),
    )
    assert isinstance(provider, BuildingProvider)
    observation = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.ROOF_STRUCTURAL_CAPACITY.value,
            location=_berlin(),
        )
    )
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.value is None
    assert "structural" in observation.assumptions[0]


def test_cannot_seed_refused_indicator() -> None:
    with pytest.raises(InvalidParameterError):
        FakeBuildingProvider(
            seeds={
                IndicatorId.ROOF_STRUCTURAL_CAPACITY.value: FakeSeed(value=1.0, unit="kN/m2"),
            }
        )


def test_derived_seed_records_calculation_method() -> None:
    provider = FakeTerrainProvider(
        seeds={
            IndicatorId.TERRAIN_SLOPE_MEAN.value: FakeSeed(
                value=6.2,
                unit="deg",
                value_origin=ValueOrigin.DERIVED,
                value_kind=ValueProvenanceKind.DERIVED_METHOD,
                calculation_method="horn_1981",
                assumptions=("regular 30 m DEM",),
            )
        },
        clock=FixedClock(T1),
    )
    observation = provider.fetch(
        FetchRequest(indicator_id=IndicatorId.TERRAIN_SLOPE_MEAN.value, location=_berlin())
    )
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.value == 6.2
    assert observation.provenance.value_origin is ValueOrigin.DERIVED
    assert observation.provenance.calculation_method == "horn_1981"


def test_named_place_without_geometry_needs_explicit_coverage() -> None:
    named = GeographicRef(name="Berlin", place_id="berlin")
    uncovered = _solar(covered_place_ids=frozenset()).fetch(_ghi_request(named))
    assert uncovered.status is ObservationStatus.UNSUPPORTED_REGION
    assert uncovered.value is None
    covered = _solar(covered_place_ids=frozenset({"berlin"})).fetch(_ghi_request(named))
    assert covered.status is ObservationStatus.OBSERVED
    assert covered.value == 1085.0


def test_provider_cache_modules_have_no_home_default() -> None:
    from solardss.providers import cache as cache_mod
    from solardss.providers import fake as fake_mod

    for module in (cache_mod, fake_mod):
        path = module.__file__
        assert path is not None
        text = Path(path).read_text(encoding="utf-8")
        assert "Path.home" not in text
        assert "expanduser" not in text
        assert "/home/" not in text
