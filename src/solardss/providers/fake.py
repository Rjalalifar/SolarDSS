"""In-memory fake providers. They never open a network or a real dataset."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

from solardss.data.contracts import DatasetRef
from solardss.data.geo import BoundingBox
from solardss.exceptions import InvalidParameterError
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.indicators.ids import IndicatorId
from solardss.provenance.coverage import SpatialCoverage, TemporalCoverage
from solardss.provenance.observation import (
    DataProvenance,
    IndicatorObservation,
    ObservationStatus,
    QualityFlag,
    ValueOrigin,
)
from solardss.providers.cache import DatasetCache
from solardss.providers.clock import Clock, UtcClock
from solardss.providers.contracts import (
    AvailabilityStatus,
    CoverageReport,
    HealthStatus,
    ProviderAvailability,
    ProviderHealth,
)
from solardss.providers.errors import CacheError
from solardss.providers.region import (
    geometry_matches_bbox,
    request_has_geometry,
    request_place_id,
)
from solardss.providers.request import FetchRequest, request_cache_key

_CACHE_SCHEMA: Final = "solardss.provider_cache.v1"

_FAKE_SPATIAL = SpatialCoverage(resolution="fake", crs="EPSG:4326")
_FAKE_TEMPORAL = TemporalCoverage(resolution="fake")

_SOLAR_SERVED = frozenset(
    {
        IndicatorId.SOLAR_GHI_ANNUAL.value,
        IndicatorId.SOLAR_DNI_ANNUAL.value,
        IndicatorId.SOLAR_DHI_ANNUAL.value,
        IndicatorId.SOLAR_MONTHLY_VARIABILITY.value,
        IndicatorId.SOLAR_INTERANNUAL_VARIABILITY.value,
        IndicatorId.SOLAR_SUNSHINE_DURATION.value,
    }
)
_WEATHER_SERVED = frozenset(
    {
        IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN.value,
        IndicatorId.CLIMATE_AIR_TEMPERATURE_MIN.value,
        IndicatorId.CLIMATE_AIR_TEMPERATURE_MAX.value,
        IndicatorId.CLIMATE_EXTREME_HEAT_DAYS.value,
        IndicatorId.CLIMATE_WIND_SPEED_MEAN.value,
        IndicatorId.CLIMATE_RELATIVE_HUMIDITY.value,
        IndicatorId.CLIMATE_CLOUD_COVER.value,
        IndicatorId.CLIMATE_PRECIPITATION.value,
        IndicatorId.CLIMATE_SNOW_RISK.value,
        IndicatorId.CLIMATE_SOILING_RISK.value,
    }
)
_TERRAIN_SERVED = frozenset(
    {
        IndicatorId.TERRAIN_ELEVATION.value,
        IndicatorId.TERRAIN_SLOPE_MEAN.value,
        IndicatorId.TERRAIN_SLOPE_MAX.value,
        IndicatorId.TERRAIN_ASPECT.value,
        IndicatorId.TERRAIN_RUGGEDNESS.value,
        IndicatorId.TERRAIN_HORIZON_OBSTRUCTION.value,
        IndicatorId.TERRAIN_SHADING_POTENTIAL.value,
    }
)
_LAND_COVER_SERVED = frozenset(
    {
        IndicatorId.LAND_COVER.value,
        IndicatorId.LAND_AVAILABLE_AREA.value,
        IndicatorId.LAND_FOREST_FRACTION.value,
        IndicatorId.LAND_WATER_FRACTION.value,
        IndicatorId.LAND_WETLAND_FRACTION.value,
        IndicatorId.LAND_AGRICULTURAL_FRACTION.value,
        IndicatorId.LAND_BUILTUP_FRACTION.value,
        IndicatorId.LAND_BARE_LAND_FRACTION.value,
    }
)
_INFRASTRUCTURE_SERVED = frozenset(
    {
        IndicatorId.INFRA_DISTANCE_TO_POWER_LINE.value,
        IndicatorId.INFRA_DISTANCE_TO_SUBSTATION.value,
        IndicatorId.INFRA_DISTANCE_TO_ROAD.value,
        IndicatorId.INFRA_DISTANCE_TO_SETTLEMENT.value,
        IndicatorId.INFRA_DISTANCE_TO_DEMAND_CENTER.value,
    }
)
_PROTECTED_SERVED = frozenset({IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value})
_HAZARD_SERVED = frozenset(
    {
        IndicatorId.HAZARD_FLOOD_RISK.value,
        IndicatorId.HAZARD_LANDSLIDE_RISK.value,
        IndicatorId.HAZARD_WILDFIRE_RISK.value,
        IndicatorId.HAZARD_HAIL_RISK.value,
    }
)
_BUILDING_SERVED = frozenset(
    {
        IndicatorId.ROOF_BUILDING_FOOTPRINT_AREA.value,
        IndicatorId.ROOF_USABLE_AREA.value,
        IndicatorId.ROOF_ORIENTATION.value,
        IndicatorId.ROOF_SLOPE.value,
        IndicatorId.ROOF_SHADING.value,
    }
)
_GRID_SERVED = frozenset({IndicatorId.INFRA_GRID_HOSTING_CAPACITY.value})

_INFRA_REFUSED = (
    (
        IndicatorId.INFRA_GRID_HOSTING_CAPACITY.value,
        "mapped infrastructure is not grid hosting capacity",
    ),
)
_BUILDING_REFUSED = (
    (
        IndicatorId.ROOF_STRUCTURAL_CAPACITY.value,
        "building footprints are not structural capacity",
    ),
)


class FakeSeed(BaseModel):
    """One catalogued value. Absence of a seed means the fake has no value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float | str | bool
    unit: str = Field(min_length=1)
    quality_flags: tuple[str, ...] = (QualityFlag.NOMINAL.value,)
    assumptions: tuple[str, ...] = ()
    value_kind: ValueProvenanceKind = ValueProvenanceKind.MEASURED_OR_PROVIDER
    value_origin: ValueOrigin = ValueOrigin.RAW
    calculation_method: str | None = None
    dataset: DatasetRef | None = None


def _as_seed(value: FakeSeed | Mapping[str, object]) -> FakeSeed:
    if isinstance(value, FakeSeed):
        return value
    return FakeSeed.model_validate(value)


def _temporal_overlap(dataset: TemporalCoverage, requested: TemporalCoverage | None) -> bool:
    if requested is None:
        return True
    if dataset.start is not None and requested.end is not None and requested.end < dataset.start:
        return False
    return not (
        dataset.end is not None and requested.start is not None and requested.start > dataset.end
    )


def _encode_observation(observation: IndicatorObservation) -> bytes:
    envelope = {
        "schema_id": _CACHE_SCHEMA,
        "observation": observation.model_dump(mode="json"),
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _decode_observation(payload: bytes) -> IndicatorObservation:
    try:
        raw: object = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheError("cache entry is not valid JSON") from exc
    if not isinstance(raw, dict) or raw.get("schema_id") != _CACHE_SCHEMA:
        raise CacheError("cache entry has unknown schema")
    body = raw.get("observation")
    if not isinstance(body, dict):
        raise CacheError("cache entry is missing an observation")
    try:
        return IndicatorObservation.model_validate(body)
    except (InvalidParameterError, ValueError, TypeError) as exc:
        raise CacheError("cache entry is not an IndicatorObservation") from exc


class FakeDataProvider:
    """Deterministic catalog. Values exist only when seeded.

    Temporary failures are never cached. Successful observations and structured
    misses (unavailable, unsupported region, missing coverage, not configured)
    are cached when a cache is attached so the same request round-trips the
    same provenance.
    """

    provider_id_default: str = "fake.provider"
    source_name_default: str = "Fake catalog"
    served_default: frozenset[str] = frozenset()
    refused_default: tuple[tuple[str, str], ...] = ()

    def __init__(
        self,
        *,
        provider_id: str | None = None,
        source_name: str | None = None,
        spatial: SpatialCoverage | None = None,
        temporal: TemporalCoverage | None = None,
        seeds: Mapping[str, FakeSeed | Mapping[str, object]] | None = None,
        place_seeds: Mapping[str, Mapping[str, FakeSeed | Mapping[str, object]]] | None = None,
        served_indicators: frozenset[str] | None = None,
        refused_indicators: Mapping[str, str] | None = None,
        operating_bbox: BoundingBox | None = None,
        coverage_bbox: BoundingBox | None = None,
        coverage_holes: Sequence[BoundingBox] = (),
        covered_place_ids: frozenset[str] | None = None,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        source_version: str | None = "fake-1",
        dataset: DatasetRef | None = None,
        configured: bool = True,
        temporary_failure: bool = False,
        temporary_failure_reason: str = "temporary network failure",
    ) -> None:
        self._provider_id = provider_id or self.provider_id_default
        self._source_name = source_name or self.source_name_default
        self._spatial = spatial if spatial is not None else _FAKE_SPATIAL
        self._temporal = temporal if temporal is not None else _FAKE_TEMPORAL
        self._seeds = {key: _as_seed(value) for key, value in (seeds or {}).items()}
        self._place_seeds = {
            place: {indicator: _as_seed(seed) for indicator, seed in mapping.items()}
            for place, mapping in (place_seeds or {}).items()
        }
        self._served = served_indicators if served_indicators is not None else self.served_default
        refused = dict(self.refused_default)
        if refused_indicators is not None:
            refused.update(refused_indicators)
        self._refused = refused
        self._operating_bbox = operating_bbox
        self._coverage_bbox = coverage_bbox
        self._coverage_holes = tuple(coverage_holes)
        self._covered_place_ids = covered_place_ids or frozenset()
        self._cache = cache
        self._clock: Clock = UtcClock() if clock is None else clock
        self._source_version = source_version
        self._dataset = dataset
        self._configured = configured
        self._temporary_failure = temporary_failure
        self._temporary_failure_reason = temporary_failure_reason
        self._reject_invalid_seeds()

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def served_indicators(self) -> frozenset[str]:
        """Indicator ids this fake will consider. Empty means none."""

        return self._served

    def _reject_invalid_seeds(self) -> None:
        seeded_ids = set(self._seeds)
        for mapping in self._place_seeds.values():
            seeded_ids.update(mapping)
        for indicator_id in sorted(seeded_ids):
            if indicator_id in self._refused:
                raise InvalidParameterError(
                    f"cannot seed refused indicator {indicator_id} on {self._provider_id}"
                )
            if self._served and indicator_id not in self._served:
                raise InvalidParameterError(
                    f"seed {indicator_id} is not served by {self._provider_id}"
                )

    def health(self) -> ProviderHealth:
        checked_at = self._clock.now()
        if not self._configured:
            return ProviderHealth(
                status=HealthStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                reason="provider is not configured",
                checked_at=checked_at,
            )
        if self._temporary_failure:
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                reason=self._temporary_failure_reason,
                checked_at=checked_at,
            )
        return ProviderHealth(
            status=HealthStatus.OK,
            provider_id=self._provider_id,
            reason="fake provider is ready",
            checked_at=checked_at,
            details={"cache": self._cache.__class__.__name__ if self._cache else "none"},
        )

    def availability(self, request: FetchRequest) -> ProviderAvailability:
        if not self._configured:
            return ProviderAvailability(
                status=AvailabilityStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="provider is not configured",
            )
        if self._temporary_failure:
            return ProviderAvailability(
                status=AvailabilityStatus.TEMPORARY_FAILURE,
                provider_id=self._provider_id,
                retryable=True,
                indicator_id=request.indicator_id,
                reason=self._temporary_failure_reason,
            )
        refused = self._refused.get(request.indicator_id)
        if refused is not None:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=refused,
            )
        if request.indicator_id not in self._served:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="indicator is not served by this provider",
            )
        region = self._region_status(request)
        if region is not None:
            return region
        if not _temporal_overlap(self._temporal, request.temporal):
            return ProviderAvailability(
                status=AvailabilityStatus.MISSING_COVERAGE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="requested period is outside the dataset temporal coverage",
            )
        return ProviderAvailability(
            status=AvailabilityStatus.AVAILABLE,
            provider_id=self._provider_id,
            retryable=False,
            indicator_id=request.indicator_id,
            reason="provider can serve this request",
        )

    def coverage(self, request: FetchRequest) -> CoverageReport:
        availability = self.availability(request)
        return CoverageReport(
            status=availability.status,
            provider_id=self._provider_id,
            spatial=self._spatial,
            temporal=self._temporal,
            reason=availability.reason,
        )

    def fetch(self, request: FetchRequest) -> IndicatorObservation:
        request_hash = request_cache_key(self._provider_id, request)
        availability = self.availability(request)
        if availability.status is AvailabilityStatus.TEMPORARY_FAILURE:
            return self._observation_from_status(
                request,
                request_hash,
                status=ObservationStatus.TEMPORARY_FAILURE,
                reason=availability.reason,
                retrieved_at=self._clock.now(),
            )
        cached = self._read_cache(request_hash)
        if cached is not None:
            return cached
        if availability.status is not AvailabilityStatus.AVAILABLE:
            observation = self._observation_from_status(
                request,
                request_hash,
                status=ObservationStatus(availability.status.value),
                reason=availability.reason,
                retrieved_at=self._clock.now(),
            )
            self._write_cache(request_hash, observation)
            return observation
        seed = self._lookup_seed(request)
        if seed is None:
            observation = self._observation_from_status(
                request,
                request_hash,
                status=ObservationStatus.UNAVAILABLE,
                reason="no catalogued value for this indicator at the requested place",
                retrieved_at=self._clock.now(),
            )
            self._write_cache(request_hash, observation)
            return observation
        observation = self._observed(request, request_hash, seed, retrieved_at=self._clock.now())
        self._write_cache(request_hash, observation)
        return observation

    def _region_status(self, request: FetchRequest) -> ProviderAvailability | None:
        has_geometry = request_has_geometry(request)
        named = self._named_place_known(request)
        if self._operating_bbox is not None:
            match = geometry_matches_bbox(request, self._operating_bbox)
            if match is False:
                return self._place_miss(
                    request,
                    AvailabilityStatus.UNSUPPORTED_REGION,
                    "requested place is outside the provider operating region",
                )
            if not has_geometry and not named:
                return self._place_miss(
                    request,
                    AvailabilityStatus.UNSUPPORTED_REGION,
                    "requested place has no geometry and is not in the operating place list",
                )
        if self._coverage_bbox is not None:
            match = geometry_matches_bbox(request, self._coverage_bbox)
            if match is False:
                return self._place_miss(
                    request,
                    AvailabilityStatus.MISSING_COVERAGE,
                    "requested place is not in the dataset spatial coverage",
                )
            if not has_geometry and not named:
                return self._place_miss(
                    request,
                    AvailabilityStatus.MISSING_COVERAGE,
                    "requested place has no geometry and is not in the coverage place list",
                )
        for hole in self._coverage_holes:
            if geometry_matches_bbox(request, hole) is True:
                return self._place_miss(
                    request,
                    AvailabilityStatus.MISSING_COVERAGE,
                    "requested place falls in a coverage hole",
                )
        if (
            not has_geometry
            and request_place_id(request) is not None
            and self._covered_place_ids
            and not named
        ):
            return self._place_miss(
                request,
                AvailabilityStatus.MISSING_COVERAGE,
                "requested place is not in the coverage place list",
            )
        return None

    def _named_place_known(self, request: FetchRequest) -> bool:
        place_id = request_place_id(request)
        if place_id is None:
            return False
        return place_id in self._covered_place_ids or place_id in self._place_seeds

    def _place_miss(
        self,
        request: FetchRequest,
        status: AvailabilityStatus,
        reason: str,
    ) -> ProviderAvailability:
        return ProviderAvailability(
            status=status,
            provider_id=self._provider_id,
            retryable=False,
            indicator_id=request.indicator_id,
            reason=reason,
        )

    def _lookup_seed(self, request: FetchRequest) -> FakeSeed | None:
        place_id = request_place_id(request)
        if place_id is not None:
            by_place = self._place_seeds.get(place_id)
            if by_place is not None and request.indicator_id in by_place:
                return by_place[request.indicator_id]
        return self._seeds.get(request.indicator_id)

    def _read_cache(self, request_hash: str) -> IndicatorObservation | None:
        if self._cache is None:
            return None
        payload = self._cache.get(request_hash)
        if payload is None:
            return None
        observation = _decode_observation(payload)
        if observation.provenance.request_hash != request_hash:
            raise CacheError("cached observation request_hash does not match the request")
        if observation.provenance.provider_id != self._provider_id:
            raise CacheError("cached observation belongs to a different provider")
        return observation

    def _write_cache(self, request_hash: str, observation: IndicatorObservation) -> None:
        if self._cache is None:
            return
        self._cache.put(request_hash, _encode_observation(observation))

    def _observation_from_status(
        self,
        request: FetchRequest,
        request_hash: str,
        *,
        status: ObservationStatus,
        reason: str,
        retrieved_at: datetime,
    ) -> IndicatorObservation:
        provenance = DataProvenance(
            provider_id=self._provider_id,
            source_name=self._source_name,
            retrieved_at=retrieved_at,
            request_hash=request_hash,
            value_kind=ValueProvenanceKind.MEASURED_OR_PROVIDER,
            value_origin=ValueOrigin.RAW,
            source_version=self._source_version,
            dataset=self._dataset,
        )
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=status,
            provenance=provenance,
            spatial=self._spatial,
            temporal=self._temporal,
            quality_flags=(),
            assumptions=(reason,),
        )

    def _observed(
        self,
        request: FetchRequest,
        request_hash: str,
        seed: FakeSeed,
        *,
        retrieved_at: datetime,
    ) -> IndicatorObservation:
        provenance = DataProvenance(
            provider_id=self._provider_id,
            source_name=self._source_name,
            retrieved_at=retrieved_at,
            request_hash=request_hash,
            value_kind=seed.value_kind,
            value_origin=seed.value_origin,
            source_version=self._source_version,
            calculation_method=seed.calculation_method,
            dataset=seed.dataset if seed.dataset is not None else self._dataset,
        )
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=ObservationStatus.OBSERVED,
            provenance=provenance,
            spatial=self._spatial,
            temporal=self._temporal,
            value=seed.value,
            unit=seed.unit,
            quality_flags=seed.quality_flags,
            assumptions=seed.assumptions,
        )


class FakeSolarResourceProvider(FakeDataProvider):
    """Seeded GHI/DNI/DHI catalog. Does not estimate irradiation."""

    provider_id_default = "fake.solar_resource"
    source_name_default = "Fake solar resource catalog"
    served_default = _SOLAR_SERVED


class FakeWeatherProvider(FakeDataProvider):
    """Seeded climate catalog. Does not run a weather model."""

    provider_id_default = "fake.weather"
    source_name_default = "Fake weather catalog"
    served_default = _WEATHER_SERVED


class FakeTerrainProvider(FakeDataProvider):
    """Seeded DEM-derived values. Does not invent a DEM."""

    provider_id_default = "fake.terrain"
    source_name_default = "Fake terrain catalog"
    served_default = _TERRAIN_SERVED


class FakeLandCoverProvider(FakeDataProvider):
    """Seeded land-cover fractions. Does not classify imagery."""

    provider_id_default = "fake.land_cover"
    source_name_default = "Fake land-cover catalog"
    served_default = _LAND_COVER_SERVED


class FakeInfrastructureProvider(FakeDataProvider):
    """Seeded distances. Refuses to serve hosting capacity."""

    provider_id_default = "fake.infrastructure"
    source_name_default = "Fake infrastructure catalog"
    served_default = _INFRASTRUCTURE_SERVED
    refused_default = _INFRA_REFUSED


class FakeProtectedAreaProvider(FakeDataProvider):
    """Seeded protected-area overlap. Does not read a cadastre file."""

    provider_id_default = "fake.protected_area"
    source_name_default = "Fake protected-area catalog"
    served_default = _PROTECTED_SERVED


class FakeHazardProvider(FakeDataProvider):
    """Seeded hazard scores. Does not run a hazard model."""

    provider_id_default = "fake.hazard"
    source_name_default = "Fake hazard catalog"
    served_default = _HAZARD_SERVED


class FakeBuildingProvider(FakeDataProvider):
    """Seeded rooftop attributes. Refuses structural capacity."""

    provider_id_default = "fake.building"
    source_name_default = "Fake building catalog"
    served_default = _BUILDING_SERVED
    refused_default = _BUILDING_REFUSED


class FakeGridCapacityProvider(FakeDataProvider):
    """Seeded hosting-capacity values. Empty catalog stays empty."""

    provider_id_default = "fake.grid_capacity"
    source_name_default = "Fake grid-capacity catalog"
    served_default = _GRID_SERVED
