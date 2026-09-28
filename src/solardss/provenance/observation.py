"""Data-source provenance and indicator observations.

Distinct from ``ProvenanceRecord``, which stamps a SolarDSS computation.
``DataProvenance`` stamps a retrieved or derived *dataset value*.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from solardss.data.contracts import DatasetRef
from solardss.exceptions import InvalidParameterError
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.provenance.coverage import SpatialCoverage, TemporalCoverage
from solardss.provenance.record import ProvenanceRecord


class ValueOrigin(str, Enum):
    """Whether the stored number is a provider field or a derivation."""

    RAW = "raw"
    DERIVED = "derived"


class ObservationStatus(str, Enum):
    """Outcome of a fetch. Non-observed statuses never carry a value."""

    OBSERVED = "observed"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED_REGION = "unsupported_region"
    MISSING_COVERAGE = "missing_coverage"
    TEMPORARY_FAILURE = "temporary_failure"
    NOT_CONFIGURED = "not_configured"


class QualityFlag(str, Enum):
    """Well-known quality markers. Unknown flags may be stored as strings."""

    NOMINAL = "nominal"
    ESTIMATED = "estimated"
    INTERPOLATED = "interpolated"
    GAP_FILLED = "gap_filled"
    PRELIMINARY = "preliminary"
    PROXY = "proxy"
    UNKNOWN = "unknown"


class DataProvenance(BaseModel):
    """Where an indicator value came from, and how it was obtained."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_id: str = Field(min_length=1)
    source_name: str = Field(min_length=1)
    retrieved_at: datetime
    request_hash: str = Field(min_length=1)
    value_kind: ValueProvenanceKind
    value_origin: ValueOrigin
    source_version: str | None = None
    calculation_method: str | None = None
    dataset: DatasetRef | None = None
    computation: ProvenanceRecord | None = None

    @field_validator("retrieved_at")
    @classmethod
    def _aware_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise InvalidParameterError("retrieved_at must be timezone-aware")
        return value.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _derived_has_method(self) -> DataProvenance:
        if self.value_origin is ValueOrigin.DERIVED and not self.calculation_method:
            raise InvalidParameterError("derived values require calculation_method")
        if self.value_origin is ValueOrigin.RAW and self.calculation_method:
            raise InvalidParameterError("raw values must not set calculation_method")
        return self


class IndicatorObservation(BaseModel):
    """One indicator value plus the provenance required to audit it.

    If ``status`` is not ``observed``, ``value`` is null. SolarDSS never fills
    a missing observation with an invented number.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    indicator_id: str = Field(min_length=1)
    status: ObservationStatus
    provenance: DataProvenance
    spatial: SpatialCoverage
    temporal: TemporalCoverage
    value: float | str | bool | None = None
    unit: str | None = None
    quality_flags: tuple[str, ...] = ()
    assumptions: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _value_matches_status(self) -> IndicatorObservation:
        if self.status is ObservationStatus.OBSERVED:
            if self.value is None:
                raise InvalidParameterError("observed indicator values must not be null")
            if not self.unit:
                raise InvalidParameterError("observed indicator values require a unit")
        elif self.value is not None:
            raise InvalidParameterError(f"status {self.status.value} must not carry a value")
        return self
