"""Availability, health, and coverage reports for data providers."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator

from solardss.core.jsonutil import JsonValue
from solardss.exceptions import InvalidParameterError
from solardss.provenance.coverage import SpatialCoverage, TemporalCoverage


class AvailabilityStatus(str, Enum):
    """Whether a provider can serve a request. Never implies a numeric value."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED_REGION = "unsupported_region"
    MISSING_COVERAGE = "missing_coverage"
    TEMPORARY_FAILURE = "temporary_failure"
    NOT_CONFIGURED = "not_configured"


class HealthStatus(str, Enum):
    """Process/configuration health, independent of a spatial request."""

    OK = "ok"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


class ProviderAvailability(BaseModel):
    """Structured availability. Adapters must not treat this as an observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: AvailabilityStatus
    provider_id: str = Field(min_length=1)
    retryable: bool
    indicator_id: str | None = None
    reason: str = Field(min_length=1)


class ProviderHealth(BaseModel):
    """Result of ``health()``. Fake providers use an injected clock."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: HealthStatus
    provider_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    checked_at: datetime | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("checked_at")
    @classmethod
    def _aware(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        if value.tzinfo is None:
            raise InvalidParameterError("checked_at must be timezone-aware")
        return value.astimezone(timezone.utc)


class CoverageReport(BaseModel):
    """Whether the dataset covers the requested place and time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: AvailabilityStatus
    provider_id: str = Field(min_length=1)
    spatial: SpatialCoverage | None = None
    temporal: TemporalCoverage | None = None
    reason: str = Field(min_length=1)
