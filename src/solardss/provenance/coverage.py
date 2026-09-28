"""Spatial and temporal coverage of a dataset or request."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, ConfigDict, Field

from solardss.data.geo import BoundingBox


class SpatialCoverage(BaseModel):
    """Where a value applies, and at what native geographic grain.

    ``resolution`` is a provider-stated grain (for example ``30m`` or
    ``0.25deg``), not a computed statistic.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolution: str = Field(min_length=1)
    crs: str | None = None
    bbox: BoundingBox | None = None
    geometry_fingerprint: str | None = None
    native_cell_size_m: float | None = None


class TemporalCoverage(BaseModel):
    """When a value applies, and at what native time grain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resolution: str = Field(min_length=1)
    start: date | None = None
    end: date | None = None
    aggregation: str | None = None
    calendar: str = "gregorian"
