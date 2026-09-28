"""Geographic identifiers. SolarDSS does not geocode or transform CRS."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, model_validator

from solardss.exceptions import InvalidParameterError


class BoundingBox(BaseModel):
    """Axis-aligned box. Coordinates are caller-supplied; CRS is a label only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float
    crs: str = "EPSG:4326"

    @model_validator(mode="after")
    def _ordered(self) -> BoundingBox:
        if self.max_lon < self.min_lon or self.max_lat < self.min_lat:
            raise InvalidParameterError("Bounding box maxima must be >= minima")
        if self.crs == "EPSG:4326":
            if not -90.0 <= self.min_lat <= 90.0 or not -90.0 <= self.max_lat <= 90.0:
                raise InvalidParameterError("EPSG:4326 latitude must be in [-90, 90]")
            if not -180.0 <= self.min_lon <= 180.0 or not -180.0 <= self.max_lon <= 180.0:
                raise InvalidParameterError("EPSG:4326 longitude must be in [-180, 180]")
        return self


class GeographicRef(BaseModel):
    """Opaque place or geometry handle. SolarDSS does not geocode or open it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = None
    place_id: str | None = None
    longitude: float | None = None
    latitude: float | None = None
    bbox: BoundingBox | None = None
    geometry_media_type: str | None = None
    geometry_fingerprint: str | None = None
    crs: str | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _identifiable(self) -> GeographicRef:
        has_point = self.longitude is not None or self.latitude is not None
        if (self.longitude is None) != (self.latitude is None):
            raise InvalidParameterError("longitude and latitude must be supplied together")
        if has_point and self.crs in (None, "EPSG:4326"):
            lon = self.longitude
            lat = self.latitude
            if lon is None or lat is None:
                raise InvalidParameterError("longitude and latitude must be supplied together")
            if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
                raise InvalidParameterError("EPSG:4326 point is out of range")
        identifiable = any(
            (
                self.name,
                self.place_id,
                has_point,
                self.bbox is not None,
                self.geometry_fingerprint,
            )
        )
        if not identifiable:
            raise InvalidParameterError(
                "GeographicRef requires a name, place_id, point, bbox, or geometry fingerprint"
            )
        return self
