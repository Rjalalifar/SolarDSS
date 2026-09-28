"""Fetch requests and stable cache keys."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solardss.core.hashing import fingerprint_canonical
from solardss.core.jsonutil import JsonValue
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.exceptions import InvalidParameterError
from solardss.provenance.coverage import TemporalCoverage


class FetchRequest(BaseModel):
    """What to retrieve. Does not name a filesystem path."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    indicator_id: str = Field(min_length=1)
    location: GeographicRef | None = None
    bbox: BoundingBox | None = None
    temporal: TemporalCoverage | None = None
    parameters: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _has_place(self) -> FetchRequest:
        if self.location is None and self.bbox is None:
            raise InvalidParameterError("FetchRequest requires location or bbox")
        return self


def request_cache_key(provider_id: str, request: FetchRequest) -> str:
    """Deterministic SHA-256 over provider id and the canonical request."""

    payload = {
        "provider_id": provider_id,
        "request": request.model_dump(mode="json"),
    }
    return fingerprint_canonical(payload)
