"""Dataset references. Identifiers are caller-supplied; no builtin paths."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class DatasetRef(BaseModel):
    """Name a dataset without implying that SolarDSS can read it.

    ``uri`` is optional and opaque. SolarDSS never opens it. Adapters may store
    a URL, a catalog key, or a content hash here.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    uri: str | None = None
    fingerprint: str | None = None
    notes: tuple[str, ...] = ()
