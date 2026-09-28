"""Provenance records attached to every computed result."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from solardss.core.jsonutil import JsonValue, json_params
from solardss.core.methods import METHOD_VERSIONS, MethodId
from solardss.version import __version__


class SoftwareStamp(BaseModel):
    """Which package and method produced a result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    package: str = "solardss"
    version: str
    method_id: str
    method_version: str


class ProvenanceRecord(BaseModel):
    """Inputs, parameters, and software identity for one computation.

    ``input_fingerprints`` are content hashes, not paths. Callers may attach
    their own dataset identifiers in ``parameters`` if they need to name sources.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stamp: SoftwareStamp
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    input_fingerprints: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def software_stamp(method_id: MethodId) -> SoftwareStamp:
    """Stamp the running SolarDSS version and the method's numerical version."""

    return SoftwareStamp(
        package="solardss",
        version=__version__,
        method_id=method_id.value,
        method_version=METHOD_VERSIONS[method_id],
    )


def provenance_record(
    method_id: MethodId,
    *,
    parameters: Mapping[str, object] | None = None,
    input_fingerprints: Sequence[str] = (),
    notes: Sequence[str] = (),
) -> ProvenanceRecord:
    """Build a frozen provenance record for a method call."""

    return ProvenanceRecord(
        stamp=software_stamp(method_id),
        parameters=json_params(parameters or {}),
        input_fingerprints=tuple(input_fingerprints),
        notes=tuple(notes),
    )
