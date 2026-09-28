"""Provenance types for SolarDSS results and retrieved indicator values.

``ProvenanceRecord`` is imported eagerly because computed results depend on it.
Observation and coverage types are loaded lazily so ``solardss.core`` can import
this package without a cycle through the indicator registry.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from solardss.provenance.record import (
    ProvenanceRecord,
    SoftwareStamp,
    provenance_record,
    software_stamp,
)

if TYPE_CHECKING:
    from solardss.provenance.coverage import SpatialCoverage, TemporalCoverage
    from solardss.provenance.observation import (
        DataProvenance,
        IndicatorObservation,
        ObservationStatus,
        QualityFlag,
        ValueOrigin,
    )

__all__ = [
    "DataProvenance",
    "IndicatorObservation",
    "ObservationStatus",
    "ProvenanceRecord",
    "QualityFlag",
    "SoftwareStamp",
    "SpatialCoverage",
    "TemporalCoverage",
    "ValueOrigin",
    "provenance_record",
    "software_stamp",
]

_LAZY_ATTRS = {
    "DataProvenance": ("solardss.provenance.observation", "DataProvenance"),
    "IndicatorObservation": ("solardss.provenance.observation", "IndicatorObservation"),
    "ObservationStatus": ("solardss.provenance.observation", "ObservationStatus"),
    "QualityFlag": ("solardss.provenance.observation", "QualityFlag"),
    "SpatialCoverage": ("solardss.provenance.coverage", "SpatialCoverage"),
    "TemporalCoverage": ("solardss.provenance.coverage", "TemporalCoverage"),
    "ValueOrigin": ("solardss.provenance.observation", "ValueOrigin"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
