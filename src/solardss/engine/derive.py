"""Derivation calculators invoked by the engine after dependencies are resolved.

Production SolarDSS ships no POA transposition. Tests may inject calculators.
A calculator must not treat a missing parent as zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Protocol

from solardss.core.hashing import fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.data.contracts import DatasetRef
from solardss.engine.contracts import DerivedInputs, EngineContext, IndicatorResult
from solardss.indicators.contracts import IndicatorSpec, ValueProvenanceKind
from solardss.provenance.coverage import SpatialCoverage, TemporalCoverage
from solardss.provenance.observation import (
    DataProvenance,
    IndicatorObservation,
    ObservationStatus,
    QualityFlag,
    ValueOrigin,
)
from solardss.provenance.record import provenance_record
from solardss.providers.request import FetchRequest


class DerivedCalculator(Protocol):
    """Engine-side derivation. Must not call providers."""

    @property
    def indicator_id(self) -> str:
        """Canonical indicator this calculator produces."""

    @property
    def method_id(self) -> str:
        """Recorded on ``DataProvenance.calculation_method``."""

    def required_inputs(self) -> tuple[str, ...]:
        """Indicator ids that must be available before ``compute``."""

    def can_compute(self, inputs: DerivedInputs, context: EngineContext) -> bool:
        """Return true only when every required parent is available."""

    def compute(
        self,
        inputs: DerivedInputs,
        *,
        spec: IndicatorSpec,
        context: EngineContext,
        request: FetchRequest,
        retrieved_at: datetime,
    ) -> IndicatorObservation:
        """Return an observation. Must not invent a value for a missing parent."""


def parent_fingerprints(results: Mapping[str, IndicatorResult]) -> tuple[str, ...]:
    """Stable hashes of available parent observations, in id order."""

    fingerprints: list[str] = []
    for indicator_id in sorted(results):
        item = results[indicator_id]
        if item.observation is None:
            continue
        fingerprints.append(item.observation.provenance.request_hash)
    return tuple(fingerprints)


def derived_observation(
    *,
    spec: IndicatorSpec,
    value: float | str | bool,
    unit: str,
    request: FetchRequest,
    retrieved_at: datetime,
    method_id: str,
    inputs: Mapping[str, IndicatorResult],
    assumptions: Sequence[str] = (),
    spatial: SpatialCoverage | None = None,
    temporal: TemporalCoverage | None = None,
) -> IndicatorObservation:
    """Build a derived observation with parent fingerprints in provenance."""

    request_hash = fingerprint_canonical(
        {
            "engine": "indicator_engine",
            "indicator_id": spec.id,
            "method_id": method_id,
            "parents": list(parent_fingerprints(inputs)),
            "request": request.model_dump(mode="json"),
        }
    )
    parent_ids = tuple(
        item.indicator_id for item in inputs.values() if item.observation is not None
    )
    computation = provenance_record(
        MethodId.INDICATOR_ENGINE_DERIVE,
        parameters={
            "indicator_id": spec.id,
            "calculator": method_id,
            "parents": ",".join(parent_ids),
        },
        input_fingerprints=parent_fingerprints(inputs),
        notes=tuple(assumptions),
    )
    return IndicatorObservation(
        indicator_id=spec.id,
        status=ObservationStatus.OBSERVED,
        provenance=DataProvenance(
            provider_id="solardss.indicator_engine",
            source_name="IndicatorEngine",
            retrieved_at=retrieved_at,
            request_hash=request_hash,
            value_kind=ValueProvenanceKind.DERIVED_METHOD,
            value_origin=ValueOrigin.DERIVED,
            calculation_method=method_id,
            dataset=DatasetRef(name="indicator-engine", media_type="application/json"),
            computation=computation,
        ),
        spatial=spatial
        or SpatialCoverage(
            resolution="derived",
            bbox=request.bbox,
            crs=None if request.location is None else request.location.crs,
        ),
        temporal=temporal or request.temporal or TemporalCoverage(resolution="derived"),
        value=value,
        unit=unit,
        quality_flags=(QualityFlag.NOMINAL.value,),
        assumptions=tuple(assumptions),
    )


class CalculatorRegistry:
    """One calculator per indicator id. Last registration wins."""

    def __init__(self, calculators: Sequence[DerivedCalculator] = ()) -> None:
        self._items: dict[str, DerivedCalculator] = {}
        for item in calculators:
            self._items[item.indicator_id] = item

    def get(self, indicator_id: str) -> DerivedCalculator | None:
        return self._items.get(indicator_id)

    def extra_deps(self) -> dict[str, tuple[str, ...]]:
        return {key: value.required_inputs() for key, value in self._items.items()}
