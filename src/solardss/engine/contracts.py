"""Public types for the indicator computation engine.

Providers never call each other. The engine walks an explicit DAG, fetches
through registered providers, then runs registered derivations.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from solardss.core.jsonutil import JsonValue
from solardss.core.quantity import Quantity
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.data.tables import CandidateTable
from solardss.indicators.contracts import DecisionMode, ProjectType
from solardss.provenance.coverage import TemporalCoverage
from solardss.provenance.observation import IndicatorObservation

ContextValue = float | str | bool | Quantity


class IndicatorOutcome(str, Enum):
    """Engine-level status. Missing values stay missing; zero is never substituted."""

    AVAILABLE = "available"
    MISSING = "missing"
    UNSUPPORTED = "unsupported"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    EXTERNALLY_REQUIRED = "externally_required"


class DependencyKind(str, Enum):
    """What a node needs before it can be resolved."""

    INDICATOR = "indicator"
    SOURCE_TYPE = "source_type"


class ResolutionPath(str, Enum):
    """How the engine may obtain a value. Paths are exclusive of implicit hops."""

    PROVIDER = "provider"
    CONTEXT = "context"
    DERIVED = "derived"
    PROVIDER_OR_DERIVED = "provider_or_derived"


class BatchScope(str, Enum):
    """Whether a candidate-set evaluation may reuse one AOI request."""

    POINT = "point"
    SHARED_AOI = "shared_aoi"


class Dependency(BaseModel):
    """One explicit edge. Source types are not indicators and are not fetched."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: DependencyKind
    target: str = Field(min_length=1)
    required: bool = True


class IndicatorNode(BaseModel):
    """One DAG node. ``implemented`` is false when SolarDSS has no method yet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    indicator_id: str = Field(min_length=1)
    dependencies: tuple[Dependency, ...] = ()
    resolution: ResolutionPath = ResolutionPath.PROVIDER
    batch_scope: BatchScope = BatchScope.POINT
    implemented: bool = True


class EngineContext(BaseModel):
    """Caller-supplied values, design variables, and batch options."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: DecisionMode = DecisionMode.FIXED_SITE
    project_type: ProjectType | None = None
    values: dict[str, ContextValue] = Field(default_factory=dict)
    parameters: dict[str, JsonValue] = Field(default_factory=dict)
    temporal: TemporalCoverage | None = None
    shared_bbox: BoundingBox | None = None
    reuse_aoi_window: bool = True


class IndicatorResult(BaseModel):
    """One indicator after engine evaluation. Non-available outcomes have no value."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    indicator_id: str = Field(min_length=1)
    outcome: IndicatorOutcome
    unit: str | None = None
    provider_id: str | None = None
    observation: IndicatorObservation | None = None
    reason: str = Field(min_length=1)
    missing_dependencies: tuple[str, ...] = ()

    @property
    def value(self) -> float | str | bool | None:
        """Observed value, or ``None`` when the outcome is not available."""

        if self.outcome is not IndicatorOutcome.AVAILABLE or self.observation is None:
            return None
        return self.observation.value


class IndicatorSet(BaseModel):
    """Structured result of ``compute``. Includes requested ids and expanded deps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested: tuple[str, ...]
    evaluation_order: tuple[str, ...]
    results: tuple[IndicatorResult, ...]
    geometry: GeographicRef | None = None
    bbox: BoundingBox | None = None

    def get(self, indicator_id: str) -> IndicatorResult | None:
        """Return the result for ``indicator_id``, if evaluated."""

        for item in self.results:
            if item.indicator_id == indicator_id:
                return item
        return None

    def outcome_of(self, indicator_id: str) -> IndicatorOutcome | None:
        """Outcome for ``indicator_id``, if evaluated."""

        item = self.get(indicator_id)
        return None if item is None else item.outcome

    def by_outcome(self, outcome: IndicatorOutcome) -> tuple[IndicatorResult, ...]:
        """Results with a given engine outcome."""

        return tuple(item for item in self.results if item.outcome is outcome)

    def as_mapping(self) -> Mapping[str, IndicatorResult]:
        """Indicator id → result."""

        return {item.indicator_id: item for item in self.results}


class CandidateIndicatorSet(BaseModel):
    """Per-candidate indicator sets plus a table that uses NA for missing values."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    requested: tuple[str, ...]
    evaluation_order: tuple[str, ...]
    table: CandidateTable
    sets: tuple[IndicatorSet, ...]
    shared_bbox: BoundingBox | None = None
    prefetch_providers: tuple[str, ...] = ()

    def for_candidate(self, candidate_id: str) -> IndicatorSet | None:
        """Return the set whose geometry name matches ``candidate_id``."""

        for item in self.sets:
            if item.geometry is not None and item.geometry.name == candidate_id:
                return item
        return None


class DerivedInputs:
    """Parent results passed to a derivation. Missing parents are present as outcomes."""

    def __init__(self, results: Mapping[str, IndicatorResult]) -> None:
        self._results = dict(results)

    def get(self, indicator_id: str) -> IndicatorResult | None:
        return self._results.get(indicator_id)

    def available_value(self, indicator_id: str) -> float | str | bool | None:
        """Return the parent value only when it is available. Never invents zero."""

        item = self._results.get(indicator_id)
        if item is None or item.outcome is not IndicatorOutcome.AVAILABLE:
            return None
        return item.value

    def items(self) -> Sequence[tuple[str, IndicatorResult]]:
        return tuple(self._results.items())
