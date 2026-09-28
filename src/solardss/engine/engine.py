"""Indicator computation engine.

Walks the explicit DAG, selects providers, fetches, derives, validates units,
and returns an ``IndicatorSet``. Providers are never asked to call each other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from solardss.core.hashing import fingerprint_canonical
from solardss.core.quantity import Quantity
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.data.tables import DEFAULT_ID_COLUMN, CandidateTable, validate_candidate_table
from solardss.decision.problem import SolarDecisionProblem
from solardss.engine.contracts import (
    BatchScope,
    CandidateIndicatorSet,
    ContextValue,
    DependencyKind,
    DerivedInputs,
    EngineContext,
    IndicatorNode,
    IndicatorOutcome,
    IndicatorResult,
    IndicatorSet,
    ResolutionPath,
)
from solardss.engine.dependencies import DEFAULT_GRAPH
from solardss.engine.derive import CalculatorRegistry, DerivedCalculator
from solardss.engine.graph import IndicatorGraph
from solardss.engine.providers import ProviderRegistry, registry_from_providers
from solardss.exceptions import SchemaError, UnitError
from solardss.indicators.catalog import registry as default_registry
from solardss.indicators.contracts import Controllability, IndicatorSpec, ValueProvenanceKind
from solardss.indicators.ids import IndicatorId
from solardss.indicators.registry import IndicatorRegistry
from solardss.provenance.coverage import SpatialCoverage, TemporalCoverage
from solardss.provenance.observation import (
    DataProvenance,
    IndicatorObservation,
    ObservationStatus,
    QualityFlag,
    ValueOrigin,
)
from solardss.providers.clock import Clock, UtcClock
from solardss.providers.contracts import AvailabilityStatus, ProviderAvailability
from solardss.providers.protocols import DataProvider
from solardss.providers.request import FetchRequest, request_cache_key

_ENGINE_PROVIDER_ID = "solardss.indicator_engine"
_POINT_PAD = 0.0002


class IndicatorEngine:
    """Resolve, fetch, and derive indicators for a site or a candidate set."""

    def __init__(
        self,
        providers: Sequence[DataProvider | tuple[DataProvider, Sequence[str]]]
        | ProviderRegistry = (),
        *,
        registry: IndicatorRegistry | None = None,
        graph: IndicatorGraph | None = None,
        calculators: Sequence[DerivedCalculator] = (),
        clock: Clock | None = None,
    ) -> None:
        self._registry = default_registry if registry is None else registry
        self._graph = DEFAULT_GRAPH if graph is None else graph
        if isinstance(providers, ProviderRegistry):
            self._providers = providers
        else:
            self._providers = registry_from_providers(providers)
        self._calculators = CalculatorRegistry(calculators)
        self._clock: Clock = UtcClock() if clock is None else clock
        self._fetch_cache: dict[str, IndicatorObservation] = {}
        self._prefetch_ids: list[str] = []

    def compute(
        self,
        *,
        geometry: GeographicRef | BoundingBox | None = None,
        indicators: Sequence[str | IndicatorId] | None = None,
        context: EngineContext | None = None,
    ) -> IndicatorSet:
        """Evaluate ``indicators`` at one geometry. Missing values stay missing."""

        ctx = context if context is not None else EngineContext()
        requested = self._requested_ids(indicators, ctx)
        location, bbox = _split_geometry(geometry, ctx.shared_bbox)
        return self._compute_place(requested, location=location, bbox=bbox, context=ctx)

    def compute_for_problem(
        self,
        problem: SolarDecisionProblem,
        *,
        indicators: Sequence[str | IndicatorId] | None = None,
        context: EngineContext | None = None,
        candidates: CandidateTable | pd.DataFrame | None = None,
        x_column: str = "longitude",
        y_column: str = "latitude",
        crs: str = "EPSG:4326",
        id_column: str = DEFAULT_ID_COLUMN,
    ) -> IndicatorSet | CandidateIndicatorSet:
        """Evaluate the problem's required indicators, or an explicit list."""

        base = context if context is not None else EngineContext()
        ctx = base.model_copy(update={"mode": problem.mode, "project_type": problem.project_type})
        wanted = indicators
        if wanted is None:
            specs = self._registry.required_for(
                problem.mode,
                problem.project_type,
                scope=problem.site_selection_scope,
            )
            wanted = tuple(spec.id for spec in specs)
        if candidates is not None:
            return self.compute_for_candidates(
                candidates,
                indicators=wanted,
                context=ctx,
                x_column=x_column,
                y_column=y_column,
                crs=crs,
                id_column=id_column,
            )
        geometry = problem.location or problem.aoi or problem.region
        return self.compute(geometry=geometry, indicators=wanted, context=ctx)

    def compute_for_candidates(
        self,
        table: CandidateTable | pd.DataFrame,
        *,
        indicators: Sequence[str | IndicatorId],
        context: EngineContext | None = None,
        x_column: str = "longitude",
        y_column: str = "latitude",
        crs: str = "EPSG:4326",
        id_column: str = DEFAULT_ID_COLUMN,
    ) -> CandidateIndicatorSet:
        """Evaluate each candidate. Shared-AOI indicators are fetched once."""

        ctx = context if context is not None else EngineContext()
        if isinstance(table, CandidateTable):
            candidates = table
        else:
            candidates = validate_candidate_table(table, id_column=id_column)
        missing = [name for name in (x_column, y_column) if name not in candidates.frame.columns]
        if missing:
            raise SchemaError(f"Candidate table is missing coordinate columns: {missing}")
        requested = self._requested_ids(indicators, ctx)
        shared_bbox = ctx.shared_bbox or _bbox_from_candidates(
            candidates, x_column=x_column, y_column=y_column, crs=crs
        )
        batch_ctx = ctx.model_copy(update={"shared_bbox": shared_bbox})
        expanded = self._expanded(requested)
        order = self._graph.topological_order(expanded)
        shared_ids = tuple(
            item for item in order if self._graph.node(item).batch_scope is BatchScope.SHARED_AOI
        )
        self._prefetch(expanded, batch_ctx, bbox=shared_bbox)
        shared_done: dict[str, IndicatorResult] = {}
        if shared_ids:
            shared_set = self._compute_place(
                shared_ids,
                location=None,
                bbox=shared_bbox,
                context=batch_ctx,
                seed=shared_done,
            )
            shared_done = dict(shared_set.as_mapping())
        sets: list[IndicatorSet] = []
        rows: list[dict[str, object]] = []
        for _, row in candidates.frame.iterrows():
            candidate_id = str(row[candidates.id_column])
            location = GeographicRef(
                name=candidate_id,
                longitude=float(row[x_column]),
                latitude=float(row[y_column]),
                crs=crs,
            )
            result = self._compute_place(
                requested,
                location=location,
                bbox=shared_bbox,
                context=batch_ctx,
                seed=dict(shared_done),
            )
            sets.append(result)
            record: dict[str, object] = {candidates.id_column: candidate_id}
            for indicator_id in requested:
                item = result.get(indicator_id)
                record[indicator_id] = pd.NA if item is None or item.value is None else item.value
                record[f"{indicator_id}_outcome"] = None if item is None else item.outcome.value
            rows.append(record)
        frame = candidates.frame.copy()
        extra = pd.DataFrame(rows)
        merged = frame.merge(extra, on=candidates.id_column, how="left")
        updated = CandidateTable(frame=merged, id_column=candidates.id_column)
        return CandidateIndicatorSet(
            requested=requested,
            evaluation_order=order,
            table=updated,
            sets=tuple(sets),
            shared_bbox=shared_bbox,
            prefetch_providers=tuple(self._prefetch_ids),
        )

    def _requested_ids(
        self,
        indicators: Sequence[str | IndicatorId] | None,
        context: EngineContext,
    ) -> tuple[str, ...]:
        if indicators is None:
            specs = self._registry.required_for(context.mode, context.project_type)
            return tuple(spec.id for spec in specs)
        return tuple(self._registry.get(item).id for item in indicators)

    def _expanded(self, requested: Sequence[str]) -> tuple[str, ...]:
        return self._graph.expand(requested, extra_indicator_deps=self._calculators.extra_deps())

    def _compute_place(
        self,
        requested: Sequence[str],
        *,
        location: GeographicRef | None,
        bbox: BoundingBox | None,
        context: EngineContext,
        seed: dict[str, IndicatorResult] | None = None,
    ) -> IndicatorSet:
        expanded = self._expanded(requested)
        order = self._graph.topological_order(expanded)
        done: dict[str, IndicatorResult] = {} if seed is None else dict(seed)
        for indicator_id in order:
            if indicator_id in done:
                continue
            done[indicator_id] = self._evaluate(
                indicator_id, done, location=location, bbox=bbox, context=context
            )
        results = tuple(done[item] for item in order)
        return IndicatorSet(
            requested=tuple(requested),
            evaluation_order=order,
            results=results,
            geometry=location,
            bbox=bbox,
        )

    def _evaluate(
        self,
        indicator_id: str,
        done: Mapping[str, IndicatorResult],
        *,
        location: GeographicRef | None,
        bbox: BoundingBox | None,
        context: EngineContext,
    ) -> IndicatorResult:
        spec = self._registry.get(indicator_id)
        node = self._graph.node(indicator_id)
        if indicator_id in context.values:
            return self._from_context(spec, context.values[indicator_id], location, bbox, context)
        if indicator_id == IndicatorId.PV_LOCATION.value:
            from_place = self._from_location(spec, location, bbox, context)
            if from_place is not None:
                return from_place
        request = _fetch_request(indicator_id, location, bbox, context)
        calculator = self._calculators.get(indicator_id)
        if node.resolution in {ResolutionPath.PROVIDER, ResolutionPath.PROVIDER_OR_DERIVED}:
            if request is not None:
                selected = self._providers.select(indicator_id, request)
                if selected is not None:
                    provider, availability = selected
                    if availability.status is AvailabilityStatus.AVAILABLE:
                        observation = self._fetch(provider, request)
                        return self._from_observation(spec, observation, provider.provider_id)
                    if node.resolution is ResolutionPath.PROVIDER:
                        return self._from_availability(spec, availability)
                elif node.resolution is ResolutionPath.PROVIDER:
                    return self._no_provider(spec)
            elif node.resolution is ResolutionPath.PROVIDER:
                return self._geometry_missing(spec)
        missing_required = _missing_required_indicators(node, done)
        if calculator is not None:
            inputs = DerivedInputs(done)
            if missing_required:
                return _blocked(spec, missing_required, done)
            if calculator.can_compute(inputs, context):
                if request is None:
                    return self._geometry_missing(spec)
                observation = calculator.compute(
                    inputs,
                    spec=spec,
                    context=context,
                    request=request,
                    retrieved_at=self._clock.now(),
                )
                return self._from_observation(spec, observation, _ENGINE_PROVIDER_ID)
            return IndicatorResult(
                indicator_id=spec.id,
                outcome=IndicatorOutcome.MISSING,
                unit=spec.unit,
                reason="derivation inputs are not available; zero is not substituted",
                missing_dependencies=calculator.required_inputs(),
            )
        if spec.controllability in {
            Controllability.CONTROLLABLE,
            Controllability.FIXED,
            Controllability.EXTERNALLY_PROVIDED,
        }:
            return IndicatorResult(
                indicator_id=spec.id,
                outcome=IndicatorOutcome.EXTERNALLY_REQUIRED,
                unit=spec.unit,
                reason="value must be supplied in context; it is not inferred",
            )
        if not node.implemented or node.resolution is ResolutionPath.DERIVED:
            if missing_required:
                return _blocked(spec, missing_required, done)
            return IndicatorResult(
                indicator_id=spec.id,
                outcome=IndicatorOutcome.UNSUPPORTED,
                unit=spec.unit,
                reason=(
                    spec.derivation_method.description
                    if spec.derivation_method is not None
                    else "SolarDSS has no implemented method for this indicator"
                ),
                missing_dependencies=missing_required,
            )
        if request is None:
            return self._geometry_missing(spec)
        return self._no_provider(spec)

    def _from_context(
        self,
        spec: IndicatorSpec,
        raw: ContextValue,
        location: GeographicRef | None,
        bbox: BoundingBox | None,
        context: EngineContext,
    ) -> IndicatorResult:
        try:
            value, unit = _coerce_context_value(spec, raw)
        except UnitError as exc:
            return IndicatorResult(
                indicator_id=spec.id,
                outcome=IndicatorOutcome.MISSING,
                unit=spec.unit,
                reason=exc.message,
            )
        retrieved_at = self._clock.now()
        request_hash = fingerprint_canonical(
            {
                "engine": "context",
                "indicator_id": spec.id,
                "value": value if not isinstance(value, bool) else str(value),
                "unit": unit,
            }
        )
        observation = IndicatorObservation(
            indicator_id=spec.id,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id="caller",
                source_name="engine context",
                retrieved_at=retrieved_at,
                request_hash=request_hash,
                value_kind=ValueProvenanceKind.EXTERNALLY_PROVIDED,
                value_origin=ValueOrigin.RAW,
            ),
            spatial=_spatial(location, bbox),
            temporal=context.temporal or TemporalCoverage(resolution="context"),
            value=value,
            unit=unit,
            quality_flags=(QualityFlag.NOMINAL.value,),
            assumptions=("caller-supplied context value",),
        )
        return IndicatorResult(
            indicator_id=spec.id,
            outcome=IndicatorOutcome.AVAILABLE,
            unit=unit,
            provider_id="caller",
            observation=observation,
            reason="supplied in engine context",
        )

    def _from_location(
        self,
        spec: IndicatorSpec,
        location: GeographicRef | None,
        bbox: BoundingBox | None,
        context: EngineContext,
    ) -> IndicatorResult | None:
        if location is None and bbox is None:
            return None
        if location is not None and location.longitude is not None:
            token = f"{location.longitude},{location.latitude}"
        elif location is not None and location.name:
            token = location.name
        elif location is not None and location.place_id:
            token = location.place_id
        elif bbox is not None:
            token = f"{bbox.min_lon},{bbox.min_lat},{bbox.max_lon},{bbox.max_lat}"
        else:
            return None
        retrieved_at = self._clock.now()
        observation = IndicatorObservation(
            indicator_id=spec.id,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id="caller",
                source_name="compute geometry",
                retrieved_at=retrieved_at,
                request_hash=fingerprint_canonical({"pv_location": token}),
                value_kind=ValueProvenanceKind.EXTERNALLY_PROVIDED,
                value_origin=ValueOrigin.RAW,
            ),
            spatial=_spatial(location, bbox),
            temporal=context.temporal or TemporalCoverage(resolution="context"),
            value=token,
            unit=spec.unit,
            quality_flags=(QualityFlag.NOMINAL.value,),
            assumptions=("PV_LOCATION is the geometry passed to compute",),
        )
        return IndicatorResult(
            indicator_id=spec.id,
            outcome=IndicatorOutcome.AVAILABLE,
            unit=spec.unit,
            provider_id="caller",
            observation=observation,
            reason="geometry supplied to compute",
        )

    def _from_observation(
        self,
        spec: IndicatorSpec,
        observation: IndicatorObservation,
        provider_id: str,
    ) -> IndicatorResult:
        if observation.status is ObservationStatus.OBSERVED:
            if observation.unit != spec.unit:
                return IndicatorResult(
                    indicator_id=spec.id,
                    outcome=IndicatorOutcome.MISSING,
                    unit=spec.unit,
                    provider_id=provider_id,
                    reason=(
                        f"unit mismatch: expected {spec.unit!r}, received {observation.unit!r}"
                    ),
                )
            return IndicatorResult(
                indicator_id=spec.id,
                outcome=IndicatorOutcome.AVAILABLE,
                unit=observation.unit,
                provider_id=provider_id,
                observation=observation,
                reason="provider observation",
            )
        outcome = _outcome_from_observation(observation.status)
        reason = observation.assumptions[0] if observation.assumptions else observation.status.value
        return IndicatorResult(
            indicator_id=spec.id,
            outcome=outcome,
            unit=spec.unit,
            provider_id=provider_id,
            observation=None,
            reason=reason,
        )

    def _from_availability(
        self, spec: IndicatorSpec, availability: ProviderAvailability
    ) -> IndicatorResult:
        outcome = _outcome_from_availability(availability.status)
        return IndicatorResult(
            indicator_id=spec.id,
            outcome=outcome,
            unit=spec.unit,
            provider_id=availability.provider_id,
            reason=availability.reason,
        )

    def _no_provider(self, spec: IndicatorSpec) -> IndicatorResult:
        if spec.controllability is Controllability.EXTERNALLY_PROVIDED:
            outcome = IndicatorOutcome.EXTERNALLY_REQUIRED
            reason = "no provider is registered; a caller or operator value is required"
        else:
            outcome = IndicatorOutcome.PROVIDER_UNAVAILABLE
            reason = "no provider is registered for this indicator"
        return IndicatorResult(
            indicator_id=spec.id,
            outcome=outcome,
            unit=spec.unit,
            reason=reason,
        )

    def _geometry_missing(self, spec: IndicatorSpec) -> IndicatorResult:
        return IndicatorResult(
            indicator_id=spec.id,
            outcome=IndicatorOutcome.MISSING,
            unit=spec.unit,
            reason="compute requires a point or bbox; zero is not substituted",
        )

    def _fetch(self, provider: DataProvider, request: FetchRequest) -> IndicatorObservation:
        key = request_cache_key(provider.provider_id, request)
        cached = self._fetch_cache.get(key)
        if cached is not None:
            return cached
        observation = provider.fetch(request)
        if observation.status is not ObservationStatus.TEMPORARY_FAILURE:
            self._fetch_cache[key] = observation
        return observation

    def _prefetch(
        self,
        indicator_ids: Sequence[str],
        context: EngineContext,
        *,
        bbox: BoundingBox | None,
    ) -> None:
        self._prefetch_ids = []
        if not context.reuse_aoi_window or bbox is None:
            return
        seen: set[str] = set()
        for indicator_id in indicator_ids:
            for provider in self._providers.providers_for(indicator_id):
                if provider.provider_id in seen:
                    continue
                prefetch = getattr(provider, "prefetch", None)
                if not callable(prefetch):
                    continue
                seen.add(provider.provider_id)
                request = FetchRequest(
                    indicator_id=indicator_id,
                    bbox=bbox,
                    temporal=context.temporal,
                    parameters=dict(context.parameters),
                )
                prefetch(request)
                self._prefetch_ids.append(provider.provider_id)


def _split_geometry(
    geometry: GeographicRef | BoundingBox | None,
    shared: BoundingBox | None,
) -> tuple[GeographicRef | None, BoundingBox | None]:
    if isinstance(geometry, BoundingBox):
        return None, geometry
    if isinstance(geometry, GeographicRef):
        box = geometry.bbox if geometry.bbox is not None else shared
        return geometry, box
    return None, shared


def _fetch_request(
    indicator_id: str,
    location: GeographicRef | None,
    bbox: BoundingBox | None,
    context: EngineContext,
) -> FetchRequest | None:
    fetch_location = location
    if location is not None and location.longitude is not None:
        fetch_location = location.model_copy(update={"name": None, "place_id": None})
    if fetch_location is None and bbox is None:
        return None
    return FetchRequest(
        indicator_id=indicator_id,
        location=fetch_location,
        bbox=bbox,
        temporal=context.temporal,
        parameters=dict(context.parameters),
    )


def _spatial(location: GeographicRef | None, bbox: BoundingBox | None) -> SpatialCoverage:
    crs = None
    if location is not None:
        crs = location.crs
    elif bbox is not None:
        crs = bbox.crs
    box = bbox
    if box is None and location is not None:
        box = location.bbox
    return SpatialCoverage(resolution="request", crs=crs, bbox=box)


def _coerce_context_value(spec: IndicatorSpec, raw: ContextValue) -> tuple[float | str | bool, str]:
    if isinstance(raw, Quantity):
        return raw.require_unit(spec.unit), spec.unit
    if isinstance(raw, bool):
        return raw, spec.unit
    if isinstance(raw, str):
        return raw, spec.unit
    return float(raw), spec.unit


def _missing_required_indicators(
    node: IndicatorNode, done: Mapping[str, IndicatorResult]
) -> tuple[str, ...]:
    missing: list[str] = []
    for dep in node.dependencies:
        if dep.kind is not DependencyKind.INDICATOR or not dep.required:
            continue
        parent = done.get(dep.target)
        if parent is None or parent.outcome is not IndicatorOutcome.AVAILABLE:
            missing.append(dep.target)
    return tuple(missing)


def _blocked(
    spec: IndicatorSpec,
    missing_required: tuple[str, ...],
    done: Mapping[str, IndicatorResult],
) -> IndicatorResult:
    external = all(
        (done.get(item) is not None and done[item].outcome is IndicatorOutcome.EXTERNALLY_REQUIRED)
        for item in missing_required
    )
    outcome = IndicatorOutcome.EXTERNALLY_REQUIRED if external else IndicatorOutcome.MISSING
    return IndicatorResult(
        indicator_id=spec.id,
        outcome=outcome,
        unit=spec.unit,
        reason="required dependencies are not available; zero is not substituted",
        missing_dependencies=missing_required,
    )


def _outcome_from_observation(status: ObservationStatus) -> IndicatorOutcome:
    if status is ObservationStatus.MISSING_COVERAGE:
        return IndicatorOutcome.MISSING
    if status is ObservationStatus.UNAVAILABLE:
        return IndicatorOutcome.MISSING
    if status is ObservationStatus.UNSUPPORTED_REGION:
        return IndicatorOutcome.UNSUPPORTED
    if status is ObservationStatus.TEMPORARY_FAILURE:
        return IndicatorOutcome.PROVIDER_UNAVAILABLE
    if status is ObservationStatus.NOT_CONFIGURED:
        return IndicatorOutcome.PROVIDER_UNAVAILABLE
    return IndicatorOutcome.MISSING


def _outcome_from_availability(status: AvailabilityStatus) -> IndicatorOutcome:
    if status is AvailabilityStatus.MISSING_COVERAGE:
        return IndicatorOutcome.MISSING
    if status is AvailabilityStatus.UNSUPPORTED_REGION:
        return IndicatorOutcome.UNSUPPORTED
    if status is AvailabilityStatus.NOT_CONFIGURED:
        return IndicatorOutcome.PROVIDER_UNAVAILABLE
    if status is AvailabilityStatus.TEMPORARY_FAILURE:
        return IndicatorOutcome.PROVIDER_UNAVAILABLE
    return IndicatorOutcome.PROVIDER_UNAVAILABLE


def _bbox_from_candidates(
    table: CandidateTable,
    *,
    x_column: str,
    y_column: str,
    crs: str,
) -> BoundingBox:
    xs = [float(value) for value in table.frame[x_column].tolist()]
    ys = [float(value) for value in table.frame[y_column].tolist()]
    pad = _POINT_PAD
    min_lon = min(xs) - pad
    max_lon = max(xs) + pad
    min_lat = min(ys) - pad
    max_lat = max(ys) + pad
    if crs == "EPSG:4326":
        min_lon = max(-180.0, min_lon)
        max_lon = min(180.0, max_lon)
        min_lat = max(-90.0, min_lat)
        max_lat = min(90.0, max_lat)
    return BoundingBox(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        crs=crs,
    )
