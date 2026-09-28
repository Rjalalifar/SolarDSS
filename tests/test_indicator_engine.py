"""Indicator engine: DAG, provider selection, cache, batch, and provenance."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import pytest

from solardss.core.quantity import Quantity
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.decision.problem import SolarDecisionProblem
from solardss.engine import (
    DEFAULT_GRAPH,
    CircularDependencyError,
    Dependency,
    DependencyKind,
    EngineContext,
    IndicatorEngine,
    IndicatorGraph,
    IndicatorNode,
    IndicatorOutcome,
    IndicatorSet,
    ResolutionPath,
    derived_observation,
)
from solardss.engine.contracts import DerivedInputs
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.indicators.ids import IndicatorId
from solardss.provenance.observation import ValueOrigin
from solardss.providers.clock import FixedClock
from solardss.providers.fake import (
    FakeLandCoverProvider,
    FakeSeed,
    FakeSolarResourceProvider,
    FakeTerrainProvider,
)
from solardss.providers.request import FetchRequest

T1 = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
BERLIN = GeographicRef(name="Berlin", longitude=13.4, latitude=52.52)
GHI = IndicatorId.SOLAR_GHI_ANNUAL.value
DNI = IndicatorId.SOLAR_DNI_ANNUAL.value
SLOPE = IndicatorId.TERRAIN_SLOPE_MEAN.value
ELEVATION = IndicatorId.TERRAIN_ELEVATION.value
FOREST = IndicatorId.LAND_FOREST_FRACTION.value
ROAD = IndicatorId.INFRA_DISTANCE_TO_ROAD.value
POA = IndicatorId.SOLAR_POA_IRRADIATION.value
TILT = IndicatorId.PV_TILT.value
AZIMUTH = IndicatorId.PV_AZIMUTH.value
MONTHLY = IndicatorId.SOLAR_MONTHLY_VARIABILITY.value
HOSTING = IndicatorId.INFRA_GRID_HOSTING_CAPACITY.value


class CountingSolar(FakeSolarResourceProvider):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.fetch_calls = 0
        self.prefetch_calls = 0

    def prefetch(self, request: FetchRequest) -> None:
        self.prefetch_calls += 1

    def fetch(self, request: FetchRequest):
        self.fetch_calls += 1
        return super().fetch(request)


class CountingLand(FakeLandCoverProvider):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self.fetch_calls = 0
        self.prefetch_calls = 0

    def prefetch(self, request: FetchRequest) -> None:
        self.prefetch_calls += 1

    def fetch(self, request: FetchRequest):
        self.fetch_calls += 1
        return super().fetch(request)


class GhiPresentIndexCalculator:
    """Test double: a variability index when GHI is available. Missing GHI stays missing."""

    indicator_id = MONTHLY
    method_id = "test_ghi_present_index"

    def required_inputs(self) -> tuple[str, ...]:
        return (GHI,)

    def can_compute(self, inputs: DerivedInputs, context: EngineContext) -> bool:
        value = inputs.available_value(GHI)
        return isinstance(value, int | float)

    def compute(self, inputs: DerivedInputs, **kwargs: object):
        ghi = inputs.available_value(GHI)
        assert ghi is not None
        return derived_observation(
            spec=kwargs["spec"],  # type: ignore[arg-type]
            value=0.2,
            unit="1",
            request=kwargs["request"],  # type: ignore[arg-type]
            retrieved_at=kwargs["retrieved_at"],  # type: ignore[arg-type]
            method_id=self.method_id,
            inputs={GHI: inputs.get(GHI)},  # type: ignore[dict-item]
            assumptions=("test index; not a climate model",),
        )


def _ghi_seed(value: float = 1085.0) -> FakeSeed:
    return FakeSeed(value=value, unit="kWh/m2/year")


def test_default_graph_declares_slope_and_road_and_poa_edges() -> None:
    slope_deps = DEFAULT_GRAPH.dependencies_of(SLOPE)
    targets = {(item.kind, item.target, item.required) for item in slope_deps}
    assert (DependencyKind.SOURCE_TYPE, "dem", True) in targets
    assert (DependencyKind.INDICATOR, ELEVATION, False) in targets
    road_deps = DEFAULT_GRAPH.dependencies_of(ROAD)
    assert any(
        item.kind is DependencyKind.SOURCE_TYPE and item.target == "osm_or_network_vector"
        for item in road_deps
    )
    poa_required = DEFAULT_GRAPH.indicator_dependencies(POA, required_only=True)
    assert TILT in poa_required
    assert AZIMUTH in poa_required
    order = DEFAULT_GRAPH.topological_order(DEFAULT_GRAPH.expand((SLOPE, GHI)))
    assert GHI in order
    assert SLOPE in order
    assert ELEVATION not in order


def test_circular_dependencies_are_rejected() -> None:
    with pytest.raises(CircularDependencyError, match="circular") as exc:
        IndicatorGraph(
            (
                IndicatorNode(
                    indicator_id="A",
                    dependencies=(Dependency(kind=DependencyKind.INDICATOR, target="B"),),
                ),
                IndicatorNode(
                    indicator_id="B",
                    dependencies=(Dependency(kind=DependencyKind.INDICATOR, target="A"),),
                ),
            )
        )
    assert "A" in exc.value.cycle
    assert "B" in exc.value.cycle


def test_expand_includes_required_edges_not_optional_or_source_types() -> None:
    expanded = DEFAULT_GRAPH.expand((POA,))
    assert TILT in expanded
    assert AZIMUTH in expanded
    assert GHI not in expanded
    with_extra = DEFAULT_GRAPH.expand((POA,), extra_indicator_deps={POA: (GHI,)})
    assert GHI in with_extra


def test_context_value_skips_provider_fetch() -> None:
    solar = CountingSolar(seeds={GHI: _ghi_seed()}, clock=FixedClock(T1))
    engine = IndicatorEngine([solar], clock=FixedClock(T1))
    result = engine.compute(
        geometry=BERLIN,
        indicators=[GHI],
        context=EngineContext(values={GHI: 1100.0}),
    )
    item = result.get(GHI)
    assert item is not None
    assert item.outcome is IndicatorOutcome.AVAILABLE
    assert item.value == 1100.0
    assert item.provider_id == "caller"
    assert solar.fetch_calls == 0


def test_provider_selection_skips_unwilling_adapters() -> None:
    dni_only = FakeSolarResourceProvider(
        provider_id="fake.solar.dni",
        served_indicators=frozenset({DNI}),
        seeds={DNI: FakeSeed(value=1400.0, unit="kWh/m2/year")},
        clock=FixedClock(T1),
    )
    ghi_provider = FakeSolarResourceProvider(
        provider_id="fake.solar.ghi",
        seeds={GHI: _ghi_seed()},
        clock=FixedClock(T1),
    )
    engine = IndicatorEngine([dni_only, ghi_provider], clock=FixedClock(T1))
    result = engine.compute(geometry=BERLIN, indicators=[GHI])
    item = result.get(GHI)
    assert item is not None
    assert item.outcome is IndicatorOutcome.AVAILABLE
    assert item.provider_id == "fake.solar.ghi"
    assert item.value == 1085.0


def test_engine_cache_avoids_repeat_fetches() -> None:
    solar = CountingSolar(seeds={GHI: _ghi_seed()}, clock=FixedClock(T1))
    engine = IndicatorEngine([solar], clock=FixedClock(T1))
    first = engine.compute(geometry=BERLIN, indicators=[GHI])
    second = engine.compute(geometry=BERLIN, indicators=[GHI])
    first_ghi = first.get(GHI)
    second_ghi = second.get(GHI)
    assert first_ghi is not None
    assert second_ghi is not None
    assert first_ghi.value == second_ghi.value
    assert solar.fetch_calls == 1


def test_missing_catalog_value_is_not_zero() -> None:
    solar = FakeSolarResourceProvider(seeds={DNI: FakeSeed(value=1400.0, unit="kWh/m2/year")})
    engine = IndicatorEngine([solar], clock=FixedClock(T1))
    result = engine.compute(geometry=BERLIN, indicators=[GHI])
    item = result.get(GHI)
    assert item is not None
    assert item.outcome is IndicatorOutcome.MISSING
    assert item.value is None
    assert item.observation is None


def test_coverage_hole_is_missing() -> None:
    solar = FakeSolarResourceProvider(
        seeds={GHI: _ghi_seed()},
        coverage_holes=(BoundingBox(min_lon=13.0, min_lat=52.0, max_lon=14.0, max_lat=53.0),),
        clock=FixedClock(T1),
    )
    engine = IndicatorEngine([solar], clock=FixedClock(T1))
    item = engine.compute(geometry=BERLIN, indicators=[GHI]).get(GHI)
    assert item is not None
    assert item.outcome is IndicatorOutcome.MISSING
    assert item.value is None


def test_unregistered_provider_is_unavailable_and_hosting_is_external() -> None:
    engine = IndicatorEngine([], clock=FixedClock(T1))
    result = engine.compute(geometry=BERLIN, indicators=[GHI, HOSTING])
    assert result.outcome_of(GHI) is IndicatorOutcome.PROVIDER_UNAVAILABLE
    assert result.outcome_of(HOSTING) is IndicatorOutcome.EXTERNALLY_REQUIRED
    ghi_item = result.get(GHI)
    hosting_item = result.get(HOSTING)
    assert ghi_item is not None
    assert hosting_item is not None
    assert ghi_item.value is None
    assert hosting_item.value is None


def test_derived_provider_value_keeps_method_provenance() -> None:
    terrain = FakeTerrainProvider(
        seeds={
            SLOPE: FakeSeed(
                value=4.5,
                unit="deg",
                value_kind=ValueProvenanceKind.DERIVED_METHOD,
                value_origin=ValueOrigin.DERIVED,
                calculation_method="horn_1981_slope_aspect",
            )
        },
        clock=FixedClock(T1),
    )
    engine = IndicatorEngine([terrain], clock=FixedClock(T1))
    item = engine.compute(geometry=BERLIN, indicators=[SLOPE]).get(SLOPE)
    assert item is not None
    assert item.outcome is IndicatorOutcome.AVAILABLE
    assert item.value == 4.5
    assert item.observation is not None
    assert item.observation.provenance.value_origin is ValueOrigin.DERIVED
    assert item.observation.provenance.calculation_method == "horn_1981_slope_aspect"


def test_engine_calculator_propagates_parent_fingerprints() -> None:
    solar = FakeSolarResourceProvider(seeds={GHI: _ghi_seed()}, clock=FixedClock(T1))
    graph = DEFAULT_GRAPH.overlay(
        (
            IndicatorNode(
                indicator_id=MONTHLY,
                dependencies=(
                    Dependency(kind=DependencyKind.INDICATOR, target=GHI, required=True),
                ),
                resolution=ResolutionPath.DERIVED,
                implemented=True,
            ),
        )
    )
    engine = IndicatorEngine(
        [solar],
        graph=graph,
        calculators=(GhiPresentIndexCalculator(),),
        clock=FixedClock(T1),
    )
    result = engine.compute(geometry=BERLIN, indicators=[MONTHLY])
    monthly = result.get(MONTHLY)
    ghi = result.get(GHI)
    assert ghi is not None and ghi.outcome is IndicatorOutcome.AVAILABLE
    assert monthly is not None
    assert monthly.outcome is IndicatorOutcome.AVAILABLE
    assert monthly.value == 0.2
    assert monthly.observation is not None
    assert monthly.observation.provenance.calculation_method == "test_ghi_present_index"
    assert monthly.observation.provenance.computation is not None
    fingerprints = monthly.observation.provenance.computation.input_fingerprints
    assert ghi.observation is not None
    assert ghi.observation.provenance.request_hash in fingerprints


def test_calculator_does_not_treat_missing_ghi_as_zero() -> None:
    solar = FakeSolarResourceProvider(seeds={DNI: FakeSeed(value=1200.0, unit="kWh/m2/year")})
    graph = DEFAULT_GRAPH.overlay(
        (
            IndicatorNode(
                indicator_id=MONTHLY,
                dependencies=(
                    Dependency(kind=DependencyKind.INDICATOR, target=GHI, required=True),
                ),
                resolution=ResolutionPath.DERIVED,
                implemented=True,
            ),
        )
    )
    engine = IndicatorEngine(
        [solar],
        graph=graph,
        calculators=(GhiPresentIndexCalculator(),),
        clock=FixedClock(T1),
    )
    result = engine.compute(geometry=BERLIN, indicators=[MONTHLY])
    monthly = result.get(MONTHLY)
    assert monthly is not None
    assert monthly.outcome is IndicatorOutcome.MISSING
    assert monthly.value is None
    assert GHI in monthly.missing_dependencies


def test_poa_without_orientation_is_externally_required() -> None:
    solar = FakeSolarResourceProvider(seeds={GHI: _ghi_seed()}, clock=FixedClock(T1))
    engine = IndicatorEngine([solar], clock=FixedClock(T1))
    result = engine.compute(geometry=BERLIN, indicators=[POA])
    poa = result.get(POA)
    assert poa is not None
    assert poa.outcome is IndicatorOutcome.EXTERNALLY_REQUIRED
    assert TILT in poa.missing_dependencies
    assert poa.value is None


def test_poa_with_orientation_but_no_method_is_unsupported() -> None:
    solar = FakeSolarResourceProvider(seeds={GHI: _ghi_seed()}, clock=FixedClock(T1))
    engine = IndicatorEngine([solar], clock=FixedClock(T1))
    result = engine.compute(
        geometry=BERLIN,
        indicators=[POA],
        context=EngineContext(values={TILT: Quantity(value=30.0, unit="deg"), AZIMUTH: 180.0}),
    )
    poa = result.get(POA)
    assert poa is not None
    assert poa.outcome is IndicatorOutcome.UNSUPPORTED
    assert poa.value is None
    assert result.outcome_of(TILT) is IndicatorOutcome.AVAILABLE


def test_context_unit_mismatch_is_missing() -> None:
    engine = IndicatorEngine([], clock=FixedClock(T1))
    result = engine.compute(
        geometry=BERLIN,
        indicators=[TILT],
        context=EngineContext(values={TILT: Quantity(value=30.0, unit="rad")}),
    )
    item = result.get(TILT)
    assert item is not None
    assert item.outcome is IndicatorOutcome.MISSING
    assert "Expected unit 'deg'" in item.reason


def test_provider_unit_mismatch_is_missing() -> None:
    solar = FakeSolarResourceProvider(
        seeds={GHI: FakeSeed(value=1085.0, unit="W/m2")},
        clock=FixedClock(T1),
    )
    engine = IndicatorEngine([solar], clock=FixedClock(T1))
    item = engine.compute(geometry=BERLIN, indicators=[GHI]).get(GHI)
    assert item is not None
    assert item.outcome is IndicatorOutcome.MISSING
    assert item.value is None
    assert "unit mismatch" in item.reason


def test_batch_shared_aoi_fetches_once_and_points_dedupe() -> None:
    land = CountingLand(
        seeds={FOREST: FakeSeed(value=0.4, unit="1")},
        clock=FixedClock(T1),
    )
    solar = CountingSolar(seeds={GHI: _ghi_seed()}, clock=FixedClock(T1))
    engine = IndicatorEngine([land, solar], clock=FixedClock(T1))
    table = pd.DataFrame(
        {
            "candidate_id": ["a", "b", "c"],
            "longitude": [13.4, 13.41, 13.4],
            "latitude": [52.52, 52.53, 52.52],
        }
    )
    batch = engine.compute_for_candidates(table, indicators=[FOREST, GHI])
    assert land.prefetch_calls == 1
    assert land.fetch_calls == 1
    assert solar.prefetch_calls == 1
    assert solar.fetch_calls == 2
    forest_values = list(batch.table.frame[FOREST])
    assert forest_values == [0.4, 0.4, 0.4]
    ghi_values = list(batch.table.frame[GHI])
    assert ghi_values[0] == 1085.0
    assert ghi_values[2] == 1085.0
    site_a = batch.for_candidate("a")
    assert site_a is not None
    assert site_a.outcome_of(FOREST) is IndicatorOutcome.AVAILABLE


def test_batch_missing_stays_na_not_zero() -> None:
    solar = FakeSolarResourceProvider(seeds={DNI: FakeSeed(value=1000.0, unit="kWh/m2/year")})
    engine = IndicatorEngine([solar], clock=FixedClock(T1))
    table = pd.DataFrame({"candidate_id": ["a"], "longitude": [13.4], "latitude": [52.52]})
    batch = engine.compute_for_candidates(table, indicators=[GHI])
    value = batch.table.frame.loc[0, GHI]
    assert pd.isna(value)
    assert batch.sets[0].outcome_of(GHI) is IndicatorOutcome.MISSING


def test_compute_for_problem_uses_required_indicators() -> None:
    solar = FakeSolarResourceProvider(seeds={GHI: _ghi_seed()}, clock=FixedClock(T1))
    engine = IndicatorEngine([solar], clock=FixedClock(T1))
    problem = SolarDecisionProblem(
        mode="fixed_site",
        project_type="residential_rooftop",
        location=BERLIN,
        target_capacity_kw=5.0,
    )
    result = engine.compute_for_problem(
        problem,
        context=EngineContext(values={IndicatorId.PV_SYSTEM_CAPACITY.value: 5.0}),
    )
    assert isinstance(result, IndicatorSet)
    assert GHI in result.requested
    assert result.outcome_of(GHI) is IndicatorOutcome.AVAILABLE
    assert result.outcome_of(IndicatorId.PV_LOCATION.value) is IndicatorOutcome.AVAILABLE
    assert result.outcome_of(IndicatorId.PV_SYSTEM_CAPACITY.value) is IndicatorOutcome.AVAILABLE
