"""Weighted Sum Model ranking and Pareto non-dominated sets."""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from pydantic import ValidationError

from solardss.core.methods import MethodId
from solardss.core.quantity import Quantity
from solardss.core.status import FeasibilityStatus
from solardss.indicators.contracts import OptimizationDirection
from solardss.indicators.ids import IndicatorId
from solardss.suitability import (
    BALANCED,
    ENERGY_FOCUSED,
    ENVIRONMENT_FOCUSED,
    INFRASTRUCTURE_FOCUSED,
    CriterionWeight,
    DecisionProfile,
    DecisionProfileError,
    MultiObjectiveResult,
    NormalizationKind,
    ObjectiveSpec,
    OmissionReason,
    SuitabilityEngine,
    normalize_criterion,
    pareto_candidates,
    rank_candidates,
)
from solardss.suitability.ranking import SuitabilityEngine as RankingEngineClass

GHI = IndicatorId.SOLAR_GHI_ANNUAL.value
SLOPE = IndicatorId.TERRAIN_SLOPE_MEAN.value
GRID = IndicatorId.INFRA_DISTANCE_TO_POWER_LINE.value
ROAD = IndicatorId.INFRA_DISTANCE_TO_ROAD.value
AREA = IndicatorId.LAND_AVAILABLE_AREA.value
OVERLAP = IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value
WATER = IndicatorId.LAND_WATER_FRACTION.value
FOREST = IndicatorId.LAND_FOREST_FRACTION.value
TILT = IndicatorId.PV_TILT.value

_SOURCE = "test.profile"
_RATIONALE = "Caller-supplied test weight. Not universal truth."


def _weight(
    indicator_id: str,
    weight: float,
    unit: str,
    kind: NormalizationKind,
    *,
    profile: str = "test",
    target: float | None = None,
) -> CriterionWeight:
    return CriterionWeight(
        indicator_id=indicator_id,
        weight=weight,
        unit=unit,
        source=_SOURCE,
        profile=profile,
        rationale=_RATIONALE,
        normalization=kind,
        target_value=target,
    )


def _profile(*criteria: CriterionWeight, profile_id: str = "test") -> DecisionProfile:
    return DecisionProfile(
        profile_id=profile_id,
        label="Test decision profile",
        criteria=criteria,
        notes=("Test configuration. Not universal truth.",),
    )


def _site(
    ghi: float,
    slope: float,
    grid: float,
    *,
    area: float = 10_000.0,
    road: float = 500.0,
    overlap: float = 0.0,
    water: float = 0.0,
    forest: float = 0.0,
) -> dict[str, float]:
    return {
        GHI: ghi,
        SLOPE: slope,
        GRID: grid,
        AREA: area,
        ROAD: road,
        OVERLAP: overlap,
        WATER: water,
        FOREST: forest,
    }


def test_benefit_cost_and_equal_range_normalization() -> None:
    benefit, stats = normalize_criterion(
        np.array([10.0, 20.0, 30.0]), kind=NormalizationKind.BENEFIT
    )
    cost, _ = normalize_criterion(np.array([10.0, 20.0, 30.0]), kind=NormalizationKind.COST)
    np.testing.assert_allclose(benefit.values, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(cost.values, [1.0, 0.5, 0.0])
    assert stats.equal_values is False
    equal, equal_stats = normalize_criterion(
        np.array([7.0, 7.0, 7.0]), kind=NormalizationKind.BENEFIT
    )
    np.testing.assert_allclose(equal.values, [1.0, 1.0, 1.0])
    assert equal_stats.equal_values is True
    assert equal_stats.range_source == "degenerate_equal"


def test_target_distance_normalization() -> None:
    scores, stats = normalize_criterion(
        np.array([20.0, 30.0, 40.0]),
        kind=NormalizationKind.TARGET,
        target=30.0,
    )
    np.testing.assert_allclose(scores.values, [0.0, 1.0, 0.0])
    assert stats.target_value == 30.0
    hit, hit_stats = normalize_criterion(
        np.array([30.0, 30.0]), kind=NormalizationKind.TARGET, target=30.0
    )
    np.testing.assert_allclose(hit.values, [1.0, 1.0])
    assert hit_stats.equal_values is True


def test_deterministic_ranking_and_full_trace() -> None:
    profile = _profile(
        _weight(GHI, 0.6, "kWh/m2/year", NormalizationKind.BENEFIT),
        _weight(SLOPE, 0.4, "deg", NormalizationKind.COST),
    )
    rows = (
        ("b", {GHI: 1600.0, SLOPE: 20.0}),
        ("a", {GHI: 2000.0, SLOPE: 5.0}),
    )
    first = rank_candidates(rows, profile)
    second = rank_candidates(tuple(reversed(rows)), profile)
    assert first.ranked_ids() == ("a", "b")
    assert second.ranked_ids() == ("a", "b")
    assert first.ranked[0].rank == 1
    assert first.ranked[1].rank == 2
    winner = first.ranked[0]
    assert winner.candidate_id == "a"
    assert winner.constraint_status is FeasibilityStatus.FEASIBLE
    assert {item.indicator_id for item in winner.contributions} == {GHI, SLOPE}
    ghi = next(item for item in winner.contributions if item.indicator_id == GHI)
    assert ghi.raw_value == 2000.0
    assert ghi.normalized_value == pytest.approx(1.0)
    assert ghi.weight == 0.6
    assert ghi.weighted_contribution == pytest.approx(0.6)
    assert ghi.source == _SOURCE
    assert ghi.rationale == _RATIONALE
    assert winner.score == pytest.approx(
        sum(item.weighted_contribution for item in winner.contributions)
    )
    assert first.provenance.stamp.method_id == MethodId.WEIGHTED_SUM_RANKING.value
    assert first.method == "weighted_sum_model"


def test_tie_breaks_on_candidate_id() -> None:
    profile = _profile(_weight(GHI, 1.0, "kWh/m2/year", NormalizationKind.BENEFIT))
    result = rank_candidates((("z", {GHI: 1500.0}), ("a", {GHI: 1500.0})), profile)
    assert result.ranked_ids() == ("a", "z")
    assert result.ranked[0].score == result.ranked[1].score


def test_excluded_candidates_omitted() -> None:
    profile = _profile(_weight(GHI, 1.0, "kWh/m2/year", NormalizationKind.BENEFIT))
    result = rank_candidates(
        (("keep", {GHI: 1000.0}), ("drop", {GHI: 2000.0})),
        profile,
        constraint_status={
            "keep": FeasibilityStatus.FEASIBLE,
            "drop": FeasibilityStatus.EXCLUDED,
        },
    )
    assert result.ranked_ids() == ("keep",)
    assert result.omitted[0].candidate_id == "drop"
    assert result.omitted[0].reason is OmissionReason.EXCLUDED
    assert result.ranked[0].constraint_status is FeasibilityStatus.FEASIBLE


def test_missing_indicator_is_not_zero() -> None:
    profile = _profile(
        _weight(GHI, 0.5, "kWh/m2/year", NormalizationKind.BENEFIT),
        _weight(SLOPE, 0.5, "deg", NormalizationKind.COST),
    )
    result = rank_candidates(
        (("full", {GHI: 1800.0, SLOPE: 8.0}), ("gap", {GHI: 2000.0})),
        profile,
    )
    assert result.ranked_ids() == ("full",)
    assert result.omitted[0].reason is OmissionReason.INCOMPLETE_INDICATORS
    assert result.omitted[0].missing_indicators == (SLOPE,)


def test_profile_change_can_change_rank() -> None:
    sunny_far = _site(ghi=2000.0, slope=6.0, grid=8000.0, road=4000.0)
    dim_near = _site(ghi=1400.0, slope=6.0, grid=80.0, road=40.0)
    rows = (("sunny", sunny_far), ("near", dim_near))
    energy = rank_candidates(rows, ENERGY_FOCUSED)
    infra = rank_candidates(rows, INFRASTRUCTURE_FOCUSED)
    assert energy.ranked_ids()[0] == "sunny"
    assert infra.ranked_ids()[0] == "near"


def test_example_profiles_are_labelled_not_universal_and_not_auto_applied() -> None:
    for profile in (ENERGY_FOCUSED, INFRASTRUCTURE_FOCUSED, ENVIRONMENT_FOCUSED, BALANCED):
        assert profile.example is True
        joined = " ".join(profile.notes).casefold()
        assert "example" in joined
        assert "universal" in joined
        assert abs(sum(item.weight for item in profile.criteria) - 1.0) < 1e-12
        for item in profile.criteria:
            assert item.source == "solardss.example_profile"
            assert item.profile == profile.profile_id
            assert item.rationale
    source = inspect.getsource(RankingEngineClass.rank)
    assert "ENERGY_FOCUSED" not in source
    signature = inspect.signature(SuitabilityEngine.rank)
    assert signature.parameters["profile"].default is inspect.Parameter.empty


def test_weight_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        CriterionWeight(
            indicator_id=GHI,
            weight=1.0,
            unit="kWh/m2/year",
            normalization=NormalizationKind.BENEFIT,
        )  # type: ignore[call-arg]
    with pytest.raises(DecisionProfileError, match="sum to 1"):
        _profile(_weight(GHI, 0.4, "kWh/m2/year", NormalizationKind.BENEFIT))


def test_target_criterion_in_wsm() -> None:
    profile = _profile(
        _weight(TILT, 1.0, "deg", NormalizationKind.TARGET, target=30.0),
    )
    result = rank_candidates(
        (("on_target", {TILT: 30.0}), ("steep", {TILT: 50.0})),
        profile,
    )
    assert result.ranked_ids() == ("on_target", "steep")
    assert result.ranked[0].contributions[0].normalized_value == pytest.approx(1.0)
    assert result.ranked[0].contributions[0].raw_value == 30.0


def test_quantity_units_and_complete_traceability() -> None:
    profile = _profile(_weight(GRID, 1.0, "m", NormalizationKind.COST))
    result = rank_candidates(
        (
            ("near", {GRID: Quantity(value=100.0, unit="m")}),
            ("far", {GRID: Quantity(value=900.0, unit="m")}),
        ),
        profile,
    )
    winner = result.ranked[0]
    contrib = winner.contributions[0]
    assert winner.candidate_id == "near"
    assert contrib.raw_value == 100.0
    assert contrib.unit == "m"
    assert contrib.normalized_value == pytest.approx(1.0)
    assert contrib.weight == 1.0
    assert contrib.weighted_contribution == pytest.approx(1.0)
    assert winner.score == pytest.approx(1.0)
    assert winner.rank == 1
    assert winner.constraint_status is FeasibilityStatus.FEASIBLE
    assert contrib.source == _SOURCE
    assert contrib.profile == "test"


def test_pareto_uses_raw_objectives_and_does_not_scalarize() -> None:
    energy = ObjectiveSpec(
        indicator_id=GHI,
        direction=OptimizationDirection.MAXIMIZE,
        unit="kWh/m2/year",
        source=_SOURCE,
        rationale="Higher GHI is preferred in this test.",
    )
    distance = ObjectiveSpec(
        indicator_id=GRID,
        direction=OptimizationDirection.MINIMIZE,
        unit="m",
        source=_SOURCE,
        rationale="Shorter grid distance is preferred in this test.",
    )
    result = pareto_candidates(
        (
            ("a", {GHI: 2000.0, GRID: 400.0}),
            ("b", {GHI: 1500.0, GRID: 50.0}),
            ("c", {GHI: 1200.0, GRID: 800.0}),
            ("gap", {GRID: 10.0}),
        ),
        (energy, distance),
        constraint_status={
            "a": FeasibilityStatus.FEASIBLE,
            "b": FeasibilityStatus.FEASIBLE,
            "c": FeasibilityStatus.FEASIBLE,
            "gap": FeasibilityStatus.FEASIBLE,
        },
    )
    assert result.method == "pareto_nondominated"
    assert "score" not in MultiObjectiveResult.model_fields
    assert set(result.nondominated_ids()) == {"a", "b"}
    dominated = next(item for item in result.points if item.candidate_id == "c")
    assert dominated.nondominated is False
    assert set(dominated.dominated_by) >= {"a"}
    omitted = next(item for item in result.omitted if item.candidate_id == "gap")
    assert omitted.reason is OmissionReason.INCOMPLETE_INDICATORS
    assert result.provenance.parameters["scalarized"] is False
    for point in result.points:
        ghi_value = next(item.value for item in point.objectives if item.indicator_id == GHI)
        assert ghi_value in {2000.0, 1500.0, 1200.0}


def test_pareto_omits_excluded_and_engine_matches_function() -> None:
    energy = ObjectiveSpec(
        indicator_id=GHI,
        direction=OptimizationDirection.MAXIMIZE,
        unit="kWh/m2/year",
        source=_SOURCE,
        rationale="test",
    )
    rows = (("keep", {GHI: 1000.0}), ("drop", {GHI: 2000.0}))
    status = {"keep": FeasibilityStatus.FEASIBLE, "drop": FeasibilityStatus.EXCLUDED}
    engine = SuitabilityEngine()
    result = engine.pareto(rows, (energy,), constraint_status=status)
    assert result.nondominated_ids() == ("keep",)
    assert result.omitted[0].reason is OmissionReason.EXCLUDED
    assert not hasattr(result.points[0], "score")
