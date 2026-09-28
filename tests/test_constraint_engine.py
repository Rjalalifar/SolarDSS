"""Hard/soft constraint engine: explicit thresholds, never universal cutoffs."""

from __future__ import annotations

import inspect
from collections.abc import Mapping

import pytest
from pydantic import ValidationError

from solardss.constraints import (
    ROOFTOP_DEFAULT_EXAMPLE,
    UTILITY_SCALE_DEFAULT_EXAMPLE,
    AllowedCategoryConstraint,
    BooleanExclusionConstraint,
    ConstraintConfigError,
    ConstraintEffect,
    ConstraintEngine,
    ConstraintKind,
    ConstraintProfile,
    DisallowedCategoryConstraint,
    DistanceConstraint,
    MaximumConstraint,
    MinimumConstraint,
    OverlapConstraint,
    Threshold,
    example_profile_for,
)
from solardss.constraints.contracts import AnyConstraint, CandidateEvaluation
from solardss.constraints.engine import IndicatorValue
from solardss.core.methods import MethodId
from solardss.core.quantity import Quantity
from solardss.core.status import FeasibilityStatus
from solardss.engine.contracts import IndicatorOutcome, IndicatorResult, IndicatorSet
from solardss.exceptions import InvalidParameterError, UnitError, UnknownIndicatorError
from solardss.indicators.contracts import ProjectType
from solardss.indicators.ids import IndicatorId

WATER = IndicatorId.LAND_WATER_FRACTION.value
OVERLAP = IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value
SLOPE = IndicatorId.TERRAIN_SLOPE_MEAN.value
AREA = IndicatorId.LAND_AVAILABLE_AREA.value
GRID = IndicatorId.INFRA_DISTANCE_TO_POWER_LINE.value
COVER = IndicatorId.LAND_COVER.value
ROOF_SLOPE = IndicatorId.ROOF_SLOPE.value
FOOTPRINT = IndicatorId.ROOF_BUILDING_FOOTPRINT_AREA.value
FLOOD = IndicatorId.HAZARD_FLOOD_RISK.value

_SOURCE = "test.profile"
_RATIONALE = "Caller-supplied bound for tests. Not a scientific default."


def _threshold(value: float, unit: str, profile: str = "test") -> Threshold:
    return Threshold(
        value=value,
        unit=unit,
        source=_SOURCE,
        profile=profile,
        rationale=_RATIONALE,
    )


def _profile(*constraints: AnyConstraint, profile_id: str = "test") -> ConstraintProfile:
    return ConstraintProfile(
        profile_id=profile_id,
        label="Test profile",
        constraints=tuple(constraints),
        notes=("Test configuration. Not a universal standard.",),
    )


def _max(
    indicator: str,
    value: float,
    unit: str,
    *,
    constraint_id: str = "max",
    effect: ConstraintEffect = ConstraintEffect.HARD_EXCLUSION,
    inclusive: bool = False,
    penalty_weight: float | None = None,
    profile: str = "test",
) -> MaximumConstraint:
    return MaximumConstraint(
        constraint_id=constraint_id,
        indicator_id=indicator,
        effect=effect,
        source=_SOURCE,
        profile=profile,
        rationale=_RATIONALE,
        penalty_weight=penalty_weight,
        inclusive=inclusive,
        threshold=_threshold(value, unit, profile),
    )


def _min(
    indicator: str,
    value: float,
    unit: str,
    *,
    constraint_id: str = "min",
    effect: ConstraintEffect = ConstraintEffect.HARD_EXCLUSION,
    inclusive: bool = False,
    profile: str = "test",
) -> MinimumConstraint:
    return MinimumConstraint(
        constraint_id=constraint_id,
        indicator_id=indicator,
        effect=effect,
        source=_SOURCE,
        profile=profile,
        rationale=_RATIONALE,
        inclusive=inclusive,
        threshold=_threshold(value, unit, profile),
    )


def _overlap(
    value: float,
    *,
    constraint_id: str = "overlap",
    effect: ConstraintEffect = ConstraintEffect.HARD_EXCLUSION,
    profile: str = "test",
) -> OverlapConstraint:
    return OverlapConstraint(
        constraint_id=constraint_id,
        indicator_id=OVERLAP,
        effect=effect,
        source=_SOURCE,
        profile=profile,
        rationale=_RATIONALE,
        penalty_weight=1.0 if effect is ConstraintEffect.PENALTY else None,
        threshold=_threshold(value, "1", profile),
    )


def _distance(
    *,
    maximum: float | None = None,
    minimum: float | None = None,
    effect: ConstraintEffect = ConstraintEffect.HARD_EXCLUSION,
    penalty_weight: float | None = None,
    profile: str = "test",
    constraint_id: str = "grid",
) -> DistanceConstraint:
    return DistanceConstraint(
        constraint_id=constraint_id,
        indicator_id=GRID,
        effect=effect,
        source=_SOURCE,
        profile=profile,
        rationale=_RATIONALE,
        penalty_weight=penalty_weight,
        maximum=None if maximum is None else _threshold(maximum, "m", profile),
        minimum=None if minimum is None else _threshold(minimum, "m", profile),
    )


def _evaluate(
    indicators: Mapping[str, IndicatorValue], *constraints: AnyConstraint
) -> CandidateEvaluation:
    return ConstraintEngine().evaluate("c1", indicators, _profile(*constraints))


def test_threshold_requires_value_unit_source_profile_rationale() -> None:
    with pytest.raises(ValidationError):
        Threshold(value=0.1, unit="1", source="s", profile="p")  # type: ignore[call-arg]


def test_water_fraction_above_threshold_is_exclusion() -> None:
    result = _evaluate(
        {WATER: Quantity(value=0.2, unit="1")},
        _max(WATER, 0.1, "1", constraint_id="water"),
    )
    assert result.feasible is False
    assert result.status is FeasibilityStatus.EXCLUDED
    assert result.failed_constraints[0].constraint_id == "water"
    assert result.failed_constraints[0].kind is ConstraintKind.MAXIMUM
    assert result.failed_constraints[0].effect is ConstraintEffect.HARD_EXCLUSION
    assert result.failed_constraints[0].source == _SOURCE
    assert result.failed_constraints[0].rationale == _RATIONALE
    assert result.indicators[0].indicator_id == WATER
    assert result.provenance.stamp.method_id == MethodId.CONSTRAINT_ENGINE.value


def test_equal_water_fraction_does_not_fail_exclusive_maximum() -> None:
    result = _evaluate({WATER: 0.1}, _max(WATER, 0.1, "1"))
    assert result.feasible is True
    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.failed_constraints == ()


def test_inclusive_maximum_excludes_equal_value() -> None:
    result = _evaluate({SLOPE: 15.0}, _max(SLOPE, 15.0, "deg", inclusive=True))
    assert result.feasible is False


def test_protected_overlap_exclusion_or_penalty_depends_on_profile() -> None:
    values = {OVERLAP: Quantity(value=0.05, unit="1")}
    hard = _evaluate(values, _overlap(0.0, effect=ConstraintEffect.HARD_EXCLUSION))
    soft = _evaluate(values, _overlap(0.0, effect=ConstraintEffect.PENALTY))
    assert hard.feasible is False
    assert hard.status is FeasibilityStatus.EXCLUDED
    assert soft.feasible is True
    assert soft.status is FeasibilityStatus.FEASIBLE
    assert soft.failed_constraints[0].effect is ConstraintEffect.PENALTY
    assert soft.failed_constraints[0].penalty_weight == 1.0
    assert any("overlap" in item for item in soft.warnings)


def test_slope_above_threshold_is_exclusion() -> None:
    result = _evaluate({SLOPE: Quantity(value=18.0, unit="deg")}, _max(SLOPE, 15.0, "deg"))
    assert result.feasible is False
    assert result.failed_constraints[0].observed == 18.0
    assert result.failed_constraints[0].threshold_value == 15.0
    assert result.failed_constraints[0].threshold_unit == "deg"


def test_available_area_below_required_is_exclusion() -> None:
    result = _evaluate({AREA: 4000.0}, _min(AREA, 5000.0, "m2", constraint_id="area"))
    assert result.feasible is False
    assert result.failed_constraints[0].kind is ConstraintKind.MINIMUM
    equal = _evaluate({AREA: 5000.0}, _min(AREA, 5000.0, "m2"))
    assert equal.feasible is True


def test_grid_distance_beyond_maximum_exclusion_or_penalty() -> None:
    far = {GRID: Quantity(value=8000.0, unit="m")}
    hard = _evaluate(far, _distance(maximum=5000.0))
    soft = _evaluate(
        far,
        _distance(maximum=5000.0, effect=ConstraintEffect.PENALTY, penalty_weight=0.5),
    )
    near = _evaluate({GRID: 1000.0}, _distance(maximum=5000.0))
    assert hard.feasible is False
    assert soft.feasible is True
    assert soft.failed_constraints[0].kind is ConstraintKind.DISTANCE
    assert near.feasible is True


def test_distance_minimum_excludes_too_close() -> None:
    result = _evaluate({GRID: 50.0}, _distance(minimum=100.0, maximum=10_000.0))
    assert result.feasible is False


def test_boolean_exclusion() -> None:
    constraint = BooleanExclusionConstraint(
        constraint_id="flood",
        indicator_id=FLOOD,
        effect=ConstraintEffect.HARD_EXCLUSION,
        source=_SOURCE,
        profile="test",
        rationale=_RATIONALE,
        true_means_excluded=True,
    )
    excluded = _evaluate({FLOOD: True}, constraint)
    allowed = _evaluate({FLOOD: False}, constraint)
    assert excluded.feasible is False
    assert allowed.feasible is True


def test_allowed_and_disallowed_categories() -> None:
    allowed = AllowedCategoryConstraint(
        constraint_id="cover_ok",
        indicator_id=COVER,
        effect=ConstraintEffect.HARD_EXCLUSION,
        source=_SOURCE,
        profile="test",
        rationale=_RATIONALE,
        allowed=("grassland", "bare"),
    )
    blocked = DisallowedCategoryConstraint(
        constraint_id="no_water",
        indicator_id=COVER,
        effect=ConstraintEffect.HARD_EXCLUSION,
        source=_SOURCE,
        profile="test",
        rationale=_RATIONALE,
        disallowed=("water",),
    )
    ok = _evaluate({COVER: "Grassland"}, allowed, blocked)
    wet = _evaluate({COVER: "water"}, allowed, blocked)
    other = _evaluate({COVER: "forest"}, allowed, blocked)
    assert ok.feasible is True
    assert wet.feasible is False
    assert {item.kind for item in wet.failed_constraints} >= {
        ConstraintKind.ALLOWED_CATEGORY,
        ConstraintKind.DISALLOWED_CATEGORY,
    }
    assert other.feasible is False
    assert other.failed_constraints[0].kind is ConstraintKind.ALLOWED_CATEGORY


def test_missing_hard_indicator_is_unknown_not_zero() -> None:
    result = _evaluate({}, _max(WATER, 0.1, "1", constraint_id="water"))
    assert result.feasible is False
    assert result.status is FeasibilityStatus.UNKNOWN
    assert result.failed_constraints == ()
    assert result.incomplete_constraints == ("water",)
    assert result.indicators[0].available is False
    assert any("zero is not substituted" in item for item in result.warnings)
    observed_zero = _evaluate({WATER: 0.0}, _max(WATER, 0.1, "1"))
    assert observed_zero.feasible is True


def test_missing_soft_constraint_does_not_exclude() -> None:
    result = _evaluate(
        {},
        _overlap(0.0, effect=ConstraintEffect.WARNING),
    )
    assert result.feasible is True
    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.incomplete_constraints == ("overlap",)


def test_indicator_set_missing_is_not_zero() -> None:
    bundle = IndicatorSet(
        requested=(WATER,),
        evaluation_order=(WATER,),
        results=(
            IndicatorResult(
                indicator_id=WATER,
                outcome=IndicatorOutcome.MISSING,
                reason="provider returned no coverage",
            ),
        ),
    )
    result = ConstraintEngine().evaluate(
        "c1", bundle, _profile(_max(WATER, 0.1, "1", constraint_id="water"))
    )
    assert result.status is FeasibilityStatus.UNKNOWN
    assert result.feasible is False


def test_unit_mismatch_raises() -> None:
    with pytest.raises(ConstraintConfigError, match="does not match indicator unit"):
        _profile(_max(SLOPE, 15.0, "percent"))
    with pytest.raises(UnitError):
        ConstraintEngine().evaluate(
            "c1",
            {SLOPE: Quantity(value=10.0, unit="percent")},
            _profile(_max(SLOPE, 15.0, "deg")),
        )


def test_unknown_indicator_rejected() -> None:
    with pytest.raises(UnknownIndicatorError):
        _profile(_max("NOT_AN_INDICATOR", 1.0, "1"))


def test_penalty_requires_weight_and_profile_ids_must_match() -> None:
    with pytest.raises(ConstraintConfigError, match="penalty_weight"):
        _max(WATER, 0.1, "1", effect=ConstraintEffect.PENALTY)
    with pytest.raises(ConstraintConfigError, match="does not match"):
        MaximumConstraint(
            constraint_id="x",
            indicator_id=WATER,
            effect=ConstraintEffect.HARD_EXCLUSION,
            source=_SOURCE,
            profile="test",
            rationale=_RATIONALE,
            threshold=_threshold(0.1, "1", profile="other"),
        )
    with pytest.raises(ConstraintConfigError, match="duplicate constraint_id"):
        _profile(
            _max(WATER, 0.1, "1", constraint_id="dup"),
            _max(SLOPE, 15.0, "deg", constraint_id="dup"),
        )
    with pytest.raises(ConstraintConfigError, match="maximum and/or minimum"):
        _distance()


def test_empty_profile_is_feasible() -> None:
    result = ConstraintEngine().evaluate("c1", {WATER: 1.0}, _profile())
    assert result.feasible is True
    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.candidate_id == "c1"
    assert result.profile_id == "test"


def test_example_profiles_are_labelled_not_universal_and_not_auto_applied() -> None:
    for profile in (UTILITY_SCALE_DEFAULT_EXAMPLE, ROOFTOP_DEFAULT_EXAMPLE):
        assert profile.example is True
        assert "example" in " ".join(profile.notes).casefold()
        assert "universal" in " ".join(profile.notes).casefold()
        for item in profile.constraints:
            assert item.source == "solardss.example_profile"
            assert item.profile == profile.profile_id
    assert example_profile_for(ProjectType.UTILITY_SCALE) is UTILITY_SCALE_DEFAULT_EXAMPLE
    assert example_profile_for(ProjectType.ROOFTOP) is ROOFTOP_DEFAULT_EXAMPLE
    source = inspect.getsource(ConstraintEngine.evaluate)
    assert "UTILITY_SCALE_DEFAULT_EXAMPLE" not in source
    assert "example_profile_for" not in source
    signature = inspect.signature(ConstraintEngine.evaluate)
    assert "profile" in signature.parameters
    assert signature.parameters["profile"].default is inspect.Parameter.empty


def test_utility_example_profile_screens_the_documented_cases() -> None:
    engine = ConstraintEngine()
    passing = engine.evaluate(
        "ok",
        {
            WATER: 0.05,
            OVERLAP: 0.0,
            SLOPE: 8.0,
            AREA: 8000.0,
            GRID: 1000.0,
            COVER: "grassland",
        },
        UTILITY_SCALE_DEFAULT_EXAMPLE,
    )
    failing = engine.evaluate(
        "bad",
        {
            WATER: 0.4,
            OVERLAP: 0.2,
            SLOPE: 25.0,
            AREA: 100.0,
            GRID: 20_000.0,
            COVER: "water",
        },
        UTILITY_SCALE_DEFAULT_EXAMPLE,
    )
    assert passing.feasible is True
    assert passing.failed_constraints == ()
    assert failing.feasible is False
    hard_ids = {
        item.constraint_id
        for item in failing.failed_constraints
        if item.effect is ConstraintEffect.HARD_EXCLUSION
    }
    assert "example_max_water_fraction" in hard_ids
    assert "example_max_protected_overlap" in hard_ids
    assert "example_max_slope" in hard_ids
    assert "example_min_available_area" in hard_ids
    assert "example_disallow_water_cover" in hard_ids
    penalty_ids = {
        item.constraint_id
        for item in failing.failed_constraints
        if item.effect is ConstraintEffect.PENALTY
    }
    assert "example_max_grid_distance" in penalty_ids


def test_rooftop_example_profile() -> None:
    engine = ConstraintEngine()
    steep = engine.evaluate(
        "roof",
        {ROOF_SLOPE: 50.0, FOOTPRINT: 40.0, OVERLAP: 0.0},
        ROOFTOP_DEFAULT_EXAMPLE,
    )
    small = engine.evaluate(
        "tiny",
        {ROOF_SLOPE: 10.0, FOOTPRINT: 5.0, OVERLAP: 0.2},
        ROOFTOP_DEFAULT_EXAMPLE,
    )
    assert steep.feasible is False
    assert small.feasible is False
    warn = engine.evaluate(
        "warn",
        {ROOF_SLOPE: 10.0, FOOTPRINT: 40.0, OVERLAP: 0.2},
        ROOFTOP_DEFAULT_EXAMPLE,
    )
    assert warn.feasible is True
    assert warn.failed_constraints[0].effect is ConstraintEffect.WARNING


def test_evaluate_many_and_indicator_id_keys() -> None:
    profile = _profile(_max(WATER, 0.1, "1", constraint_id="water"))
    bundle = ConstraintEngine().evaluate_many(
        (
            ("a", {IndicatorId.LAND_WATER_FRACTION: 0.0}),
            ("b", {WATER: 0.9}),
        ),
        profile,
    )
    assert bundle.feasible_ids() == ("a",)
    assert bundle.excluded_ids() == ("b",)
    assert bundle.profile_id == "test"


def test_non_numeric_value_rejected() -> None:
    with pytest.raises(InvalidParameterError, match="non-numeric"):
        _evaluate({WATER: "wet"}, _max(WATER, 0.1, "1"))


def test_example_profile_notes_required() -> None:
    with pytest.raises(ConstraintConfigError, match="example profiles must state"):
        ConstraintProfile(
            profile_id="test",
            label="Unlabelled example",
            example=True,
            constraints=(),
            notes=("A profile without the required wording.",),
        )
