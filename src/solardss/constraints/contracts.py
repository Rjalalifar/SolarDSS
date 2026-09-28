"""Explicit hard/soft constraint configuration.

Thresholds are caller- or profile-supplied. None of them is a universal
scientific cutoff.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solardss.constraints.errors import ConstraintConfigError
from solardss.core.status import FeasibilityStatus
from solardss.indicators.catalog import registry as default_registry
from solardss.indicators.contracts import IndicatorSpec
from solardss.indicators.registry import IndicatorRegistry
from solardss.provenance.record import ProvenanceRecord


class ConstraintKind(str, Enum):
    """Configured comparison. The kind does not imply a default threshold."""

    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    BOOLEAN_EXCLUSION = "boolean_exclusion"
    ALLOWED_CATEGORY = "allowed_category"
    DISALLOWED_CATEGORY = "disallowed_category"
    OVERLAP = "overlap"
    DISTANCE = "distance"


class ConstraintEffect(str, Enum):
    """What happens when the configured test fires. Soft effects do not exclude."""

    HARD_EXCLUSION = "hard_exclusion"
    PENALTY = "penalty"
    WARNING = "warning"


class Threshold(BaseModel):
    """One numeric bound. Value, unit, source, profile, and rationale are required."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float
    unit: str = Field(min_length=1)
    source: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def _finite(self) -> Threshold:
        if self.value != self.value or self.value in (float("inf"), float("-inf")):
            raise ConstraintConfigError("threshold value must be finite")
        return self


class _ConstraintFields(BaseModel):
    """Shared identity and provenance for every constraint object."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    constraint_id: str = Field(min_length=1)
    indicator_id: str = Field(min_length=1)
    effect: ConstraintEffect
    source: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    penalty_weight: float | None = None
    inclusive: bool = False

    @model_validator(mode="after")
    def _effect_weight(self) -> _ConstraintFields:
        if self.effect is ConstraintEffect.PENALTY:
            if self.penalty_weight is None:
                raise ConstraintConfigError("penalty constraints require penalty_weight")
            weight = self.penalty_weight
            if weight != weight or weight in (float("inf"), float("-inf")) or weight < 0.0:
                raise ConstraintConfigError("penalty_weight must be a non-negative finite value")
        elif self.penalty_weight is not None:
            raise ConstraintConfigError("penalty_weight is only valid when effect is penalty")
        return self


class MaximumConstraint(_ConstraintFields):
    """Fail when the indicator is above the configured maximum."""

    kind: Literal[ConstraintKind.MAXIMUM] = ConstraintKind.MAXIMUM
    threshold: Threshold

    @model_validator(mode="after")
    def _threshold_profile(self) -> MaximumConstraint:
        _align_threshold(self.threshold, self)
        return self


class MinimumConstraint(_ConstraintFields):
    """Fail when the indicator is below the configured minimum."""

    kind: Literal[ConstraintKind.MINIMUM] = ConstraintKind.MINIMUM
    threshold: Threshold

    @model_validator(mode="after")
    def _threshold_profile(self) -> MinimumConstraint:
        _align_threshold(self.threshold, self)
        return self


class BooleanExclusionConstraint(_ConstraintFields):
    """Fail when a boolean indicator matches the configured excluded polarity."""

    kind: Literal[ConstraintKind.BOOLEAN_EXCLUSION] = ConstraintKind.BOOLEAN_EXCLUSION
    true_means_excluded: bool = True


class AllowedCategoryConstraint(_ConstraintFields):
    """Fail when a categorical indicator is not in the allowed set."""

    kind: Literal[ConstraintKind.ALLOWED_CATEGORY] = ConstraintKind.ALLOWED_CATEGORY
    allowed: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _nonempty_tokens(self) -> AllowedCategoryConstraint:
        _reject_blank_categories(self.allowed)
        return self


class DisallowedCategoryConstraint(_ConstraintFields):
    """Fail when a categorical indicator is in the disallowed set."""

    kind: Literal[ConstraintKind.DISALLOWED_CATEGORY] = ConstraintKind.DISALLOWED_CATEGORY
    disallowed: tuple[str, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _nonempty_tokens(self) -> DisallowedCategoryConstraint:
        _reject_blank_categories(self.disallowed)
        return self


class OverlapConstraint(_ConstraintFields):
    """Fail when an overlap fraction/area exceeds the configured maximum."""

    kind: Literal[ConstraintKind.OVERLAP] = ConstraintKind.OVERLAP
    threshold: Threshold

    @model_validator(mode="after")
    def _threshold_profile(self) -> OverlapConstraint:
        _align_threshold(self.threshold, self)
        return self


class DistanceConstraint(_ConstraintFields):
    """Fail when a distance is beyond a maximum and/or inside a minimum."""

    kind: Literal[ConstraintKind.DISTANCE] = ConstraintKind.DISTANCE
    maximum: Threshold | None = None
    minimum: Threshold | None = None

    @model_validator(mode="after")
    def _bounds(self) -> DistanceConstraint:
        if self.maximum is None and self.minimum is None:
            raise ConstraintConfigError(
                "DistanceConstraint requires maximum and/or minimum; no distance is implied"
            )
        if self.maximum is not None:
            _align_threshold(self.maximum, self)
        if self.minimum is not None:
            _align_threshold(self.minimum, self)
        if (
            self.maximum is not None
            and self.minimum is not None
            and self.maximum.unit != self.minimum.unit
        ):
            raise ConstraintConfigError("distance maximum and minimum must use the same unit")
        if (
            self.maximum is not None
            and self.minimum is not None
            and self.minimum.value > self.maximum.value
        ):
            raise ConstraintConfigError("distance minimum must not exceed maximum")
        return self


ConstraintConfig = Annotated[
    MaximumConstraint
    | MinimumConstraint
    | BooleanExclusionConstraint
    | AllowedCategoryConstraint
    | DisallowedCategoryConstraint
    | OverlapConstraint
    | DistanceConstraint,
    Field(discriminator="kind"),
]


class ConstraintProfile(BaseModel):
    """Named bundle of explicit constraints.

    Example profiles shipped with SolarDSS are labelled ``example=True``. They
    are not universal standards and are never applied unless the caller passes
    them in.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    constraints: tuple[ConstraintConfig, ...] = ()
    example: bool = False
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> ConstraintProfile:
        seen: set[str] = set()
        for item in self.constraints:
            if item.constraint_id in seen:
                raise ConstraintConfigError(f"duplicate constraint_id {item.constraint_id!r}")
            seen.add(item.constraint_id)
            if item.profile != self.profile_id:
                raise ConstraintConfigError(
                    f"constraint {item.constraint_id!r} profile {item.profile!r} "
                    f"does not match profile_id {self.profile_id!r}"
                )
        validate_profile_against_registry(self)
        if self.example and not any("example" in note.casefold() for note in self.notes):
            raise ConstraintConfigError(
                "example profiles must state in notes that they are examples, "
                "not universal standards"
            )
        return self


class IndicatorSnapshot(BaseModel):
    """Observed (or missing) indicator used by a constraint evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    indicator_id: str = Field(min_length=1)
    available: bool
    value: float | str | bool | None = None
    unit: str | None = None


class ConstraintFailure(BaseModel):
    """One configured test that fired. Soft effects do not by themselves exclude."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    constraint_id: str = Field(min_length=1)
    kind: ConstraintKind
    effect: ConstraintEffect
    indicator_id: str = Field(min_length=1)
    message: str = Field(min_length=1)
    observed: float | str | bool | None = None
    observed_unit: str | None = None
    threshold_value: float | None = None
    threshold_unit: str | None = None
    penalty_weight: float | None = None
    source: str = Field(min_length=1)
    profile: str = Field(min_length=1)
    rationale: str = Field(min_length=1)


class CandidateEvaluation(BaseModel):
    """Hard/soft screening outcome for one candidate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    feasible: bool
    status: FeasibilityStatus
    failed_constraints: tuple[ConstraintFailure, ...] = ()
    warnings: tuple[str, ...] = ()
    indicators: tuple[IndicatorSnapshot, ...] = ()
    incomplete_constraints: tuple[str, ...] = ()
    provenance: ProvenanceRecord
    profile_id: str = Field(min_length=1)


class ConstraintEvaluationSet(BaseModel):
    """Batch of candidate evaluations against one profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(min_length=1)
    evaluations: tuple[CandidateEvaluation, ...]
    provenance: ProvenanceRecord

    def feasible_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.evaluations if item.feasible)

    def excluded_ids(self) -> tuple[str, ...]:
        return tuple(
            item.candidate_id
            for item in self.evaluations
            if item.status is FeasibilityStatus.EXCLUDED
        )


def _align_threshold(threshold: Threshold, constraint: _ConstraintFields) -> None:
    if threshold.profile != constraint.profile:
        raise ConstraintConfigError(
            f"threshold profile {threshold.profile!r} does not match "
            f"constraint profile {constraint.profile!r}"
        )
    if threshold.source != constraint.source:
        raise ConstraintConfigError(
            f"threshold source {threshold.source!r} does not match "
            f"constraint source {constraint.source!r}"
        )
    if threshold.rationale != constraint.rationale:
        raise ConstraintConfigError("threshold rationale must match the constraint rationale")


def _reject_blank_categories(values: tuple[str, ...]) -> None:
    if any(not item.strip() for item in values):
        raise ConstraintConfigError("category tokens must be non-empty strings")


def validate_profile_against_registry(
    profile: ConstraintProfile, catalog: IndicatorRegistry | None = None
) -> None:
    """Reject unknown indicators and threshold units that disagree with the registry."""

    table = default_registry if catalog is None else catalog
    for item in profile.constraints:
        spec = table.get(item.indicator_id)
        _validate_units(item, spec)


def _validate_units(item: ConstraintConfig, spec: IndicatorSpec) -> None:
    if isinstance(item, MaximumConstraint | MinimumConstraint | OverlapConstraint):
        if item.threshold.unit != spec.unit:
            raise ConstraintConfigError(
                f"{item.constraint_id}: threshold unit {item.threshold.unit!r} "
                f"does not match indicator unit {spec.unit!r}"
            )
        return
    if isinstance(item, DistanceConstraint):
        for bound in (item.maximum, item.minimum):
            if bound is not None and bound.unit != spec.unit:
                raise ConstraintConfigError(
                    f"{item.constraint_id}: distance unit {bound.unit!r} "
                    f"does not match indicator unit {spec.unit!r}"
                )


# Union alias for type checkers that do not unwrap Annotated.
AnyConstraint = (
    MaximumConstraint
    | MinimumConstraint
    | BooleanExclusionConstraint
    | AllowedCategoryConstraint
    | DisallowedCategoryConstraint
    | OverlapConstraint
    | DistanceConstraint
)
