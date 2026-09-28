"""Evaluate explicit hard/soft constraints. Missing values are not treated as zero."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from solardss.constraints.contracts import (
    AllowedCategoryConstraint,
    AnyConstraint,
    BooleanExclusionConstraint,
    CandidateEvaluation,
    ConstraintEffect,
    ConstraintEvaluationSet,
    ConstraintFailure,
    ConstraintKind,
    ConstraintProfile,
    DisallowedCategoryConstraint,
    DistanceConstraint,
    IndicatorSnapshot,
    MaximumConstraint,
    MinimumConstraint,
    OverlapConstraint,
    Threshold,
    validate_profile_against_registry,
)
from solardss.core.hashing import fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.core.quantity import Quantity
from solardss.core.status import FeasibilityStatus
from solardss.engine.contracts import IndicatorOutcome, IndicatorSet
from solardss.exceptions import InvalidParameterError
from solardss.indicators.catalog import registry as default_registry
from solardss.indicators.ids import IndicatorId
from solardss.indicators.registry import IndicatorRegistry
from solardss.provenance.record import provenance_record

IndicatorValue = Quantity | float | str | bool


class ConstraintEngine:
    """Apply a ``ConstraintProfile`` to one candidate or a batch.

    A profile is never inferred from project type. Missing indicator values stay
    missing: they are not substituted with zero.
    """

    def __init__(self, catalog: IndicatorRegistry | None = None) -> None:
        self._catalog = default_registry if catalog is None else catalog

    def evaluate(
        self,
        candidate_id: str,
        indicators: Mapping[str | IndicatorId, IndicatorValue] | IndicatorSet,
        profile: ConstraintProfile,
    ) -> CandidateEvaluation:
        """Screen one candidate. ``feasible`` is true only when every hard test passed."""

        validate_profile_against_registry(profile, self._catalog)
        values = _normalize_indicators(indicators)
        failures: list[ConstraintFailure] = []
        warnings: list[str] = []
        incomplete: list[str] = []
        snapshots: list[IndicatorSnapshot] = []
        seen_ids: set[str] = set()
        for constraint in profile.constraints:
            spec = self._catalog.get(constraint.indicator_id)
            if spec.id not in seen_ids:
                snapshots.append(_snapshot(spec.id, spec.unit, values.get(spec.id)))
                seen_ids.add(spec.id)
            raw = values.get(spec.id)
            if raw is None:
                incomplete.append(constraint.constraint_id)
                warnings.append(
                    f"{constraint.constraint_id}: {spec.id} is missing; zero is not substituted"
                )
                continue
            fired = _evaluate_constraint(constraint, raw, spec.unit)
            if fired is None:
                continue
            failures.append(fired)
            if constraint.effect is not ConstraintEffect.HARD_EXCLUSION:
                warnings.append(fired.message)
        hard_fail = any(item.effect is ConstraintEffect.HARD_EXCLUSION for item in failures)
        hard_incomplete = any(
            item.constraint_id in set(incomplete) and item.effect is ConstraintEffect.HARD_EXCLUSION
            for item in profile.constraints
        )
        if hard_fail:
            status = FeasibilityStatus.EXCLUDED
            feasible = False
        elif hard_incomplete:
            status = FeasibilityStatus.UNKNOWN
            feasible = False
        else:
            status = FeasibilityStatus.FEASIBLE
            feasible = True
        record = provenance_record(
            MethodId.CONSTRAINT_ENGINE,
            parameters={
                "candidate_id": candidate_id,
                "profile_id": profile.profile_id,
                "example_profile": profile.example,
                "constraint_ids": [item.constraint_id for item in profile.constraints],
                "feasible": feasible,
                "status": status.value,
            },
            input_fingerprints=(
                fingerprint_canonical(
                    {
                        "candidate_id": candidate_id,
                        "profile_id": profile.profile_id,
                        "indicators": _fingerprint_values(values),
                    }
                ),
            ),
            notes=(
                "thresholds are configuration, not universal scientific cutoffs",
                *profile.notes[:2],
            ),
        )
        return CandidateEvaluation(
            candidate_id=candidate_id,
            feasible=feasible,
            status=status,
            failed_constraints=tuple(failures),
            warnings=tuple(warnings),
            indicators=tuple(snapshots),
            incomplete_constraints=tuple(incomplete),
            provenance=record,
            profile_id=profile.profile_id,
        )

    def evaluate_many(
        self,
        rows: Sequence[tuple[str, Mapping[str | IndicatorId, IndicatorValue] | IndicatorSet]],
        profile: ConstraintProfile,
    ) -> ConstraintEvaluationSet:
        """Evaluate each ``(candidate_id, indicators)`` pair against ``profile``."""

        evaluations = tuple(self.evaluate(ident, values, profile) for ident, values in rows)
        record = provenance_record(
            MethodId.CONSTRAINT_ENGINE,
            parameters={
                "profile_id": profile.profile_id,
                "example_profile": profile.example,
                "count": len(evaluations),
                "feasible_count": sum(1 for item in evaluations if item.feasible),
            },
            input_fingerprints=tuple(item.provenance.input_fingerprints[0] for item in evaluations)
            if evaluations
            else (),
            notes=("batch constraint evaluation; ranking is not performed",),
        )
        return ConstraintEvaluationSet(
            profile_id=profile.profile_id,
            evaluations=evaluations,
            provenance=record,
        )


def _normalize_indicators(
    indicators: Mapping[str | IndicatorId, IndicatorValue] | IndicatorSet,
) -> dict[str, IndicatorValue]:
    if isinstance(indicators, IndicatorSet):
        return _from_indicator_set(indicators)
    out: dict[str, IndicatorValue] = {}
    for key, value in indicators.items():
        ident = key.value if isinstance(key, IndicatorId) else key
        out[ident] = value
    return out


def _from_indicator_set(bundle: IndicatorSet) -> dict[str, IndicatorValue]:
    out: dict[str, IndicatorValue] = {}
    for item in bundle.results:
        if item.outcome is not IndicatorOutcome.AVAILABLE or item.value is None:
            continue
        if isinstance(item.value, bool | str):
            out[item.indicator_id] = item.value
        else:
            unit = item.unit or "1"
            out[item.indicator_id] = Quantity(value=float(item.value), unit=unit)
    return out


def _snapshot(indicator_id: str, unit: str, raw: IndicatorValue | None) -> IndicatorSnapshot:
    if raw is None:
        return IndicatorSnapshot(indicator_id=indicator_id, available=False, unit=unit)
    if isinstance(raw, Quantity):
        return IndicatorSnapshot(
            indicator_id=indicator_id,
            available=True,
            value=raw.value,
            unit=raw.unit,
        )
    if isinstance(raw, bool | str):
        return IndicatorSnapshot(indicator_id=indicator_id, available=True, value=raw, unit=unit)
    return IndicatorSnapshot(indicator_id=indicator_id, available=True, value=float(raw), unit=unit)


def _evaluate_constraint(
    constraint: AnyConstraint, raw: IndicatorValue, spec_unit: str
) -> ConstraintFailure | None:
    if isinstance(constraint, MaximumConstraint | OverlapConstraint):
        value = _as_float(raw, spec_unit, constraint.threshold.unit)
        if not _above(value, constraint.threshold.value, constraint.inclusive):
            return None
        return _numeric_failure(constraint, constraint.kind, value, constraint.threshold)
    if isinstance(constraint, MinimumConstraint):
        value = _as_float(raw, spec_unit, constraint.threshold.unit)
        if not _below(value, constraint.threshold.value, constraint.inclusive):
            return None
        return _numeric_failure(constraint, constraint.kind, value, constraint.threshold)
    if isinstance(constraint, DistanceConstraint):
        bound = constraint.maximum or constraint.minimum
        if bound is None:
            raise InvalidParameterError("DistanceConstraint is missing both bounds")
        value = _as_float(raw, spec_unit, bound.unit)
        hit: Threshold | None = None
        if constraint.maximum is not None and _above(
            value, constraint.maximum.value, constraint.inclusive
        ):
            hit = constraint.maximum
        elif constraint.minimum is not None and _below(
            value, constraint.minimum.value, constraint.inclusive
        ):
            hit = constraint.minimum
        if hit is None:
            return None
        return _numeric_failure(constraint, ConstraintKind.DISTANCE, value, hit)
    if isinstance(constraint, BooleanExclusionConstraint):
        if not isinstance(raw, bool):
            raise InvalidParameterError(
                f"{constraint.constraint_id} requires a boolean indicator value"
            )
        excluded = raw if constraint.true_means_excluded else not raw
        if not excluded:
            return None
        return _failure(
            constraint,
            ConstraintKind.BOOLEAN_EXCLUSION,
            observed=raw,
            observed_unit=None,
            message=(
                f"{constraint.indicator_id} is {raw}; "
                f"{constraint.effect.value} ({constraint.constraint_id})"
            ),
        )
    if isinstance(constraint, AllowedCategoryConstraint):
        token = _as_category(raw, constraint.constraint_id)
        allowed = {item.casefold() for item in constraint.allowed}
        if token.casefold() in allowed:
            return None
        return _failure(
            constraint,
            ConstraintKind.ALLOWED_CATEGORY,
            observed=token,
            observed_unit="categorical",
            message=(
                f"{constraint.indicator_id}={token!r} is not in allowed "
                f"{constraint.allowed} ({constraint.constraint_id})"
            ),
        )
    assert isinstance(constraint, DisallowedCategoryConstraint)
    token = _as_category(raw, constraint.constraint_id)
    blocked = {item.casefold() for item in constraint.disallowed}
    if token.casefold() not in blocked:
        return None
    return _failure(
        constraint,
        ConstraintKind.DISALLOWED_CATEGORY,
        observed=token,
        observed_unit="categorical",
        message=(f"{constraint.indicator_id}={token!r} is disallowed ({constraint.constraint_id})"),
    )


def _as_float(raw: IndicatorValue, spec_unit: str, required_unit: str) -> float:
    if isinstance(raw, Quantity):
        return raw.require_unit(required_unit)
    if isinstance(raw, bool | str):
        raise InvalidParameterError("numeric constraint received a non-numeric indicator value")
    return Quantity(value=float(raw), unit=spec_unit).require_unit(required_unit)


def _as_category(raw: IndicatorValue, constraint_id: str) -> str:
    if isinstance(raw, str):
        if not raw.strip():
            raise InvalidParameterError(f"{constraint_id} received an empty category")
        return raw
    raise InvalidParameterError(f"{constraint_id} requires a categorical string value")


def _above(value: float, threshold: float, inclusive: bool) -> bool:
    return value >= threshold if inclusive else value > threshold


def _below(value: float, threshold: float, inclusive: bool) -> bool:
    return value <= threshold if inclusive else value < threshold


def _numeric_failure(
    constraint: AnyConstraint,
    kind: ConstraintKind,
    observed: float,
    threshold: Threshold,
) -> ConstraintFailure:
    relation = "exceeds" if kind is not ConstraintKind.MINIMUM else "is below"
    if kind is ConstraintKind.DISTANCE and constraint.kind is ConstraintKind.DISTANCE:
        distance = constraint
        if distance.minimum is not None and threshold is distance.minimum:
            relation = "is below"
    message = (
        f"{constraint.indicator_id} {observed} {threshold.unit} {relation} "
        f"{threshold.value} {threshold.unit} ({constraint.effect.value}, "
        f"{constraint.constraint_id})"
    )
    return _failure(
        constraint,
        kind,
        observed=observed,
        observed_unit=threshold.unit,
        message=message,
        threshold=threshold,
    )


def _failure(
    constraint: AnyConstraint,
    kind: ConstraintKind,
    *,
    observed: float | str | bool | None,
    observed_unit: str | None,
    message: str,
    threshold: Threshold | None = None,
) -> ConstraintFailure:
    return ConstraintFailure(
        constraint_id=constraint.constraint_id,
        kind=kind,
        effect=constraint.effect,
        indicator_id=constraint.indicator_id,
        message=message,
        observed=observed,
        observed_unit=observed_unit,
        threshold_value=None if threshold is None else threshold.value,
        threshold_unit=None if threshold is None else threshold.unit,
        penalty_weight=constraint.penalty_weight,
        source=constraint.source,
        profile=constraint.profile,
        rationale=constraint.rationale,
    )


def _fingerprint_values(values: Mapping[str, IndicatorValue]) -> dict[str, str | float | bool]:
    payload: dict[str, str | float | bool] = {}
    for key in sorted(values):
        raw = values[key]
        if isinstance(raw, Quantity):
            payload[key] = f"{raw.value}:{raw.unit}"
        else:
            payload[key] = raw
    return payload
