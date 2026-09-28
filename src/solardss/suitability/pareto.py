"""Non-dominated set from raw objective values. No scalarization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from solardss.core.hashing import fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.core.quantity import Quantity
from solardss.core.status import EvaluationStatus, FeasibilityStatus
from solardss.exceptions import InvalidParameterError
from solardss.indicators.catalog import registry as default_registry
from solardss.indicators.contracts import OptimizationDirection
from solardss.indicators.ids import IndicatorId
from solardss.indicators.registry import IndicatorRegistry
from solardss.provenance.record import provenance_record
from solardss.suitability.contracts import (
    MultiObjectiveResult,
    ObjectiveSpec,
    ObjectiveValue,
    OmissionReason,
    OmittedCandidate,
    ParetoPoint,
    validate_objectives,
)
from solardss.suitability.inputs import (
    CandidateRows,
    ConstraintStatusInput,
    ScoreValue,
    map_indicator_values,
    require_unique_ids,
    resolve_constraint_statuses,
)

_ATOL = 1e-12


def pareto_candidates(
    rows: CandidateRows,
    objectives: Sequence[ObjectiveSpec],
    *,
    constraint_status: ConstraintStatusInput = None,
    catalog: IndicatorRegistry | None = None,
) -> MultiObjectiveResult:
    """Identify non-dominated candidates using actual objective values.

    Does not compute a Weighted Sum Model score.
    """

    table = default_registry if catalog is None else catalog
    specs = tuple(objectives)
    validate_objectives(specs, table)
    require_unique_ids(rows)
    statuses = resolve_constraint_statuses(rows, constraint_status)
    omitted: list[OmittedCandidate] = []
    considered: list[tuple[str, FeasibilityStatus, dict[str, float]]] = []
    for ident, values in rows:
        status = statuses[ident]
        if status is FeasibilityStatus.EXCLUDED:
            omitted.append(
                OmittedCandidate(
                    candidate_id=ident,
                    reason=OmissionReason.EXCLUDED,
                    constraint_status=status,
                )
            )
            continue
        if status is FeasibilityStatus.UNKNOWN:
            omitted.append(
                OmittedCandidate(
                    candidate_id=ident,
                    reason=OmissionReason.UNKNOWN_FEASIBILITY,
                    constraint_status=status,
                )
            )
            continue
        parsed, missing = _parse_objectives(ident, values, specs)
        if missing:
            omitted.append(
                OmittedCandidate(
                    candidate_id=ident,
                    reason=OmissionReason.INCOMPLETE_INDICATORS,
                    constraint_status=status,
                    missing_indicators=missing,
                )
            )
            continue
        considered.append((ident, status, parsed))

    points: list[ParetoPoint] = []
    for ident, status, parsed in considered:
        dominated_by = tuple(
            other_id
            for other_id, _, other_values in considered
            if other_id != ident and _dominates(other_values, parsed, specs)
        )
        points.append(
            ParetoPoint(
                candidate_id=ident,
                objectives=tuple(
                    ObjectiveValue(
                        indicator_id=spec.indicator_id,
                        available=True,
                        value=parsed[spec.indicator_id],
                        unit=spec.unit,
                    )
                    for spec in specs
                ),
                nondominated=not dominated_by,
                dominated_by=dominated_by,
                constraint_status=status,
            )
        )
    points.sort(key=lambda item: item.candidate_id)
    record = provenance_record(
        MethodId.PARETO_FRONT,
        parameters={
            "method": "pareto_nondominated",
            "objective_ids": [item.indicator_id for item in specs],
            "directions": [item.direction.value for item in specs],
            "point_count": len(points),
            "nondominated_count": sum(1 for item in points if item.nondominated),
            "omitted_count": len(omitted),
            "scalarized": False,
        },
        input_fingerprints=(
            fingerprint_canonical(
                {
                    "ids": [ident for ident, _ in rows],
                    "objectives": [item.indicator_id for item in specs],
                }
            ),
        ),
        notes=(
            "non-dominated set uses raw objective values; WSM score is not computed",
            "missing values are not treated as zero",
        ),
    )
    return MultiObjectiveResult(
        objectives=specs,
        points=tuple(points),
        omitted=tuple(omitted),
        provenance=record,
        status=EvaluationStatus.EVALUATED,
    )


def _parse_objectives(
    candidate_id: str,
    values: Mapping[str | IndicatorId, ScoreValue],
    specs: tuple[ObjectiveSpec, ...],
) -> tuple[dict[str, float], tuple[str, ...]]:
    mapped = map_indicator_values(values)
    parsed: dict[str, float] = {}
    missing: list[str] = []
    for spec in specs:
        raw = mapped.get(spec.indicator_id)
        if raw is None:
            missing.append(spec.indicator_id)
            continue
        parsed[spec.indicator_id] = _numeric(raw, spec, candidate_id)
    return parsed, tuple(missing)


def _numeric(raw: ScoreValue, spec: ObjectiveSpec, candidate_id: str) -> float:
    if isinstance(raw, Quantity):
        value = raw.require_unit(spec.unit)
    else:
        value = Quantity(value=float(raw), unit=spec.unit).require_unit(spec.unit)
    if value != value or value in (float("inf"), float("-inf")):
        raise InvalidParameterError(
            f"{candidate_id}: {spec.indicator_id} must be finite; missing stays missing"
        )
    return value


def _dominates(
    left: Mapping[str, float],
    right: Mapping[str, float],
    specs: tuple[ObjectiveSpec, ...],
) -> bool:
    """True if ``left`` is at least as good on every objective and better on one."""

    at_least = True
    better = False
    for spec in specs:
        a = left[spec.indicator_id]
        b = right[spec.indicator_id]
        if spec.direction is OptimizationDirection.MAXIMIZE:
            if a + _ATOL < b:
                at_least = False
                break
            if a > b + _ATOL:
                better = True
        else:
            if a - _ATOL > b:
                at_least = False
                break
            if a < b - _ATOL:
                better = True
    return at_least and better
