"""Shared candidate-row checks for ranking and Pareto analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from solardss.constraints.contracts import CandidateEvaluation, ConstraintEvaluationSet
from solardss.core.quantity import Quantity
from solardss.core.status import FeasibilityStatus
from solardss.indicators.ids import IndicatorId
from solardss.suitability.errors import RankingConfigError

ScoreValue = Quantity | float
ConstraintStatusInput = (
    ConstraintEvaluationSet | Mapping[str, FeasibilityStatus | CandidateEvaluation] | None
)
CandidateRows = Sequence[tuple[str, Mapping[str | IndicatorId, ScoreValue]]]


def require_unique_ids(rows: CandidateRows) -> None:
    ids = [ident for ident, _ in rows]
    if any(not ident for ident in ids):
        raise RankingConfigError("candidate ids must be non-empty strings")
    if len(set(ids)) != len(ids):
        raise RankingConfigError("candidate ids must be unique")


def resolve_constraint_statuses(
    rows: CandidateRows, constraint_status: ConstraintStatusInput
) -> dict[str, FeasibilityStatus]:
    if constraint_status is None:
        return {ident: FeasibilityStatus.FEASIBLE for ident, _ in rows}
    lookup: dict[str, FeasibilityStatus] = {}
    if isinstance(constraint_status, ConstraintEvaluationSet):
        for item in constraint_status.evaluations:
            lookup[item.candidate_id] = item.status
    else:
        for key, value in constraint_status.items():
            lookup[key] = value.status if isinstance(value, CandidateEvaluation) else value
    return {ident: lookup.get(ident, FeasibilityStatus.UNKNOWN) for ident, _ in rows}


def map_indicator_values(
    values: Mapping[str | IndicatorId, ScoreValue],
) -> dict[str, ScoreValue]:
    mapped: dict[str, ScoreValue] = {}
    for key, value in values.items():
        ident = key.value if isinstance(key, IndicatorId) else key
        mapped[ident] = value
    return mapped
