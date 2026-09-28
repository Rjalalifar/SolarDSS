"""Greedy budget selection. This is not a global optimizer."""

from __future__ import annotations

from solardss.core.methods import MethodId
from solardss.core.results import SelectionResult
from solardss.core.status import EvaluationStatus
from solardss.data.tables import CandidateTable
from solardss.exceptions import InvalidParameterError, SchemaError
from solardss.provenance.record import provenance_record


def greedy_select(
    table: CandidateTable,
    *,
    score_column: str,
    cost_column: str,
    budget: float,
) -> SelectionResult:
    """Select candidates by descending score until ``budget`` is exhausted.

    Tie-break is ``candidate_id`` ascending (stable and deterministic).
    A candidate whose cost exceeds the remaining budget is skipped, not treated
    as a stopping condition. Negative scores or costs are rejected.

    This is the greedy-by-score heuristic, not an optimal knapsack solver.
    """

    if budget < 0.0:
        raise InvalidParameterError("budget must be non-negative")
    frame = table.frame
    for column in (score_column, cost_column):
        if column not in frame.columns:
            raise SchemaError(f"Selection table is missing column {column!r}")
    scores = frame[score_column].to_numpy(dtype=float)
    costs = frame[cost_column].to_numpy(dtype=float)
    if not (len(scores) == len(costs) == len(frame)):
        raise SchemaError("Score and cost columns must align with the table")
    if (scores < 0.0).any() or (costs < 0.0).any():
        raise InvalidParameterError("greedy_select rejects negative scores or costs")

    order = frame.sort_values(
        by=[score_column, table.id_column],
        ascending=[False, True],
        kind="mergesort",
    ).index

    remaining = float(budget)
    selected: list[str] = []
    skipped: list[str] = []
    total_score = 0.0
    total_cost = 0.0
    for index in order:
        cost = float(frame.loc[index, cost_column])
        ident = str(frame.loc[index, table.id_column])
        if cost > remaining:
            skipped.append(ident)
            continue
        selected.append(ident)
        remaining -= cost
        total_score += float(frame.loc[index, score_column])
        total_cost += cost

    status = EvaluationStatus.SELECTED if selected else EvaluationStatus.EVALUATED
    warnings: tuple[str, ...] = ("greedy_by_score_not_global_optimum",)
    return SelectionResult(
        status=status,
        selected_ids=tuple(selected),
        remaining_budget=remaining,
        total_score=total_score,
        total_cost=total_cost,
        skipped_over_budget_ids=tuple(skipped),
        provenance=provenance_record(
            MethodId.GREEDY_BUDGET_SELECTION,
            parameters={
                "score_column": score_column,
                "cost_column": cost_column,
                "budget": budget,
                "rule": "greedy_by_score_skip_over_budget",
            },
            input_fingerprints=(table.fingerprint(),),
            notes=warnings,
        ),
        warnings=warnings,
    )
