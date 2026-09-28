"""Structured missing-requirement reports for a SolarDecisionProblem.

Reasons are field-level and non-conversational. Ariadne (or any adapter) decides
whether to ask a user.
"""

from __future__ import annotations

from solardss.decision.contracts import MissingRequirement
from solardss.decision.problem import SolarDecisionProblem
from solardss.indicators.contracts import DecisionMode, SiteSelectionScope
from solardss.indicators.ids import IndicatorId

_OPTIONAL_CAPACITY_REASON = "needed only for capacity-specific sizing"
_OPTIONAL_BUDGET_REASON = "needed only for cost-constrained selection or design"
_OPTIONAL_OBJECTIVE_REASON = "needed to select a registered objective; none is inferred"
_OPTIONAL_GRID_FLAG_REASON = "needed only when interconnection policy is in scope"
_OPTIONAL_GRID_DISTANCE_REASON = "needed when grid_connection_required is true"
_OPTIONAL_ENV_REASON = "needed to weight environmental indicators; weights are not inferred"
_OPTIONAL_BATTERY_REASON = "needed only if storage is in scope"
_OPTIONAL_MODULE_REASON = "needed only for technology-specific design"


def get_missing_requirements(problem: SolarDecisionProblem) -> tuple[MissingRequirement, ...]:
    """Return gaps as structured records, required first, then optional.

    Spatial mode rules:

    * SITE_SELECTION + REGION_SEARCH → ``region`` required
    * SITE_SELECTION + AOI_SEARCH → ``aoi`` required
    * SITE_SELECTION + CANDIDATE_RANKING → ``candidates`` required
    * FIXED_SITE → ``location`` required
    """

    missing: list[MissingRequirement] = []
    _append_spatial(problem, missing)
    _append_optional(problem, missing)
    return tuple(missing)


def missing_requirement_records(problem: SolarDecisionProblem) -> list[dict[str, object]]:
    """JSON-like dicts matching the adapter contract (list of field/required/reason)."""

    return [
        {"field": item.field, "required": item.required, "reason": item.reason}
        for item in get_missing_requirements(problem)
    ]


def _append_spatial(
    problem: SolarDecisionProblem,
    missing: list[MissingRequirement],
) -> None:
    if problem.mode is DecisionMode.FIXED_SITE:
        if problem.location is None:
            missing.append(
                MissingRequirement(
                    field="location",
                    required=True,
                    reason="fixed-site evaluation requires location as given context",
                )
            )
        return
    scope = problem.site_selection_scope
    if scope is SiteSelectionScope.REGION_SEARCH and problem.region is None:
        missing.append(
            MissingRequirement(
                field="region",
                required=True,
                reason="region_search requires region",
            )
        )
    elif scope is SiteSelectionScope.AOI_SEARCH and problem.aoi is None:
        missing.append(
            MissingRequirement(
                field="aoi",
                required=True,
                reason="aoi_search requires aoi",
            )
        )
    elif scope is SiteSelectionScope.CANDIDATE_RANKING and problem.candidates is None:
        missing.append(
            MissingRequirement(
                field="candidates",
                required=True,
                reason="candidate_ranking requires candidates",
            )
        )


def _append_optional(
    problem: SolarDecisionProblem,
    missing: list[MissingRequirement],
) -> None:
    if problem.target_capacity_kw_resolved is None:
        missing.append(
            MissingRequirement(
                field="target_capacity_kw",
                required=False,
                reason=_OPTIONAL_CAPACITY_REASON,
            )
        )
    if problem.budget is None:
        missing.append(
            MissingRequirement(
                field="budget",
                required=False,
                reason=_OPTIONAL_BUDGET_REASON,
            )
        )
    if problem.objective is None:
        missing.append(
            MissingRequirement(
                field="objective",
                required=False,
                reason=_OPTIONAL_OBJECTIVE_REASON,
            )
        )
    if problem.grid_connection_required is None:
        missing.append(
            MissingRequirement(
                field="grid_connection_required",
                required=False,
                reason=_OPTIONAL_GRID_FLAG_REASON,
            )
        )
    elif problem.grid_connection_required is True and problem.maximum_grid_distance is None:
        missing.append(
            MissingRequirement(
                field="maximum_grid_distance",
                required=True,
                reason=_OPTIONAL_GRID_DISTANCE_REASON,
            )
        )
    env_needed = problem.objective == IndicatorId.OBJECTIVE_MIN_ENVIRONMENTAL_IMPACT.value
    if problem.environmental_preferences is None and (
        env_needed or problem.mode is DecisionMode.SITE_SELECTION
    ):
        missing.append(
            MissingRequirement(
                field="environmental_preferences",
                required=False,
                reason=_OPTIONAL_ENV_REASON,
            )
        )
    if problem.battery_required is None:
        missing.append(
            MissingRequirement(
                field="battery_required",
                required=False,
                reason=_OPTIONAL_BATTERY_REASON,
            )
        )
    if problem.module_preferences is None:
        missing.append(
            MissingRequirement(
                field="module_preferences",
                required=False,
                reason=_OPTIONAL_MODULE_REASON,
            )
        )
