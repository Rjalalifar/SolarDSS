"""SolarDecisionProblem: two decision modes, no hidden execution."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solardss.core.jsonutil import JsonValue, json_params
from solardss.core.quantity import Quantity
from solardss.data.geo import GeographicRef
from solardss.decision.contracts import (
    CandidateSet,
    DecisionOperation,
    EnvironmentalPreferences,
    MissingRequirement,
    ModulePreferences,
    UserConstraint,
)
from solardss.decision.errors import OperationNotAllowedError
from solardss.exceptions import InvalidParameterError
from solardss.indicators.catalog import registry
from solardss.indicators.contracts import (
    Controllability,
    DecisionMode,
    IndicatorRole,
    ProjectType,
    SiteSelectionScope,
)
from solardss.indicators.ids import IndicatorId

_CAPACITY_RTOL = 1e-9
_CAPACITY_ATOL = 1e-9

_SITE_SELECTION_OPERATIONS: tuple[DecisionOperation, ...] = (
    DecisionOperation.CANDIDATE_GENERATION,
    DecisionOperation.SITE_RANKING,
    DecisionOperation.SYSTEM_EVALUATION,
    DecisionOperation.SYSTEM_DESIGN,
)
_CANDIDATE_RANKING_OPERATIONS: tuple[DecisionOperation, ...] = (
    DecisionOperation.SITE_RANKING,
    DecisionOperation.SYSTEM_EVALUATION,
    DecisionOperation.SYSTEM_DESIGN,
)
_FIXED_SITE_OPERATIONS: tuple[DecisionOperation, ...] = (
    DecisionOperation.SYSTEM_EVALUATION,
    DecisionOperation.SYSTEM_DESIGN,
)


def _kw_from_mw(megawatts: float) -> float:
    return megawatts * 1000.0


class SolarDecisionProblem(BaseModel):
    """A well-typed PV decision problem.

    ``SITE_SELECTION`` treats location as a decision variable (see
    ``PV_LOCATION`` in the indicator registry). ``FIXED_SITE`` treats location
    as given context. This object never generates candidates or ranks sites.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: DecisionMode
    project_type: ProjectType
    site_selection_scope: SiteSelectionScope | None = None
    location: GeographicRef | None = None
    region: GeographicRef | None = None
    aoi: GeographicRef | None = None
    candidates: CandidateSet | None = None
    target_capacity_kw: float | None = None
    target_capacity_mw: float | None = None
    budget: Quantity | None = None
    objective: str | None = None
    grid_connection_required: bool | None = None
    maximum_grid_distance: Quantity | None = None
    environmental_preferences: EnvironmentalPreferences | None = None
    battery_required: bool | None = None
    module_preferences: ModulePreferences | None = None
    user_constraints: tuple[UserConstraint, ...] = ()
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_problem(self) -> SolarDecisionProblem:
        if self.mode is DecisionMode.SITE_SELECTION:
            if self.site_selection_scope is None:
                raise InvalidParameterError(
                    "SITE_SELECTION requires site_selection_scope "
                    "(region_search, aoi_search, or candidate_ranking)"
                )
        elif self.site_selection_scope is not None:
            raise InvalidParameterError("FIXED_SITE must not set site_selection_scope")

        if self.mode is DecisionMode.FIXED_SITE and (
            self.region is not None or self.aoi is not None or self.candidates is not None
        ):
            raise InvalidParameterError(
                "FIXED_SITE forbids region, aoi, and candidates; "
                "candidate generation and site ranking must not run"
            )

        if self.mode is DecisionMode.SITE_SELECTION:
            scope = self.site_selection_scope
            if scope is SiteSelectionScope.REGION_SEARCH and self.aoi is not None:
                raise InvalidParameterError("REGION_SEARCH uses region, not aoi")
            if scope is SiteSelectionScope.REGION_SEARCH and self.candidates is not None:
                raise InvalidParameterError(
                    "REGION_SEARCH does not take a pre-ranked candidate set"
                )
            if scope is SiteSelectionScope.AOI_SEARCH and self.region is not None:
                raise InvalidParameterError("AOI_SEARCH uses aoi, not region")
            if scope is SiteSelectionScope.AOI_SEARCH and self.candidates is not None:
                raise InvalidParameterError("AOI_SEARCH does not take a pre-ranked candidate set")
            if scope is SiteSelectionScope.CANDIDATE_RANKING and self.region is not None:
                raise InvalidParameterError("CANDIDATE_RANKING uses candidates, not region")
            if scope is SiteSelectionScope.CANDIDATE_RANKING and self.aoi is not None:
                raise InvalidParameterError("CANDIDATE_RANKING uses candidates, not aoi")

        _validate_capacity(self.target_capacity_kw, self.target_capacity_mw)
        if self.budget is not None and self.budget.value < 0.0:
            raise InvalidParameterError("budget must be non-negative")
        if self.maximum_grid_distance is not None:
            metres = self.maximum_grid_distance.require_unit("m")
            if metres < 0.0:
                raise InvalidParameterError("maximum_grid_distance must be non-negative")
        if self.objective is not None:
            _validate_objective(self.objective)
        for constraint in self.user_constraints:
            if constraint.indicator_id is not None:
                registry.get(constraint.indicator_id)
        if self.environmental_preferences is not None:
            for indicator_id in self.environmental_preferences.minimize_indicator_ids:
                registry.get(indicator_id)
        normalized = json_params(self.metadata)
        if normalized != self.metadata:
            return self.model_copy(update={"metadata": normalized})
        return self

    @property
    def location_role(self) -> IndicatorRole:
        """Role of ``PV_LOCATION`` under this problem's decision mode."""

        return registry.resolve(IndicatorId.PV_LOCATION, mode=self.mode).role

    @property
    def location_controllability(self) -> Controllability:
        """Controllability of ``PV_LOCATION`` under this problem's decision mode."""

        return registry.resolve(IndicatorId.PV_LOCATION, mode=self.mode).controllability

    @property
    def target_capacity_kw_resolved(self) -> float | None:
        """Capacity in kW if either capacity field is set."""

        if self.target_capacity_kw is not None:
            return self.target_capacity_kw
        if self.target_capacity_mw is not None:
            return _kw_from_mw(self.target_capacity_mw)
        return None

    def allowed_operations(self) -> tuple[DecisionOperation, ...]:
        """Operations this problem statement authorizes. Nothing is executed."""

        if self.mode is DecisionMode.FIXED_SITE:
            return _FIXED_SITE_OPERATIONS
        if self.site_selection_scope is SiteSelectionScope.CANDIDATE_RANKING:
            return _CANDIDATE_RANKING_OPERATIONS
        return _SITE_SELECTION_OPERATIONS

    def operation_allowed(self, operation: DecisionOperation | str) -> bool:
        """Return whether ``operation`` is in scope for this problem."""

        wanted = _as_operation(operation)
        return wanted in self.allowed_operations()

    def require_operation(self, operation: DecisionOperation | str) -> None:
        """Raise if candidate generation or ranking is requested on a fixed site."""

        wanted = _as_operation(operation)
        if wanted not in self.allowed_operations():
            scope = None if self.site_selection_scope is None else self.site_selection_scope.value
            raise OperationNotAllowedError(
                f"{wanted.value} is not allowed for mode={self.mode.value} scope={scope}"
            )

    def missing_requirements(self) -> tuple[MissingRequirement, ...]:
        """Return structured gaps; see ``get_missing_requirements``."""

        from solardss.decision.requirements import get_missing_requirements

        return get_missing_requirements(self)


def _as_operation(value: DecisionOperation | str) -> DecisionOperation:
    if isinstance(value, DecisionOperation):
        return value
    try:
        return DecisionOperation(value)
    except ValueError as exc:
        raise InvalidParameterError(f"Unknown decision operation {value!r}") from exc


def _validate_capacity(kw: float | None, mw: float | None) -> None:
    if kw is not None and kw < 0.0:
        raise InvalidParameterError("target_capacity_kw must be non-negative")
    if mw is not None and mw < 0.0:
        raise InvalidParameterError("target_capacity_mw must be non-negative")
    if kw is not None and mw is not None:
        expected = _kw_from_mw(mw)
        scale = max(abs(kw), abs(expected), 1.0)
        if abs(kw - expected) > max(_CAPACITY_ATOL, _CAPACITY_RTOL * scale):
            raise InvalidParameterError(
                f"target_capacity_kw and target_capacity_mw disagree ({kw} kW vs {mw} MW)"
            )


def _validate_objective(objective_id: str) -> None:
    spec = registry.get(objective_id)
    if spec.role is not IndicatorRole.OBJECTIVE:
        raise InvalidParameterError(
            f"{objective_id!r} is registered but is not an objective (role={spec.role.value})"
        )
