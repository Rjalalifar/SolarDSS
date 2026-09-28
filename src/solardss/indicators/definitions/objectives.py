"""Decision objectives. These name preferences; they are not computed here."""

from __future__ import annotations

from solardss.indicators.contracts import (
    Controllability,
    DecisionMode,
    IndicatorCategory,
    IndicatorRole,
    OptimizationDirection,
    RequirementLevel,
    SourceType,
)
from solardss.indicators.defaults import ALL_MODES, ALL_PROJECTS, GROUND_PROJECTS, ROOF_PROJECTS
from solardss.indicators.factory import define
from solardss.indicators.ids import IndicatorId

_OBJECTIVE_SOURCES = (SourceType.DERIVED_FROM_REGISTERED_INDICATORS, SourceType.CALLER_PROVIDED)
_OBJECTIVE_METHOD_NOTE = (
    "Objective identity only. SolarDSS does not scalarize or optimize this until "
    "a method is registered and inputs are supplied."
)

OBJECTIVE_INDICATORS = (
    define(
        IndicatorId.OBJECTIVE_MAX_ANNUAL_ENERGY,
        name="Maximize annual energy",
        description="Preference for higher annual AC or DC energy, given supplied yield inputs.",
        category=IndicatorCategory.OBJECTIVE,
        role=IndicatorRole.OBJECTIVE,
        controllability=Controllability.DERIVED,
        unit="kWh/year",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_OBJECTIVE_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        notes=_OBJECTIVE_METHOD_NOTE,
    ),
    define(
        IndicatorId.OBJECTIVE_MAX_SPECIFIC_YIELD,
        name="Maximize specific yield",
        description="Preference for higher kWh per installed kWp.",
        category=IndicatorCategory.OBJECTIVE,
        role=IndicatorRole.OBJECTIVE,
        controllability=Controllability.DERIVED,
        unit="kWh/kWp/year",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_OBJECTIVE_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        notes=_OBJECTIVE_METHOD_NOTE,
    ),
    define(
        IndicatorId.OBJECTIVE_MIN_INITIAL_COST,
        name="Minimize initial cost",
        description="Preference for lower capital cost. Cost data must be supplied externally.",
        category=IndicatorCategory.OBJECTIVE,
        role=IndicatorRole.OBJECTIVE,
        controllability=Controllability.DERIVED,
        unit="currency",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=(SourceType.CALLER_PROVIDED, SourceType.ENGINEERING_DATASET),
        required_or_optional=RequirementLevel.OPTIONAL,
        notes="SolarDSS has no cost database.",
    ),
    define(
        IndicatorId.OBJECTIVE_MIN_GRID_CONNECTION_DISTANCE,
        name="Minimize grid-connection distance",
        description="Preference for shorter electrical connection distance in site selection.",
        category=IndicatorCategory.OBJECTIVE,
        role=IndicatorRole.OBJECTIVE,
        controllability=Controllability.DERIVED,
        unit="m",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=(DecisionMode.SITE_SELECTION,),
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=_OBJECTIVE_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        notes=_OBJECTIVE_METHOD_NOTE,
    ),
    define(
        IndicatorId.OBJECTIVE_MIN_ENVIRONMENTAL_IMPACT,
        name="Minimize environmental impact",
        description=(
            "Preference for lower environmental pressure using registered environmental "
            "indicators. No composite impact index is invented."
        ),
        category=IndicatorCategory.OBJECTIVE,
        role=IndicatorRole.OBJECTIVE,
        controllability=Controllability.DERIVED,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_OBJECTIVE_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        notes="Weights over environmental indicators must be supplied explicitly.",
    ),
    define(
        IndicatorId.OBJECTIVE_MAX_SELF_CONSUMPTION,
        name="Maximize self-consumption",
        description="Preference for on-site consumption of generated electricity.",
        category=IndicatorCategory.OBJECTIVE,
        role=IndicatorRole.OBJECTIVE,
        controllability=Controllability.DERIVED,
        unit="1",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ROOF_PROJECTS,
        preferred_source_types=_OBJECTIVE_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        notes="Requires a demand profile supplied outside SolarDSS.",
    ),
    define(
        IndicatorId.OBJECTIVE_BALANCED,
        name="Balanced multi-objective",
        description=(
            "Explicit multi-criteria compromise. Requires caller-supplied weights or an "
            "AHP matrix; SolarDSS does not choose the trade-off."
        ),
        category=IndicatorCategory.OBJECTIVE,
        role=IndicatorRole.OBJECTIVE,
        controllability=Controllability.DERIVED,
        unit="1",
        optimization_direction=OptimizationDirection.TARGET,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_OBJECTIVE_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
    ),
)
