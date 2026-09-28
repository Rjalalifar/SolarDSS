"""Infrastructure vocabulary. Distances are mapped; hosting capacity is never inferred."""

from __future__ import annotations

from solardss.indicators.contracts import (
    Controllability,
    IndicatorCategory,
    IndicatorRole,
    OptimizationDirection,
    RequirementLevel,
    SourceType,
)
from solardss.indicators.defaults import (
    ALL_MODES,
    ALL_PROJECTS,
    GROUND_PROJECTS,
    OSM_NEAREST_DISTANCE,
)
from solardss.indicators.factory import define
from solardss.indicators.ids import IndicatorId

INFRASTRUCTURE_INDICATORS = (
    define(
        IndicatorId.INFRA_DISTANCE_TO_POWER_LINE,
        name="Distance to power line",
        description=(
            "Straight-line distance to a mapped OSM power line in a bounded window. "
            "SolarDSS does not route or snap to networks."
        ),
        category=IndicatorCategory.INFRASTRUCTURE,
        role=IndicatorRole.INFRASTRUCTURE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="m",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(
            SourceType.OSM_OR_NETWORK_VECTOR,
            SourceType.GRID_OPERATOR,
            SourceType.CALLER_PROVIDED,
        ),
        required_or_optional=RequirementLevel.OPTIONAL,
        aliases=("distance_to_grid_m",),
        derivation_method=OSM_NEAREST_DISTANCE,
        notes="A nearby line is not hosting capacity.",
    ),
    define(
        IndicatorId.INFRA_DISTANCE_TO_SUBSTATION,
        name="Distance to substation",
        description="Straight-line distance to a mapped OSM substation in a bounded window.",
        category=IndicatorCategory.INFRASTRUCTURE,
        role=IndicatorRole.INFRASTRUCTURE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="m",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(
            SourceType.OSM_OR_NETWORK_VECTOR,
            SourceType.GRID_OPERATOR,
            SourceType.CALLER_PROVIDED,
        ),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=OSM_NEAREST_DISTANCE,
        notes="A nearby substation is not hosting capacity.",
    ),
    define(
        IndicatorId.INFRA_DISTANCE_TO_ROAD,
        name="Distance to road",
        description=(
            "Straight-line distance to a mapped OSM highway from an explicit class "
            "whitelist. Not network / routing distance."
        ),
        category=IndicatorCategory.INFRASTRUCTURE,
        role=IndicatorRole.INFRASTRUCTURE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="m",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(SourceType.OSM_OR_NETWORK_VECTOR, SourceType.CALLER_PROVIDED),
        required_or_optional=RequirementLevel.OPTIONAL,
        aliases=("distance_to_road_m",),
        derivation_method=OSM_NEAREST_DISTANCE,
    ),
    define(
        IndicatorId.INFRA_DISTANCE_TO_SETTLEMENT,
        name="Distance to settlement",
        description="Straight-line distance to a mapped OSM populated place or residential area.",
        category=IndicatorCategory.INFRASTRUCTURE,
        role=IndicatorRole.INFRASTRUCTURE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="m",
        optimization_direction=OptimizationDirection.NONE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=(SourceType.OSM_OR_NETWORK_VECTOR, SourceType.CALLER_PROVIDED),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=OSM_NEAREST_DISTANCE,
        notes="Setback versus access is policy; not a default maximize or minimize score.",
    ),
    define(
        IndicatorId.INFRA_DISTANCE_TO_DEMAND_CENTER,
        name="Distance to demand center",
        description=(
            "Distance to a declared demand center. The center must be supplied as "
            "coordinates (or an OSM id inside the search window). Settlements and "
            "industrial areas are not treated as demand centers."
        ),
        category=IndicatorCategory.INFRASTRUCTURE,
        role=IndicatorRole.INFRASTRUCTURE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="m",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=(SourceType.CALLER_PROVIDED, SourceType.OSM_OR_NETWORK_VECTOR),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=OSM_NEAREST_DISTANCE,
        notes="Not computed unless the caller declares the demand-center points.",
    ),
    define(
        IndicatorId.INFRA_GRID_HOSTING_CAPACITY,
        name="Grid hosting capacity",
        description=(
            "Available interconnection / hosting capacity at the relevant grid node. "
            "Not derivable from OpenStreetMap, from the presence of a power line, or "
            "from the presence of a substation."
        ),
        category=IndicatorCategory.INFRASTRUCTURE,
        role=IndicatorRole.INFRASTRUCTURE_INDICATOR,
        controllability=Controllability.EXTERNALLY_PROVIDED,
        unit="MW",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=(SourceType.GRID_OPERATOR, SourceType.CALLER_PROVIDED),
        required_or_optional=RequirementLevel.OPTIONAL,
        provenance_notes="Must come from a grid operator, utility study, or equivalent dataset.",
        notes="Do not infer capacity from mapped electrical features.",
    ),
)
