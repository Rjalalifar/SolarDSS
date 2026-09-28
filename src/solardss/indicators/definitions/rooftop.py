"""Rooftop-specific vocabulary. Structural safety is never inferred from footprint."""

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
    OSM_FOOTPRINT_AREA,
    OSM_ROOF_TAG_ORIENTATION,
    OSM_ROOF_TAG_SLOPE,
    ROOF_PROJECTS,
    ROOF_USABLE_FROM_FOOTPRINT,
)
from solardss.indicators.factory import define
from solardss.indicators.ids import IndicatorId

ROOFTOP_INDICATORS = (
    define(
        IndicatorId.ROOF_BUILDING_FOOTPRINT_AREA,
        name="Building footprint area",
        description="Plan-view building footprint area. Not usable PV area.",
        category=IndicatorCategory.ROOFTOP,
        role=IndicatorRole.ROOFTOP_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="m2",
        optimization_direction=OptimizationDirection.NONE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ROOF_PROJECTS,
        preferred_source_types=(SourceType.BUILDING_FOOTPRINT, SourceType.CADASTRAL_OR_LAND_ADMIN),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=OSM_FOOTPRINT_AREA,
        notes="Do not treat footprint as structurally usable roof area.",
    ),
    define(
        IndicatorId.ROOF_USABLE_AREA,
        name="Usable roof area",
        description=(
            "Area available for modules after setbacks, obstructions, and slope. "
            "Not equal to building footprint."
        ),
        category=IndicatorCategory.ROOFTOP,
        role=IndicatorRole.ROOFTOP_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="m2",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ROOF_PROJECTS,
        preferred_source_types=(
            SourceType.ENGINEERING_DATASET,
            SourceType.CALLER_PROVIDED,
            SourceType.DERIVED_FROM_REGISTERED_INDICATORS,
        ),
        required_or_optional=RequirementLevel.REQUIRED,
        derivation_method=ROOF_USABLE_FROM_FOOTPRINT,
    ),
    define(
        IndicatorId.ROOF_ORIENTATION,
        name="Roof orientation",
        description=(
            "Azimuth of the roof plane, degrees clockwise from north. Mapped from "
            "OSM roof:direction only when that tag is a compass point or degrees."
        ),
        category=IndicatorCategory.ROOFTOP,
        role=IndicatorRole.ROOFTOP_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="deg",
        optimization_direction=OptimizationDirection.TARGET,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ROOF_PROJECTS,
        preferred_source_types=(
            SourceType.BUILDING_FOOTPRINT,
            SourceType.ENGINEERING_DATASET,
            SourceType.CALLER_PROVIDED,
        ),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=OSM_ROOF_TAG_ORIENTATION,
        notes="Not inferred from footprint long-axis or roof:shape.",
    ),
    define(
        IndicatorId.ROOF_SLOPE,
        name="Roof slope",
        description=(
            "Slope of the roof plane in degrees. Distinct from terrain slope. "
            "Mapped from OSM roof:angle, or 0 when roof:shape=flat."
        ),
        category=IndicatorCategory.ROOFTOP,
        role=IndicatorRole.ROOFTOP_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="deg",
        optimization_direction=OptimizationDirection.TARGET,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ROOF_PROJECTS,
        preferred_source_types=(
            SourceType.BUILDING_FOOTPRINT,
            SourceType.ENGINEERING_DATASET,
            SourceType.CALLER_PROVIDED,
        ),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=OSM_ROOF_TAG_SLOPE,
        notes="gabled/hipped/skillion without roof:angle do not invent a slope.",
    ),
    define(
        IndicatorId.ROOF_SHADING,
        name="Roof shading",
        description="Near-field shading of the roof (parapets, HVAC, neighbouring buildings).",
        category=IndicatorCategory.ROOFTOP,
        role=IndicatorRole.ROOFTOP_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ROOF_PROJECTS,
        preferred_source_types=(SourceType.ENGINEERING_DATASET, SourceType.CALLER_PROVIDED),
        required_or_optional=RequirementLevel.OPTIONAL,
        notes="Not inferred from footprint area.",
    ),
    define(
        IndicatorId.ROOF_STRUCTURAL_CAPACITY,
        name="Roof structural capacity",
        description=(
            "Structural load capacity of the roof. Must be externally provided unless a "
            "reliable engineering dataset exists. Never inferred from building footprint."
        ),
        category=IndicatorCategory.ROOFTOP,
        role=IndicatorRole.ROOFTOP_INDICATOR,
        controllability=Controllability.EXTERNALLY_PROVIDED,
        unit="kN/m2",
        optimization_direction=OptimizationDirection.NONE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ROOF_PROJECTS,
        preferred_source_types=(SourceType.ENGINEERING_DATASET, SourceType.CALLER_PROVIDED),
        required_or_optional=RequirementLevel.REQUIRED,
        provenance_notes="Engineering value only. Footprint geometry is not an allowed source.",
        notes="Absence of this value means structural feasibility is unknown, not assumed.",
    ),
)
