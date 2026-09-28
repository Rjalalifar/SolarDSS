"""Land and environment vocabulary. Classification maps are explicit, not inferred."""

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
from solardss.indicators.defaults import (
    ALL_MODES,
    GROUND_PROJECTS,
    LAND_AVAILABLE_AREA_PROFILE,
    LAND_DOMINANT_FROM_COVER,
    LAND_FRACTION_FROM_COVER,
    PROTECTED_AREA_OVERLAP,
)
from solardss.indicators.factory import define
from solardss.indicators.ids import IndicatorId

LAND_INDICATORS = (
    define(
        IndicatorId.LAND_COVER,
        name="Land cover",
        description=(
            "Dominant land-cover category in the AOI or candidate, taken from a "
            "land-cover map. SolarDSS does not classify imagery."
        ),
        category=IndicatorCategory.LAND,
        role=IndicatorRole.LAND_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="categorical",
        optimization_direction=OptimizationDirection.NONE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(SourceType.LAND_COVER_MAP,),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=LAND_DOMINANT_FROM_COVER,
    ),
    define(
        IndicatorId.LAND_AVAILABLE_AREA,
        name="Available land area",
        description=(
            "Area remaining after an explicit available/excluded class profile "
            "and optional extra constraint masks. Non-water is not automatically available."
        ),
        category=IndicatorCategory.LAND,
        role=IndicatorRole.LAND_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="m2",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=(DecisionMode.SITE_SELECTION,),
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(
            SourceType.LAND_COVER_MAP,
            SourceType.CADASTRAL_OR_LAND_ADMIN,
            SourceType.DERIVED_FROM_REGISTERED_INDICATORS,
            SourceType.CALLER_PROVIDED,
        ),
        required_or_optional=RequirementLevel.REQUIRED,
        derivation_method=LAND_AVAILABLE_AREA_PROFILE,
        notes=(
            "Requires an AvailableLandProfile (available classes + excluded classes). "
            "SolarDSS never treats every non-water pixel as PV-suitable."
        ),
    ),
    define(
        IndicatorId.LAND_FOREST_FRACTION,
        name="Forest fraction",
        description="Fraction of valid AOI area mapped to the forest category.",
        category=IndicatorCategory.LAND,
        role=IndicatorRole.LAND_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(SourceType.LAND_COVER_MAP,),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=LAND_FRACTION_FROM_COVER,
    ),
    define(
        IndicatorId.LAND_WATER_FRACTION,
        name="Water fraction",
        description="Fraction of valid AOI area mapped to the water category.",
        category=IndicatorCategory.LAND,
        role=IndicatorRole.LAND_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(SourceType.LAND_COVER_MAP,),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=LAND_FRACTION_FROM_COVER,
        notes="Boolean water exclusion is a constraint mask, not this fraction.",
    ),
    define(
        IndicatorId.LAND_WETLAND_FRACTION,
        name="Wetland fraction",
        description="Fraction of valid AOI area mapped to the wetland category.",
        category=IndicatorCategory.LAND,
        role=IndicatorRole.LAND_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(SourceType.LAND_COVER_MAP,),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=LAND_FRACTION_FROM_COVER,
    ),
    define(
        IndicatorId.LAND_AGRICULTURAL_FRACTION,
        name="Agricultural fraction",
        description="Fraction of valid AOI area mapped to the agricultural category.",
        category=IndicatorCategory.LAND,
        role=IndicatorRole.LAND_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="1",
        optimization_direction=OptimizationDirection.NONE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(SourceType.LAND_COVER_MAP,),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=LAND_FRACTION_FROM_COVER,
    ),
    define(
        IndicatorId.LAND_BUILTUP_FRACTION,
        name="Built-up fraction",
        description="Fraction of valid AOI area mapped to the built-up category.",
        category=IndicatorCategory.LAND,
        role=IndicatorRole.LAND_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(SourceType.LAND_COVER_MAP, SourceType.BUILDING_FOOTPRINT),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=LAND_FRACTION_FROM_COVER,
    ),
    define(
        IndicatorId.LAND_BARE_LAND_FRACTION,
        name="Bare-land fraction",
        description="Fraction of valid AOI area mapped to bare land or sparse vegetation.",
        category=IndicatorCategory.LAND,
        role=IndicatorRole.LAND_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="1",
        optimization_direction=OptimizationDirection.NONE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(SourceType.LAND_COVER_MAP,),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=LAND_FRACTION_FROM_COVER,
    ),
    define(
        IndicatorId.LAND_PROTECTED_AREA_OVERLAP,
        name="Protected-area overlap",
        description=(
            "Overlap fraction of the AOI with designated protected areas from a "
            "region-specific cadastre. Not inferred from land cover."
        ),
        category=IndicatorCategory.LAND,
        role=IndicatorRole.ENVIRONMENTAL_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=GROUND_PROJECTS,
        preferred_source_types=(SourceType.PROTECTED_AREA_CADASTRE,),
        required_or_optional=RequirementLevel.REQUIRED,
        derivation_method=PROTECTED_AREA_OVERLAP,
        notes=(
            "Intersection area, percent, site identifiers, protection categories, "
            "and source provenance are recorded on the observation. Whether overlap "
            "is a hard exclusion, penalty, or warning comes from a methodology/"
            "legal ProtectedAreaConstraintProfile. Vegetation is not protection status. "
            "SolarDSS does not assume a global unrestricted dataset."
        ),
    ),
)
