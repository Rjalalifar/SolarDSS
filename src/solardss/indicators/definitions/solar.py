"""Solar-resource vocabulary. Values must be supplied; SolarDSS does not estimate them."""

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
    POA_FROM_COMPONENTS,
    YIELD_FROM_RESOURCE_AND_DESIGN,
)
from solardss.indicators.factory import define
from solardss.indicators.ids import IndicatorId

_SOLAR_SOURCES = (
    SourceType.SATELLITE_SOLAR,
    SourceType.CLIMATE_REANALYSIS,
    SourceType.GROUND_STATION,
)

SOLAR_INDICATORS = (
    define(
        IndicatorId.SOLAR_GHI_ANNUAL,
        name="Annual Global Horizontal Irradiation",
        description=(
            "Annual GHI at the site or search cell. SolarDSS does not estimate "
            "irradiation; a meteorological provider value is required."
        ),
        category=IndicatorCategory.SOLAR_RESOURCE,
        role=IndicatorRole.SOLAR_RESOURCE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="kWh/m2/year",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_SOLAR_SOURCES,
        required_or_optional=RequirementLevel.REQUIRED,
        aliases=("ghi_kwh_m2",),
        notes="Canonical solar-resource screening indicator.",
    ),
    define(
        IndicatorId.SOLAR_DNI_ANNUAL,
        name="Annual Direct Normal Irradiation",
        description="Annual DNI from a solar-resource provider. Not computed by SolarDSS.",
        category=IndicatorCategory.SOLAR_RESOURCE,
        role=IndicatorRole.SOLAR_RESOURCE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="kWh/m2/year",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_SOLAR_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        notes="Relevant for tracking and for POA transposition when a method is supplied later.",
    ),
    define(
        IndicatorId.SOLAR_DHI_ANNUAL,
        name="Annual Diffuse Horizontal Irradiation",
        description="Annual DHI from a solar-resource provider. Not computed by SolarDSS.",
        category=IndicatorCategory.SOLAR_RESOURCE,
        role=IndicatorRole.SOLAR_RESOURCE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="kWh/m2/year",
        optimization_direction=OptimizationDirection.NONE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_SOLAR_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        notes="A resource component; not scored by itself as a siting benefit.",
    ),
    define(
        IndicatorId.SOLAR_POA_IRRADIATION,
        name="Plane-of-array irradiation",
        description=(
            "Irradiation on the array plane. Requires resource components and orientation. "
            "SolarDSS does not currently transpose GHI/DNI/DHI to POA."
        ),
        category=IndicatorCategory.SOLAR_RESOURCE,
        role=IndicatorRole.DERIVED_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="kWh/m2/year",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=(
            SourceType.SATELLITE_SOLAR,
            SourceType.DERIVED_FROM_REGISTERED_INDICATORS,
            SourceType.CALLER_PROVIDED,
        ),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=POA_FROM_COMPONENTS,
        notes="Supply a provider POA product or wait for a registered transposition method.",
    ),
    define(
        IndicatorId.SOLAR_SPECIFIC_YIELD,
        name="Expected energy production per installed kWp",
        description=(
            "Specific yield. Requires a resource and a specified system; SolarDSS does "
            "not run a PV performance model."
        ),
        category=IndicatorCategory.SOLAR_RESOURCE,
        role=IndicatorRole.DERIVED_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="kWh/kWp/year",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=(
            SourceType.SATELLITE_SOLAR,
            SourceType.DERIVED_FROM_REGISTERED_INDICATORS,
            SourceType.CALLER_PROVIDED,
        ),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=YIELD_FROM_RESOURCE_AND_DESIGN,
    ),
    define(
        IndicatorId.SOLAR_PVOUT,
        name="Expected photovoltaic output",
        description=(
            "Expected PV output as supplied by a PVOUT/yield product or a performance "
            "model outside SolarDSS."
        ),
        category=IndicatorCategory.SOLAR_RESOURCE,
        role=IndicatorRole.DERIVED_INDICATOR,
        controllability=Controllability.DERIVED,
        unit="kWh/kWp/year",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=(
            SourceType.SATELLITE_SOLAR,
            SourceType.CALLER_PROVIDED,
            SourceType.DERIVED_FROM_REGISTERED_INDICATORS,
        ),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=YIELD_FROM_RESOURCE_AND_DESIGN,
        notes="Do not treat PVOUT as measured irradiation.",
    ),
    define(
        IndicatorId.SOLAR_MONTHLY_VARIABILITY,
        name="Monthly/seasonal solar-resource variability",
        description="Seasonal variation of the solar resource when the provider supplies it.",
        category=IndicatorCategory.SOLAR_RESOURCE,
        role=IndicatorRole.SOLAR_RESOURCE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_SOLAR_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        notes="Unit is the provider's variability index; do not invent a coefficient of variation.",
    ),
    define(
        IndicatorId.SOLAR_INTERANNUAL_VARIABILITY,
        name="Interannual solar-resource variability",
        description="Year-to-year resource variability where a multi-year provider series exists.",
        category=IndicatorCategory.SOLAR_RESOURCE,
        role=IndicatorRole.SOLAR_RESOURCE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_SOLAR_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        notes="Omit rather than impute when the source has no multi-year record.",
    ),
    define(
        IndicatorId.SOLAR_SUNSHINE_DURATION,
        name="Sunshine duration",
        description="Sunshine-duration information only when a valid provider supplies it.",
        category=IndicatorCategory.SOLAR_RESOURCE,
        role=IndicatorRole.SOLAR_RESOURCE_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="h/year",
        optimization_direction=OptimizationDirection.MAXIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=(SourceType.GROUND_STATION, SourceType.CLIMATE_REANALYSIS),
        required_or_optional=RequirementLevel.OPTIONAL,
        notes="Not a substitute for GHI. Do not derive from cloud cover inside SolarDSS.",
    ),
)
