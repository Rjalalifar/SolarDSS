"""Hazard vocabulary. Risk layers are supplied; SolarDSS does not model hazards."""

from __future__ import annotations

from solardss.indicators.contracts import (
    Controllability,
    IndicatorCategory,
    IndicatorRole,
    OptimizationDirection,
    ProvenanceRequirements,
    RequirementLevel,
    SourceType,
    ValueProvenanceKind,
)
from solardss.indicators.defaults import ALL_MODES, ALL_PROJECTS, HAZARD_SOURCE_MAPPING
from solardss.indicators.factory import define
from solardss.indicators.ids import IndicatorId

_HAZARD_SOURCES = (SourceType.HAZARD_MAP, SourceType.CALLER_PROVIDED)
_HAZARD_NOTES = (
    "Sampled from a caller-supplied map. Categories such as low/medium/high require "
    "that dataset's explicit legend; SolarDSS does not invent thresholds or model the hazard."
)
_HAZARD_PROVENANCE = ProvenanceRequirements(
    allowed_value_kinds=(
        ValueProvenanceKind.MEASURED_OR_PROVIDER,
        ValueProvenanceKind.DERIVED_METHOD,
        ValueProvenanceKind.DERIVED_PROXY,
    ),
    source_citation_required=True,
    distinguish_proxy_from_measurement=True,
    notes=(
        "Record whether the value is a sampled map cell, a mapped legend class, "
        "or a proxy such as water proximity or thunderstorm counts."
    ),
)

HAZARD_INDICATORS = (
    define(
        IndicatorId.HAZARD_FLOOD_RISK,
        name="Flood risk",
        description="Flood hazard or risk class from a hazard map. Not modelled by SolarDSS.",
        category=IndicatorCategory.HAZARD,
        role=IndicatorRole.HAZARD_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_HAZARD_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=HAZARD_SOURCE_MAPPING,
        provenance_requirements=_HAZARD_PROVENANCE,
        provenance_notes="A flood-plain class is a provider value; proximity to water is a proxy.",
        notes=_HAZARD_NOTES,
    ),
    define(
        IndicatorId.HAZARD_LANDSLIDE_RISK,
        name="Landslide risk",
        description="Landslide hazard or risk class from a hazard map.",
        category=IndicatorCategory.HAZARD,
        role=IndicatorRole.HAZARD_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_HAZARD_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=HAZARD_SOURCE_MAPPING,
        provenance_requirements=_HAZARD_PROVENANCE,
        provenance_notes=(
            "Slope is not landslide risk unless a published susceptibility model is cited."
        ),
        notes=_HAZARD_NOTES,
    ),
    define(
        IndicatorId.HAZARD_WILDFIRE_RISK,
        name="Wildfire risk",
        description="Wildfire hazard or risk class from a hazard map.",
        category=IndicatorCategory.HAZARD,
        role=IndicatorRole.HAZARD_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=_HAZARD_SOURCES,
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=HAZARD_SOURCE_MAPPING,
        provenance_requirements=_HAZARD_PROVENANCE,
        notes=_HAZARD_NOTES,
    ),
    define(
        IndicatorId.HAZARD_HAIL_RISK,
        name="Hail risk",
        description="Hail hazard or risk from a climate or hazard provider.",
        category=IndicatorCategory.HAZARD,
        role=IndicatorRole.HAZARD_INDICATOR,
        controllability=Controllability.UNCONTROLLABLE,
        unit="1",
        optimization_direction=OptimizationDirection.MINIMIZE,
        applicable_decision_modes=ALL_MODES,
        applicable_project_types=ALL_PROJECTS,
        preferred_source_types=(SourceType.HAZARD_MAP, SourceType.CLIMATE_REANALYSIS),
        required_or_optional=RequirementLevel.OPTIONAL,
        derivation_method=HAZARD_SOURCE_MAPPING,
        provenance_requirements=_HAZARD_PROVENANCE,
        provenance_notes=(
            "A hail climatology product is a provider value; thunderstorm counts are a proxy."
        ),
        notes=_HAZARD_NOTES,
    ),
)
