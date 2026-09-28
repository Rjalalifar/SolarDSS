"""Builder for registry entries. Keeps definition modules free of boilerplate."""

from __future__ import annotations

from solardss.indicators.contracts import (
    Controllability,
    DecisionMode,
    DerivationMethod,
    IndicatorCategory,
    IndicatorRole,
    IndicatorSpec,
    ModeContext,
    OptimizationDirection,
    ProjectType,
    ProvenanceRequirements,
    RequirementLevel,
    SiteSelectionScope,
    SourceType,
)
from solardss.indicators.defaults import ALL_SCOPES, provenance_for
from solardss.indicators.ids import IndicatorId


def define(
    indicator_id: IndicatorId,
    *,
    name: str,
    description: str,
    category: IndicatorCategory,
    role: IndicatorRole,
    controllability: Controllability,
    unit: str,
    optimization_direction: OptimizationDirection,
    applicable_decision_modes: tuple[DecisionMode, ...],
    applicable_project_types: tuple[ProjectType, ...],
    preferred_source_types: tuple[SourceType, ...],
    required_or_optional: RequirementLevel,
    derivation_method: DerivationMethod | None = None,
    provenance_requirements: ProvenanceRequirements | None = None,
    notes: str = "",
    aliases: tuple[str, ...] = (),
    applicable_site_selection_scopes: tuple[SiteSelectionScope, ...] | None = None,
    mode_contexts: tuple[ModeContext, ...] = (),
    distinguish_proxy: bool = False,
    provenance_notes: str = "",
) -> IndicatorSpec:
    """Construct a frozen indicator spec with default provenance and scopes."""

    if applicable_site_selection_scopes is None:
        scopes = ALL_SCOPES if DecisionMode.SITE_SELECTION in applicable_decision_modes else ()
    else:
        scopes = applicable_site_selection_scopes
    provenance = provenance_requirements or provenance_for(
        controllability,
        distinguish_proxy=distinguish_proxy,
        notes=provenance_notes,
    )
    return IndicatorSpec(
        id=indicator_id.value,
        name=name,
        description=description,
        category=category,
        role=role,
        controllability=controllability,
        unit=unit,
        optimization_direction=optimization_direction,
        applicable_decision_modes=applicable_decision_modes,
        applicable_project_types=applicable_project_types,
        preferred_source_types=preferred_source_types,
        required_or_optional=required_or_optional,
        derivation_method=derivation_method,
        provenance_requirements=provenance,
        notes=notes,
        aliases=aliases,
        applicable_site_selection_scopes=scopes,
        mode_contexts=mode_contexts,
    )
