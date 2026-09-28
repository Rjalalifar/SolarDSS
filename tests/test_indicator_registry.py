"""Scientific indicator registry: vocabulary, not data acquisition."""

from __future__ import annotations

import pytest

from solardss.exceptions import UnknownIndicatorError
from solardss.indicators import (
    Controllability,
    DecisionMode,
    IndicatorCategory,
    IndicatorId,
    IndicatorRole,
    IndicatorSpec,
    OptimizationDirection,
    ProjectType,
    RequirementLevel,
    SiteSelectionScope,
    SourceType,
    ValueProvenanceKind,
    catalog_ids,
    get_indicator,
    registry,
)
from solardss.indicators.definitions import ALL_INDICATORS
from solardss.indicators.errors import IndicatorRegistryError


def test_registry_ids_are_unique_and_match_enum() -> None:
    ids = [spec.id for spec in registry.list_all()]
    assert len(ids) == len(set(ids))
    assert ids == sorted(ids)
    assert set(ids) == {member.value for member in IndicatorId}
    assert len(ALL_INDICATORS) == len(IndicatorId)
    assert catalog_ids() == tuple(ids)


def test_aliases_do_not_collide_with_ids_or_each_other() -> None:
    ids = {spec.id for spec in registry.list_all()}
    aliases = [alias for spec in registry.list_all() for alias in spec.aliases]
    assert len(aliases) == len(set(aliases))
    assert ids.isdisjoint(aliases)


def test_no_duplicated_definitions() -> None:
    names = [spec.name for spec in registry.list_all()]
    assert len(names) == len(set(names))
    assert sum(spec.id == "PV_LOCATION" for spec in registry.list_all()) == 1
    assert sum(spec.id == "SOLAR_GHI_ANNUAL" for spec in ALL_INDICATORS) == 1


def test_every_spec_has_complete_metadata() -> None:
    for spec in registry.list_all():
        assert isinstance(spec, IndicatorSpec)
        assert spec.id
        assert spec.name
        assert spec.description
        assert spec.unit
        assert spec.applicable_decision_modes
        assert spec.applicable_project_types
        assert spec.preferred_source_types
        assert spec.provenance_requirements.allowed_value_kinds
        if DecisionMode.SITE_SELECTION in spec.applicable_decision_modes:
            assert spec.applicable_site_selection_scopes


def test_ghi_unit_and_alias() -> None:
    spec = registry.get("SOLAR_GHI_ANNUAL")
    assert spec.unit == "kWh/m2/year"
    assert spec.optimization_direction is OptimizationDirection.MAXIMIZE
    assert spec.computed_by_solardss is False
    assert get_indicator("ghi_kwh_m2") is spec
    assert get_indicator(IndicatorId.SOLAR_GHI_ANNUAL) is spec


def test_slope_is_horn_derived() -> None:
    spec = get_indicator("slope_deg")
    assert spec.id == "TERRAIN_SLOPE_MEAN"
    assert spec.unit == "deg"
    assert spec.computed_by_solardss is True
    assert spec.derivation_method is not None
    assert spec.derivation_method.id == "horn_1981_slope_aspect"


def test_soiling_distinguishes_proxy_from_measurement() -> None:
    spec = registry.get("CLIMATE_SOILING_RISK")
    req = spec.provenance_requirements
    assert req.distinguish_proxy_from_measurement is True
    assert ValueProvenanceKind.MEASURED_OR_PROVIDER in req.allowed_value_kinds
    assert ValueProvenanceKind.DERIVED_PROXY in req.allowed_value_kinds
    assert spec.computed_by_solardss is False


def test_hosting_capacity_is_externally_provided() -> None:
    spec = registry.get("INFRA_GRID_HOSTING_CAPACITY")
    assert spec.controllability is Controllability.EXTERNALLY_PROVIDED
    assert SourceType.OSM_OR_NETWORK_VECTOR not in spec.preferred_source_types
    assert SourceType.GRID_OPERATOR in spec.preferred_source_types
    assert spec.computed_by_solardss is False


def test_structural_capacity_is_not_inferred_from_footprint() -> None:
    spec = registry.get("ROOF_STRUCTURAL_CAPACITY")
    assert spec.controllability is Controllability.EXTERNALLY_PROVIDED
    assert SourceType.BUILDING_FOOTPRINT not in spec.preferred_source_types
    assert SourceType.ENGINEERING_DATASET in spec.preferred_source_types
    assert spec.required_or_optional is RequirementLevel.REQUIRED


def test_by_category() -> None:
    solar = registry.by_category(IndicatorCategory.SOLAR_RESOURCE)
    assert {spec.id for spec in solar} >= {
        "SOLAR_GHI_ANNUAL",
        "SOLAR_DNI_ANNUAL",
        "SOLAR_POA_IRRADIATION",
    }
    assert all(spec.category is IndicatorCategory.SOLAR_RESOURCE for spec in solar)


def test_for_mode_and_scope() -> None:
    site = {spec.id for spec in registry.for_mode(DecisionMode.SITE_SELECTION)}
    fixed = {spec.id for spec in registry.for_mode(DecisionMode.FIXED_SITE)}
    assert "PV_LOCATION" in site
    assert "PV_LOCATION" in fixed
    assert "LAND_AVAILABLE_AREA" in site
    assert "LAND_AVAILABLE_AREA" not in fixed
    assert "OBJECTIVE_MIN_GRID_CONNECTION_DISTANCE" in site
    ranking = {
        spec.id
        for spec in registry.for_mode(
            DecisionMode.SITE_SELECTION,
            scope=SiteSelectionScope.CANDIDATE_RANKING,
        )
    }
    assert "SOLAR_GHI_ANNUAL" in ranking


def test_project_type_filtering() -> None:
    residential = {spec.id for spec in registry.for_project_type(ProjectType.RESIDENTIAL_ROOFTOP)}
    utility = {spec.id for spec in registry.for_project_type(ProjectType.UTILITY_SCALE)}
    assert "ROOF_USABLE_AREA" in residential
    assert "ROOF_STRUCTURAL_CAPACITY" in residential
    assert "LAND_AVAILABLE_AREA" not in residential
    assert "PV_ROW_SPACING" not in residential
    assert "LAND_AVAILABLE_AREA" in utility
    assert "TERRAIN_SLOPE_MEAN" in utility
    assert "ROOF_USABLE_AREA" not in utility


def test_required_for_utility_site_selection() -> None:
    required = {
        spec.id
        for spec in registry.required_for(
            DecisionMode.SITE_SELECTION,
            ProjectType.UTILITY_SCALE,
        )
    }
    assert required >= {
        "SOLAR_GHI_ANNUAL",
        "TERRAIN_SLOPE_MEAN",
        "LAND_AVAILABLE_AREA",
        "LAND_PROTECTED_AREA_OVERLAP",
        "PV_LOCATION",
        "PV_SYSTEM_CAPACITY",
    }
    assert "CLIMATE_SOILING_RISK" not in required
    assert "INFRA_GRID_HOSTING_CAPACITY" not in required


def test_pv_location_mode_resolution() -> None:
    base = registry.get("PV_LOCATION")
    site = registry.resolve("PV_LOCATION", mode=DecisionMode.SITE_SELECTION)
    fixed = registry.resolve("PV_LOCATION", mode=DecisionMode.FIXED_SITE)
    assert site.definition is base
    assert fixed.definition is base
    assert site.id == "PV_LOCATION"
    assert site.role is IndicatorRole.DECISION_VARIABLE
    assert site.controllability is Controllability.CONTROLLABLE
    assert fixed.role is IndicatorRole.CONTEXT
    assert fixed.controllability is Controllability.FIXED
    assert site.required_or_optional is RequirementLevel.REQUIRED
    assert fixed.required_or_optional is RequirementLevel.REQUIRED


def test_resolve_rejects_inapplicable_mode() -> None:
    with pytest.raises(IndicatorRegistryError):
        registry.resolve("LAND_AVAILABLE_AREA", mode=DecisionMode.FIXED_SITE)


def test_unknown_indicator() -> None:
    with pytest.raises(UnknownIndicatorError):
        registry.get("made_up_irradiance_model")
