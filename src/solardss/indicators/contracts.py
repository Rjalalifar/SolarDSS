"""Indicator and decision-variable contracts.

The registry is a vocabulary: it describes indicators, it does not acquire data
or compute physical values. ``Polarity`` remains the computational orientation
used by min-max normalization and is derived from ``OptimizationDirection``.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DecisionMode(str, Enum):
    """Whether the site is chosen by the DSS or already fixed."""

    SITE_SELECTION = "site_selection"
    FIXED_SITE = "fixed_site"


class SiteSelectionScope(str, Enum):
    """Spatial grain of a site-selection study."""

    REGION_SEARCH = "region_search"
    AOI_SEARCH = "aoi_search"
    CANDIDATE_RANKING = "candidate_ranking"


class ProjectType(str, Enum):
    """PV deployment class. Rooftop subtypes are distinct from utility/ground."""

    UTILITY_SCALE = "utility_scale"
    GROUND_MOUNTED = "ground_mounted"
    ROOFTOP = "rooftop"
    COMMERCIAL_ROOFTOP = "commercial_rooftop"
    RESIDENTIAL_ROOFTOP = "residential_rooftop"


class IndicatorCategory(str, Enum):
    """Section of the SolarDSS vocabulary (not a computed value)."""

    SOLAR_RESOURCE = "solar_resource"
    CLIMATE = "climate"
    TERRAIN = "terrain"
    LAND = "land"
    INFRASTRUCTURE = "infrastructure"
    HAZARD = "hazard"
    ROOFTOP = "rooftop"
    PV_DESIGN = "pv_design"
    OBJECTIVE = "objective"


class IndicatorRole(str, Enum):
    """How a registered item participates in a PV decision problem."""

    ENVIRONMENTAL_INDICATOR = "environmental_indicator"
    SOLAR_RESOURCE_INDICATOR = "solar_resource_indicator"
    CLIMATE_INDICATOR = "climate_indicator"
    TERRAIN_INDICATOR = "terrain_indicator"
    LAND_INDICATOR = "land_indicator"
    INFRASTRUCTURE_INDICATOR = "infrastructure_indicator"
    HAZARD_INDICATOR = "hazard_indicator"
    ROOFTOP_INDICATOR = "rooftop_indicator"
    CONSTRAINT = "constraint"
    DERIVED_INDICATOR = "derived_indicator"
    DECISION_VARIABLE = "decision_variable"
    OBJECTIVE = "objective"
    CONTEXT = "context"


class Controllability(str, Enum):
    """Whether a planner can choose the quantity, or must take it as given."""

    UNCONTROLLABLE = "uncontrollable"
    CONTROLLABLE = "controllable"
    FIXED = "fixed"
    DERIVED = "derived"
    EXTERNALLY_PROVIDED = "externally_provided"


class OptimizationDirection(str, Enum):
    """Preferred orientation if the item is used as a score or objective."""

    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"
    TARGET = "target"
    NONE = "none"


class Polarity(str, Enum):
    """Computational orientation for min-max normalization.

    Derived from ``OptimizationDirection``: maximize → benefit, minimize → cost.
    ``NONE`` covers target and non-scored items.
    """

    BENEFIT = "benefit"
    COST = "cost"
    NONE = "none"


class RequirementLevel(str, Enum):
    """Whether a value must be supplied before a mode/project can proceed."""

    REQUIRED = "required"
    OPTIONAL = "optional"
    CONTEXT_DEPENDENT = "context_dependent"


class SourceType(str, Enum):
    """Preferred *kinds* of source. The registry never opens these sources."""

    SATELLITE_SOLAR = "satellite_solar"
    CLIMATE_REANALYSIS = "climate_reanalysis"
    GROUND_STATION = "ground_station"
    DEM = "dem"
    LAND_COVER_MAP = "land_cover_map"
    PROTECTED_AREA_CADASTRE = "protected_area_cadastre"
    OSM_OR_NETWORK_VECTOR = "osm_or_network_vector"
    GRID_OPERATOR = "grid_operator"
    HAZARD_MAP = "hazard_map"
    BUILDING_FOOTPRINT = "building_footprint"
    ENGINEERING_DATASET = "engineering_dataset"
    CADASTRAL_OR_LAND_ADMIN = "cadastral_or_land_admin"
    CALLER_PROVIDED = "caller_provided"
    DERIVED_FROM_REGISTERED_INDICATORS = "derived_from_registered_indicators"


class ValueProvenanceKind(str, Enum):
    """How a concrete value may be obtained. Distinct from controllability."""

    MEASURED_OR_PROVIDER = "measured_or_provider"
    DERIVED_PROXY = "derived_proxy"
    DERIVED_METHOD = "derived_method"
    EXTERNALLY_PROVIDED = "externally_provided"


class DerivationMethod(BaseModel):
    """Named derivation, when the indicator is not a raw provider field.

    ``implemented_in_solardss`` is true only if this package currently computes
    the quantity. A documented method that is not implemented still requires
    an externally supplied value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    description: str
    implemented_in_solardss: bool = False


class ProvenanceRequirements(BaseModel):
    """What must be recorded when a value is attached to this indicator."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed_value_kinds: tuple[ValueProvenanceKind, ...] = Field(min_length=1)
    source_citation_required: bool = True
    distinguish_proxy_from_measurement: bool = False
    notes: str = ""


class ModeContext(BaseModel):
    """Mode-specific role/controllability overlay. Does not duplicate the id."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: DecisionMode
    role: IndicatorRole
    controllability: Controllability
    required_or_optional: RequirementLevel | None = None


def polarity_from_direction(direction: OptimizationDirection) -> Polarity:
    """Map an optimization direction onto min-max polarity."""

    if direction is OptimizationDirection.MAXIMIZE:
        return Polarity.BENEFIT
    if direction is OptimizationDirection.MINIMIZE:
        return Polarity.COST
    return Polarity.NONE


class IndicatorSpec(BaseModel):
    """One registered indicator, constraint, decision variable, or objective.

    This is metadata only. Presence in the registry does not mean SolarDSS can
    acquire or compute the value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    category: IndicatorCategory
    role: IndicatorRole
    controllability: Controllability
    unit: str = Field(min_length=1)
    optimization_direction: OptimizationDirection
    applicable_decision_modes: tuple[DecisionMode, ...] = Field(min_length=1)
    applicable_project_types: tuple[ProjectType, ...] = Field(min_length=1)
    preferred_source_types: tuple[SourceType, ...] = Field(min_length=1)
    required_or_optional: RequirementLevel
    derivation_method: DerivationMethod | None = None
    provenance_requirements: ProvenanceRequirements
    notes: str = ""
    aliases: tuple[str, ...] = ()
    applicable_site_selection_scopes: tuple[SiteSelectionScope, ...] = ()
    mode_contexts: tuple[ModeContext, ...] = ()

    @model_validator(mode="after")
    def _scopes_match_modes(self) -> IndicatorSpec:
        has_site_selection = DecisionMode.SITE_SELECTION in self.applicable_decision_modes
        if has_site_selection and not self.applicable_site_selection_scopes:
            msg = f"{self.id} applies to site_selection but lists no site-selection scopes"
            raise ValueError(msg)
        if not has_site_selection and self.applicable_site_selection_scopes:
            msg = f"{self.id} lists site-selection scopes without site_selection mode"
            raise ValueError(msg)
        modes = tuple(ctx.mode for ctx in self.mode_contexts)
        if len(modes) != len(set(modes)):
            msg = f"{self.id} has duplicate mode_contexts"
            raise ValueError(msg)
        for ctx in self.mode_contexts:
            if ctx.mode not in self.applicable_decision_modes:
                msg = f"{self.id} has a mode context for {ctx.mode.value} which is not applicable"
                raise ValueError(msg)
        return self

    @property
    def polarity(self) -> Polarity:
        """Min-max polarity implied by ``optimization_direction``."""

        return polarity_from_direction(self.optimization_direction)

    @property
    def computed_by_solardss(self) -> bool:
        """True only when a SolarDSS method currently implements the derivation."""

        return self.derivation_method is not None and self.derivation_method.implemented_in_solardss

    @property
    def title(self) -> str:
        """Alias of ``name`` for the original catalog field."""

        return self.name


class ResolvedIndicator(BaseModel):
    """An ``IndicatorSpec`` after applying decision-mode context.

    ``definition`` is the single registered object. Role and controllability may
    differ from the base spec when ``mode_contexts`` are defined.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    definition: IndicatorSpec
    mode: DecisionMode
    project_type: ProjectType | None = None
    role: IndicatorRole
    controllability: Controllability
    required_or_optional: RequirementLevel

    @property
    def id(self) -> str:
        return self.definition.id

    @property
    def polarity(self) -> Polarity:
        return self.definition.polarity

    @property
    def computed_by_solardss(self) -> bool:
        return self.definition.computed_by_solardss
