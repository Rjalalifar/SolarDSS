"""Shared derivation methods and provenance defaults for registry definitions."""

from __future__ import annotations

from solardss.indicators.contracts import (
    Controllability,
    DecisionMode,
    DerivationMethod,
    ProjectType,
    ProvenanceRequirements,
    SiteSelectionScope,
    ValueProvenanceKind,
)

ALL_MODES: tuple[DecisionMode, ...] = (
    DecisionMode.SITE_SELECTION,
    DecisionMode.FIXED_SITE,
)
ALL_SCOPES: tuple[SiteSelectionScope, ...] = (
    SiteSelectionScope.REGION_SEARCH,
    SiteSelectionScope.AOI_SEARCH,
    SiteSelectionScope.CANDIDATE_RANKING,
)
GROUND_PROJECTS: tuple[ProjectType, ...] = (
    ProjectType.UTILITY_SCALE,
    ProjectType.GROUND_MOUNTED,
)
ROOF_PROJECTS: tuple[ProjectType, ...] = (
    ProjectType.ROOFTOP,
    ProjectType.COMMERCIAL_ROOFTOP,
    ProjectType.RESIDENTIAL_ROOFTOP,
)
ALL_PROJECTS: tuple[ProjectType, ...] = GROUND_PROJECTS + ROOF_PROJECTS

HORN_SLOPE_ASPECT = DerivationMethod(
    id="horn_1981_slope_aspect",
    description="Horn (1981) 8-neighbor slope and aspect on a regular elevation grid.",
    implemented_in_solardss=True,
)

POA_FROM_COMPONENTS = DerivationMethod(
    id="poa_from_components_and_orientation",
    description=(
        "Plane-of-array irradiation from GHI/DNI/DHI and array orientation. "
        "SolarDSS does not currently compute this transposition."
    ),
    implemented_in_solardss=False,
)

YIELD_FROM_RESOURCE_AND_DESIGN = DerivationMethod(
    id="yield_from_resource_and_design",
    description=(
        "Specific yield / PVOUT from a supplied resource and a fully specified "
        "PV design. SolarDSS does not currently run a PV performance model."
    ),
    implemented_in_solardss=False,
)

TERRAIN_FROM_DEM = DerivationMethod(
    id="terrain_from_dem",
    description="Derived from a caller-supplied DEM. Not implemented unless noted.",
    implemented_in_solardss=False,
)

RILEY_TRI = DerivationMethod(
    id="riley_1999_terrain_ruggedness_index",
    description=(
        "Riley et al. (1999) TRI: sqrt of the sum of squared elevation "
        "differences to the 8 neighbors, in metres."
    ),
    implemented_in_solardss=True,
)

DEM_ELEVATION_ZONAL = DerivationMethod(
    id="dem_elevation_zonal",
    description="Zonal mean/median/max/percentiles of DEM elevation over a mask.",
    implemented_in_solardss=True,
)

LAND_FRACTION_FROM_COVER = DerivationMethod(
    id="land_fraction_from_cover",
    description=(
        "Class-area fraction from a land-cover map: category area divided by "
        "valid (non-nodata) area using projected cell area or WGS84 authalic "
        "trapezoids on geographic grids."
    ),
    implemented_in_solardss=True,
)

LAND_DOMINANT_FROM_COVER = DerivationMethod(
    id="land_dominant_class",
    description=(
        "Dominant land-cover category by area (not pixel count when cell areas "
        "differ). Ties break to the lowest dataset class code."
    ),
    implemented_in_solardss=True,
)

LAND_AVAILABLE_AREA_PROFILE = DerivationMethod(
    id="land_available_area_profile",
    description=(
        "Available land area from an explicit profile of available and excluded "
        "cover classes, optionally intersected with a caller constraint mask. "
        "Non-water pixels are not treated as available unless the profile says so."
    ),
    implemented_in_solardss=True,
)

ROOF_USABLE_FROM_FOOTPRINT = DerivationMethod(
    id="roof_usable_from_geometry_and_setbacks",
    description=(
        "Usable roof area from footprint, slope, setbacks, and shading. "
        "Not inferred from footprint alone."
    ),
    implemented_in_solardss=False,
)

OSM_NEAREST_DISTANCE = DerivationMethod(
    id="osm_nearest_distance",
    description=(
        "Nearest-feature distance on a projected Euclidean plane or a WGS84 "
        "authalic equirectangular plane centred on the origin. Empty extracts "
        "stay empty. Not network distance and not hosting capacity."
    ),
    implemented_in_solardss=True,
)

OSM_FOOTPRINT_AREA = DerivationMethod(
    id="osm_footprint_area",
    description=(
        "Plan-view building footprint area from a closed OSM polygon using "
        "projected shoelace or WGS84 authalic trapezoids. Not usable PV area."
    ),
    implemented_in_solardss=True,
)

OSM_ROOF_TAG_ORIENTATION = DerivationMethod(
    id="osm_roof_tag_orientation",
    description=(
        "Roof azimuth from OSM roof:direction when the tag is a compass point "
        "or a degree value. Missing or unparseable tags stay missing."
    ),
    implemented_in_solardss=True,
)

OSM_ROOF_TAG_SLOPE = DerivationMethod(
    id="osm_roof_tag_slope",
    description=(
        "Roof slope from OSM roof:angle, or 0° when roof:shape=flat. Other "
        "shapes and missing tags do not invent a slope."
    ),
    implemented_in_solardss=True,
)

PROTECTED_AREA_OVERLAP = DerivationMethod(
    id="protected_area_overlap",
    description=(
        "Intersection of the AOI with a caller-supplied protected-area cadastre. "
        "Hard exclusion, penalty, or warning come only from an explicit "
        "ProtectedAreaConstraintProfile. Missing profiles stay informational. "
        "SolarDSS does not assume a global unrestricted dataset."
    ),
    implemented_in_solardss=True,
)

HAZARD_SOURCE_MAPPING = DerivationMethod(
    id="hazard_source_mapping",
    description=(
        "Sample a caller-supplied hazard raster or vector and apply that dataset's "
        "explicit legend. SolarDSS does not model flood, wildfire, landslide, or hail, "
        "and does not invent Low/Medium/High thresholds."
    ),
    implemented_in_solardss=True,
)


def provenance_for(
    controllability: Controllability,
    *,
    distinguish_proxy: bool = False,
    notes: str = "",
) -> ProvenanceRequirements:
    """Default allowed value kinds for a controllability class."""

    if distinguish_proxy:
        return ProvenanceRequirements(
            allowed_value_kinds=(
                ValueProvenanceKind.MEASURED_OR_PROVIDER,
                ValueProvenanceKind.DERIVED_PROXY,
            ),
            source_citation_required=True,
            distinguish_proxy_from_measurement=True,
            notes=notes or "Record whether the value is a provider measurement or a derived proxy.",
        )
    if controllability is Controllability.DERIVED:
        return ProvenanceRequirements(
            allowed_value_kinds=(
                ValueProvenanceKind.DERIVED_METHOD,
                ValueProvenanceKind.EXTERNALLY_PROVIDED,
            ),
            source_citation_required=True,
            notes=notes,
        )
    if controllability is Controllability.CONTROLLABLE:
        return ProvenanceRequirements(
            allowed_value_kinds=(ValueProvenanceKind.EXTERNALLY_PROVIDED,),
            source_citation_required=False,
            notes=notes or "Caller-selected decision variable.",
        )
    if controllability is Controllability.FIXED:
        return ProvenanceRequirements(
            allowed_value_kinds=(
                ValueProvenanceKind.EXTERNALLY_PROVIDED,
                ValueProvenanceKind.MEASURED_OR_PROVIDER,
            ),
            source_citation_required=True,
            notes=notes,
        )
    if controllability is Controllability.EXTERNALLY_PROVIDED:
        return ProvenanceRequirements(
            allowed_value_kinds=(ValueProvenanceKind.EXTERNALLY_PROVIDED,),
            source_citation_required=True,
            notes=notes,
        )
    return ProvenanceRequirements(
        allowed_value_kinds=(
            ValueProvenanceKind.MEASURED_OR_PROVIDER,
            ValueProvenanceKind.EXTERNALLY_PROVIDED,
        ),
        source_citation_required=True,
        notes=notes,
    )
