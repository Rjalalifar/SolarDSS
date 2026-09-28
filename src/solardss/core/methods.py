"""Stable method identifiers and versions recorded in provenance.

A method version changes only when the numerical definition changes. Adding
warnings or extra metadata does not bump the version.
"""

from __future__ import annotations

from enum import Enum


class MethodId(str, Enum):
    """Identifiers for implemented, deterministic methods."""

    MINMAX_NORMALIZE = "minmax_normalize"
    WEIGHTED_LINEAR_COMBINATION = "weighted_linear_combination"
    AHP_GEOMETRIC_MEAN = "ahp_geometric_mean"
    BOOLEAN_EXCLUSION = "boolean_exclusion"
    CANDIDATES_FROM_MASK = "candidates_from_mask"
    CANDIDATES_REGULAR_GRID = "candidates_regular_grid"
    CANDIDATES_HEXAGONAL_GRID = "candidates_hexagonal_grid"
    CANDIDATES_FROM_FEATURES = "candidates_from_features"
    CANDIDATES_CONTIGUOUS_GROUPS = "candidates_contiguous_groups"
    RETAIN_FEASIBLE = "retain_feasible"
    GREEDY_BUDGET_SELECTION = "greedy_budget_selection"
    HORN_SLOPE_ASPECT = "horn_slope_aspect"
    RILEY_TRI = "riley_tri"
    DEM_ELEVATION_ZONAL = "dem_elevation_zonal"
    CAPACITY_FACTOR_ENERGY = "capacity_factor_energy"
    IRRADIATION_AREA_EFFICIENCY_ENERGY = "irradiation_area_efficiency_energy"
    LAND_CLASS_AREA_FRACTION = "land_class_area_fraction"
    LAND_DOMINANT_CLASS = "land_dominant_class"
    LAND_AVAILABLE_AREA_PROFILE = "land_available_area_profile"
    OSM_NEAREST_DISTANCE = "osm_nearest_distance"
    OSM_FOOTPRINT_AREA = "osm_footprint_area"
    OSM_ROOF_TAG_ORIENTATION = "osm_roof_tag_orientation"
    OSM_ROOF_TAG_SLOPE = "osm_roof_tag_slope"
    PROTECTED_AREA_OVERLAP = "protected_area_overlap"
    HAZARD_SOURCE_MAPPING = "hazard_source_mapping"
    INDICATOR_ENGINE_DERIVE = "indicator_engine_derive"
    CONSTRAINT_ENGINE = "constraint_engine"
    CRITERION_NORMALIZE = "criterion_normalize"
    WEIGHTED_SUM_RANKING = "weighted_sum_ranking"
    PARETO_FRONT = "pareto_front"
    PV_CATALOG_LOAD = "pv_catalog_load"
    PV_DESIGN_EVALUATE = "pv_design_evaluate"
    PV_DESIGN_ENUMERATE = "pv_design_enumerate"
    PVGIS_PV_YIELD = "pvgis_pv_yield"
    PVLIB_PV_YIELD = "pvlib_pv_yield"


METHOD_VERSIONS: dict[MethodId, str] = {
    MethodId.MINMAX_NORMALIZE: "1",
    MethodId.WEIGHTED_LINEAR_COMBINATION: "1",
    MethodId.AHP_GEOMETRIC_MEAN: "1",
    MethodId.BOOLEAN_EXCLUSION: "1",
    MethodId.CANDIDATES_FROM_MASK: "1",
    MethodId.CANDIDATES_REGULAR_GRID: "1",
    MethodId.CANDIDATES_HEXAGONAL_GRID: "1",
    MethodId.CANDIDATES_FROM_FEATURES: "1",
    MethodId.CANDIDATES_CONTIGUOUS_GROUPS: "1",
    MethodId.RETAIN_FEASIBLE: "1",
    MethodId.GREEDY_BUDGET_SELECTION: "1",
    MethodId.HORN_SLOPE_ASPECT: "1",
    MethodId.RILEY_TRI: "1",
    MethodId.DEM_ELEVATION_ZONAL: "1",
    MethodId.CAPACITY_FACTOR_ENERGY: "1",
    MethodId.IRRADIATION_AREA_EFFICIENCY_ENERGY: "1",
    MethodId.LAND_CLASS_AREA_FRACTION: "1",
    MethodId.LAND_DOMINANT_CLASS: "1",
    MethodId.LAND_AVAILABLE_AREA_PROFILE: "1",
    MethodId.OSM_NEAREST_DISTANCE: "1",
    MethodId.OSM_FOOTPRINT_AREA: "1",
    MethodId.OSM_ROOF_TAG_ORIENTATION: "1",
    MethodId.OSM_ROOF_TAG_SLOPE: "1",
    MethodId.PROTECTED_AREA_OVERLAP: "1",
    MethodId.HAZARD_SOURCE_MAPPING: "1",
    MethodId.INDICATOR_ENGINE_DERIVE: "1",
    MethodId.CONSTRAINT_ENGINE: "1",
    MethodId.CRITERION_NORMALIZE: "1",
    MethodId.WEIGHTED_SUM_RANKING: "1",
    MethodId.PARETO_FRONT: "1",
    MethodId.PV_CATALOG_LOAD: "1",
    MethodId.PV_DESIGN_EVALUATE: "1",
    MethodId.PV_DESIGN_ENUMERATE: "1",
    MethodId.PVGIS_PV_YIELD: "1",
    MethodId.PVLIB_PV_YIELD: "1",
}
