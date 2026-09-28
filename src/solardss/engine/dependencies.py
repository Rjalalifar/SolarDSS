"""Default explicit indicator DAG.

Edges are declared here. Providers do not discover each other. A DEM adapter
may compute slope from its own raster; that is one provider, not a hidden hop
to an elevation provider.
"""

from __future__ import annotations

from solardss.engine.contracts import (
    BatchScope,
    Dependency,
    DependencyKind,
    IndicatorNode,
    ResolutionPath,
)
from solardss.engine.graph import IndicatorGraph
from solardss.indicators.catalog import registry
from solardss.indicators.contracts import Controllability, IndicatorSpec
from solardss.indicators.ids import IndicatorId
from solardss.indicators.registry import IndicatorRegistry

_ID = IndicatorId


def _ind(indicator_id: str | IndicatorId, *, required: bool = True) -> Dependency:
    target = indicator_id.value if isinstance(indicator_id, IndicatorId) else indicator_id
    return Dependency(kind=DependencyKind.INDICATOR, target=target, required=required)


def _src(source_type: str, *, required: bool = True) -> Dependency:
    return Dependency(kind=DependencyKind.SOURCE_TYPE, target=source_type, required=required)


def _node(
    indicator_id: IndicatorId,
    *dependencies: Dependency,
    resolution: ResolutionPath = ResolutionPath.PROVIDER,
    batch_scope: BatchScope = BatchScope.POINT,
    implemented: bool = True,
) -> IndicatorNode:
    return IndicatorNode(
        indicator_id=indicator_id.value,
        dependencies=dependencies,
        resolution=resolution,
        batch_scope=batch_scope,
        implemented=implemented,
    )


def _leaf_from_spec(spec: IndicatorSpec) -> IndicatorNode:
    if spec.controllability in {
        Controllability.CONTROLLABLE,
        Controllability.FIXED,
    }:
        resolution = ResolutionPath.CONTEXT
        implemented = True
    elif spec.controllability is Controllability.EXTERNALLY_PROVIDED or (
        spec.derivation_method is not None and not spec.derivation_method.implemented_in_solardss
    ):
        resolution = ResolutionPath.PROVIDER_OR_DERIVED
        implemented = False
    else:
        resolution = ResolutionPath.PROVIDER
        implemented = True
    return IndicatorNode(
        indicator_id=spec.id,
        resolution=resolution,
        implemented=implemented,
    )


_EXPLICIT: tuple[IndicatorNode, ...] = (
    _node(
        _ID.TERRAIN_ELEVATION,
        _src("dem"),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.TERRAIN_SLOPE_MEAN,
        _src("dem"),
        _ind(_ID.TERRAIN_ELEVATION, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.TERRAIN_SLOPE_MAX,
        _src("dem"),
        _ind(_ID.TERRAIN_ELEVATION, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.TERRAIN_ASPECT,
        _src("dem"),
        _ind(_ID.TERRAIN_ELEVATION, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.TERRAIN_RUGGEDNESS,
        _src("dem"),
        _ind(_ID.TERRAIN_ELEVATION, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.TERRAIN_HORIZON_OBSTRUCTION,
        _src("dem"),
        _ind(_ID.TERRAIN_ELEVATION, required=False),
        implemented=False,
        resolution=ResolutionPath.PROVIDER_OR_DERIVED,
    ),
    _node(
        _ID.TERRAIN_SHADING_POTENTIAL,
        _src("dem"),
        _ind(_ID.TERRAIN_SLOPE_MEAN, required=False),
        implemented=False,
        resolution=ResolutionPath.PROVIDER_OR_DERIVED,
    ),
    _node(
        _ID.SOLAR_POA_IRRADIATION,
        _ind(_ID.SOLAR_GHI_ANNUAL, required=False),
        _ind(_ID.SOLAR_DNI_ANNUAL, required=False),
        _ind(_ID.SOLAR_DHI_ANNUAL, required=False),
        _ind(_ID.PV_TILT),
        _ind(_ID.PV_AZIMUTH),
        resolution=ResolutionPath.PROVIDER_OR_DERIVED,
        implemented=False,
    ),
    _node(
        _ID.SOLAR_SPECIFIC_YIELD,
        _ind(_ID.SOLAR_POA_IRRADIATION, required=False),
        _ind(_ID.SOLAR_GHI_ANNUAL, required=False),
        _ind(_ID.PV_SYSTEM_CAPACITY, required=False),
        resolution=ResolutionPath.PROVIDER_OR_DERIVED,
        implemented=False,
    ),
    _node(
        _ID.SOLAR_PVOUT,
        _ind(_ID.SOLAR_SPECIFIC_YIELD, required=False),
        _ind(_ID.SOLAR_GHI_ANNUAL, required=False),
        resolution=ResolutionPath.PROVIDER_OR_DERIVED,
        implemented=False,
    ),
    _node(_ID.LAND_COVER, _src("land_cover_map"), batch_scope=BatchScope.SHARED_AOI),
    _node(
        _ID.LAND_FOREST_FRACTION,
        _src("land_cover_map"),
        _ind(_ID.LAND_COVER, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.LAND_WATER_FRACTION,
        _src("land_cover_map"),
        _ind(_ID.LAND_COVER, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.LAND_WETLAND_FRACTION,
        _src("land_cover_map"),
        _ind(_ID.LAND_COVER, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.LAND_AGRICULTURAL_FRACTION,
        _src("land_cover_map"),
        _ind(_ID.LAND_COVER, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.LAND_BUILTUP_FRACTION,
        _src("land_cover_map"),
        _ind(_ID.LAND_COVER, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.LAND_BARE_LAND_FRACTION,
        _src("land_cover_map"),
        _ind(_ID.LAND_COVER, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.LAND_AVAILABLE_AREA,
        _src("land_cover_map"),
        _ind(_ID.LAND_COVER, required=False),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.LAND_PROTECTED_AREA_OVERLAP,
        _src("protected_area_cadastre"),
        batch_scope=BatchScope.SHARED_AOI,
    ),
    _node(
        _ID.INFRA_DISTANCE_TO_POWER_LINE,
        _src("osm_or_network_vector"),
    ),
    _node(
        _ID.INFRA_DISTANCE_TO_SUBSTATION,
        _src("osm_or_network_vector"),
    ),
    _node(
        _ID.INFRA_DISTANCE_TO_ROAD,
        _src("osm_or_network_vector"),
    ),
    _node(
        _ID.INFRA_DISTANCE_TO_SETTLEMENT,
        _src("osm_or_network_vector"),
    ),
    _node(
        _ID.INFRA_DISTANCE_TO_DEMAND_CENTER,
        _src("caller_provided"),
        _src("osm_or_network_vector", required=False),
    ),
    _node(
        _ID.INFRA_GRID_HOSTING_CAPACITY,
        _src("grid_operator"),
        resolution=ResolutionPath.PROVIDER_OR_DERIVED,
        implemented=False,
    ),
    _node(_ID.HAZARD_FLOOD_RISK, _src("hazard_map")),
    _node(_ID.HAZARD_WILDFIRE_RISK, _src("hazard_map")),
    _node(_ID.HAZARD_LANDSLIDE_RISK, _src("hazard_map")),
    _node(_ID.HAZARD_HAIL_RISK, _src("hazard_map")),
    _node(_ID.ROOF_BUILDING_FOOTPRINT_AREA, _src("building_footprint")),
    _node(
        _ID.ROOF_ORIENTATION,
        _src("building_footprint"),
        _ind(_ID.ROOF_BUILDING_FOOTPRINT_AREA, required=False),
    ),
    _node(
        _ID.ROOF_SLOPE,
        _src("building_footprint"),
        _ind(_ID.ROOF_BUILDING_FOOTPRINT_AREA, required=False),
    ),
    _node(
        _ID.ROOF_USABLE_AREA,
        _ind(_ID.ROOF_BUILDING_FOOTPRINT_AREA),
        _ind(_ID.ROOF_SLOPE, required=False),
        resolution=ResolutionPath.PROVIDER_OR_DERIVED,
        implemented=False,
    ),
    _node(
        _ID.ROOF_SHADING,
        _ind(_ID.ROOF_BUILDING_FOOTPRINT_AREA, required=False),
        resolution=ResolutionPath.PROVIDER_OR_DERIVED,
        implemented=False,
    ),
    _node(
        _ID.ROOF_STRUCTURAL_CAPACITY,
        _src("engineering_dataset"),
        _ind(_ID.ROOF_BUILDING_FOOTPRINT_AREA, required=False),
        resolution=ResolutionPath.PROVIDER_OR_DERIVED,
        implemented=False,
    ),
)


def build_default_graph(
    indicator_registry: IndicatorRegistry | None = None,
) -> IndicatorGraph:
    """One node per registered indicator, with explicit overrides applied."""

    catalog = registry if indicator_registry is None else indicator_registry
    leaves = [_leaf_from_spec(spec) for spec in catalog.list_all()]
    overlay = {item.indicator_id: item for item in _EXPLICIT}
    nodes = [overlay.get(item.indicator_id, item) for item in leaves]
    return IndicatorGraph(tuple(nodes))


DEFAULT_GRAPH = build_default_graph()
