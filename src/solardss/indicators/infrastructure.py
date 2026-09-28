"""Nearest-feature distances on projected or geographic coordinates.

This module does not query OpenStreetMap. A provider supplies an origin and
``InfraFeature`` geometries. Empty feature lists stay empty: distance is not
invented as zero or infinity. Hosting capacity is not computed here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from math import atan2, cos, hypot, radians, sin, sqrt

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from solardss.core.arrays import FloatArray
from solardss.core.hashing import fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.exceptions import InvalidParameterError
from solardss.provenance.record import provenance_record

CALC_NEAREST_DISTANCE = "osm_nearest_distance"
DISTANCE_METHOD_PROJECTED = "projected_euclidean_m"
DISTANCE_METHOD_EQUIRECTANGULAR = "wgs84_authalic_equirectangular_m"
DISTANCE_METHOD_HAVERSINE = "wgs84_authalic_haversine_m"

# WGS84 authalic radius (m). Same sphere as land-cover cell areas.
WGS84_AUTHALIC_RADIUS_M = 6371007.180918475

_GEOGRAPHIC_EXACT = frozenset(
    {
        "EPSG:4326",
        "EPSG:4269",
        "EPSG:4258",
        "CRS:84",
        "OGC:CRS84",
        "WGS84",
        "WGS 84",
    }
)


class FeatureKind(str, Enum):
    """Mapped infrastructure classes. Demand centers are caller-declared only."""

    POWER_LINE = "power_line"
    SUBSTATION = "substation"
    ROAD = "road"
    SETTLEMENT = "settlement"
    BUILDING = "building"
    INDUSTRIAL = "industrial"
    DEMAND_CENTER = "demand_center"


class GeometryKind(str, Enum):
    """How ``coordinates`` should be interpreted."""

    POINT = "point"
    LINESTRING = "linestring"
    POLYGON = "polygon"


class InfraFeature(BaseModel):
    """One vector feature. Coordinates are ``(x, y)`` / ``(lon, lat)`` pairs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: FeatureKind
    geometry_kind: GeometryKind
    coordinates: tuple[tuple[float, float], ...] = Field(min_length=1)
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()
    osm_type: str | None = None
    osm_id: int | None = None
    name: str | None = None
    tags: tuple[tuple[str, str], ...] = ()

    def tag_map(self) -> dict[str, str]:
        return {key: value for key, value in self.tags}


class DistanceHit(BaseModel):
    """Nearest feature and the metre distance to it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    distance_m: float
    method: str = Field(min_length=1)
    feature: InfraFeature
    origin_x: float
    origin_y: float
    crs: str | None = None
    inside_polygon: bool = False


@dataclass(frozen=True, slots=True)
class DistanceAnalysisResult:
    """Nearest-neighbour result plus a provenance layer."""

    hit: DistanceHit | None
    kind: FeatureKind
    feature_count: int
    layer: LayerResult


def crs_is_geographic(crs: str | None) -> bool:
    """True when ``crs`` names a geographic (degree) coordinate system."""

    if crs is None:
        return False
    compact = crs.strip().upper().replace(" ", "")
    if compact in {item.replace(" ", "") for item in _GEOGRAPHIC_EXACT}:
        return True
    if compact.startswith("EPSG:4326") or compact.startswith("EPSG:4269"):
        return True
    return "GEOGCS" in compact or "GEOGRAPHIC" in compact


def wrap_delta_deg(delta: float) -> float:
    """Wrap a longitude difference into ``(-180, 180]``."""

    wrapped = (float(delta) + 180.0) % 360.0 - 180.0
    if wrapped == -180.0:
        return 180.0
    return wrapped


def local_en_m(
    x: float, y: float, origin_x: float, origin_y: float, *, crs: str | None
) -> tuple[float, float]:
    """East/north metres of ``(x, y)`` relative to the origin.

    Geographic coordinates use an authalic equirectangular plane centred on the
    origin. Projected coordinates are treated as metres already.
    """

    if crs_is_geographic(crs):
        radius = WGS84_AUTHALIC_RADIUS_M
        phi0 = radians(origin_y)
        east = radius * radians(wrap_delta_deg(x - origin_x)) * cos(phi0)
        north = radius * radians(y - origin_y)
        return east, north
    return float(x) - float(origin_x), float(y) - float(origin_y)


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance on the WGS84 authalic sphere, in metres."""

    radius = WGS84_AUTHALIC_RADIUS_M
    phi1 = radians(lat1)
    phi2 = radians(lat2)
    dphi = phi2 - phi1
    dlam = radians(wrap_delta_deg(lon2 - lon1))
    chord = sin(dphi / 2.0) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2.0) ** 2
    angle = 2.0 * atan2(sqrt(chord), sqrt(max(0.0, 1.0 - chord)))
    return radius * angle


def _point_to_segment_m(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx = bx - ax
    dy = by - ay
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        return hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length2
    t = min(1.0, max(0.0, t))
    return hypot(px - (ax + t * dx), py - (ay + t * dy))


def _closed_ring(ring: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if not ring:
        return ()
    if ring[0] == ring[-1]:
        return tuple(ring)
    return (*tuple(ring), ring[0])


def point_in_ring(x: float, y: float, ring: Sequence[tuple[float, float]]) -> bool:
    """Even-odd ray test. Vertices are ``(x, y)`` already in the analysis plane."""

    closed = _closed_ring(ring)
    if len(closed) < 4:
        return False
    inside = False
    for index in range(len(closed) - 1):
        x1, y1 = closed[index]
        x2, y2 = closed[index + 1]
        crosses = (y1 > y) != (y2 > y)
        if crosses:
            at_x = (x2 - x1) * (y - y1) / (y2 - y1 + 0.0) + x1
            if x < at_x:
                inside = not inside
    return inside


def _ring_to_local(
    ring: Sequence[tuple[float, float]], origin_x: float, origin_y: float, crs: str | None
) -> tuple[tuple[float, float], ...]:
    return tuple(local_en_m(x, y, origin_x, origin_y, crs=crs) for x, y in ring)


def _distance_to_ring_m(px: float, py: float, ring: Sequence[tuple[float, float]]) -> float:
    closed = _closed_ring(ring)
    if len(closed) < 2:
        if not closed:
            return float("inf")
        return hypot(px - closed[0][0], py - closed[0][1])
    best = float("inf")
    for index in range(len(closed) - 1):
        ax, ay = closed[index]
        bx, by = closed[index + 1]
        best = min(best, _point_to_segment_m(px, py, ax, ay, bx, by))
    return best


def feature_distance_m(
    origin_x: float,
    origin_y: float,
    feature: InfraFeature,
    *,
    crs: str | None,
) -> tuple[float, bool]:
    """Metre distance from the origin to ``feature``. Polygon interior is 0."""

    method_crs = crs
    verts = _ring_to_local(feature.coordinates, origin_x, origin_y, method_crs)
    if feature.geometry_kind is GeometryKind.POINT:
        east, north = verts[0]
        return hypot(east, north), False
    if feature.geometry_kind is GeometryKind.LINESTRING:
        return _distance_to_ring_m(0.0, 0.0, verts), False
    if feature.geometry_kind is GeometryKind.POLYGON:
        outer = verts
        if point_in_ring(0.0, 0.0, outer):
            in_hole = False
            for hole in feature.holes:
                hole_xy = _ring_to_local(hole, origin_x, origin_y, method_crs)
                if point_in_ring(0.0, 0.0, hole_xy):
                    in_hole = True
                    break
            if not in_hole:
                return 0.0, True
        best = _distance_to_ring_m(0.0, 0.0, outer)
        for hole in feature.holes:
            hole_xy = _ring_to_local(hole, origin_x, origin_y, method_crs)
            best = min(best, _distance_to_ring_m(0.0, 0.0, hole_xy))
        return best, False
    raise InvalidParameterError(f"unsupported geometry kind {feature.geometry_kind.value}")


def distance_method_for_crs(crs: str | None) -> str:
    if crs_is_geographic(crs):
        return DISTANCE_METHOD_EQUIRECTANGULAR
    return DISTANCE_METHOD_PROJECTED


def nearest_feature(
    origin_x: float,
    origin_y: float,
    features: Sequence[InfraFeature],
    *,
    crs: str | None = "EPSG:4326",
    kind: FeatureKind | None = None,
) -> DistanceHit | None:
    """Closest feature of ``kind``. ``None`` when the list is empty after filtering."""

    method = distance_method_for_crs(crs)
    best: DistanceHit | None = None
    for feature in features:
        if kind is not None and feature.kind is not kind:
            continue
        distance, inside = feature_distance_m(origin_x, origin_y, feature, crs=crs)
        if best is None or distance < best.distance_m:
            best = DistanceHit(
                distance_m=distance,
                method=method,
                feature=feature,
                origin_x=origin_x,
                origin_y=origin_y,
                crs=crs,
                inside_polygon=inside,
            )
    return best


def analyze_nearest_distance(
    origin_x: float,
    origin_y: float,
    features: Sequence[InfraFeature],
    *,
    kind: FeatureKind,
    crs: str | None = "EPSG:4326",
) -> DistanceAnalysisResult:
    """Provenanced nearest-neighbour distance for one infrastructure class."""

    matching = tuple(item for item in features if item.kind is kind)
    hit = nearest_feature(origin_x, origin_y, matching, crs=crs, kind=kind)
    warnings: list[str] = []
    if not matching:
        warnings.append(f"no {kind.value} features supplied; distance is not invented")
        status = EvaluationStatus.INCOMPLETE_INPUTS
        values: FloatArray = np.asarray([], dtype=np.float64)
    else:
        status = EvaluationStatus.EVALUATED
        values = np.asarray([hit.distance_m if hit is not None else np.nan], dtype=np.float64)
    parameters: dict[str, object] = {
        "kind": kind.value,
        "crs": crs,
        "origin_x": origin_x,
        "origin_y": origin_y,
        "feature_count": len(matching),
        "distance_method": distance_method_for_crs(crs),
        "distance_m": hit.distance_m if hit is not None else None,
        "nearest_osm_id": hit.feature.osm_id if hit is not None else None,
        "inside_polygon": hit.inside_polygon if hit is not None else False,
    }
    layer = LayerResult(
        name=f"distance_{kind.value}",
        status=status,
        method_id=MethodId.OSM_NEAREST_DISTANCE.value,
        provenance=provenance_record(
            MethodId.OSM_NEAREST_DISTANCE,
            parameters=parameters,
            input_fingerprints=(
                fingerprint_canonical(
                    {
                        "kind": kind.value,
                        "n": len(matching),
                        "ids": [item.osm_id for item in matching],
                    }
                ),
            ),
            notes=tuple(warnings),
        ),
        summary=summarize_array(values),
        warnings=tuple(warnings),
    )
    return DistanceAnalysisResult(hit=hit, kind=kind, feature_count=len(matching), layer=layer)


def demand_center_features(
    points: Sequence[tuple[float, float]],
    *,
    names: Sequence[str] | None = None,
) -> tuple[InfraFeature, ...]:
    """Caller-declared demand-center points. Not inferred from settlements."""

    if names is not None and len(names) != len(points):
        raise InvalidParameterError("demand-center names must match the number of points")
    features: list[InfraFeature] = []
    for index, (x, y) in enumerate(points):
        label = names[index] if names is not None else f"demand_center_{index}"
        features.append(
            InfraFeature(
                kind=FeatureKind.DEMAND_CENTER,
                geometry_kind=GeometryKind.POINT,
                coordinates=((float(x), float(y)),),
                name=label,
            )
        )
    return tuple(features)


__all__ = [
    "CALC_NEAREST_DISTANCE",
    "DISTANCE_METHOD_EQUIRECTANGULAR",
    "DISTANCE_METHOD_HAVERSINE",
    "DISTANCE_METHOD_PROJECTED",
    "WGS84_AUTHALIC_RADIUS_M",
    "DistanceAnalysisResult",
    "DistanceHit",
    "FeatureKind",
    "GeometryKind",
    "InfraFeature",
    "analyze_nearest_distance",
    "crs_is_geographic",
    "demand_center_features",
    "distance_method_for_crs",
    "feature_distance_m",
    "haversine_distance_m",
    "local_en_m",
    "nearest_feature",
    "point_in_ring",
    "wrap_delta_deg",
]
