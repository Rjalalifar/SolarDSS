"""Building-footprint area and OSM roof-tag mapping.

Footprint area is plan-view geometry, not usable PV area. Roof orientation and
slope are mapped only from explicit tags; missing tags stay missing. SolarDSS
never invents a roof plane from a footprint rectangle.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from solardss.core.arrays import FloatArray
from solardss.core.hashing import fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.exceptions import InvalidParameterError
from solardss.indicators.infrastructure import (
    WGS84_AUTHALIC_RADIUS_M,
    InfraFeature,
    crs_is_geographic,
    wrap_delta_deg,
)
from solardss.provenance.record import provenance_record

CALC_FOOTPRINT_AREA = "osm_footprint_area"
CALC_ROOF_ORIENTATION = "osm_roof_tag_orientation"
CALC_ROOF_SLOPE = "osm_roof_tag_slope"
AREA_METHOD_PROJECTED = "projected_shoelace_m2"
AREA_METHOD_AUTHALIC = "wgs84_authalic_trapezoid_m2"

# Cardinal / compass points actually present on OSM ``roof:direction``.
# Values are degrees clockwise from north. Unknown tokens are not guessed.
ROOF_DIRECTION_DEGREES: dict[str, float] = {
    "N": 0.0,
    "NORTH": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "NORTHEAST": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "EAST": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SOUTHEAST": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SOUTH": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "SOUTHWEST": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WEST": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NORTHWEST": 315.0,
    "NNW": 337.5,
}

# Only shapes that themselves state a numeric slope. ``gabled`` is not 0 or 45.
ROOF_SHAPE_SLOPE_DEG: dict[str, float] = {
    "flat": 0.0,
}

PRESERVED_ROOF_KEYS: tuple[str, ...] = (
    "roof:shape",
    "roof:direction",
    "roof:angle",
    "building:levels",
)


class RoofAttributes(BaseModel):
    """Raw OSM roof tags plus mapped orientation/slope when the tags support them."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    shape: str | None = None
    direction_raw: str | None = None
    angle_raw: str | None = None
    levels_raw: str | None = None
    orientation_deg: float | None = None
    slope_deg: float | None = None
    preserved: tuple[tuple[str, str], ...] = ()
    notes: tuple[str, ...] = ()


class FootprintArea(BaseModel):
    """Plan-view area of one building ring."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    area_m2: float = Field(ge=0.0)
    method: str = Field(min_length=1)
    vertex_count: int
    has_holes: bool = False


@dataclass(frozen=True, slots=True)
class RooftopAnalysisResult:
    """Footprint and optional roof-tag mapping for one building candidate."""

    footprint: FootprintArea | None
    roof: RoofAttributes
    feature: InfraFeature | None
    layer: LayerResult


def _closed_ring(ring: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if not ring:
        return ()
    if ring[0] == ring[-1]:
        return tuple(ring)
    return (*tuple(ring), ring[0])


def projected_ring_area_m2(ring: Sequence[tuple[float, float]]) -> float:
    """Planar shoelace area. Coordinates must already be metres."""

    closed = _closed_ring(ring)
    if len(closed) < 4:
        return 0.0
    total = 0.0
    for index in range(len(closed) - 1):
        x1, y1 = closed[index]
        x2, y2 = closed[index + 1]
        total += x1 * y2 - x2 * y1
    return abs(total) * 0.5


def authalic_ring_area_m2(ring: Sequence[tuple[float, float]]) -> float:
    """WGS84 authalic-sphere trapezoid area of a lon/lat ring, in square metres."""

    closed = _closed_ring(ring)
    if len(closed) < 4:
        return 0.0
    radius = WGS84_AUTHALIC_RADIUS_M
    total = 0.0
    for index in range(len(closed) - 1):
        lon1, lat1 = closed[index]
        lon2, lat2 = closed[index + 1]
        dlam = radians(wrap_delta_deg(lon2 - lon1))
        total += dlam * (sin(radians(lat1)) + sin(radians(lat2))) * 0.5
    return abs(total) * radius * radius


def polygon_area_m2(
    ring: Sequence[tuple[float, float]],
    *,
    crs: str | None = "EPSG:4326",
    holes: Sequence[Sequence[tuple[float, float]]] = (),
) -> tuple[float, str]:
    """Outer area minus holes. Geographic rings use authalic trapezoids."""

    if crs_is_geographic(crs):
        area = authalic_ring_area_m2(ring)
        for hole in holes:
            area -= authalic_ring_area_m2(hole)
        return max(0.0, area), AREA_METHOD_AUTHALIC
    area = projected_ring_area_m2(ring)
    for hole in holes:
        area -= projected_ring_area_m2(hole)
    return max(0.0, area), AREA_METHOD_PROJECTED


def footprint_area_of(
    feature: InfraFeature, *, crs: str | None = "EPSG:4326"
) -> FootprintArea | None:
    """Plan-view area when the feature has a closed polygonal ring."""

    if len(feature.coordinates) < 3:
        return None
    unique = {item for item in feature.coordinates}
    if len(unique) < 3:
        return None
    area, method = polygon_area_m2(feature.coordinates, crs=crs, holes=feature.holes)
    if area <= 0.0:
        return None
    return FootprintArea(
        area_m2=area,
        method=method,
        vertex_count=len(feature.coordinates),
        has_holes=bool(feature.holes),
    )


def _parse_degrees(raw: str) -> float | None:
    text = raw.strip().upper().replace("°", "").replace("DEG", "").replace("DEGREES", "")
    text = text.replace("DEGREE", "").strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def parse_roof_orientation(raw: str | None) -> tuple[float | None, str | None]:
    """Map ``roof:direction`` to degrees clockwise from north, if possible."""

    if raw is None or not raw.strip():
        return None, "roof:direction is absent; orientation is not invented"
    token = raw.strip()
    mapped = ROOF_DIRECTION_DEGREES.get(token.upper())
    if mapped is not None:
        return mapped, None
    numeric = _parse_degrees(token)
    if numeric is None:
        return None, f"roof:direction={token!r} is not a compass point or degree value"
    if numeric < 0.0 or numeric > 360.0:
        return None, f"roof:direction={token!r} is outside [0, 360]"
    if numeric == 360.0:
        numeric = 0.0
    return numeric, None


def parse_roof_slope(angle_raw: str | None, shape: str | None) -> tuple[float | None, str | None]:
    """Map ``roof:angle`` or ``roof:shape=flat``. Other shapes do not imply a slope."""

    if angle_raw is not None and angle_raw.strip():
        numeric = _parse_degrees(angle_raw)
        if numeric is None:
            return None, f"roof:angle={angle_raw!r} is not numeric; slope is not invented"
        if numeric < 0.0 or numeric > 90.0:
            return None, f"roof:angle={angle_raw!r} is outside [0, 90]"
        return numeric, None
    if shape is not None:
        mapped = ROOF_SHAPE_SLOPE_DEG.get(shape.strip().lower())
        if mapped is not None:
            return mapped, f"slope mapped from roof:shape={shape!r} only"
        return None, (f"roof:shape={shape!r} is preserved but does not imply a numeric slope")
    return None, "roof:angle is absent; slope is not invented from the footprint"


def parse_roof_tags(tags: Mapping[str, str]) -> RoofAttributes:
    """Preserve listed OSM roof tags; map orientation/slope only when supported."""

    preserved = tuple((key, tags[key]) for key in PRESERVED_ROOF_KEYS if tags.get(key))
    shape = tags.get("roof:shape") or None
    direction_raw = tags.get("roof:direction") or None
    angle_raw = tags.get("roof:angle") or None
    levels_raw = tags.get("building:levels") or None
    notes: list[str] = [
        "OSM roof tags are incomplete in general and are not treated as a survey",
    ]
    orientation, orientation_note = parse_roof_orientation(direction_raw)
    slope, slope_note = parse_roof_slope(angle_raw, shape)
    if orientation_note:
        notes.append(orientation_note)
    if slope_note:
        notes.append(slope_note)
    if levels_raw is not None:
        notes.append(f"building:levels={levels_raw!r} is preserved; not a structural rating")
    return RoofAttributes(
        shape=shape,
        direction_raw=direction_raw,
        angle_raw=angle_raw,
        levels_raw=levels_raw,
        orientation_deg=orientation,
        slope_deg=slope,
        preserved=preserved,
        notes=tuple(notes),
    )


def analyze_rooftop(
    feature: InfraFeature | None,
    *,
    crs: str | None = "EPSG:4326",
) -> RooftopAnalysisResult:
    """Footprint area and tag mapping. Missing geometry stays missing."""

    roof = parse_roof_tags(feature.tag_map() if feature is not None else {})
    footprint = footprint_area_of(feature, crs=crs) if feature is not None else None
    warnings = list(roof.notes)
    if feature is None:
        warnings.append("no building geometry supplied; footprint is not invented")
        status = EvaluationStatus.INCOMPLETE_INPUTS
        values: FloatArray = np.asarray([], dtype=np.float64)
        area_value: float | None = None
    elif footprint is None:
        warnings.append("building is not a closed polygonal footprint; area is not invented")
        status = EvaluationStatus.INCOMPLETE_INPUTS
        values = np.asarray([], dtype=np.float64)
        area_value = None
    else:
        status = EvaluationStatus.EVALUATED
        values = np.asarray([footprint.area_m2], dtype=np.float64)
        area_value = footprint.area_m2
    parameters: dict[str, object] = {
        "crs": crs,
        "osm_id": feature.osm_id if feature is not None else None,
        "area_m2": area_value,
        "orientation_deg": roof.orientation_deg,
        "slope_deg": roof.slope_deg,
        "roof_shape": roof.shape,
        "roof_direction": roof.direction_raw,
        "roof_angle": roof.angle_raw,
        "building_levels": roof.levels_raw,
        "usable_area_not_inferred": True,
    }
    layer = LayerResult(
        name="roof_footprint",
        status=status,
        method_id=MethodId.OSM_FOOTPRINT_AREA.value,
        provenance=provenance_record(
            MethodId.OSM_FOOTPRINT_AREA,
            parameters=parameters,
            input_fingerprints=(
                fingerprint_canonical(
                    {
                        "osm_id": feature.osm_id if feature is not None else None,
                        "n": len(feature.coordinates) if feature is not None else 0,
                    }
                ),
            ),
            notes=tuple(warnings),
        ),
        summary=summarize_array(values),
        warnings=tuple(warnings),
    )
    return RooftopAnalysisResult(footprint=footprint, roof=roof, feature=feature, layer=layer)


def local_square_lonlat(
    west: float, south: float, size_m: float, *, lat0: float | None = None
) -> tuple[tuple[float, float], ...]:
    """Helper for tests: axis-aligned square in lon/lat of side ``size_m``."""

    if size_m <= 0.0:
        raise InvalidParameterError("size_m must be positive")
    phi = radians(south if lat0 is None else lat0)
    dlat = size_m / WGS84_AUTHALIC_RADIUS_M
    dlon = size_m / (WGS84_AUTHALIC_RADIUS_M * max(cos(phi), 1e-6))
    north = south + (dlat * 180.0 / np.pi)
    east = west + (dlon * 180.0 / np.pi)
    return ((west, south), (east, south), (east, north), (west, north), (west, south))


__all__ = [
    "AREA_METHOD_AUTHALIC",
    "AREA_METHOD_PROJECTED",
    "CALC_FOOTPRINT_AREA",
    "CALC_ROOF_ORIENTATION",
    "CALC_ROOF_SLOPE",
    "PRESERVED_ROOF_KEYS",
    "ROOF_DIRECTION_DEGREES",
    "ROOF_SHAPE_SLOPE_DEG",
    "FootprintArea",
    "RoofAttributes",
    "RooftopAnalysisResult",
    "analyze_rooftop",
    "authalic_ring_area_m2",
    "footprint_area_of",
    "local_square_lonlat",
    "parse_roof_orientation",
    "parse_roof_slope",
    "parse_roof_tags",
    "polygon_area_m2",
    "projected_ring_area_m2",
]
