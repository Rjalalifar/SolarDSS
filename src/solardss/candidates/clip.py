"""Clip candidate polygons to a search boundary.

Convex clippers use the in-repo Sutherland-Hodgman implementation. Concave
clippers and boundaries with holes use shapely, imported lazily.
"""

from __future__ import annotations

from typing import Any

from solardss.candidates.contracts import BoundaryGeometry, PolygonPart
from solardss.candidates.errors import CandidateClipError
from solardss.candidates.geometry import (
    bboxes_disjoint,
    clip_ring,
    envelope_of,
    ring_bbox,
    ring_is_convex,
)
from solardss.exceptions import MissingExtraError


def clip_parts_to_boundary(
    parts: tuple[PolygonPart, ...],
    boundary: BoundaryGeometry,
) -> tuple[PolygonPart, ...]:
    """Return the intersection of ``parts`` with ``boundary``. Empty means no overlap."""

    env = envelope_of(boundary)
    convex = ring_is_convex(boundary.exterior) and not boundary.holes
    kept: list[PolygonPart] = []
    for part in parts:
        if bboxes_disjoint(ring_bbox(part.exterior), env):
            continue
        if convex:
            clipped = _clip_convex(part, boundary.exterior)
            if clipped is not None:
                kept.append(clipped)
            continue
        kept.extend(_clip_with_shapely((part,), boundary))
    return tuple(kept)


def _clip_convex(part: PolygonPart, clipper: tuple[tuple[float, float], ...]) -> PolygonPart | None:
    outer = clip_ring(part.exterior, clipper)
    if len(outer) < 4:
        return None
    holes: list[tuple[tuple[float, float], ...]] = []
    for hole in part.holes:
        clipped_hole = clip_ring(hole, clipper)
        if len(clipped_hole) >= 4:
            holes.append(clipped_hole)
    return PolygonPart(exterior=outer, holes=tuple(holes))


def _clip_with_shapely(
    parts: tuple[PolygonPart, ...], boundary: BoundaryGeometry
) -> tuple[PolygonPart, ...]:
    polygon_cls = _require_shapely()
    clipper = _part_to_shapely(
        polygon_cls, PolygonPart(exterior=boundary.exterior, holes=boundary.holes)
    )
    if clipper.is_empty:
        return ()
    result: list[PolygonPart] = []
    for part in parts:
        subject = _part_to_shapely(polygon_cls, part)
        if subject.is_empty:
            continue
        inter = subject.intersection(clipper)
        result.extend(_shapely_to_parts(inter))
    return tuple(result)


def _part_to_shapely(polygon_cls: Any, part: PolygonPart) -> Any:
    holes = [list(hole) for hole in part.holes]
    return polygon_cls(list(part.exterior), holes)


def _shapely_to_parts(geom: Any) -> tuple[PolygonPart, ...]:
    if geom.is_empty:
        return ()
    if geom.geom_type == "Polygon":
        return (_polygon_part(geom),)
    if geom.geom_type == "MultiPolygon":
        return tuple(_polygon_part(item) for item in geom.geoms if not item.is_empty)
    if geom.geom_type == "GeometryCollection":
        found: list[PolygonPart] = []
        for item in geom.geoms:
            found.extend(_shapely_to_parts(item))
        return tuple(found)
    return ()


def _polygon_part(geom: Any) -> PolygonPart:
    exterior = tuple((float(x), float(y)) for x, y in geom.exterior.coords)
    holes = tuple(tuple((float(x), float(y)) for x, y in ring.coords) for ring in geom.interiors)
    return PolygonPart(exterior=exterior, holes=holes)


def _require_shapely() -> Any:
    try:
        from shapely.geometry import Polygon
    except ImportError as exc:
        raise MissingExtraError(
            "Concave AOI/region clip or hole subtraction needs shapely. "
            "Install with: pip install 'solardss[gis]'."
        ) from exc
    return Polygon


def require_clip_possible(boundary: BoundaryGeometry) -> None:
    """Raise early when a concave clipper cannot be handled without shapely."""

    if ring_is_convex(boundary.exterior) and not boundary.holes:
        return
    try:
        _require_shapely()
    except MissingExtraError as exc:
        raise CandidateClipError(
            "boundary is concave or has holes; correct clip requires the GIS extra (shapely)"
        ) from exc
