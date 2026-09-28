"""Ring and CRS helpers for candidate polygons. No GIS extra at import."""

from __future__ import annotations

from collections.abc import Sequence

from solardss.candidates.contracts import BoundaryGeometry, CandidateGeometry, PolygonPart
from solardss.candidates.errors import CandidateGridError
from solardss.core.hashing import fingerprint_canonical
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.exceptions import IncompleteInputsError, InvalidParameterError
from solardss.indicators.infrastructure import crs_is_geographic
from solardss.indicators.protected import clip_polygon
from solardss.indicators.rooftop import polygon_area_m2

_CRS_COMPACT_SEP = ""
_AREA_EPS_M2 = 1e-12
_CROSS_EPS = 1e-12


def closed_ring(ring: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Return a ring that repeats the first vertex at the end."""

    if not ring:
        return ()
    if ring[0] == ring[-1]:
        return tuple(ring)
    return (*tuple(ring), ring[0])


def unsigned_ring(ring: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    closed = closed_ring(ring)
    if len(closed) < 2:
        return closed
    return closed[:-1]


def normalize_crs(crs: str) -> str:
    return crs.strip().casefold().replace(" ", _CRS_COMPACT_SEP)


def same_crs(left: str, right: str) -> bool:
    return normalize_crs(left) == normalize_crs(right)


def require_same_crs(left: str, right: str, *, what: str) -> None:
    if not same_crs(left, right):
        raise InvalidParameterError(f"{what} CRS {left!r} does not match {right!r}")


def require_grid_units(crs: str, cell_size_unit: str) -> None:
    """Projected grids use metres; geographic grids use degrees. No silent conversion."""

    geographic = crs_is_geographic(crs)
    if geographic and cell_size_unit != "deg":
        raise CandidateGridError(
            "geographic CRS requires cell_size_unit='deg'; "
            "pass a projected CRS if cell_size is in metres"
        )
    if not geographic and cell_size_unit != "m":
        raise CandidateGridError(
            "projected CRS requires cell_size_unit='m'; "
            "degree cell sizes are only valid on a geographic CRS"
        )


def ring_bbox(
    ring: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float]:
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def bboxes_disjoint(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1]


def ring_is_convex(ring: Sequence[tuple[float, float]]) -> bool:
    """True when consecutive edge cross-products do not change sign."""

    verts = unsigned_ring(ring)
    if len(verts) < 3:
        return False
    sign: bool | None = None
    count = len(verts)
    for index in range(count):
        x1, y1 = verts[index]
        x2, y2 = verts[(index + 1) % count]
        x3, y3 = verts[(index + 2) % count]
        cross = (x2 - x1) * (y3 - y2) - (y2 - y1) * (x3 - x2)
        if abs(cross) <= _CROSS_EPS:
            continue
        current = cross > 0.0
        if sign is None:
            sign = current
        elif current is not sign:
            return False
    return True


def geometry_area_m2(geometry: CandidateGeometry) -> tuple[float, str]:
    """Plan-view area. Geographic parts use authalic trapezoids."""

    total = 0.0
    method = "projected_shoelace_m2"
    for part in geometry.parts:
        area, method = polygon_area_m2(part.exterior, crs=geometry.crs, holes=part.holes)
        total += area
    return total, method


def geometry_centroid(geometry: CandidateGeometry) -> tuple[float, float]:
    """Area-weighted centroid of the parts, in the geometry CRS."""

    total_area = 0.0
    sum_x = 0.0
    sum_y = 0.0
    for part in geometry.parts:
        cx, cy, area = _part_centroid(part.exterior)
        if area <= 0.0:
            continue
        hole_x = 0.0
        hole_y = 0.0
        hole_area = 0.0
        for hole in part.holes:
            hx, hy, ha = _part_centroid(hole)
            hole_x += hx * ha
            hole_y += hy * ha
            hole_area += ha
        net = area - hole_area
        if net <= 0.0:
            continue
        if hole_area > 0.0:
            cx = (cx * area - hole_x) / net
            cy = (cy * area - hole_y) / net
        sum_x += cx * net
        sum_y += cy * net
        total_area += net
    if total_area <= 0.0:
        xs = [pt[0] for part in geometry.parts for pt in part.exterior]
        ys = [pt[1] for part in geometry.parts for pt in part.exterior]
        return (sum(xs) / len(xs), sum(ys) / len(ys))
    return (sum_x / total_area, sum_y / total_area)


def _part_centroid(ring: Sequence[tuple[float, float]]) -> tuple[float, float, float]:
    closed = closed_ring(ring)
    area2 = 0.0
    cx = 0.0
    cy = 0.0
    for index in range(len(closed) - 1):
        x1, y1 = closed[index]
        x2, y2 = closed[index + 1]
        cross = x1 * y2 - x2 * y1
        area2 += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    area = area2 * 0.5
    if abs(area) <= _AREA_EPS_M2:
        xs = [pt[0] for pt in unsigned_ring(ring)]
        ys = [pt[1] for pt in unsigned_ring(ring)]
        if not xs:
            return (0.0, 0.0, 0.0)
        return (sum(xs) / len(xs), sum(ys) / len(ys), 0.0)
    return (cx / (3.0 * area2), cy / (3.0 * area2), abs(area))


def ring_to_wkt(ring: Sequence[tuple[float, float]]) -> str:
    closed = closed_ring(ring)
    body = ", ".join(f"{x} {y}" for x, y in closed)
    return f"({body})"


def geometry_wkt(geometry: CandidateGeometry) -> str:
    polygons = []
    for part in geometry.parts:
        shells = [ring_to_wkt(part.exterior), *[ring_to_wkt(hole) for hole in part.holes]]
        polygons.append("(" + ", ".join(shells) + ")")
    joined = ", ".join(polygons)
    if len(polygons) == 1:
        return f"POLYGON {joined}"
    return f"MULTIPOLYGON ({joined})"


def geometry_fingerprint(geometry: CandidateGeometry) -> str:
    payload = {
        "crs": geometry.crs,
        "parts": [
            {
                "exterior": [list(pt) for pt in part.exterior],
                "holes": [[list(pt) for pt in hole] for hole in part.holes],
            }
            for part in geometry.parts
        ],
    }
    return fingerprint_canonical(payload)


def polygon_geometry(
    exterior: Sequence[tuple[float, float]],
    *,
    crs: str,
    holes: Sequence[Sequence[tuple[float, float]]] = (),
) -> CandidateGeometry:
    part = PolygonPart(
        exterior=closed_ring(exterior),
        holes=tuple(closed_ring(hole) for hole in holes),
    )
    return CandidateGeometry(crs=crs, parts=(part,))


def boundary_area_m2(boundary: BoundaryGeometry) -> tuple[float, str]:
    """Plan-view area of a clip boundary, including holes."""

    return polygon_area_m2(boundary.exterior, crs=boundary.crs, holes=boundary.holes)


def envelope_of(boundary: BoundaryGeometry) -> tuple[float, float, float, float]:
    return ring_bbox(boundary.exterior)


def boundary_from_bbox(bbox: BoundingBox) -> BoundaryGeometry:
    return BoundaryGeometry.from_bbox(bbox)


def boundary_from_ref(ref: GeographicRef) -> BoundaryGeometry:
    """Build a rectangle from ``ref.bbox``. A point is not a search region."""

    if ref.bbox is None:
        raise IncompleteInputsError(
            "region/AOI GeographicRef must include bbox; a point is not a generation boundary"
        )
    if ref.crs is not None:
        require_same_crs(ref.crs, ref.bbox.crs, what="GeographicRef")
    return BoundaryGeometry.from_bbox(ref.bbox)


def clip_ring(
    subject: Sequence[tuple[float, float]], clipper: Sequence[tuple[float, float]]
) -> tuple[tuple[float, float], ...]:
    """Convex clipper, subject may be concave. Delegates to Sutherland-Hodgman."""

    return clip_polygon(subject, clipper)
