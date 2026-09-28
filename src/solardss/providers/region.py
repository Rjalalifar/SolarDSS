"""Geographic containment checks. No CRS transformation is performed."""

from __future__ import annotations

from solardss.data.geo import BoundingBox
from solardss.providers.request import FetchRequest


def point_in_bbox(longitude: float, latitude: float, bbox: BoundingBox) -> bool:
    """Axis-aligned containment in the bbox's stated CRS. No wrapping at 180°."""

    return bbox.min_lon <= longitude <= bbox.max_lon and bbox.min_lat <= latitude <= bbox.max_lat


def bboxes_intersect(left: BoundingBox, right: BoundingBox) -> bool:
    """True when two axis-aligned boxes overlap or touch. CRS labels must match."""

    if left.crs != right.crs:
        return False
    return not (
        left.max_lon < right.min_lon
        or left.min_lon > right.max_lon
        or left.max_lat < right.min_lat
        or left.min_lat > right.max_lat
    )


def request_point(request: FetchRequest) -> tuple[float, float] | None:
    """Longitude/latitude on the request location, if both are present."""

    location = request.location
    if location is None or location.longitude is None or location.latitude is None:
        return None
    return (location.longitude, location.latitude)


def request_bbox(request: FetchRequest) -> BoundingBox | None:
    """Preferred bbox: explicit request bbox, else the location's bbox."""

    if request.bbox is not None:
        return request.bbox
    if request.location is not None:
        return request.location.bbox
    return None


def request_place_id(request: FetchRequest) -> str | None:
    location = request.location
    if location is None:
        return None
    return location.place_id


def request_has_geometry(request: FetchRequest) -> bool:
    """True when the request includes a comparable point or bbox."""

    return request_point(request) is not None or request_bbox(request) is not None


def geometry_matches_bbox(request: FetchRequest, bbox: BoundingBox) -> bool | None:
    """Whether the request lies in ``bbox``.

    Returns ``None`` when the request has no comparable geometry (opaque name
    or place id only). Callers must not treat ``None`` as coverage.
    """

    point = request_point(request)
    if point is not None:
        return point_in_bbox(point[0], point[1], bbox)
    box = request_bbox(request)
    if box is not None:
        return bboxes_intersect(box, bbox)
    return None
