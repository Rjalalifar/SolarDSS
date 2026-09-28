"""Bounded OpenStreetMap / Overpass adapter for infrastructure and buildings.

SolarDSS does not import Ariadne Thread or any other project's OSM stack.
Queries are rendered from an explicit selector table and always include a
``[bbox:]`` clause. Global or unfiltered Overpass statements are rejected.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime
from math import cos, pi, radians
from typing import Final, Literal, Protocol

import httpx
from pydantic import BaseModel, ConfigDict, Field

from solardss.core.hashing import fingerprint_canonical
from solardss.core.jsonutil import JsonValue
from solardss.data.contracts import DatasetRef
from solardss.data.geo import BoundingBox
from solardss.exceptions import InvalidParameterError
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.indicators.ids import IndicatorId
from solardss.indicators.infrastructure import (
    CALC_NEAREST_DISTANCE,
    WGS84_AUTHALIC_RADIUS_M,
    DistanceHit,
    FeatureKind,
    GeometryKind,
    InfraFeature,
    analyze_nearest_distance,
    crs_is_geographic,
    demand_center_features,
    nearest_feature,
)
from solardss.indicators.rooftop import (
    CALC_FOOTPRINT_AREA,
    CALC_ROOF_ORIENTATION,
    CALC_ROOF_SLOPE,
    analyze_rooftop,
)
from solardss.provenance.coverage import SpatialCoverage, TemporalCoverage
from solardss.provenance.observation import (
    DataProvenance,
    IndicatorObservation,
    ObservationStatus,
    QualityFlag,
    ValueOrigin,
)
from solardss.providers.cache import DatasetCache
from solardss.providers.clock import Clock, UtcClock
from solardss.providers.contracts import (
    AvailabilityStatus,
    CoverageReport,
    HealthStatus,
    ProviderAvailability,
    ProviderHealth,
)
from solardss.providers.errors import (
    CacheError,
    OSMError,
    OSMHTTPError,
    OSMQueryBoundError,
    OSMResponseError,
)
from solardss.providers.region import request_bbox, request_point
from solardss.providers.request import FetchRequest, request_cache_key
from solardss.version import __version__

OSM_SOURCE_NAME: Final = "OpenStreetMap"
OSM_LICENSE: Final = "ODbL"
OVERPASS_DEFAULT_URL: Final = "https://overpass-api.de/api/interpreter"
OVERPASS_PATH_ENV: Final = "SOLARDSS_OVERPASS_URL"
OSM_USER_AGENT: Final = f"solardss/{__version__} (OSM Overpass client)"
DEFAULT_SEARCH_RADIUS_M: Final = 5000.0
MAX_SEARCH_RADIUS_M: Final = 50_000.0
MAX_BBOX_SPAN_DEG: Final = 1.0
DEFAULT_OVERPASS_TIMEOUT_S: Final = 25
DEFAULT_HTTP_TIMEOUT_S: Final = 30.0
DEFAULT_MAXSIZE_BYTES: Final = 16_777_216
_CACHE_SCHEMA: Final = "solardss.overpass.v1"
_RETRY_STATUS: Final = frozenset({429, 500, 502, 503, 504})

OsmType = Literal["node", "way", "relation"]

POWER_LINE_VALUES: Final = ("line", "minor_line", "cable")
SUBSTATION_VALUES: Final = ("substation", "sub_station")
HIGHWAY_VALUES: Final = (
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "service",
    "living_street",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
)
PLACE_VALUES: Final = ("city", "town", "village", "hamlet", "suburb", "municipality")

_UNFILTERED_STATEMENT = re.compile(r"(?m)^\s*(?:node|way|relation)\s*(?:\(\s*\)\s*)?;\s*$")
_GLOBAL_BBOX = re.compile(
    r"\[bbox:\s*-90(?:\.0+)?\s*,\s*-180(?:\.0+)?\s*,\s*90(?:\.0+)?\s*,\s*180(?:\.0+)?\s*\]",
    re.IGNORECASE,
)
_BBOX_CLAUSE = re.compile(
    r"\[bbox:\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,"
    r"\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\]",
    re.IGNORECASE,
)

_INFRA_KIND: dict[str, FeatureKind] = {
    IndicatorId.INFRA_DISTANCE_TO_POWER_LINE.value: FeatureKind.POWER_LINE,
    IndicatorId.INFRA_DISTANCE_TO_SUBSTATION.value: FeatureKind.SUBSTATION,
    IndicatorId.INFRA_DISTANCE_TO_ROAD.value: FeatureKind.ROAD,
    IndicatorId.INFRA_DISTANCE_TO_SETTLEMENT.value: FeatureKind.SETTLEMENT,
    IndicatorId.INFRA_DISTANCE_TO_DEMAND_CENTER.value: FeatureKind.DEMAND_CENTER,
}

_INFRA_SERVED: Final = frozenset(_INFRA_KIND)
_ROOF_SERVED: Final = frozenset(
    {
        IndicatorId.ROOF_BUILDING_FOOTPRINT_AREA.value,
        IndicatorId.ROOF_ORIENTATION.value,
        IndicatorId.ROOF_SLOPE.value,
    }
)
_ROOF_REFUSED: Final = {
    IndicatorId.ROOF_USABLE_AREA.value: (
        "OSM building footprint is not usable roof area; setbacks and shading are required"
    ),
    IndicatorId.ROOF_SHADING.value: "OpenStreetMap does not provide roof shading",
    IndicatorId.ROOF_STRUCTURAL_CAPACITY.value: ("building footprints are not structural capacity"),
}
_HOSTING_REFUSED = (
    "OpenStreetMap power features are not grid hosting capacity; "
    "grid existence is not available capacity"
)


@dataclass(frozen=True, slots=True)
class OverpassSelector:
    """One Overpass statement. ``value`` is None when only the key must exist."""

    osm_type: OsmType
    kind: FeatureKind
    key: str
    value: str | None = None

    def render(self) -> str:
        quoted_key = _escape_overpass(self.key)
        if self.value is None:
            return f'{self.osm_type}["{quoted_key}"]'
        return f'{self.osm_type}["{quoted_key}"="{_escape_overpass(self.value)}"]'


def _escape_overpass(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _selectors_for_values(
    osm_types: Sequence[OsmType], kind: FeatureKind, key: str, values: Sequence[str]
) -> tuple[OverpassSelector, ...]:
    return tuple(
        OverpassSelector(osm_type=osm_type, kind=kind, key=key, value=value)
        for osm_type in osm_types
        for value in values
    )


OVERPASS_SELECTORS: tuple[OverpassSelector, ...] = (
    *_selectors_for_values(("way",), FeatureKind.POWER_LINE, "power", POWER_LINE_VALUES),
    *_selectors_for_values(
        ("node", "way", "relation"), FeatureKind.SUBSTATION, "power", SUBSTATION_VALUES
    ),
    *_selectors_for_values(("way",), FeatureKind.ROAD, "highway", HIGHWAY_VALUES),
    *_selectors_for_values(
        ("node", "way", "relation"), FeatureKind.SETTLEMENT, "place", PLACE_VALUES
    ),
    OverpassSelector("way", FeatureKind.SETTLEMENT, "landuse", "residential"),
    OverpassSelector("relation", FeatureKind.SETTLEMENT, "landuse", "residential"),
    OverpassSelector("way", FeatureKind.BUILDING, "building", None),
    OverpassSelector("relation", FeatureKind.BUILDING, "building", None),
    OverpassSelector("node", FeatureKind.BUILDING, "building", None),
    OverpassSelector("way", FeatureKind.INDUSTRIAL, "landuse", "industrial"),
    OverpassSelector("relation", FeatureKind.INDUSTRIAL, "landuse", "industrial"),
)


class OverpassQuerySpec(BaseModel):
    """Bounded Overpass request. ``kinds`` selects from ``OVERPASS_SELECTORS``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bbox: BoundingBox
    kinds: tuple[FeatureKind, ...] = Field(min_length=1)
    timeout_seconds: int = DEFAULT_OVERPASS_TIMEOUT_S
    maxsize_bytes: int = DEFAULT_MAXSIZE_BYTES


class OverpassSnapshot(BaseModel):
    """Parsed Overpass extract. ``timestamp_osm_base`` is the OSM data timestamp."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    features: tuple[InfraFeature, ...]
    timestamp_osm_base: str | None = None
    copyright: str | None = None
    query: str
    bbox: BoundingBox
    remark: str | None = None


def selectors_for(kinds: Sequence[FeatureKind]) -> tuple[OverpassSelector, ...]:
    wanted = frozenset(kinds)
    selected = tuple(item for item in OVERPASS_SELECTORS if item.kind in wanted)
    if not selected:
        raise InvalidParameterError("no Overpass selectors for the requested feature kinds")
    return selected


def bbox_span_ok(bbox: BoundingBox) -> bool:
    if bbox.crs != "EPSG:4326" and not crs_is_geographic(bbox.crs):
        return False
    lat_span = bbox.max_lat - bbox.min_lat
    lon_span = bbox.max_lon - bbox.min_lon
    return lat_span <= MAX_BBOX_SPAN_DEG and lon_span <= MAX_BBOX_SPAN_DEG


def require_bounded_bbox(bbox: BoundingBox) -> BoundingBox:
    """Reject world-scale or non-geographic windows before any HTTP call."""

    if bbox.crs != "EPSG:4326" and not crs_is_geographic(bbox.crs):
        raise OSMQueryBoundError("Overpass queries require a geographic (EPSG:4326) bbox")
    world = (
        bbox.min_lat <= -90.0
        and bbox.max_lat >= 90.0
        and bbox.min_lon <= -180.0
        and bbox.max_lon >= 180.0
    )
    if world:
        raise OSMQueryBoundError("global Overpass bboxes are not allowed")
    if not bbox_span_ok(bbox):
        raise OSMQueryBoundError(
            f"Overpass bbox span exceeds {MAX_BBOX_SPAN_DEG} degrees; query is not bounded"
        )
    return bbox


def build_overpass_query(spec: OverpassQuerySpec) -> str:
    """Render a bbox-bounded Overpass QL string from explicit selectors."""

    bbox = require_bounded_bbox(spec.bbox)
    if spec.timeout_seconds < 1 or spec.timeout_seconds > 60:
        raise InvalidParameterError("Overpass timeout_seconds must be in [1, 60]")
    if spec.maxsize_bytes < 1_048_576 or spec.maxsize_bytes > 67_108_864:
        raise InvalidParameterError("Overpass maxsize_bytes is outside the allowed range")
    statements = [item.render() + ";" for item in selectors_for(spec.kinds)]
    header = (
        f"[out:json][timeout:{spec.timeout_seconds}][maxsize:{spec.maxsize_bytes}]"
        f"[bbox:{bbox.min_lat:.7f},{bbox.min_lon:.7f},{bbox.max_lat:.7f},{bbox.max_lon:.7f}];"
    )
    body = "\n  ".join(statements)
    query = f"{header}\n(\n  {body}\n);\nout geom;"
    assert_bounded_overpass_query(query, bbox)
    return query


def assert_bounded_overpass_query(query: str, bbox: BoundingBox | None = None) -> None:
    """Reject unrestricted Overpass QL. Used on generated and injected queries."""

    compact = query.strip()
    if not compact:
        raise OSMQueryBoundError("Overpass query is empty")
    if _GLOBAL_BBOX.search(compact):
        raise OSMQueryBoundError("global Overpass bbox is not allowed")
    if _UNFILTERED_STATEMENT.search(compact):
        raise OSMQueryBoundError("unfiltered Overpass node/way/relation dump is not allowed")
    match = _BBOX_CLAUSE.search(compact)
    if match is None:
        raise OSMQueryBoundError("Overpass query must include a [bbox:s,w,n,e] clause")
    south, west, north, east = (float(match.group(i)) for i in range(1, 5))
    parsed = BoundingBox(min_lon=west, min_lat=south, max_lon=east, max_lat=north)
    require_bounded_bbox(parsed)
    if bbox is not None:
        if abs(south - bbox.min_lat) > 1e-7 or abs(west - bbox.min_lon) > 1e-7:
            raise OSMQueryBoundError("Overpass bbox does not match the request window")
        if abs(north - bbox.max_lat) > 1e-7 or abs(east - bbox.max_lon) > 1e-7:
            raise OSMQueryBoundError("Overpass bbox does not match the request window")


def classify_osm_tags(tags: Mapping[str, str]) -> tuple[FeatureKind, ...]:
    """Map OSM tags onto SolarDSS feature kinds using the selector table."""

    kinds: list[FeatureKind] = []
    power = tags.get("power")
    if power in POWER_LINE_VALUES:
        kinds.append(FeatureKind.POWER_LINE)
    if power in SUBSTATION_VALUES:
        kinds.append(FeatureKind.SUBSTATION)
    highway = tags.get("highway")
    if highway in HIGHWAY_VALUES:
        kinds.append(FeatureKind.ROAD)
    place = tags.get("place")
    if place in PLACE_VALUES or tags.get("landuse") == "residential":
        kinds.append(FeatureKind.SETTLEMENT)
    if "building" in tags:
        kinds.append(FeatureKind.BUILDING)
    if tags.get("landuse") == "industrial":
        kinds.append(FeatureKind.INDUSTRIAL)
    return tuple(dict.fromkeys(kinds))


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _coords_from_geometry(geometry: object) -> tuple[tuple[float, float], ...] | None:
    if not isinstance(geometry, list) or not geometry:
        return None
    points: list[tuple[float, float]] = []
    for item in geometry:
        if not isinstance(item, dict):
            return None
        lon = _as_float(item.get("lon"))
        lat = _as_float(item.get("lat"))
        if lon is None or lat is None:
            return None
        points.append((lon, lat))
    return tuple(points)


def _geometry_kind(kind: FeatureKind, coords: Sequence[tuple[float, float]]) -> GeometryKind:
    if kind in {FeatureKind.POWER_LINE, FeatureKind.ROAD}:
        return GeometryKind.LINESTRING if len(coords) >= 2 else GeometryKind.POINT
    if len(coords) >= 3:
        return GeometryKind.POLYGON
    if len(coords) >= 2:
        return GeometryKind.LINESTRING
    return GeometryKind.POINT


def _tag_pairs(tags: Mapping[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple((str(key), str(value)) for key, value in sorted(tags.items()))


def _feature(
    kind: FeatureKind,
    osm_type: str,
    osm_id: int,
    coords: tuple[tuple[float, float], ...],
    holes: tuple[tuple[tuple[float, float], ...], ...],
    tags: Mapping[str, str],
) -> InfraFeature:
    return InfraFeature(
        kind=kind,
        geometry_kind=_geometry_kind(kind, coords),
        coordinates=coords,
        holes=holes,
        osm_type=osm_type,
        osm_id=osm_id,
        name=tags.get("name"),
        tags=_tag_pairs(tags),
    )


def _features_from_element(element: Mapping[str, object]) -> tuple[InfraFeature, ...]:
    osm_type = element.get("type")
    osm_id = element.get("id")
    raw_tags = element.get("tags")
    if osm_type not in {"node", "way", "relation"}:
        return ()
    if not isinstance(osm_id, int) or isinstance(osm_id, bool):
        return ()
    tags: dict[str, str] = {}
    if isinstance(raw_tags, dict):
        tags = {str(key): str(value) for key, value in raw_tags.items() if value is not None}
    kinds = classify_osm_tags(tags)
    if not kinds:
        return ()
    coords: tuple[tuple[float, float], ...] | None = None
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()
    if osm_type == "node":
        lon = _as_float(element.get("lon"))
        lat = _as_float(element.get("lat"))
        if lon is not None and lat is not None:
            coords = ((lon, lat),)
    elif osm_type == "way":
        coords = _coords_from_geometry(element.get("geometry"))
    else:
        members = element.get("members")
        outers: list[tuple[tuple[float, float], ...]] = []
        inners: list[tuple[tuple[float, float], ...]] = []
        if isinstance(members, list):
            for member in members:
                if not isinstance(member, dict):
                    continue
                geom = _coords_from_geometry(member.get("geometry"))
                if geom is None:
                    continue
                if member.get("role") == "inner":
                    inners.append(geom)
                else:
                    outers.append(geom)
        if len(outers) == 1:
            coords = outers[0]
            holes = tuple(inners)
        elif outers:
            return tuple(
                _feature(kind, osm_type, osm_id, ring, (), tags)
                for ring in outers
                for kind in kinds
            )
    if coords is None:
        return ()
    return tuple(_feature(kind, osm_type, osm_id, coords, holes, tags) for kind in kinds)


def parse_overpass_json(
    payload: Mapping[str, object], *, query: str, bbox: BoundingBox
) -> OverpassSnapshot:
    """Interpret an Overpass ``out geom`` JSON document."""

    remark_raw = payload.get("remark")
    remark = remark_raw if isinstance(remark_raw, str) else None
    osm3s = payload.get("osm3s")
    timestamp: str | None = None
    copyright_text: str | None = None
    if isinstance(osm3s, dict):
        stamp = osm3s.get("timestamp_osm_base")
        timestamp = stamp if isinstance(stamp, str) else None
        copy = osm3s.get("copyright")
        copyright_text = copy if isinstance(copy, str) else None
    elements = payload.get("elements")
    if elements is None:
        elements = []
    if not isinstance(elements, list):
        raise OSMResponseError("Overpass JSON elements must be a list")
    features: list[InfraFeature] = []
    for item in elements:
        if isinstance(item, dict):
            features.extend(_features_from_element(item))
    return OverpassSnapshot(
        features=tuple(features),
        timestamp_osm_base=timestamp,
        copyright=copyright_text,
        query=query,
        bbox=bbox,
        remark=remark,
    )


def _feature_intersects_bbox(feature: InfraFeature, bbox: BoundingBox) -> bool:
    lons = [point[0] for point in feature.coordinates]
    lats = [point[1] for point in feature.coordinates]
    if not lons:
        return False
    return not (
        max(lons) < bbox.min_lon
        or min(lons) > bbox.max_lon
        or max(lats) < bbox.min_lat
        or min(lats) > bbox.max_lat
    )


class OSMFeatureSource(Protocol):
    """Return features for a bounded Overpass spec. File or HTTP implementations."""

    def query(self, spec: OverpassQuerySpec) -> OverpassSnapshot:
        """Run a bounded extract. Must not issue a global query."""


@dataclass(frozen=True, slots=True)
class MemoryOSMSource:
    """In-memory OSM extract used by tests. Filters by bbox and kind."""

    features: tuple[InfraFeature, ...]
    timestamp_osm_base: str | None = "2024-06-01T12:00:00Z"
    copyright: str | None = "The data included in this document is from www.openstreetmap.org."

    def query(self, spec: OverpassQuerySpec) -> OverpassSnapshot:
        query = build_overpass_query(spec)
        wanted = frozenset(spec.kinds)
        selected = tuple(
            item
            for item in self.features
            if item.kind in wanted and _feature_intersects_bbox(item, spec.bbox)
        )
        return OverpassSnapshot(
            features=selected,
            timestamp_osm_base=self.timestamp_osm_base,
            copyright=self.copyright,
            query=query,
            bbox=spec.bbox,
        )


class OverpassClient:
    """POST-only Overpass client with timeout, bounded retries, and raw caching."""

    def __init__(
        self,
        *,
        base_url: str = OVERPASS_DEFAULT_URL,
        timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_S,
        max_attempts: int = 3,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        user_agent: str = OSM_USER_AGENT,
    ) -> None:
        if timeout_seconds <= 0:
            raise InvalidParameterError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise InvalidParameterError("max_attempts must be >= 1")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._cache = cache
        self._clock: Clock = UtcClock() if clock is None else clock
        self._sleep = sleep
        self._user_agent = user_agent
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(
                timeout=httpx.Timeout(timeout_seconds),
                headers={"User-Agent": user_agent, "Accept": "application/json"},
                transport=transport,
                follow_redirects=True,
            )

    @property
    def base_url(self) -> str:
        return self._base_url

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OverpassClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def query(self, spec: OverpassQuerySpec) -> OverpassSnapshot:
        query = build_overpass_query(spec)
        cache_key = fingerprint_canonical({"base_url": self._base_url, "query": query})
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return self._snapshot_from_cache(cached, query=query, bbox=spec.bbox)
        retrieved_at = self._clock.now()
        last_message = "Overpass request failed"
        last_status: int | None = None
        for attempt in range(self._max_attempts):
            try:
                response = self._client.post(self._base_url, data={"data": query})
            except httpx.TimeoutException as exc:
                last_message = f"Overpass timeout after {self._timeout}s"
                last_status = None
                if attempt + 1 >= self._max_attempts:
                    raise OSMHTTPError(last_message) from exc
                self._backoff(attempt, retry_after=None)
                continue
            except httpx.TransportError as exc:
                last_message = "Overpass transport error"
                last_status = None
                if attempt + 1 >= self._max_attempts:
                    raise OSMHTTPError(last_message) from exc
                self._backoff(attempt, retry_after=None)
                continue
            last_status = response.status_code
            if response.status_code in _RETRY_STATUS:
                last_message = f"Overpass HTTP {response.status_code}"
                if attempt + 1 >= self._max_attempts:
                    raise OSMHTTPError(last_message, status_code=response.status_code)
                self._backoff(attempt, retry_after=response.headers.get("Retry-After"))
                continue
            if response.status_code != 200:
                raise OSMHTTPError(
                    f"Overpass HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            payload = _parse_overpass_payload(response.content)
            snapshot = parse_overpass_json(payload, query=query, bbox=spec.bbox)
            if self._cache is not None:
                self._cache.put(
                    cache_key,
                    _encode_overpass_cache(payload, query=query, retrieved_at=retrieved_at),
                )
            return snapshot
        raise OSMHTTPError(last_message, status_code=last_status)

    def _snapshot_from_cache(
        self, blob: bytes, *, query: str, bbox: BoundingBox
    ) -> OverpassSnapshot:
        try:
            raw: object = json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CacheError("cached Overpass payload is not JSON") from exc
        if not isinstance(raw, dict) or raw.get("schema_id") != _CACHE_SCHEMA:
            raise CacheError("cached Overpass payload has unknown schema")
        body = raw.get("payload")
        stored_query = raw.get("query")
        if not isinstance(body, dict) or not isinstance(stored_query, str):
            raise CacheError("cached Overpass payload is incomplete")
        if stored_query != query:
            raise CacheError("cached Overpass query does not match the request")
        return parse_overpass_json(body, query=query, bbox=bbox)

    def _backoff(self, attempt: int, *, retry_after: str | None) -> None:
        if self._sleep is None:
            return
        seconds = 0.25 * (2**attempt)
        if retry_after is not None:
            with suppress(ValueError):
                seconds = max(seconds, float(retry_after))
        self._sleep(min(seconds, 2.0))


def _parse_overpass_payload(content: bytes) -> dict[str, object]:
    try:
        raw: object = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OSMResponseError("Overpass response is not JSON") from exc
    if not isinstance(raw, dict):
        raise OSMResponseError("Overpass JSON must be an object")
    return raw


def _encode_overpass_cache(
    payload: Mapping[str, object], *, query: str, retrieved_at: datetime
) -> bytes:
    envelope = {
        "schema_id": _CACHE_SCHEMA,
        "query": query,
        "retrieved_at": retrieved_at.isoformat(),
        "payload": payload,
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def search_bbox_from_point(
    lon: float, lat: float, radius_m: float, *, crs: str = "EPSG:4326"
) -> BoundingBox:
    """Geographic bbox around a point, capped at ``MAX_BBOX_SPAN_DEG``."""

    if radius_m <= 0.0:
        raise InvalidParameterError("search radius must be positive")
    if radius_m > MAX_SEARCH_RADIUS_M:
        raise OSMQueryBoundError(
            f"search radius {radius_m} m exceeds the {MAX_SEARCH_RADIUS_M} m bound"
        )
    metres_per_deg = WGS84_AUTHALIC_RADIUS_M * pi / 180.0
    dlat = min(radius_m / metres_per_deg, MAX_BBOX_SPAN_DEG / 2.0)
    cos_lat = max(abs(cos(radians(lat))), 0.05)
    dlon = min(radius_m / (metres_per_deg * cos_lat), MAX_BBOX_SPAN_DEG / 2.0)
    return BoundingBox(
        min_lon=max(-180.0, lon - dlon),
        min_lat=max(-90.0, lat - dlat),
        max_lon=min(180.0, lon + dlon),
        max_lat=min(90.0, lat + dlat),
        crs=crs,
    )


def _float_param(parameters: Mapping[str, JsonValue], key: str) -> float | None:
    value = parameters.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _int_param(parameters: Mapping[str, JsonValue], key: str) -> int | None:
    value = parameters.get(key)
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    return None


def _float_list_param(parameters: Mapping[str, JsonValue], key: str) -> tuple[float, ...] | None:
    value = parameters.get(key)
    if value is None:
        return None
    if isinstance(value, list):
        items: list[float] = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int | float):
                return None
            items.append(float(item))
        return tuple(items)
    return None


def parse_demand_centers(
    parameters: Mapping[str, JsonValue],
) -> tuple[tuple[float, float], ...] | None:
    """Caller-declared demand-center points. Settlements are not a fallback."""

    lon = _float_param(parameters, "demand_center_lon")
    lat = _float_param(parameters, "demand_center_lat")
    if (lon is None) != (lat is None):
        raise InvalidParameterError(
            "demand_center_lon and demand_center_lat must be supplied together"
        )
    points: list[tuple[float, float]] = []
    if lon is not None and lat is not None:
        points.append((lon, lat))
    lons = _float_list_param(parameters, "demand_center_lons")
    lats = _float_list_param(parameters, "demand_center_lats")
    if (lons is None) != (lats is None):
        raise InvalidParameterError(
            "demand_center_lons and demand_center_lats must be supplied together"
        )
    if lons is not None and lats is not None:
        if len(lons) != len(lats):
            raise InvalidParameterError(
                "demand_center_lons and demand_center_lats must have equal length"
            )
        points.extend(zip(lons, lats, strict=True))
    if not points:
        return None
    return tuple(points)


def _search_radius_m(parameters: Mapping[str, JsonValue]) -> float:
    raw = _float_param(parameters, "search_radius_m")
    radius = DEFAULT_SEARCH_RADIUS_M if raw is None else raw
    if radius <= 0.0:
        raise InvalidParameterError("search_radius_m must be positive")
    if radius > MAX_SEARCH_RADIUS_M:
        raise OSMQueryBoundError(
            f"search_radius_m {radius} exceeds the {MAX_SEARCH_RADIUS_M} m bound"
        )
    return radius


@dataclass(frozen=True, slots=True)
class OSMAnalysisBundle:
    """Parsed extract plus mapped indicator observations."""

    snapshot: OverpassSnapshot | None
    observations: tuple[IndicatorObservation, ...]
    origin: tuple[float, float] | None
    quality_warnings: tuple[str, ...]


class OSMOverpassProvider:
    """Bounded OSM adapter. Serves infrastructure distances and rooftop tags."""

    provider_id_default = "osm.overpass"
    source_name_default = OSM_SOURCE_NAME
    served = _INFRA_SERVED | _ROOF_SERVED

    def __init__(
        self,
        *,
        source: OSMFeatureSource | None = None,
        client: OverpassClient | None = None,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        transport: httpx.BaseTransport | None = None,
        base_url: str | None = None,
        timeout_seconds: float = DEFAULT_HTTP_TIMEOUT_S,
        sleep: Callable[[float], None] | None = None,
        provider_id: str | None = None,
        environ: Mapping[str, str] | None = None,
        served: frozenset[str] | None = None,
    ) -> None:
        self._source = source
        self._clock: Clock = UtcClock() if clock is None else clock
        self._provider_id = provider_id or self.provider_id_default
        self._served = served if served is not None else self.served
        env = environ if environ is not None else None
        url = base_url
        if url is None:
            raw = None if env is None else env.get(OVERPASS_PATH_ENV)
            if raw is None and environ is None:
                import os

                raw = os.environ.get(OVERPASS_PATH_ENV)
            url = raw
        self._owns_client = client is None and source is None
        if client is not None:
            self._client: OverpassClient | None = client
        elif source is None:
            self._client = OverpassClient(
                base_url=url or OVERPASS_DEFAULT_URL,
                timeout_seconds=timeout_seconds,
                cache=cache,
                clock=self._clock,
                transport=transport,
                sleep=sleep,
            )
        else:
            self._client = None

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            self._client.close()

    def __enter__(self) -> OSMOverpassProvider:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def health(self) -> ProviderHealth:
        checked_at = self._clock.now()
        details: dict[str, JsonValue] = {
            "source": OSM_SOURCE_NAME,
            "license": OSM_LICENSE,
            "overpass_url": None if self._client is None else self._client.base_url,
            "injected_source": self._source is not None,
            "max_search_radius_m": MAX_SEARCH_RADIUS_M,
            "max_bbox_span_deg": MAX_BBOX_SPAN_DEG,
        }
        return ProviderHealth(
            status=HealthStatus.OK,
            provider_id=self._provider_id,
            reason="OpenStreetMap Overpass adapter is configured",
            checked_at=checked_at,
            details=details,
        )

    def availability(self, request: FetchRequest) -> ProviderAvailability:
        indicator = request.indicator_id
        if indicator == IndicatorId.INFRA_GRID_HOSTING_CAPACITY.value:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason=_HOSTING_REFUSED,
            )
        if indicator in _ROOF_REFUSED:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason=_ROOF_REFUSED[indicator],
            )
        if indicator not in self._served:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason="indicator is not served by the OpenStreetMap provider",
            )
        try:
            self._window(request)
        except OSMQueryBoundError as exc:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason=exc.message,
            )
        except InvalidParameterError as exc:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason=exc.message,
            )
        point = request_point(request)
        box = request_bbox(request)
        if point is None and box is None:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason="OpenStreetMap analysis requires a point or bbox",
            )
        if indicator == IndicatorId.INFRA_DISTANCE_TO_DEMAND_CENTER.value:
            try:
                centers = parse_demand_centers(request.parameters)
            except InvalidParameterError as exc:
                return ProviderAvailability(
                    status=AvailabilityStatus.UNAVAILABLE,
                    provider_id=self._provider_id,
                    retryable=False,
                    indicator_id=indicator,
                    reason=exc.message,
                )
            osm_id = _int_param(request.parameters, "demand_center_osm_id")
            if centers is None and osm_id is None:
                return ProviderAvailability(
                    status=AvailabilityStatus.NOT_CONFIGURED,
                    provider_id=self._provider_id,
                    retryable=False,
                    indicator_id=indicator,
                    reason=(
                        "INFRA_DISTANCE_TO_DEMAND_CENTER requires declared "
                        "demand_center_lon/lat (or demand_center_osm_id); "
                        "settlements are not treated as demand centers"
                    ),
                )
        return ProviderAvailability(
            status=AvailabilityStatus.AVAILABLE,
            provider_id=self._provider_id,
            retryable=False,
            indicator_id=indicator,
            reason="bounded OpenStreetMap extract can be queried for this request",
        )

    def coverage(self, request: FetchRequest) -> CoverageReport:
        availability = self.availability(request)
        spatial = SpatialCoverage(
            resolution="osm-vector",
            crs="EPSG:4326",
            bbox=request_bbox(request),
        )
        return CoverageReport(
            status=availability.status,
            provider_id=self._provider_id,
            spatial=spatial,
            temporal=request.temporal or TemporalCoverage(resolution="snapshot"),
            reason=availability.reason,
        )

    def fetch(self, request: FetchRequest) -> IndicatorObservation:
        bundle = self.fetch_osm(request)
        for item in bundle.observations:
            if item.indicator_id == request.indicator_id:
                return item
        return self._miss(
            request,
            ObservationStatus.UNAVAILABLE,
            "indicator was not produced from the OpenStreetMap extract",
            self._clock.now(),
        )

    def fetch_osm(self, request: FetchRequest) -> OSMAnalysisBundle:
        availability = self.availability(request)
        retrieved_at = self._clock.now()
        if availability.status is not AvailabilityStatus.AVAILABLE:
            miss = self._miss(
                request,
                ObservationStatus(availability.status.value),
                availability.reason,
                retrieved_at,
            )
            return OSMAnalysisBundle(
                snapshot=None,
                observations=(miss,),
                origin=None,
                quality_warnings=(availability.reason,),
            )
        try:
            origin, bbox = self._window(request)
            snapshot = None
            features: tuple[InfraFeature, ...] = ()
            timestamp = None
            copyright_text = None
            if request.indicator_id == IndicatorId.INFRA_DISTANCE_TO_DEMAND_CENTER.value:
                centers = parse_demand_centers(request.parameters)
                osm_id = _int_param(request.parameters, "demand_center_osm_id")
                declared = demand_center_features(centers or ())
                if osm_id is not None:
                    snapshot = self._query(bbox, (FeatureKind.INDUSTRIAL, FeatureKind.SETTLEMENT))
                    timestamp = snapshot.timestamp_osm_base
                    copyright_text = snapshot.copyright
                    looked_up = tuple(
                        InfraFeature(
                            kind=FeatureKind.DEMAND_CENTER,
                            geometry_kind=item.geometry_kind,
                            coordinates=item.coordinates,
                            holes=item.holes,
                            osm_type=item.osm_type,
                            osm_id=item.osm_id,
                            name=item.name,
                            tags=item.tags,
                        )
                        for item in snapshot.features
                        if item.osm_id == osm_id
                    )
                    features = declared + looked_up
                else:
                    features = declared
            else:
                kinds = self._kinds_for(request.indicator_id)
                snapshot = self._query(bbox, kinds)
                features = snapshot.features
                timestamp = snapshot.timestamp_osm_base
                copyright_text = snapshot.copyright
            observation = self._observation(
                request,
                origin=origin,
                bbox=bbox,
                features=features,
                timestamp=timestamp,
                copyright_text=copyright_text,
                retrieved_at=retrieved_at,
            )
        except OSMQueryBoundError as exc:
            miss = self._miss(request, ObservationStatus.UNAVAILABLE, exc.message, retrieved_at)
            return OSMAnalysisBundle(
                snapshot=None,
                observations=(miss,),
                origin=None,
                quality_warnings=(exc.message,),
            )
        except (OSMHTTPError, OSMResponseError, OSMError) as exc:
            miss = self._miss(
                request, ObservationStatus.TEMPORARY_FAILURE, exc.message, retrieved_at
            )
            return OSMAnalysisBundle(
                snapshot=None,
                observations=(miss,),
                origin=None,
                quality_warnings=(exc.message,),
            )
        except InvalidParameterError as exc:
            miss = self._miss(request, ObservationStatus.UNAVAILABLE, exc.message, retrieved_at)
            return OSMAnalysisBundle(
                snapshot=None,
                observations=(miss,),
                origin=None,
                quality_warnings=(exc.message,),
            )
        return OSMAnalysisBundle(
            snapshot=snapshot,
            observations=(observation,),
            origin=origin,
            quality_warnings=observation.assumptions,
        )

    def _kinds_for(self, indicator: str) -> tuple[FeatureKind, ...]:
        if indicator in _ROOF_SERVED:
            return (FeatureKind.BUILDING,)
        kind = _INFRA_KIND.get(indicator)
        if kind is None or kind is FeatureKind.DEMAND_CENTER:
            raise InvalidParameterError(f"OpenStreetMap provider does not map {indicator}")
        return (kind, FeatureKind.INDUSTRIAL)

    def _query(self, bbox: BoundingBox, kinds: Sequence[FeatureKind]) -> OverpassSnapshot:
        spec = OverpassQuerySpec(bbox=bbox, kinds=tuple(kinds))
        if self._source is not None:
            return self._source.query(spec)
        if self._client is None:
            raise OSMError("OpenStreetMap source is not configured")
        return self._client.query(spec)

    def _window(self, request: FetchRequest) -> tuple[tuple[float, float], BoundingBox]:
        radius = _search_radius_m(request.parameters)
        point = request_point(request)
        box = request_bbox(request)
        loc_crs = "EPSG:4326"
        if request.location is not None and request.location.crs:
            loc_crs = request.location.crs
        elif box is not None:
            loc_crs = box.crs
        if loc_crs != "EPSG:4326" and not crs_is_geographic(loc_crs):
            raise OSMQueryBoundError("OpenStreetMap adapter requires geographic coordinates")
        if point is not None:
            origin = point
            search = search_bbox_from_point(point[0], point[1], radius, crs=loc_crs)
            if box is not None:
                min_lon = max(search.min_lon, box.min_lon)
                min_lat = max(search.min_lat, box.min_lat)
                max_lon = min(search.max_lon, box.max_lon)
                max_lat = min(search.max_lat, box.max_lat)
                if min_lon > max_lon or min_lat > max_lat:
                    raise OSMQueryBoundError("search radius and request bbox do not intersect")
                search = BoundingBox(
                    min_lon=min_lon,
                    min_lat=min_lat,
                    max_lon=max_lon,
                    max_lat=max_lat,
                    crs=loc_crs,
                )
            return origin, require_bounded_bbox(search)
        if box is None:
            raise InvalidParameterError("OpenStreetMap analysis requires a point or bbox")
        origin = ((box.min_lon + box.max_lon) / 2.0, (box.min_lat + box.max_lat) / 2.0)
        return origin, require_bounded_bbox(box)

    def _select_building(
        self,
        origin: tuple[float, float],
        features: Sequence[InfraFeature],
        request: FetchRequest,
    ) -> InfraFeature | None:
        buildings = [item for item in features if item.kind is FeatureKind.BUILDING]
        osm_id = _int_param(request.parameters, "osm_id")
        if osm_id is not None:
            for item in buildings:
                if item.osm_id == osm_id:
                    return item
            return None
        containing: list[InfraFeature] = []
        for item in buildings:
            hit = nearest_feature(
                origin[0], origin[1], (item,), crs="EPSG:4326", kind=FeatureKind.BUILDING
            )
            if hit is not None and hit.inside_polygon:
                containing.append(item)
        if containing:
            ranked: list[tuple[float, InfraFeature]] = []
            for item in containing:
                rooftop = analyze_rooftop(item, crs="EPSG:4326")
                area = rooftop.footprint.area_m2 if rooftop.footprint is not None else float("inf")
                ranked.append((area, item))
            ranked.sort(key=lambda pair: (pair[0], pair[1].osm_id or 0))
            return ranked[0][1]
        hit = nearest_feature(
            origin[0], origin[1], buildings, crs="EPSG:4326", kind=FeatureKind.BUILDING
        )
        return None if hit is None else hit.feature

    def _observation(
        self,
        request: FetchRequest,
        *,
        origin: tuple[float, float],
        bbox: BoundingBox,
        features: Sequence[InfraFeature],
        timestamp: str | None,
        copyright_text: str | None,
        retrieved_at: datetime,
    ) -> IndicatorObservation:
        indicator = request.indicator_id
        industrial_count = sum(1 for item in features if item.kind is FeatureKind.INDUSTRIAL)
        dataset = DatasetRef(
            name=OSM_SOURCE_NAME,
            media_type="application/json",
            uri=None if self._client is None else self._client.base_url,
            notes=(
                f"source={OSM_SOURCE_NAME}",
                f"license={OSM_LICENSE}",
                f"timestamp_osm_base={timestamp or 'unknown'}",
                f"industrial_areas={industrial_count}",
                f"search_bbox={bbox.min_lat},{bbox.min_lon},{bbox.max_lat},{bbox.max_lon}",
            ),
        )
        spatial = SpatialCoverage(resolution="osm-vector", crs="EPSG:4326", bbox=bbox)
        temporal = request.temporal or TemporalCoverage(
            resolution="snapshot",
            aggregation=timestamp,
        )
        if indicator in _ROOF_SERVED:
            return self._roof_observation(
                request,
                origin=origin,
                features=features,
                dataset=dataset,
                spatial=spatial,
                temporal=temporal,
                timestamp=timestamp,
                copyright_text=copyright_text,
                retrieved_at=retrieved_at,
            )
        kind = _INFRA_KIND[indicator]
        matching = tuple(item for item in features if item.kind is kind)
        analysis = analyze_nearest_distance(
            origin[0], origin[1], matching, kind=kind, crs="EPSG:4326"
        )
        hit = analysis.hit
        if hit is None:
            reason = (
                f"no OSM {kind.value} features in the bounded window"
                if kind is not FeatureKind.DEMAND_CENTER
                else "declared demand center was not found in the bounded window"
            )
            return self._miss(
                request,
                ObservationStatus.MISSING_COVERAGE,
                reason,
                retrieved_at,
                dataset=dataset,
                spatial=spatial,
            )
        extra = self._distance_assumptions(hit, timestamp, copyright_text, kind)
        return IndicatorObservation(
            indicator_id=indicator,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=OSM_SOURCE_NAME,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=ValueProvenanceKind.DERIVED_METHOD,
                value_origin=ValueOrigin.DERIVED,
                source_version=timestamp,
                calculation_method=CALC_NEAREST_DISTANCE,
                dataset=dataset,
            ),
            spatial=spatial,
            temporal=temporal,
            value=float(hit.distance_m),
            unit="m",
            quality_flags=(QualityFlag.NOMINAL.value,),
            assumptions=extra,
        )

    def _roof_observation(
        self,
        request: FetchRequest,
        *,
        origin: tuple[float, float],
        features: Sequence[InfraFeature],
        dataset: DatasetRef,
        spatial: SpatialCoverage,
        temporal: TemporalCoverage,
        timestamp: str | None,
        copyright_text: str | None,
        retrieved_at: datetime,
    ) -> IndicatorObservation:
        building = self._select_building(origin, features, request)
        rooftop = analyze_rooftop(building, crs="EPSG:4326")
        indicator = request.indicator_id
        if indicator == IndicatorId.ROOF_BUILDING_FOOTPRINT_AREA.value:
            value = None if rooftop.footprint is None else rooftop.footprint.area_m2
            unit = "m2"
            method = CALC_FOOTPRINT_AREA
        elif indicator == IndicatorId.ROOF_ORIENTATION.value:
            value = rooftop.roof.orientation_deg
            unit = "deg"
            method = CALC_ROOF_ORIENTATION
        else:
            value = rooftop.roof.slope_deg
            unit = "deg"
            method = CALC_ROOF_SLOPE
        preserved = ", ".join(f"{key}={val}" for key, val in rooftop.roof.preserved)
        copy_note = (copyright_text,) if copyright_text else ()
        assumptions = (
            f"source={OSM_SOURCE_NAME}",
            f"timestamp_osm_base={timestamp or 'unknown'}",
            f"license={OSM_LICENSE}",
            *rooftop.roof.notes,
            f"preserved_tags={preserved or 'none'}",
            "OSM roof tags are not a complete or surveyed roof model",
            "footprint is not usable PV area and not structural capacity",
            *copy_note,
        )
        if value is None:
            missing_building = building is None
            return self._miss(
                request,
                ObservationStatus.MISSING_COVERAGE
                if missing_building
                else ObservationStatus.UNAVAILABLE,
                (
                    "no OSM building in the bounded window"
                    if missing_building
                    else "OSM tags/geometry do not support this rooftop indicator"
                ),
                retrieved_at,
                dataset=dataset,
                spatial=spatial,
            )
        return IndicatorObservation(
            indicator_id=indicator,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=OSM_SOURCE_NAME,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=ValueProvenanceKind.DERIVED_METHOD,
                value_origin=ValueOrigin.DERIVED,
                source_version=timestamp,
                calculation_method=method,
                dataset=dataset,
            ),
            spatial=spatial,
            temporal=temporal,
            value=float(value),
            unit=unit,
            quality_flags=(QualityFlag.NOMINAL.value,),
            assumptions=assumptions,
        )

    def _distance_assumptions(
        self,
        hit: DistanceHit,
        timestamp: str | None,
        copyright_text: str | None,
        kind: FeatureKind,
    ) -> tuple[str, ...]:
        feature = hit.feature
        notes = [
            f"source={OSM_SOURCE_NAME}",
            f"timestamp_osm_base={timestamp or 'unknown'}",
            f"license={OSM_LICENSE}",
            f"distance_method={hit.method}",
            f"nearest_osm_type={feature.osm_type}",
            f"nearest_osm_id={feature.osm_id}",
            f"nearest_kind={kind.value}",
            "distance is straight-line, not network routing",
            "grid existence is not hosting capacity",
        ]
        if kind is FeatureKind.DEMAND_CENTER:
            notes.append("demand center is caller-declared; not inferred from OSM settlements")
        if hit.inside_polygon:
            notes.append("origin lies inside the nearest polygon (distance 0)")
        if copyright_text:
            notes.append(copyright_text)
        return tuple(notes)

    def _miss(
        self,
        request: FetchRequest,
        status: ObservationStatus,
        reason: str,
        retrieved_at: datetime,
        *,
        dataset: DatasetRef | None = None,
        spatial: SpatialCoverage | None = None,
    ) -> IndicatorObservation:
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=status,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=OSM_SOURCE_NAME,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=ValueProvenanceKind.MEASURED_OR_PROVIDER,
                value_origin=ValueOrigin.RAW,
                dataset=dataset or DatasetRef(name=OSM_SOURCE_NAME, media_type="application/json"),
            ),
            spatial=spatial or SpatialCoverage(resolution="osm-vector", crs="EPSG:4326"),
            temporal=request.temporal or TemporalCoverage(resolution="snapshot"),
            assumptions=(reason, f"source={OSM_SOURCE_NAME}"),
        )


class OSMInfrastructureProvider(OSMOverpassProvider):
    """OpenStreetMap distances to power, roads, and settlements."""

    provider_id_default = "osm.infrastructure"
    served = _INFRA_SERVED


class OSMBuildingProvider(OSMOverpassProvider):
    """OpenStreetMap building footprints and roof tags."""

    provider_id_default = "osm.building"
    served = _ROOF_SERVED


__all__ = [
    "DEFAULT_SEARCH_RADIUS_M",
    "HIGHWAY_VALUES",
    "MAX_BBOX_SPAN_DEG",
    "MAX_SEARCH_RADIUS_M",
    "OSM_SOURCE_NAME",
    "OVERPASS_DEFAULT_URL",
    "OVERPASS_SELECTORS",
    "PLACE_VALUES",
    "POWER_LINE_VALUES",
    "SUBSTATION_VALUES",
    "MemoryOSMSource",
    "OSMAnalysisBundle",
    "OSMBuildingProvider",
    "OSMFeatureSource",
    "OSMInfrastructureProvider",
    "OSMOverpassProvider",
    "OverpassClient",
    "OverpassQuerySpec",
    "OverpassSelector",
    "OverpassSnapshot",
    "assert_bounded_overpass_query",
    "bbox_span_ok",
    "build_overpass_query",
    "classify_osm_tags",
    "parse_demand_centers",
    "parse_overpass_json",
    "require_bounded_bbox",
    "search_bbox_from_point",
    "selectors_for",
]
