"""Protected-area cadastre adapters. There is no global unrestricted dataset.

Vector files (GeoJSON without GIS extras; GeoPackage and shapefile via
``geopandas``) are the first implementation. Official national cadastres and
European products (Natura 2000, CDDA, licensed WDPA extracts, ...) are
region-specific adapters injected into ``RegionalProtectedAreaProvider``.
SolarDSS never downloads WDPA or any other worldwide product.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol

from pydantic import BaseModel, ConfigDict, Field

from solardss.core.jsonutil import JsonValue
from solardss.data.contracts import DatasetRef
from solardss.data.geo import BoundingBox
from solardss.data.gis import gis_module_available
from solardss.exceptions import InvalidParameterError, MissingExtraError
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.indicators.ids import IndicatorId
from solardss.indicators.protected import (
    CALC_PROTECTED_OVERLAP,
    IUCN_IA_IB_HARD_EXCLUSION,
    NAMED_CONSTRAINT_PROFILES,
    ProtectedAreaConstraintProfile,
    ProtectedAreaFeature,
    ProtectedOverlapResult,
    analyze_point_in_protected,
    analyze_protected_overlap,
    bbox_ring,
    resolve_constraint_profile,
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
from solardss.providers.errors import CacheError, ProtectedAreaError
from solardss.providers.region import (
    bboxes_intersect,
    request_bbox,
    request_place_id,
    request_point,
)
from solardss.providers.request import FetchRequest, request_cache_key

PROTECTED_AREA_PATH_ENV: Final = "SOLARDSS_PROTECTED_AREA_PATH"
VECTOR_PROVIDER_ID: Final = "vector.protected_area"
REGIONAL_PROVIDER_ID: Final = "regional.protected_area"
_CACHE_SCHEMA: Final = "solardss.protected_obs.v1"
_SERVED: Final = frozenset({IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value})
_GEOJSON_SUFFIXES: Final = frozenset({".geojson", ".json"})
_GEOPANDAS_SUFFIXES: Final = frozenset({".gpkg", ".shp"})
_Ring = tuple[tuple[float, float], ...]
_PolygonPart = tuple[_Ring, tuple[_Ring, ...]]


class VectorFieldMap(BaseModel):
    """Column names on a caller-supplied vector file. Nothing is inferred."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    map_id: str = Field(min_length=1)
    site_id: str = "id"
    name: str | None = "name"
    category: str | None = "category"
    designation: str | None = "designation"
    notes: tuple[str, ...] = ()


GENERIC_FIELD_MAP = VectorFieldMap(
    map_id="generic",
    notes=("Generic id/name/category/designation columns.",),
)
NATURA2000_FIELD_MAP = VectorFieldMap(
    map_id="natura2000",
    site_id="SITECODE",
    name="SITENAME",
    category="SITETYPE",
    designation="SITETYPE",
    notes=(
        "Use only when the caller supplies a Natura 2000 extract.",
        "SITETYPE A/B/C is not treated as a universal exclusion.",
    ),
)
WDPA_FIELD_MAP = VectorFieldMap(
    map_id="wdpa",
    site_id="WDPAID",
    name="NAME",
    category="IUCN_CAT",
    designation="DESIG_ENG",
    notes=(
        "Use only when the caller supplies a licensed WDPA extract.",
        "SolarDSS does not download WDPA.",
    ),
)
NAMED_FIELD_MAPS: dict[str, VectorFieldMap] = {
    GENERIC_FIELD_MAP.map_id: GENERIC_FIELD_MAP,
    NATURA2000_FIELD_MAP.map_id: NATURA2000_FIELD_MAP,
    WDPA_FIELD_MAP.map_id: WDPA_FIELD_MAP,
}


class ProtectedAreaSource(Protocol):
    """Read designated-site polygons. Implementations are region- or file-specific."""

    @property
    def crs(self) -> str:
        """Coordinate system of feature rings."""

    @property
    def source_name(self) -> str:
        """Provenance source_name recorded on observations."""

    @property
    def source_dataset(self) -> str:
        """Dataset identity recorded on each feature."""

    def features_in(self, bbox: BoundingBox | None) -> tuple[ProtectedAreaFeature, ...]:
        """Sites that intersect ``bbox``. ``None`` returns the full extract."""


@dataclass(frozen=True, slots=True)
class ProtectedAreaAnalysisBundle:
    """Overlap result plus the observation produced for ``LAND_PROTECTED_AREA_OVERLAP``."""

    analysis: object | None
    observations: tuple[IndicatorObservation, ...]
    source_crs: str | None
    quality_warnings: tuple[str, ...] = ()


def _text_param(parameters: Mapping[str, JsonValue], key: str) -> str | None:
    value = parameters.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _str_tuple_param(parameters: Mapping[str, JsonValue], key: str) -> tuple[str, ...] | None:
    value = parameters.get(key)
    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, list):
        items = [item.strip() for item in value if isinstance(item, str) and item.strip()]
        return tuple(items)
    return None


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


def _as_text(value: object) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        if value.is_integer():
            return str(int(value))
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _property_text(properties: Mapping[str, object], key: str | None) -> str | None:
    if not key:
        return None
    if key not in properties:
        return None
    return _as_text(properties[key])


def field_map_from_name(name: str) -> VectorFieldMap:
    if name not in NAMED_FIELD_MAPS:
        raise InvalidParameterError(
            f"unknown protected-area field map {name!r}; "
            "pass generic, natura2000, or wdpa, or inject a VectorFieldMap"
        )
    return NAMED_FIELD_MAPS[name]


def ring_bounds(ring: Sequence[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if not ring:
        return None
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return (min(xs), min(ys), max(xs), max(ys))


def feature_intersects_bbox(feature: ProtectedAreaFeature, bbox: BoundingBox) -> bool:
    bounds = ring_bounds(feature.coordinates)
    if bounds is None:
        return False
    min_x, min_y, max_x, max_y = bounds
    feature_box = BoundingBox(
        min_lon=min_x,
        min_lat=min_y,
        max_lon=max_x,
        max_lat=max_y,
        crs=bbox.crs,
    )
    return bboxes_intersect(feature_box, bbox)


def _coords_to_ring(raw: object) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list) or len(raw) < 3:
        return ()
    points: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, list) or len(item) < 2:
            continue
        x, y = item[0], item[1]
        if isinstance(x, bool) or isinstance(y, bool):
            continue
        if isinstance(x, int | float) and isinstance(y, int | float):
            points.append((float(x), float(y)))
    if len(points) < 3:
        return ()
    return tuple(points)


def _polygon_parts(geometry: Mapping[str, object]) -> tuple[_PolygonPart, ...]:
    geom_type = geometry.get("type")
    coords = geometry.get("coordinates")
    if geom_type == "Polygon" and isinstance(coords, list) and coords:
        outer = _coords_to_ring(coords[0])
        if len(outer) < 3:
            return ()
        holes = tuple(_coords_to_ring(ring) for ring in coords[1:] if isinstance(ring, list))
        holes = tuple(ring for ring in holes if len(ring) >= 3)
        return ((outer, holes),)
    if geom_type == "MultiPolygon" and isinstance(coords, list):
        parts: list[_PolygonPart] = []
        for polygon in coords:
            if not isinstance(polygon, list) or not polygon:
                continue
            outer = _coords_to_ring(polygon[0])
            if len(outer) < 3:
                continue
            holes = tuple(_coords_to_ring(ring) for ring in polygon[1:] if isinstance(ring, list))
            holes = tuple(ring for ring in holes if len(ring) >= 3)
            parts.append((outer, holes))
        return tuple(parts)
    return ()


def features_from_geojson(
    payload: Mapping[str, object],
    *,
    source_dataset: str,
    field_map: VectorFieldMap,
    default_crs: str = "EPSG:4326",
) -> tuple[ProtectedAreaFeature, ...]:
    """Parse a GeoJSON FeatureCollection or Feature. Points and lines are ignored."""

    raw_features: list[Mapping[str, object]]
    kind = payload.get("type")
    if kind == "FeatureCollection":
        items = payload.get("features")
        if not isinstance(items, list):
            return ()
        raw_features = [item for item in items if isinstance(item, dict)]
    elif kind == "Feature":
        raw_features = [payload]
    elif kind in {"Polygon", "MultiPolygon"}:
        raw_features = [{"type": "Feature", "properties": {}, "geometry": dict(payload)}]
    else:
        raise ProtectedAreaError("GeoJSON must be a Feature, FeatureCollection, or Polygon")

    features: list[ProtectedAreaFeature] = []
    for index, item in enumerate(raw_features):
        geometry = item.get("geometry")
        if not isinstance(geometry, dict):
            continue
        properties_raw = item.get("properties")
        properties: dict[str, object] = (
            dict(properties_raw) if isinstance(properties_raw, dict) else {}
        )
        site_id = _property_text(properties, field_map.site_id) or f"feature-{index}"
        name = _property_text(properties, field_map.name)
        category = _property_text(properties, field_map.category)
        designation = _property_text(properties, field_map.designation)
        attributes = tuple(
            (str(key), text)
            for key, value in properties.items()
            if (text := _as_text(value)) is not None
        )
        parts = _polygon_parts(geometry)
        if not parts:
            continue
        for part_index, (outer, holes) in enumerate(parts):
            part_id = site_id if len(parts) == 1 else f"{site_id}#{part_index}"
            features.append(
                ProtectedAreaFeature(
                    site_id=part_id,
                    name=name,
                    category=category,
                    designation=designation,
                    coordinates=outer,
                    holes=holes,
                    source_dataset=source_dataset,
                    attributes=attributes,
                )
            )
    _ = default_crs
    return tuple(features)


def load_geojson_path(
    path: Path,
    *,
    source_dataset: str,
    field_map: VectorFieldMap,
) -> tuple[ProtectedAreaFeature, ...]:
    try:
        payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtectedAreaError(f"could not read GeoJSON {path}") from exc
    if not isinstance(payload_obj, dict):
        raise ProtectedAreaError("GeoJSON root must be an object")
    return features_from_geojson(payload_obj, source_dataset=source_dataset, field_map=field_map)


def parse_polygon_wkt(wkt: str) -> tuple[tuple[float, float], ...]:
    """Parse a simple ``POLYGON((x y, ...))`` ring. Holes and Z are rejected."""

    compact = " ".join(wkt.strip().split())
    upper = compact.upper()
    if not upper.startswith("POLYGON"):
        raise InvalidParameterError(
            "geometry_wkt must be a simple POLYGON((x y, ...)); "
            "install solardss[gis] for other WKT types"
        )
    if "POLYGON Z" in upper or "EMPTY" in upper:
        raise InvalidParameterError("protected-area AOI WKT does not accept EMPTY or Z polygons")
    start = compact.find("(")
    end = compact.rfind(")")
    if start < 0 or end <= start:
        raise InvalidParameterError("geometry_wkt POLYGON is missing parentheses")
    body = compact[start : end + 1]
    if body.count("(") > 2:
        raise InvalidParameterError("protected-area AOI WKT does not accept holes")
    inner = body.strip()
    while inner.startswith("(") and inner.endswith(")"):
        inner = inner[1:-1].strip()
    points: list[tuple[float, float]] = []
    for token in inner.split(","):
        parts = token.strip().replace(",", " ").split()
        if len(parts) < 2:
            raise InvalidParameterError("geometry_wkt vertex must have x and y")
        try:
            points.append((float(parts[0]), float(parts[1])))
        except ValueError as exc:
            raise InvalidParameterError("geometry_wkt vertex is not numeric") from exc
    if len(points) < 3:
        raise InvalidParameterError("geometry_wkt POLYGON needs at least three vertices")
    return tuple(points)


def _require_geopandas() -> Any:
    try:
        import geopandas
    except ImportError as exc:
        raise MissingExtraError(
            "GeoPackage and shapefile protected-area files require geopandas. "
            "Install with: pip install 'solardss[gis]'."
        ) from exc
    return geopandas


def features_from_geodataframe(
    frame: Any,
    *,
    source_dataset: str,
    field_map: VectorFieldMap,
    target_crs: str | None,
) -> tuple[ProtectedAreaFeature, ...]:
    gdf = frame
    if target_crs is not None and getattr(gdf, "crs", None) is not None:
        crs_label = str(gdf.crs)
        if crs_label and crs_label != target_crs:
            gdf = gdf.to_crs(target_crs)
    features: list[ProtectedAreaFeature] = []
    for index, row in gdf.iterrows():
        geom = row.geometry if hasattr(row, "geometry") else None
        if geom is None or geom is True or geom is False:
            continue
        geo = geom.__geo_interface__ if hasattr(geom, "__geo_interface__") else None
        if not isinstance(geo, dict):
            continue
        properties = {str(key): row[key] for key in gdf.columns if key != "geometry"}
        site_id = _property_text(properties, field_map.site_id) or f"feature-{index}"
        name = _property_text(properties, field_map.name)
        category = _property_text(properties, field_map.category)
        designation = _property_text(properties, field_map.designation)
        attributes = tuple(
            (str(key), text)
            for key, value in properties.items()
            if (text := _as_text(value)) is not None
        )
        parts = _polygon_parts(geo)
        for part_index, (outer, holes) in enumerate(parts):
            part_id = site_id if len(parts) == 1 else f"{site_id}#{part_index}"
            features.append(
                ProtectedAreaFeature(
                    site_id=str(part_id),
                    name=name,
                    category=category,
                    designation=designation,
                    coordinates=outer,
                    holes=holes,
                    source_dataset=source_dataset,
                    attributes=attributes,
                )
            )
    return tuple(features)


class MemoryProtectedAreaSource:
    """In-memory cadastre. Used in tests and as an injected regional extract."""

    def __init__(
        self,
        features: Sequence[ProtectedAreaFeature],
        *,
        crs: str = "EPSG:4326",
        source_name: str = "injected protected-area cadastre",
        source_dataset: str = "memory",
    ) -> None:
        self._features = tuple(features)
        self._crs = crs
        self._source_name = source_name
        self._source_dataset = source_dataset

    @property
    def crs(self) -> str:
        return self._crs

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def source_dataset(self) -> str:
        return self._source_dataset

    def features_in(self, bbox: BoundingBox | None) -> tuple[ProtectedAreaFeature, ...]:
        if bbox is None:
            return self._features
        return tuple(item for item in self._features if feature_intersects_bbox(item, bbox))


class VectorFileProtectedAreaSource:
    """GeoJSON without GIS extras; GeoPackage and shapefile via geopandas."""

    def __init__(
        self,
        path: Path | str,
        *,
        field_map: VectorFieldMap | None = None,
        source_name: str | None = None,
        source_dataset: str | None = None,
        crs: str = "EPSG:4326",
        layer: str | None = None,
    ) -> None:
        self._path = Path(path)
        self._field_map = field_map or GENERIC_FIELD_MAP
        self._crs = crs
        self._layer = layer
        suffix = self._path.suffix.lower()
        if suffix in _GEOJSON_SUFFIXES:
            kind = "geojson"
        elif suffix in _GEOPANDAS_SUFFIXES:
            kind = "geopandas"
        else:
            raise InvalidParameterError(
                f"unsupported protected-area vector format {self._path.suffix!r}; "
                "use GeoJSON, GeoPackage, or shapefile"
            )
        self._kind = kind
        self._source_dataset = source_dataset or self._path.name
        self._source_name = source_name or f"vector protected-area ({self._path.name})"
        self._features: tuple[ProtectedAreaFeature, ...] | None = None

    @property
    def crs(self) -> str:
        return self._crs

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def source_dataset(self) -> str:
        return self._source_dataset

    def features_in(self, bbox: BoundingBox | None) -> tuple[ProtectedAreaFeature, ...]:
        loaded = self._load()
        if bbox is None:
            return loaded
        return tuple(item for item in loaded if feature_intersects_bbox(item, bbox))

    def _load(self) -> tuple[ProtectedAreaFeature, ...]:
        if self._features is not None:
            return self._features
        if not self._path.exists():
            raise ProtectedAreaError(f"protected-area path does not exist: {self._path}")
        if self._kind == "geojson":
            self._features = load_geojson_path(
                self._path,
                source_dataset=self._source_dataset,
                field_map=self._field_map,
            )
            return self._features
        geopandas = _require_geopandas()
        kwargs: dict[str, object] = {}
        if self._layer and self._path.suffix.lower() == ".gpkg":
            kwargs["layer"] = self._layer
        try:
            frame = geopandas.read_file(self._path, **kwargs)
        except (OSError, ValueError) as exc:
            raise ProtectedAreaError(f"could not read protected-area vector {self._path}") from exc
        if getattr(frame, "crs", None) is not None:
            self._crs = str(frame.crs) or self._crs
        self._features = features_from_geodataframe(
            frame,
            source_dataset=self._source_dataset,
            field_map=self._field_map,
            target_crs=None,
        )
        return self._features


def _encode_observation(observation: IndicatorObservation) -> bytes:
    envelope = {
        "schema_id": _CACHE_SCHEMA,
        "observation": observation.model_dump(mode="json"),
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _decode_observation(payload: bytes) -> IndicatorObservation:
    try:
        raw: object = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheError("cache entry is not valid JSON") from exc
    if not isinstance(raw, dict) or raw.get("schema_id") != _CACHE_SCHEMA:
        raise CacheError("cache entry has unknown schema")
    body = raw.get("observation")
    if not isinstance(body, dict):
        raise CacheError("cache entry is missing an observation")
    try:
        return IndicatorObservation.model_validate(body)
    except (InvalidParameterError, ValueError, TypeError) as exc:
        raise CacheError("cache entry is not an IndicatorObservation") from exc


class VectorProtectedAreaProvider:
    """Generic vector-file protected-area provider.

    GeoJSON is parsed without GIS extras. GeoPackage and shapefile require
    ``geopandas``. Overlap is always computed; hard exclusion is never implied.
    """

    provider_id_default = VECTOR_PROVIDER_ID
    source_name_default = "caller-supplied protected-area vector"
    served = _SERVED

    def __init__(
        self,
        *,
        source: ProtectedAreaSource | None = None,
        vector_path: Path | str | None = None,
        field_map: VectorFieldMap | None = None,
        profiles: Mapping[str, ProtectedAreaConstraintProfile] | None = None,
        default_profile: ProtectedAreaConstraintProfile | None = None,
        source_name: str | None = None,
        source_dataset: str | None = None,
        source_version: str | None = None,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        provider_id: str | None = None,
        environ: Mapping[str, str] | None = None,
        layer: str | None = None,
        crs: str | None = None,
    ) -> None:
        path = vector_path
        if path is None:
            env = environ if environ is not None else None
            raw = None if env is None else env.get(PROTECTED_AREA_PATH_ENV)
            if raw is None and environ is None:
                import os

                raw = os.environ.get(PROTECTED_AREA_PATH_ENV)
            path = raw
        self._field_map = field_map or GENERIC_FIELD_MAP
        self._source = source
        self._path = Path(path) if path else None
        self._layer = layer
        self._crs = crs
        self._profiles = dict(profiles) if profiles is not None else dict(NAMED_CONSTRAINT_PROFILES)
        self._default_profile = default_profile
        self._source_name = source_name
        self._source_dataset = source_dataset
        self._source_version = source_version
        self._cache = cache
        self._clock: Clock = UtcClock() if clock is None else clock
        self._provider_id = provider_id or self.provider_id_default
        self._file_source: VectorFileProtectedAreaSource | None = None

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def _configured(self) -> bool:
        return self._source is not None or self._path is not None

    def _resolve_source(self) -> ProtectedAreaSource:
        if self._source is not None:
            return self._source
        if self._path is None:
            raise ProtectedAreaError("protected-area vector is not configured")
        if self._file_source is None:
            self._file_source = VectorFileProtectedAreaSource(
                self._path,
                field_map=self._field_map,
                source_name=self._source_name,
                source_dataset=self._source_dataset,
                crs=self._crs or "EPSG:4326",
                layer=self._layer,
            )
        return self._file_source

    def health(self) -> ProviderHealth:
        checked_at = self._clock.now()
        details: dict[str, JsonValue] = {
            "vector_path": None if self._path is None else str(self._path),
            "injected_source": self._source is not None,
            "geopandas": gis_module_available("geopandas"),
            "field_map": self._field_map.map_id,
            "named_profiles": list(self._profiles),
            "has_default_profile": self._default_profile is not None,
        }
        if not self._configured():
            return ProviderHealth(
                status=HealthStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                reason=(
                    "protected-area cadastre is not configured: pass a ProtectedAreaSource, "
                    "a vector path, or SOLARDSS_PROTECTED_AREA_PATH"
                ),
                checked_at=checked_at,
                details=details,
            )
        if self._path is not None and not self._path.exists() and self._source is None:
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                reason="protected-area path does not exist",
                checked_at=checked_at,
                details=details,
            )
        return ProviderHealth(
            status=HealthStatus.OK,
            provider_id=self._provider_id,
            reason="protected-area provider is configured",
            checked_at=checked_at,
            details=details,
        )

    def _field_map_from_request(self, request: FetchRequest) -> VectorFieldMap:
        name = _text_param(request.parameters, "protected_field_map")
        if name is None:
            return self._field_map
        return field_map_from_name(name)

    def _profile_from_request(self, request: FetchRequest) -> ProtectedAreaConstraintProfile | None:
        profile_id = _text_param(request.parameters, "protected_area_profile") or _text_param(
            request.parameters, "constraint_profile_id"
        )
        unmatched = _text_param(request.parameters, "unmatched_disposition")
        return resolve_constraint_profile(
            profile_id=profile_id,
            hard_exclusion_categories=_str_tuple_param(
                request.parameters, "hard_exclusion_categories"
            ),
            warning_categories=_str_tuple_param(request.parameters, "warning_categories"),
            penalty_categories=_str_tuple_param(request.parameters, "penalty_categories"),
            penalty_weight=_float_param(request.parameters, "penalty_weight"),
            unmatched_disposition=unmatched,
            catalog=self._profiles,
        )

    def availability(self, request: FetchRequest) -> ProviderAvailability:
        if request.indicator_id not in self.served:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="indicator is not served by the protected-area provider",
            )
        if not self._configured():
            return ProviderAvailability(
                status=AvailabilityStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="protected-area cadastre is not configured",
            )
        try:
            self._profile_from_request(request)
            self._field_map_from_request(request)
            if _text_param(request.parameters, "geometry_wkt") is not None:
                parse_polygon_wkt(_text_param(request.parameters, "geometry_wkt") or "")
        except InvalidParameterError as exc:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=exc.message,
            )
        if (
            request_point(request) is None
            and request_bbox(request) is None
            and _text_param(request.parameters, "geometry_wkt") is None
        ):
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="protected-area overlap requires a point, bbox, or geometry_wkt",
            )
        return ProviderAvailability(
            status=AvailabilityStatus.AVAILABLE,
            provider_id=self._provider_id,
            retryable=False,
            indicator_id=request.indicator_id,
            reason="protected-area cadastre can be intersected for this request",
        )

    def coverage(self, request: FetchRequest) -> CoverageReport:
        availability = self.availability(request)
        spatial = SpatialCoverage(
            resolution="protected-area-vector",
            crs=self._crs or "EPSG:4326",
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
        request_hash = request_cache_key(self._provider_id, request)
        availability = self.availability(request)
        if availability.status is not AvailabilityStatus.AVAILABLE:
            return self._miss(
                request,
                ObservationStatus(availability.status.value),
                availability.reason,
                self._clock.now(),
            )
        cached = self._read_cache(request_hash)
        if cached is not None:
            return cached
        bundle = self.fetch_protected(request)
        for item in bundle.observations:
            if item.status is ObservationStatus.OBSERVED:
                self._write_cache(item.provenance.request_hash, item)
            if item.indicator_id == request.indicator_id:
                return item
        return self._miss(
            request,
            ObservationStatus.UNAVAILABLE,
            "indicator was not produced from the protected-area cadastre",
            self._clock.now(),
        )

    def fetch_protected(self, request: FetchRequest) -> ProtectedAreaAnalysisBundle:
        availability = self.availability(request)
        retrieved_at = self._clock.now()
        if availability.status is not AvailabilityStatus.AVAILABLE:
            miss = self._miss(
                request,
                ObservationStatus(availability.status.value),
                availability.reason,
                retrieved_at,
            )
            return ProtectedAreaAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                quality_warnings=(availability.reason,),
            )
        try:
            source = self._resolve_source()
            profile = self._profile_from_request(request) or self._default_profile
            aoi, point, analysis_crs, analysis_bbox = self._place(request, source.crs)
            features = source.features_in(analysis_bbox)
            if point is not None and aoi is None:
                analysis = analyze_point_in_protected(
                    point[0], point[1], features, crs=analysis_crs, profile=profile
                )
            elif aoi is not None:
                analysis = analyze_protected_overlap(
                    aoi, features, crs=analysis_crs, profile=profile
                )
            else:
                raise InvalidParameterError(
                    "protected-area overlap requires a point, bbox, or geometry_wkt"
                )
            observation = self._observation_from_analysis(
                request, analysis, source, retrieved_at, analysis_crs
            )
        except MissingExtraError as exc:
            miss = self._miss(request, ObservationStatus.NOT_CONFIGURED, exc.message, retrieved_at)
            return ProtectedAreaAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                quality_warnings=(exc.message,),
            )
        except ProtectedAreaError as exc:
            miss = self._miss(
                request, ObservationStatus.TEMPORARY_FAILURE, exc.message, retrieved_at
            )
            return ProtectedAreaAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                quality_warnings=(exc.message,),
            )
        except InvalidParameterError as exc:
            miss = self._miss(request, ObservationStatus.UNAVAILABLE, exc.message, retrieved_at)
            return ProtectedAreaAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                quality_warnings=(exc.message,),
            )
        return ProtectedAreaAnalysisBundle(
            analysis=analysis,
            observations=(observation,),
            source_crs=analysis_crs,
            quality_warnings=analysis.summary.warnings,
        )

    def _place(
        self, request: FetchRequest, source_crs: str
    ) -> tuple[
        tuple[tuple[float, float], ...] | None,
        tuple[float, float] | None,
        str,
        BoundingBox | None,
    ]:
        wkt = _text_param(request.parameters, "geometry_wkt")
        geometry_crs = _text_param(request.parameters, "geometry_crs")
        box = request_bbox(request)
        point = request_point(request)
        loc_crs = source_crs
        if request.location is not None and request.location.crs:
            loc_crs = request.location.crs
        elif box is not None:
            loc_crs = box.crs
        analysis_crs = geometry_crs or loc_crs or source_crs
        if wkt is not None:
            ring = parse_polygon_wkt(wkt)
            xs = [item[0] for item in ring]
            ys = [item[1] for item in ring]
            analysis_bbox = BoundingBox(
                min_lon=min(xs),
                min_lat=min(ys),
                max_lon=max(xs),
                max_lat=max(ys),
                crs=analysis_crs,
            )
            return ring, None, analysis_crs, analysis_bbox
        if box is not None:
            ring = bbox_ring(box.min_lon, box.min_lat, box.max_lon, box.max_lat)
            return ring, None, box.crs or analysis_crs, box
        if point is not None:
            point_box = BoundingBox(
                min_lon=point[0],
                min_lat=point[1],
                max_lon=point[0],
                max_lat=point[1],
                crs=analysis_crs,
            )
            return None, point, analysis_crs, point_box
        return None, None, analysis_crs, box

    def _observation_from_analysis(
        self,
        request: FetchRequest,
        analysis: ProtectedOverlapResult,
        source: ProtectedAreaSource,
        retrieved_at: datetime,
        analysis_crs: str,
    ) -> IndicatorObservation:
        summary = analysis.summary
        source_name = self._source_name or source.source_name
        dataset = DatasetRef(
            name=source_name,
            media_type="application/geo+json",
            uri=None if self._path is None else str(self._path),
            notes=(
                f"source_dataset={source.source_dataset}",
                f"source_name={source_name}",
                f"field_map={self._field_map.map_id}",
                f"crs={analysis_crs}",
                f"area_method={summary.area_method}",
                f"profile_id={summary.profile_id}",
                f"overall_disposition={summary.overall_disposition.value}",
            ),
        )
        spatial = SpatialCoverage(
            resolution="protected-area-vector",
            crs=analysis_crs,
            bbox=request_bbox(request),
        )
        temporal = request.temporal or TemporalCoverage(resolution="snapshot")
        extra = (
            f"intersection_area_m2={summary.intersection_area_m2}",
            f"intersection_percent={summary.overlap_percent}",
            f"site_ids={','.join(summary.site_ids) if summary.site_ids else 'none'}",
            f"categories={','.join(summary.categories) if summary.categories else 'none'}",
            f"designations={','.join(summary.designations) if summary.designations else 'none'}",
            f"overall_disposition={summary.overall_disposition.value}",
            f"union_is_exact={summary.union_is_exact}",
            f"source_dataset={source.source_dataset}",
            "overlap is not a universal exclusion; disposition comes from the constraint profile",
            "SolarDSS does not assume a global protected-area dataset",
        )
        assumptions = tuple([*extra, *summary.warnings])
        flags: tuple[str, ...]
        if summary.union_is_exact:
            flags = (QualityFlag.NOMINAL.value,)
        else:
            flags = (QualityFlag.ESTIMATED.value,)
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=source_name,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=ValueProvenanceKind.DERIVED_METHOD,
                value_origin=ValueOrigin.DERIVED,
                source_version=self._source_version,
                calculation_method=CALC_PROTECTED_OVERLAP,
                dataset=dataset,
            ),
            spatial=spatial,
            temporal=temporal,
            value=float(summary.overlap_fraction),
            unit="1",
            quality_flags=flags,
            assumptions=assumptions,
        )

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
        source_name = self._source_name or self.source_name_default
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=status,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=source_name,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=ValueProvenanceKind.MEASURED_OR_PROVIDER,
                value_origin=ValueOrigin.RAW,
                source_version=self._source_version,
                dataset=dataset or DatasetRef(name=source_name, media_type="application/geo+json"),
            ),
            spatial=spatial
            or SpatialCoverage(resolution="protected-area-vector", crs=self._crs or "EPSG:4326"),
            temporal=request.temporal or TemporalCoverage(resolution="snapshot"),
            assumptions=(reason,),
        )

    def _read_cache(self, request_hash: str) -> IndicatorObservation | None:
        if self._cache is None:
            return None
        payload = self._cache.get(request_hash)
        if payload is None:
            return None
        observation = _decode_observation(payload)
        if observation.provenance.request_hash != request_hash:
            raise CacheError("cached observation request_hash does not match the request")
        return observation

    def _write_cache(self, request_hash: str, observation: IndicatorObservation) -> None:
        if self._cache is None:
            return
        self._cache.put(request_hash, _encode_observation(observation))


class RegionalProtectedAreaProvider:
    """Dispatch to a region-specific cadastre. There is no global fallback."""

    provider_id_default = REGIONAL_PROVIDER_ID

    def __init__(
        self,
        adapters: Mapping[str, VectorProtectedAreaProvider],
        *,
        provider_id: str | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._adapters = {key: value for key, value in adapters.items()}
        self._provider_id = provider_id or self.provider_id_default
        self._clock: Clock = UtcClock() if clock is None else clock

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def _region_id(self, request: FetchRequest) -> str | None:
        return _text_param(request.parameters, "region_id") or request_place_id(request)

    def _adapter(self, request: FetchRequest) -> VectorProtectedAreaProvider | None:
        region = self._region_id(request)
        if region is None:
            return None
        return self._adapters.get(region)

    def health(self) -> ProviderHealth:
        checked_at = self._clock.now()
        details: dict[str, JsonValue] = {
            "regions": list(self._adapters),
            "global_dataset": False,
        }
        if not self._adapters:
            return ProviderHealth(
                status=HealthStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                reason=(
                    "no region-specific protected-area adapters were injected; "
                    "SolarDSS has no global unrestricted cadastre"
                ),
                checked_at=checked_at,
                details=details,
            )
        return ProviderHealth(
            status=HealthStatus.OK,
            provider_id=self._provider_id,
            reason="region-specific protected-area adapters are configured",
            checked_at=checked_at,
            details=details,
        )

    def availability(self, request: FetchRequest) -> ProviderAvailability:
        if request.indicator_id not in _SERVED:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="indicator is not served by the protected-area provider",
            )
        region = self._region_id(request)
        if region is None:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=("region_id is required; SolarDSS has no global protected-area dataset"),
            )
        adapter = self._adapters.get(region)
        if adapter is None:
            return ProviderAvailability(
                status=AvailabilityStatus.UNSUPPORTED_REGION,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=(
                    f"no protected-area adapter for region {region!r}; "
                    "official national and European cadastres must be injected"
                ),
            )
        return adapter.availability(request)

    def coverage(self, request: FetchRequest) -> CoverageReport:
        adapter = self._adapter(request)
        if adapter is None:
            availability = self.availability(request)
            return CoverageReport(
                status=availability.status,
                provider_id=self._provider_id,
                spatial=SpatialCoverage(
                    resolution="protected-area-vector",
                    crs="EPSG:4326",
                    bbox=request_bbox(request),
                ),
                temporal=request.temporal or TemporalCoverage(resolution="snapshot"),
                reason=availability.reason,
            )
        return adapter.coverage(request)

    def fetch(self, request: FetchRequest) -> IndicatorObservation:
        adapter = self._adapter(request)
        if adapter is None:
            availability = self.availability(request)
            source_name = "regional protected-area dispatcher"
            return IndicatorObservation(
                indicator_id=request.indicator_id,
                status=ObservationStatus(availability.status.value),
                provenance=DataProvenance(
                    provider_id=self._provider_id,
                    source_name=source_name,
                    retrieved_at=self._clock.now(),
                    request_hash=request_cache_key(self._provider_id, request),
                    value_kind=ValueProvenanceKind.MEASURED_OR_PROVIDER,
                    value_origin=ValueOrigin.RAW,
                    dataset=DatasetRef(name=source_name, media_type="application/geo+json"),
                ),
                spatial=SpatialCoverage(
                    resolution="protected-area-vector",
                    crs="EPSG:4326",
                    bbox=request_bbox(request),
                ),
                temporal=request.temporal or TemporalCoverage(resolution="snapshot"),
                assumptions=(availability.reason,),
            )
        return adapter.fetch(request)


GenericVectorProtectedAreaProvider = VectorProtectedAreaProvider


__all__ = [
    "GENERIC_FIELD_MAP",
    "IUCN_IA_IB_HARD_EXCLUSION",
    "NATURA2000_FIELD_MAP",
    "PROTECTED_AREA_PATH_ENV",
    "WDPA_FIELD_MAP",
    "GenericVectorProtectedAreaProvider",
    "MemoryProtectedAreaSource",
    "RegionalProtectedAreaProvider",
    "VectorFieldMap",
    "VectorFileProtectedAreaSource",
    "VectorProtectedAreaProvider",
    "features_from_geojson",
    "field_map_from_name",
    "parse_polygon_wkt",
]
