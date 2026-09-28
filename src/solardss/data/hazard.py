"""Generic hazard raster and vector adapters. There is no global hazard dataset.

JSON grids and GeoJSON are readable without GIS extras. GeoTIFF uses rasterio;
GeoPackage and shapefile use geopandas. Categories such as low/medium/high are
applied only from an explicit ``HazardValueMapping``. SolarDSS never downloads
a worldwide flood, fire, landslide, or hail product.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from solardss.core.arrays import FloatArray
from solardss.core.jsonutil import JsonValue
from solardss.data.contracts import DatasetRef
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.data.gis import gis_module_available
from solardss.data.tables import DEFAULT_ID_COLUMN, CandidateTable, validate_candidate_table
from solardss.exceptions import InvalidParameterError, MissingExtraError, SchemaError
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.indicators.hazard import (
    CALC_HAZARD_SAMPLE,
    KIND_TO_INDICATOR,
    HazardAggregation,
    HazardAnalysisResult,
    HazardGrid,
    HazardKind,
    HazardLayerMeta,
    HazardPolygon,
    HazardRecord,
    HazardValueMapping,
    aggregate_hazard_cells,
    analyze_hazard,
    mask_from_bbox,
    sample_hazard_cell,
    sample_hazard_polygon,
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
from solardss.providers.errors import CacheError, HazardCoverageError, HazardError
from solardss.providers.region import request_bbox, request_point
from solardss.providers.request import FetchRequest, request_cache_key

FLOOD_PATH_ENV: Final = "SOLARDSS_HAZARD_FLOOD_PATH"
WILDFIRE_PATH_ENV: Final = "SOLARDSS_HAZARD_WILDFIRE_PATH"
LANDSLIDE_PATH_ENV: Final = "SOLARDSS_HAZARD_LANDSLIDE_PATH"
HAIL_PATH_ENV: Final = "SOLARDSS_HAZARD_HAIL_PATH"
_KIND_PATH_ENV: dict[HazardKind, str] = {
    HazardKind.FLOOD: FLOOD_PATH_ENV,
    HazardKind.WILDFIRE: WILDFIRE_PATH_ENV,
    HazardKind.LANDSLIDE: LANDSLIDE_PATH_ENV,
    HazardKind.HAIL: HAIL_PATH_ENV,
}
_CACHE_SCHEMA: Final = "solardss.hazard_obs.v1"
_GEOJSON_SUFFIXES: Final = frozenset({".geojson"})
_JSON_GRID_SUFFIXES: Final = frozenset({".json"})
_RASTERIO_SUFFIXES: Final = frozenset({".tif", ".tiff", ".img"})
_GEOPANDAS_SUFFIXES: Final = frozenset({".gpkg", ".shp"})
_Ring = tuple[tuple[float, float], ...]
_PolygonPart = tuple[_Ring, tuple[_Ring, ...]]


class HazardFieldMap(BaseModel):
    """Attribute names on a caller-supplied vector hazard file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: str = "value"
    category: str | None = None
    return_period: str | None = None
    site_id: str = "id"


GENERIC_HAZARD_FIELD_MAP = HazardFieldMap()


class HazardRasterSource(Protocol):
    """Read a north-up hazard raster window."""

    @property
    def crs(self) -> str:
        """Native CRS of the raster."""

    def coverage_bbox(self) -> BoundingBox | None:
        """Dataset extent, if known."""

    def read_window(self, bbox: BoundingBox) -> HazardGrid:
        """Return cells covering ``bbox``. Values are not resampled."""

    def layer_meta(self) -> HazardLayerMeta:
        """Source identity for provenance."""


class HazardVectorSource(Protocol):
    """Read designated hazard polygons."""

    @property
    def crs(self) -> str:
        """Coordinate system of feature rings."""

    def features_in(self, bbox: BoundingBox | None) -> tuple[HazardPolygon, ...]:
        """Polygons that intersect ``bbox``. ``None`` returns the full extract."""

    def layer_meta(self) -> HazardLayerMeta:
        """Source identity for provenance."""


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


def _as_raw(value: object) -> str | float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return None
        return float(value)
    if isinstance(value, int):
        return float(value)
    if isinstance(value, str) and value.strip():
        token = value.strip()
        try:
            numeric = float(token)
        except ValueError:
            return token
        return numeric
    return None


def _text_param(parameters: Mapping[str, JsonValue], key: str) -> str | None:
    value = parameters.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _coords_to_ring(raw: object) -> _Ring:
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


def _window_indices(
    *,
    origin_x: float,
    origin_y: float,
    cell_size_x: float,
    cell_size_y: float,
    rows: int,
    cols: int,
    bbox: BoundingBox,
) -> tuple[int, int, int, int]:
    col0 = int(np.floor((bbox.min_lon - origin_x) / cell_size_x))
    col1 = int(np.ceil((bbox.max_lon - origin_x) / cell_size_x))
    row0 = int(np.floor((origin_y - bbox.max_lat) / cell_size_y))
    row1 = int(np.ceil((origin_y - bbox.min_lat) / cell_size_y))
    if col1 <= col0:
        col1 = col0 + 1
    if row1 <= row0:
        row1 = row0 + 1
    row0 = max(row0, 0)
    col0 = max(col0, 0)
    row1 = min(row1, rows)
    col1 = min(col1, cols)
    if row1 <= row0 or col1 <= col0:
        raise HazardCoverageError("requested window does not intersect the hazard raster")
    return row0, row1, col0, col1


def _point_window_bbox(point: tuple[float, float], crs: str) -> BoundingBox:
    """Expand a degenerate point so the raster window contains at least one cell."""

    token = crs.strip().upper().replace(" ", "")
    geographic = token in {"EPSG:4326", "WGS84"} or token.endswith(":4326")
    pad = 0.0002 if geographic else 30.0
    min_lon = point[0] - pad
    max_lon = point[0] + pad
    min_lat = point[1] - pad
    max_lat = point[1] + pad
    if geographic:
        min_lon = max(-180.0, min_lon)
        max_lon = min(180.0, max_lon)
        min_lat = max(-90.0, min_lat)
        max_lat = min(90.0, max_lat)
    return BoundingBox(
        min_lon=min_lon,
        min_lat=min_lat,
        max_lon=max_lon,
        max_lat=max_lat,
        crs=crs,
    )


def load_hazard_mapping(path: Path | str) -> HazardValueMapping:
    """Load a JSON legend. This is never inferred from raster statistics."""

    payload_obj: object = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload_obj, dict):
        raise HazardError("hazard mapping JSON must be an object")
    return HazardValueMapping.model_validate(payload_obj)


def _require_rasterio() -> Any:
    try:
        import rasterio
    except ImportError as exc:
        raise MissingExtraError(
            "GeoTIFF hazard rasters require rasterio. Install with: pip install 'solardss[gis]'."
        ) from exc
    return rasterio


def _require_geopandas() -> Any:
    try:
        import geopandas
    except ImportError as exc:
        raise MissingExtraError(
            "GeoPackage and shapefile hazard files require geopandas. "
            "Install with: pip install 'solardss[gis]'."
        ) from exc
    return geopandas


@dataclass(frozen=True, slots=True)
class MemoryHazardRaster:
    """In-memory north-up hazard raster used by tests and injected fixtures."""

    values: FloatArray
    coordinate_system: str
    origin_x: float
    origin_y: float
    cell_size_x: float
    cell_size_y: float
    nodata: float | None = None
    source_name: str = "array hazard raster"
    source_dataset: str = "memory"
    source_version: str | None = "fixture"
    source_year: int | None = None
    resolution: str = "synthetic"
    return_period: str | None = None
    uncertainty_notes: tuple[str, ...] = ()

    @property
    def crs(self) -> str:
        return self.coordinate_system

    def coverage_bbox(self) -> BoundingBox | None:
        rows, cols = np.asarray(self.values).shape
        return BoundingBox(
            min_lon=self.origin_x,
            min_lat=self.origin_y - rows * self.cell_size_y,
            max_lon=self.origin_x + cols * self.cell_size_x,
            max_lat=self.origin_y,
            crs=self.coordinate_system,
        )

    def read_window(self, bbox: BoundingBox) -> HazardGrid:
        values = np.asarray(self.values, dtype=np.float64)
        rows, cols = values.shape
        row0, row1, col0, col1 = _window_indices(
            origin_x=self.origin_x,
            origin_y=self.origin_y,
            cell_size_x=self.cell_size_x,
            cell_size_y=self.cell_size_y,
            rows=rows,
            cols=cols,
            bbox=bbox,
        )
        return HazardGrid(
            values=np.asarray(values[row0:row1, col0:col1], dtype=np.float64),
            cell_size_x=self.cell_size_x,
            cell_size_y=self.cell_size_y,
            origin_x=self.origin_x + col0 * self.cell_size_x,
            origin_y=self.origin_y - row0 * self.cell_size_y,
            nodata=self.nodata,
            crs=self.coordinate_system,
        )

    def layer_meta(self) -> HazardLayerMeta:
        return HazardLayerMeta(
            source_name=self.source_name,
            source_dataset=self.source_dataset,
            source_version=self.source_version,
            source_year=self.source_year,
            resolution=self.resolution,
            return_period=self.return_period,
            crs=self.coordinate_system,
            uncertainty_notes=self.uncertainty_notes,
        )


class JsonHazardRasterSource:
    """JSON grid (``values`` matrix plus origin/cell size). No GIS extra required."""

    def __init__(self, path: Path | str, *, meta: HazardLayerMeta | None = None) -> None:
        self._path = Path(path)
        payload_obj: object = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload_obj, dict):
            raise HazardError("hazard JSON grid must be an object")
        payload = payload_obj
        raw_values = payload.get("values")
        if not isinstance(raw_values, list) or not raw_values:
            raise HazardError("hazard JSON grid requires a values matrix")
        matrix = np.asarray(raw_values, dtype=np.float64)
        if matrix.ndim != 2:
            raise HazardError("hazard JSON values must be a 2-D matrix")
        nodata_raw = payload.get("nodata")
        nodata = float(nodata_raw) if isinstance(nodata_raw, int | float) else None
        crs = str(payload.get("crs") or "EPSG:4326")
        year_raw = payload.get("source_year")
        self._memory = MemoryHazardRaster(
            values=matrix,
            coordinate_system=crs,
            origin_x=float(payload.get("origin_x") or 0.0),
            origin_y=float(payload.get("origin_y") or 0.0),
            cell_size_x=float(payload.get("cell_size_x") or payload.get("dx") or 1.0),
            cell_size_y=float(payload.get("cell_size_y") or payload.get("dy") or 1.0),
            nodata=nodata,
            source_name=str(payload.get("source_name") or self._path.name),
            source_dataset=str(payload.get("source_dataset") or self._path.name),
            source_version=_as_text(payload.get("source_version")),
            source_year=int(year_raw) if isinstance(year_raw, int) else None,
            resolution=str(payload.get("resolution") or "json-grid"),
            return_period=_as_text(payload.get("return_period")),
            uncertainty_notes=tuple(
                item
                for item in (payload.get("uncertainty_notes") or ())
                if isinstance(item, str) and item.strip()
            ),
        )
        self._meta = meta or self._memory.layer_meta()

    @property
    def crs(self) -> str:
        return self._memory.crs

    def coverage_bbox(self) -> BoundingBox | None:
        return self._memory.coverage_bbox()

    def read_window(self, bbox: BoundingBox) -> HazardGrid:
        return self._memory.read_window(bbox)

    def layer_meta(self) -> HazardLayerMeta:
        return self._meta


class RasterioHazardSource:
    """GeoTIFF/COG reader. rasterio is imported only when a window is read."""

    def __init__(self, path: Path | str, *, meta: HazardLayerMeta | None = None) -> None:
        self._path = Path(path)
        self._meta = meta
        self._crs = meta.crs if meta is not None and meta.crs else "EPSG:4326"

    @property
    def crs(self) -> str:
        return self._crs

    def coverage_bbox(self) -> BoundingBox | None:
        rasterio = _require_rasterio()
        with rasterio.open(self._path) as dataset:
            self._crs = str(dataset.crs) if dataset.crs else self._crs
            bounds = dataset.bounds
            return BoundingBox(
                min_lon=float(bounds.left),
                min_lat=float(bounds.bottom),
                max_lon=float(bounds.right),
                max_lat=float(bounds.top),
                crs=self._crs,
            )

    def read_window(self, bbox: BoundingBox) -> HazardGrid:
        rasterio = _require_rasterio()
        from rasterio.enums import Resampling
        from rasterio.errors import RasterioError
        from rasterio.windows import Window, from_bounds

        try:
            with rasterio.open(self._path) as dataset:
                self._crs = str(dataset.crs) if dataset.crs else self._crs
                transform = dataset.transform
                window = from_bounds(
                    bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat, transform
                )
                window = Window(
                    col_off=int(np.floor(float(window.col_off))),
                    row_off=int(np.floor(float(window.row_off))),
                    width=max(1, int(np.ceil(float(window.width)))),
                    height=max(1, int(np.ceil(float(window.height)))),
                )
                data = dataset.read(
                    1,
                    window=window,
                    boundless=True,
                    masked=True,
                    resampling=Resampling.nearest,
                )
                window_transform = dataset.window_transform(window)
                nodata = dataset.nodata
        except RasterioError as exc:
            raise HazardError(f"failed to read hazard raster window: {exc}") from exc
        masked = np.ma.asarray(data)  # type: ignore[no-untyped-call]
        filled = np.asarray(
            np.ma.filled(masked.astype(np.float64), np.nan),  # type: ignore[no-untyped-call]
            dtype=np.float64,
        )
        return HazardGrid(
            values=filled,
            cell_size_x=abs(float(window_transform.a)),
            cell_size_y=abs(float(window_transform.e)),
            origin_x=float(window_transform.c),
            origin_y=float(window_transform.f),
            nodata=None if nodata is None else float(nodata),
            crs=self._crs,
        )

    def layer_meta(self) -> HazardLayerMeta:
        if self._meta is not None:
            return self._meta
        return HazardLayerMeta(
            source_name=self._path.name,
            source_dataset=self._path.name,
            resolution="raster",
            crs=self._crs,
        )


def _feature_bbox(ring: Sequence[tuple[float, float]], crs: str) -> BoundingBox | None:
    if not ring:
        return None
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return BoundingBox(
        min_lon=min(xs),
        min_lat=min(ys),
        max_lon=max(xs),
        max_lat=max(ys),
        crs=crs,
    )


class MemoryHazardVector:
    """In-memory hazard polygons."""

    def __init__(
        self,
        features: Sequence[HazardPolygon],
        *,
        crs: str = "EPSG:4326",
        meta: HazardLayerMeta | None = None,
    ) -> None:
        self._features = tuple(features)
        self._crs = crs
        self._meta = meta or HazardLayerMeta(
            source_name="array hazard vector",
            source_dataset="memory",
            resolution="vector",
            crs=crs,
        )

    @property
    def crs(self) -> str:
        return self._crs

    def features_in(self, bbox: BoundingBox | None) -> tuple[HazardPolygon, ...]:
        if bbox is None:
            return self._features
        kept: list[HazardPolygon] = []
        for item in self._features:
            feature_box = _feature_bbox(item.coordinates, bbox.crs)
            if feature_box is None:
                continue
            if not (
                feature_box.max_lon < bbox.min_lon
                or feature_box.min_lon > bbox.max_lon
                or feature_box.max_lat < bbox.min_lat
                or feature_box.min_lat > bbox.max_lat
            ):
                kept.append(item)
        return tuple(kept)

    def layer_meta(self) -> HazardLayerMeta:
        return self._meta


def polygons_from_geojson(
    payload: Mapping[str, object],
    *,
    field_map: HazardFieldMap,
) -> tuple[HazardPolygon, ...]:
    kind = payload.get("type")
    raw_features: list[Mapping[str, object]]
    if kind == "FeatureCollection":
        items = payload.get("features")
        if not isinstance(items, list):
            return ()
        raw_features = [item for item in items if isinstance(item, dict)]
    elif kind == "Feature":
        raw_features = [payload]
    else:
        raise HazardError("hazard GeoJSON must be a Feature or FeatureCollection")
    features: list[HazardPolygon] = []
    for index, item in enumerate(raw_features):
        geometry = item.get("geometry")
        if not isinstance(geometry, dict):
            continue
        properties_raw = item.get("properties")
        properties: dict[str, object] = (
            dict(properties_raw) if isinstance(properties_raw, dict) else {}
        )
        site_id = _as_text(properties.get(field_map.site_id)) or f"feature-{index}"
        raw_value = _as_raw(properties.get(field_map.value))
        source_category = (
            _as_text(properties.get(field_map.category)) if field_map.category else None
        )
        return_period = (
            _as_text(properties.get(field_map.return_period)) if field_map.return_period else None
        )
        attributes = tuple(
            (str(key), text)
            for key, value in properties.items()
            if (text := _as_text(value)) is not None
        )
        parts = _polygon_parts(geometry)
        for part_index, (outer, holes) in enumerate(parts):
            part_id = site_id if len(parts) == 1 else f"{site_id}#{part_index}"
            features.append(
                HazardPolygon(
                    site_id=part_id,
                    coordinates=outer,
                    holes=holes,
                    raw_value=raw_value,
                    source_category=source_category,
                    return_period=return_period,
                    attributes=attributes,
                )
            )
    return tuple(features)


class GeoJSONHazardSource:
    """GeoJSON polygons without GIS extras."""

    def __init__(
        self,
        path: Path | str,
        *,
        field_map: HazardFieldMap | None = None,
        meta: HazardLayerMeta | None = None,
        crs: str = "EPSG:4326",
    ) -> None:
        self._path = Path(path)
        payload_obj: object = json.loads(self._path.read_text(encoding="utf-8"))
        if not isinstance(payload_obj, dict):
            raise HazardError("hazard GeoJSON root must be an object")
        self._crs = crs
        self._features = polygons_from_geojson(
            payload_obj, field_map=field_map or GENERIC_HAZARD_FIELD_MAP
        )
        self._meta = meta or HazardLayerMeta(
            source_name=self._path.name,
            source_dataset=self._path.name,
            resolution="vector",
            crs=crs,
        )

    @property
    def crs(self) -> str:
        return self._crs

    def features_in(self, bbox: BoundingBox | None) -> tuple[HazardPolygon, ...]:
        return MemoryHazardVector(self._features, crs=self._crs, meta=self._meta).features_in(bbox)

    def layer_meta(self) -> HazardLayerMeta:
        return self._meta


class GeopandasHazardSource:
    """GeoPackage or shapefile polygons. geopandas is imported on first read."""

    def __init__(
        self,
        path: Path | str,
        *,
        field_map: HazardFieldMap | None = None,
        meta: HazardLayerMeta | None = None,
        layer: str | None = None,
        crs: str = "EPSG:4326",
    ) -> None:
        self._path = Path(path)
        self._field_map = field_map or GENERIC_HAZARD_FIELD_MAP
        self._meta = meta
        self._layer = layer
        self._crs = crs
        self._features: tuple[HazardPolygon, ...] | None = None

    @property
    def crs(self) -> str:
        return self._crs

    def features_in(self, bbox: BoundingBox | None) -> tuple[HazardPolygon, ...]:
        loaded = self._load()
        return MemoryHazardVector(loaded, crs=self._crs, meta=self.layer_meta()).features_in(bbox)

    def layer_meta(self) -> HazardLayerMeta:
        if self._meta is not None:
            return self._meta
        return HazardLayerMeta(
            source_name=self._path.name,
            source_dataset=self._path.name,
            resolution="vector",
            crs=self._crs,
        )

    def _load(self) -> tuple[HazardPolygon, ...]:
        if self._features is not None:
            return self._features
        geopandas = _require_geopandas()
        kwargs: dict[str, str] = {}
        if self._layer and self._path.suffix.lower() == ".gpkg":
            kwargs["layer"] = self._layer
        try:
            frame = geopandas.read_file(self._path, **kwargs)
        except (OSError, ValueError) as exc:
            raise HazardError(f"could not read hazard vector {self._path}") from exc
        if getattr(frame, "crs", None) is not None:
            self._crs = str(frame.crs) or self._crs
        features: list[HazardPolygon] = []
        for index, row in frame.iterrows():
            geom = row.geometry if hasattr(row, "geometry") else None
            if geom is None:
                continue
            geo = geom.__geo_interface__ if hasattr(geom, "__geo_interface__") else None
            if not isinstance(geo, dict):
                continue
            properties = {str(key): row[key] for key in frame.columns if key != "geometry"}
            site_id = _as_text(properties.get(self._field_map.site_id)) or f"feature-{index}"
            parts = _polygon_parts(geo)
            for part_index, (outer, holes) in enumerate(parts):
                part_id = site_id if len(parts) == 1 else f"{site_id}#{part_index}"
                features.append(
                    HazardPolygon(
                        site_id=str(part_id),
                        coordinates=outer,
                        holes=holes,
                        raw_value=_as_raw(properties.get(self._field_map.value)),
                        source_category=(
                            _as_text(properties.get(self._field_map.category))
                            if self._field_map.category
                            else None
                        ),
                        return_period=(
                            _as_text(properties.get(self._field_map.return_period))
                            if self._field_map.return_period
                            else None
                        ),
                    )
                )
        self._features = tuple(features)
        return self._features


@dataclass(frozen=True, slots=True)
class HazardSiteResult:
    """One site sample plus the indicator observation."""

    record: HazardRecord
    observation: IndicatorObservation
    analysis: HazardAnalysisResult | None


@dataclass(frozen=True, slots=True)
class CandidateHazardResult:
    """Per-candidate records merged onto a copy of the input table."""

    table: CandidateTable
    records: tuple[HazardRecord, ...]
    observations: tuple[IndicatorObservation, ...]


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


def _open_path(
    path: Path,
    *,
    field_map: HazardFieldMap,
    meta: HazardLayerMeta | None,
    layer: str | None,
    crs: str,
) -> tuple[HazardRasterSource | None, HazardVectorSource | None]:
    suffix = path.suffix.lower()
    if suffix in _RASTERIO_SUFFIXES:
        return RasterioHazardSource(path, meta=meta), None
    if suffix in _GEOPANDAS_SUFFIXES:
        return None, GeopandasHazardSource(
            path, field_map=field_map, meta=meta, layer=layer, crs=crs
        )
    if suffix in _GEOJSON_SUFFIXES:
        return None, GeoJSONHazardSource(path, field_map=field_map, meta=meta, crs=crs)
    if suffix in _JSON_GRID_SUFFIXES:
        payload_obj: object = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload_obj, dict) and payload_obj.get("type") in {
            "FeatureCollection",
            "Feature",
        }:
            return None, GeoJSONHazardSource(path, field_map=field_map, meta=meta, crs=crs)
        return JsonHazardRasterSource(path, meta=meta), None
    raise InvalidParameterError(
        f"unsupported hazard file type {path.suffix!r}; use JSON grid, GeoTIFF, "
        "GeoJSON, GeoPackage, or shapefile"
    )


class HazardProviderBase:
    """Generic raster/vector hazard sampler bound to one ``HazardKind``."""

    hazard_kind: HazardKind = HazardKind.FLOOD
    provider_id_default: str = "hazard.generic"
    require_explicit_kind: bool = False

    def __init__(
        self,
        *,
        kind: HazardKind | None = None,
        raster_source: HazardRasterSource | None = None,
        vector_source: HazardVectorSource | None = None,
        path: Path | str | None = None,
        mapping: HazardValueMapping | None = None,
        mapping_path: Path | str | None = None,
        field_map: HazardFieldMap | None = None,
        meta: HazardLayerMeta | None = None,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        provider_id: str | None = None,
        environ: Mapping[str, str] | None = None,
        layer: str | None = None,
        crs: str = "EPSG:4326",
    ) -> None:
        if kind is None and self.require_explicit_kind:
            raise InvalidParameterError(
                f"{type(self).__name__} requires kind; SolarDSS has no default hazard dataset"
            )
        self.hazard_kind = kind or self.hazard_kind
        if raster_source is not None and vector_source is not None:
            raise InvalidParameterError("pass either a raster source or a vector source, not both")
        resolved_path = path
        if resolved_path is None and raster_source is None and vector_source is None:
            env_key = _KIND_PATH_ENV[self.hazard_kind]
            env = environ if environ is not None else None
            raw = None if env is None else env.get(env_key)
            if raw is None and environ is None:
                import os

                raw = os.environ.get(env_key)
            resolved_path = raw
        self._raster = raster_source
        self._vector = vector_source
        self._path = Path(resolved_path) if resolved_path else None
        self._field_map = field_map or GENERIC_HAZARD_FIELD_MAP
        self._meta_override = meta
        self._layer = layer
        self._crs = crs
        self._mapping: HazardValueMapping | None
        if mapping is not None:
            self._mapping = mapping
        elif mapping_path is not None:
            self._mapping = load_hazard_mapping(mapping_path)
        else:
            self._mapping = None
        self._cache = cache
        self._clock: Clock = UtcClock() if clock is None else clock
        self._provider_id = provider_id or self.provider_id_default

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def indicator_id(self) -> str:
        return KIND_TO_INDICATOR[self.hazard_kind]

    def _configured(self) -> bool:
        return self._raster is not None or self._vector is not None or self._path is not None

    def _ensure_source(self) -> None:
        if self._raster is not None or self._vector is not None:
            return
        if self._path is None:
            raise HazardError("hazard layer is not configured")
        if not self._path.exists():
            raise HazardError(f"hazard path does not exist: {self._path}")
        self._raster, self._vector = _open_path(
            self._path,
            field_map=self._field_map,
            meta=self._meta_override,
            layer=self._layer,
            crs=self._crs,
        )

    def _meta(self) -> HazardLayerMeta:
        if self._meta_override is not None:
            return self._meta_override
        if self._raster is not None:
            return self._raster.layer_meta()
        if self._vector is not None:
            return self._vector.layer_meta()
        return HazardLayerMeta(
            source_name="unconfigured hazard layer",
            source_dataset="none",
            crs=self._crs,
        )

    def health(self) -> ProviderHealth:
        checked_at = self._clock.now()
        details: dict[str, JsonValue] = {
            "kind": self.hazard_kind.value,
            "indicator_id": self.indicator_id,
            "path": None if self._path is None else str(self._path),
            "raster": self._raster is not None,
            "vector": self._vector is not None,
            "has_mapping": self._mapping is not None,
            "rasterio": gis_module_available("rasterio"),
            "geopandas": gis_module_available("geopandas"),
            "global_dataset": False,
        }
        if not self._configured():
            return ProviderHealth(
                status=HealthStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                reason=(
                    f"{self.hazard_kind.value} hazard layer is not configured; "
                    "SolarDSS has no global hazard dataset"
                ),
                checked_at=checked_at,
                details=details,
            )
        if (
            self._path is not None
            and not self._path.exists()
            and self._raster is None
            and self._vector is None
        ):
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                reason=f"hazard path does not exist: {self._path}",
                checked_at=checked_at,
                details=details,
            )
        return ProviderHealth(
            status=HealthStatus.OK,
            provider_id=self._provider_id,
            reason=f"{self.hazard_kind.value} hazard provider is configured",
            checked_at=checked_at,
            details=details,
        )

    def hazard_available(self, request: FetchRequest | None = None) -> ProviderAvailability:
        """Whether this provider can serve its hazard. No place is required."""

        indicator = self.indicator_id if request is None else request.indicator_id
        if request is not None and request.indicator_id != self.indicator_id:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=(
                    f"{self.__class__.__name__} serves {self.indicator_id}, "
                    f"not {request.indicator_id}"
                ),
            )
        if not self._configured():
            return ProviderAvailability(
                status=AvailabilityStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason=(
                    f"{self.hazard_kind.value} hazard layer is not configured; "
                    "SolarDSS has no global hazard dataset"
                ),
            )
        if (
            self._path is not None
            and not self._path.exists()
            and self._raster is None
            and self._vector is None
        ):
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason=f"hazard path does not exist: {self._path}",
            )
        if request is None:
            return ProviderAvailability(
                status=AvailabilityStatus.AVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason=f"{self.hazard_kind.value} hazard layer is configured",
            )
        point = request_point(request)
        box = request_bbox(request)
        if point is None and box is None:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason="hazard sampling requires a point or bbox",
            )
        try:
            self._aggregation(request)
        except InvalidParameterError as exc:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=indicator,
                reason=exc.message,
            )
        return ProviderAvailability(
            status=AvailabilityStatus.AVAILABLE,
            provider_id=self._provider_id,
            retryable=False,
            indicator_id=indicator,
            reason=f"{self.hazard_kind.value} hazard layer can be sampled for this request",
        )

    def availability(self, request: FetchRequest) -> ProviderAvailability:
        return self.hazard_available(request)

    def coverage(self, request: FetchRequest) -> CoverageReport:
        availability = self.hazard_available(request)
        meta = self._meta()
        return CoverageReport(
            status=availability.status,
            provider_id=self._provider_id,
            spatial=SpatialCoverage(
                resolution=meta.resolution or "hazard-layer",
                crs=meta.crs,
                bbox=request_bbox(request),
            ),
            temporal=request.temporal
            or TemporalCoverage(
                resolution="snapshot",
                aggregation=None if meta.source_year is None else str(meta.source_year),
            ),
            reason=availability.reason,
        )

    def _aggregation(self, request: FetchRequest) -> HazardAggregation | None:
        raw = _text_param(request.parameters, "hazard_aggregation")
        if raw is None:
            return None
        try:
            return HazardAggregation(raw)
        except ValueError as exc:
            raise InvalidParameterError(f"unknown hazard_aggregation {raw!r}") from exc

    def _sample_request(self, request: FetchRequest) -> tuple[HazardAnalysisResult, str | None]:
        self._ensure_source()
        meta = self._meta()
        aggregation = self._aggregation(request)
        point = request_point(request)
        box = request_bbox(request)
        source_category: str | None = None
        if self._raster is not None:
            if box is None and point is not None:
                loc_crs = "EPSG:4326"
                if request.location is not None and request.location.crs:
                    loc_crs = request.location.crs
                box = _point_window_bbox(point, loc_crs)
            if box is None:
                raise InvalidParameterError("hazard raster sampling requires a point or bbox")
            grid = self._raster.read_window(box)
            if point is not None and aggregation in (None, HazardAggregation.CELL):
                sample = sample_hazard_cell(grid, point[0], point[1])
            else:
                chosen = aggregation or HazardAggregation.MAJORITY
                mask = mask_from_bbox(grid, box.min_lon, box.min_lat, box.max_lon, box.max_lat)
                sample = aggregate_hazard_cells(
                    grid, mask, aggregation=chosen, mapping=self._mapping
                )
        elif self._vector is not None:
            if point is None:
                raise InvalidParameterError("hazard vector sampling requires a point")
            features = self._vector.features_in(box)
            sample = sample_hazard_polygon(point[0], point[1], features)
            if not sample.nodata:
                for item in features:
                    matched = (
                        item.raw_value == sample.raw_value
                        or item.source_category == sample.raw_value
                    )
                    if not matched:
                        continue
                    source_category = item.source_category
                    if item.return_period and meta.return_period is None:
                        meta = meta.model_copy(update={"return_period": item.return_period})
                    break
        else:
            raise HazardError("hazard layer is not configured")
        analysis = analyze_hazard(
            sample,
            kind=self.hazard_kind,
            meta=meta,
            mapping=self._mapping,
            source_category=source_category,
        )
        return_period = meta.return_period
        return analysis, return_period

    def _observation_from_analysis(
        self,
        request: FetchRequest,
        analysis: HazardAnalysisResult,
        retrieved_at: datetime,
    ) -> IndicatorObservation:
        record = analysis.record
        meta = self._meta()
        dataset = DatasetRef(
            name=record.source_name,
            media_type="application/json",
            notes=(
                f"source_dataset={record.source_dataset}",
                f"source_version={record.source_version}",
                f"source_year={record.source_year}",
                f"resolution={record.resolution}",
                f"return_period={record.return_period}",
                f"mapping_id={record.mapping_id}",
                f"aggregation={record.aggregation.value}",
            ),
        )
        spatial = SpatialCoverage(
            resolution=record.resolution or "hazard-layer",
            crs=meta.crs,
            bbox=request_bbox(request),
        )
        temporal = request.temporal or TemporalCoverage(
            resolution="snapshot",
            aggregation=None if record.source_year is None else str(record.source_year),
        )
        if record.missing:
            return self._miss(
                request,
                ObservationStatus.MISSING_COVERAGE,
                record.warnings[0] if record.warnings else "hazard sample is missing",
                retrieved_at,
                dataset=dataset,
                spatial=spatial,
            )
        if record.score is not None:
            value: float | str = float(record.score)
            unit = "1"
        elif isinstance(record.raw_value, float):
            value = float(record.raw_value)
            unit = "1"
        elif record.category is not None:
            value = record.category
            unit = "categorical"
        else:
            value = str(record.raw_value)
            unit = "categorical"
        flags = (
            (QualityFlag.UNKNOWN.value,)
            if (
                record.category is None and self._mapping is not None and self._mapping.has_legend()
            )
            else (QualityFlag.NOMINAL.value,)
        )
        assumptions = (
            f"raw_value={record.raw_value}",
            f"category={record.category}",
            f"score={record.score}",
            f"source={record.source_name}",
            f"source_dataset={record.source_dataset}",
            f"resolution={record.resolution}",
            f"return_period={record.return_period}",
            f"source_year={record.source_year}",
            f"source_version={record.source_version}",
            f"mapping_id={record.mapping_id}",
            f"aggregation={record.aggregation.value}",
            "SolarDSS does not model this hazard",
            "Low/Medium/High are not assigned without an explicit source legend",
            *record.uncertainty_notes,
            *record.warnings,
        )
        origin = (
            ValueOrigin.DERIVED
            if record.category is not None or record.score is not None
            else ValueOrigin.RAW
        )
        kind = (
            ValueProvenanceKind.DERIVED_METHOD
            if origin is ValueOrigin.DERIVED
            else ValueProvenanceKind.MEASURED_OR_PROVIDER
        )
        return IndicatorObservation(
            indicator_id=self.indicator_id,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=record.source_name,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=kind,
                value_origin=origin,
                source_version=record.source_version,
                calculation_method=CALC_HAZARD_SAMPLE if origin is ValueOrigin.DERIVED else None,
                dataset=dataset,
            ),
            spatial=spatial,
            temporal=temporal,
            value=value,
            unit=unit,
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
        meta = self._meta()
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=status,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=meta.source_name,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=ValueProvenanceKind.MEASURED_OR_PROVIDER,
                value_origin=ValueOrigin.RAW,
                source_version=meta.source_version,
                dataset=dataset or DatasetRef(name=meta.source_name, media_type="application/json"),
            ),
            spatial=spatial
            or SpatialCoverage(
                resolution=meta.resolution or "hazard-layer",
                crs=meta.crs,
                bbox=request_bbox(request),
            ),
            temporal=request.temporal or TemporalCoverage(resolution="snapshot"),
            assumptions=(reason,),
        )

    def hazard_for_site(self, request: FetchRequest) -> HazardSiteResult:
        """Sample the hazard at one site. Missing coverage stays missing."""

        retrieved_at = self._clock.now()
        availability = self.hazard_available(request)
        if availability.status is not AvailabilityStatus.AVAILABLE:
            observation = self._miss(
                request,
                ObservationStatus(availability.status.value),
                availability.reason,
                retrieved_at,
            )
            record = HazardRecord(
                kind=self.hazard_kind,
                indicator_id=self.indicator_id,
                source_name=self._meta().source_name,
                source_dataset=self._meta().source_dataset,
                missing=True,
                warnings=(availability.reason,),
            )
            return HazardSiteResult(record=record, observation=observation, analysis=None)
        request_hash = request_cache_key(self._provider_id, request)
        cached = self._read_cache(request_hash)
        try:
            analysis, _return_period = self._sample_request(request)
        except HazardCoverageError as exc:
            observation = self._miss(
                request, ObservationStatus.MISSING_COVERAGE, exc.message, retrieved_at
            )
            record = HazardRecord(
                kind=self.hazard_kind,
                indicator_id=self.indicator_id,
                source_name=self._meta().source_name,
                source_dataset=self._meta().source_dataset,
                missing=True,
                warnings=(exc.message,),
            )
            return HazardSiteResult(record=record, observation=observation, analysis=None)
        except (HazardError, InvalidParameterError, MissingExtraError) as exc:
            status = (
                ObservationStatus.NOT_CONFIGURED
                if isinstance(exc, MissingExtraError)
                else ObservationStatus.UNAVAILABLE
            )
            observation = self._miss(request, status, exc.message, retrieved_at)
            record = HazardRecord(
                kind=self.hazard_kind,
                indicator_id=self.indicator_id,
                source_name=self._meta().source_name,
                source_dataset=self._meta().source_dataset,
                missing=True,
                warnings=(exc.message,),
            )
            return HazardSiteResult(record=record, observation=observation, analysis=None)
        if cached is not None:
            record = analysis.record.model_copy(update={"quality_flags": cached.quality_flags})
            return HazardSiteResult(record=record, observation=cached, analysis=analysis)
        observation = self._observation_from_analysis(request, analysis, retrieved_at)
        record = analysis.record.model_copy(update={"quality_flags": observation.quality_flags})
        if observation.status is ObservationStatus.OBSERVED:
            self._write_cache(observation.provenance.request_hash, observation)
        return HazardSiteResult(record=record, observation=observation, analysis=analysis)

    def fetch(self, request: FetchRequest) -> IndicatorObservation:
        return self.hazard_for_site(request).observation

    def hazard_for_candidates(
        self,
        table: CandidateTable | pd.DataFrame,
        *,
        x_column: str = "longitude",
        y_column: str = "latitude",
        crs: str = "EPSG:4326",
        id_column: str = DEFAULT_ID_COLUMN,
    ) -> CandidateHazardResult:
        """Sample the hazard at each candidate point. Empty cells stay missing."""

        if isinstance(table, CandidateTable):
            candidates = table
        else:
            candidates = validate_candidate_table(table, id_column=id_column)
        missing_cols = [
            name for name in (x_column, y_column) if name not in candidates.frame.columns
        ]
        if missing_cols:
            raise SchemaError(f"Candidate table is missing coordinate columns: {missing_cols}")
        records: list[HazardRecord] = []
        observations: list[IndicatorObservation] = []
        for _, row in candidates.frame.iterrows():
            candidate_id = str(row[candidates.id_column])
            x = float(row[x_column])
            y = float(row[y_column])
            request = FetchRequest(
                indicator_id=self.indicator_id,
                location=GeographicRef(name=candidate_id, longitude=x, latitude=y, crs=crs),
            )
            result = self.hazard_for_site(request)
            records.append(result.record.model_copy(update={"candidate_id": candidate_id}))
            observations.append(result.observation)
        prefix = self.indicator_id
        frame = candidates.frame.copy()
        frame[f"{prefix}_raw"] = [item.raw_value for item in records]
        frame[f"{prefix}_category"] = [item.category for item in records]
        frame[f"{prefix}_score"] = [item.score for item in records]
        frame[f"{prefix}_status"] = [item.status.value for item in observations]
        frame[f"{prefix}_source"] = [item.source_name for item in records]
        frame[f"{prefix}_resolution"] = [item.resolution for item in records]
        frame[f"{prefix}_return_period"] = [item.return_period for item in records]
        frame[f"{prefix}_year"] = [item.source_year for item in records]
        frame[f"{prefix}_version"] = [item.source_version for item in records]
        updated = CandidateTable(frame=frame, id_column=candidates.id_column)
        return CandidateHazardResult(
            table=updated, records=tuple(records), observations=tuple(observations)
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


class RasterHazardProvider(HazardProviderBase):
    """Generic raster adapter. ``kind`` selects the registered hazard indicator."""

    provider_id_default = "hazard.raster"
    require_explicit_kind = True


class VectorHazardProvider(HazardProviderBase):
    """Generic vector adapter. ``kind`` selects the registered hazard indicator."""

    provider_id_default = "hazard.vector"
    require_explicit_kind = True


class FloodRiskProvider(HazardProviderBase):
    """Flood hazard map. Does not infer floodplain from water or elevation."""

    hazard_kind = HazardKind.FLOOD
    provider_id_default = "hazard.flood"


class WildfireRiskProvider(HazardProviderBase):
    """Wildfire hazard map. Does not infer fire risk from land cover."""

    hazard_kind = HazardKind.WILDFIRE
    provider_id_default = "hazard.wildfire"


class LandslideRiskProvider(HazardProviderBase):
    """Landslide hazard map. Slope is not treated as landslide risk."""

    hazard_kind = HazardKind.LANDSLIDE
    provider_id_default = "hazard.landslide"


class HailRiskProvider(HazardProviderBase):
    """Hail hazard map. Thunderstorm counts are not treated as hail risk."""

    hazard_kind = HazardKind.HAIL
    provider_id_default = "hazard.hail"


class HazardCatalog:
    """Dispatch by hazard kind. Missing kinds stay not configured."""

    def __init__(
        self,
        providers: Mapping[HazardKind, HazardProviderBase] | Sequence[HazardProviderBase],
        *,
        provider_id: str = "hazard.catalog",
    ) -> None:
        if isinstance(providers, Mapping):
            self._providers = dict(providers)
        else:
            self._providers = {item.hazard_kind: item for item in providers}
        self._provider_id = provider_id

    def _provider(self, kind: HazardKind) -> HazardProviderBase | None:
        return self._providers.get(kind)

    def hazard_available(
        self, kind: HazardKind, request: FetchRequest | None = None
    ) -> ProviderAvailability:
        provider = self._provider(kind)
        if provider is None:
            return ProviderAvailability(
                status=AvailabilityStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=KIND_TO_INDICATOR[kind],
                reason=(
                    f"no {kind.value} hazard adapter is registered; "
                    "SolarDSS has no global hazard dataset"
                ),
            )
        return provider.hazard_available(request)

    def hazard_for_site(self, kind: HazardKind, request: FetchRequest) -> HazardSiteResult:
        provider = self._provider(kind)
        if provider is None:
            availability = self.hazard_available(kind, request)
            meta_name = "hazard catalog"
            observation = IndicatorObservation(
                indicator_id=KIND_TO_INDICATOR[kind],
                status=ObservationStatus.NOT_CONFIGURED,
                provenance=DataProvenance(
                    provider_id=self._provider_id,
                    source_name=meta_name,
                    retrieved_at=UtcClock().now(),
                    request_hash=request_cache_key(self._provider_id, request),
                    value_kind=ValueProvenanceKind.MEASURED_OR_PROVIDER,
                    value_origin=ValueOrigin.RAW,
                    dataset=DatasetRef(name=meta_name, media_type="application/json"),
                ),
                spatial=SpatialCoverage(
                    resolution="hazard-layer",
                    bbox=request_bbox(request),
                ),
                temporal=request.temporal or TemporalCoverage(resolution="snapshot"),
                assumptions=(availability.reason,),
            )
            record = HazardRecord(
                kind=kind,
                indicator_id=KIND_TO_INDICATOR[kind],
                source_name=meta_name,
                source_dataset="none",
                missing=True,
                warnings=(availability.reason,),
            )
            return HazardSiteResult(record=record, observation=observation, analysis=None)
        return provider.hazard_for_site(request)

    def hazard_for_candidates(
        self,
        kind: HazardKind,
        table: CandidateTable | pd.DataFrame,
        *,
        x_column: str = "longitude",
        y_column: str = "latitude",
        crs: str = "EPSG:4326",
        id_column: str = DEFAULT_ID_COLUMN,
    ) -> CandidateHazardResult:
        provider = self._provider(kind)
        if provider is None:
            raise InvalidParameterError(
                f"no {kind.value} hazard adapter is registered; "
                "SolarDSS has no global hazard dataset"
            )
        return provider.hazard_for_candidates(
            table, x_column=x_column, y_column=y_column, crs=crs, id_column=id_column
        )


def hazard_available(
    provider: HazardProviderBase | HazardCatalog,
    kind: HazardKind | None = None,
    request: FetchRequest | None = None,
) -> ProviderAvailability:
    """Configuration/coverage check. Catalogs require ``kind``."""

    if isinstance(provider, HazardCatalog):
        if kind is None:
            raise InvalidParameterError("hazard_available on a catalog requires kind")
        return provider.hazard_available(kind, request)
    return provider.hazard_available(request)


def hazard_for_site(
    provider: HazardProviderBase | HazardCatalog,
    request: FetchRequest,
    *,
    kind: HazardKind | None = None,
) -> HazardSiteResult:
    """Sample one site. Catalogs require ``kind``."""

    if isinstance(provider, HazardCatalog):
        if kind is None:
            raise InvalidParameterError("hazard_for_site on a catalog requires kind")
        return provider.hazard_for_site(kind, request)
    return provider.hazard_for_site(request)


def hazard_for_candidates(
    provider: HazardProviderBase | HazardCatalog,
    table: CandidateTable | pd.DataFrame,
    *,
    kind: HazardKind | None = None,
    x_column: str = "longitude",
    y_column: str = "latitude",
    crs: str = "EPSG:4326",
    id_column: str = DEFAULT_ID_COLUMN,
) -> CandidateHazardResult:
    """Sample each candidate point. Catalogs require ``kind``."""

    if isinstance(provider, HazardCatalog):
        if kind is None:
            raise InvalidParameterError("hazard_for_candidates on a catalog requires kind")
        return provider.hazard_for_candidates(
            kind, table, x_column=x_column, y_column=y_column, crs=crs, id_column=id_column
        )
    return provider.hazard_for_candidates(
        table, x_column=x_column, y_column=y_column, crs=crs, id_column=id_column
    )


__all__ = [
    "FLOOD_PATH_ENV",
    "GENERIC_HAZARD_FIELD_MAP",
    "HAIL_PATH_ENV",
    "LANDSLIDE_PATH_ENV",
    "WILDFIRE_PATH_ENV",
    "CandidateHazardResult",
    "FloodRiskProvider",
    "GeoJSONHazardSource",
    "GeopandasHazardSource",
    "HailRiskProvider",
    "HazardCatalog",
    "HazardFieldMap",
    "HazardProviderBase",
    "HazardSiteResult",
    "JsonHazardRasterSource",
    "LandslideRiskProvider",
    "MemoryHazardRaster",
    "MemoryHazardVector",
    "RasterHazardProvider",
    "RasterioHazardSource",
    "VectorHazardProvider",
    "WildfireRiskProvider",
    "hazard_available",
    "hazard_for_candidates",
    "hazard_for_site",
    "load_hazard_mapping",
    "polygons_from_geojson",
]
