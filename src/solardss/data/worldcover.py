"""ESA WorldCover land-cover provider. Retrieval is injected, not hardcoded.

Scientific zonal fractions live in ``solardss.indicators.land``. This module
windows a classified raster, preserves native codes (no bilinear resampling),
and maps an explicit legend onto land indicators. ESA WorldCover 10 m is the
first dataset; any north-up classified raster can be supplied through
``LandCoverRasterSource``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from solardss.core.arrays import BoolArray, FloatArray, IntArray
from solardss.core.jsonutil import JsonValue
from solardss.data.contracts import DatasetRef
from solardss.data.geo import BoundingBox
from solardss.data.gis import gis_module_available
from solardss.exceptions import InvalidGridError, InvalidParameterError, MissingExtraError
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.indicators.ids import IndicatorId
from solardss.indicators.land import (
    CALC_AVAILABLE_AREA,
    CALC_CLASS_FRACTION,
    CALC_DOMINANT_CLASS,
    FRACTION_INDICATORS,
    AvailableLandProfile,
    LandCoverAnalysisResult,
    LandCoverClass,
    LandCoverGrid,
    LandCoverLegend,
    analyze_land_cover,
    category_fraction,
    crs_is_geographic,
    resolve_available_profile,
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
from solardss.providers.errors import CacheError, LandCoverCoverageError, LandCoverError
from solardss.providers.region import request_bbox, request_point
from solardss.providers.request import FetchRequest, request_cache_key

ESA_WORLDCOVER_ID: Final = "ESA-WorldCover-10m"
ESA_WORLDCOVER_VERSION: Final = "v200"
ESA_WORLDCOVER_YEAR: Final = 2021
ESA_WORLDCOVER_SOURCE_NAME: Final = "ESA WorldCover"
ESA_WORLDCOVER_RESOLUTION: Final = "10m"
ESA_WORLDCOVER_NATIVE_CRS: Final = "EPSG:4326"
ESA_WORLDCOVER_CATALOGUE_URL: Final = "https://esa-worldcover.org"
ESA_WORLDCOVER_NODATA: Final = 0
WORLDCOVER_PATH_ENV: Final = "SOLARDSS_WORLDCOVER_PATH"
_CACHE_SCHEMA: Final = "solardss.worldcover_obs.v1"

# ESA WorldCover 10 m Product User Manual class table (v100/v200 share codes).
ESA_WORLDCOVER_LEGEND = LandCoverLegend(
    dataset_id=ESA_WORLDCOVER_ID,
    classes=(
        LandCoverClass(code=10, name="Tree cover", category="forest"),
        LandCoverClass(code=20, name="Shrubland", category="shrubland"),
        LandCoverClass(code=30, name="Grassland", category="grassland"),
        LandCoverClass(code=40, name="Cropland", category="agricultural"),
        LandCoverClass(code=50, name="Built-up", category="builtup"),
        LandCoverClass(code=60, name="Bare / sparse vegetation", category="bare_land"),
        LandCoverClass(code=70, name="Snow and ice", category="snow_ice"),
        LandCoverClass(code=80, name="Permanent water bodies", category="water"),
        LandCoverClass(code=90, name="Herbaceous wetland", category="wetland"),
        LandCoverClass(
            code=95,
            name="Mangroves",
            category="wetland",
        ),
        LandCoverClass(code=100, name="Moss and lichen", category="moss_lichen"),
    ),
)

WORLDCOVER_BARE_SPARSE_ONLY = AvailableLandProfile(
    profile_id="worldcover_bare_sparse_only",
    available_categories=("bare_land",),
    excluded_categories=(
        "water",
        "wetland",
        "snow_ice",
        "builtup",
        "forest",
        "agricultural",
        "shrubland",
        "grassland",
        "moss_lichen",
    ),
    notes=(
        "ESA WorldCover: only class 60 (bare / sparse vegetation) is available.",
        "Non-water classes are not treated as available.",
    ),
)

WORLDCOVER_OPEN_GROUND = AvailableLandProfile(
    profile_id="worldcover_open_ground",
    available_categories=("bare_land", "grassland"),
    excluded_categories=(
        "water",
        "wetland",
        "snow_ice",
        "builtup",
        "forest",
        "agricultural",
    ),
    notes=(
        "ESA WorldCover: classes 60 and 30 (bare / sparse, grassland) are available.",
        "Cropland, forest, water, wetland, and built-up remain excluded.",
    ),
)

WORLDCOVER_PROFILES: dict[str, AvailableLandProfile] = {
    WORLDCOVER_BARE_SPARSE_ONLY.profile_id: WORLDCOVER_BARE_SPARSE_ONLY,
    WORLDCOVER_OPEN_GROUND.profile_id: WORLDCOVER_OPEN_GROUND,
}

_SERVED: Final = frozenset(
    {
        IndicatorId.LAND_COVER.value,
        IndicatorId.LAND_AVAILABLE_AREA.value,
        IndicatorId.LAND_FOREST_FRACTION.value,
        IndicatorId.LAND_WATER_FRACTION.value,
        IndicatorId.LAND_WETLAND_FRACTION.value,
        IndicatorId.LAND_AGRICULTURAL_FRACTION.value,
        IndicatorId.LAND_BUILTUP_FRACTION.value,
        IndicatorId.LAND_BARE_LAND_FRACTION.value,
    }
)
_FRACTIONS: Final = tuple(FRACTION_INDICATORS)
LandCoverRetriever = Callable[[BoundingBox, Path], Path]


@dataclass(frozen=True, slots=True)
class LandCoverWindow:
    """North-up classified window. ``cell_size_y`` is the positive southward spacing."""

    codes: IntArray
    crs: str
    origin_x: float
    origin_y: float
    cell_size_x: float
    cell_size_y: float
    nodata: int | None = ESA_WORLDCOVER_NODATA
    source_name: str = ESA_WORLDCOVER_SOURCE_NAME
    source_resolution: str = ESA_WORLDCOVER_RESOLUTION
    source_version: str = ESA_WORLDCOVER_VERSION
    source_year: int = ESA_WORLDCOVER_YEAR


class LandCoverRasterSource(Protocol):
    """Read a classified spatial window. File, object, or CDN implementations."""

    @property
    def crs(self) -> str:
        """Native CRS label of the land-cover raster."""

    def coverage_bbox(self) -> BoundingBox | None:
        """Dataset extent in the native CRS, if known."""

    def read_window(self, bbox: BoundingBox, *, pad_cells: int = 0) -> LandCoverWindow:
        """Return class codes covering ``bbox``. Codes are not resampled."""


class RasterioLandCoverSource:
    """GeoTIFF/COG classified reader. rasterio is imported only when a window is read."""

    def __init__(
        self,
        path: Path | str,
        *,
        source_name: str = ESA_WORLDCOVER_SOURCE_NAME,
        source_resolution: str = ESA_WORLDCOVER_RESOLUTION,
        source_version: str = ESA_WORLDCOVER_VERSION,
        source_year: int = ESA_WORLDCOVER_YEAR,
        nodata: int | None = ESA_WORLDCOVER_NODATA,
    ) -> None:
        self._path = Path(path)
        self._source_name = source_name
        self._source_resolution = source_resolution
        self._source_version = source_version
        self._source_year = source_year
        self._nodata = nodata
        self._crs = ESA_WORLDCOVER_NATIVE_CRS

    @property
    def crs(self) -> str:
        return self._crs

    def coverage_bbox(self) -> BoundingBox | None:
        rasterio = _require_rasterio()
        with rasterio.open(self._path) as dataset:
            self._crs = str(dataset.crs) if dataset.crs else ESA_WORLDCOVER_NATIVE_CRS
            bounds = dataset.bounds
            return BoundingBox(
                min_lon=float(bounds.left),
                min_lat=float(bounds.bottom),
                max_lon=float(bounds.right),
                max_lat=float(bounds.top),
                crs=self._crs,
            )

    def read_window(self, bbox: BoundingBox, *, pad_cells: int = 0) -> LandCoverWindow:
        rasterio = _require_rasterio()
        from rasterio.enums import Resampling
        from rasterio.errors import RasterioError
        from rasterio.windows import Window, from_bounds

        try:
            with rasterio.open(self._path) as dataset:
                self._crs = str(dataset.crs) if dataset.crs else ESA_WORLDCOVER_NATIVE_CRS
                transform = dataset.transform
                window = from_bounds(
                    bbox.min_lon, bbox.min_lat, bbox.max_lon, bbox.max_lat, transform
                )
                window = Window(
                    col_off=int(np.floor(float(window.col_off))) - pad_cells,
                    row_off=int(np.floor(float(window.row_off))) - pad_cells,
                    width=int(np.ceil(float(window.width))) + 2 * pad_cells,
                    height=int(np.ceil(float(window.height))) + 2 * pad_cells,
                )
                if window.width <= 0 or window.height <= 0:
                    raise LandCoverCoverageError(
                        "requested window does not intersect the land-cover raster"
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
            raise LandCoverError(f"failed to read land-cover window: {exc}") from exc
        masked = np.ma.asarray(data)  # type: ignore[no-untyped-call]
        filled = np.asarray(
            np.ma.filled(masked, ESA_WORLDCOVER_NODATA),  # type: ignore[no-untyped-call]
            dtype=np.int16,
        )
        fill_nodata = int(nodata) if nodata is not None else self._nodata
        return LandCoverWindow(
            codes=np.asarray(filled, dtype=np.int16),
            crs=self._crs,
            origin_x=float(window_transform.c),
            origin_y=float(window_transform.f),
            cell_size_x=abs(float(window_transform.a)),
            cell_size_y=abs(float(window_transform.e)),
            nodata=fill_nodata,
            source_name=self._source_name,
            source_resolution=self._source_resolution,
            source_version=self._source_version,
            source_year=self._source_year,
        )


@dataclass(frozen=True, slots=True)
class MemoryLandCoverSource:
    """In-memory north-up classified raster used by tests and injected fixtures."""

    codes: IntArray
    coordinate_system: str
    origin_x: float
    origin_y: float
    cell_size_x: float
    cell_size_y: float
    nodata: int | None = ESA_WORLDCOVER_NODATA
    source_name: str = "array land cover"
    source_resolution: str = "synthetic"
    source_version: str = "fixture"
    source_year: int = ESA_WORLDCOVER_YEAR

    @property
    def crs(self) -> str:
        return self.coordinate_system

    def coverage_bbox(self) -> BoundingBox | None:
        rows, cols = np.asarray(self.codes).shape
        return BoundingBox(
            min_lon=self.origin_x,
            min_lat=self.origin_y - rows * self.cell_size_y,
            max_lon=self.origin_x + cols * self.cell_size_x,
            max_lat=self.origin_y,
            crs=self.coordinate_system,
        )

    def read_window(self, bbox: BoundingBox, *, pad_cells: int = 0) -> LandCoverWindow:
        codes = np.asarray(self.codes, dtype=np.int16)
        rows, cols = codes.shape
        col0 = int(np.floor((bbox.min_lon - self.origin_x) / self.cell_size_x)) - pad_cells
        col1 = int(np.ceil((bbox.max_lon - self.origin_x) / self.cell_size_x)) + pad_cells
        row0 = int(np.floor((self.origin_y - bbox.max_lat) / self.cell_size_y)) - pad_cells
        row1 = int(np.ceil((self.origin_y - bbox.min_lat) / self.cell_size_y)) + pad_cells
        row0 = max(row0, 0)
        col0 = max(col0, 0)
        row1 = min(row1, rows)
        col1 = min(col1, cols)
        if row1 <= row0 or col1 <= col0:
            raise LandCoverCoverageError(
                "requested window does not intersect the in-memory land-cover raster"
            )
        return LandCoverWindow(
            codes=np.asarray(codes[row0:row1, col0:col1], dtype=np.int16),
            crs=self.coordinate_system,
            origin_x=self.origin_x + col0 * self.cell_size_x,
            origin_y=self.origin_y - row0 * self.cell_size_y,
            cell_size_x=self.cell_size_x,
            cell_size_y=self.cell_size_y,
            nodata=self.nodata,
            source_name=self.source_name,
            source_resolution=self.source_resolution,
            source_version=self.source_version,
            source_year=self.source_year,
        )


@dataclass(frozen=True, slots=True)
class LandCoverAnalysisBundle:
    """Normalized land indicators plus zonal class accounting."""

    analysis: LandCoverAnalysisResult | None
    observations: tuple[IndicatorObservation, ...]
    source_crs: str | None
    quality_warnings: tuple[str, ...]


def _require_rasterio() -> Any:
    try:
        import rasterio
    except ImportError as exc:
        raise MissingExtraError(
            "Optional GIS extra is not installed. Install with: pip install 'solardss[gis]'."
        ) from exc
    return rasterio


def _require_pyproj() -> Any:
    try:
        import pyproj
    except ImportError as exc:
        raise MissingExtraError(
            "Optional GIS extra is not installed. Install with: pip install 'solardss[gis]'."
        ) from exc
    return pyproj


def _require_shapely() -> Any:
    try:
        import shapely
        from shapely import wkt as shapely_wkt
        from shapely.ops import transform as shapely_transform
    except ImportError as exc:
        raise MissingExtraError(
            "Optional GIS extra is not installed. Install with: pip install 'solardss[gis]'."
        ) from exc
    return shapely, shapely_wkt, shapely_transform


def transform_bbox(bbox: BoundingBox, dst_crs: str) -> BoundingBox:
    """Reproject an axis-aligned box. Requires pyproj when CRS labels differ."""

    if bbox.crs == dst_crs:
        return bbox
    pyproj = _require_pyproj()
    transformer = pyproj.Transformer.from_crs(bbox.crs, dst_crs, always_xy=True)
    xs = [bbox.min_lon, bbox.max_lon, bbox.min_lon, bbox.max_lon]
    ys = [bbox.min_lat, bbox.min_lat, bbox.max_lat, bbox.max_lat]
    tx, ty = transformer.transform(xs, ys)
    return BoundingBox(
        min_lon=float(min(tx)),
        min_lat=float(min(ty)),
        max_lon=float(max(tx)),
        max_lat=float(max(ty)),
        crs=dst_crs,
    )


def window_to_grid(window: LandCoverWindow) -> LandCoverGrid:
    return LandCoverGrid(
        codes=window.codes,
        cell_size_x=window.cell_size_x,
        cell_size_y=window.cell_size_y,
        nodata=window.nodata,
        crs=window.crs,
        origin_x=window.origin_x,
        origin_y=window.origin_y,
    )


def _cell_centers(grid: LandCoverGrid) -> tuple[FloatArray, FloatArray]:
    rows, cols = np.asarray(grid.codes).shape
    origin_x = 0.0 if grid.origin_x is None else float(grid.origin_x)
    origin_y = 0.0 if grid.origin_y is None else float(grid.origin_y)
    xs = origin_x + (np.arange(cols, dtype=np.float64) + 0.5) * float(grid.cell_size_x)
    ys = origin_y - (np.arange(rows, dtype=np.float64) + 0.5) * float(grid.cell_size_y)
    xx, yy = np.meshgrid(xs, ys)
    return np.asarray(xx, dtype=np.float64), np.asarray(yy, dtype=np.float64)


def mask_from_bbox(grid: LandCoverGrid, bbox: BoundingBox) -> BoolArray:
    xx, yy = _cell_centers(grid)
    return np.asarray(
        (xx >= bbox.min_lon) & (xx <= bbox.max_lon) & (yy >= bbox.min_lat) & (yy <= bbox.max_lat),
        dtype=bool,
    )


def mask_from_point(grid: LandCoverGrid, x: float, y: float) -> BoolArray:
    xx, yy = _cell_centers(grid)
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    mask = np.zeros(xx.shape, dtype=bool)
    index = np.unravel_index(int(np.argmin(dist2)), dist2.shape)
    mask[index] = True
    return mask


def mask_from_wkt(grid: LandCoverGrid, wkt: str, *, geometry_crs: str) -> BoolArray:
    shapely, shapely_wkt, shapely_transform = _require_shapely()
    geom = shapely_wkt.loads(wkt)
    if grid.crs and geometry_crs != grid.crs:
        pyproj = _require_pyproj()
        transformer = pyproj.Transformer.from_crs(geometry_crs, grid.crs, always_xy=True)
        geom = shapely_transform(transformer.transform, geom)
    xx, yy = _cell_centers(grid)
    points = shapely.points(xx, yy)
    return np.asarray(shapely.contains(geom, points), dtype=bool)


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


def _int_tuple_param(parameters: Mapping[str, JsonValue], key: str) -> tuple[int, ...] | None:
    value = parameters.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return (value,)
    if isinstance(value, list):
        items = [
            int(item) for item in value if isinstance(item, int) and not isinstance(item, bool)
        ]
        return tuple(items)
    return None


class LandCoverRasterProvider:
    """Window a classified raster and map legend statistics onto land indicators.

    Download is optional and injected as ``retriever``. Classified values are
    never bilinearly resampled.
    """

    provider_id_default = "esa.worldcover"
    source_name_default = ESA_WORLDCOVER_SOURCE_NAME
    served = _SERVED

    def __init__(
        self,
        *,
        source: LandCoverRasterSource | None = None,
        raster_path: Path | str | None = None,
        retriever: LandCoverRetriever | None = None,
        legend: LandCoverLegend | None = None,
        profiles: Mapping[str, AvailableLandProfile] | None = None,
        default_profile: AvailableLandProfile | None = None,
        dataset_id: str = ESA_WORLDCOVER_ID,
        source_name: str | None = None,
        source_version: str = ESA_WORLDCOVER_VERSION,
        source_year: int = ESA_WORLDCOVER_YEAR,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        provider_id: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        path = raster_path
        if path is None:
            env = environ if environ is not None else None
            raw = None if env is None else env.get(WORLDCOVER_PATH_ENV)
            if raw is None and environ is None:
                import os

                raw = os.environ.get(WORLDCOVER_PATH_ENV)
            path = raw
        self._source = source
        self._path = Path(path) if path else None
        self._retriever = retriever
        self._legend = legend or ESA_WORLDCOVER_LEGEND
        self._profiles = dict(profiles) if profiles is not None else dict(WORLDCOVER_PROFILES)
        self._default_profile = default_profile
        self._dataset_id = dataset_id
        self._source_name = source_name or self.source_name_default
        self._source_version = source_version
        self._source_year = source_year
        self._cache = cache
        self._clock: Clock = UtcClock() if clock is None else clock
        self._provider_id = provider_id or self.provider_id_default
        self._temp_dirs: list[Any] = []

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def _configured(self) -> bool:
        if self._source is not None:
            return True
        if self._retriever is not None:
            return True
        return self._path is not None

    def health(self) -> ProviderHealth:
        checked_at = self._clock.now()
        details: dict[str, JsonValue] = {
            "dataset_id": self._dataset_id,
            "source_version": self._source_version,
            "source_year": self._source_year,
            "source_url": ESA_WORLDCOVER_CATALOGUE_URL,
            "has_local_raster": self._path is not None or self._source is not None,
            "rasterio": gis_module_available("rasterio"),
            "legend_classes": [item.name for item in self._legend.classes],
        }
        if not self._configured():
            return ProviderHealth(
                status=HealthStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                reason=(
                    "land-cover raster is not configured: pass a LandCoverRasterSource, "
                    "SOLARDSS_WORLDCOVER_PATH, or an injected retriever"
                ),
                checked_at=checked_at,
                details=details,
            )
        if self._path is not None and not self._path.exists() and self._source is None:
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                reason="land-cover path does not exist",
                checked_at=checked_at,
                details=details,
            )
        return ProviderHealth(
            status=HealthStatus.OK,
            provider_id=self._provider_id,
            reason="land-cover provider is configured",
            checked_at=checked_at,
            details=details,
        )

    def availability(self, request: FetchRequest) -> ProviderAvailability:
        if request.indicator_id not in self.served:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="indicator is not served by the land-cover provider",
            )
        if not self._configured():
            return ProviderAvailability(
                status=AvailabilityStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="land-cover raster is not configured",
            )
        try:
            profile = self._profile_from_request(request)
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
                reason="land-cover analysis requires a point, bbox, or geometry_wkt",
            )
        if request.indicator_id == IndicatorId.LAND_AVAILABLE_AREA.value and profile is None:
            return ProviderAvailability(
                status=AvailabilityStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=(
                    "LAND_AVAILABLE_AREA requires an explicit available-land profile "
                    "(available_land_profile or available_categories); non-water is "
                    "not treated as available"
                ),
            )
        return ProviderAvailability(
            status=AvailabilityStatus.AVAILABLE,
            provider_id=self._provider_id,
            retryable=False,
            indicator_id=request.indicator_id,
            reason="land-cover raster can be windowed for this request",
        )

    def coverage(self, request: FetchRequest) -> CoverageReport:
        availability = self.availability(request)
        spatial = SpatialCoverage(
            resolution=ESA_WORLDCOVER_RESOLUTION,
            crs=ESA_WORLDCOVER_NATIVE_CRS,
            bbox=request_bbox(request),
        )
        temporal = request.temporal or TemporalCoverage(
            resolution="annual",
            start=None,
            end=None,
            aggregation=str(self._source_year),
        )
        return CoverageReport(
            status=availability.status,
            provider_id=self._provider_id,
            spatial=spatial,
            temporal=temporal,
            reason=availability.reason,
        )

    def fetch(self, request: FetchRequest) -> IndicatorObservation:
        request_hash = request_cache_key(self._provider_id, request)
        availability = self.availability(request)
        if availability.status is AvailabilityStatus.TEMPORARY_FAILURE:
            return self._miss(
                request,
                ObservationStatus.TEMPORARY_FAILURE,
                availability.reason,
                self._clock.now(),
            )
        cached = self._read_cache(request_hash)
        if cached is not None:
            return cached
        bundle = self.fetch_land_cover(request)
        for item in bundle.observations:
            if item.status is ObservationStatus.OBSERVED:
                self._write_cache(item.provenance.request_hash, item)
            if item.indicator_id == request.indicator_id:
                return item
        return self._miss(
            request,
            ObservationStatus.UNAVAILABLE,
            "indicator was not produced from the land-cover window",
            self._clock.now(),
        )

    def fetch_land_cover(self, request: FetchRequest) -> LandCoverAnalysisBundle:
        availability = self.availability(request)
        retrieved_at = self._clock.now()
        if availability.status is not AvailabilityStatus.AVAILABLE:
            if (
                request.indicator_id == IndicatorId.LAND_AVAILABLE_AREA.value
                and availability.status is AvailabilityStatus.NOT_CONFIGURED
                and self._configured()
            ):
                miss = self._miss(
                    request,
                    ObservationStatus.NOT_CONFIGURED,
                    availability.reason,
                    retrieved_at,
                )
                return LandCoverAnalysisBundle(
                    analysis=None,
                    observations=(miss,),
                    source_crs=None,
                    quality_warnings=(availability.reason,),
                )
            miss = self._miss(
                request,
                ObservationStatus(availability.status.value),
                availability.reason,
                retrieved_at,
            )
            return LandCoverAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                quality_warnings=(availability.reason,),
            )
        try:
            source = self._resolve_source(request)
            request_box, geometry_wkt, geometry_crs, point_xy = self._place(request, source.crs)
            native_box = transform_bbox(request_box, source.crs)
            window = source.read_window(native_box)
            grid = window_to_grid(window)
            mask = self._mask(grid, request_box, geometry_wkt, geometry_crs, point_xy)
            profile = self._profile_from_request(request)
            analysis = analyze_land_cover(grid, self._legend, mask=mask, profile=profile)
        except MissingExtraError as exc:
            miss = self._miss(request, ObservationStatus.NOT_CONFIGURED, exc.message, retrieved_at)
            return LandCoverAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                quality_warnings=(exc.message,),
            )
        except LandCoverCoverageError as exc:
            miss = self._miss(
                request, ObservationStatus.MISSING_COVERAGE, exc.message, retrieved_at
            )
            return LandCoverAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                quality_warnings=(exc.message,),
            )
        except (
            LandCoverError,
            InvalidGridError,
            InvalidParameterError,
            ValueError,
            OSError,
        ) as exc:
            reason = getattr(exc, "message", str(exc))
            miss = self._miss(
                request, ObservationStatus.TEMPORARY_FAILURE, str(reason), retrieved_at
            )
            return LandCoverAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                quality_warnings=(str(reason),),
            )
        observations = self._observations_from_analysis(request, analysis, window, retrieved_at)
        return LandCoverAnalysisBundle(
            analysis=analysis,
            observations=observations,
            source_crs=window.crs,
            quality_warnings=analysis.summary.warnings,
        )

    def _profile_from_request(self, request: FetchRequest) -> AvailableLandProfile | None:
        parameters = request.parameters
        return (
            resolve_available_profile(
                profile_id=_text_param(parameters, "available_land_profile"),
                available_categories=_str_tuple_param(parameters, "available_categories"),
                excluded_categories=_str_tuple_param(parameters, "excluded_categories"),
                available_codes=_int_tuple_param(parameters, "available_codes"),
                excluded_codes=_int_tuple_param(parameters, "excluded_codes"),
                catalog=self._profiles,
            )
            or self._default_profile
        )

    def _resolve_source(self, request: FetchRequest) -> LandCoverRasterSource:
        if self._source is not None:
            return self._source
        if self._path is not None and self._path.exists():
            return RasterioLandCoverSource(
                self._path,
                source_name=self._source_name,
                source_version=self._source_version,
                source_year=self._source_year,
            )
        if self._retriever is not None:
            box = request_bbox(request)
            if box is None:
                point = request_point(request)
                if point is None:
                    raise LandCoverError("retriever requires a bbox or point")
                loc_crs = (
                    request.location.crs
                    if request.location is not None and request.location.crs
                    else "EPSG:4326"
                )
                pad = 0.01
                box = BoundingBox(
                    min_lon=point[0] - pad,
                    min_lat=point[1] - pad,
                    max_lon=point[0] + pad,
                    max_lat=point[1] + pad,
                    crs=loc_crs,
                )
            import tempfile

            tmp = tempfile.TemporaryDirectory(prefix="solardss-worldcover-")
            self._temp_dirs.append(tmp)
            target = Path(tmp.name) / "worldcover.tif"
            written = self._retriever(box, target)
            return RasterioLandCoverSource(
                written,
                source_name=self._source_name,
                source_version=self._source_version,
                source_year=self._source_year,
            )
        raise LandCoverError("land-cover source is not configured")

    def _place(
        self, request: FetchRequest, source_crs: str
    ) -> tuple[BoundingBox, str | None, str, tuple[float, float] | None]:
        geometry_wkt = _text_param(request.parameters, "geometry_wkt")
        default_crs = "EPSG:4326"
        if request.bbox is not None:
            default_crs = request.bbox.crs
        elif request.location is not None and request.location.crs:
            default_crs = request.location.crs
        else:
            default_crs = source_crs
        geometry_crs = _text_param(request.parameters, "geometry_crs") or default_crs
        point = request_point(request)
        box = request_bbox(request)
        if box is not None:
            return box, geometry_wkt, geometry_crs, point
        if point is not None:
            location = request.location
            loc_crs = location.crs if location is not None and location.crs else "EPSG:4326"
            if loc_crs == "EPSG:4326" or crs_is_geographic(loc_crs):
                deg = 0.0002
                box = BoundingBox(
                    min_lon=point[0] - deg,
                    min_lat=point[1] - deg,
                    max_lon=point[0] + deg,
                    max_lat=point[1] + deg,
                    crs=loc_crs,
                )
            else:
                pad = 30.0
                box = BoundingBox(
                    min_lon=point[0] - pad,
                    min_lat=point[1] - pad,
                    max_lon=point[0] + pad,
                    max_lat=point[1] + pad,
                    crs=loc_crs,
                )
            return box, geometry_wkt, geometry_crs, point
        if geometry_wkt is not None:
            _, shapely_wkt, _ = _require_shapely()
            geom = shapely_wkt.loads(geometry_wkt)
            minx, miny, maxx, maxy = geom.bounds
            box = BoundingBox(
                min_lon=float(minx),
                min_lat=float(miny),
                max_lon=float(maxx),
                max_lat=float(maxy),
                crs=geometry_crs,
            )
            return box, geometry_wkt, geometry_crs, None
        raise LandCoverError("land-cover analysis requires a point, bbox, or geometry_wkt")

    def _mask(
        self,
        grid: LandCoverGrid,
        request_box: BoundingBox,
        geometry_wkt: str | None,
        geometry_crs: str,
        point_xy: tuple[float, float] | None,
    ) -> BoolArray:
        analysis_crs = grid.crs or request_box.crs
        if geometry_wkt:
            return mask_from_wkt(grid, geometry_wkt, geometry_crs=geometry_crs)
        if point_xy is not None:
            x, y = point_xy
            loc_crs = request_box.crs
            if grid.crs and loc_crs != grid.crs:
                pyproj = _require_pyproj()
                transformer = pyproj.Transformer.from_crs(loc_crs, grid.crs, always_xy=True)
                x, y = transformer.transform(x, y)
            return mask_from_point(grid, float(x), float(y))
        analysis_box = request_box
        if grid.crs and request_box.crs != grid.crs:
            analysis_box = transform_bbox(request_box, analysis_crs)
        return mask_from_bbox(grid, analysis_box)

    def _observations_from_analysis(
        self,
        request: FetchRequest,
        analysis: LandCoverAnalysisResult,
        window: LandCoverWindow,
        retrieved_at: datetime,
    ) -> tuple[IndicatorObservation, ...]:
        items: list[IndicatorObservation] = []
        produced = (
            IndicatorId.LAND_COVER.value,
            IndicatorId.LAND_AVAILABLE_AREA.value,
            *_FRACTIONS,
        )
        for indicator in produced:
            if (
                indicator == IndicatorId.LAND_AVAILABLE_AREA.value
                and analysis.summary.available_area_m2 is None
            ):
                item_request = request.model_copy(update={"indicator_id": indicator})
                items.append(
                    self._miss(
                        item_request,
                        ObservationStatus.NOT_CONFIGURED,
                        (
                            "LAND_AVAILABLE_AREA requires an explicit available-land "
                            "profile; non-water is not treated as available"
                        ),
                        retrieved_at,
                    )
                )
                continue
            item_request = request.model_copy(update={"indicator_id": indicator})
            items.append(
                self._observation_from_analysis(item_request, analysis, window, retrieved_at)
            )
        return tuple(items)

    def _observation_from_analysis(
        self,
        request: FetchRequest,
        analysis: LandCoverAnalysisResult,
        window: LandCoverWindow,
        retrieved_at: datetime,
    ) -> IndicatorObservation:
        summary = analysis.summary
        dataset = DatasetRef(
            name=window.source_name,
            media_type="image/tiff",
            uri=ESA_WORLDCOVER_CATALOGUE_URL,
            notes=(
                f"dataset_id={self._dataset_id}",
                f"source_version={window.source_version}",
                f"source_year={window.source_year}",
                f"native_crs={window.crs}",
                f"source_resolution={window.source_resolution}",
                f"area_method={summary.area_method}",
                f"legend={self._legend.dataset_id}",
            ),
        )
        spatial = SpatialCoverage(
            resolution=window.source_resolution,
            crs=window.crs,
            native_cell_size_m=(
                window.cell_size_x if window.crs and not crs_is_geographic(window.crs) else None
            ),
        )
        temporal = request.temporal or TemporalCoverage(
            resolution="annual",
            aggregation=str(window.source_year),
        )
        indicator = request.indicator_id
        value, unit, method, kind, extra = self._map_indicator(indicator, analysis)
        quality = list(summary.warnings)
        flags: tuple[str, ...]
        if (
            summary.nodata_cell_count
            or summary.unmapped_cell_count
            or summary.valid_cell_count == 0
        ):
            flags = (QualityFlag.UNKNOWN.value,)
        else:
            flags = (QualityFlag.NOMINAL.value,)
        if value is None:
            return self._miss(
                request,
                ObservationStatus.UNAVAILABLE,
                "no valid land-cover samples in the analysis mask",
                retrieved_at,
                dataset=dataset,
                spatial=spatial,
            )
        share_notes = tuple(
            f"{share.key}: area_m2={share.area_m2:.6f} percent={share.percent:.6f} "
            f"cells={share.cell_count}"
            for share in summary.category_shares
        )
        assumptions = tuple(
            [
                *extra,
                *quality,
                *share_notes,
                "classes are windowed with nearest-neighbour sampling; no bilinear resampling",
                f"land-cover source={window.source_name}",
                f"source_year={window.source_year}",
                f"source_version={window.source_version}",
                f"processing={method}",
                f"area_method={summary.area_method}",
                f"valid_cells={summary.valid_cell_count}",
                f"nodata_cells={summary.nodata_cell_count}",
                f"dominant_class={summary.dominant_category}",
                f"dominant_code={summary.dominant_code}",
                f"dominant_name={summary.dominant_name}",
            ]
        )
        origin = ValueOrigin.DERIVED
        return IndicatorObservation(
            indicator_id=indicator,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=self._source_name,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=kind,
                value_origin=origin,
                source_version=f"{window.source_version}-{window.source_year}",
                calculation_method=method,
                dataset=dataset,
            ),
            spatial=spatial,
            temporal=temporal,
            value=value,
            unit=unit,
            quality_flags=flags,
            assumptions=assumptions,
        )

    def _map_indicator(
        self, indicator: str, analysis: LandCoverAnalysisResult
    ) -> tuple[float | str | None, str, str, ValueProvenanceKind, tuple[str, ...]]:
        summary = analysis.summary
        extra: tuple[str, ...]
        if indicator == IndicatorId.LAND_COVER.value:
            if summary.dominant_category is None:
                return (
                    None,
                    "categorical",
                    CALC_DOMINANT_CLASS,
                    ValueProvenanceKind.MEASURED_OR_PROVIDER,
                    (),
                )
            extra = (
                f"dominant_category={summary.dominant_category}",
                f"dominant_code={summary.dominant_code}",
                f"dominant_name={summary.dominant_name}",
            )
            return (
                summary.dominant_category,
                "categorical",
                CALC_DOMINANT_CLASS,
                ValueProvenanceKind.MEASURED_OR_PROVIDER,
                extra,
            )
        if indicator == IndicatorId.LAND_AVAILABLE_AREA.value:
            extra = (
                f"profile_id={summary.profile_id}",
                f"available_fraction={summary.available_fraction}",
                f"available_cells={summary.available_cell_count}",
                "non-water is not automatically available",
            )
            return (
                summary.available_area_m2,
                "m2",
                CALC_AVAILABLE_AREA,
                ValueProvenanceKind.DERIVED_METHOD,
                extra,
            )
        category = FRACTION_INDICATORS.get(indicator)
        if category is None:
            raise InvalidParameterError(f"land-cover provider does not map indicator {indicator}")
        extra = (f"category={category}", f"valid_area_m2={summary.valid_area_m2}")
        return (
            category_fraction(summary, category),
            "1",
            CALC_CLASS_FRACTION,
            ValueProvenanceKind.DERIVED_METHOD,
            extra,
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
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=status,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=self._source_name,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=ValueProvenanceKind.MEASURED_OR_PROVIDER,
                value_origin=ValueOrigin.RAW,
                source_version=f"{self._source_version}-{self._source_year}",
                dataset=dataset
                or DatasetRef(
                    name=self._source_name,
                    media_type="image/tiff",
                    uri=ESA_WORLDCOVER_CATALOGUE_URL,
                ),
            ),
            spatial=spatial
            or SpatialCoverage(resolution="land-cover", crs=ESA_WORLDCOVER_NATIVE_CRS),
            temporal=request.temporal
            or TemporalCoverage(resolution="annual", aggregation=str(self._source_year)),
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


class ESAWorldCoverProvider(LandCoverRasterProvider):
    """ESA WorldCover 10 m (v200 / 2021 defaults) land-cover provider."""

    provider_id_default = "esa.worldcover"


WorldCoverLandCoverProvider = ESAWorldCoverProvider


__all__ = [
    "ESA_WORLDCOVER_CATALOGUE_URL",
    "ESA_WORLDCOVER_ID",
    "ESA_WORLDCOVER_LEGEND",
    "ESA_WORLDCOVER_NATIVE_CRS",
    "ESA_WORLDCOVER_RESOLUTION",
    "ESA_WORLDCOVER_SOURCE_NAME",
    "ESA_WORLDCOVER_VERSION",
    "ESA_WORLDCOVER_YEAR",
    "WORLDCOVER_BARE_SPARSE_ONLY",
    "WORLDCOVER_OPEN_GROUND",
    "WORLDCOVER_PATH_ENV",
    "WORLDCOVER_PROFILES",
    "ESAWorldCoverProvider",
    "LandCoverAnalysisBundle",
    "LandCoverRasterProvider",
    "LandCoverRasterSource",
    "LandCoverRetriever",
    "LandCoverWindow",
    "MemoryLandCoverSource",
    "RasterioLandCoverSource",
    "WorldCoverLandCoverProvider",
    "mask_from_bbox",
    "mask_from_point",
    "mask_from_wkt",
    "transform_bbox",
    "window_to_grid",
]
