"""Copernicus-compatible DEM provider. Retrieval is injected, not hardcoded.

Scientific slope/TRI live in ``solardss.terrain.analysis``. This module windows
a raster, reprojects geographic DEMs to a local UTM CRS, and maps zonal
statistics onto terrain indicators. Copernicus GLO-30 metadata is the default
catalogue description; any north-up DEM with linear or geographic CRS can be
supplied through ``DEMRasterSource``.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from solardss.core.arrays import BoolArray, FloatArray
from solardss.core.jsonutil import JsonValue
from solardss.data.contracts import DatasetRef
from solardss.data.geo import BoundingBox
from solardss.data.gis import gis_module_available
from solardss.exceptions import InvalidGridError, InvalidParameterError, MissingExtraError
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.indicators.ids import IndicatorId
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
from solardss.providers.errors import CacheError, DEMCoverageError, DEMError
from solardss.providers.region import request_bbox, request_point
from solardss.providers.request import FetchRequest, request_cache_key
from solardss.terrain.analysis import (
    CALC_ASPECT_CIRCULAR,
    CALC_ELEVATION_ZONAL,
    CALC_HORN,
    CALC_RILEY_TRI,
    TerrainAnalysisResult,
    ZonalStats,
    analyze_terrain,
    crs_is_geographic,
    utm_epsg,
)
from solardss.terrain.contracts import ElevationGrid

COPERNICUS_DEM_GLO30_ID: Final = "COP-DEM_GLO-30-DGED"
COPERNICUS_DEM_SOURCE_NAME: Final = "Copernicus DEM GLO-30"
COPERNICUS_DEM_RESOLUTION_M: Final = 30.0
COPERNICUS_DEM_NATIVE_CRS: Final = "EPSG:4326"
COPERNICUS_DEM_CATALOGUE_URL: Final = (
    "https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model"
)
DEM_PATH_ENV: Final = "SOLARDSS_DEM_PATH"
_CACHE_SCHEMA: Final = "solardss.dem_obs.v1"
_PAD_CELLS: Final = 2

_SERVED: Final = frozenset(
    {
        IndicatorId.TERRAIN_ELEVATION.value,
        IndicatorId.TERRAIN_SLOPE_MEAN.value,
        IndicatorId.TERRAIN_SLOPE_MAX.value,
        IndicatorId.TERRAIN_ASPECT.value,
        IndicatorId.TERRAIN_RUGGEDNESS.value,
        IndicatorId.TERRAIN_HORIZON_OBSTRUCTION.value,
        IndicatorId.TERRAIN_SHADING_POTENTIAL.value,
    }
)
_COMPUTED: Final = (
    IndicatorId.TERRAIN_ELEVATION.value,
    IndicatorId.TERRAIN_SLOPE_MEAN.value,
    IndicatorId.TERRAIN_SLOPE_MAX.value,
    IndicatorId.TERRAIN_ASPECT.value,
    IndicatorId.TERRAIN_RUGGEDNESS.value,
)
_COMPUTED_SET: Final = frozenset(_COMPUTED)

DEMRetriever = Callable[[BoundingBox, Path], Path]


@dataclass(frozen=True, slots=True)
class DEMWindow:
    """North-up raster window. ``cell_size_y`` is the positive southward spacing."""

    z: FloatArray
    crs: str
    origin_x: float
    origin_y: float
    cell_size_x: float
    cell_size_y: float
    nodata: float | None = None
    source_name: str = COPERNICUS_DEM_SOURCE_NAME
    source_resolution: str = "30m"


class DEMRasterSource(Protocol):
    """Read a spatial window. Implementations may use files, objects, or a CDN."""

    @property
    def crs(self) -> str:
        """Native CRS label of the DEM."""

    def coverage_bbox(self) -> BoundingBox | None:
        """Dataset extent in the native CRS, if known."""

    def read_window(
        self,
        bbox: BoundingBox,
        *,
        pad_cells: int = _PAD_CELLS,
    ) -> DEMWindow:
        """Return elevation cells covering ``bbox`` plus a Horn/TRI neighborhood."""


class RasterioDEMSource:
    """GeoTIFF/COG reader. rasterio is imported only when a window is read."""

    def __init__(
        self,
        path: Path | str,
        *,
        source_name: str = COPERNICUS_DEM_SOURCE_NAME,
        source_resolution: str = "30m",
    ) -> None:
        self._path = Path(path)
        self._source_name = source_name
        self._source_resolution = source_resolution
        self._crs = COPERNICUS_DEM_NATIVE_CRS

    @property
    def crs(self) -> str:
        return self._crs

    def coverage_bbox(self) -> BoundingBox | None:
        rasterio = _require_rasterio()
        with rasterio.open(self._path) as dataset:
            self._crs = str(dataset.crs) if dataset.crs else COPERNICUS_DEM_NATIVE_CRS
            bounds = dataset.bounds
            return BoundingBox(
                min_lon=float(bounds.left),
                min_lat=float(bounds.bottom),
                max_lon=float(bounds.right),
                max_lat=float(bounds.top),
                crs=self._crs,
            )

    def read_window(self, bbox: BoundingBox, *, pad_cells: int = _PAD_CELLS) -> DEMWindow:
        rasterio = _require_rasterio()
        from rasterio.errors import RasterioError
        from rasterio.windows import Window, from_bounds

        try:
            with rasterio.open(self._path) as dataset:
                self._crs = str(dataset.crs) if dataset.crs else COPERNICUS_DEM_NATIVE_CRS
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
                    raise DEMCoverageError("requested window does not intersect the DEM")
                data = dataset.read(1, window=window, boundless=True, masked=True)
                window_transform = dataset.window_transform(window)
                nodata = dataset.nodata
        except RasterioError as exc:
            raise DEMError(f"failed to read DEM window: {exc}") from exc
        masked = np.ma.asarray(data)  # type: ignore[no-untyped-call]
        z = np.asarray(
            np.ma.filled(masked.astype(np.float64), np.nan),  # type: ignore[no-untyped-call]
            dtype=np.float64,
        )
        if nodata is not None:
            z = np.where(z == float(nodata), np.nan, z)
        return DEMWindow(
            z=np.asarray(z, dtype=np.float64),
            crs=self._crs,
            origin_x=float(window_transform.c),
            origin_y=float(window_transform.f),
            cell_size_x=abs(float(window_transform.a)),
            cell_size_y=abs(float(window_transform.e)),
            nodata=None,
            source_name=self._source_name,
            source_resolution=self._source_resolution,
        )


@dataclass(frozen=True, slots=True)
class DEMAnalysisBundle:
    """Normalized indicators plus zonal statistics and quality."""

    analysis: TerrainAnalysisResult | None
    observations: tuple[IndicatorObservation, ...]
    source_crs: str | None
    analysis_crs: str | None
    quality_warnings: tuple[str, ...]


def _require_rasterio() -> Any:
    try:
        import rasterio
    except ImportError as exc:
        msg = "Optional GIS extra is not installed. Install with: pip install 'solardss[gis]'."
        raise MissingExtraError(msg) from exc
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


def window_to_grid(window: DEMWindow) -> ElevationGrid:
    """Build an ``ElevationGrid`` without changing CRS."""

    return ElevationGrid(
        z=window.z,
        cell_size_x=window.cell_size_x,
        cell_size_y=window.cell_size_y,
        nodata=window.nodata,
        crs=window.crs,
        origin_x=window.origin_x,
        origin_y=window.origin_y,
    )


def project_window_for_analysis(window: DEMWindow, *, dst_crs: str | None = None) -> ElevationGrid:
    """Return a linearly spaced grid. Geographic windows are warped to UTM."""

    if not crs_is_geographic(window.crs):
        return window_to_grid(window)
    rasterio = _require_rasterio()
    from rasterio.crs import CRS
    from rasterio.errors import RasterioError
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    rows, cols = window.z.shape
    try:
        src_crs = CRS.from_user_input(window.crs)
        src_transform = rasterio.Affine(
            window.cell_size_x,
            0.0,
            window.origin_x,
            0.0,
            -window.cell_size_y,
            window.origin_y,
        )
        lon = window.origin_x + 0.5 * cols * window.cell_size_x
        lat = window.origin_y - 0.5 * rows * window.cell_size_y
        target = dst_crs or utm_epsg(lon, lat)
        dst_crs_obj = CRS.from_user_input(target)
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src_crs,
            dst_crs_obj,
            cols,
            rows,
            window.origin_x,
            window.origin_y - rows * window.cell_size_y,
            window.origin_x + cols * window.cell_size_x,
            window.origin_y,
        )
        destination = np.full((dst_height, dst_width), np.nan, dtype=np.float64)
        reproject(
            source=window.z,
            destination=destination,
            src_transform=src_transform,
            src_crs=src_crs,
            dst_transform=dst_transform,
            dst_crs=dst_crs_obj,
            src_nodata=np.nan if window.nodata is None else window.nodata,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    except RasterioError as exc:
        raise DEMError(f"failed to reproject DEM window: {exc}") from exc
    return ElevationGrid(
        z=destination,
        cell_size_x=abs(float(dst_transform.a)),
        cell_size_y=abs(float(dst_transform.e)),
        nodata=None,
        crs=target,
        origin_x=float(dst_transform.c),
        origin_y=float(dst_transform.f),
    )


def _cell_centers(grid: ElevationGrid) -> tuple[FloatArray, FloatArray]:
    rows, cols = np.asarray(grid.z).shape
    origin_x = 0.0 if grid.origin_x is None else float(grid.origin_x)
    origin_y = 0.0 if grid.origin_y is None else float(grid.origin_y)
    xs = origin_x + (np.arange(cols, dtype=np.float64) + 0.5) * float(grid.cell_size_x)
    ys = origin_y - (np.arange(rows, dtype=np.float64) + 0.5) * float(grid.cell_size_y)
    xx, yy = np.meshgrid(xs, ys)
    return np.asarray(xx, dtype=np.float64), np.asarray(yy, dtype=np.float64)


def mask_from_bbox(grid: ElevationGrid, bbox: BoundingBox) -> BoolArray:
    xx, yy = _cell_centers(grid)
    return np.asarray(
        (xx >= bbox.min_lon) & (xx <= bbox.max_lon) & (yy >= bbox.min_lat) & (yy <= bbox.max_lat),
        dtype=bool,
    )


def mask_from_point(grid: ElevationGrid, x: float, y: float) -> BoolArray:
    xx, yy = _cell_centers(grid)
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    mask = np.zeros(xx.shape, dtype=bool)
    index = np.unravel_index(int(np.argmin(dist2)), dist2.shape)
    mask[index] = True
    return mask


def mask_from_wkt(grid: ElevationGrid, wkt: str, *, geometry_crs: str) -> BoolArray:
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


class DEMTerrainProvider:
    """Window a DEM, project if needed, and map Horn/TRI statistics to indicators.

    Download is optional and injected as ``retriever``. There is no built-in
    Copernicus STAC/OpenSearch client.
    """

    provider_id_default = "copernicus.dem"
    source_name_default = COPERNICUS_DEM_SOURCE_NAME
    served = _SERVED

    def __init__(
        self,
        *,
        source: DEMRasterSource | None = None,
        dem_path: Path | str | None = None,
        retriever: DEMRetriever | None = None,
        dataset_id: str = COPERNICUS_DEM_GLO30_ID,
        source_name: str | None = None,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        provider_id: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        path = dem_path
        if path is None:
            env = environ if environ is not None else None
            raw = None if env is None else env.get(DEM_PATH_ENV)
            if raw is None and environ is None:
                import os

                raw = os.environ.get(DEM_PATH_ENV)
            path = raw
        self._source = source
        self._path = Path(path) if path else None
        self._retriever = retriever
        self._dataset_id = dataset_id
        self._source_name = source_name or self.source_name_default
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
            "source_url": COPERNICUS_DEM_CATALOGUE_URL,
            "has_local_dem": self._path is not None or self._source is not None,
            "rasterio": gis_module_available("rasterio"),
            "pyproj": gis_module_available("pyproj"),
        }
        if not self._configured():
            return ProviderHealth(
                status=HealthStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                reason=(
                    "DEM is not configured: pass a DEMRasterSource, SOLARDSS_DEM_PATH, "
                    "or an injected retriever"
                ),
                checked_at=checked_at,
                details=details,
            )
        if self._path is not None and not self._path.exists() and self._source is None:
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                reason="DEM path does not exist",
                checked_at=checked_at,
                details=details,
            )
        return ProviderHealth(
            status=HealthStatus.OK,
            provider_id=self._provider_id,
            reason="DEM provider is configured",
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
                reason="indicator is not served by the DEM provider",
            )
        if not self._configured():
            return ProviderAvailability(
                status=AvailabilityStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="DEM is not configured",
            )
        if request.indicator_id not in _COMPUTED_SET:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=(
                    "horizon obstruction and shading potential are not computed; "
                    "slope is not a substitute"
                ),
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
                reason="DEM analysis requires a point, bbox, or geometry_wkt",
            )
        return ProviderAvailability(
            status=AvailabilityStatus.AVAILABLE,
            provider_id=self._provider_id,
            retryable=False,
            indicator_id=request.indicator_id,
            reason="DEM can be windowed for this request",
        )

    def coverage(self, request: FetchRequest) -> CoverageReport:
        availability = self.availability(request)
        spatial = SpatialCoverage(
            resolution="30m" if self._dataset_id == COPERNICUS_DEM_GLO30_ID else "dem",
            crs=COPERNICUS_DEM_NATIVE_CRS,
            bbox=request_bbox(request),
        )
        return CoverageReport(
            status=availability.status,
            provider_id=self._provider_id,
            spatial=spatial,
            temporal=request.temporal,
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
        bundle = self.fetch_terrain(request)
        for item in bundle.observations:
            if item.status is ObservationStatus.OBSERVED:
                self._write_cache(item.provenance.request_hash, item)
            if item.indicator_id == request.indicator_id:
                return item
        miss = self._miss(
            request,
            ObservationStatus.UNAVAILABLE,
            "indicator was not produced from the DEM window",
            self._clock.now(),
        )
        return miss

    def fetch_terrain(self, request: FetchRequest) -> DEMAnalysisBundle:
        availability = self.availability(request)
        retrieved_at = self._clock.now()
        if availability.status is not AvailabilityStatus.AVAILABLE:
            miss = self._miss(
                request,
                ObservationStatus(availability.status.value),
                availability.reason,
                retrieved_at,
            )
            return DEMAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                analysis_crs=None,
                quality_warnings=(availability.reason,),
            )
        try:
            source = self._resolve_source(request)
            request_box, geometry_wkt, geometry_crs, point_xy = self._place(request, source.crs)
            native_box = transform_bbox(request_box, source.crs)
            window = source.read_window(native_box)
            grid = project_window_for_analysis(window)
            mask = self._mask(grid, request_box, geometry_wkt, geometry_crs, point_xy)
            analysis = analyze_terrain(grid, mask=mask)
        except MissingExtraError as exc:
            miss = self._miss(request, ObservationStatus.NOT_CONFIGURED, exc.message, retrieved_at)
            return DEMAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                analysis_crs=None,
                quality_warnings=(exc.message,),
            )
        except DEMCoverageError as exc:
            miss = self._miss(
                request, ObservationStatus.MISSING_COVERAGE, exc.message, retrieved_at
            )
            return DEMAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                analysis_crs=None,
                quality_warnings=(exc.message,),
            )
        except (DEMError, InvalidGridError, InvalidParameterError, ValueError, OSError) as exc:
            reason = getattr(exc, "message", str(exc))
            miss = self._miss(
                request, ObservationStatus.TEMPORARY_FAILURE, str(reason), retrieved_at
            )
            return DEMAnalysisBundle(
                analysis=None,
                observations=(miss,),
                source_crs=None,
                analysis_crs=None,
                quality_warnings=(str(reason),),
            )
        observations = self._observations_from_analysis(request, analysis, window, retrieved_at)
        return DEMAnalysisBundle(
            analysis=analysis,
            observations=observations,
            source_crs=window.crs,
            analysis_crs=analysis.analysis_crs,
            quality_warnings=analysis.quality.warnings,
        )

    def _resolve_source(self, request: FetchRequest) -> DEMRasterSource:
        if self._source is not None:
            return self._source
        if self._path is not None and self._path.exists():
            return RasterioDEMSource(self._path, source_name=self._source_name)
        if self._retriever is not None:
            box = request_bbox(request)
            if box is None:
                point = request_point(request)
                if point is None:
                    raise DEMError("retriever requires a bbox or point")
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

            tmp = tempfile.TemporaryDirectory(prefix="solardss-dem-")
            self._temp_dirs.append(tmp)
            target = Path(tmp.name) / "dem.tif"
            written = self._retriever(box, target)
            return RasterioDEMSource(written, source_name=self._source_name)
        raise DEMError("DEM source is not configured")

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
            pad = COPERNICUS_DEM_RESOLUTION_M * (_PAD_CELLS + 2)
            if loc_crs == "EPSG:4326" or crs_is_geographic(loc_crs):
                deg = 0.002
                box = BoundingBox(
                    min_lon=point[0] - deg,
                    min_lat=point[1] - deg,
                    max_lon=point[0] + deg,
                    max_lat=point[1] + deg,
                    crs=loc_crs,
                )
            else:
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
        raise DEMError("DEM analysis requires a point, bbox, or geometry_wkt")

    def _mask(
        self,
        grid: ElevationGrid,
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
        analysis: TerrainAnalysisResult,
        window: DEMWindow,
        retrieved_at: datetime,
    ) -> tuple[IndicatorObservation, ...]:
        items: list[IndicatorObservation] = []
        for indicator in _COMPUTED:
            item_request = request.model_copy(update={"indicator_id": indicator})
            items.append(
                self._observation_from_analysis(item_request, analysis, window, retrieved_at)
            )
        return tuple(items)

    def _observation_from_analysis(
        self,
        request: FetchRequest,
        analysis: TerrainAnalysisResult,
        window: DEMWindow,
        retrieved_at: datetime,
    ) -> IndicatorObservation:
        dataset = DatasetRef(
            name=window.source_name,
            media_type="image/tiff",
            uri=COPERNICUS_DEM_CATALOGUE_URL,
            notes=(
                f"dataset_id={self._dataset_id}",
                f"native_crs={window.crs}",
                f"analysis_crs={analysis.analysis_crs}",
                f"cell_size_x={analysis.cell_size_x}",
                f"cell_size_y={analysis.cell_size_y}",
                f"source_resolution={window.source_resolution}",
            ),
        )
        spatial = SpatialCoverage(
            resolution=window.source_resolution,
            crs=analysis.analysis_crs,
            native_cell_size_m=(
                analysis.cell_size_x
                if analysis.analysis_crs and not crs_is_geographic(analysis.analysis_crs)
                else None
            ),
        )
        temporal = request.temporal or TemporalCoverage(resolution="static-dem")
        indicator = request.indicator_id
        zonal, value, unit, method, kind, extra = self._map_indicator(indicator, analysis)
        quality = list(analysis.quality.warnings)
        flags: tuple[str, ...]
        if zonal.nodata_count > 0 or analysis.quality.nodata_fraction > 0.0:
            flags = (QualityFlag.UNKNOWN.value,)
            quality.append(f"nodata_fraction={analysis.quality.nodata_fraction:.6f}")
            quality.append(f"zonal_nodata_count={zonal.nodata_count}")
        else:
            flags = (QualityFlag.NOMINAL.value,)
        if value is None:
            return self._miss(
                request,
                ObservationStatus.UNAVAILABLE,
                "no finite DEM samples in the analysis mask",
                retrieved_at,
                dataset=dataset,
                spatial=spatial,
            )
        assumptions = tuple(
            [
                *extra,
                *quality,
                f"DEM source={window.source_name}",
                f"processing={method}",
                f"statistic={zonal.statistic_used}",
                f"finite_count={zonal.finite_count}",
                f"nodata_count={zonal.nodata_count}",
            ]
        )
        return IndicatorObservation(
            indicator_id=indicator,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=self._source_name,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=kind,
                value_origin=ValueOrigin.DERIVED,
                source_version=self._dataset_id,
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
        self, indicator: str, analysis: TerrainAnalysisResult
    ) -> tuple[ZonalStats, float | None, str, str, ValueProvenanceKind, tuple[str, ...]]:
        if indicator == IndicatorId.TERRAIN_ELEVATION.value:
            zonal = analysis.elevation_zonal
            extra: tuple[str, ...] = (
                f"mean={zonal.mean}",
                f"median={zonal.median}",
                f"max={zonal.max}",
                f"p10={zonal.p10}",
                f"p90={zonal.p90}",
            )
            return (
                zonal,
                zonal.mean,
                "m",
                CALC_ELEVATION_ZONAL,
                ValueProvenanceKind.MEASURED_OR_PROVIDER,
                extra,
            )
        if indicator == IndicatorId.TERRAIN_SLOPE_MEAN.value:
            zonal = analysis.slope_zonal
            extra = (
                f"mean={zonal.mean}",
                f"median={zonal.median}",
                f"max={zonal.max}",
                f"p10={zonal.p10}",
                f"p90={zonal.p90}",
            )
            return (
                zonal,
                zonal.mean,
                "deg",
                CALC_HORN,
                ValueProvenanceKind.DERIVED_METHOD,
                extra,
            )
        if indicator == IndicatorId.TERRAIN_SLOPE_MAX.value:
            zonal = analysis.slope_zonal
            extra = (f"max={zonal.max}", f"mean={zonal.mean}", f"p90={zonal.p90}")
            return (
                zonal,
                zonal.max,
                "deg",
                CALC_HORN,
                ValueProvenanceKind.DERIVED_METHOD,
                extra,
            )
        if indicator == IndicatorId.TERRAIN_ASPECT.value:
            zonal = analysis.aspect_zonal
            extra = (
                f"circular_mean={zonal.circular_mean}",
                "arithmetic mean of aspect degrees is not used",
                f"median={zonal.median}",
            )
            return (
                zonal,
                zonal.circular_mean,
                "deg",
                CALC_ASPECT_CIRCULAR,
                ValueProvenanceKind.DERIVED_METHOD,
                extra,
            )
        if indicator == IndicatorId.TERRAIN_RUGGEDNESS.value:
            zonal = analysis.ruggedness_zonal
            extra = (f"mean={zonal.mean}", f"max={zonal.max}", f"median={zonal.median}")
            return (
                zonal,
                zonal.mean,
                "m",
                CALC_RILEY_TRI,
                ValueProvenanceKind.DERIVED_METHOD,
                extra,
            )
        raise InvalidParameterError(f"DEM provider does not map indicator {indicator}")

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
                source_version=self._dataset_id,
                dataset=dataset
                or DatasetRef(
                    name=self._source_name,
                    media_type="image/tiff",
                    uri=COPERNICUS_DEM_CATALOGUE_URL,
                ),
            ),
            spatial=spatial or SpatialCoverage(resolution="dem", crs=COPERNICUS_DEM_NATIVE_CRS),
            temporal=request.temporal or TemporalCoverage(resolution="static-dem"),
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


CopernicusDEMProvider = DEMTerrainProvider


@dataclass(frozen=True, slots=True)
class MemoryDEMSource:
    """In-memory north-up DEM used by tests and injected fixtures."""

    """In-memory north-up DEM used by tests and injected fixtures."""

    elevation: FloatArray
    coordinate_system: str
    origin_x: float
    origin_y: float
    cell_size_x: float
    cell_size_y: float
    nodata: float | None = None
    source_name: str = "array DEM"
    source_resolution: str = "synthetic"

    @property
    def crs(self) -> str:
        return self.coordinate_system

    def coverage_bbox(self) -> BoundingBox | None:
        rows, cols = np.asarray(self.elevation).shape
        return BoundingBox(
            min_lon=self.origin_x,
            min_lat=self.origin_y - rows * self.cell_size_y,
            max_lon=self.origin_x + cols * self.cell_size_x,
            max_lat=self.origin_y,
            crs=self.coordinate_system,
        )

    def read_window(self, bbox: BoundingBox, *, pad_cells: int = _PAD_CELLS) -> DEMWindow:
        z = np.asarray(self.elevation, dtype=np.float64)
        rows, cols = z.shape
        col0 = int(np.floor((bbox.min_lon - self.origin_x) / self.cell_size_x)) - pad_cells
        col1 = int(np.ceil((bbox.max_lon - self.origin_x) / self.cell_size_x)) + pad_cells
        row0 = int(np.floor((self.origin_y - bbox.max_lat) / self.cell_size_y)) - pad_cells
        row1 = int(np.ceil((self.origin_y - bbox.min_lat) / self.cell_size_y)) + pad_cells
        row0 = max(row0, 0)
        col0 = max(col0, 0)
        row1 = min(row1, rows)
        col1 = min(col1, cols)
        if row1 <= row0 or col1 <= col0:
            raise DEMCoverageError("requested window does not intersect the in-memory DEM")
        subset = z[row0:row1, col0:col1]
        return DEMWindow(
            z=np.asarray(subset, dtype=np.float64),
            crs=self.coordinate_system,
            origin_x=self.origin_x + col0 * self.cell_size_x,
            origin_y=self.origin_y - row0 * self.cell_size_y,
            cell_size_x=self.cell_size_x,
            cell_size_y=self.cell_size_y,
            nodata=self.nodata,
            source_name=self.source_name,
            source_resolution=self.source_resolution,
        )


__all__ = [
    "COPERNICUS_DEM_CATALOGUE_URL",
    "COPERNICUS_DEM_GLO30_ID",
    "COPERNICUS_DEM_NATIVE_CRS",
    "COPERNICUS_DEM_RESOLUTION_M",
    "COPERNICUS_DEM_SOURCE_NAME",
    "DEM_PATH_ENV",
    "CopernicusDEMProvider",
    "DEMAnalysisBundle",
    "DEMRasterSource",
    "DEMRetriever",
    "DEMTerrainProvider",
    "DEMWindow",
    "MemoryDEMSource",
    "RasterioDEMSource",
    "mask_from_bbox",
    "mask_from_point",
    "mask_from_wkt",
    "project_window_for_analysis",
    "transform_bbox",
    "window_to_grid",
]
