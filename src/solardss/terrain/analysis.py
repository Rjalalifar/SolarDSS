"""Terrain derivatives from a regular elevation grid.

Slope and aspect use Horn (1981). Ruggedness uses Riley et al. (1999) TRI.
These methods require **linear** cell sizes in the same unit as elevation
(typically metres). Geographic degree spacings are rejected; reproject first.

This module does not open rasters, download Copernicus DEM tiles, or choose a
retrieval API. Windowing and CRS conversion belong in a DEM provider.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from solardss.core.arrays import BoolArray, FloatArray
from solardss.core.hashing import fingerprint_array
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, NumericResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.exceptions import InvalidGridError, UnsupportedMethodError
from solardss.provenance.record import provenance_record
from solardss.terrain.contracts import ElevationGrid
from solardss.terrain.horn import SlopeAspectResult, horn_slope_aspect

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

CALC_HORN = "horn_1981_slope_aspect"
CALC_RILEY_TRI = "riley_1999_terrain_ruggedness_index"
CALC_ELEVATION_ZONAL = "dem_zonal_elevation_statistics"
CALC_ASPECT_CIRCULAR = "circular_mean_aspect_from_north_clockwise"


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


def utm_epsg(longitude: float, latitude: float) -> str:
    """UTM zone EPSG from longitude/latitude using the standard 6° formula.

    Norway/Svalbard exceptions are not applied. The result is a projected CRS
    in metres, suitable for Horn/TRI on a local window.
    """

    zone = int((longitude + 180.0) // 6.0) + 1
    zone = min(max(zone, 1), 60)
    base = 32600 if latitude >= 0.0 else 32700
    return f"EPSG:{base + zone}"


def require_projected_grid(grid: ElevationGrid) -> ElevationGrid:
    """Reject geographic CRS so slope is never computed from degree distances."""

    if crs_is_geographic(grid.crs):
        raise InvalidGridError(
            "slope/aspect/TRI require linear cell sizes in the same unit as "
            "elevation; reproject this geographic DEM before analysis"
        )
    return grid


def circular_mean_deg(values: FloatArray) -> float | None:
    """Circular mean of compass bearings in degrees clockwise from north."""

    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    radians = np.deg2rad(finite)
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    if mean_sin == 0.0 and mean_cos == 0.0:
        return None
    return float(np.mod(np.rad2deg(np.arctan2(mean_sin, mean_cos)), 360.0))


class ZonalStats(BaseModel):
    """Finite-aware zonal summary. Missing samples stay null, never filled."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int
    finite_count: int
    nodata_count: int
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    median: float | None = None
    p10: float | None = None
    p90: float | None = None
    circular_mean: float | None = None
    statistic_used: str = Field(min_length=1)


class TerrainQuality(BaseModel):
    """Data-quality counters. Incomplete coverage is reported, not repaired."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    valid_fraction: float
    nodata_fraction: float
    edge_cells_excluded: bool
    finite_elevation_count: int
    finite_slope_count: int
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class TerrainAnalysisResult:
    """Full-grid derivatives plus zonal statistics for the analysis mask."""

    elevation: NumericResult
    slope_aspect: SlopeAspectResult
    ruggedness_m: NumericResult
    elevation_zonal: ZonalStats
    slope_zonal: ZonalStats
    aspect_zonal: ZonalStats
    ruggedness_zonal: ZonalStats
    quality: TerrainQuality
    analysis_crs: str | None
    cell_size_x: float
    cell_size_y: float


def riley_tri(grid: ElevationGrid) -> NumericResult:
    """Riley et al. (1999) terrain ruggedness index in elevation units (m).

    ``TRI = sqrt(sum_{i=1..8} (z_i - z_0)^2)``. The 1-cell border and any
    window containing nodata are NaN. Not interpolated.
    """

    grid = require_projected_grid(grid).validated()
    z = _prepare_elevation(grid)
    tri = np.full(z.shape, np.nan, dtype=np.float64)
    window = (
        z[0:-2, 0:-2],
        z[0:-2, 1:-1],
        z[0:-2, 2:],
        z[1:-1, 0:-2],
        z[1:-1, 1:-1],
        z[1:-1, 2:],
        z[2:, 0:-2],
        z[2:, 1:-1],
        z[2:, 2:],
    )
    finite = np.ones(window[0].shape, dtype=bool)
    for cell in window:
        finite &= np.isfinite(cell)
    center = window[4]
    acc = np.zeros(center.shape, dtype=np.float64)
    for index, cell in enumerate(window):
        if index == 4:
            continue
        delta = cell - center
        acc = acc + delta * delta
    interior = np.sqrt(acc)
    tri[1:-1, 1:-1] = np.where(finite, interior, np.nan)
    parameters = {
        "cell_size_x": float(grid.cell_size_x),
        "cell_size_y": float(grid.cell_size_y),
        "nodata": grid.nodata,
        "crs": grid.crs,
        "formula": "sqrt(sum_i (z_i - z_0)^2) over 8 neighbors",
        "reference": "Riley, DeGloria, Elliot 1999",
    }
    layer = LayerResult(
        name="tri_m",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.RILEY_TRI.value,
        provenance=provenance_record(
            MethodId.RILEY_TRI,
            parameters=parameters,
            input_fingerprints=(fingerprint_array(z),),
        ),
        summary=summarize_array(tri),
    )
    return NumericResult(values=np.asarray(tri, dtype=np.float64), layer=layer)


def zonal_statistics(
    values: FloatArray,
    mask: BoolArray | None = None,
    *,
    circular: bool = False,
    statistic_used: str = "mean",
) -> ZonalStats:
    """Summarize ``values`` where ``mask`` is true. Nodata stays nodata."""

    data = np.asarray(values, dtype=np.float64)
    if mask is None:
        selected = data.reshape(-1)
    else:
        if mask.shape != data.shape:
            raise InvalidGridError("terrain mask must match the elevation grid shape")
        selected = data[np.asarray(mask, dtype=bool)]
    count = int(selected.size)
    finite = selected[np.isfinite(selected)]
    finite_count = int(finite.size)
    nodata_count = count - finite_count
    if finite_count == 0:
        return ZonalStats(
            count=count,
            finite_count=0,
            nodata_count=nodata_count,
            circular_mean=None,
            statistic_used=statistic_used,
        )
    p10, median, p90 = (float(item) for item in np.percentile(finite, [10.0, 50.0, 90.0]))
    return ZonalStats(
        count=count,
        finite_count=finite_count,
        nodata_count=nodata_count,
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
        median=median,
        p10=p10,
        p90=p90,
        circular_mean=circular_mean_deg(finite) if circular else None,
        statistic_used=statistic_used,
    )


def analyze_terrain(
    grid: ElevationGrid,
    *,
    mask: BoolArray | None = None,
) -> TerrainAnalysisResult:
    """Horn slope/aspect, Riley TRI, and zonal statistics on ``mask``.

    Derivatives are computed on the full grid (neighbors outside the mask are
    used). The mask only selects cells that enter zonal statistics. The outer
    one-cell Horn/TRI ring is NaN (edge effect) and is reported in quality.
    """

    grid = require_projected_grid(grid).validated()
    z = _prepare_elevation(grid)
    slope_aspect = horn_slope_aspect(grid)
    ruggedness = riley_tri(grid)
    elevation_layer = LayerResult(
        name="elevation_m",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.DEM_ELEVATION_ZONAL.value,
        provenance=provenance_record(
            MethodId.DEM_ELEVATION_ZONAL,
            parameters={
                "crs": grid.crs,
                "cell_size_x": float(grid.cell_size_x),
                "cell_size_y": float(grid.cell_size_y),
                "nodata": grid.nodata,
            },
            input_fingerprints=(fingerprint_array(z),),
        ),
        summary=summarize_array(z),
    )
    elevation = NumericResult(values=z, layer=elevation_layer)
    slope_mask = mask
    elevation_zonal = zonal_statistics(z, mask, statistic_used="mean")
    slope_zonal = zonal_statistics(slope_aspect.slope_deg.values, slope_mask, statistic_used="mean")
    aspect_zonal = zonal_statistics(
        slope_aspect.aspect_deg.values,
        slope_mask,
        circular=True,
        statistic_used="circular_mean",
    )
    ruggedness_zonal = zonal_statistics(ruggedness.values, slope_mask, statistic_used="mean")
    rows, cols = z.shape
    border = np.ones((rows, cols), dtype=bool)
    border[1:-1, 1:-1] = False
    if mask is not None:
        masked_border = bool(np.any(np.asarray(mask, dtype=bool) & border))
    else:
        masked_border = True
    finite_elev = int(np.isfinite(z).sum())
    finite_slope = int(np.isfinite(slope_aspect.slope_deg.values).sum())
    total = int(z.size)
    nodata_fraction = float((total - finite_elev) / total) if total else 1.0
    valid_fraction = (
        float(slope_zonal.finite_count / slope_zonal.count) if slope_zonal.count else 0.0
    )
    warnings: list[str] = []
    if grid.crs is None:
        warnings.append(
            "CRS omitted; cell_size_x/cell_size_y are treated as linear units matching z"
        )
    if masked_border:
        warnings.append("Horn/TRI leave a 1-cell NaN border; those cells are excluded")
    if slope_zonal.nodata_count:
        warnings.append(
            f"{slope_zonal.nodata_count} masked cells have no finite slope "
            "(nodata, edge, or incomplete 8-neighborhood)"
        )
    if valid_fraction < 1.0:
        warnings.append(f"valid_slope_fraction={valid_fraction:.6f}")
    quality = TerrainQuality(
        valid_fraction=valid_fraction,
        nodata_fraction=nodata_fraction,
        edge_cells_excluded=masked_border or finite_slope < finite_elev,
        finite_elevation_count=finite_elev,
        finite_slope_count=finite_slope,
        warnings=tuple(warnings),
    )
    return TerrainAnalysisResult(
        elevation=elevation,
        slope_aspect=slope_aspect,
        ruggedness_m=ruggedness,
        elevation_zonal=elevation_zonal,
        slope_zonal=slope_zonal,
        aspect_zonal=aspect_zonal,
        ruggedness_zonal=ruggedness_zonal,
        quality=quality,
        analysis_crs=grid.crs,
        cell_size_x=float(grid.cell_size_x),
        cell_size_y=float(grid.cell_size_y),
    )


def horizon_obstruction(_grid: ElevationGrid) -> None:
    """Reserved. Far-field horizon angle is not computed from the DEM yet."""

    raise UnsupportedMethodError(
        "TERRAIN_HORIZON_OBSTRUCTION is not implemented; slope is not a horizon angle"
    )


def shading_potential(_grid: ElevationGrid) -> None:
    """Reserved. Terrain shading is not inferred from slope."""

    raise UnsupportedMethodError(
        "TERRAIN_SHADING_POTENTIAL is not implemented; do not equate slope with shading"
    )


def _prepare_elevation(grid: ElevationGrid) -> FloatArray:
    z = np.asarray(grid.z, dtype=np.float64)
    if grid.nodata is not None:
        z = np.where(z == grid.nodata, np.nan, z)
    return np.asarray(z, dtype=np.float64)


__all__ = [
    "CALC_ASPECT_CIRCULAR",
    "CALC_ELEVATION_ZONAL",
    "CALC_HORN",
    "CALC_RILEY_TRI",
    "TerrainAnalysisResult",
    "TerrainQuality",
    "ZonalStats",
    "analyze_terrain",
    "circular_mean_deg",
    "crs_is_geographic",
    "horizon_obstruction",
    "require_projected_grid",
    "riley_tri",
    "shading_potential",
    "utm_epsg",
    "zonal_statistics",
]
