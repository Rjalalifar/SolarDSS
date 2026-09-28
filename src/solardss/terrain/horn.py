"""Horn (1981) 8-neighbor slope and aspect on a regular grid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solardss.core.arrays import FloatArray
from solardss.core.hashing import fingerprint_array
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, NumericResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.provenance.record import provenance_record
from solardss.terrain.contracts import ElevationGrid


@dataclass(frozen=True, slots=True)
class SlopeAspectResult:
    """Paired slope (degrees) and aspect (degrees clockwise from north)."""

    slope_deg: NumericResult
    aspect_deg: NumericResult


def _prepare_elevation(grid: ElevationGrid) -> FloatArray:
    z = np.asarray(grid.z, dtype=np.float64)
    if grid.nodata is not None:
        z = np.where(z == grid.nodata, np.nan, z)
    return np.asarray(z, dtype=np.float64)


def horn_slope_aspect(grid: ElevationGrid) -> SlopeAspectResult:
    """Compute Horn (1981) slope and aspect.

    Slope is ``atan(sqrt((dz/dx)^2 + (dz/dy)^2))`` in degrees, in ``[0, 90]``.
    Aspect is the direction of **steepest descent**, degrees clockwise from
    north, in ``[0, 360)``. Flat cells (slope 0) and the 1-cell border are
    NaN. A cell is NaN if any of its 8 neighbors is non-finite.

    ``dz/dx`` is eastward; ``dz/dy`` is northward. This matches a north-up
    raster (row index increases south).
    """

    from solardss.terrain.analysis import require_projected_grid

    grid = require_projected_grid(grid).validated()
    z = _prepare_elevation(grid)
    dx = float(grid.cell_size_x)
    dy = float(grid.cell_size_y)

    slope = np.full(z.shape, np.nan, dtype=np.float64)
    aspect = np.full(z.shape, np.nan, dtype=np.float64)

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
    z_nw, z_n, z_ne, z_w, _, z_e, z_sw, z_s, z_se = window

    dz_dx = ((z_ne + 2.0 * z_e + z_se) - (z_nw + 2.0 * z_w + z_sw)) / (8.0 * dx)
    dz_dy = ((z_nw + 2.0 * z_n + z_ne) - (z_sw + 2.0 * z_s + z_se)) / (8.0 * dy)

    rise = np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)
    slope_interior = np.degrees(np.arctan(rise))
    # Steepest ascent azimuth from north, clockwise; add 180° for descent.
    ascent_deg = np.degrees(np.arctan2(dz_dx, dz_dy))
    descent_deg = np.mod(ascent_deg + 180.0, 360.0)
    flat = rise == 0.0
    descent_deg = np.where(flat, np.nan, descent_deg)

    slope[1:-1, 1:-1] = np.where(finite, slope_interior, np.nan)
    aspect[1:-1, 1:-1] = np.where(finite, descent_deg, np.nan)

    fingerprints = (fingerprint_array(z),)
    parameters = {
        "cell_size_x": dx,
        "cell_size_y": dy,
        "nodata": grid.nodata,
        "crs": grid.crs,
        "origin_x": grid.origin_x,
        "origin_y": grid.origin_y,
        "convention": "horn_1981_steepest_descent_from_north_clockwise",
    }
    slope_layer = LayerResult(
        name="slope_deg",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.HORN_SLOPE_ASPECT.value,
        provenance=provenance_record(
            MethodId.HORN_SLOPE_ASPECT,
            parameters=parameters,
            input_fingerprints=fingerprints,
            notes=("slope component",),
        ),
        summary=summarize_array(slope),
    )
    aspect_layer = LayerResult(
        name="aspect_deg",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.HORN_SLOPE_ASPECT.value,
        provenance=provenance_record(
            MethodId.HORN_SLOPE_ASPECT,
            parameters=parameters,
            input_fingerprints=fingerprints,
            notes=("aspect component",),
        ),
        summary=summarize_array(aspect),
    )
    return SlopeAspectResult(
        slope_deg=NumericResult(values=slope, layer=slope_layer),
        aspect_deg=NumericResult(values=aspect, layer=aspect_layer),
    )
