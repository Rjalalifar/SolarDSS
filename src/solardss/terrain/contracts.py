"""Regular elevation grids. Units of z and cell size must already match."""

from __future__ import annotations

from dataclasses import dataclass

from solardss.core.arrays import AnyArray
from solardss.exceptions import InvalidGridError


@dataclass(frozen=True, slots=True)
class ElevationGrid:
    """North-up image convention: row index increases south, column increases east.

    ``cell_size_x`` and ``cell_size_y`` are positive spacings in the same linear
    unit as ``z``. CRS and origin are provenance metadata only; they are not
    used in the Horn derivative.
    """

    z: AnyArray
    cell_size_x: float
    cell_size_y: float
    nodata: float | None = None
    crs: str | None = None
    origin_x: float | None = None
    origin_y: float | None = None

    def validated(self) -> ElevationGrid:
        if self.z.ndim != 2:
            raise InvalidGridError("Elevation grid must be 2-dimensional")
        rows, cols = self.z.shape
        if rows < 3 or cols < 3:
            raise InvalidGridError("Horn slope requires a grid of at least 3x3 cells")
        if self.cell_size_x <= 0.0 or self.cell_size_y <= 0.0:
            raise InvalidGridError("cell_size_x and cell_size_y must be positive")
        return self
