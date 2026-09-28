"""Regular and hexagonal covering cells. Cell size is caller-supplied."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from math import ceil, cos, floor, radians, sin, sqrt

from solardss.candidates.contracts import BoundaryGeometry, GenerationStrategy
from solardss.candidates.errors import CandidateGridError
from solardss.candidates.geometry import closed_ring, envelope_of, ring_bbox

SQRT3 = sqrt(3.0)


@dataclass(frozen=True, slots=True)
class GridCellSpec:
    """One unclipped covering cell plus its grid indices."""

    row: int
    col: int
    exterior: tuple[tuple[float, float], ...]


def square_cell(
    col: int, row: int, *, origin_x: float, origin_y: float, cell_size: float
) -> tuple[tuple[float, float], ...]:
    x0 = origin_x + col * cell_size
    y0 = origin_y + row * cell_size
    x1 = x0 + cell_size
    y1 = y0 + cell_size
    return closed_ring(((x0, y0), (x1, y0), (x1, y1), (x0, y1)))


def hex_center(
    q: int, r: int, *, origin_x: float, origin_y: float, hex_size: float
) -> tuple[float, float]:
    """Pointy-top axial centre. ``hex_size`` is centre-to-vertex."""

    x = origin_x + hex_size * (SQRT3 * q + SQRT3 / 2.0 * r)
    y = origin_y + hex_size * (1.5 * r)
    return x, y


def hex_cell(
    q: int, r: int, *, origin_x: float, origin_y: float, hex_size: float
) -> tuple[tuple[float, float], ...]:
    cx, cy = hex_center(q, r, origin_x=origin_x, origin_y=origin_y, hex_size=hex_size)
    verts: list[tuple[float, float]] = []
    for index in range(6):
        angle = radians(60 * index - 30)
        verts.append((cx + hex_size * cos(angle), cy + hex_size * sin(angle)))
    return closed_ring(verts)


def iter_covering_cells(
    boundary: BoundaryGeometry,
    *,
    strategy: GenerationStrategy,
    cell_size: float,
    origin_x: float | None,
    origin_y: float | None,
    max_candidates: int | None,
) -> Iterator[GridCellSpec]:
    """Yield unclipped cells whose envelope intersects the boundary envelope."""

    minx, miny, maxx, maxy = envelope_of(boundary)
    if strategy is GenerationStrategy.REGULAR_GRID:
        yield from _iter_squares(
            minx,
            miny,
            maxx,
            maxy,
            cell_size=cell_size,
            origin_x=origin_x,
            origin_y=origin_y,
            max_candidates=max_candidates,
        )
        return
    if strategy is GenerationStrategy.HEXAGONAL_GRID:
        yield from _iter_hexes(
            minx,
            miny,
            maxx,
            maxy,
            cell_size=cell_size,
            origin_x=origin_x,
            origin_y=origin_y,
            max_candidates=max_candidates,
        )
        return
    raise CandidateGridError(f"{strategy.value} is not a grid covering strategy")


def _iter_squares(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    cell_size: float,
    origin_x: float | None,
    origin_y: float | None,
    max_candidates: int | None,
) -> Iterator[GridCellSpec]:
    x0 = minx if origin_x is None else origin_x
    y0 = miny if origin_y is None else origin_y
    col0 = floor((minx - x0) / cell_size) - 1
    col1 = ceil((maxx - x0) / cell_size) + 1
    row0 = floor((miny - y0) / cell_size) - 1
    row1 = ceil((maxy - y0) / cell_size) + 1
    estimated = (col1 - col0) * (row1 - row0)
    _check_max(estimated, max_candidates)
    envelope = (minx, miny, maxx, maxy)
    for row in range(row0, row1):
        for col in range(col0, col1):
            ring = square_cell(col, row, origin_x=x0, origin_y=y0, cell_size=cell_size)
            if _disjoint(ring, envelope):
                continue
            yield GridCellSpec(row=row, col=col, exterior=ring)


def _iter_hexes(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    cell_size: float,
    origin_x: float | None,
    origin_y: float | None,
    max_candidates: int | None,
) -> Iterator[GridCellSpec]:
    # cell_size is centre-to-centre = flat-to-flat; hex_size is centre-to-vertex.
    hex_size = cell_size / SQRT3
    x0 = minx if origin_x is None else origin_x
    y0 = miny if origin_y is None else origin_y
    vert = 1.5 * hex_size
    horiz = SQRT3 * hex_size
    r0 = floor((miny - y0) / vert) - 2
    r1 = ceil((maxy - y0) / vert) + 2
    estimated = 0
    ranges: list[tuple[int, int, int]] = []
    for row in range(r0, r1 + 1):
        offset = (SQRT3 / 2.0) * hex_size * row
        q0 = floor((minx - x0 - offset) / horiz) - 2
        q1 = ceil((maxx - x0 - offset) / horiz) + 2
        estimated += q1 - q0
        ranges.append((row, q0, q1))
    _check_max(estimated, max_candidates)
    envelope = (minx, miny, maxx, maxy)
    for row, q0, q1 in ranges:
        for col in range(q0, q1):
            ring = hex_cell(col, row, origin_x=x0, origin_y=y0, hex_size=hex_size)
            if _disjoint(ring, envelope):
                continue
            yield GridCellSpec(row=row, col=col, exterior=ring)


def _disjoint(
    ring: tuple[tuple[float, float], ...],
    envelope: tuple[float, float, float, float],
) -> bool:
    box = ring_bbox(ring)
    return (
        box[2] < envelope[0] or envelope[2] < box[0] or box[3] < envelope[1] or envelope[3] < box[1]
    )


def _check_max(estimated: int, max_candidates: int | None) -> None:
    if max_candidates is not None and estimated > max_candidates:
        raise CandidateGridError(
            f"covering grid estimates {estimated} cells, above max_candidates={max_candidates}"
        )
