"""Adjacency grouping for later merge of contiguous ground-mounted cells.

This does not rank cells and does not dissolve geometries. Isolated cells are
one-cell groups. Features without grid indices cannot be grouped this way.
"""

from __future__ import annotations

from collections.abc import Mapping

from solardss.candidates.contracts import CandidateCollection, GenerationStrategy
from solardss.core.hashing import fingerprint_sequence
from solardss.core.methods import MethodId
from solardss.exceptions import InvalidParameterError
from solardss.provenance.record import ProvenanceRecord, provenance_record

_SQUARE_ROOK = ((1, 0), (-1, 0), (0, 1), (0, -1))
_HEX_AXIAL = ((1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1))


def contiguous_groups(
    collection: CandidateCollection,
    *,
    suitable: Mapping[str, bool] | None = None,
) -> tuple[tuple[tuple[str, ...], ...], ProvenanceRecord]:
    """Group cells that share an edge.

    ``suitable`` is a boolean mask by candidate id. Missing keys are treated as
    not suitable (not as True). Scores are ignored: this is not ranking.
    """

    if collection.strategy is GenerationStrategy.FEATURES:
        raise InvalidParameterError(
            "contiguous grouping requires a grid; parcel/building adjacency is not inferred"
        )
    offsets = (
        _HEX_AXIAL if collection.strategy is GenerationStrategy.HEXAGONAL_GRID else _SQUARE_ROOK
    )
    indexed: dict[tuple[int, int], str] = {}
    for item in collection.candidates:
        if item.grid_row is None or item.grid_col is None:
            raise InvalidParameterError(
                f"candidate {item.candidate_id!r} is missing grid indices needed for grouping"
            )
        if suitable is not None and not suitable.get(item.candidate_id, False):
            continue
        indexed[(item.grid_col, item.grid_row)] = item.candidate_id
    parent = {ident: ident for ident in indexed.values()}

    def find(ident: str) -> str:
        while parent[ident] != ident:
            parent[ident] = parent[parent[ident]]
            ident = parent[ident]
        return ident

    def union(left: str, right: str) -> None:
        root_l, root_r = find(left), find(right)
        if root_l == root_r:
            return
        if root_l < root_r:
            parent[root_r] = root_l
        else:
            parent[root_l] = root_r

    for (col, row), ident in indexed.items():
        for dcol, drow in offsets:
            neighbor = indexed.get((col + dcol, row + drow))
            if neighbor is not None:
                union(ident, neighbor)

    buckets: dict[str, list[str]] = {}
    for ident in indexed.values():
        buckets.setdefault(find(ident), []).append(ident)
    groups = tuple(tuple(sorted(members)) for _, members in sorted(buckets.items()))
    record = provenance_record(
        MethodId.CANDIDATES_CONTIGUOUS_GROUPS,
        parameters={
            "strategy": collection.strategy.value,
            "suitable_filter": suitable is not None,
            "group_count": len(groups),
        },
        input_fingerprints=(fingerprint_sequence(collection.ids()),),
        notes=("edge-adjacent cells only; geometries are not dissolved",),
    )
    return groups, record
