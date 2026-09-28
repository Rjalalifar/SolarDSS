"""Tabular candidate records backed by pandas."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from solardss.core.hashing import fingerprint_frame
from solardss.exceptions import SchemaError

DEFAULT_ID_COLUMN = "candidate_id"


@dataclass(frozen=True, slots=True)
class CandidateTable:
    """A table of candidate sites or cells.

    Construction copies the frame. ``candidate_id`` values must be unique
    strings. Geometry, if present, is an opaque column and is never interpreted
    unless a GIS extra method is used later.
    """

    frame: pd.DataFrame
    id_column: str = DEFAULT_ID_COLUMN

    def ids(self) -> tuple[str, ...]:
        return tuple(self.frame[self.id_column].astype(str).tolist())

    def fingerprint(self) -> str:
        return fingerprint_frame(self.frame, id_column=self.id_column)

    def __len__(self) -> int:
        return len(self.frame)


def validate_candidate_table(
    frame: pd.DataFrame,
    *,
    id_column: str = DEFAULT_ID_COLUMN,
    required_columns: Sequence[str] = (),
) -> CandidateTable:
    """Copy and validate a candidate table.

    Required columns are caller-defined. SolarDSS does not assume coordinates.
    """

    if id_column not in frame.columns:
        msg = f"Candidate table is missing id column {id_column!r}"
        raise SchemaError(msg)
    missing = [name for name in required_columns if name not in frame.columns]
    if missing:
        msg = f"Candidate table is missing required columns: {missing}"
        raise SchemaError(msg)
    copied = frame.copy(deep=True)
    copied[id_column] = copied[id_column].astype(str)
    ids = copied[id_column]
    if ids.duplicated().any():
        raise SchemaError("Candidate ids must be unique")
    if ids.str.len().eq(0).any():
        raise SchemaError("Candidate ids must be non-empty strings")
    return CandidateTable(frame=copied, id_column=id_column)
