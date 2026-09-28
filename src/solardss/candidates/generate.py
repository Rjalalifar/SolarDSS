"""Build candidate tables from feasibility masks or screened records."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solardss.core.arrays import BoolArray
from solardss.core.hashing import fingerprint_array
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.data.tables import DEFAULT_ID_COLUMN, CandidateTable, validate_candidate_table
from solardss.exceptions import InvalidParameterError, SchemaError
from solardss.provenance.record import provenance_record


def candidates_from_mask(
    feasible: BoolArray,
    *,
    id_template: str = "r{row}c{col}",
) -> tuple[CandidateTable, LayerResult]:
    """Each True cell becomes a candidate with row/col and a derived id.

    ``id_template`` must contain ``{row}`` and ``{col}``. This does not invent
    coordinates; it only records array indices.
    """

    mask = np.asarray(feasible)
    if mask.dtype != bool:
        raise InvalidParameterError("feasible mask must have dtype bool")
    if mask.ndim != 2:
        raise InvalidParameterError("feasible mask must be 2-dimensional")
    if "{row}" not in id_template or "{col}" not in id_template:
        raise InvalidParameterError("id_template must contain {row} and {col}")

    rows, cols = np.nonzero(mask)
    ids = [id_template.format(row=int(r), col=int(c)) for r, c in zip(rows, cols, strict=True)]
    frame = pd.DataFrame(
        {
            DEFAULT_ID_COLUMN: ids,
            "row": rows.astype(np.int64),
            "col": cols.astype(np.int64),
        }
    )
    table = validate_candidate_table(frame)
    layer = LayerResult(
        name="candidates",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.CANDIDATES_FROM_MASK.value,
        provenance=provenance_record(
            MethodId.CANDIDATES_FROM_MASK,
            parameters={"id_template": id_template, "shape": list(mask.shape)},
            input_fingerprints=(fingerprint_array(mask.astype(np.uint8)),),
        ),
        summary=summarize_array(mask.astype(np.float64)),
        feasible_count=len(table),
        excluded_count=int(mask.size - len(table)),
    )
    return table, layer


def retain_feasible(
    table: CandidateTable,
    excluded: pd.Series,
) -> tuple[CandidateTable, LayerResult]:
    """Keep rows where ``excluded`` is False. Empty output remains empty."""

    if len(excluded) != len(table):
        raise SchemaError("excluded flags must align with the candidate table")
    flags = excluded.to_numpy()
    if flags.dtype != bool:
        raise InvalidParameterError("excluded flags must be boolean")
    kept = table.frame.loc[~excluded.to_numpy()].reset_index(drop=True)
    retained = validate_candidate_table(kept, id_column=table.id_column)
    layer = LayerResult(
        name="feasible_candidates",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.RETAIN_FEASIBLE.value,
        provenance=provenance_record(
            MethodId.RETAIN_FEASIBLE,
            parameters={"id_column": table.id_column},
            input_fingerprints=(table.fingerprint(), fingerprint_array(flags.astype(np.uint8))),
        ),
        feasible_count=len(retained),
        excluded_count=len(table) - len(retained),
    )
    return retained, layer
