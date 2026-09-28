"""Boolean exclusion screening. Layers are caller-supplied; none are implied."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from solardss.core.arrays import AnyArray, BoolArray
from solardss.core.hashing import fingerprint_array
from solardss.core.methods import MethodId
from solardss.core.results import ScreeningResult
from solardss.core.status import EvaluationStatus, FeasibilityStatus
from solardss.data.tables import CandidateTable
from solardss.exceptions import InvalidParameterError, SchemaError
from solardss.provenance.record import provenance_record


def apply_exclusions(
    layers: Mapping[str, AnyArray],
    *,
    true_means_excluded: bool = True,
) -> tuple[BoolArray, BoolArray, ScreeningResult]:
    """Combine boolean exclusion layers with logical OR.

    Returns ``(excluded, feasible, result)``. A cell is excluded if any layer is
    true (when ``true_means_excluded`` is true). Non-boolean arrays are rejected.
    Empty ``layers`` is incomplete: SolarDSS will not assume that everything is
    feasible without an explicit empty-layer policy.

    Pass a single all-false layer if the caller has evaluated constraints and
    found none.
    """

    if not layers:
        raise InvalidParameterError(
            "apply_exclusions requires at least one layer; pass an all-false "
            "mask if constraints were evaluated and none apply"
        )
    arrays = []
    names = []
    shape: tuple[int, ...] | None = None
    for name, raw in layers.items():
        array = np.asarray(raw)
        if array.dtype != bool:
            raise InvalidParameterError(f"Exclusion layer {name!r} must have dtype bool")
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise InvalidParameterError("Exclusion layers must share the same shape")
        arrays.append(array)
        names.append(name)

    stacked = np.stack(arrays, axis=0)
    any_true = np.any(stacked, axis=0)
    excluded = any_true if true_means_excluded else ~any_true
    feasible = ~excluded
    fingerprints = tuple(
        fingerprint_array(np.asarray(layers[name]).astype(np.uint8, copy=False)) for name in names
    )
    result = ScreeningResult(
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.BOOLEAN_EXCLUSION.value,
        excluded_count=int(np.count_nonzero(excluded)),
        feasible_count=int(np.count_nonzero(feasible)),
        unknown_count=0,
        default_feasibility=FeasibilityStatus.FEASIBLE,
        provenance=provenance_record(
            MethodId.BOOLEAN_EXCLUSION,
            parameters={
                "true_means_excluded": true_means_excluded,
                "layer_names": names,
            },
            input_fingerprints=fingerprints,
        ),
    )
    return excluded, feasible, result


def apply_exclusion_columns(
    table: CandidateTable,
    columns: tuple[str, ...],
    *,
    true_means_excluded: bool = True,
) -> tuple[pd.Series, ScreeningResult]:
    """OR-combine boolean columns of a candidate table. Returns excluded flags."""

    if not columns:
        raise InvalidParameterError("apply_exclusion_columns requires at least one column")
    missing = [name for name in columns if name not in table.frame.columns]
    if missing:
        raise SchemaError(f"Exclusion columns missing: {missing}")
    layers = {name: table.frame[name].to_numpy() for name in columns}
    for name, array in layers.items():
        if array.dtype != bool:
            raise InvalidParameterError(f"Exclusion column {name!r} must be boolean")
    excluded, _feasible, result = apply_exclusions(
        layers,
        true_means_excluded=true_means_excluded,
    )
    flags = pd.Series(excluded, index=table.frame.index, name="excluded")
    fingerprints = (*result.provenance.input_fingerprints, table.fingerprint())
    updated = result.model_copy(
        update={
            "provenance": result.provenance.model_copy(update={"input_fingerprints": fingerprints}),
        }
    )
    return flags, updated
