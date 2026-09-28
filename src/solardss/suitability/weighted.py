"""Weighted linear combination of caller-supplied score layers."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from solardss.core.arrays import AnyArray, FloatArray
from solardss.core.hashing import fingerprint_array
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, NumericResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.exceptions import InvalidWeightsError
from solardss.provenance.record import provenance_record

_WEIGHT_ATOL = 1e-12


def weighted_linear_combination(
    scores: Mapping[str, AnyArray],
    weights: Mapping[str, float],
    *,
    normalize_weights: bool = False,
    name: str = "suitability",
) -> NumericResult:
    """Compute ``sum(w_i * x_i)`` on aligned arrays.

    Weights must be non-negative. If ``normalize_weights`` is false (default),
    they must sum to 1 within ``1e-12``. If true, they are divided by their
    sum and that choice is recorded in provenance.

    NaN scores propagate. No missing-score imputation is performed.
    """

    if not scores:
        raise InvalidWeightsError("weighted_linear_combination requires at least one score layer")
    score_keys = set(scores)
    weight_keys = set(weights)
    if score_keys != weight_keys:
        raise InvalidWeightsError(
            "Score layer names must match weight names exactly; "
            f"scores={sorted(score_keys)}, weights={sorted(weight_keys)}"
        )
    names = tuple(sorted(score_keys))
    shape: tuple[int, ...] | None = None
    arrays: dict[str, FloatArray] = {}
    for key in names:
        array = np.asarray(scores[key], dtype=np.float64)
        if shape is None:
            shape = array.shape
        elif array.shape != shape:
            raise InvalidWeightsError("Score layers must share the same shape")
        arrays[key] = array
        if weights[key] < 0.0:
            raise InvalidWeightsError(f"Weight for {key!r} is negative")
    if shape is None:
        raise InvalidWeightsError("weighted_linear_combination requires at least one score layer")

    raw_sum = float(sum(float(weights[key]) for key in names))
    if raw_sum <= 0.0:
        raise InvalidWeightsError("Weights must sum to a positive value")
    if normalize_weights:
        used = {key: float(weights[key]) / raw_sum for key in names}
        weight_mode = "normalized"
    else:
        if abs(raw_sum - 1.0) > _WEIGHT_ATOL:
            raise InvalidWeightsError("Weights must sum to 1, or pass normalize_weights=True")
        used = {key: float(weights[key]) for key in names}
        weight_mode = "as_supplied"

    combined = np.zeros(shape, dtype=np.float64)
    for key in names:
        combined = combined + used[key] * arrays[key]

    layer = LayerResult(
        name=name,
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.WEIGHTED_LINEAR_COMBINATION.value,
        provenance=provenance_record(
            MethodId.WEIGHTED_LINEAR_COMBINATION,
            parameters={
                "weights": used,
                "weight_mode": weight_mode,
                "layers": list(names),
            },
            input_fingerprints=tuple(fingerprint_array(arrays[key]) for key in names),
        ),
        summary=summarize_array(combined),
    )
    return NumericResult(values=combined, layer=layer)
