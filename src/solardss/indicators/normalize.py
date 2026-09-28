"""Min-max normalization with explicit polarity. No imputation."""

from __future__ import annotations

import numpy as np

from solardss.core.arrays import AnyArray
from solardss.core.hashing import fingerprint_array
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, NumericResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.exceptions import IncompleteInputsError, InvalidParameterError
from solardss.indicators.contracts import Polarity
from solardss.provenance.record import provenance_record

_RANGE_ATOL = 0.0


def minmax_normalize(
    values: AnyArray,
    *,
    polarity: Polarity | str,
    range_min: float | None = None,
    range_max: float | None = None,
    name: str = "normalized",
) -> NumericResult:
    """Rescale finite values onto ``[0, 1]``.

    Benefit: ``(x - min) / (max - min)``.
    Cost: ``(max - x) / (max - min)``.

    If ``range_min`` and ``range_max`` are omitted, they are the finite sample
    min and max. A zero range cannot be normalized: this raises
    ``IncompleteInputsError`` rather than inventing a score.

    Non-finite inputs stay non-finite. ``Polarity.NONE`` is rejected.
    """

    if isinstance(polarity, str):
        polarity = Polarity(polarity)
    if polarity is Polarity.NONE:
        raise InvalidParameterError("minmax_normalize requires benefit or cost polarity")

    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if range_min is None or range_max is None:
        if finite.size == 0:
            raise IncompleteInputsError(
                "Cannot infer a min-max range from an array with no finite values"
            )
        inferred_min = float(np.min(finite)) if range_min is None else range_min
        inferred_max = float(np.max(finite)) if range_max is None else range_max
        range_min = inferred_min
        range_max = inferred_max
        range_source = "sample"
    else:
        range_source = "explicit"

    if range_max < range_min:
        raise InvalidParameterError("range_max must be greater than or equal to range_min")
    span = range_max - range_min
    if span <= _RANGE_ATOL:
        raise IncompleteInputsError(
            "min-max range is zero; supply a non-degenerate range_min and range_max"
        )

    scaled = (array - range_min) / span
    if polarity is Polarity.COST:
        scaled = 1.0 - scaled
    scaled = np.where(np.isfinite(array), scaled, np.nan)

    layer = LayerResult(
        name=name,
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.MINMAX_NORMALIZE.value,
        provenance=provenance_record(
            MethodId.MINMAX_NORMALIZE,
            parameters={
                "polarity": polarity.value,
                "range_min": range_min,
                "range_max": range_max,
                "range_source": range_source,
            },
            input_fingerprints=(fingerprint_array(array),),
        ),
        summary=summarize_array(scaled),
    )
    return NumericResult(values=scaled, layer=layer)
