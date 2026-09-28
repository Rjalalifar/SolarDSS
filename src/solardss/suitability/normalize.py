"""Criterion normalization for the Weighted Sum Model.

Benefit: ``(x - min) / (max - min)``.
Cost: ``(max - x) / (max - min)``.
Target: ``1 - |x - t| / max_i |x_i - t|`` (explicit range uses
``max(|min - t|, |max - t|)`` as the denominator).

Equal min/max (or zero target deviation) is a defined edge case: every finite
value scores ``1.0``. Missing values stay missing; zero is not substituted.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from solardss.core.arrays import AnyArray, FloatArray
from solardss.core.hashing import fingerprint_array
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, NumericResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.exceptions import IncompleteInputsError, InvalidParameterError
from solardss.provenance.record import provenance_record
from solardss.suitability.contracts import NormalizationKind, NormalizationStats
from solardss.suitability.errors import RankingConfigError

_RANGE_ATOL = 0.0


def _as_kind(value: NormalizationKind | str) -> NormalizationKind:
    if isinstance(value, NormalizationKind):
        return value
    try:
        return NormalizationKind(value)
    except ValueError as exc:
        raise InvalidParameterError(f"unknown normalization kind {value!r}") from exc


def _finite_minmax(array: FloatArray) -> tuple[float, float]:
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise IncompleteInputsError(
            "Cannot infer a normalization range from an array with no finite values"
        )
    return float(np.min(finite)), float(np.max(finite))


def normalize_criterion(
    values: AnyArray,
    *,
    kind: NormalizationKind | str,
    range_min: float | None = None,
    range_max: float | None = None,
    target: float | None = None,
    indicator_id: str = "criterion",
    unit: str = "1",
    name: str = "normalized",
) -> tuple[NumericResult, NormalizationStats]:
    """Scale finite values for WSM. Equal-range inputs score 1.0, not NaN."""

    polarity = _as_kind(kind)
    if polarity is NormalizationKind.TARGET and target is None:
        raise InvalidParameterError("target normalization requires target")
    if polarity is not NormalizationKind.TARGET and target is not None:
        raise InvalidParameterError("target is only valid for target normalization")
    if (range_min is None) ^ (range_max is None):
        raise InvalidParameterError("range_min and range_max must be supplied together")

    array = np.asarray(values, dtype=np.float64)
    if range_min is None or range_max is None:
        lo, hi = _finite_minmax(array)
        range_source = "sample"
    else:
        if range_max < range_min:
            raise InvalidParameterError("range_max must be greater than or equal to range_min")
        lo, hi = range_min, range_max
        range_source = "explicit"

    equal = (hi - lo) <= _RANGE_ATOL
    scaled = np.full(array.shape, np.nan, dtype=np.float64)
    finite_mask = np.isfinite(array)

    if polarity is NormalizationKind.TARGET:
        assert target is not None
        if range_source == "explicit":
            denom = max(abs(lo - target), abs(hi - target))
        else:
            denom = float(np.max(np.abs(array[finite_mask] - target))) if finite_mask.any() else 0.0
        if denom <= _RANGE_ATOL:
            scaled = np.where(finite_mask, 1.0, np.nan)
            range_source = "degenerate_equal"
            equal = True
        else:
            scaled = np.where(finite_mask, 1.0 - np.abs(array - target) / denom, np.nan)
            equal = False
        stats = NormalizationStats(
            indicator_id=indicator_id,
            kind=polarity,
            range_min=lo,
            range_max=hi,
            range_source=range_source,
            equal_values=equal,
            target_value=target,
            unit=unit,
        )
    elif equal:
        scaled = np.where(finite_mask, 1.0, np.nan)
        range_source = "degenerate_equal" if range_source == "sample" else range_source
        stats = NormalizationStats(
            indicator_id=indicator_id,
            kind=polarity,
            range_min=lo,
            range_max=hi,
            range_source=range_source,
            equal_values=True,
            unit=unit,
        )
    else:
        span = hi - lo
        if polarity is NormalizationKind.BENEFIT:
            scaled = np.where(finite_mask, (array - lo) / span, np.nan)
        else:
            scaled = np.where(finite_mask, (hi - array) / span, np.nan)
        stats = NormalizationStats(
            indicator_id=indicator_id,
            kind=polarity,
            range_min=lo,
            range_max=hi,
            range_source=range_source,
            equal_values=False,
            unit=unit,
        )

    layer = LayerResult(
        name=name,
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.CRITERION_NORMALIZE.value,
        provenance=provenance_record(
            MethodId.CRITERION_NORMALIZE,
            parameters={
                "kind": polarity.value,
                "range_min": stats.range_min,
                "range_max": stats.range_max,
                "range_source": stats.range_source,
                "equal_values": stats.equal_values,
                "target": stats.target_value,
                "indicator_id": indicator_id,
            },
            input_fingerprints=(fingerprint_array(array),),
            notes=("equal min/max scores 1.0; missing values stay missing",),
        ),
        summary=summarize_array(scaled),
    )
    return NumericResult(values=scaled, layer=layer), stats


def scores_from_optional(
    values: Sequence[float | None],
    *,
    kind: NormalizationKind,
    range_min: float | None = None,
    range_max: float | None = None,
    target: float | None = None,
    indicator_id: str,
    unit: str,
) -> tuple[tuple[float | None, ...], NormalizationStats]:
    """Normalize a list that may contain missing observations."""

    if all(item is None for item in values):
        raise RankingConfigError(
            f"{indicator_id}: every observation is missing; zero is not substituted"
        )
    raw = np.asarray(
        [float("nan") if item is None else item for item in values],
        dtype=np.float64,
    )
    result, stats = normalize_criterion(
        raw,
        kind=kind,
        range_min=range_min,
        range_max=range_max,
        target=target,
        indicator_id=indicator_id,
        unit=unit,
    )
    out: list[float | None] = []
    for original, scaled in zip(values, result.values.tolist(), strict=True):
        if original is None:
            out.append(None)
        else:
            out.append(float(scaled))
    return tuple(out), stats
