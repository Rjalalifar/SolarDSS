"""AHP priority vectors via the geometric-mean method.

This implements a published decision method on a caller-supplied pairwise
matrix. It does not invent criteria or judgments.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict

from solardss.core.arrays import AnyArray, FloatArray
from solardss.core.hashing import fingerprint_array
from solardss.core.methods import MethodId
from solardss.core.status import EvaluationStatus
from solardss.exceptions import InvalidParameterError, InvalidWeightsError
from solardss.provenance.record import ProvenanceRecord, provenance_record

# Saaty, T. L. (1980). The Analytic Hierarchy Process. McGraw-Hill.
# Random index for n = 1..10 as commonly tabulated for CR = CI / RI.
SAATY_RANDOM_INDEX: dict[int, float] = {
    1: 0.00,
    2: 0.00,
    3: 0.58,
    4: 0.90,
    5: 1.12,
    6: 1.24,
    7: 1.32,
    8: 1.41,
    9: 1.45,
    10: 1.49,
}

_RECIPROCAL_RTOL = 1e-9


class AhpResult(BaseModel):
    """Priority vector, λ_max, and Saaty consistency ratio."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    criteria: tuple[str, ...]
    weights: tuple[float, ...]
    lambda_max: float
    consistency_index: float
    consistency_ratio: float | None
    status: EvaluationStatus
    method_id: str
    provenance: ProvenanceRecord
    warnings: tuple[str, ...] = ()


def _validate_matrix(matrix: AnyArray, criteria: tuple[str, ...]) -> FloatArray:
    array = np.asarray(matrix, dtype=np.float64)
    n = len(criteria)
    if array.shape != (n, n):
        raise InvalidParameterError("Pairwise matrix shape must be (n, n) for n criteria")
    if n < 1:
        raise InvalidParameterError("AHP requires at least one criterion")
    if not np.all(np.isfinite(array)) or np.any(array <= 0.0):
        raise InvalidParameterError("Pairwise matrix entries must be finite and positive")
    if np.max(np.abs(np.diag(array) - 1.0)) > _RECIPROCAL_RTOL:
        raise InvalidParameterError("Pairwise matrix diagonal must be 1")
    reciprocal = array * array.T
    if np.max(np.abs(reciprocal - 1.0)) > _RECIPROCAL_RTOL * 10.0:
        raise InvalidParameterError("Pairwise matrix must be reciprocal (a_ji = 1 / a_ij)")
    if len(set(criteria)) != n:
        raise InvalidParameterError("AHP criteria names must be unique")
    return array


def ahp_geometric_mean(
    matrix: AnyArray,
    criteria: tuple[str, ...],
    *,
    max_consistency_ratio: float | None = None,
) -> AhpResult:
    """Geometric-mean priority vector (Crawford & Williams, 1985) plus Saaty CR.

    For a consistent matrix this coincides with the principal eigenvector.
    ``consistency_ratio`` is ``None`` when n < 3 (RI is defined as 0 and CR is
    not used). For n > 10 the published RI table used here does not apply and
    CR is returned as ``None`` with a warning; λ_max and CI are still reported.

    If ``max_consistency_ratio`` is set and CR exceeds it, this raises
    ``InvalidWeightsError`` rather than silently using inconsistent weights.
    """

    names = tuple(criteria)
    array = _validate_matrix(matrix, names)
    n = len(names)
    log_values = np.log(array)
    geometric = np.exp(np.mean(log_values, axis=1))
    weight_sum = float(np.sum(geometric))
    weights = geometric / weight_sum
    aw = array @ weights
    lambda_components = aw / weights
    lambda_max = float(np.mean(lambda_components))
    consistency_index = float((lambda_max - n) / (n - 1)) if n > 1 else 0.0
    warnings: list[str] = []
    ri = SAATY_RANDOM_INDEX.get(n)
    if n < 3:
        consistency_ratio: float | None = 0.0
    elif ri is None:
        consistency_ratio = None
        warnings.append("saaty_ri_undefined_for_n")
    elif ri == 0.0:
        consistency_ratio = 0.0
    else:
        consistency_ratio = float(consistency_index / ri)

    if (
        max_consistency_ratio is not None
        and consistency_ratio is not None
        and consistency_ratio > max_consistency_ratio
    ):
        raise InvalidWeightsError(
            "AHP consistency ratio "
            f"{consistency_ratio} exceeds max_consistency_ratio {max_consistency_ratio}"
        )

    provenance = provenance_record(
        MethodId.AHP_GEOMETRIC_MEAN,
        parameters={
            "criteria": list(names),
            "max_consistency_ratio": max_consistency_ratio,
            "random_index": ri,
        },
        input_fingerprints=(fingerprint_array(array),),
        notes=tuple(warnings),
    )
    return AhpResult(
        criteria=names,
        weights=tuple(float(value) for value in weights),
        lambda_max=lambda_max,
        consistency_index=consistency_index,
        consistency_ratio=consistency_ratio,
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.AHP_GEOMETRIC_MEAN.value,
        provenance=provenance,
        warnings=tuple(warnings),
    )
