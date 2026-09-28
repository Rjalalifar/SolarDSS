"""Core status, result, and quantity types."""

from __future__ import annotations

from solardss.core.hashing import (
    fingerprint_array,
    fingerprint_canonical,
    fingerprint_frame,
    fingerprint_sequence,
)
from solardss.core.methods import METHOD_VERSIONS, MethodId
from solardss.core.quantity import Quantity
from solardss.core.results import (
    ArraySummary,
    LayerResult,
    NumericResult,
    RunResult,
    ScreeningResult,
    SelectionResult,
    summarize_array,
)
from solardss.core.status import EvaluationStatus, FeasibilityStatus

__all__ = [
    "METHOD_VERSIONS",
    "ArraySummary",
    "EvaluationStatus",
    "FeasibilityStatus",
    "LayerResult",
    "MethodId",
    "NumericResult",
    "Quantity",
    "RunResult",
    "ScreeningResult",
    "SelectionResult",
    "fingerprint_array",
    "fingerprint_canonical",
    "fingerprint_frame",
    "fingerprint_sequence",
    "summarize_array",
]
