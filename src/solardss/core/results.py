"""Typed result models returned by SolarDSS methods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from solardss.core.arrays import AnyArray, FloatArray
from solardss.core.hashing import fingerprint_array
from solardss.core.status import EvaluationStatus, FeasibilityStatus
from solardss.provenance.record import ProvenanceRecord


class ArraySummary(BaseModel):
    """JSON-safe numeric summary. Full arrays stay in memory, not in this model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    count: int
    finite_count: int
    min: float | None
    max: float | None
    mean: float | None


class LayerResult(BaseModel):
    """Status and provenance for one computed layer or table column."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    status: EvaluationStatus
    method_id: str
    provenance: ProvenanceRecord
    summary: ArraySummary | None = None
    warnings: tuple[str, ...] = ()
    excluded_count: int | None = None
    feasible_count: int | None = None


class RunResult(BaseModel):
    """Optional container for composing several layer results.

    SolarDSS does not run a hidden pipeline. Adapters may assemble ``RunResult``
    after calling individual methods.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    status: EvaluationStatus
    layers: tuple[LayerResult, ...] = ()
    selected_ids: tuple[str, ...] = ()
    provenance: ProvenanceRecord
    warnings: tuple[str, ...] = ()


class SelectionResult(BaseModel):
    """Outcome of a documented selection rule (currently greedy-by-score)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EvaluationStatus
    selected_ids: tuple[str, ...]
    remaining_budget: float
    total_score: float
    total_cost: float
    skipped_over_budget_ids: tuple[str, ...] = ()
    provenance: ProvenanceRecord
    warnings: tuple[str, ...] = ()


class ScreeningResult(BaseModel):
    """Constraint-screening outcome with counts only (masks stay in memory)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EvaluationStatus
    method_id: str
    excluded_count: int
    feasible_count: int
    unknown_count: int = 0
    default_feasibility: FeasibilityStatus = FeasibilityStatus.FEASIBLE
    provenance: ProvenanceRecord
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NumericResult:
    """In-memory array plus its serializable layer metadata."""

    values: FloatArray
    layer: LayerResult

    def fingerprint(self) -> str:
        return fingerprint_array(self.values)


def summarize_array(values: AnyArray) -> ArraySummary:
    """Finite-aware min/max/mean. Empty or all-nonfinite arrays have null stats."""

    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        return ArraySummary(
            count=int(flat.size),
            finite_count=0,
            min=None,
            max=None,
            mean=None,
        )
    return ArraySummary(
        count=int(flat.size),
        finite_count=int(finite.size),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
        mean=float(np.mean(finite)),
    )
