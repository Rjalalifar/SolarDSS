"""Core status, quantity, and result models."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from solardss.core import (
    EvaluationStatus,
    LayerResult,
    Quantity,
    RunResult,
    fingerprint_array,
    summarize_array,
)
from solardss.core.methods import MethodId
from solardss.exceptions import UnitError
from solardss.provenance import provenance_record


def test_quantity_unit_match() -> None:
    qty = Quantity(value=3.5, unit="kW")
    assert qty.require_unit("kW") == 3.5


def test_quantity_unit_mismatch() -> None:
    qty = Quantity(value=3.5, unit="kW")
    with pytest.raises(UnitError):
        qty.require_unit("kWh")


def test_quantity_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        Quantity(value=1.0, unit="kW", source="invented")  # type: ignore[call-arg]


def test_array_summary_all_nan() -> None:
    summary = summarize_array(np.array([np.nan, np.nan]))
    assert summary.finite_count == 0
    assert summary.min is None


def test_fingerprint_is_deterministic() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])
    assert fingerprint_array(values) == fingerprint_array(values.copy())


def test_run_result_assembles_without_hidden_pipeline() -> None:
    provenance = provenance_record(
        MethodId.WEIGHTED_LINEAR_COMBINATION, parameters={"weights": {"a": 1.0}}
    )
    layer = LayerResult(
        name="suitability",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.WEIGHTED_LINEAR_COMBINATION.value,
        provenance=provenance,
    )
    run = RunResult(
        run_id="abc123",
        status=EvaluationStatus.EVALUATED,
        layers=(layer,),
        provenance=provenance,
    )
    assert run.selected_ids == ()
    assert run.layers[0].name == "suitability"
