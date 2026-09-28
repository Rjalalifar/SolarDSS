"""Candidate extraction and greedy selection."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solardss.candidates import candidates_from_mask, retain_feasible
from solardss.core import EvaluationStatus
from solardss.data import validate_candidate_table
from solardss.exceptions import SchemaError
from solardss.optimization import greedy_select


def test_candidates_from_mask() -> None:
    mask = np.array([[True, False], [False, True]])
    table, layer = candidates_from_mask(mask)
    assert table.ids() == ("r0c0", "r1c1")
    assert layer.feasible_count == 2
    assert layer.excluded_count == 2


def test_empty_mask_stays_empty() -> None:
    table, layer = candidates_from_mask(np.zeros((2, 2), dtype=bool))
    assert len(table) == 0
    assert layer.feasible_count == 0


def test_retain_feasible() -> None:
    table = validate_candidate_table(
        pd.DataFrame({"candidate_id": ["a", "b", "c"], "score": [1.0, 2.0, 3.0]})
    )
    excluded = pd.Series([False, True, False])
    kept, layer = retain_feasible(table, excluded)
    assert kept.ids() == ("a", "c")
    assert layer.excluded_count == 1


def test_greedy_select_skips_over_budget_items() -> None:
    table = validate_candidate_table(
        pd.DataFrame(
            {
                "candidate_id": ["low", "expensive", "mid"],
                "score": [1.0, 10.0, 5.0],
                "cost": [1.0, 10.0, 2.0],
            }
        )
    )
    result = greedy_select(table, score_column="score", cost_column="cost", budget=3.0)
    assert result.selected_ids == ("mid", "low")
    assert result.skipped_over_budget_ids == ("expensive",)
    assert result.total_cost == pytest.approx(3.0)
    assert result.status is EvaluationStatus.SELECTED
    assert "greedy_by_score_not_global_optimum" in result.warnings


def test_greedy_tie_breaks_by_id() -> None:
    table = validate_candidate_table(
        pd.DataFrame(
            {
                "candidate_id": ["b", "a"],
                "score": [1.0, 1.0],
                "cost": [1.0, 1.0],
            }
        )
    )
    result = greedy_select(table, score_column="score", cost_column="cost", budget=1.0)
    assert result.selected_ids == ("a",)


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(SchemaError):
        validate_candidate_table(pd.DataFrame({"candidate_id": ["a", "a"]}))
