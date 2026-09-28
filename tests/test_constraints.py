"""Constraint screening."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from solardss.constraints import apply_exclusion_columns, apply_exclusions
from solardss.data import validate_candidate_table
from solardss.exceptions import InvalidParameterError


def test_exclusion_or_combine() -> None:
    excluded, feasible, result = apply_exclusions(
        {
            "water": np.array([True, False, False]),
            "protect": np.array([False, False, True]),
        }
    )
    np.testing.assert_array_equal(excluded, [True, False, True])
    np.testing.assert_array_equal(feasible, [False, True, False])
    assert result.excluded_count == 2
    assert result.feasible_count == 1


def test_empty_layers_rejected() -> None:
    with pytest.raises(InvalidParameterError):
        apply_exclusions({})


def test_non_boolean_rejected() -> None:
    with pytest.raises(InvalidParameterError):
        apply_exclusions({"water": np.array([1.0, 0.0])})


def test_explicit_all_false_means_all_feasible() -> None:
    excluded, feasible, result = apply_exclusions({"none": np.array([False, False])})
    np.testing.assert_array_equal(excluded, [False, False])
    np.testing.assert_array_equal(feasible, [True, True])
    assert result.feasible_count == 2


def test_table_exclusion_columns() -> None:
    table = validate_candidate_table(
        pd.DataFrame(
            {
                "candidate_id": ["a", "b"],
                "water": [False, True],
            }
        )
    )
    flags, result = apply_exclusion_columns(table, ("water",))
    np.testing.assert_array_equal(flags.to_numpy(), [False, True])
    assert result.excluded_count == 1
