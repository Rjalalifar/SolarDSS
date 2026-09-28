"""Weighted overlay and AHP weights."""

from __future__ import annotations

import numpy as np
import pytest

from solardss.exceptions import InvalidParameterError, InvalidWeightsError
from solardss.suitability import ahp_geometric_mean, weighted_linear_combination


def test_weighted_linear_combination() -> None:
    result = weighted_linear_combination(
        {"slope": np.array([0.0, 1.0]), "ghi": np.array([1.0, 0.0])},
        {"slope": 0.4, "ghi": 0.6},
    )
    np.testing.assert_allclose(result.values, [0.6, 0.4])
    assert result.layer.provenance.parameters["weight_mode"] == "as_supplied"


def test_weights_must_sum_to_one_unless_normalized() -> None:
    scores = {"a": np.array([1.0])}
    with pytest.raises(InvalidWeightsError):
        weighted_linear_combination(scores, {"a": 2.0})
    result = weighted_linear_combination(scores, {"a": 2.0}, normalize_weights=True)
    np.testing.assert_allclose(result.values, [1.0])
    weights = result.layer.provenance.parameters["weights"]
    assert isinstance(weights, dict)
    assert weights["a"] == 1.0


def test_mismatched_weight_names() -> None:
    with pytest.raises(InvalidWeightsError):
        weighted_linear_combination({"a": np.array([1.0])}, {"b": 1.0})


def test_ahp_consistent_matrix() -> None:
    matrix = np.array(
        [
            [1.0, 2.0, 4.0],
            [0.5, 1.0, 2.0],
            [0.25, 0.5, 1.0],
        ]
    )
    result = ahp_geometric_mean(matrix, ("slope", "ghi", "grid"))
    np.testing.assert_allclose(result.weights, (4 / 7, 2 / 7, 1 / 7))
    assert result.consistency_ratio is not None
    assert result.consistency_ratio == pytest.approx(0.0, abs=1e-12)


def test_ahp_rejects_non_reciprocal() -> None:
    matrix = np.array([[1.0, 2.0], [2.0, 1.0]])
    with pytest.raises(InvalidParameterError):
        ahp_geometric_mean(matrix, ("a", "b"))


def test_ahp_max_consistency_ratio() -> None:
    matrix = np.array(
        [
            [1.0, 9.0, 1 / 9],
            [1 / 9, 1.0, 9.0],
            [9.0, 1 / 9, 1.0],
        ]
    )
    with pytest.raises(InvalidWeightsError):
        ahp_geometric_mean(matrix, ("a", "b", "c"), max_consistency_ratio=0.1)
