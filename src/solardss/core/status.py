"""Status enumerations for evaluation and constraint screening."""

from __future__ import annotations

from enum import Enum


class EvaluationStatus(str, Enum):
    """Whether a method produced values, and how those values may be used.

    ``INCOMPLETE_INPUTS`` is a result status, not a silent fallback: the library
    never invents missing physical quantities.
    """

    NOT_EVALUATED = "not_evaluated"
    INCOMPLETE_INPUTS = "incomplete_inputs"
    EXCLUDED = "excluded"
    EVALUATED = "evaluated"
    SELECTED = "selected"
    FAILED = "failed"


class FeasibilityStatus(str, Enum):
    """Constraint-screening outcome for a cell or candidate record."""

    UNKNOWN = "unknown"
    FEASIBLE = "feasible"
    EXCLUDED = "excluded"
