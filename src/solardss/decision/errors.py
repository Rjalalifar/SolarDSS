"""Errors for decision-problem construction and disallowed operations."""

from __future__ import annotations

from solardss.exceptions import SolarDSSError, UnsupportedMethodError


class DecisionProblemError(SolarDSSError):
    """The decision problem is contradictory or not usable as stated."""

    code = "decision_problem_error"


class OperationNotAllowedError(UnsupportedMethodError):
    """An operation is not part of the stated decision mode/scope.

    Fixed-site problems must not run candidate generation or site ranking.
    """

    code = "operation_not_allowed"
