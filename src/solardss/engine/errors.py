"""Errors raised by the indicator computation engine."""

from __future__ import annotations

from solardss.exceptions import InvalidParameterError, SolarDSSError


class IndicatorEngineError(SolarDSSError):
    """The engine could not assemble a valid computation."""

    code = "indicator_engine_error"


class CircularDependencyError(IndicatorEngineError, InvalidParameterError):
    """The indicator dependency graph contains a cycle."""

    code = "circular_indicator_dependency"

    def __init__(self, message: str, *, cycle: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.cycle = cycle
