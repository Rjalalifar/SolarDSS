"""Errors from the hard/soft constraint engine."""

from __future__ import annotations

from solardss.exceptions import InvalidParameterError, SolarDSSError


class ConstraintEngineError(SolarDSSError):
    """A constraint profile or evaluation could not be applied as stated."""

    code = "constraint_engine_error"


class ConstraintConfigError(ConstraintEngineError, InvalidParameterError):
    """A constraint object is missing required threshold metadata or is inconsistent."""

    code = "constraint_config_error"
