"""Errors from PV configuration generation and evaluation."""

from __future__ import annotations

from solardss.exceptions import InvalidParameterError, SolarDSSError


class PVDesignError(SolarDSSError):
    """A PV design configuration could not be generated or evaluated as stated."""

    code = "pv_design_error"


class PVDesignConfigError(PVDesignError, InvalidParameterError):
    """The design problem, search space, or configuration is inconsistent."""

    code = "pv_design_config_error"
