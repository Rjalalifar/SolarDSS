"""Public error hierarchy.

Each error has a stable ``code`` so adapters can map failures without parsing
message text. None of these errors imply a computed scientific result.
"""

from __future__ import annotations


class SolarDSSError(Exception):
    """Base class for all SolarDSS errors."""

    code = "solardss_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class IncompleteInputsError(SolarDSSError):
    """A method was asked to run without every required scientific input."""

    code = "incomplete_inputs"


class InvalidParameterError(SolarDSSError):
    """A caller-supplied parameter is outside the method's domain."""

    code = "invalid_parameter"


class InvalidWeightsError(InvalidParameterError):
    """A weight vector is missing, negative, or does not satisfy the chosen rule."""

    code = "invalid_weights"


class InvalidGridError(InvalidParameterError):
    """An elevation or raster-like grid cannot be used as stated."""

    code = "invalid_grid"


class SchemaError(InvalidParameterError):
    """A table is missing required columns or has inconsistent lengths."""

    code = "schema_error"


class UnitError(InvalidParameterError):
    """A quantity's unit does not match the unit required by the identity."""

    code = "unit_error"


class UnknownIndicatorError(SolarDSSError):
    """The requested indicator id is not in the catalog."""

    code = "unknown_indicator"


class MissingExtraError(SolarDSSError):
    """An optional extra (GIS, ERA5, or pvlib) is required but not installed."""

    code = "missing_extra"


class UnsupportedMethodError(SolarDSSError):
    """The requested physical or decision method is not implemented."""

    code = "unsupported_method"
