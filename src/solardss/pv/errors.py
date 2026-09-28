"""Errors from the PV catalog, design, and yield engines."""

from __future__ import annotations

from solardss.exceptions import InvalidParameterError, SchemaError, SolarDSSError


class PVCatalogError(SolarDSSError):
    """A PV component catalog could not be loaded or queried as stated."""

    code = "pv_catalog_error"


class PVCatalogConfigError(PVCatalogError, InvalidParameterError):
    """A component record is missing required fields or is internally inconsistent."""

    code = "pv_catalog_config_error"


class PVCatalogSchemaError(PVCatalogError, SchemaError):
    """A catalog file does not match the expected JSON or CSV layout."""

    code = "pv_catalog_schema_error"


class UnknownComponentError(PVCatalogError):
    """The requested component id is not in the catalog."""

    code = "unknown_component"


class PVYieldError(SolarDSSError):
    """A PV yield evaluation could not be completed as stated."""

    code = "pv_yield_error"


class PVYieldConfigError(PVYieldError, InvalidParameterError):
    """A yield request is outside the method's domain."""

    code = "pv_yield_config_error"
