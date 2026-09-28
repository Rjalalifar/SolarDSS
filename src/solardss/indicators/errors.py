"""Registry error for invalid indicator vocabulary."""

from __future__ import annotations

from solardss.exceptions import SolarDSSError


class IndicatorRegistryError(SolarDSSError):
    """The indicator registry is inconsistent or a lookup is invalid."""

    code = "indicator_registry_error"
