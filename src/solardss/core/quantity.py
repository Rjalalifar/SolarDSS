"""Dimensioned scalars. SolarDSS labels units; it is not a unit-conversion library."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from solardss.exceptions import UnitError


class Quantity(BaseModel):
    """A finite scalar with an explicit unit string.

    Units are compared as exact strings (``kW`` is not ``kWh``). Callers must
    convert before invoking a method.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float
    unit: str = Field(min_length=1)

    def require_unit(self, unit: str) -> float:
        """Return ``value`` if ``unit`` matches, otherwise raise ``UnitError``."""

        if self.unit != unit:
            msg = f"Expected unit {unit!r}, received {self.unit!r}"
            raise UnitError(msg)
        return self.value
