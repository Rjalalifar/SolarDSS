"""PV identities require fully specified inputs."""

from __future__ import annotations

import pytest

from solardss.core import Quantity
from solardss.exceptions import InvalidParameterError, UnitError
from solardss.pv import energy_from_capacity_factor, energy_from_irradiation


def test_capacity_factor_identity() -> None:
    energy, layer = energy_from_capacity_factor(
        Quantity(value=100.0, unit="kW"),
        0.2,
        Quantity(value=10.0, unit="h"),
    )
    assert energy.unit == "kWh"
    assert energy.value == pytest.approx(200.0)
    assert layer.provenance.parameters["identity"] == "E = P_rated * CF * t"


def test_capacity_factor_rejects_out_of_range() -> None:
    with pytest.raises(InvalidParameterError):
        energy_from_capacity_factor(
            Quantity(value=1.0, unit="kW"),
            1.2,
            Quantity(value=1.0, unit="h"),
        )


def test_capacity_factor_rejects_wrong_units() -> None:
    with pytest.raises(UnitError):
        energy_from_capacity_factor(
            Quantity(value=1.0, unit="MW"),
            0.2,
            Quantity(value=1.0, unit="h"),
        )


def test_irradiation_identity() -> None:
    energy, layer = energy_from_irradiation(
        Quantity(value=5.0, unit="kWh/m2"),
        Quantity(value=10.0, unit="m2"),
        0.2,
    )
    assert energy.value == pytest.approx(10.0)
    assert layer.provenance.parameters["identity"] == "E = G * A * eta"
