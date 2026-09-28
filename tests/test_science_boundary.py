"""Guardrails: SolarDSS must not invent physical solar quantities."""

from __future__ import annotations

import inspect

import solardss
from solardss.indicators import get_indicator


def test_irradiation_is_input_not_a_method() -> None:
    assert not hasattr(solardss, "estimate_ghi")
    assert not hasattr(solardss, "estimate_irradiation")
    assert not hasattr(solardss, "pvwatts")
    assert get_indicator("ghi_kwh_m2").computed_by_solardss is False


def test_public_callables_are_documented() -> None:
    for name in solardss.__all__:
        obj = getattr(solardss, name)
        if inspect.isfunction(obj) or inspect.isclass(obj):
            assert obj.__doc__, f"{name} needs a docstring"
