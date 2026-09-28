"""Assembled indicator vocabulary."""

from __future__ import annotations

from solardss.indicators.contracts import IndicatorSpec
from solardss.indicators.definitions.climate import CLIMATE_INDICATORS
from solardss.indicators.definitions.design import DESIGN_INDICATORS
from solardss.indicators.definitions.hazard import HAZARD_INDICATORS
from solardss.indicators.definitions.infrastructure import INFRASTRUCTURE_INDICATORS
from solardss.indicators.definitions.land import LAND_INDICATORS
from solardss.indicators.definitions.objectives import OBJECTIVE_INDICATORS
from solardss.indicators.definitions.rooftop import ROOFTOP_INDICATORS
from solardss.indicators.definitions.solar import SOLAR_INDICATORS
from solardss.indicators.definitions.terrain import TERRAIN_INDICATORS

ALL_INDICATORS: tuple[IndicatorSpec, ...] = (
    *SOLAR_INDICATORS,
    *CLIMATE_INDICATORS,
    *TERRAIN_INDICATORS,
    *LAND_INDICATORS,
    *INFRASTRUCTURE_INDICATORS,
    *HAZARD_INDICATORS,
    *ROOFTOP_INDICATORS,
    *DESIGN_INDICATORS,
    *OBJECTIVE_INDICATORS,
)
