"""Shipped example decision profiles. They are configuration, not universal truth."""

from __future__ import annotations

from solardss.indicators.ids import IndicatorId
from solardss.suitability.contracts import (
    CriterionWeight,
    DecisionProfile,
    NormalizationKind,
)

_EXAMPLE_SOURCE = "solardss.example_profile"
_EXAMPLE_RATIONALE = (
    "Configurable example weight for demonstration and tests. "
    "Not universal truth and not produced by an LLM."
)

_ENERGY_ID = "energy_focused"
_INFRA_ID = "infrastructure_focused"
_ENV_ID = "environment_focused"
_BALANCED_ID = "balanced"

_EXAMPLE_NOTES = (
    "Configurable EXAMPLE profile only. Weights are not universal truth "
    "and are not applied unless the caller passes this object in.",
    "Replace every weight with a project-specific value, source, and rationale.",
)


def _weight(
    indicator_id: str,
    weight: float,
    unit: str,
    profile_id: str,
    normalization: NormalizationKind,
) -> CriterionWeight:
    return CriterionWeight(
        indicator_id=indicator_id,
        weight=weight,
        unit=unit,
        source=_EXAMPLE_SOURCE,
        profile=profile_id,
        rationale=_EXAMPLE_RATIONALE,
        normalization=normalization,
    )


ENERGY_FOCUSED = DecisionProfile(
    profile_id=_ENERGY_ID,
    label="Energy-focused example profile (not universal truth)",
    example=True,
    notes=_EXAMPLE_NOTES,
    criteria=(
        _weight(
            IndicatorId.SOLAR_GHI_ANNUAL.value,
            0.55,
            "kWh/m2/year",
            _ENERGY_ID,
            NormalizationKind.BENEFIT,
        ),
        _weight(
            IndicatorId.TERRAIN_SLOPE_MEAN.value,
            0.15,
            "deg",
            _ENERGY_ID,
            NormalizationKind.COST,
        ),
        _weight(
            IndicatorId.INFRA_DISTANCE_TO_POWER_LINE.value,
            0.15,
            "m",
            _ENERGY_ID,
            NormalizationKind.COST,
        ),
        _weight(
            IndicatorId.LAND_AVAILABLE_AREA.value,
            0.15,
            "m2",
            _ENERGY_ID,
            NormalizationKind.BENEFIT,
        ),
    ),
)

INFRASTRUCTURE_FOCUSED = DecisionProfile(
    profile_id=_INFRA_ID,
    label="Infrastructure-focused example profile (not universal truth)",
    example=True,
    notes=_EXAMPLE_NOTES,
    criteria=(
        _weight(
            IndicatorId.INFRA_DISTANCE_TO_POWER_LINE.value,
            0.45,
            "m",
            _INFRA_ID,
            NormalizationKind.COST,
        ),
        _weight(
            IndicatorId.INFRA_DISTANCE_TO_ROAD.value,
            0.25,
            "m",
            _INFRA_ID,
            NormalizationKind.COST,
        ),
        _weight(
            IndicatorId.SOLAR_GHI_ANNUAL.value,
            0.20,
            "kWh/m2/year",
            _INFRA_ID,
            NormalizationKind.BENEFIT,
        ),
        _weight(
            IndicatorId.TERRAIN_SLOPE_MEAN.value,
            0.10,
            "deg",
            _INFRA_ID,
            NormalizationKind.COST,
        ),
    ),
)

ENVIRONMENT_FOCUSED = DecisionProfile(
    profile_id=_ENV_ID,
    label="Environment-focused example profile (not universal truth)",
    example=True,
    notes=_EXAMPLE_NOTES,
    criteria=(
        _weight(
            IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value,
            0.40,
            "1",
            _ENV_ID,
            NormalizationKind.COST,
        ),
        _weight(
            IndicatorId.LAND_WATER_FRACTION.value,
            0.20,
            "1",
            _ENV_ID,
            NormalizationKind.COST,
        ),
        _weight(
            IndicatorId.TERRAIN_SLOPE_MEAN.value,
            0.15,
            "deg",
            _ENV_ID,
            NormalizationKind.COST,
        ),
        _weight(
            IndicatorId.SOLAR_GHI_ANNUAL.value,
            0.15,
            "kWh/m2/year",
            _ENV_ID,
            NormalizationKind.BENEFIT,
        ),
        _weight(
            IndicatorId.LAND_FOREST_FRACTION.value,
            0.10,
            "1",
            _ENV_ID,
            NormalizationKind.COST,
        ),
    ),
)

BALANCED = DecisionProfile(
    profile_id=_BALANCED_ID,
    label="Balanced example profile (not universal truth)",
    example=True,
    notes=_EXAMPLE_NOTES,
    criteria=(
        _weight(
            IndicatorId.SOLAR_GHI_ANNUAL.value,
            0.25,
            "kWh/m2/year",
            _BALANCED_ID,
            NormalizationKind.BENEFIT,
        ),
        _weight(
            IndicatorId.TERRAIN_SLOPE_MEAN.value,
            0.20,
            "deg",
            _BALANCED_ID,
            NormalizationKind.COST,
        ),
        _weight(
            IndicatorId.INFRA_DISTANCE_TO_POWER_LINE.value,
            0.20,
            "m",
            _BALANCED_ID,
            NormalizationKind.COST,
        ),
        _weight(
            IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value,
            0.20,
            "1",
            _BALANCED_ID,
            NormalizationKind.COST,
        ),
        _weight(
            IndicatorId.LAND_AVAILABLE_AREA.value,
            0.15,
            "m2",
            _BALANCED_ID,
            NormalizationKind.BENEFIT,
        ),
    ),
)

EXAMPLE_DECISION_PROFILES: dict[str, DecisionProfile] = {
    ENERGY_FOCUSED.profile_id: ENERGY_FOCUSED,
    INFRASTRUCTURE_FOCUSED.profile_id: INFRASTRUCTURE_FOCUSED,
    ENVIRONMENT_FOCUSED.profile_id: ENVIRONMENT_FOCUSED,
    BALANCED.profile_id: BALANCED,
}
