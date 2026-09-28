"""Shipped example constraint profiles. They are not universal standards."""

from __future__ import annotations

from solardss.constraints.contracts import (
    ConstraintEffect,
    ConstraintProfile,
    DisallowedCategoryConstraint,
    DistanceConstraint,
    MaximumConstraint,
    MinimumConstraint,
    OverlapConstraint,
    Threshold,
)
from solardss.exceptions import InvalidParameterError
from solardss.indicators.contracts import ProjectType
from solardss.indicators.ids import IndicatorId

_EXAMPLE_SOURCE = "solardss.example_profile"
_EXAMPLE_RATIONALE = (
    "Configurable example threshold for demonstration and tests. "
    "Not a universal scientific standard."
)

_UTILITY_ID = "utility_scale_default_example"
_ROOFTOP_ID = "rooftop_default_example"


def _example_threshold(value: float, unit: str, profile_id: str) -> Threshold:
    return Threshold(
        value=value,
        unit=unit,
        source=_EXAMPLE_SOURCE,
        profile=profile_id,
        rationale=_EXAMPLE_RATIONALE,
    )


UTILITY_SCALE_DEFAULT_EXAMPLE = ConstraintProfile(
    profile_id=_UTILITY_ID,
    label="Utility-scale example profile (not a universal standard)",
    example=True,
    notes=(
        "Configurable EXAMPLE profile only. Thresholds are not a universal "
        "scientific standard and are not applied unless the caller passes this object in.",
        "Replace every bound with a project-specific value, unit, source, and rationale.",
    ),
    constraints=(
        MaximumConstraint(
            constraint_id="example_max_water_fraction",
            indicator_id=IndicatorId.LAND_WATER_FRACTION.value,
            effect=ConstraintEffect.HARD_EXCLUSION,
            source=_EXAMPLE_SOURCE,
            profile=_UTILITY_ID,
            rationale=_EXAMPLE_RATIONALE,
            threshold=_example_threshold(0.1, "1", _UTILITY_ID),
        ),
        OverlapConstraint(
            constraint_id="example_max_protected_overlap",
            indicator_id=IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value,
            effect=ConstraintEffect.HARD_EXCLUSION,
            source=_EXAMPLE_SOURCE,
            profile=_UTILITY_ID,
            rationale=_EXAMPLE_RATIONALE,
            threshold=_example_threshold(0.0, "1", _UTILITY_ID),
        ),
        MaximumConstraint(
            constraint_id="example_max_slope",
            indicator_id=IndicatorId.TERRAIN_SLOPE_MEAN.value,
            effect=ConstraintEffect.HARD_EXCLUSION,
            source=_EXAMPLE_SOURCE,
            profile=_UTILITY_ID,
            rationale=_EXAMPLE_RATIONALE,
            threshold=_example_threshold(15.0, "deg", _UTILITY_ID),
        ),
        MinimumConstraint(
            constraint_id="example_min_available_area",
            indicator_id=IndicatorId.LAND_AVAILABLE_AREA.value,
            effect=ConstraintEffect.HARD_EXCLUSION,
            source=_EXAMPLE_SOURCE,
            profile=_UTILITY_ID,
            rationale=_EXAMPLE_RATIONALE,
            threshold=_example_threshold(5000.0, "m2", _UTILITY_ID),
        ),
        DistanceConstraint(
            constraint_id="example_max_grid_distance",
            indicator_id=IndicatorId.INFRA_DISTANCE_TO_POWER_LINE.value,
            effect=ConstraintEffect.PENALTY,
            source=_EXAMPLE_SOURCE,
            profile=_UTILITY_ID,
            rationale=_EXAMPLE_RATIONALE,
            penalty_weight=1.0,
            maximum=_example_threshold(5000.0, "m", _UTILITY_ID),
        ),
        DisallowedCategoryConstraint(
            constraint_id="example_disallow_water_cover",
            indicator_id=IndicatorId.LAND_COVER.value,
            effect=ConstraintEffect.HARD_EXCLUSION,
            source=_EXAMPLE_SOURCE,
            profile=_UTILITY_ID,
            rationale=_EXAMPLE_RATIONALE,
            disallowed=("water",),
        ),
    ),
)

ROOFTOP_DEFAULT_EXAMPLE = ConstraintProfile(
    profile_id=_ROOFTOP_ID,
    label="Rooftop example profile (not a universal standard)",
    example=True,
    notes=(
        "Configurable EXAMPLE profile only. Thresholds are not a universal "
        "scientific standard and are not applied unless the caller passes this object in.",
        "Rooftop screening still requires project-specific bounds and a legal overlap policy.",
    ),
    constraints=(
        MaximumConstraint(
            constraint_id="example_max_roof_slope",
            indicator_id=IndicatorId.ROOF_SLOPE.value,
            effect=ConstraintEffect.HARD_EXCLUSION,
            source=_EXAMPLE_SOURCE,
            profile=_ROOFTOP_ID,
            rationale=_EXAMPLE_RATIONALE,
            threshold=_example_threshold(45.0, "deg", _ROOFTOP_ID),
        ),
        MinimumConstraint(
            constraint_id="example_min_footprint_area",
            indicator_id=IndicatorId.ROOF_BUILDING_FOOTPRINT_AREA.value,
            effect=ConstraintEffect.HARD_EXCLUSION,
            source=_EXAMPLE_SOURCE,
            profile=_ROOFTOP_ID,
            rationale=_EXAMPLE_RATIONALE,
            threshold=_example_threshold(15.0, "m2", _ROOFTOP_ID),
        ),
        OverlapConstraint(
            constraint_id="example_protected_overlap_warning",
            indicator_id=IndicatorId.LAND_PROTECTED_AREA_OVERLAP.value,
            effect=ConstraintEffect.WARNING,
            source=_EXAMPLE_SOURCE,
            profile=_ROOFTOP_ID,
            rationale=_EXAMPLE_RATIONALE,
            threshold=_example_threshold(0.0, "1", _ROOFTOP_ID),
        ),
    ),
)

EXAMPLE_CONSTRAINT_PROFILES: dict[str, ConstraintProfile] = {
    UTILITY_SCALE_DEFAULT_EXAMPLE.profile_id: UTILITY_SCALE_DEFAULT_EXAMPLE,
    ROOFTOP_DEFAULT_EXAMPLE.profile_id: ROOFTOP_DEFAULT_EXAMPLE,
}


def example_profile_for(project_type: ProjectType | str) -> ConstraintProfile:
    """Return a labelled example profile. SolarDSS never applies it automatically."""

    token = project_type.value if isinstance(project_type, ProjectType) else project_type
    if token in {
        ProjectType.UTILITY_SCALE.value,
        ProjectType.GROUND_MOUNTED.value,
    }:
        return UTILITY_SCALE_DEFAULT_EXAMPLE
    if token in {
        ProjectType.ROOFTOP.value,
        ProjectType.COMMERCIAL_ROOFTOP.value,
        ProjectType.RESIDENTIAL_ROOFTOP.value,
    }:
        return ROOFTOP_DEFAULT_EXAMPLE
    raise InvalidParameterError(f"no example constraint profile for project_type {token!r}")
