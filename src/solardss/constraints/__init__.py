"""Constraint screening from caller-supplied layers and explicit profiles."""

from __future__ import annotations

from solardss.constraints.contracts import (
    AllowedCategoryConstraint,
    BooleanExclusionConstraint,
    CandidateEvaluation,
    ConstraintEffect,
    ConstraintEvaluationSet,
    ConstraintFailure,
    ConstraintKind,
    ConstraintProfile,
    DisallowedCategoryConstraint,
    DistanceConstraint,
    IndicatorSnapshot,
    MaximumConstraint,
    MinimumConstraint,
    OverlapConstraint,
    Threshold,
)
from solardss.constraints.engine import ConstraintEngine
from solardss.constraints.errors import ConstraintConfigError, ConstraintEngineError
from solardss.constraints.profile import (
    EXAMPLE_CONSTRAINT_PROFILES,
    ROOFTOP_DEFAULT_EXAMPLE,
    UTILITY_SCALE_DEFAULT_EXAMPLE,
    example_profile_for,
)
from solardss.constraints.screening import apply_exclusion_columns, apply_exclusions

__all__ = [
    "EXAMPLE_CONSTRAINT_PROFILES",
    "ROOFTOP_DEFAULT_EXAMPLE",
    "UTILITY_SCALE_DEFAULT_EXAMPLE",
    "AllowedCategoryConstraint",
    "BooleanExclusionConstraint",
    "CandidateEvaluation",
    "ConstraintConfigError",
    "ConstraintEffect",
    "ConstraintEngine",
    "ConstraintEngineError",
    "ConstraintEvaluationSet",
    "ConstraintFailure",
    "ConstraintKind",
    "ConstraintProfile",
    "DisallowedCategoryConstraint",
    "DistanceConstraint",
    "IndicatorSnapshot",
    "MaximumConstraint",
    "MinimumConstraint",
    "OverlapConstraint",
    "Threshold",
    "apply_exclusion_columns",
    "apply_exclusions",
    "example_profile_for",
]
