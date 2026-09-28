"""Decision-problem modeling for SolarDSS."""

from __future__ import annotations

from solardss.data.geo import BoundingBox, GeographicRef
from solardss.decision.contracts import (
    CandidateSet,
    ConstraintOperator,
    DecisionOperation,
    EnvironmentalPreferences,
    MissingRequirement,
    ModulePreferences,
    UserConstraint,
)
from solardss.decision.errors import DecisionProblemError, OperationNotAllowedError
from solardss.decision.problem import SolarDecisionProblem
from solardss.decision.requirements import get_missing_requirements, missing_requirement_records

__all__ = [
    "BoundingBox",
    "CandidateSet",
    "ConstraintOperator",
    "DecisionOperation",
    "DecisionProblemError",
    "EnvironmentalPreferences",
    "GeographicRef",
    "MissingRequirement",
    "ModulePreferences",
    "OperationNotAllowedError",
    "SolarDecisionProblem",
    "UserConstraint",
    "get_missing_requirements",
    "missing_requirement_records",
]
