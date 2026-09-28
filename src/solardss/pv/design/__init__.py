"""PV configuration generation and evaluation after a site is resolved."""

from __future__ import annotations

from solardss.pv.design.contracts import (
    ConfigurationSearchSpace,
    DerivedDesignQuantities,
    DesignConstraintFailure,
    DesignConstraintKind,
    MountingType,
    PVDesignProblem,
    PVDesignResult,
    PVDesignSet,
    PVSystemConfiguration,
    TrackingMode,
    is_ground_project,
    is_rooftop_project,
    mounting_for,
)
from solardss.pv.design.engine import PVDesignEngine, design_problem_from_decision
from solardss.pv.design.errors import PVDesignConfigError, PVDesignError
from solardss.pv.design.evaluate import evaluate_configuration
from solardss.pv.design.generator import (
    KNOWN_FUTURE_OPTIMIZERS,
    BoundedEnumerationGenerator,
    PVConfigurationGenerator,
    UnimplementedOptimizer,
)

__all__ = [
    "KNOWN_FUTURE_OPTIMIZERS",
    "BoundedEnumerationGenerator",
    "ConfigurationSearchSpace",
    "DerivedDesignQuantities",
    "DesignConstraintFailure",
    "DesignConstraintKind",
    "MountingType",
    "PVConfigurationGenerator",
    "PVDesignConfigError",
    "PVDesignEngine",
    "PVDesignError",
    "PVDesignProblem",
    "PVDesignResult",
    "PVDesignSet",
    "PVSystemConfiguration",
    "TrackingMode",
    "UnimplementedOptimizer",
    "design_problem_from_decision",
    "evaluate_configuration",
    "is_ground_project",
    "is_rooftop_project",
    "mounting_for",
]
