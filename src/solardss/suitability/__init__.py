"""Suitability scoring: weighted overlay, AHP weights, WSM ranking, and Pareto sets."""

from __future__ import annotations

from solardss.suitability.ahp import SAATY_RANDOM_INDEX, AhpResult, ahp_geometric_mean
from solardss.suitability.contracts import (
    CriterionContribution,
    CriterionWeight,
    DecisionProfile,
    MultiObjectiveResult,
    NormalizationKind,
    NormalizationStats,
    ObjectiveSpec,
    ObjectiveValue,
    OmissionReason,
    OmittedCandidate,
    ParetoPoint,
    RankedCandidate,
    RankingResult,
)
from solardss.suitability.errors import DecisionProfileError, RankingConfigError, SuitabilityError
from solardss.suitability.normalize import normalize_criterion
from solardss.suitability.pareto import pareto_candidates
from solardss.suitability.profiles import (
    BALANCED,
    ENERGY_FOCUSED,
    ENVIRONMENT_FOCUSED,
    EXAMPLE_DECISION_PROFILES,
    INFRASTRUCTURE_FOCUSED,
)
from solardss.suitability.ranking import SuitabilityEngine, rank_candidates
from solardss.suitability.weighted import weighted_linear_combination

__all__ = [
    "BALANCED",
    "ENERGY_FOCUSED",
    "ENVIRONMENT_FOCUSED",
    "EXAMPLE_DECISION_PROFILES",
    "INFRASTRUCTURE_FOCUSED",
    "SAATY_RANDOM_INDEX",
    "AhpResult",
    "CriterionContribution",
    "CriterionWeight",
    "DecisionProfile",
    "DecisionProfileError",
    "MultiObjectiveResult",
    "NormalizationKind",
    "NormalizationStats",
    "ObjectiveSpec",
    "ObjectiveValue",
    "OmissionReason",
    "OmittedCandidate",
    "ParetoPoint",
    "RankedCandidate",
    "RankingConfigError",
    "RankingResult",
    "SuitabilityEngine",
    "SuitabilityError",
    "ahp_geometric_mean",
    "normalize_criterion",
    "pareto_candidates",
    "rank_candidates",
    "weighted_linear_combination",
]
