"""Errors from suitability ranking and multi-objective analysis."""

from __future__ import annotations

from solardss.exceptions import InvalidParameterError, InvalidWeightsError, SolarDSSError


class SuitabilityError(SolarDSSError):
    """A decision profile or ranking could not be applied as stated."""

    code = "suitability_error"


class DecisionProfileError(SuitabilityError, InvalidWeightsError):
    """A decision profile is missing weight provenance or is inconsistent."""

    code = "decision_profile_error"


class RankingConfigError(SuitabilityError, InvalidParameterError):
    """Ranking or Pareto inputs cannot be used as stated."""

    code = "ranking_config_error"
