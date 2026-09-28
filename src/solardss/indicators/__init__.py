"""Indicator vocabulary, registry, and deterministic normalization."""

from __future__ import annotations

from solardss.indicators.catalog import INDICATOR_CATALOG, catalog_ids, get_indicator, registry
from solardss.indicators.contracts import (
    Controllability,
    DecisionMode,
    DerivationMethod,
    IndicatorCategory,
    IndicatorRole,
    IndicatorSpec,
    ModeContext,
    OptimizationDirection,
    Polarity,
    ProjectType,
    ProvenanceRequirements,
    RequirementLevel,
    ResolvedIndicator,
    SiteSelectionScope,
    SourceType,
    ValueProvenanceKind,
    polarity_from_direction,
)
from solardss.indicators.errors import IndicatorRegistryError
from solardss.indicators.ids import IndicatorId
from solardss.indicators.normalize import minmax_normalize
from solardss.indicators.registry import IndicatorRegistry

__all__ = [
    "INDICATOR_CATALOG",
    "Controllability",
    "DecisionMode",
    "DerivationMethod",
    "IndicatorCategory",
    "IndicatorId",
    "IndicatorRegistry",
    "IndicatorRegistryError",
    "IndicatorRole",
    "IndicatorSpec",
    "ModeContext",
    "OptimizationDirection",
    "Polarity",
    "ProjectType",
    "ProvenanceRequirements",
    "RequirementLevel",
    "ResolvedIndicator",
    "SiteSelectionScope",
    "SourceType",
    "ValueProvenanceKind",
    "catalog_ids",
    "get_indicator",
    "minmax_normalize",
    "polarity_from_direction",
    "registry",
]
