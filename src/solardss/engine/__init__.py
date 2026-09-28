"""Indicator computation engine: DAG, provider selection, and IndicatorSet."""

from __future__ import annotations

from solardss.engine.contracts import (
    BatchScope,
    CandidateIndicatorSet,
    Dependency,
    DependencyKind,
    EngineContext,
    IndicatorNode,
    IndicatorOutcome,
    IndicatorResult,
    IndicatorSet,
    ResolutionPath,
)
from solardss.engine.dependencies import DEFAULT_GRAPH, build_default_graph
from solardss.engine.derive import CalculatorRegistry, DerivedCalculator, derived_observation
from solardss.engine.engine import IndicatorEngine
from solardss.engine.errors import CircularDependencyError, IndicatorEngineError
from solardss.engine.graph import IndicatorGraph
from solardss.engine.providers import ProviderRegistry

__all__ = [
    "DEFAULT_GRAPH",
    "BatchScope",
    "CalculatorRegistry",
    "CandidateIndicatorSet",
    "CircularDependencyError",
    "Dependency",
    "DependencyKind",
    "DerivedCalculator",
    "EngineContext",
    "IndicatorEngine",
    "IndicatorEngineError",
    "IndicatorGraph",
    "IndicatorNode",
    "IndicatorOutcome",
    "IndicatorResult",
    "IndicatorSet",
    "ProviderRegistry",
    "ResolutionPath",
    "build_default_graph",
    "derived_observation",
]
