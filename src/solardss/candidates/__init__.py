"""Candidate generation from masks, grids, and supplied features."""

from __future__ import annotations

from solardss.candidates.contracts import (
    BoundaryGeometry,
    CandidateCollection,
    CandidateFeature,
    CandidateGenerationConfig,
    CandidateGeometry,
    CandidateSource,
    GenerationStrategy,
    InputFeature,
    PolygonPart,
)
from solardss.candidates.errors import (
    CandidateClipError,
    CandidateGenerationError,
    CandidateGridError,
)
from solardss.candidates.features import (
    candidates_from_buildings,
    close_input_polygon,
    input_from_building,
)
from solardss.candidates.generate import candidates_from_mask, retain_feasible
from solardss.candidates.group import contiguous_groups
from solardss.candidates.spatial import generate_candidates, generate_for_problem

__all__ = [
    "BoundaryGeometry",
    "CandidateClipError",
    "CandidateCollection",
    "CandidateFeature",
    "CandidateGenerationConfig",
    "CandidateGenerationError",
    "CandidateGeometry",
    "CandidateGridError",
    "CandidateSource",
    "GenerationStrategy",
    "InputFeature",
    "PolygonPart",
    "candidates_from_buildings",
    "candidates_from_mask",
    "close_input_polygon",
    "contiguous_groups",
    "generate_candidates",
    "generate_for_problem",
    "input_from_building",
    "retain_feasible",
]
