"""Terrain derivatives from caller-supplied elevation grids."""

from __future__ import annotations

from solardss.terrain.analysis import (
    TerrainAnalysisResult,
    analyze_terrain,
    circular_mean_deg,
    horizon_obstruction,
    riley_tri,
    shading_potential,
    zonal_statistics,
)
from solardss.terrain.contracts import ElevationGrid
from solardss.terrain.horn import SlopeAspectResult, horn_slope_aspect

__all__ = [
    "ElevationGrid",
    "SlopeAspectResult",
    "TerrainAnalysisResult",
    "analyze_terrain",
    "circular_mean_deg",
    "horizon_obstruction",
    "horn_slope_aspect",
    "riley_tri",
    "shading_potential",
    "zonal_statistics",
]
