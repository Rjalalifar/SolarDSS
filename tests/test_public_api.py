"""Public package surface and independence from agent/LLM stacks."""

from __future__ import annotations

import ast
from pathlib import Path

import solardss

FORBIDDEN_MODULES = {
    "ariadne",
    "ariadne_thread",
    "langchain",
    "langgraph",
    "openai",
    "ollama",
    "anthropic",
    "fastapi",
    "flask",
    "streamlit",
    "dash",
    "gradio",
    "qdrant_client",
    "chromadb",
    "requests",
}

SRC_ROOT = Path(solardss.__file__).resolve().parent


def test_version_is_stable() -> None:
    assert solardss.__version__ == "0.1.0"


def test_public_api_exports() -> None:
    for name in (
        "EvaluationStatus",
        "FeasibilityStatus",
        "Quantity",
        "LayerResult",
        "RunResult",
        "minmax_normalize",
        "weighted_linear_combination",
        "rank_candidates",
        "RankingResult",
        "ahp_geometric_mean",
        "apply_exclusions",
        "ConstraintEngine",
        "ConstraintProfile",
        "candidates_from_mask",
        "generate_candidates",
        "greedy_select",
        "horn_slope_aspect",
        "energy_from_capacity_factor",
        "PVComponentCatalog",
        "PVDesignEngine",
        "PVDesignProblem",
        "PVDesignResult",
        "PVGISYieldEngine",
        "PVSystemConfiguration",
        "PVYieldEngine",
        "PVYieldRequest",
        "PVYieldResult",
        "PvlibYieldEngine",
        "PVModule",
        "PVInverter",
        "BatterySystem",
        "empty_pv_catalog",
        "load_pv_catalog",
        "provenance_record",
        "INDICATOR_CATALOG",
        "registry",
        "DecisionMode",
        "SolarDecisionProblem",
        "get_missing_requirements",
        "IndicatorObservation",
        "IndicatorEngine",
        "DataProvenance",
        "SpatialCoverage",
        "TemporalCoverage",
        "ProviderAvailability",
    ):
        assert hasattr(solardss, name)


def test_source_has_no_forbidden_imports() -> None:
    found: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in FORBIDDEN_MODULES:
                    found.append(f"{path.name}:{name}")
    assert found == []


def test_httpx_is_confined_to_pvgis_adapter() -> None:
    found: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            if "httpx" in names and path.name not in {"pvgis.py", "osm.py"}:
                found.append(path.name)
    assert found == []


def test_xarray_is_confined_to_era5_adapter() -> None:
    found: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            if "xarray" in names and path.name != "era5_land.py":
                found.append(path.name)
    assert found == []


def test_pvlib_is_confined_to_pvlib_yield_engine() -> None:
    found: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            if "pvlib" in names and path.name != "pvlib_yield.py":
                found.append(path.name)
    assert found == []


def test_rasterio_shapely_pyproj_are_confined_to_raster_adapters() -> None:
    found: list[str] = []
    confined = {"rasterio", "shapely", "pyproj", "geopandas"}
    allowed = {"dem.py", "worldcover.py", "protected.py", "hazard.py", "clip.py"}
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name.split(".")[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            for name in names:
                if name in confined and path.name not in allowed:
                    found.append(f"{path.name}:{name}")
    assert found == []
