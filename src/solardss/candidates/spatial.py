"""Materialise candidates for the three SITE_SELECTION scopes.

REGION_SEARCH and AOI_SEARCH tessellate a boundary with a configured grid and
clip to that boundary. CANDIDATE_RANKING copies supplied features and never
creates new locations. Nothing here ranks or scores sites.
"""

from __future__ import annotations

from collections.abc import Sequence

from solardss.candidates.clip import clip_parts_to_boundary, require_clip_possible
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
from solardss.candidates.errors import CandidateGenerationError
from solardss.candidates.features import adopt_features
from solardss.candidates.geometry import (
    boundary_area_m2,
    boundary_from_ref,
    geometry_area_m2,
    geometry_fingerprint,
    require_grid_units,
    require_same_crs,
)
from solardss.candidates.grid import iter_covering_cells
from solardss.core.hashing import fingerprint_canonical, fingerprint_sequence
from solardss.core.methods import MethodId
from solardss.core.quantity import Quantity
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.decision.contracts import DecisionOperation
from solardss.decision.problem import SolarDecisionProblem
from solardss.exceptions import IncompleteInputsError, InvalidParameterError
from solardss.indicators.contracts import DecisionMode, ProjectType, SiteSelectionScope
from solardss.provenance.record import provenance_record

_GRID_SOURCE = {
    GenerationStrategy.REGULAR_GRID: CandidateSource.REGULAR_GRID,
    GenerationStrategy.HEXAGONAL_GRID: CandidateSource.HEXAGONAL_GRID,
}
_GRID_METHOD = {
    GenerationStrategy.REGULAR_GRID: MethodId.CANDIDATES_REGULAR_GRID,
    GenerationStrategy.HEXAGONAL_GRID: MethodId.CANDIDATES_HEXAGONAL_GRID,
}
_DEFAULT_ID = {
    GenerationStrategy.REGULAR_GRID: "r{row}c{col}",
    GenerationStrategy.HEXAGONAL_GRID: "q{col}r{row}",
}


def generate_candidates(
    *,
    scope: SiteSelectionScope | str,
    config: CandidateGenerationConfig,
    boundary: BoundaryGeometry | BoundingBox | GeographicRef | None = None,
    features: Sequence[InputFeature] = (),
    problem: SolarDecisionProblem | None = None,
) -> CandidateCollection:
    """Build candidates for one site-selection scope."""

    wanted = _as_scope(scope)
    _validate_problem(problem, wanted)
    if wanted is SiteSelectionScope.CANDIDATE_RANKING:
        return _from_features(config, features, problem=problem)
    return _from_grid(wanted, config, boundary, problem=problem)


def generate_for_problem(
    problem: SolarDecisionProblem,
    config: CandidateGenerationConfig,
    *,
    boundary: BoundaryGeometry | BoundingBox | GeographicRef | None = None,
    features: Sequence[InputFeature] = (),
) -> CandidateCollection:
    """Dispatch on ``problem.site_selection_scope``."""

    if problem.mode is DecisionMode.FIXED_SITE:
        problem.require_operation(DecisionOperation.CANDIDATE_GENERATION)
    if problem.site_selection_scope is None:
        raise InvalidParameterError("generate_for_problem requires a site-selection scope")
    resolved = boundary
    if resolved is None:
        scope = problem.site_selection_scope
        if scope is SiteSelectionScope.REGION_SEARCH and problem.region is not None:
            resolved = problem.region
        elif scope is SiteSelectionScope.AOI_SEARCH and problem.aoi is not None:
            resolved = problem.aoi
    return generate_candidates(
        scope=problem.site_selection_scope,
        config=config,
        boundary=resolved,
        features=features,
        problem=problem,
    )


def _validate_problem(problem: SolarDecisionProblem | None, scope: SiteSelectionScope) -> None:
    if problem is None:
        return
    if problem.mode is DecisionMode.FIXED_SITE:
        problem.require_operation(DecisionOperation.CANDIDATE_GENERATION)
    if problem.site_selection_scope is not None and problem.site_selection_scope is not scope:
        raise InvalidParameterError(
            f"scope {scope.value} does not match problem scope {problem.site_selection_scope.value}"
        )
    if scope is SiteSelectionScope.CANDIDATE_RANKING:
        return
    problem.require_operation(DecisionOperation.CANDIDATE_GENERATION)


def _from_features(
    config: CandidateGenerationConfig,
    features: Sequence[InputFeature],
    *,
    problem: SolarDecisionProblem | None,
) -> CandidateCollection:
    if config.strategy is not GenerationStrategy.FEATURES:
        raise InvalidParameterError(
            "CANDIDATE_RANKING uses supplied features and must not run a grid strategy"
        )
    adopted = adopt_features(features, crs=config.crs)
    warnings: list[str] = []
    if problem is not None and _is_rooftop(problem.project_type):
        non_roof = [
            item.candidate_id
            for item in adopted
            if item.source is not CandidateSource.BUILDING_FOOTPRINT
        ]
        if non_roof:
            warnings.append(
                "rooftop problems typically use building footprints as candidate geometries"
            )
    collection_prov = provenance_record(
        MethodId.CANDIDATES_FROM_FEATURES,
        parameters={
            "scope": SiteSelectionScope.CANDIDATE_RANKING.value,
            "crs": config.crs,
            "count": len(adopted),
        },
        input_fingerprints=(fingerprint_sequence([item.candidate_id for item in adopted]),),
        notes=("exactly the supplied features; no new locations",),
    )
    return CandidateCollection(
        scope=SiteSelectionScope.CANDIDATE_RANKING,
        strategy=GenerationStrategy.FEATURES,
        crs=config.crs,
        candidates=adopted,
        provenance=collection_prov,
        covering_count=len(adopted),
        dropped_count=0,
        warnings=tuple(warnings),
    )


def _from_grid(
    scope: SiteSelectionScope,
    config: CandidateGenerationConfig,
    boundary: BoundaryGeometry | BoundingBox | GeographicRef | None,
    *,
    problem: SolarDecisionProblem | None,
) -> CandidateCollection:
    if config.strategy not in {GenerationStrategy.REGULAR_GRID, GenerationStrategy.HEXAGONAL_GRID}:
        raise InvalidParameterError(
            f"{scope.value} requires regular_grid or hexagonal_grid; "
            "FEATURES is only valid for candidate_ranking"
        )
    if config.cell_size is None or config.cell_size_unit is None:
        raise InvalidParameterError("grid generation requires cell_size and cell_size_unit")
    require_grid_units(config.crs, config.cell_size_unit)
    clip_boundary = _as_boundary(boundary, config.crs)
    require_clip_possible(clip_boundary)
    template = config.id_template or _DEFAULT_ID[config.strategy]
    _validate_template(config.strategy, template)
    method_id = _GRID_METHOD[config.strategy]
    source = _GRID_SOURCE[config.strategy]
    warnings: list[str] = []
    if problem is not None and _is_rooftop(problem.project_type):
        warnings.append(
            "rooftop problems typically use building footprints rather than a search grid"
        )
    boundary_area = boundary_area_m2(clip_boundary)[0]
    if boundary_area <= 0.0:
        warnings.append("search boundary has no area; no candidates were generated")
        return _empty_grid(scope, config, method_id, warnings)
    covering = 0
    dropped = 0
    built: list[CandidateFeature] = []
    seen: set[str] = set()
    for cell in iter_covering_cells(
        clip_boundary,
        strategy=config.strategy,
        cell_size=config.cell_size,
        origin_x=config.origin_x,
        origin_y=config.origin_y,
        max_candidates=config.max_candidates,
    ):
        covering += 1
        parts = clip_parts_to_boundary((PolygonPart(exterior=cell.exterior),), clip_boundary)
        if not parts:
            dropped += 1
            continue
        geometry = CandidateGeometry(crs=config.crs, parts=parts)
        area, area_method = geometry_area_m2(geometry)
        if area <= 0.0:
            dropped += 1
            continue
        if config.min_area_m2 is not None and area < config.min_area_m2:
            dropped += 1
            continue
        ident = template.format(row=cell.row, col=cell.col, q=cell.col, r=cell.row)
        if ident in seen:
            raise CandidateGenerationError(f"duplicate generated candidate_id {ident!r}")
        seen.add(ident)
        built.append(
            CandidateFeature(
                candidate_id=ident,
                geometry=geometry,
                area=Quantity(value=area, unit="m2"),
                source=source,
                source_id=ident,
                metadata={"grid_row": cell.row, "grid_col": cell.col},
                provenance=provenance_record(
                    method_id,
                    parameters={
                        "row": cell.row,
                        "col": cell.col,
                        "area_method": area_method,
                        "cell_size": config.cell_size,
                        "cell_size_unit": config.cell_size_unit,
                    },
                    input_fingerprints=(geometry_fingerprint(geometry),),
                ),
                grid_row=cell.row,
                grid_col=cell.col,
            )
        )
    built.sort(key=lambda item: (item.grid_row or 0, item.grid_col or 0, item.candidate_id))
    if config.max_candidates is not None and len(built) > config.max_candidates:
        raise CandidateGenerationError(
            f"clipped grid produced {len(built)} candidates, "
            f"above max_candidates={config.max_candidates}"
        )
    collection_prov = provenance_record(
        method_id,
        parameters={
            "scope": scope.value,
            "crs": config.crs,
            "cell_size": config.cell_size,
            "cell_size_unit": config.cell_size_unit,
            "covering_count": covering,
            "kept": len(built),
            "dropped": dropped,
            "boundary": fingerprint_canonical(
                {"crs": clip_boundary.crs, "exterior": [list(pt) for pt in clip_boundary.exterior]}
            ),
        },
        input_fingerprints=(fingerprint_sequence([item.candidate_id for item in built]),),
        notes=("clipped to the search boundary; not ranked",),
    )
    return CandidateCollection(
        scope=scope,
        strategy=config.strategy,
        crs=config.crs,
        candidates=tuple(built),
        provenance=collection_prov,
        covering_count=covering,
        dropped_count=dropped,
        warnings=tuple(warnings),
    )


def _empty_grid(
    scope: SiteSelectionScope,
    config: CandidateGenerationConfig,
    method_id: MethodId,
    warnings: list[str],
) -> CandidateCollection:
    record = provenance_record(
        method_id,
        parameters={
            "scope": scope.value,
            "crs": config.crs,
            "cell_size": config.cell_size if config.cell_size is not None else 0.0,
            "cell_size_unit": config.cell_size_unit or "",
            "covering_count": 0,
            "kept": 0,
            "dropped": 0,
        },
        notes=("clipped to the search boundary; not ranked", *warnings),
    )
    return CandidateCollection(
        scope=scope,
        strategy=config.strategy,
        crs=config.crs,
        candidates=(),
        provenance=record,
        covering_count=0,
        dropped_count=0,
        warnings=tuple(warnings),
    )


def _as_boundary(
    boundary: BoundaryGeometry | BoundingBox | GeographicRef | None, crs: str
) -> BoundaryGeometry:
    if boundary is None:
        raise IncompleteInputsError("region/AOI search requires a boundary geometry")
    if isinstance(boundary, BoundingBox):
        resolved = BoundaryGeometry.from_bbox(boundary)
    elif isinstance(boundary, GeographicRef):
        resolved = boundary_from_ref(boundary)
    else:
        resolved = boundary
    require_same_crs(resolved.crs, crs, what="boundary")
    return resolved


def _as_scope(value: SiteSelectionScope | str) -> SiteSelectionScope:
    if isinstance(value, SiteSelectionScope):
        return value
    try:
        return SiteSelectionScope(value)
    except ValueError as exc:
        raise InvalidParameterError(f"Unknown site-selection scope {value!r}") from exc


def _validate_template(strategy: GenerationStrategy, template: str) -> None:
    if strategy is GenerationStrategy.REGULAR_GRID:
        if "{row}" not in template or "{col}" not in template:
            raise InvalidParameterError("regular_grid id_template must contain {row} and {col}")
        return
    if "{row}" not in template and "{r}" not in template:
        raise InvalidParameterError("hexagonal_grid id_template must contain {row} or {r}")
    if "{col}" not in template and "{q}" not in template:
        raise InvalidParameterError("hexagonal_grid id_template must contain {col} or {q}")


def _is_rooftop(project_type: ProjectType) -> bool:
    return project_type in {
        ProjectType.ROOFTOP,
        ProjectType.COMMERCIAL_ROOFTOP,
        ProjectType.RESIDENTIAL_ROOFTOP,
    }
