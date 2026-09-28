"""Spatial candidate generation: scopes, grids, clip, CRS, grouping."""

from __future__ import annotations

from math import sqrt

import pytest

from solardss.candidates import (
    BoundaryGeometry,
    CandidateGenerationConfig,
    CandidateSource,
    GenerationStrategy,
    candidates_from_buildings,
    close_input_polygon,
    contiguous_groups,
    generate_candidates,
    generate_for_problem,
)
from solardss.candidates.errors import CandidateClipError, CandidateGridError
from solardss.candidates.geometry import geometry_centroid
from solardss.candidates.grid import hex_center
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.decision.errors import OperationNotAllowedError
from solardss.decision.problem import SolarDecisionProblem
from solardss.exceptions import IncompleteInputsError, InvalidParameterError
from solardss.indicators.contracts import SiteSelectionScope
from solardss.indicators.infrastructure import FeatureKind, GeometryKind, InfraFeature

CRS = "EPSG:32632"
SQRT3 = sqrt(3.0)


def _square(size: float, crs: str = CRS) -> BoundingBox:
    return BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=size, max_lat=size, crs=crs)


def _grid_config(
    cell_size: float = 10.0,
    *,
    strategy: GenerationStrategy = GenerationStrategy.REGULAR_GRID,
    crs: str = CRS,
    unit: str = "m",
    origin_x: float = 0.0,
    origin_y: float = 0.0,
) -> CandidateGenerationConfig:
    return CandidateGenerationConfig(
        strategy=strategy,
        crs=crs,
        cell_size=cell_size,
        cell_size_unit=unit,
        origin_x=origin_x,
        origin_y=origin_y,
    )


def test_config_rejects_missing_cell_size() -> None:
    with pytest.raises(InvalidParameterError, match="cell_size"):
        CandidateGenerationConfig(strategy=GenerationStrategy.REGULAR_GRID, crs=CRS)


def test_features_strategy_rejects_cell_size() -> None:
    with pytest.raises(InvalidParameterError, match="FEATURES"):
        CandidateGenerationConfig(
            strategy=GenerationStrategy.FEATURES,
            crs=CRS,
            cell_size=10.0,
            cell_size_unit="m",
        )


def test_projected_crs_rejects_degree_cell_size() -> None:
    with pytest.raises(CandidateGridError, match="projected CRS"):
        generate_candidates(
            scope=SiteSelectionScope.REGION_SEARCH,
            config=_grid_config(unit="deg"),
            boundary=_square(30.0),
        )


def test_geographic_crs_rejects_metre_cell_size() -> None:
    bbox = BoundingBox(min_lon=13.40, min_lat=52.50, max_lon=13.42, max_lat=52.52, crs="EPSG:4326")
    with pytest.raises(CandidateGridError, match="geographic CRS"):
        generate_candidates(
            scope=SiteSelectionScope.REGION_SEARCH,
            config=_grid_config(crs="EPSG:4326", unit="m"),
            boundary=bbox,
        )


def test_crs_mismatch_is_rejected() -> None:
    bbox = BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=30.0, max_lat=30.0, crs="EPSG:4326")
    with pytest.raises(InvalidParameterError, match="does not match"):
        generate_candidates(
            scope=SiteSelectionScope.REGION_SEARCH,
            config=_grid_config(),
            boundary=bbox,
        )


def test_regular_grid_aligned_squares_have_configured_area() -> None:
    bundle = generate_candidates(
        scope=SiteSelectionScope.REGION_SEARCH,
        config=_grid_config(10.0),
        boundary=_square(30.0),
    )
    assert bundle.strategy is GenerationStrategy.REGULAR_GRID
    full = [item for item in bundle.candidates if item.area.value == pytest.approx(100.0)]
    assert len(full) == 9
    assert {item.geometry.crs for item in bundle.candidates} == {CRS}
    assert all(item.source is CandidateSource.REGULAR_GRID for item in full)
    table = bundle.to_table()
    assert "geometry" in table.frame.columns
    assert table.frame["crs"].unique().tolist() == [CRS]


def test_aoi_search_clips_to_boundary() -> None:
    aoi = BoundingBox(min_lon=5.0, min_lat=5.0, max_lon=25.0, max_lat=25.0, crs=CRS)
    bundle = generate_candidates(
        scope=SiteSelectionScope.AOI_SEARCH,
        config=_grid_config(10.0),
        boundary=aoi,
    )
    assert bundle.scope is SiteSelectionScope.AOI_SEARCH
    assert bundle.candidates
    for item in bundle.candidates:
        for part in item.geometry.parts:
            for x, y in part.exterior:
                assert 5.0 - 1e-9 <= x <= 25.0 + 1e-9
                assert 5.0 - 1e-9 <= y <= 25.0 + 1e-9
        cx, cy = geometry_centroid(item.geometry)
        assert 5.0 <= cx <= 25.0
        assert 5.0 <= cy <= 25.0
        assert item.area.value > 0.0
        assert item.area.value <= 100.0 + 1e-9
    edge = [item for item in bundle.candidates if item.area.value < 99.0]
    assert edge, "boundary cells must be clipped, not left as full squares outside the AOI"


def test_empty_aoi_stays_empty() -> None:
    aoi = BoundingBox(min_lon=0.0, min_lat=0.0, max_lon=0.0, max_lat=0.0, crs=CRS)
    bundle = generate_candidates(
        scope=SiteSelectionScope.AOI_SEARCH,
        config=_grid_config(10.0),
        boundary=aoi,
    )
    assert len(bundle) == 0
    assert bundle.to_table().ids() == ()


def test_hex_grid_neighbour_spacing_matches_cell_size() -> None:
    bundle = generate_candidates(
        scope=SiteSelectionScope.REGION_SEARCH,
        config=_grid_config(10.0, strategy=GenerationStrategy.HEXAGONAL_GRID),
        boundary=_square(80.0),
    )
    by_index = {(item.grid_col, item.grid_row): item for item in bundle.candidates}
    origin = next(item for item in bundle.candidates if item.grid_col == 2 and item.grid_row == 2)
    hex_size = 10.0 / SQRT3
    cx, cy = hex_center(2, 2, origin_x=0.0, origin_y=0.0, hex_size=hex_size)
    expected_area = (SQRT3 / 2.0) * 10.0 * 10.0
    if origin.area.value == pytest.approx(expected_area, rel=1e-6):
        neighbors = (
            (3, 2),
            (3, 1),
            (2, 1),
            (1, 2),
            (1, 3),
            (2, 3),
        )
        found = 0
        for nq, nr in neighbors:
            other = by_index.get((nq, nr))
            if other is None:
                continue
            ox, oy = hex_center(nq, nr, origin_x=0.0, origin_y=0.0, hex_size=hex_size)
            dist = sqrt((ox - cx) ** 2 + (oy - cy) ** 2)
            assert dist == pytest.approx(10.0, rel=1e-6)
            found += 1
        assert found == 6
    else:
        raise AssertionError("interior hexagon area does not match configured cell size")


def test_ranking_uses_exact_features_and_order() -> None:
    first = close_input_polygon(
        "parcel-a",
        ((0.0, 0.0), (8.0, 0.0), (8.0, 5.0), (0.0, 5.0)),
        crs=CRS,
        source=CandidateSource.PARCEL,
        source_id="cadastre-1",
    )
    second = close_input_polygon(
        "parcel-b",
        ((20.0, 20.0), (24.0, 20.0), (24.0, 24.0), (20.0, 24.0)),
        crs=CRS,
        source=CandidateSource.PARCEL,
        source_id="cadastre-2",
    )
    bundle = generate_candidates(
        scope=SiteSelectionScope.CANDIDATE_RANKING,
        config=CandidateGenerationConfig(strategy=GenerationStrategy.FEATURES, crs=CRS),
        features=(first, second),
    )
    assert bundle.ids() == ("parcel-a", "parcel-b")
    assert bundle.candidates[0].area.value == pytest.approx(40.0)
    assert bundle.candidates[1].area.value == pytest.approx(16.0)
    assert bundle.candidates[0].source is CandidateSource.PARCEL
    assert bundle.covering_count == 2
    assert bundle.dropped_count == 0


def test_ranking_rejects_grid_strategy() -> None:
    feature = close_input_polygon(
        "only",
        ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
        crs=CRS,
        source=CandidateSource.PREDEFINED_SITE,
    )
    with pytest.raises(InvalidParameterError, match="must not run a grid"):
        generate_candidates(
            scope=SiteSelectionScope.CANDIDATE_RANKING,
            config=_grid_config(),
            features=(feature,),
        )


def test_region_search_rejects_feature_strategy() -> None:
    with pytest.raises(InvalidParameterError, match="regular_grid or hexagonal_grid"):
        generate_candidates(
            scope=SiteSelectionScope.REGION_SEARCH,
            config=CandidateGenerationConfig(strategy=GenerationStrategy.FEATURES, crs=CRS),
            boundary=_square(30.0),
        )


def test_building_footprints_are_candidates() -> None:
    building = InfraFeature(
        kind=FeatureKind.BUILDING,
        geometry_kind=GeometryKind.POLYGON,
        coordinates=((0.0, 0.0), (12.0, 0.0), (12.0, 8.0), (0.0, 8.0), (0.0, 0.0)),
        osm_type="way",
        osm_id=42,
        name="hall",
        tags=(("building", "yes"),),
    )
    features = candidates_from_buildings((building,), crs=CRS)
    bundle = generate_candidates(
        scope=SiteSelectionScope.CANDIDATE_RANKING,
        config=CandidateGenerationConfig(strategy=GenerationStrategy.FEATURES, crs=CRS),
        features=features,
    )
    assert bundle.ids() == ("osm-way-42",)
    assert bundle.candidates[0].source is CandidateSource.BUILDING_FOOTPRINT
    assert bundle.candidates[0].area.value == pytest.approx(96.0)
    assert bundle.candidates[0].source_id == "way/42"


def test_contiguous_groups_rook_adjacency_without_ranking() -> None:
    bundle = generate_candidates(
        scope=SiteSelectionScope.REGION_SEARCH,
        config=_grid_config(10.0),
        boundary=_square(30.0),
    )
    suitable = {
        "r0c0": True,
        "r0c1": True,
        "r1c0": True,
        "r2c2": True,
    }
    groups, record = contiguous_groups(bundle, suitable=suitable)
    grouped = {frozenset(group) for group in groups}
    assert frozenset({"r0c0", "r0c1", "r1c0"}) in grouped
    assert frozenset({"r2c2"}) in grouped
    assert len(groups) == 2
    assert "not dissolved" in " ".join(record.notes)


def test_missing_boundary_is_incomplete() -> None:
    with pytest.raises(IncompleteInputsError, match="boundary"):
        generate_candidates(scope=SiteSelectionScope.REGION_SEARCH, config=_grid_config())


def test_fixed_site_cannot_generate() -> None:
    problem = SolarDecisionProblem(
        mode="fixed_site",
        project_type="utility_scale",
        location=GeographicRef(name="site", longitude=13.4, latitude=52.52),
        target_capacity_kw=5.0,
    )
    with pytest.raises(OperationNotAllowedError):
        generate_for_problem(problem, _grid_config(), boundary=_square(30.0))


def test_generate_for_problem_region_uses_ref_bbox() -> None:
    problem = SolarDecisionProblem(
        mode="site_selection",
        project_type="ground_mounted",
        site_selection_scope=SiteSelectionScope.REGION_SEARCH,
        region=GeographicRef(name="box", bbox=_square(20.0), crs=CRS),
        target_capacity_kw=10.0,
    )
    bundle = generate_for_problem(problem, _grid_config(10.0))
    assert bundle.scope is SiteSelectionScope.REGION_SEARCH
    assert len(bundle) == 4
    assert all(item.area.value == pytest.approx(100.0) for item in bundle.candidates)


def test_geographic_degree_grid_keeps_crs() -> None:
    bbox = BoundingBox(min_lon=13.40, min_lat=52.50, max_lon=13.42, max_lat=52.52, crs="EPSG:4326")
    bundle = generate_candidates(
        scope=SiteSelectionScope.AOI_SEARCH,
        config=CandidateGenerationConfig(
            strategy=GenerationStrategy.REGULAR_GRID,
            crs="EPSG:4326",
            cell_size=0.01,
            cell_size_unit="deg",
            origin_x=13.40,
            origin_y=52.50,
        ),
        boundary=bbox,
    )
    assert bundle.candidates
    assert all(item.geometry.crs == "EPSG:4326" for item in bundle.candidates)
    assert all(item.area.unit == "m2" for item in bundle.candidates)
    assert all(item.area.value > 0.0 for item in bundle.candidates)


def test_concave_aoi_clip() -> None:
    pytest.importorskip("shapely")
    exterior = (
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 10.0),
        (10.0, 10.0),
        (10.0, 20.0),
        (0.0, 20.0),
        (0.0, 0.0),
    )
    boundary = BoundaryGeometry(crs=CRS, exterior=exterior)
    bundle = generate_candidates(
        scope=SiteSelectionScope.AOI_SEARCH,
        config=_grid_config(10.0),
        boundary=boundary,
    )
    assert bundle.candidates
    for item in bundle.candidates:
        cx, cy = geometry_centroid(item.geometry)
        in_missing_corner = cx > 10.0 + 1e-6 and cy > 10.0 + 1e-6
        assert not in_missing_corner
        assert item.area.value > 0.0


def test_concave_without_shapely_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    exterior = (
        (0.0, 0.0),
        (20.0, 0.0),
        (20.0, 10.0),
        (10.0, 10.0),
        (10.0, 20.0),
        (0.0, 20.0),
        (0.0, 0.0),
    )
    boundary = BoundaryGeometry(crs=CRS, exterior=exterior)

    def _missing() -> object:
        from solardss.exceptions import MissingExtraError

        raise MissingExtraError("no shapely")

    monkeypatch.setattr("solardss.candidates.clip._require_shapely", _missing)
    with pytest.raises(CandidateClipError, match="concave"):
        generate_candidates(
            scope=SiteSelectionScope.AOI_SEARCH,
            config=_grid_config(10.0),
            boundary=boundary,
        )
