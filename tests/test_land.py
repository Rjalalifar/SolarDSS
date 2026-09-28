"""Land-cover zonal fractions, geodesic area, and explicit available-area profiles."""

from __future__ import annotations

import numpy as np
import pytest

from solardss.core.arrays import IntArray
from solardss.data.worldcover import ESA_WORLDCOVER_LEGEND
from solardss.exceptions import InvalidGridError, InvalidParameterError
from solardss.indicators.land import (
    AREA_METHOD_AUTHALIC,
    AREA_METHOD_PROJECTED,
    UNMAPPED_CATEGORY,
    AvailableLandProfile,
    LandCoverClass,
    LandCoverGrid,
    LandCoverLegend,
    analyze_land_cover,
    authalic_rectangle_area_m2,
    category_fraction,
    cell_area_m2,
    resolve_available_profile,
)

DX = 10.0
CRS = "EPSG:32632"


def _legend() -> LandCoverLegend:
    return ESA_WORLDCOVER_LEGEND


def _codes() -> IntArray:
    # 8 cropland, 4 tree cover, 2 water, 2 built-up.
    return np.asarray(
        [
            [40, 40, 40, 40],
            [40, 40, 40, 40],
            [10, 10, 10, 10],
            [80, 80, 50, 50],
        ],
        dtype=np.int16,
    )


def _grid(codes: IntArray | None = None) -> LandCoverGrid:
    return LandCoverGrid(
        codes=_codes() if codes is None else codes,
        cell_size_x=DX,
        cell_size_y=DX,
        nodata=0,
        crs=CRS,
        origin_x=500_000.0,
        origin_y=5_800_000.0,
    )


def test_known_class_proportions_on_equal_area_cells() -> None:
    result = analyze_land_cover(_grid(), _legend())
    summary = result.summary
    assert summary.area_method == AREA_METHOD_PROJECTED
    assert summary.valid_cell_count == 16
    np.testing.assert_allclose(summary.valid_area_m2, 1600.0)
    np.testing.assert_allclose(category_fraction(summary, "agricultural"), 0.5)
    np.testing.assert_allclose(category_fraction(summary, "forest"), 0.25)
    np.testing.assert_allclose(category_fraction(summary, "water"), 0.125)
    np.testing.assert_allclose(category_fraction(summary, "builtup"), 0.125)
    np.testing.assert_allclose(category_fraction(summary, "bare_land"), 0.0)
    np.testing.assert_allclose(category_fraction(summary, "wetland"), 0.0)
    agri = next(item for item in summary.category_shares if item.key == "agricultural")
    np.testing.assert_allclose(agri.area_m2, 800.0)
    np.testing.assert_allclose(agri.percent, 50.0)
    assert summary.dominant_category == "agricultural"
    assert summary.dominant_code == 40
    assert summary.dominant_name == "Cropland"
    assert summary.available_area_m2 is None


def test_non_water_is_not_automatically_available() -> None:
    with pytest.raises(InvalidParameterError, match="non-water"):
        AvailableLandProfile(profile_id="oops")
    result = analyze_land_cover(_grid(), _legend())
    assert result.available_mask is None
    water_area = category_fraction(result.summary, "water") * result.summary.valid_area_m2
    non_water = result.summary.valid_area_m2 - water_area
    np.testing.assert_allclose(non_water, 1400.0)
    bare_only = AvailableLandProfile(
        profile_id="bare_only",
        available_categories=("bare_land",),
        excluded_categories=("water", "wetland", "forest", "agricultural", "builtup"),
    )
    with_profile = analyze_land_cover(_grid(), _legend(), profile=bare_only)
    assert with_profile.summary.available_area_m2 is not None
    np.testing.assert_allclose(with_profile.summary.available_area_m2, 0.0)
    assert with_profile.summary.available_area_m2 != non_water


def test_available_area_uses_profile_and_exclusion_wins() -> None:
    profile = AvailableLandProfile(
        profile_id="crops_not_water",
        available_categories=("agricultural", "water"),
        excluded_categories=("water",),
    )
    result = analyze_land_cover(_grid(), _legend(), profile=profile)
    assert result.summary.available_area_m2 is not None
    assert result.summary.available_fraction is not None
    np.testing.assert_allclose(result.summary.available_area_m2, 800.0)
    np.testing.assert_allclose(result.summary.available_fraction, 0.5)
    constraint = np.zeros((4, 4), dtype=bool)
    constraint[0, :] = True
    constrained = analyze_land_cover(
        _grid(), _legend(), profile=profile, constraint_mask=constraint
    )
    assert constrained.summary.available_area_m2 is not None
    np.testing.assert_allclose(constrained.summary.available_area_m2, 400.0)


def test_nodata_and_unmapped_codes_are_reported() -> None:
    codes = _codes()
    codes[0, 0] = 0
    codes[3, 3] = 12
    result = analyze_land_cover(_grid(codes), _legend())
    assert result.summary.nodata_cell_count == 1
    assert result.summary.unmapped_cell_count == 1
    np.testing.assert_allclose(result.summary.valid_area_m2, 1500.0)
    np.testing.assert_allclose(category_fraction(result.summary, "agricultural"), 700.0 / 1500.0)
    unmapped = next(
        item for item in result.summary.category_shares if item.key == UNMAPPED_CATEGORY
    )
    np.testing.assert_allclose(unmapped.area_m2, 100.0)
    assert any("outside the legend" in item for item in result.summary.warnings)


def test_geographic_rows_use_authalic_area_not_pixel_counts() -> None:
    codes = np.asarray(
        [
            [80, 80, 80, 80],
            [40, 40, 40, 40],
        ],
        dtype=np.int16,
    )
    grid = LandCoverGrid(
        codes=codes,
        cell_size_x=0.01,
        cell_size_y=0.01,
        nodata=0,
        crs="EPSG:4326",
        origin_x=8.0,
        origin_y=50.02,
    )
    areas, method = cell_area_m2(grid)
    assert method == AREA_METHOD_AUTHALIC
    north = float(np.asarray(authalic_rectangle_area_m2(50.02, 50.01, 0.01)))
    south = float(np.asarray(authalic_rectangle_area_m2(50.01, 50.00, 0.01)))
    assert south > north
    np.testing.assert_allclose(areas[0, :], north)
    np.testing.assert_allclose(areas[1, :], south)
    result = analyze_land_cover(grid, _legend())
    water = category_fraction(result.summary, "water")
    agri = category_fraction(result.summary, "agricultural")
    assert agri > 0.5
    assert water < 0.5
    assert result.summary.dominant_category == "agricultural"
    np.testing.assert_allclose(result.summary.valid_area_m2, 4.0 * north + 4.0 * south)


def test_mask_mismatch_and_named_profile_lookup() -> None:
    with pytest.raises(InvalidGridError, match="mask"):
        analyze_land_cover(_grid(), _legend(), mask=np.ones((2, 2), dtype=bool))
    catalog = {
        "bare_only": AvailableLandProfile(
            profile_id="bare_only", available_categories=("bare_land",)
        )
    }
    found = resolve_available_profile(profile_id="bare_only", catalog=catalog)
    assert found is not None
    assert found.profile_id == "bare_only"
    with pytest.raises(InvalidParameterError, match="unknown available-land profile"):
        resolve_available_profile(profile_id="missing", catalog=catalog)
    custom = resolve_available_profile(
        profile_id=None,
        available_categories=("grassland",),
        excluded_categories=("water",),
        catalog=catalog,
    )
    assert custom is not None
    assert custom.profile_id == "caller_supplied"


def test_mangroves_map_to_wetland_not_forest() -> None:
    assert _legend().category_for(95) == "wetland"
    assert _legend().category_for(10) == "forest"
    codes = np.full((3, 3), 95, dtype=np.int16)
    grid = LandCoverGrid(codes=codes, cell_size_x=DX, cell_size_y=DX, crs=CRS)
    result = analyze_land_cover(grid, _legend())
    np.testing.assert_allclose(category_fraction(result.summary, "wetland"), 1.0)
    np.testing.assert_allclose(category_fraction(result.summary, "forest"), 0.0)


def test_legend_rejects_duplicate_codes() -> None:
    with pytest.raises(InvalidParameterError, match="unique"):
        LandCoverLegend(
            dataset_id="dup",
            classes=(
                LandCoverClass(code=10, name="a", category="forest"),
                LandCoverClass(code=10, name="b", category="forest"),
            ),
        )
