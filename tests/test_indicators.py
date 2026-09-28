"""Indicator catalog and min-max normalization."""

from __future__ import annotations

import numpy as np
import pytest

from solardss.exceptions import IncompleteInputsError, InvalidParameterError, UnknownIndicatorError
from solardss.indicators import Polarity, catalog_ids, get_indicator, minmax_normalize


def test_catalog_marks_irradiation_as_not_computed() -> None:
    spec = get_indicator("ghi_kwh_m2")
    assert spec.computed_by_solardss is False
    assert spec.polarity is Polarity.BENEFIT


def test_catalog_marks_slope_as_computed() -> None:
    spec = get_indicator("slope_deg")
    assert spec.computed_by_solardss is True


def test_catalog_marks_elevation_and_ruggedness_as_computed() -> None:
    elevation = get_indicator("TERRAIN_ELEVATION")
    ruggedness = get_indicator("TERRAIN_RUGGEDNESS")
    horizon = get_indicator("TERRAIN_HORIZON_OBSTRUCTION")
    assert elevation.computed_by_solardss is True
    assert ruggedness.computed_by_solardss is True
    assert horizon.computed_by_solardss is False


def test_catalog_marks_land_cover_fractions_as_computed() -> None:
    forest = get_indicator("LAND_FOREST_FRACTION")
    available = get_indicator("LAND_AVAILABLE_AREA")
    cover = get_indicator("LAND_COVER")
    protected = get_indicator("LAND_PROTECTED_AREA_OVERLAP")
    assert forest.computed_by_solardss is True
    assert available.computed_by_solardss is True
    assert cover.computed_by_solardss is True
    assert protected.computed_by_solardss is True


def test_catalog_marks_osm_distances_and_footprint_as_computed() -> None:
    power = get_indicator("INFRA_DISTANCE_TO_POWER_LINE")
    demand = get_indicator("INFRA_DISTANCE_TO_DEMAND_CENTER")
    hosting = get_indicator("INFRA_GRID_HOSTING_CAPACITY")
    footprint = get_indicator("ROOF_BUILDING_FOOTPRINT_AREA")
    orientation = get_indicator("ROOF_ORIENTATION")
    slope = get_indicator("ROOF_SLOPE")
    usable = get_indicator("ROOF_USABLE_AREA")
    assert power.computed_by_solardss is True
    assert demand.computed_by_solardss is True
    assert hosting.computed_by_solardss is False
    assert footprint.computed_by_solardss is True
    assert orientation.computed_by_solardss is True
    assert slope.computed_by_solardss is True
    assert usable.computed_by_solardss is False


def test_catalog_marks_hazard_sampling_as_computed() -> None:
    flood = get_indicator("HAZARD_FLOOD_RISK")
    wildfire = get_indicator("HAZARD_WILDFIRE_RISK")
    landslide = get_indicator("HAZARD_LANDSLIDE_RISK")
    hail = get_indicator("HAZARD_HAIL_RISK")
    assert flood.computed_by_solardss is True
    assert wildfire.computed_by_solardss is True
    assert landslide.computed_by_solardss is True
    assert hail.computed_by_solardss is True
    assert flood.derivation_method is not None
    assert flood.derivation_method.id == "hazard_source_mapping"


def test_unknown_indicator() -> None:
    with pytest.raises(UnknownIndicatorError):
        get_indicator("made_up_irradiance_model")


def test_catalog_ids_are_sorted() -> None:
    ids = catalog_ids()
    assert ids == tuple(sorted(ids))


def test_minmax_benefit_and_cost() -> None:
    values = np.array([10.0, 20.0, 30.0])
    benefit = minmax_normalize(values, polarity="benefit", range_min=10.0, range_max=30.0)
    cost = minmax_normalize(values, polarity="cost", range_min=10.0, range_max=30.0)
    np.testing.assert_allclose(benefit.values, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(cost.values, [1.0, 0.5, 0.0])
    assert benefit.layer.provenance.parameters["range_source"] == "explicit"


def test_minmax_sample_range() -> None:
    values = np.array([2.0, 4.0])
    result = minmax_normalize(values, polarity=Polarity.BENEFIT)
    np.testing.assert_allclose(result.values, [0.0, 1.0])
    assert result.layer.provenance.parameters["range_source"] == "sample"


def test_minmax_zero_range_is_incomplete() -> None:
    with pytest.raises(IncompleteInputsError):
        minmax_normalize(np.array([3.0, 3.0, 3.0]), polarity="benefit")


def test_minmax_rejects_none_polarity() -> None:
    with pytest.raises(InvalidParameterError):
        minmax_normalize(np.array([1.0, 2.0]), polarity=Polarity.NONE)


def test_minmax_preserves_nan() -> None:
    result = minmax_normalize(
        np.array([0.0, np.nan, 10.0]),
        polarity="benefit",
        range_min=0.0,
        range_max=10.0,
    )
    assert np.isnan(result.values[1])
    np.testing.assert_allclose(result.values[0], 0.0)
    np.testing.assert_allclose(result.values[2], 1.0)
