"""Footprint area and OSM roof-tag mapping. Missing tags stay missing."""

from __future__ import annotations

import numpy as np

from solardss.indicators.infrastructure import FeatureKind, GeometryKind, InfraFeature
from solardss.indicators.rooftop import (
    AREA_METHOD_AUTHALIC,
    AREA_METHOD_PROJECTED,
    analyze_rooftop,
    footprint_area_of,
    local_square_lonlat,
    parse_roof_tags,
    polygon_area_m2,
)


def test_projected_square_area() -> None:
    ring = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0), (0.0, 0.0))
    area, method = polygon_area_m2(ring, crs="EPSG:32632")
    assert method == AREA_METHOD_PROJECTED
    np.testing.assert_allclose(area, 100.0)


def test_geographic_ten_metre_square_area() -> None:
    ring = local_square_lonlat(13.4, 52.52, 10.0)
    feature = InfraFeature(
        kind=FeatureKind.BUILDING,
        geometry_kind=GeometryKind.POLYGON,
        coordinates=ring,
        osm_type="way",
        osm_id=1,
        tags=(("building", "yes"),),
    )
    footprint = footprint_area_of(feature, crs="EPSG:4326")
    assert footprint is not None
    assert footprint.method == AREA_METHOD_AUTHALIC
    np.testing.assert_allclose(footprint.area_m2, 100.0, rtol=1e-3)


def test_roof_direction_and_angle_are_mapped() -> None:
    tags = parse_roof_tags(
        {
            "roof:shape": "gabled",
            "roof:direction": "S",
            "roof:angle": "35",
            "building:levels": "3",
        }
    )
    np.testing.assert_allclose(tags.orientation_deg or -1.0, 180.0)
    np.testing.assert_allclose(tags.slope_deg or -1.0, 35.0)
    assert tags.shape == "gabled"
    assert ("roof:shape", "gabled") in tags.preserved
    assert ("building:levels", "3") in tags.preserved


def test_numeric_direction_and_flat_shape() -> None:
    numeric = parse_roof_tags({"roof:direction": "90"})
    np.testing.assert_allclose(numeric.orientation_deg or -1.0, 90.0)
    flat = parse_roof_tags({"roof:shape": "flat"})
    assert flat.slope_deg == 0.0


def test_missing_and_unparseable_tags_are_not_invented() -> None:
    empty = parse_roof_tags({})
    assert empty.orientation_deg is None
    assert empty.slope_deg is None
    hipped = parse_roof_tags({"roof:shape": "hipped"})
    assert hipped.slope_deg is None
    assert any("does not imply" in note for note in hipped.notes)
    junk = parse_roof_tags({"roof:direction": "towards-the-park", "roof:angle": "steep"})
    assert junk.orientation_deg is None
    assert junk.slope_deg is None


def test_open_ring_does_not_invent_area() -> None:
    line = InfraFeature(
        kind=FeatureKind.BUILDING,
        geometry_kind=GeometryKind.LINESTRING,
        coordinates=((0.0, 0.0), (1.0, 0.0)),
        osm_type="way",
        osm_id=2,
        tags=(("building", "yes"),),
    )
    result = analyze_rooftop(line, crs="EPSG:4326")
    assert result.footprint is None
    assert any("not invented" in item for item in result.layer.warnings)


def test_no_building_does_not_invent_footprint() -> None:
    result = analyze_rooftop(None)
    assert result.footprint is None
    assert result.roof.orientation_deg is None
