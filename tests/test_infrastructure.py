"""Nearest-feature distances: geodesic-safe geometry, empty extracts stay empty."""

from __future__ import annotations

from math import pi

import numpy as np
import pytest

from solardss.exceptions import InvalidParameterError
from solardss.indicators.infrastructure import (
    DISTANCE_METHOD_EQUIRECTANGULAR,
    DISTANCE_METHOD_PROJECTED,
    WGS84_AUTHALIC_RADIUS_M,
    FeatureKind,
    GeometryKind,
    InfraFeature,
    analyze_nearest_distance,
    demand_center_features,
    feature_distance_m,
    haversine_distance_m,
    nearest_feature,
    point_in_ring,
)


def _point(kind: FeatureKind, x: float, y: float, osm_id: int = 1) -> InfraFeature:
    return InfraFeature(
        kind=kind,
        geometry_kind=GeometryKind.POINT,
        coordinates=((x, y),),
        osm_type="node",
        osm_id=osm_id,
    )


def test_meridian_distance_matches_authalic_arc() -> None:
    expected = WGS84_AUTHALIC_RADIUS_M * 0.01 * pi / 180.0
    feature = _point(FeatureKind.SUBSTATION, 0.0, 0.01)
    hit = nearest_feature(0.0, 0.0, (feature,), crs="EPSG:4326")
    assert hit is not None
    np.testing.assert_allclose(hit.distance_m, expected, rtol=1e-10)
    assert hit.method == DISTANCE_METHOD_EQUIRECTANGULAR
    haversine = haversine_distance_m(0.0, 0.0, 0.0, 0.01)
    np.testing.assert_allclose(hit.distance_m, haversine, rtol=1e-8)


def test_point_to_vertical_line_is_east_offset() -> None:
    expected = WGS84_AUTHALIC_RADIUS_M * 0.001 * pi / 180.0
    line = InfraFeature(
        kind=FeatureKind.POWER_LINE,
        geometry_kind=GeometryKind.LINESTRING,
        coordinates=((0.001, -0.001), (0.001, 0.001)),
        osm_type="way",
        osm_id=9,
    )
    hit = nearest_feature(0.0, 0.0, (line,), crs="EPSG:4326")
    assert hit is not None
    np.testing.assert_allclose(hit.distance_m, expected, rtol=1e-6)


def test_inside_polygon_distance_is_zero() -> None:
    square = InfraFeature(
        kind=FeatureKind.SETTLEMENT,
        geometry_kind=GeometryKind.POLYGON,
        coordinates=((0.0, 0.0), (0.02, 0.0), (0.02, 0.02), (0.0, 0.02), (0.0, 0.0)),
        osm_type="way",
        osm_id=3,
    )
    hit = nearest_feature(0.01, 0.01, (square,), crs="EPSG:4326")
    assert hit is not None
    assert hit.inside_polygon is True
    np.testing.assert_allclose(hit.distance_m, 0.0)
    assert point_in_ring(0.5, 0.5, ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))


def test_empty_features_are_not_invented() -> None:
    assert nearest_feature(0.0, 0.0, (), crs="EPSG:4326") is None
    result = analyze_nearest_distance(0.0, 0.0, (), kind=FeatureKind.ROAD, crs="EPSG:4326")
    assert result.hit is None
    assert result.feature_count == 0
    assert "not invented" in result.layer.warnings[0]


def test_projected_3_4_5_triangle() -> None:
    feature = _point(FeatureKind.ROAD, 300.0, 400.0)
    distance, _inside = feature_distance_m(0.0, 0.0, feature, crs="EPSG:32632")
    np.testing.assert_allclose(distance, 500.0)
    hit = nearest_feature(0.0, 0.0, (feature,), crs="EPSG:32632")
    assert hit is not None
    assert hit.method == DISTANCE_METHOD_PROJECTED


def test_demand_centers_are_caller_points_not_settlements() -> None:
    points = demand_center_features(((13.4, 52.5),))
    assert points[0].kind is FeatureKind.DEMAND_CENTER
    settlement = _point(FeatureKind.SETTLEMENT, 13.401, 52.5)
    hit = nearest_feature(
        13.4, 52.5, (*points, settlement), crs="EPSG:4326", kind=FeatureKind.DEMAND_CENTER
    )
    assert hit is not None
    np.testing.assert_allclose(hit.distance_m, 0.0)
    with pytest.raises(InvalidParameterError, match="names"):
        demand_center_features(((0.0, 0.0),), names=("a", "b"))
