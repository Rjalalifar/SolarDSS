"""Adopt caller-supplied polygons as candidates. No new locations are created."""

from __future__ import annotations

from collections.abc import Sequence

from solardss.candidates.contracts import (
    CandidateFeature,
    CandidateGeometry,
    CandidateSource,
    InputFeature,
    PolygonPart,
)
from solardss.candidates.geometry import closed_ring, geometry_area_m2, polygon_geometry
from solardss.core.hashing import fingerprint_canonical
from solardss.core.jsonutil import JsonValue
from solardss.core.methods import MethodId
from solardss.core.quantity import Quantity
from solardss.exceptions import InvalidParameterError, SchemaError
from solardss.indicators.infrastructure import FeatureKind, GeometryKind, InfraFeature
from solardss.provenance.record import provenance_record


def input_from_building(
    feature: InfraFeature,
    *,
    crs: str,
    candidate_id: str | None = None,
    source: CandidateSource = CandidateSource.BUILDING_FOOTPRINT,
) -> InputFeature:
    """Use a building footprint ring as a candidate geometry."""

    if feature.kind is not FeatureKind.BUILDING:
        raise InvalidParameterError("only BUILDING features can be used as rooftop candidates")
    if feature.geometry_kind is not GeometryKind.POLYGON:
        raise InvalidParameterError("building candidates require a polygonal footprint")
    ident = candidate_id
    if ident is None:
        if feature.osm_id is None:
            raise InvalidParameterError(
                "building feature needs candidate_id or osm_id; an id is not invented"
            )
        osm_type = feature.osm_type or "way"
        ident = f"osm-{osm_type}-{feature.osm_id}"
    source_id = None if feature.osm_id is None else f"{feature.osm_type or 'way'}/{feature.osm_id}"
    metadata: dict[str, JsonValue] = {}
    if feature.name:
        metadata["name"] = feature.name
    return InputFeature(
        candidate_id=ident,
        geometry=polygon_geometry(feature.coordinates, crs=crs, holes=feature.holes),
        source=source,
        source_id=source_id,
        metadata=metadata,
    )


def candidates_from_buildings(
    features: Sequence[InfraFeature],
    *,
    crs: str,
) -> tuple[InputFeature, ...]:
    """Map building footprints to input features, preserving caller order."""

    return tuple(input_from_building(item, crs=crs) for item in features)


def adopt_features(
    features: Sequence[InputFeature],
    *,
    crs: str,
) -> tuple[CandidateFeature, ...]:
    """Copy supplied polygons. Order and ids are preserved; nothing is tessellated."""

    if not features:
        raise InvalidParameterError(
            "CANDIDATE_RANKING requires at least one feature; new locations are not generated"
        )
    seen: set[str] = set()
    adopted: list[CandidateFeature] = []
    for item in features:
        if item.candidate_id in seen:
            raise SchemaError(f"duplicate candidate_id {item.candidate_id!r}")
        seen.add(item.candidate_id)
        if item.geometry.crs != crs:
            raise InvalidParameterError(
                f"feature {item.candidate_id!r} CRS {item.geometry.crs!r} does not match {crs!r}"
            )
        area, method = geometry_area_m2(item.geometry)
        if area <= 0.0:
            raise InvalidParameterError(
                f"feature {item.candidate_id!r} has no positive area; empty geometries are not kept"
            )
        provenance = provenance_record(
            MethodId.CANDIDATES_FROM_FEATURES,
            parameters={
                "candidate_id": item.candidate_id,
                "source": item.source.value,
                "area_method": method,
            },
            input_fingerprints=(fingerprint_canonical({"id": item.candidate_id, "crs": crs}),),
            notes=("supplied feature; no grid was generated",),
        )
        adopted.append(
            CandidateFeature(
                candidate_id=item.candidate_id,
                geometry=item.geometry,
                area=Quantity(value=area, unit="m2"),
                source=item.source,
                source_id=item.source_id,
                metadata=dict(item.metadata),
                provenance=provenance,
            )
        )
    return tuple(adopted)


def close_input_polygon(
    candidate_id: str,
    exterior: Sequence[tuple[float, float]],
    *,
    crs: str,
    source: CandidateSource = CandidateSource.FEATURE,
    source_id: str | None = None,
    holes: Sequence[Sequence[tuple[float, float]]] = (),
) -> InputFeature:
    """Build an input feature from a ring. Closing the ring does not add a site."""

    part = PolygonPart(
        exterior=closed_ring(exterior),
        holes=tuple(closed_ring(hole) for hole in holes),
    )
    return InputFeature(
        candidate_id=candidate_id,
        geometry=CandidateGeometry(crs=crs, parts=(part,)),
        source=source,
        source_id=source_id,
    )
