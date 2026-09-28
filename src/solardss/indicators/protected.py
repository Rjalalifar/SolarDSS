"""Protected-area overlap and explicit legal/methodology constraint profiles.

Overlap with a cadastre is a measured intersection. It is not a universal
exclusion: hard exclusion, penalty, or warning come only from a
``ProtectedAreaConstraintProfile``. Empty or missing profiles leave overlap
informational. SolarDSS does not assume a global protected-area dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from solardss.core.arrays import FloatArray
from solardss.core.hashing import fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.exceptions import InvalidParameterError
from solardss.indicators.infrastructure import crs_is_geographic, local_en_m, point_in_ring
from solardss.indicators.rooftop import projected_ring_area_m2
from solardss.provenance.record import provenance_record

CALC_PROTECTED_OVERLAP = "protected_area_overlap"
AREA_METHOD_LOCAL_EQUIRECTANGULAR = "local_equirectangular_intersection_m2"
AREA_METHOD_PROJECTED = "projected_intersection_m2"


class ConstraintDisposition(str, Enum):
    """How overlap with a class may be treated. Never implied by geometry alone."""

    INFORMATIONAL = "informational"
    WARNING = "warning"
    PENALTY = "penalty"
    HARD_EXCLUSION = "hard_exclusion"


_SEVERITY: dict[ConstraintDisposition, int] = {
    ConstraintDisposition.INFORMATIONAL: 0,
    ConstraintDisposition.WARNING: 1,
    ConstraintDisposition.PENALTY: 2,
    ConstraintDisposition.HARD_EXCLUSION: 3,
}


class ProtectedAreaFeature(BaseModel):
    """One designated site. Coordinates are ``(x, y)`` / ``(lon, lat)`` rings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    site_id: str = Field(min_length=1)
    name: str | None = None
    category: str | None = None
    designation: str | None = None
    coordinates: tuple[tuple[float, float], ...] = Field(min_length=3)
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()
    source_dataset: str = Field(min_length=1)
    attributes: tuple[tuple[str, str], ...] = ()


class ProtectedAreaRule(BaseModel):
    """One explicit class → disposition mapping. Empty class lists are rejected."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    categories: tuple[str, ...] = ()
    designations: tuple[str, ...] = ()
    disposition: ConstraintDisposition
    penalty_weight: float | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _explicit_classes(self) -> ProtectedAreaRule:
        if not self.categories and not self.designations:
            raise InvalidParameterError(
                "ProtectedAreaRule requires categories or designations; "
                "SolarDSS will not treat every protected-area class as excluded"
            )
        if self.disposition is ConstraintDisposition.PENALTY:
            if self.penalty_weight is None:
                raise InvalidParameterError("penalty rules require penalty_weight")
            if self.penalty_weight < 0.0:
                raise InvalidParameterError("penalty_weight must be non-negative")
        elif self.penalty_weight is not None:
            raise InvalidParameterError("penalty_weight is only valid for penalty rules")
        return self


class ProtectedAreaConstraintProfile(BaseModel):
    """Methodology/legal profile. Missing unmatched classes stay informational."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(min_length=1)
    region_id: str | None = None
    rules: tuple[ProtectedAreaRule, ...] = ()
    unmatched_disposition: ConstraintDisposition = ConstraintDisposition.INFORMATIONAL
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _no_universal_hard_exclusion(self) -> ProtectedAreaConstraintProfile:
        if not self.rules and self.unmatched_disposition is ConstraintDisposition.HARD_EXCLUSION:
            raise InvalidParameterError(
                "ProtectedAreaConstraintProfile refuses universal hard exclusion "
                "of all protected-area classes; list the classes to exclude"
            )
        return self


class ProtectedOverlapRecord(BaseModel):
    """Intersection of the AOI with one designated site."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    site_id: str = Field(min_length=1)
    name: str | None = None
    category: str | None = None
    designation: str | None = None
    intersection_area_m2: float
    intersection_fraction: float
    intersection_percent: float
    disposition: ConstraintDisposition
    source_dataset: str = Field(min_length=1)


class ProtectedOverlapSummary(BaseModel):
    """AOI-level overlap accounting plus profile advice."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    aoi_area_m2: float
    intersection_area_m2: float
    sum_of_feature_intersections_m2: float
    overlap_fraction: float
    overlap_percent: float
    union_is_exact: bool
    site_ids: tuple[str, ...] = ()
    categories: tuple[str, ...] = ()
    designations: tuple[str, ...] = ()
    records: tuple[ProtectedOverlapRecord, ...] = ()
    profile_id: str | None = None
    overall_disposition: ConstraintDisposition = ConstraintDisposition.INFORMATIONAL
    hard_exclusion_site_ids: tuple[str, ...] = ()
    penalty_site_ids: tuple[str, ...] = ()
    warning_site_ids: tuple[str, ...] = ()
    area_method: str = Field(min_length=1)
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProtectedOverlapResult:
    """Overlap summary plus a provenance layer."""

    summary: ProtectedOverlapSummary
    layer: LayerResult


def _norm_token(value: str | None) -> str | None:
    if value is None:
        return None
    compact = value.strip()
    return compact.casefold() if compact else None


def bbox_ring(
    min_x: float, min_y: float, max_x: float, max_y: float
) -> tuple[tuple[float, float], ...]:
    """Closed axis-aligned rectangle, counter-clockwise from the south-west corner."""

    return (
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y),
    )


def _closed(ring: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    if not ring:
        return ()
    if ring[0] == ring[-1]:
        return tuple(ring)
    return (*tuple(ring), ring[0])


def _unsigned(ring: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    closed = _closed(ring)
    if len(closed) < 2:
        return closed
    return closed[:-1]


def _signed_area(ring: Sequence[tuple[float, float]]) -> float:
    verts = _unsigned(ring)
    if len(verts) < 3:
        return 0.0
    total = 0.0
    count = len(verts)
    for index in range(count):
        x1, y1 = verts[index]
        x2, y2 = verts[(index + 1) % count]
        total += x1 * y2 - x2 * y1
    return 0.5 * total


def ensure_ccw(ring: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Return a closed ring with counter-clockwise orientation."""

    closed = _closed(ring)
    if _signed_area(closed) < 0.0:
        body = list(_unsigned(closed))
        body.reverse()
        return _closed(body)
    return closed


def _inside(
    point: tuple[float, float], start: tuple[float, float], end: tuple[float, float]
) -> bool:
    """True when ``point`` is to the left of or on directed edge ``start→end``."""

    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    ) >= -1e-12


def _edge_intersect(
    start: tuple[float, float],
    end: tuple[float, float],
    clip_a: tuple[float, float],
    clip_b: tuple[float, float],
) -> tuple[float, float]:
    s_x, s_y = start[0] - end[0], start[1] - end[1]
    c_x, c_y = clip_a[0] - clip_b[0], clip_a[1] - clip_b[1]
    denom = s_x * c_y - s_y * c_x
    n1 = start[0] * end[1] - start[1] * end[0]
    n2 = clip_a[0] * clip_b[1] - clip_a[1] * clip_b[0]
    if abs(denom) < 1e-18:
        return end
    return ((n1 * c_x - n2 * s_x) / denom, (n1 * c_y - n2 * s_y) / denom)


def clip_polygon(
    subject: Sequence[tuple[float, float]],
    clipper: Sequence[tuple[float, float]],
) -> tuple[tuple[float, float], ...]:
    """Sutherland-Hodgman clip. ``clipper`` must be convex; ``subject`` may be concave."""

    output = list(_unsigned(ensure_ccw(subject)))
    clip_verts = list(_unsigned(ensure_ccw(clipper)))
    if len(output) < 3 or len(clip_verts) < 3:
        return ()
    for index, clip_a in enumerate(clip_verts):
        clip_b = clip_verts[(index + 1) % len(clip_verts)]
        incoming = output
        output = []
        if not incoming:
            break
        prev = incoming[-1]
        for current in incoming:
            curr_in = _inside(current, clip_a, clip_b)
            prev_in = _inside(prev, clip_a, clip_b)
            if curr_in:
                if not prev_in:
                    output.append(_edge_intersect(prev, current, clip_a, clip_b))
                output.append(current)
            elif prev_in:
                output.append(_edge_intersect(prev, current, clip_a, clip_b))
            prev = current
    if len(output) < 3:
        return ()
    return _closed(output)


def ring_to_local(
    ring: Sequence[tuple[float, float]],
    origin_x: float,
    origin_y: float,
    *,
    crs: str | None,
) -> tuple[tuple[float, float], ...]:
    return tuple(local_en_m(x, y, origin_x, origin_y, crs=crs) for x, y in ring)


def _centroid(ring: Sequence[tuple[float, float]]) -> tuple[float, float]:
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def _area_method(crs: str | None) -> str:
    if crs_is_geographic(crs):
        return AREA_METHOD_LOCAL_EQUIRECTANGULAR
    return AREA_METHOD_PROJECTED


def _tokens_match(value: str | None, allowed: Sequence[str]) -> bool:
    token = _norm_token(value)
    if token is None:
        return False
    allowed_norm = {_norm_token(item) for item in allowed}
    return token in allowed_norm


def disposition_for(
    feature: ProtectedAreaFeature,
    profile: ProtectedAreaConstraintProfile | None,
) -> ConstraintDisposition:
    """Most severe matching rule; unmatched classes stay informational unless set."""

    if profile is None:
        return ConstraintDisposition.INFORMATIONAL
    matched: list[ConstraintDisposition] = []
    for rule in profile.rules:
        by_category = _tokens_match(feature.category, rule.categories)
        by_designation = _tokens_match(feature.designation, rule.designations)
        if by_category or by_designation:
            matched.append(rule.disposition)
    if not matched:
        return profile.unmatched_disposition
    return max(matched, key=lambda item: _SEVERITY[item])


AREA_EPS_M2 = 1e-8


def _pairwise_disjoint(rings: Sequence[Sequence[tuple[float, float]]]) -> bool:
    for left_index, left in enumerate(rings):
        for right in rings[left_index + 1 :]:
            inter = clip_polygon(left, right)
            if inter and projected_ring_area_m2(inter) > AREA_EPS_M2:
                return False
    return True


def _unique_labels(values: Sequence[str | None]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        if item is None or not item.strip():
            continue
        key = item.strip().casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(item.strip())
    return tuple(ordered)


def _unique_vertices(ring: Sequence[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    ordered: list[tuple[float, float]] = []
    for point in _unsigned(ring):
        if point not in ordered:
            ordered.append(point)
    return tuple(ordered)


def _clipped_area_m2(
    outer: Sequence[tuple[float, float]],
    holes: Sequence[Sequence[tuple[float, float]]],
    clipper: Sequence[tuple[float, float]],
) -> tuple[float, tuple[tuple[float, float], ...]]:
    intersection = clip_polygon(outer, clipper)
    if not intersection:
        return 0.0, ()
    area = projected_ring_area_m2(intersection)
    for hole in holes:
        hole_clip = clip_polygon(hole, clipper)
        if hole_clip:
            area -= projected_ring_area_m2(hole_clip)
    return max(0.0, area), intersection


def _overall_disposition(
    dispositions: Sequence[ConstraintDisposition],
) -> ConstraintDisposition:
    if not dispositions:
        return ConstraintDisposition.INFORMATIONAL
    return max(dispositions, key=lambda item: _SEVERITY[item])


def hard_exclusion_applies(summary: ProtectedOverlapSummary) -> bool:
    """True only when a profile mapped at least one overlapping site to exclusion."""

    return summary.overall_disposition is ConstraintDisposition.HARD_EXCLUSION


def _layer_for(
    summary: ProtectedOverlapSummary,
    *,
    aoi: Sequence[tuple[float, float]],
    crs: str | None,
) -> LayerResult:
    values: FloatArray = np.asarray([summary.overlap_fraction], dtype=np.float64)
    parameters: dict[str, object] = {
        "crs": crs,
        "aoi_area_m2": summary.aoi_area_m2,
        "intersection_area_m2": summary.intersection_area_m2,
        "overlap_fraction": summary.overlap_fraction,
        "overlap_percent": summary.overlap_percent,
        "union_is_exact": summary.union_is_exact,
        "profile_id": summary.profile_id,
        "overall_disposition": summary.overall_disposition.value,
        "area_method": summary.area_method,
        "site_ids": list(summary.site_ids),
        "categories": list(summary.categories),
        "designations": list(summary.designations),
        "hard_exclusion_site_ids": list(summary.hard_exclusion_site_ids),
        "penalty_site_ids": list(summary.penalty_site_ids),
        "warning_site_ids": list(summary.warning_site_ids),
    }
    return LayerResult(
        name="protected_area_overlap",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.PROTECTED_AREA_OVERLAP.value,
        provenance=provenance_record(
            MethodId.PROTECTED_AREA_OVERLAP,
            parameters=parameters,
            input_fingerprints=(
                fingerprint_canonical(
                    {
                        "aoi": [list(point) for point in aoi],
                        "site_ids": list(summary.site_ids),
                        "crs": crs,
                    }
                ),
            ),
            notes=summary.warnings,
        ),
        summary=summarize_array(values),
        warnings=summary.warnings,
    )


def analyze_point_in_protected(
    x: float,
    y: float,
    features: Sequence[ProtectedAreaFeature],
    *,
    crs: str | None = "EPSG:4326",
    profile: ProtectedAreaConstraintProfile | None = None,
) -> ProtectedOverlapResult:
    """Point AOI: area is 0; fraction is 1 if the point lies in a designated site."""

    records: list[ProtectedOverlapRecord] = []
    for feature in features:
        outer = ring_to_local(feature.coordinates, x, y, crs=crs)
        if not point_in_ring(0.0, 0.0, outer):
            continue
        in_hole = False
        for hole in feature.holes:
            hole_xy = ring_to_local(hole, x, y, crs=crs)
            if point_in_ring(0.0, 0.0, hole_xy):
                in_hole = True
                break
        if in_hole:
            continue
        disposition = disposition_for(feature, profile)
        records.append(
            ProtectedOverlapRecord(
                site_id=feature.site_id,
                name=feature.name,
                category=feature.category,
                designation=feature.designation,
                intersection_area_m2=0.0,
                intersection_fraction=1.0,
                intersection_percent=100.0,
                disposition=disposition,
                source_dataset=feature.source_dataset,
            )
        )
    overlap_fraction = 1.0 if records else 0.0
    warnings = [
        "point AOI has no area; overlap is membership (1 inside, 0 outside)",
        "overlap is not a universal exclusion; disposition comes from the constraint profile",
    ]
    if profile is None:
        warnings.append(
            "no constraint profile; overlap is informational and is not a hard exclusion"
        )
    else:
        warnings.append(f"constraint profile={profile.profile_id}")
    summary = ProtectedOverlapSummary(
        aoi_area_m2=0.0,
        intersection_area_m2=0.0,
        sum_of_feature_intersections_m2=0.0,
        overlap_fraction=overlap_fraction,
        overlap_percent=100.0 * overlap_fraction,
        union_is_exact=True,
        site_ids=tuple(item.site_id for item in records),
        categories=_unique_labels([item.category for item in records]),
        designations=_unique_labels([item.designation for item in records]),
        records=tuple(records),
        profile_id=None if profile is None else profile.profile_id,
        overall_disposition=_overall_disposition([item.disposition for item in records]),
        hard_exclusion_site_ids=tuple(
            item.site_id
            for item in records
            if item.disposition is ConstraintDisposition.HARD_EXCLUSION
        ),
        penalty_site_ids=tuple(
            item.site_id for item in records if item.disposition is ConstraintDisposition.PENALTY
        ),
        warning_site_ids=tuple(
            item.site_id for item in records if item.disposition is ConstraintDisposition.WARNING
        ),
        area_method=_area_method(crs),
        warnings=tuple(warnings),
    )
    return ProtectedOverlapResult(
        summary=summary,
        layer=_layer_for(summary, aoi=((x, y),), crs=crs),
    )


def analyze_protected_overlap(
    aoi: Sequence[tuple[float, float]],
    features: Sequence[ProtectedAreaFeature],
    *,
    crs: str | None = "EPSG:4326",
    profile: ProtectedAreaConstraintProfile | None = None,
) -> ProtectedOverlapResult:
    """Intersect ``aoi`` with designated sites. Disposition is profile-driven only."""

    unique = _unique_vertices(aoi)
    if len(unique) == 1:
        return analyze_point_in_protected(
            unique[0][0], unique[0][1], features, crs=crs, profile=profile
        )
    if len(unique) < 3:
        raise InvalidParameterError("protected-area AOI must be a polygon or a point")

    origin_x, origin_y = _centroid(aoi)
    local_aoi = ensure_ccw(ring_to_local(aoi, origin_x, origin_y, crs=crs))
    aoi_area = projected_ring_area_m2(local_aoi)
    if aoi_area <= AREA_EPS_M2:
        raise InvalidParameterError("protected-area AOI polygon has zero area")

    records: list[ProtectedOverlapRecord] = []
    inter_rings: list[tuple[tuple[float, float], ...]] = []
    feature_sum = 0.0
    for feature in features:
        outer = ring_to_local(feature.coordinates, origin_x, origin_y, crs=crs)
        holes = tuple(ring_to_local(hole, origin_x, origin_y, crs=crs) for hole in feature.holes)
        area, inter = _clipped_area_m2(outer, holes, local_aoi)
        if area <= AREA_EPS_M2:
            continue
        feature_sum += area
        fraction = min(1.0, area / aoi_area)
        if inter:
            inter_rings.append(inter)
        disposition = disposition_for(feature, profile)
        records.append(
            ProtectedOverlapRecord(
                site_id=feature.site_id,
                name=feature.name,
                category=feature.category,
                designation=feature.designation,
                intersection_area_m2=area,
                intersection_fraction=fraction,
                intersection_percent=100.0 * fraction,
                disposition=disposition,
                source_dataset=feature.source_dataset,
            )
        )

    warnings: list[str] = [
        "overlap is not a universal exclusion; disposition comes from the constraint profile",
        "SolarDSS does not assume a global protected-area dataset",
    ]
    if len(local_aoi) > 5:
        warnings.append("AOI clipper is treated as convex; concave AOI intersection may be inexact")
    union_exact = True
    intersection_area = feature_sum
    if len(inter_rings) > 1 and not _pairwise_disjoint(inter_rings):
        union_exact = False
        intersection_area = min(feature_sum, aoi_area)
        warnings.append(
            "overlapping protected-area designations; union area is the sum of "
            "intersections capped at AOI area"
        )
    elif feature_sum - aoi_area > AREA_EPS_M2:
        intersection_area = aoi_area
        union_exact = False
        warnings.append("sum of intersections exceeded AOI area; capped at AOI area")
    overlap_fraction = min(1.0, intersection_area / aoi_area) if aoi_area else 0.0
    if profile is None:
        warnings.append(
            "no constraint profile; overlap is informational and is not a hard exclusion"
        )
    else:
        warnings.append(f"constraint profile={profile.profile_id}")
        if profile.notes:
            warnings.extend(profile.notes)

    overall = _overall_disposition([item.disposition for item in records])
    hard_ids = tuple(
        item.site_id for item in records if item.disposition is ConstraintDisposition.HARD_EXCLUSION
    )
    penalty_ids = tuple(
        item.site_id for item in records if item.disposition is ConstraintDisposition.PENALTY
    )
    warning_ids = tuple(
        item.site_id for item in records if item.disposition is ConstraintDisposition.WARNING
    )
    if hard_ids:
        warnings.append(f"hard_exclusion sites={','.join(hard_ids)}")
    if penalty_ids:
        warnings.append(f"penalty sites={','.join(penalty_ids)}")
    if warning_ids:
        warnings.append(f"warning sites={','.join(warning_ids)}")

    summary = ProtectedOverlapSummary(
        aoi_area_m2=aoi_area,
        intersection_area_m2=intersection_area,
        sum_of_feature_intersections_m2=feature_sum,
        overlap_fraction=overlap_fraction,
        overlap_percent=100.0 * overlap_fraction,
        union_is_exact=union_exact,
        site_ids=tuple(item.site_id for item in records),
        categories=_unique_labels([item.category for item in records]),
        designations=_unique_labels([item.designation for item in records]),
        records=tuple(records),
        profile_id=None if profile is None else profile.profile_id,
        overall_disposition=overall,
        hard_exclusion_site_ids=hard_ids,
        penalty_site_ids=penalty_ids,
        warning_site_ids=warning_ids,
        area_method=_area_method(crs),
        warnings=tuple(warnings),
    )
    return ProtectedOverlapResult(summary=summary, layer=_layer_for(summary, aoi=aoi, crs=crs))


IUCN_IA_IB_HARD_EXCLUSION = ProtectedAreaConstraintProfile(
    profile_id="iucn_ia_ib_hard_exclusion",
    rules=(
        ProtectedAreaRule(
            categories=("Ia", "Ib"),
            disposition=ConstraintDisposition.HARD_EXCLUSION,
            notes=("Named opt-in: IUCN Ia/Ib only. Other classes are not excluded.",),
        ),
    ),
    unmatched_disposition=ConstraintDisposition.INFORMATIONAL,
    notes=(
        "Only IUCN category Ia and Ib are hard exclusions under this profile.",
        "Other IUCN categories, Natura 2000 types, and undesignated sites stay informational.",
    ),
)

NAMED_CONSTRAINT_PROFILES: dict[str, ProtectedAreaConstraintProfile] = {
    IUCN_IA_IB_HARD_EXCLUSION.profile_id: IUCN_IA_IB_HARD_EXCLUSION,
}


def _as_disposition(value: ConstraintDisposition | str | None) -> ConstraintDisposition | None:
    if value is None:
        return None
    if isinstance(value, ConstraintDisposition):
        return value
    token = value.strip()
    try:
        return ConstraintDisposition(token)
    except ValueError as exc:
        raise InvalidParameterError(f"unknown constraint disposition {value!r}") from exc


def resolve_constraint_profile(
    *,
    profile_id: str | None = None,
    hard_exclusion_categories: tuple[str, ...] | None = None,
    warning_categories: tuple[str, ...] | None = None,
    penalty_categories: tuple[str, ...] | None = None,
    penalty_weight: float | None = None,
    unmatched_disposition: ConstraintDisposition | str | None = None,
    catalog: dict[str, ProtectedAreaConstraintProfile] | None = None,
) -> ProtectedAreaConstraintProfile | None:
    """Look up or build a profile. ``None`` leaves overlap informational."""

    has_lists = bool(hard_exclusion_categories or warning_categories or penalty_categories)
    unmatched = _as_disposition(unmatched_disposition)
    if has_lists:
        rules: list[ProtectedAreaRule] = []
        if hard_exclusion_categories:
            rules.append(
                ProtectedAreaRule(
                    categories=hard_exclusion_categories,
                    disposition=ConstraintDisposition.HARD_EXCLUSION,
                )
            )
        if warning_categories:
            rules.append(
                ProtectedAreaRule(
                    categories=warning_categories,
                    disposition=ConstraintDisposition.WARNING,
                )
            )
        if penalty_categories:
            rules.append(
                ProtectedAreaRule(
                    categories=penalty_categories,
                    disposition=ConstraintDisposition.PENALTY,
                    penalty_weight=1.0 if penalty_weight is None else penalty_weight,
                )
            )
        return ProtectedAreaConstraintProfile(
            profile_id=profile_id or "caller_supplied",
            rules=tuple(rules),
            unmatched_disposition=unmatched or ConstraintDisposition.INFORMATIONAL,
            notes=("caller-supplied protected-area class dispositions",),
        )
    if profile_id:
        table = catalog if catalog is not None else NAMED_CONSTRAINT_PROFILES
        if profile_id not in table:
            raise InvalidParameterError(f"unknown protected-area constraint profile {profile_id!r}")
        return table[profile_id]
    if unmatched is not None:
        return ProtectedAreaConstraintProfile(
            profile_id="caller_unmatched",
            unmatched_disposition=unmatched,
            notes=("caller unmatched_disposition with no class lists",),
        )
    return None


__all__ = [
    "AREA_METHOD_LOCAL_EQUIRECTANGULAR",
    "AREA_METHOD_PROJECTED",
    "CALC_PROTECTED_OVERLAP",
    "IUCN_IA_IB_HARD_EXCLUSION",
    "NAMED_CONSTRAINT_PROFILES",
    "ConstraintDisposition",
    "ProtectedAreaConstraintProfile",
    "ProtectedAreaFeature",
    "ProtectedAreaRule",
    "ProtectedOverlapRecord",
    "ProtectedOverlapResult",
    "ProtectedOverlapSummary",
    "analyze_point_in_protected",
    "analyze_protected_overlap",
    "bbox_ring",
    "clip_polygon",
    "disposition_for",
    "ensure_ccw",
    "hard_exclusion_applies",
    "resolve_constraint_profile",
    "ring_to_local",
]
