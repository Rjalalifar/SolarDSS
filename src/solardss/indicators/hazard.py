"""Hazard-map sampling with explicit source legends.

SolarDSS does not model flood, wildfire, landslide, or hail. A provider supplies
a raster or vector layer. Categories such as low/medium/high are recorded only
when a ``HazardValueMapping`` from that dataset's legend is supplied. Empty
mappings leave the raw value uncategorised. There is no global hazard dataset
and no default tertile scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from solardss.core.arrays import BoolArray, FloatArray
from solardss.core.hashing import fingerprint_array, fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.exceptions import InvalidGridError, InvalidParameterError
from solardss.indicators.ids import IndicatorId
from solardss.indicators.infrastructure import point_in_ring
from solardss.provenance.record import provenance_record

CALC_HAZARD_SAMPLE = "hazard_source_mapping"


class HazardKind(str, Enum):
    """Hazard family. Each maps to one registered indicator."""

    FLOOD = "flood"
    WILDFIRE = "wildfire"
    LANDSLIDE = "landslide"
    HAIL = "hail"


class HazardAggregation(str, Enum):
    """How multiple cells or polygons are reduced. Never implied by the dataset."""

    CELL = "cell"
    MAJORITY = "majority"
    MAX = "max"
    MIN = "min"
    MEAN = "mean"


KIND_TO_INDICATOR: dict[HazardKind, str] = {
    HazardKind.FLOOD: IndicatorId.HAZARD_FLOOD_RISK.value,
    HazardKind.WILDFIRE: IndicatorId.HAZARD_WILDFIRE_RISK.value,
    HazardKind.LANDSLIDE: IndicatorId.HAZARD_LANDSLIDE_RISK.value,
    HazardKind.HAIL: IndicatorId.HAZARD_HAIL_RISK.value,
}

INDICATOR_TO_KIND: dict[str, HazardKind] = {
    indicator: kind for kind, indicator in KIND_TO_INDICATOR.items()
}


class HazardClass(BaseModel):
    """One source code or token and the category the dataset legend assigns it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: str = Field(min_length=1)
    category: str = Field(min_length=1)
    score: float | None = None
    notes: tuple[str, ...] = ()


class HazardBin(BaseModel):
    """One explicit numeric interval from a dataset legend. Edges are not inferred."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_value: float
    max_value: float
    category: str = Field(min_length=1)
    score: float | None = None
    max_inclusive: bool = False
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _ordered(self) -> HazardBin:
        if self.max_value < self.min_value:
            raise InvalidParameterError("HazardBin max_value must be >= min_value")
        return self


class HazardValueMapping(BaseModel):
    """Source-specific legend. Empty mappings do not invent Low/Medium/High."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mapping_id: str = Field(min_length=1)
    dataset_id: str = Field(min_length=1)
    classes: tuple[HazardClass, ...] = ()
    bins: tuple[HazardBin, ...] = ()
    continuous: bool = False
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _explicit_legend(self) -> HazardValueMapping:
        if self.classes and self.bins:
            raise InvalidParameterError(
                "HazardValueMapping cannot mix class codes and numeric bins"
            )
        if self.continuous and self.classes:
            raise InvalidParameterError(
                "continuous=True is for numeric scores; class codes are not continuous"
            )
        raws = [item.raw.strip().casefold() for item in self.classes]
        if len(raws) != len(set(raws)):
            raise InvalidParameterError("HazardValueMapping class raw tokens must be unique")
        return self

    def has_legend(self) -> bool:
        return bool(self.classes) or bool(self.bins)


class HazardLayerMeta(BaseModel):
    """Dataset identity recorded on every sample. Nothing here is a risk model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_name: str = Field(min_length=1)
    source_dataset: str = Field(min_length=1)
    source_version: str | None = None
    source_year: int | None = None
    resolution: str | None = None
    return_period: str | None = None
    crs: str | None = None
    uncertainty_notes: tuple[str, ...] = ()


class HazardRecord(BaseModel):
    """Structured hazard sample: raw value plus optional mapped category."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: HazardKind
    indicator_id: str = Field(min_length=1)
    raw_value: str | float | None = None
    category: str | None = None
    score: float | None = None
    source_category: str | None = None
    source_name: str = Field(min_length=1)
    source_dataset: str = Field(min_length=1)
    source_version: str | None = None
    source_year: int | None = None
    resolution: str | None = None
    return_period: str | None = None
    aggregation: HazardAggregation = HazardAggregation.CELL
    mapping_id: str | None = None
    candidate_id: str | None = None
    quality_flags: tuple[str, ...] = ()
    uncertainty_notes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    missing: bool = False


class HazardPolygon(BaseModel):
    """One vector hazard feature. Coordinates are ``(x, y)`` / ``(lon, lat)``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    site_id: str = Field(min_length=1)
    coordinates: tuple[tuple[float, float], ...] = Field(min_length=3)
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()
    raw_value: str | float | None = None
    source_category: str | None = None
    return_period: str | None = None
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class HazardGrid:
    """North-up hazard raster. Row index increases south, column increases east."""

    values: FloatArray
    cell_size_x: float
    cell_size_y: float
    origin_x: float
    origin_y: float
    nodata: float | None = None
    crs: str | None = None

    def validated(self) -> HazardGrid:
        values = np.asarray(self.values)
        if values.ndim != 2:
            raise InvalidGridError("hazard grid must be 2-dimensional")
        if self.cell_size_x <= 0.0 or self.cell_size_y <= 0.0:
            raise InvalidGridError("cell_size_x and cell_size_y must be positive")
        return self


@dataclass(frozen=True, slots=True)
class HazardSample:
    """Raw cells reduced to one source value, before legend mapping."""

    raw_value: float | str | None
    nodata: bool
    sample_count: int
    aggregation: HazardAggregation
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class HazardAnalysisResult:
    """Mapped record plus a provenance layer."""

    record: HazardRecord
    sample: HazardSample
    layer: LayerResult


def raw_key(value: str | float | int) -> str:
    """Canonical lookup key for a source code or token."""

    if isinstance(value, bool):
        raise InvalidParameterError("boolean is not a hazard source value")
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise InvalidParameterError("hazard raw value is not finite")
        if value.is_integer():
            return str(int(value))
        return format(value, ".12g")
    if isinstance(value, int):
        return str(value)
    token = value.strip()
    if not token:
        raise InvalidParameterError("hazard raw token is empty")
    return token


def _class_lookup(mapping: HazardValueMapping) -> dict[str, HazardClass]:
    table: dict[str, HazardClass] = {}
    for item in mapping.classes:
        table[item.raw.strip().casefold()] = item
        table[raw_key(item.raw).casefold()] = item
    return table


def map_hazard_value(
    raw: str | float | None,
    mapping: HazardValueMapping | None,
) -> tuple[str | None, float | None, tuple[str, ...]]:
    """Apply an explicit legend. Missing mapping leaves category unset."""

    if raw is None:
        return None, None, ("raw hazard value is missing",)
    warnings: list[str] = []
    if mapping is None or not mapping.has_legend():
        warnings.append(
            "no source legend; category is not inferred and Low/Medium/High are not assigned"
        )
        return None, None, tuple(warnings)
    if mapping.classes:
        key = raw_key(raw).casefold()
        hit = _class_lookup(mapping).get(key)
        if hit is None:
            warnings.append(
                f"raw value {raw!r} is not in mapping {mapping.mapping_id}; category is not guessed"
            )
            return None, None, tuple(warnings)
        return hit.category, hit.score, tuple(warnings)
    numeric: float
    if isinstance(raw, str):
        try:
            numeric = float(raw.strip())
        except ValueError:
            warnings.append(
                f"raw value {raw!r} is not numeric; bin legend {mapping.mapping_id} does not apply"
            )
            return None, None, tuple(warnings)
    else:
        numeric = float(raw)
    for bin_rule in mapping.bins:
        if numeric < bin_rule.min_value:
            continue
        if bin_rule.max_inclusive:
            inside = numeric <= bin_rule.max_value
        else:
            inside = numeric < bin_rule.max_value or (
                numeric == bin_rule.max_value and bin_rule.min_value == bin_rule.max_value
            )
        if inside:
            return bin_rule.category, bin_rule.score, tuple(warnings)
    warnings.append(
        f"raw value {numeric} falls outside bins of mapping {mapping.mapping_id}; "
        "category is not guessed"
    )
    return None, None, tuple(warnings)


def _cell_index(grid: HazardGrid, x: float, y: float) -> tuple[int, int] | None:
    values = np.asarray(grid.values)
    rows, cols = values.shape
    west = float(grid.origin_x)
    north = float(grid.origin_y)
    east = west + cols * float(grid.cell_size_x)
    south = north - rows * float(grid.cell_size_y)
    if x < west or x > east or y < south or y > north:
        return None
    col = int(np.floor((x - west) / float(grid.cell_size_x)))
    row = int(np.floor((north - y) / float(grid.cell_size_y)))
    if col == cols:
        col = cols - 1
    if row == rows:
        row = rows - 1
    if row < 0 or col < 0 or row >= rows or col >= cols:
        return None
    return row, col


def _is_nodata(value: float, nodata: float | None) -> bool:
    if value != value:
        return True
    if nodata is None:
        return False
    return bool(value == float(nodata))


def cell_centers(grid: HazardGrid) -> tuple[FloatArray, FloatArray]:
    """Centre coordinates of each cell, north-up."""

    checked = grid.validated()
    rows, cols = np.asarray(checked.values).shape
    xs = float(checked.origin_x) + (np.arange(cols, dtype=np.float64) + 0.5) * float(
        checked.cell_size_x
    )
    ys = float(checked.origin_y) - (np.arange(rows, dtype=np.float64) + 0.5) * float(
        checked.cell_size_y
    )
    xx, yy = np.meshgrid(xs, ys)
    return np.asarray(xx, dtype=np.float64), np.asarray(yy, dtype=np.float64)


def mask_from_bbox(
    grid: HazardGrid,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> BoolArray:
    xx, yy = cell_centers(grid)
    return np.asarray(
        (xx >= min_x) & (xx <= max_x) & (yy >= min_y) & (yy <= max_y),
        dtype=bool,
    )


def sample_hazard_cell(grid: HazardGrid, x: float, y: float) -> HazardSample:
    """Nearest containing cell. Out-of-raster points stay missing."""

    checked = grid.validated()
    index = _cell_index(checked, x, y)
    if index is None:
        return HazardSample(
            raw_value=None,
            nodata=True,
            sample_count=0,
            aggregation=HazardAggregation.CELL,
            warnings=("point does not intersect the hazard raster",),
        )
    row, col = index
    value = float(np.asarray(checked.values)[row, col])
    if _is_nodata(value, checked.nodata):
        return HazardSample(
            raw_value=None,
            nodata=True,
            sample_count=0,
            aggregation=HazardAggregation.CELL,
            warnings=("hazard cell is nodata",),
        )
    return HazardSample(
        raw_value=value,
        nodata=False,
        sample_count=1,
        aggregation=HazardAggregation.CELL,
    )


def _finite_samples(grid: HazardGrid, mask: BoolArray) -> FloatArray:
    values = np.asarray(grid.values, dtype=np.float64)
    selected = np.asarray(values[np.asarray(mask, dtype=bool)], dtype=np.float64)
    if grid.nodata is not None:
        selected = selected[selected != float(grid.nodata)]
    return np.asarray(selected[np.isfinite(selected)], dtype=np.float64)


def aggregate_hazard_cells(
    grid: HazardGrid,
    mask: BoolArray,
    *,
    aggregation: HazardAggregation,
    mapping: HazardValueMapping | None = None,
) -> HazardSample:
    """Reduce masked cells. Mean is refused unless the legend marks values continuous."""

    checked = grid.validated()
    if aggregation is HazardAggregation.CELL:
        raise InvalidParameterError("CELL aggregation is only valid for a single point sample")
    if aggregation is HazardAggregation.MEAN and (mapping is None or not mapping.continuous):
        raise InvalidParameterError(
            "mean aggregation requires HazardValueMapping.continuous=True; "
            "class codes are not averaged and tertiles are not inferred"
        )
    samples = _finite_samples(checked, mask)
    if samples.size == 0:
        return HazardSample(
            raw_value=None,
            nodata=True,
            sample_count=0,
            aggregation=aggregation,
            warnings=("no valid hazard cells in the analysis mask",),
        )
    if aggregation is HazardAggregation.MAX:
        raw = float(np.max(samples))
    elif aggregation is HazardAggregation.MIN:
        raw = float(np.min(samples))
    elif aggregation is HazardAggregation.MEAN:
        raw = float(np.mean(samples))
    else:
        rounded = np.rint(samples)
        unique, counts = np.unique(rounded, return_counts=True)
        raw = float(unique[int(np.argmax(counts))])
    return HazardSample(
        raw_value=raw,
        nodata=False,
        sample_count=int(samples.size),
        aggregation=aggregation,
    )


def sample_hazard_polygon(
    x: float,
    y: float,
    features: Sequence[HazardPolygon],
) -> HazardSample:
    """First containing polygon, ties broken by site_id. Empty extracts stay empty."""

    hits: list[HazardPolygon] = []
    for feature in features:
        if not point_in_ring(x, y, feature.coordinates):
            continue
        in_hole = False
        for hole in feature.holes:
            if point_in_ring(x, y, hole):
                in_hole = True
                break
        if not in_hole:
            hits.append(feature)
    if not hits:
        return HazardSample(
            raw_value=None,
            nodata=True,
            sample_count=0,
            aggregation=HazardAggregation.CELL,
            warnings=("point does not intersect a hazard polygon",),
        )
    hits.sort(key=lambda item: item.site_id)
    chosen = hits[0]
    raw: str | float | None = chosen.raw_value
    if raw is None and chosen.source_category is not None:
        raw = chosen.source_category
    extra: tuple[str, ...] = ()
    if len(hits) > 1:
        extra = (f"overlapping polygons; selected site_id={chosen.site_id}",)
    return HazardSample(
        raw_value=raw,
        nodata=raw is None,
        sample_count=1,
        aggregation=HazardAggregation.CELL,
        warnings=extra,
    )


def analyze_hazard(
    sample: HazardSample,
    *,
    kind: HazardKind,
    meta: HazardLayerMeta,
    mapping: HazardValueMapping | None = None,
    source_category: str | None = None,
    candidate_id: str | None = None,
) -> HazardAnalysisResult:
    """Attach an explicit legend and provenance. Missing samples stay missing."""

    indicator = KIND_TO_INDICATOR[kind]
    category, score, map_notes = map_hazard_value(sample.raw_value, mapping)
    warnings = list(sample.warnings)
    warnings.extend(map_notes)
    warnings.extend(meta.uncertainty_notes)
    if mapping is not None:
        warnings.extend(mapping.notes)
        warnings.append(f"hazard mapping_id={mapping.mapping_id} dataset_id={mapping.dataset_id}")
    else:
        warnings.append("SolarDSS does not model this hazard; the value is a sampled map cell")
    missing = sample.nodata or sample.raw_value is None
    record = HazardRecord(
        kind=kind,
        indicator_id=indicator,
        raw_value=sample.raw_value,
        category=None if missing else category,
        score=None if missing else score,
        source_category=source_category,
        source_name=meta.source_name,
        source_dataset=meta.source_dataset,
        source_version=meta.source_version,
        source_year=meta.source_year,
        resolution=meta.resolution,
        return_period=meta.return_period,
        aggregation=sample.aggregation,
        mapping_id=None if mapping is None else mapping.mapping_id,
        candidate_id=candidate_id,
        quality_flags=(),
        uncertainty_notes=meta.uncertainty_notes,
        warnings=tuple(warnings),
        missing=missing,
    )
    parameters: dict[str, object] = {
        "kind": kind.value,
        "indicator_id": indicator,
        "raw_value": None if record.raw_value is None else record.raw_value,
        "category": record.category,
        "score": record.score,
        "source_name": meta.source_name,
        "source_dataset": meta.source_dataset,
        "source_version": meta.source_version,
        "source_year": meta.source_year,
        "resolution": meta.resolution,
        "return_period": meta.return_period,
        "aggregation": sample.aggregation.value,
        "mapping_id": record.mapping_id,
        "sample_count": sample.sample_count,
        "missing": missing,
        "category_inferred": False,
    }
    values: FloatArray
    if record.score is not None:
        values = np.asarray([record.score], dtype=np.float64)
    elif isinstance(record.raw_value, float):
        values = np.asarray([record.raw_value], dtype=np.float64)
    else:
        values = np.asarray([], dtype=np.float64)
    fingerprints = (
        fingerprint_canonical(parameters),
        fingerprint_array(values) if values.size else fingerprint_canonical({"empty": True}),
    )
    layer = LayerResult(
        name=f"hazard_{kind.value}",
        status=EvaluationStatus.INCOMPLETE_INPUTS if missing else EvaluationStatus.EVALUATED,
        method_id=MethodId.HAZARD_SOURCE_MAPPING.value,
        provenance=provenance_record(
            MethodId.HAZARD_SOURCE_MAPPING,
            parameters=parameters,
            input_fingerprints=fingerprints,
            notes=tuple(warnings),
        ),
        summary=summarize_array(values) if values.size else None,
        warnings=tuple(warnings),
    )
    return HazardAnalysisResult(record=record, sample=sample, layer=layer)


__all__ = [
    "CALC_HAZARD_SAMPLE",
    "INDICATOR_TO_KIND",
    "KIND_TO_INDICATOR",
    "HazardAggregation",
    "HazardAnalysisResult",
    "HazardBin",
    "HazardClass",
    "HazardGrid",
    "HazardKind",
    "HazardLayerMeta",
    "HazardPolygon",
    "HazardRecord",
    "HazardSample",
    "HazardValueMapping",
    "aggregate_hazard_cells",
    "analyze_hazard",
    "cell_centers",
    "map_hazard_value",
    "mask_from_bbox",
    "raw_key",
    "sample_hazard_cell",
    "sample_hazard_polygon",
]
