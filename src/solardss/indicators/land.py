"""Land-cover zonal statistics and explicit available-area profiles.

This module does not open rasters or download ESA WorldCover tiles. A provider
supplies a ``LandCoverGrid`` plus a reviewable ``LandCoverLegend``. Available
area is computed only from an ``AvailableLandProfile``; non-water pixels are
never treated as PV-suitable by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from solardss.core.arrays import AnyArray, BoolArray, FloatArray, IntArray
from solardss.core.hashing import fingerprint_array
from solardss.core.methods import MethodId
from solardss.core.results import LayerResult, summarize_array
from solardss.core.status import EvaluationStatus
from solardss.exceptions import InvalidGridError, InvalidParameterError
from solardss.provenance.record import provenance_record

CALC_CLASS_FRACTION = "land_fraction_from_cover"
CALC_DOMINANT_CLASS = "land_dominant_class"
CALC_AVAILABLE_AREA = "land_available_area_profile"
AREA_METHOD_PROJECTED = "projected_cell_m2"
AREA_METHOD_AUTHALIC = "wgs84_authalic_trapezoid"
AREA_METHOD_CALLER = "caller_cell_area_m2"
UNMAPPED_CATEGORY = "unmapped"

# WGS84 authalic radius (m). Trapezoid area matches the equal-area sphere.
WGS84_AUTHALIC_RADIUS_M = 6371007.180918475

_GEOGRAPHIC_EXACT = frozenset(
    {
        "EPSG:4326",
        "EPSG:4269",
        "EPSG:4258",
        "CRS:84",
        "OGC:CRS84",
        "WGS84",
        "WGS 84",
    }
)

FRACTION_INDICATORS: dict[str, str] = {
    "LAND_FOREST_FRACTION": "forest",
    "LAND_WATER_FRACTION": "water",
    "LAND_WETLAND_FRACTION": "wetland",
    "LAND_AGRICULTURAL_FRACTION": "agricultural",
    "LAND_BUILTUP_FRACTION": "builtup",
    "LAND_BARE_LAND_FRACTION": "bare_land",
}


def crs_is_geographic(crs: str | None) -> bool:
    """True when ``crs`` names a geographic (degree) coordinate system."""

    if crs is None:
        return False
    compact = crs.strip().upper().replace(" ", "")
    if compact in {item.replace(" ", "") for item in _GEOGRAPHIC_EXACT}:
        return True
    if compact.startswith("EPSG:4326") or compact.startswith("EPSG:4269"):
        return True
    return "GEOGCS" in compact or "GEOGRAPHIC" in compact


class LandCoverClass(BaseModel):
    """One dataset class code and the SolarDSS category it maps to."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: int
    name: str = Field(min_length=1)
    category: str = Field(min_length=1)


class LandCoverLegend(BaseModel):
    """Explicit, reviewable mapping from integer codes to SolarDSS categories."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(min_length=1)
    classes: tuple[LandCoverClass, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_codes(self) -> LandCoverLegend:
        codes = [item.code for item in self.classes]
        if len(codes) != len(set(codes)):
            raise InvalidParameterError("land-cover legend codes must be unique")
        return self

    def category_for(self, code: int) -> str | None:
        """Return the SolarDSS category for ``code``, or None if unmapped."""

        for item in self.classes:
            if item.code == code:
                return item.category
        return None

    def name_for(self, code: int) -> str | None:
        for item in self.classes:
            if item.code == code:
                return item.name
        return None

    def codes_for(self, category: str) -> tuple[int, ...]:
        return tuple(item.code for item in self.classes if item.category == category)

    def as_code_map(self) -> dict[int, str]:
        return {item.code: item.category for item in self.classes}


class AvailableLandProfile(BaseModel):
    """Which cover classes may count as available land for PV.

    A cell is available when it matches an available category/code **and** does
    not match an excluded category/code. Exclusion wins. Empty available lists
    are rejected: that is not a silent 'everything except water' rule.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_id: str = Field(min_length=1)
    available_categories: tuple[str, ...] = ()
    excluded_categories: tuple[str, ...] = ()
    available_codes: tuple[int, ...] = ()
    excluded_codes: tuple[int, ...] = ()
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _has_available(self) -> AvailableLandProfile:
        if not self.available_categories and not self.available_codes:
            raise InvalidParameterError(
                "AvailableLandProfile requires available_categories or available_codes; "
                "non-water is not available by default"
            )
        return self


@dataclass(frozen=True, slots=True)
class LandCoverGrid:
    """North-up classified raster. Row index increases south, column increases east.

    ``cell_size_x`` / ``cell_size_y`` are positive spacings in the CRS unit.
    Geographic grids use degree spacings; area then uses WGS84 authalic trapezoids
    unless ``cell_area_m2`` is supplied.
    """

    codes: IntArray
    cell_size_x: float
    cell_size_y: float
    nodata: int | None = 0
    crs: str | None = None
    origin_x: float | None = None
    origin_y: float | None = None
    cell_area_m2: FloatArray | None = None

    def validated(self) -> LandCoverGrid:
        codes = np.asarray(self.codes)
        if codes.ndim != 2:
            raise InvalidGridError("land-cover grid must be 2-dimensional")
        if self.cell_size_x <= 0.0 or self.cell_size_y <= 0.0:
            raise InvalidGridError("cell_size_x and cell_size_y must be positive")
        if self.cell_area_m2 is not None and np.asarray(self.cell_area_m2).shape != codes.shape:
            raise InvalidGridError("cell_area_m2 must match the land-cover grid shape")
        if (
            crs_is_geographic(self.crs)
            and self.cell_area_m2 is None
            and (self.origin_x is None or self.origin_y is None)
        ):
            raise InvalidGridError(
                "geographic land-cover grids need origin_x/origin_y for geodesic area"
            )
        return self


class ClassAreaShare(BaseModel):
    """Area and fraction of one class code or aggregated category."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    code: int | None = None
    cell_count: int
    area_m2: float
    fraction: float
    percent: float


class LandCoverSummary(BaseModel):
    """Zonal land-cover accounting for one mask."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    valid_cell_count: int
    nodata_cell_count: int
    unmapped_cell_count: int
    total_area_m2: float
    valid_area_m2: float
    nodata_area_m2: float
    dominant_category: str | None = None
    dominant_code: int | None = None
    dominant_name: str | None = None
    category_shares: tuple[ClassAreaShare, ...] = ()
    code_shares: tuple[ClassAreaShare, ...] = ()
    available_area_m2: float | None = None
    available_fraction: float | None = None
    available_cell_count: int | None = None
    profile_id: str | None = None
    area_method: str = Field(min_length=1)
    tie_broken_by_lowest_code: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LandCoverAnalysisResult:
    """Full-grid categories plus zonal summary and provenance layer."""

    summary: LandCoverSummary
    categories: AnyArray
    cell_area_m2: FloatArray
    available_mask: BoolArray | None
    layer: LayerResult


def authalic_rectangle_area_m2(
    lat_north_deg: FloatArray | float,
    lat_south_deg: FloatArray | float,
    dlon_deg: float,
) -> FloatArray:
    """WGS84 authalic-sphere area of a lat/lon rectangle, in square metres."""

    radius = WGS84_AUTHALIC_RADIUS_M
    phi_north = np.deg2rad(np.asarray(lat_north_deg, dtype=np.float64))
    phi_south = np.deg2rad(np.asarray(lat_south_deg, dtype=np.float64))
    dlam = np.deg2rad(abs(float(dlon_deg)))
    area = radius * radius * dlam * np.abs(np.sin(phi_north) - np.sin(phi_south))
    return np.asarray(area, dtype=np.float64)


def cell_area_m2(grid: LandCoverGrid) -> tuple[FloatArray, str]:
    """Per-cell area. Projected grids use dx*dy; geographic grids use authalic trapezoids."""

    grid = grid.validated()
    rows, cols = np.asarray(grid.codes).shape
    if grid.cell_area_m2 is not None:
        areas = np.asarray(grid.cell_area_m2, dtype=np.float64)
        if np.any(~np.isfinite(areas)) or np.any(areas < 0.0):
            raise InvalidGridError("cell_area_m2 must be finite and non-negative")
        return areas, AREA_METHOD_CALLER
    if crs_is_geographic(grid.crs):
        origin_y = float(grid.origin_y) if grid.origin_y is not None else 0.0
        dlat = float(grid.cell_size_y)
        dlon = float(grid.cell_size_x)
        row = np.arange(rows, dtype=np.float64)
        lat_north = origin_y - row * dlat
        lat_south = lat_north - dlat
        if np.any(lat_north > 90.0) or np.any(lat_south < -90.0):
            raise InvalidGridError("geographic land-cover rows fall outside [-90, 90]")
        row_area = authalic_rectangle_area_m2(lat_north, lat_south, dlon)
        areas = np.repeat(row_area.reshape(rows, 1), cols, axis=1)
        return np.asarray(areas, dtype=np.float64), AREA_METHOD_AUTHALIC
    uniform = float(grid.cell_size_x) * float(grid.cell_size_y)
    return np.full((rows, cols), uniform, dtype=np.float64), AREA_METHOD_PROJECTED


def resolve_available_profile(
    *,
    profile_id: str | None = None,
    available_categories: tuple[str, ...] | None = None,
    excluded_categories: tuple[str, ...] | None = None,
    available_codes: tuple[int, ...] | None = None,
    excluded_codes: tuple[int, ...] | None = None,
    catalog: dict[str, AvailableLandProfile] | None = None,
) -> AvailableLandProfile | None:
    """Build or look up a profile. ``None`` means available area is not requested."""

    has_lists = bool(available_categories) or bool(available_codes)
    if has_lists:
        return AvailableLandProfile(
            profile_id=profile_id or "caller_supplied",
            available_categories=available_categories or (),
            excluded_categories=excluded_categories or (),
            available_codes=available_codes or (),
            excluded_codes=excluded_codes or (),
            notes=("caller-supplied available/excluded class lists",),
        )
    if profile_id:
        table = catalog or {}
        if profile_id not in table:
            raise InvalidParameterError(f"unknown available-land profile {profile_id!r}")
        return table[profile_id]
    return None


def category_fraction(summary: LandCoverSummary, category: str) -> float:
    """Valid-area fraction of ``category``. Missing categories are 0, not invented cover."""

    for share in summary.category_shares:
        if share.key == category:
            return share.fraction
    return 0.0


def available_mask(
    categories: AnyArray,
    codes: IntArray,
    profile: AvailableLandProfile,
    *,
    valid: BoolArray,
) -> BoolArray:
    """True where the profile allows the cell. Exclusion wins over inclusion."""

    cat = np.asarray(categories)
    code = np.asarray(codes)
    included = np.zeros(cat.shape, dtype=bool)
    if profile.available_categories:
        included |= np.isin(cat, np.asarray(profile.available_categories))
    if profile.available_codes:
        included |= np.isin(code, np.asarray(profile.available_codes, dtype=code.dtype))
    excluded = np.zeros(cat.shape, dtype=bool)
    if profile.excluded_categories:
        excluded |= np.isin(cat, np.asarray(profile.excluded_categories))
    if profile.excluded_codes:
        excluded |= np.isin(code, np.asarray(profile.excluded_codes, dtype=code.dtype))
    return np.asarray(valid & included & ~excluded, dtype=bool)


def analyze_land_cover(
    grid: LandCoverGrid,
    legend: LandCoverLegend,
    *,
    mask: BoolArray | None = None,
    profile: AvailableLandProfile | None = None,
    constraint_mask: BoolArray | None = None,
) -> LandCoverAnalysisResult:
    """Area, percent, and dominant class inside ``mask``.

    ``constraint_mask`` is an extra AND applied only to available area (for
    example a slope or cadastre mask). It does not change class fractions.
    """

    grid = grid.validated()
    codes = np.asarray(grid.codes, dtype=np.int16)
    rows, cols = codes.shape
    select: BoolArray
    if mask is None:
        select = np.ones((rows, cols), dtype=bool)
    else:
        if np.asarray(mask).shape != codes.shape:
            raise InvalidGridError("land-cover mask must match the classification grid shape")
        select = np.asarray(mask, dtype=bool)
    if constraint_mask is not None and np.asarray(constraint_mask).shape != codes.shape:
        raise InvalidGridError("constraint_mask must match the classification grid shape")

    areas, area_method = cell_area_m2(grid)
    nodata = grid.nodata
    finite = np.ones(codes.shape, dtype=bool) if nodata is None else codes != int(nodata)
    valid = select & finite
    nodata_sel = select & ~finite
    code_map = legend.as_code_map()
    categories = np.empty(codes.shape, dtype=object)
    categories[:] = UNMAPPED_CATEGORY
    mapped = np.zeros(codes.shape, dtype=bool)
    for code, category in code_map.items():
        hit = codes == np.int16(code)
        categories[hit] = category
        mapped[hit] = True
    unmapped = valid & ~mapped
    valid_mapped = valid & mapped

    total_area = float(areas[select].sum()) if np.any(select) else 0.0
    valid_area = float(areas[valid].sum()) if np.any(valid) else 0.0
    nodata_area = float(areas[nodata_sel].sum()) if np.any(nodata_sel) else 0.0
    warnings: list[str] = []
    if grid.crs is None:
        warnings.append("CRS omitted; cell sizes are treated as metres")
    if np.any(unmapped):
        warnings.append(
            f"{int(unmapped.sum())} valid cells have codes outside the legend (counted as unmapped)"
        )
    if valid_area == 0.0:
        warnings.append("no valid land-cover samples in the analysis mask")

    code_shares = _shares_by_key(
        keys=codes[valid],
        areas=areas[valid],
        valid_area=valid_area,
        as_code=True,
    )
    category_keys = np.asarray(categories[valid], dtype=object)
    category_shares = _shares_by_key(
        keys=category_keys,
        areas=areas[valid],
        valid_area=valid_area,
        as_code=False,
    )

    dominant_code: int | None = None
    dominant_category: str | None = None
    dominant_name: str | None = None
    tie = False
    mapped_code_shares = [item for item in code_shares if item.key != str(nodata)]
    if mapped_code_shares and valid_area > 0.0:
        mapped_only = [item for item in mapped_code_shares if legend.category_for(int(item.key))]
        pool = mapped_only or mapped_code_shares
        max_area = max(item.area_m2 for item in pool)
        tied = [item for item in pool if item.area_m2 == max_area]
        tie = len(tied) > 1
        chosen = min(tied, key=lambda item: int(item.key))
        dominant_code = int(chosen.key)
        dominant_category = legend.category_for(dominant_code) or UNMAPPED_CATEGORY
        dominant_name = legend.name_for(dominant_code)
        if tie:
            warnings.append(
                f"dominant-class tie on area={max_area}; selected lowest code {dominant_code}"
            )

    avail_mask: BoolArray | None = None
    available_area: float | None = None
    available_fraction: float | None = None
    available_count: int | None = None
    profile_id: str | None = None
    if profile is not None:
        profile_id = profile.profile_id
        avail_mask = available_mask(categories, codes, profile, valid=valid_mapped)
        if constraint_mask is not None:
            avail_mask = np.asarray(
                avail_mask & np.asarray(constraint_mask, dtype=bool), dtype=bool
            )
            warnings.append("available area intersected with an extra constraint mask")
        available_count = int(avail_mask.sum())
        available_area = float(areas[avail_mask].sum()) if available_count else 0.0
        available_fraction = (available_area / valid_area) if valid_area else 0.0
        notes = ", ".join(profile.notes) if profile.notes else "none"
        warnings.append(
            f"available-land profile={profile.profile_id}; "
            f"available_categories={list(profile.available_categories)}; "
            f"excluded_categories={list(profile.excluded_categories)}; notes={notes}"
        )

    summary = LandCoverSummary(
        valid_cell_count=int(valid.sum()),
        nodata_cell_count=int(nodata_sel.sum()),
        unmapped_cell_count=int(unmapped.sum()),
        total_area_m2=total_area,
        valid_area_m2=valid_area,
        nodata_area_m2=nodata_area,
        dominant_category=dominant_category,
        dominant_code=dominant_code,
        dominant_name=dominant_name,
        category_shares=category_shares,
        code_shares=code_shares,
        available_area_m2=available_area,
        available_fraction=available_fraction,
        available_cell_count=available_count,
        profile_id=profile_id,
        area_method=area_method,
        tie_broken_by_lowest_code=tie,
        warnings=tuple(warnings),
    )
    parameters: dict[str, object] = {
        "dataset_id": legend.dataset_id,
        "crs": grid.crs,
        "cell_size_x": float(grid.cell_size_x),
        "cell_size_y": float(grid.cell_size_y),
        "nodata": grid.nodata,
        "area_method": area_method,
        "profile_id": profile_id,
        "available_categories": list(profile.available_categories) if profile else [],
        "excluded_categories": list(profile.excluded_categories) if profile else [],
        "dominant_category": dominant_category,
        "dominant_code": dominant_code,
        "valid_area_m2": valid_area,
        "available_area_m2": available_area,
    }
    for share in category_shares:
        parameters[f"area_m2_{share.key}"] = share.area_m2
        parameters[f"percent_{share.key}"] = share.percent
        parameters[f"fraction_{share.key}"] = share.fraction
    layer = LayerResult(
        name="land_cover",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.LAND_CLASS_AREA_FRACTION.value,
        provenance=provenance_record(
            MethodId.LAND_CLASS_AREA_FRACTION,
            parameters=parameters,
            input_fingerprints=(fingerprint_array(codes), fingerprint_array(areas)),
            notes=tuple(warnings),
        ),
        summary=summarize_array(areas),
        warnings=tuple(warnings),
    )
    return LandCoverAnalysisResult(
        summary=summary,
        categories=categories,
        cell_area_m2=areas,
        available_mask=avail_mask,
        layer=layer,
    )


def _shares_by_key(
    keys: AnyArray,
    areas: FloatArray,
    valid_area: float,
    *,
    as_code: bool,
) -> tuple[ClassAreaShare, ...]:
    if keys.size == 0:
        return ()
    unique, inverse = np.unique(keys, return_inverse=True)
    area_sums = np.bincount(inverse, weights=np.asarray(areas, dtype=np.float64))
    counts = np.bincount(inverse)
    shares: list[ClassAreaShare] = []
    for index, key in enumerate(unique):
        area = float(area_sums[index])
        fraction = (area / valid_area) if valid_area else 0.0
        label = str(int(key)) if as_code else str(key)
        code = int(key) if as_code else None
        shares.append(
            ClassAreaShare(
                key=label,
                code=code,
                cell_count=int(counts[index]),
                area_m2=area,
                fraction=fraction,
                percent=100.0 * fraction,
            )
        )
    shares.sort(key=lambda item: (-item.area_m2, item.key))
    return tuple(shares)


__all__ = [
    "AREA_METHOD_AUTHALIC",
    "AREA_METHOD_CALLER",
    "AREA_METHOD_PROJECTED",
    "CALC_AVAILABLE_AREA",
    "CALC_CLASS_FRACTION",
    "CALC_DOMINANT_CLASS",
    "FRACTION_INDICATORS",
    "UNMAPPED_CATEGORY",
    "WGS84_AUTHALIC_RADIUS_M",
    "AvailableLandProfile",
    "ClassAreaShare",
    "LandCoverAnalysisResult",
    "LandCoverClass",
    "LandCoverGrid",
    "LandCoverLegend",
    "LandCoverSummary",
    "analyze_land_cover",
    "authalic_rectangle_area_m2",
    "available_mask",
    "category_fraction",
    "cell_area_m2",
    "crs_is_geographic",
    "resolve_available_profile",
]
