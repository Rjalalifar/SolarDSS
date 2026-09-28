"""Candidate-generation contracts. Resolution is never implied."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solardss.core.jsonutil import JsonValue
from solardss.core.quantity import Quantity
from solardss.data.geo import BoundingBox
from solardss.data.tables import DEFAULT_ID_COLUMN, CandidateTable, validate_candidate_table
from solardss.exceptions import InvalidParameterError
from solardss.indicators.contracts import SiteSelectionScope
from solardss.provenance.record import ProvenanceRecord


class GenerationStrategy(str, Enum):
    """How locations are obtained. Ranking never invents a grid."""

    REGULAR_GRID = "regular_grid"
    HEXAGONAL_GRID = "hexagonal_grid"
    FEATURES = "features"


class CandidateSource(str, Enum):
    """Where a candidate polygon came from."""

    REGULAR_GRID = "regular_grid"
    HEXAGONAL_GRID = "hexagonal_grid"
    BUILDING_FOOTPRINT = "building_footprint"
    PARCEL = "parcel"
    PREDEFINED_SITE = "predefined_site"
    FEATURE = "feature"


class PolygonPart(BaseModel):
    """One polygon shell plus optional holes. Rings are closed ``(x, y)`` vertices."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    exterior: tuple[tuple[float, float], ...] = Field(min_length=4)
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()


class CandidateGeometry(BaseModel):
    """CRS-labelled polygon or multipolygon. Coordinates are not reprojected here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    crs: str = Field(min_length=1)
    parts: tuple[PolygonPart, ...] = Field(min_length=1)

    @property
    def kind(self) -> str:
        return "polygon" if len(self.parts) == 1 else "multipolygon"


class BoundaryGeometry(BaseModel):
    """Clip / covering boundary in one CRS. Holes are interior exclusions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    crs: str = Field(min_length=1)
    exterior: tuple[tuple[float, float], ...] = Field(min_length=4)
    holes: tuple[tuple[tuple[float, float], ...], ...] = ()

    @classmethod
    def from_bbox(cls, bbox: BoundingBox) -> BoundaryGeometry:
        """Axis-aligned rectangle from a bounding box."""

        ring = (
            (bbox.min_lon, bbox.min_lat),
            (bbox.max_lon, bbox.min_lat),
            (bbox.max_lon, bbox.max_lat),
            (bbox.min_lon, bbox.max_lat),
            (bbox.min_lon, bbox.min_lat),
        )
        return cls(crs=bbox.crs, exterior=ring)


class CandidateGenerationConfig(BaseModel):
    """Grid or feature materialisation settings.

    ``cell_size`` is required for every grid strategy. SolarDSS does not pick a
    default resolution. ``cell_size_unit`` must match the CRS: ``m`` on a
    projected CRS, ``deg`` on a geographic CRS. Coordinates are not transformed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: GenerationStrategy
    crs: str = Field(min_length=1)
    cell_size: float | None = None
    cell_size_unit: str | None = None
    origin_x: float | None = None
    origin_y: float | None = None
    min_area_m2: float | None = None
    max_candidates: int | None = None
    id_template: str | None = None

    @model_validator(mode="after")
    def _consistent(self) -> CandidateGenerationConfig:
        if (self.origin_x is None) != (self.origin_y is None):
            raise InvalidParameterError("origin_x and origin_y must be supplied together")
        if self.min_area_m2 is not None and (
            self.min_area_m2 < 0.0 or self.min_area_m2 != self.min_area_m2
        ):
            raise InvalidParameterError("min_area_m2 must be a non-negative finite value")
        if self.max_candidates is not None and self.max_candidates < 1:
            raise InvalidParameterError("max_candidates must be a positive integer")
        if self.strategy is GenerationStrategy.FEATURES:
            if self.cell_size is not None or self.cell_size_unit is not None:
                raise InvalidParameterError(
                    "FEATURES strategy does not take cell_size; ranking uses supplied geometries"
                )
            return self
        if self.cell_size is None:
            raise InvalidParameterError(
                "grid generation requires cell_size; SolarDSS does not choose a resolution"
            )
        if self.cell_size_unit is None:
            raise InvalidParameterError(
                "grid generation requires cell_size_unit ('m' or 'deg') matching the CRS"
            )
        if self.cell_size_unit not in {"m", "deg"}:
            raise InvalidParameterError("cell_size_unit must be 'm' or 'deg'")
        if self.cell_size != self.cell_size or self.cell_size <= 0.0:
            raise InvalidParameterError("cell_size must be a positive finite value")
        return self


class InputFeature(BaseModel):
    """A caller-supplied candidate polygon. Ranking copies these without tessellating."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    geometry: CandidateGeometry
    source: CandidateSource = CandidateSource.FEATURE
    source_id: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CandidateFeature(BaseModel):
    """One generated or adopted candidate site."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_id: str = Field(min_length=1)
    geometry: CandidateGeometry
    area: Quantity
    source: CandidateSource
    source_id: str | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    provenance: ProvenanceRecord
    grid_row: int | None = None
    grid_col: int | None = None

    @model_validator(mode="after")
    def _area_unit(self) -> CandidateFeature:
        self.area.require_unit("m2")
        return self


class CandidateCollection(BaseModel):
    """Deterministic candidate set for one site-selection scope. Not a ranking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: SiteSelectionScope
    strategy: GenerationStrategy
    crs: str = Field(min_length=1)
    candidates: tuple[CandidateFeature, ...]
    provenance: ProvenanceRecord
    covering_count: int = 0
    dropped_count: int = 0
    warnings: tuple[str, ...] = ()

    def ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.candidates)

    def __len__(self) -> int:
        return len(self.candidates)

    def to_table(self) -> CandidateTable:
        """Tabular view. Geometry is WKT; missing grid indices stay as NA-compatible None."""

        import pandas as pd

        from solardss.candidates.geometry import geometry_centroid, geometry_wkt

        rows: list[dict[str, object]] = []
        for item in self.candidates:
            centroid_x, centroid_y = geometry_centroid(item.geometry)
            rows.append(
                {
                    DEFAULT_ID_COLUMN: item.candidate_id,
                    "source": item.source.value,
                    "source_id": item.source_id,
                    "area_m2": item.area.value,
                    "crs": item.geometry.crs,
                    "geometry": geometry_wkt(item.geometry),
                    "centroid_x": centroid_x,
                    "centroid_y": centroid_y,
                    "grid_row": item.grid_row,
                    "grid_col": item.grid_col,
                }
            )
        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(
                columns=[
                    DEFAULT_ID_COLUMN,
                    "source",
                    "source_id",
                    "area_m2",
                    "crs",
                    "geometry",
                    "centroid_x",
                    "centroid_y",
                    "grid_row",
                    "grid_col",
                ]
            )
        return validate_candidate_table(frame)
