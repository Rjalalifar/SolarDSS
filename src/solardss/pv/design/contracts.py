"""PV system configuration, design problem, and evaluation records.

Location is not chosen here. After a site is resolved, remaining variables are
module, count, capacity, tilt, azimuth, mounting, tracking, inverter, DC/AC
ratio, row spacing, GCR, and optional battery size.
"""

from __future__ import annotations

import math
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from solardss.core.status import FeasibilityStatus
from solardss.data.geo import GeographicRef
from solardss.decision.contracts import ModulePreferences, UserConstraint
from solardss.indicators.contracts import DecisionMode, ProjectType
from solardss.indicators.defaults import GROUND_PROJECTS, ROOF_PROJECTS
from solardss.provenance.record import ProvenanceRecord
from solardss.pv.design.errors import PVDesignConfigError

_ATOL = 1e-9
_MAX_CANDIDATES = 10_000


class MountingType(str, Enum):
    """Array mounting class. Implied by project type during enumeration."""

    ROOFTOP = "rooftop"
    GROUND = "ground"


class TrackingMode(str, Enum):
    """Tracking class. Rooftop tracking is not implied."""

    FIXED = "fixed"
    SINGLE_AXIS = "single_axis"
    DUAL_AXIS = "dual_axis"


class DesignConstraintKind(str, Enum):
    """Which design rule fired. None of these invent a missing datasheet value."""

    AVAILABLE_AREA = "available_area"
    ROOF_AREA = "roof_area"
    REQUESTED_CAPACITY = "requested_capacity"
    ELECTRICAL_LIMIT = "electrical_limit"
    USER_PREFERENCE = "user_preference"
    MODULE_DIMENSION = "module_dimension"
    INVERTER_LIMIT = "inverter_limit"
    ROOF_TILT = "roof_tilt"
    ROOF_ORIENTATION = "roof_orientation"
    MOUNTING = "mounting"
    LOCATION = "location"
    BATTERY = "battery"


class ConfigurationSearchSpace(BaseModel):
    """Discrete values for bounded enumeration. Empty tuples mean 'derive or skip'.

    This is not a continuous optimizer and not an LLM prompt.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    module_ids: tuple[str, ...] = ()
    inverter_ids: tuple[str, ...] = ()
    battery_ids: tuple[str, ...] = ()
    module_counts: tuple[int, ...] = ()
    inverter_counts: tuple[int, ...] = (1,)
    tilt_deg: tuple[float, ...] = ()
    azimuth_deg: tuple[float, ...] = ()
    tracking_modes: tuple[TrackingMode, ...] = ()
    row_spacing_m: tuple[float, ...] = ()
    ground_coverage_ratio: tuple[float, ...] = ()
    battery_capacity_kwh: tuple[float, ...] = ()
    max_candidates: int = 256

    @field_validator("tracking_modes", mode="before")
    @classmethod
    def _tracking_tuple(cls, value: object) -> object:
        if isinstance(value, str | TrackingMode):
            return (value,)
        if isinstance(value, list | tuple):
            return tuple(
                TrackingMode(item) if not isinstance(item, TrackingMode) else item for item in value
            )
        return value

    @model_validator(mode="after")
    def _bounds(self) -> ConfigurationSearchSpace:
        if self.max_candidates < 1 or self.max_candidates > _MAX_CANDIDATES:
            raise PVDesignConfigError(f"max_candidates must be in [1, {_MAX_CANDIDATES}]")
        for count in self.module_counts:
            if count < 1:
                raise PVDesignConfigError("module_counts must be positive integers")
        for count in self.inverter_counts:
            if count < 1:
                raise PVDesignConfigError("inverter_counts must be positive integers")
        for tilt in self.tilt_deg:
            _require_finite(tilt, "tilt_deg")
            if tilt < 0.0 or tilt > 90.0:
                raise PVDesignConfigError("tilt_deg must be in [0, 90]")
        for azimuth in self.azimuth_deg:
            _require_finite(azimuth, "azimuth_deg")
            if azimuth < 0.0 or azimuth > 360.0:
                raise PVDesignConfigError("azimuth_deg must be in [0, 360]")
        for spacing in self.row_spacing_m:
            _require_positive(spacing, "row_spacing_m")
        for gcr in self.ground_coverage_ratio:
            _require_finite(gcr, "ground_coverage_ratio")
            if gcr <= 0.0 or gcr > 1.0:
                raise PVDesignConfigError("ground_coverage_ratio must be in (0, 1]")
        for capacity in self.battery_capacity_kwh:
            _require_finite(capacity, "battery_capacity_kwh")
            if capacity < 0.0:
                raise PVDesignConfigError("battery_capacity_kwh must be non-negative")
        for ident in (
            *self.module_ids,
            *self.inverter_ids,
            *self.battery_ids,
        ):
            if not str(ident).strip():
                raise PVDesignConfigError("search-space component ids must be non-empty")
        return self


class PVSystemConfiguration(BaseModel):
    """One fully specified candidate system after the site is known."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    configuration_id: str = ""
    location: GeographicRef
    module_id: str = Field(min_length=1)
    module_count: int
    installed_capacity_w: float | None = None
    tilt_deg: float
    azimuth_deg: float
    mounting_type: MountingType
    tracking_mode: TrackingMode
    inverter_id: str = Field(min_length=1)
    inverter_count: int = 1
    dc_ac_ratio: float | None = None
    row_spacing_m: float | None = None
    ground_coverage_ratio: float | None = None
    battery_id: str | None = None
    battery_capacity_kwh: float | None = None
    notes: tuple[str, ...] = ()

    @field_validator("tracking_mode", mode="before")
    @classmethod
    def _tracking(cls, value: object) -> object:
        if isinstance(value, str):
            return TrackingMode(value)
        return value

    @field_validator("mounting_type", mode="before")
    @classmethod
    def _mounting(cls, value: object) -> object:
        if isinstance(value, str):
            return MountingType(value)
        return value

    @model_validator(mode="after")
    def _numeric(self) -> PVSystemConfiguration:
        if self.module_count < 1:
            raise PVDesignConfigError("module_count must be >= 1")
        if self.inverter_count < 1:
            raise PVDesignConfigError("inverter_count must be >= 1")
        _require_finite(self.tilt_deg, "tilt_deg")
        if self.tilt_deg < 0.0 or self.tilt_deg > 90.0:
            raise PVDesignConfigError("tilt_deg must be in [0, 90]")
        _require_finite(self.azimuth_deg, "azimuth_deg")
        if self.azimuth_deg < 0.0 or self.azimuth_deg > 360.0:
            raise PVDesignConfigError("azimuth_deg must be in [0, 360]")
        if self.installed_capacity_w is not None:
            _require_positive(self.installed_capacity_w, "installed_capacity_w")
        if self.dc_ac_ratio is not None:
            _require_positive(self.dc_ac_ratio, "dc_ac_ratio")
        if self.row_spacing_m is not None:
            _require_positive(self.row_spacing_m, "row_spacing_m")
        if self.ground_coverage_ratio is not None:
            _require_finite(self.ground_coverage_ratio, "ground_coverage_ratio")
            if self.ground_coverage_ratio <= 0.0 or self.ground_coverage_ratio > 1.0:
                raise PVDesignConfigError("ground_coverage_ratio must be in (0, 1]")
        if self.battery_capacity_kwh is not None:
            _require_finite(self.battery_capacity_kwh, "battery_capacity_kwh")
            if self.battery_capacity_kwh < 0.0:
                raise PVDesignConfigError("battery_capacity_kwh must be non-negative")
        if self.battery_id is not None and not self.battery_id.strip():
            raise PVDesignConfigError("battery_id must be non-empty when set")
        if self.mounting_type is MountingType.ROOFTOP and (
            self.row_spacing_m is not None or self.ground_coverage_ratio is not None
        ):
            raise PVDesignConfigError(
                "row_spacing_m and ground_coverage_ratio are ground-mount variables"
            )
        ident = self.configuration_id.strip() if self.configuration_id else ""
        if not ident:
            object.__setattr__(self, "configuration_id", _configuration_id(self))
        return self


class PVDesignProblem(BaseModel):
    """PV design after the site is resolved. Location is not re-optimized here.

    In ``SITE_SELECTION``, pass the chosen candidate as ``location``. In
    ``FIXED_SITE``, ``location`` is the given site. Tilt and azimuth on a roof
    are constraints when ``roof_tilt_deg`` / ``roof_azimuth_deg`` are set.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    location: GeographicRef
    project_type: ProjectType
    decision_mode: DecisionMode = DecisionMode.FIXED_SITE
    available_area_m2: float | None = None
    roof_area_m2: float | None = None
    roof_tilt_deg: float | None = None
    roof_azimuth_deg: float | None = None
    target_capacity_w: float | None = None
    min_capacity_w: float | None = None
    max_capacity_w: float | None = None
    min_dc_ac_ratio: float | None = None
    max_dc_ac_ratio: float | None = None
    max_module_length_m: float | None = None
    max_module_width_m: float | None = None
    max_module_thickness_m: float | None = None
    max_module_weight_kg: float | None = None
    module_preferences: ModulePreferences | None = None
    user_constraints: tuple[UserConstraint, ...] = ()
    battery_required: bool = False
    include_fixtures: bool = False
    search_space: ConfigurationSearchSpace = Field(default_factory=ConfigurationSearchSpace)
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _consistent(self) -> PVDesignProblem:
        roof = is_rooftop_project(self.project_type)
        if roof:
            if self.available_area_m2 is not None:
                raise PVDesignConfigError("rooftop design uses roof_area_m2, not available_area_m2")
        elif self.roof_area_m2 is not None or self.roof_tilt_deg is not None:
            raise PVDesignConfigError(
                "ground-mount design does not take roof_area_m2 or roof_tilt_deg"
            )
        elif self.roof_azimuth_deg is not None:
            raise PVDesignConfigError("ground-mount design does not take roof_azimuth_deg")
        for name, value in (
            ("available_area_m2", self.available_area_m2),
            ("roof_area_m2", self.roof_area_m2),
            ("max_module_length_m", self.max_module_length_m),
            ("max_module_width_m", self.max_module_width_m),
            ("max_module_thickness_m", self.max_module_thickness_m),
            ("max_module_weight_kg", self.max_module_weight_kg),
        ):
            if value is not None:
                _require_positive(value, name)
        for name, value in (
            ("target_capacity_w", self.target_capacity_w),
            ("min_capacity_w", self.min_capacity_w),
            ("max_capacity_w", self.max_capacity_w),
            ("min_dc_ac_ratio", self.min_dc_ac_ratio),
            ("max_dc_ac_ratio", self.max_dc_ac_ratio),
        ):
            if value is not None:
                _require_positive(value, name)
        if self.roof_tilt_deg is not None:
            _require_finite(self.roof_tilt_deg, "roof_tilt_deg")
            if self.roof_tilt_deg < 0.0 or self.roof_tilt_deg > 90.0:
                raise PVDesignConfigError("roof_tilt_deg must be in [0, 90]")
        if self.roof_azimuth_deg is not None:
            _require_finite(self.roof_azimuth_deg, "roof_azimuth_deg")
            if self.roof_azimuth_deg < 0.0 or self.roof_azimuth_deg > 360.0:
                raise PVDesignConfigError("roof_azimuth_deg must be in [0, 360]")
        _ordered(self.min_capacity_w, self.max_capacity_w, "capacity_w")
        _ordered(self.min_dc_ac_ratio, self.max_dc_ac_ratio, "dc_ac_ratio")
        if (
            self.min_capacity_w is None
            and self.target_capacity_w is not None
            and self.max_capacity_w is not None
            and self.target_capacity_w - self.max_capacity_w > _ATOL
        ):
            raise PVDesignConfigError("target_capacity_w must be <= max_capacity_w")
        return self

    @property
    def capacity_floor_w(self) -> float | None:
        """Minimum accepted DC nameplate, watts. ``target_capacity_w`` is a floor."""

        if self.min_capacity_w is not None:
            return self.min_capacity_w
        return self.target_capacity_w


class DerivedDesignQuantities(BaseModel):
    """Values computed from the configuration and catalog. Missing stays missing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    installed_capacity_w: float
    dc_ac_ratio: float | None = None
    module_area_m2: float | None = None
    array_area_m2: float | None = None
    occupied_area_m2: float | None = None
    ground_coverage_ratio: float | None = None
    row_spacing_m: float | None = None
    battery_capacity_kwh: float | None = None


class DesignConstraintFailure(BaseModel):
    """One design rule that failed or could not be evaluated."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    constraint_id: str = Field(min_length=1)
    kind: DesignConstraintKind
    message: str = Field(min_length=1)
    actual: float | str | bool | None = None
    limit: float | str | bool | None = None
    unknown: bool = False


class PVDesignResult(BaseModel):
    """Feasibility of one configuration against an explicit design problem."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    configuration: PVSystemConfiguration
    status: FeasibilityStatus
    feasible: bool
    derived: DerivedDesignQuantities
    failed_constraints: tuple[DesignConstraintFailure, ...] = ()
    unknown_constraints: tuple[DesignConstraintFailure, ...] = ()
    warnings: tuple[str, ...] = ()
    provenance: ProvenanceRecord


class PVDesignSet(BaseModel):
    """Evaluated enumeration. Truncation is reported; a 'best' design is not chosen."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    results: tuple[PVDesignResult, ...] = ()
    generator_id: str = Field(min_length=1)
    truncated: bool = False
    provenance: ProvenanceRecord
    warnings: tuple[str, ...] = ()

    def feasible_results(self) -> tuple[PVDesignResult, ...]:
        """Configurations that passed every stated design constraint."""

        return tuple(item for item in self.results if item.feasible)


def is_rooftop_project(project_type: ProjectType) -> bool:
    """True for rooftop project classes."""

    return project_type in ROOF_PROJECTS


def is_ground_project(project_type: ProjectType) -> bool:
    """True for utility-scale and ground-mounted classes."""

    return project_type in GROUND_PROJECTS


def mounting_for(project_type: ProjectType) -> MountingType:
    """Map project type to mounting. Not a performance assumption."""

    if is_rooftop_project(project_type):
        return MountingType.ROOFTOP
    if is_ground_project(project_type):
        return MountingType.GROUND
    raise PVDesignConfigError(f"unsupported project_type {project_type.value!r}")


def _configuration_id(config: PVSystemConfiguration) -> str:
    from solardss.core.hashing import fingerprint_canonical

    payload = config.model_dump(mode="json")
    payload.pop("configuration_id", None)
    digest = fingerprint_canonical(payload)
    return f"pv.{digest[:12]}"


def _require_finite(value: float, name: str) -> None:
    if value != value or value in (float("inf"), float("-inf")):
        raise PVDesignConfigError(f"{name} must be finite")


def _require_positive(value: float, name: str) -> None:
    _require_finite(value, name)
    if value <= 0.0:
        raise PVDesignConfigError(f"{name} must be positive")


def _ordered(lo: float | None, hi: float | None, name: str) -> None:
    if lo is not None and hi is not None and hi + _ATOL < lo:
        raise PVDesignConfigError(f"max_{name} must be >= min_{name}")


def almost_equal(left: float, right: float, *, atol: float = _ATOL) -> bool:
    """Absolute-and-relative comparison for datasheet identities."""

    scale = max(abs(left), abs(right), 1.0)
    return abs(left - right) <= max(atol, atol * scale)


def azimuth_delta(left: float, right: float) -> float:
    """Smallest absolute difference on a 360-degree circle."""

    raw = abs(left - right) % 360.0
    return min(raw, 360.0 - raw)


def ceil_count(total: float, unit: float) -> int:
    """Smallest positive integer ``n`` with ``n * unit >= total``."""

    if unit <= 0.0:
        raise PVDesignConfigError("rated power used to size module_count must be positive")
    return max(1, math.ceil(total / unit - _ATOL))
