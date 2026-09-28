"""Annual PV energy-yield request, result, and engine protocol.

PVGIS PVcalc and optional pvlib implement the same ``PVYieldEngine`` protocol.
Numeric fields are tagged by origin. SolarDSS does not invent missing weather,
module coefficients, or PVGIS quantities, and does not compute generation with
an LLM.
"""

from __future__ import annotations

import math
from datetime import datetime
from enum import Enum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from solardss.core.status import EvaluationStatus
from solardss.data.geo import GeographicRef
from solardss.provenance.record import ProvenanceRecord
from solardss.pv.contracts import ModuleTechnology, PVInverter, PVModule
from solardss.pv.design.contracts import MountingType, TrackingMode
from solardss.pv.errors import PVYieldConfigError


class YieldValueOrigin(str, Enum):
    """Who produced a numeric yield field.

    ``pvgis_provider`` is a value PVGIS published. ``pvlib_model`` is computed
    by a named pvlib physical model. ``solardss_derived`` is an identity
    SolarDSS applied to those values (never an independent generation model).
    ``caller_supplied`` is an input assumption, not a calculated yield.
    """

    PVGIS_PROVIDER = "pvgis_provider"
    PVLIB_MODEL = "pvlib_model"
    SOLARDSS_DERIVED = "solardss_derived"
    CALLER_SUPPLIED = "caller_supplied"


class SolarPositionModel(str, Enum):
    """pvlib solar-position method. Numba/ephem backends are not implied."""

    NREL_NUMPY = "nrel_numpy"


class TranspositionModel(str, Enum):
    """POA transposition model passed to ``pvlib.irradiance.get_total_irradiance``."""

    ISOTROPIC = "isotropic"
    HAYDAVIES = "haydavies"
    PEREZ = "perez"
    KING = "king"
    KLUCHER = "klucher"


class TemperatureModel(str, Enum):
    """Module-temperature model. Coefficients are never inferred from mounting."""

    SAPM = "sapm"
    FAIMAN = "faiman"


class DcModel(str, Enum):
    """Array DC model. SAPM/CEC coefficient sets are not loaded from a database."""

    PVWATTS = "pvwatts"


class AcModel(str, Enum):
    """Inverter AC model. Sandia inverter databases are not loaded implicitly."""

    PVWATTS = "pvwatts"


class AoiLossModel(str, Enum):
    """Incidence-angle modifier. ``none`` uses POA global as effective irradiance."""

    NONE = "none"
    PHYSICAL = "physical"
    ASHRAE = "ashrae"


class TaggedYieldValue(BaseModel):
    """One scalar with unit and origin. Absence means the value was not produced."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    value: float
    unit: str = Field(min_length=1)
    origin: YieldValueOrigin
    source_field: str | None = None
    calculation_method: str | None = None
    notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _finite(self) -> TaggedYieldValue:
        if not math.isfinite(self.value):
            raise PVYieldConfigError("yield values must be finite")
        if self.origin is YieldValueOrigin.SOLARDSS_DERIVED and not self.calculation_method:
            raise PVYieldConfigError("SolarDSS-derived values must name the identity used")
        return self


class MonthlyYieldValue(BaseModel):
    """One calendar month of energy or irradiation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    month: int
    value: TaggedYieldValue

    @field_validator("month")
    @classmethod
    def _month(cls, value: int) -> int:
        if value < 1 or value > 12:
            raise PVYieldConfigError("month must be in 1..12")
        return value


class HourlyPowerPoint(BaseModel):
    """One weather-aligned AC (and optional DC/POA) timestep."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    ac_power_w: TaggedYieldValue
    dc_power_w: TaggedYieldValue | None = None
    poa_global_w_m2: TaggedYieldValue | None = None

    @field_validator("timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise PVYieldConfigError("hourly timestamps must be timezone-aware")
        return value


class WeatherObservation(BaseModel):
    """One weather timestep. Missing DNI/DHI/temperature/wind stay missing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: datetime
    ghi_w_m2: float
    dni_w_m2: float | None = None
    dhi_w_m2: float | None = None
    temp_air_c: float | None = None
    wind_speed_m_s: float | None = None

    @field_validator("timestamp")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise PVYieldConfigError("weather timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _domain(self) -> WeatherObservation:
        for name, value in (
            ("ghi_w_m2", self.ghi_w_m2),
            ("dni_w_m2", self.dni_w_m2),
            ("dhi_w_m2", self.dhi_w_m2),
            ("wind_speed_m_s", self.wind_speed_m_s),
        ):
            if value is None:
                continue
            _require_finite(value, name)
            if value < 0.0:
                raise PVYieldConfigError(f"{name} must be non-negative")
        if self.temp_air_c is not None:
            _require_finite(self.temp_air_c, "temp_air_c")
        return self


class PvlibModelSelection(BaseModel):
    """Explicit pvlib model chain. Missing coefficients do not trigger a fallback."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    solar_position: SolarPositionModel = SolarPositionModel.NREL_NUMPY
    transposition: TranspositionModel = TranspositionModel.HAYDAVIES
    temperature: TemperatureModel = TemperatureModel.SAPM
    dc: DcModel = DcModel.PVWATTS
    ac: AcModel = AcModel.PVWATTS
    aoi: AoiLossModel = AoiLossModel.NONE


class PVLossAssumptions(BaseModel):
    """Caller system-loss input plus provider or model loss terms when present.

    PVGIS ``l_*`` fields are provider values. pvlib inverter/system derates are
    recorded separately. 14% is never invented.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    requested_system_loss_percent: TaggedYieldValue | None = None
    pvgis_aoi_loss_percent: TaggedYieldValue | None = None
    pvgis_spectral_loss_percent: TaggedYieldValue | None = None
    pvgis_temperature_irradiance_loss_percent: TaggedYieldValue | None = None
    pvgis_total_loss_percent: TaggedYieldValue | None = None
    inverter_loss_percent: TaggedYieldValue | None = None
    applied_system_derate_percent: TaggedYieldValue | None = None
    notes: tuple[str, ...] = ()


class PVYieldEngineMetadata(BaseModel):
    """Engine/tool metadata for one yield evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider_id: str
    source_name: str
    tool: str | None = None
    api_version: str | None = None
    source_url: str | None = None
    from_cache: bool = False
    status_code: int | None = None
    radiation_db: str | None = None
    meteo_db: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    elevation_m: float | None = None
    output_mounting_key: str | None = None
    pvgis_pvtechchoice: str | None = None
    pvgis_mountingplace: str | None = None
    pvgis_aspect: float | None = None
    pvgis_angle: float | None = None
    pvgis_trackingtype: int | None = None
    retrieved_at: datetime | None = None
    models_used: dict[str, str] = Field(default_factory=dict)
    assumptions: tuple[str, ...] = ()
    defaults_applied: tuple[str, ...] = ()
    weather_source: str | None = None
    module_data_source: str | None = None
    inverter_data_source: str | None = None
    pvlib_version: str | None = None


PVGISYieldMetadata = PVYieldEngineMetadata


class PVYieldRequest(BaseModel):
    """Inputs for a PV yield evaluation (PVGIS PVcalc or pvlib).

    Shared geometry uses SolarDSS azimuth (clockwise from north). PVGIS converts
    that to ``aspect`` (0 = south); pvlib uses the same convention as SolarDSS.
    ``system_loss_percent`` is never defaulted to 14%. Weather, module, and
    inverter fields are required only by pvlib.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    location: GeographicRef
    installed_capacity_kw: float | None = None
    tilt_deg: float | None = None
    azimuth_deg: float | None = None
    mounting: MountingType
    tracking: TrackingMode = TrackingMode.FIXED
    technology: str | None = None
    system_loss_percent: float | None = None
    raddatabase: str | None = None
    usehorizon: bool | None = None
    pvgis_trackingtype: int | None = None
    weather: tuple[WeatherObservation, ...] = ()
    timezone: str | None = None
    elevation_m: float | None = None
    albedo: float | None = None
    module: PVModule | None = None
    inverter: PVInverter | None = None
    module_count: int | None = None
    interval_hours: float | None = None
    models: PvlibModelSelection | None = None
    sapm_temperature_model: str | None = None
    temperature_model_parameters: dict[str, float] | None = None
    gamma_pdc: float | None = None
    inverter_efficiency: float | None = None
    axis_tilt_deg: float | None = None
    axis_azimuth_deg: float | None = None
    max_angle_deg: float | None = None
    backtrack: bool | None = None
    ground_coverage_ratio: float | None = None
    aoi_model_parameters: dict[str, float] | None = None
    weather_source: str | None = None
    module_data_source: str | None = None
    notes: tuple[str, ...] = ()

    @field_validator("mounting", mode="before")
    @classmethod
    def _mounting(cls, value: object) -> object:
        if isinstance(value, str):
            return MountingType(value)
        return value

    @field_validator("tracking", mode="before")
    @classmethod
    def _tracking(cls, value: object) -> object:
        if isinstance(value, str):
            return TrackingMode(value)
        return value

    @field_validator("technology", mode="before")
    @classmethod
    def _technology(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, ModuleTechnology):
            return value.value
        if isinstance(value, str):
            stripped = value.strip()
            return stripped if stripped else None
        return value

    @model_validator(mode="after")
    def _domain(self) -> PVYieldRequest:
        if self.installed_capacity_kw is not None:
            _require_finite(self.installed_capacity_kw, "installed_capacity_kw")
            if self.installed_capacity_kw <= 0.0:
                raise PVYieldConfigError("installed_capacity_kw must be positive")
        if self.tilt_deg is not None:
            _require_finite(self.tilt_deg, "tilt_deg")
            if self.tilt_deg < 0.0 or self.tilt_deg > 90.0:
                raise PVYieldConfigError("tilt_deg must be in [0, 90]")
        if self.azimuth_deg is not None:
            _require_finite(self.azimuth_deg, "azimuth_deg")
            if self.azimuth_deg < 0.0 or self.azimuth_deg > 360.0:
                raise PVYieldConfigError("azimuth_deg must be in [0, 360]")
        if self.system_loss_percent is not None:
            _require_finite(self.system_loss_percent, "system_loss_percent")
            if self.system_loss_percent < 0.0 or self.system_loss_percent > 100.0:
                raise PVYieldConfigError("system_loss_percent must be in [0, 100]")
        if self.pvgis_trackingtype is not None and self.pvgis_trackingtype not in range(0, 6):
            raise PVYieldConfigError("pvgis_trackingtype must be an integer in 0..5")
        if self.raddatabase is not None and not self.raddatabase.strip():
            raise PVYieldConfigError("raddatabase must be non-empty when set")
        if self.elevation_m is not None:
            _require_finite(self.elevation_m, "elevation_m")
        if self.albedo is not None:
            _require_finite(self.albedo, "albedo")
            if self.albedo < 0.0 or self.albedo > 1.0:
                raise PVYieldConfigError("albedo must be in [0, 1]")
        if self.module_count is not None and self.module_count < 1:
            raise PVYieldConfigError("module_count must be >= 1")
        if self.interval_hours is not None:
            _require_finite(self.interval_hours, "interval_hours")
            if self.interval_hours <= 0.0:
                raise PVYieldConfigError("interval_hours must be positive")
        if self.gamma_pdc is not None:
            _require_finite(self.gamma_pdc, "gamma_pdc")
        if self.inverter_efficiency is not None:
            _require_finite(self.inverter_efficiency, "inverter_efficiency")
            if not 0.0 < self.inverter_efficiency <= 1.0:
                raise PVYieldConfigError("inverter_efficiency must be in (0, 1]")
        if self.axis_tilt_deg is not None:
            _require_finite(self.axis_tilt_deg, "axis_tilt_deg")
            if self.axis_tilt_deg < 0.0 or self.axis_tilt_deg > 90.0:
                raise PVYieldConfigError("axis_tilt_deg must be in [0, 90]")
        if self.axis_azimuth_deg is not None:
            _require_finite(self.axis_azimuth_deg, "axis_azimuth_deg")
            if self.axis_azimuth_deg < 0.0 or self.axis_azimuth_deg > 360.0:
                raise PVYieldConfigError("axis_azimuth_deg must be in [0, 360]")
        if self.max_angle_deg is not None:
            _require_finite(self.max_angle_deg, "max_angle_deg")
            if self.max_angle_deg <= 0.0 or self.max_angle_deg > 90.0:
                raise PVYieldConfigError("max_angle_deg must be in (0, 90]")
        if self.ground_coverage_ratio is not None:
            _require_finite(self.ground_coverage_ratio, "ground_coverage_ratio")
            if self.ground_coverage_ratio <= 0.0 or self.ground_coverage_ratio > 1.0:
                raise PVYieldConfigError("ground_coverage_ratio must be in (0, 1]")
        if self.timezone is not None and not self.timezone.strip():
            raise PVYieldConfigError("timezone must be non-empty when set")
        if self.sapm_temperature_model is not None and not self.sapm_temperature_model.strip():
            raise PVYieldConfigError("sapm_temperature_model must be non-empty when set")
        if self.temperature_model_parameters:
            for key, value in self.temperature_model_parameters.items():
                _require_finite(value, f"temperature_model_parameters.{key}")
        if self.aoi_model_parameters:
            for key, value in self.aoi_model_parameters.items():
                _require_finite(value, f"aoi_model_parameters.{key}")
        return self


class PVYieldResult(BaseModel):
    """Yield from PVGIS PVcalc or pvlib. Distinct from adapter ``PVGISYieldResult``.

    Provider/model fields stay missing when the source did not produce them.
    Specific yield is always SolarDSS-derived as annual energy / peak power.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: EvaluationStatus
    request: PVYieldRequest
    annual_energy_kwh: TaggedYieldValue | None = None
    specific_yield_kwh_per_kwp: TaggedYieldValue | None = None
    monthly_energy_kwh: tuple[MonthlyYieldValue, ...] = ()
    monthly_irradiation: tuple[MonthlyYieldValue, ...] = ()
    annual_irradiation: TaggedYieldValue | None = None
    hourly_ac_power: tuple[HourlyPowerPoint, ...] = ()
    capacity_factor: TaggedYieldValue | None = None
    loss_assumptions: PVLossAssumptions = Field(default_factory=PVLossAssumptions)
    provider_metadata: PVYieldEngineMetadata
    provenance: ProvenanceRecord
    warnings: tuple[str, ...] = ()


@runtime_checkable
class PVYieldEngine(Protocol):
    """PVGIS and pvlib implement this evaluation surface."""

    def evaluate(self, request: PVYieldRequest) -> PVYieldResult:
        """Evaluate yield. Missing physics or provider fields stay missing."""
        ...


def _require_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise PVYieldConfigError(f"{name} must be finite")
