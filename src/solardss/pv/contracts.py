"""Typed PV module, inverter, and battery records.

SolarDSS does not ship manufacturer datasheets. Missing electrical fields stay
missing; they are not invented from other columns.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from solardss.data.contracts import DatasetRef
from solardss.provenance.record import ProvenanceRecord
from solardss.pv.errors import PVCatalogConfigError

_COEFF_ATOL = 1e-12


class ModuleTechnology(str, Enum):
    """Technology *class*. Not a product, and not an exhaustive market list."""

    MONO_CSI = "mono-c-Si"
    POLY_CSI = "poly-c-Si"
    CDTE = "CdTe"
    CIGS = "CIGS"
    HJT = "HJT"
    PERC = "PERC"
    TOPCON = "TOPCon"
    OTHER = "other"


class ComponentSourceKind(str, Enum):
    """Where a catalog row came from. ``test_fixture`` is never a real product."""

    DATASHEET = "datasheet"
    CATALOG_FILE = "catalog_file"
    TEST_FIXTURE = "test_fixture"
    ADAPTER = "adapter"


class ComponentProvenance(BaseModel):
    """Audit trail for one catalog row. Distinct from a computed ``ProvenanceRecord``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_name: str = Field(min_length=1)
    source_kind: ComponentSourceKind
    fixture: bool = False
    dataset: DatasetRef | None = None
    notes: tuple[str, ...] = ()
    record: ProvenanceRecord | None = None

    @model_validator(mode="after")
    def _fixture_kind(self) -> ComponentProvenance:
        if self.fixture and self.source_kind is not ComponentSourceKind.TEST_FIXTURE:
            raise PVCatalogConfigError("fixture components must use source_kind='test_fixture'")
        if self.source_kind is ComponentSourceKind.TEST_FIXTURE and not self.fixture:
            raise PVCatalogConfigError("test_fixture source_kind requires fixture=True")
        if self.fixture and not any("fixture" in note.casefold() for note in self.notes):
            raise PVCatalogConfigError(
                "fixture provenance must state in notes that the row is a test fixture, "
                "not a commercial product"
            )
        return self


class _ComponentBase(BaseModel):
    """Shared identity and provenance for every catalogued device."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    manufacturer: str = Field(min_length=1)
    model: str = Field(min_length=1)
    metadata: dict[str, str] = Field(default_factory=dict)
    provenance: ComponentProvenance

    @model_validator(mode="after")
    def _fixture_label(self) -> _ComponentBase:
        if self.provenance.fixture:
            token = f"{self.manufacturer} {self.model}".casefold()
            if "test" not in token and "fixture" not in token:
                raise PVCatalogConfigError(
                    "fixture manufacturer/model must be labelled TEST or FIXTURE; "
                    "SolarDSS does not invent commercial products"
                )
        return self


class PVModule(_ComponentBase):
    """One PV module datasheet row. Optional electrical fields stay unset if unknown."""

    module_id: str = Field(min_length=1)
    technology: str = Field(min_length=1)
    rated_power_w: float
    efficiency: float | None = None
    area_m2: float | None = None
    length_m: float | None = None
    width_m: float | None = None
    thickness_m: float | None = None
    temperature_coefficient_pmax: float | None = None
    temperature_coefficient_pmax_unit: str = "%/C"
    voc: float | None = None
    isc: float | None = None
    vmp: float | None = None
    imp: float | None = None
    bifacial: bool = False
    bifaciality_factor: float | None = None
    degradation_rate: float | None = None
    max_system_voltage: float | None = None
    weight_kg: float | None = None

    @model_validator(mode="after")
    def _numeric(self) -> PVModule:
        _require_positive(self.rated_power_w, "rated_power_w")
        _optional_unit_interval(self.efficiency, "efficiency")
        _optional_positive(self.area_m2, "area_m2")
        _optional_positive(self.length_m, "length_m")
        _optional_positive(self.width_m, "width_m")
        _optional_positive(self.thickness_m, "thickness_m")
        _optional_positive(self.weight_kg, "weight_kg")
        _optional_positive(self.max_system_voltage, "max_system_voltage")
        _optional_positive(self.voc, "voc")
        _optional_positive(self.isc, "isc")
        _optional_positive(self.vmp, "vmp")
        _optional_positive(self.imp, "imp")
        if self.degradation_rate is not None:
            _require_finite(self.degradation_rate, "degradation_rate")
            if self.degradation_rate < 0.0 or self.degradation_rate >= 1.0:
                raise PVCatalogConfigError("degradation_rate must be in [0, 1) per year")
        if self.voc is not None and self.vmp is not None and self.vmp - self.voc > _COEFF_ATOL:
            raise PVCatalogConfigError("vmp must not exceed voc")
        if self.isc is not None and self.imp is not None and self.imp - self.isc > _COEFF_ATOL:
            raise PVCatalogConfigError("imp must not exceed isc")
        if self.bifacial:
            if self.bifaciality_factor is None:
                raise PVCatalogConfigError(
                    "bifacial modules require bifaciality_factor; it is not defaulted"
                )
            _optional_unit_interval(self.bifaciality_factor, "bifaciality_factor")
        elif self.bifaciality_factor is not None and self.bifaciality_factor > _COEFF_ATOL:
            raise PVCatalogConfigError("bifaciality_factor is only valid when bifacial is true")
        if self.temperature_coefficient_pmax is not None:
            _require_finite(self.temperature_coefficient_pmax, "temperature_coefficient_pmax")
        if not self.temperature_coefficient_pmax_unit.strip():
            raise PVCatalogConfigError("temperature_coefficient_pmax_unit must be non-empty")
        return self


class PVInverter(_ComponentBase):
    """One inverter datasheet row. Limits are not inferred from rated AC power."""

    inverter_id: str = Field(min_length=1)
    rated_ac_power_w: float
    max_dc_input_w: float | None = None
    mppt_voltage_min: float | None = None
    mppt_voltage_max: float | None = None
    efficiency: float | None = None
    max_dc_voltage: float | None = None
    max_dc_current: float | None = None
    min_dc_voltage: float | None = None
    max_ac_current: float | None = None
    mppt_count: int | None = None

    @model_validator(mode="after")
    def _numeric(self) -> PVInverter:
        _require_positive(self.rated_ac_power_w, "rated_ac_power_w")
        _optional_positive(self.max_dc_input_w, "max_dc_input_w")
        _optional_positive(self.mppt_voltage_min, "mppt_voltage_min")
        _optional_positive(self.mppt_voltage_max, "mppt_voltage_max")
        _optional_positive(self.max_dc_voltage, "max_dc_voltage")
        _optional_positive(self.max_dc_current, "max_dc_current")
        _optional_positive(self.min_dc_voltage, "min_dc_voltage")
        _optional_positive(self.max_ac_current, "max_ac_current")
        _optional_unit_interval(self.efficiency, "efficiency")
        if self.mppt_count is not None and self.mppt_count < 1:
            raise PVCatalogConfigError("mppt_count must be >= 1")
        if (
            self.mppt_voltage_min is not None
            and self.mppt_voltage_max is not None
            and self.mppt_voltage_max < self.mppt_voltage_min
        ):
            raise PVCatalogConfigError("mppt_voltage_max must be >= mppt_voltage_min")
        if (
            self.min_dc_voltage is not None
            and self.max_dc_voltage is not None
            and self.max_dc_voltage < self.min_dc_voltage
        ):
            raise PVCatalogConfigError("max_dc_voltage must be >= min_dc_voltage")
        return self


class BatterySystem(_ComponentBase):
    """One battery system row. Chemistry and limits are caller-supplied."""

    battery_id: str = Field(min_length=1)
    chemistry: str | None = None
    usable_capacity_kwh: float
    rated_power_w: float | None = None
    roundtrip_efficiency: float | None = None
    voltage_nominal: float | None = None

    @model_validator(mode="after")
    def _numeric(self) -> BatterySystem:
        _require_positive(self.usable_capacity_kwh, "usable_capacity_kwh")
        _optional_positive(self.rated_power_w, "rated_power_w")
        _optional_positive(self.voltage_nominal, "voltage_nominal")
        _optional_unit_interval(self.roundtrip_efficiency, "roundtrip_efficiency")
        return self


class ModuleFilter(BaseModel):
    """Module selection bounds. Missing datasheet fields fail a bound rather than pass."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    technology: tuple[str, ...] | None = None
    min_efficiency: float | None = None
    max_efficiency: float | None = None
    min_rated_power_w: float | None = None
    max_rated_power_w: float | None = None
    min_temperature_coefficient_pmax: float | None = None
    max_temperature_coefficient_pmax: float | None = None
    max_length_m: float | None = None
    max_width_m: float | None = None
    max_thickness_m: float | None = None
    max_area_m2: float | None = None
    max_weight_kg: float | None = None
    bifacial: bool | None = None
    include_fixtures: bool = False

    @field_validator("technology", mode="before")
    @classmethod
    def _tech_tuple(cls, value: object) -> object:
        if isinstance(value, ModuleTechnology):
            return (value.value,)
        if isinstance(value, str):
            return (value,)
        if isinstance(value, list | tuple):
            return tuple(
                item.value if isinstance(item, ModuleTechnology) else item for item in value
            )
        return value

    @model_validator(mode="after")
    def _bounds(self) -> ModuleFilter:
        _ordered(self.min_efficiency, self.max_efficiency, "efficiency")
        _ordered(self.min_rated_power_w, self.max_rated_power_w, "rated_power_w")
        _ordered(
            self.min_temperature_coefficient_pmax,
            self.max_temperature_coefficient_pmax,
            "temperature_coefficient_pmax",
        )
        for name, value in (
            ("max_length_m", self.max_length_m),
            ("max_width_m", self.max_width_m),
            ("max_thickness_m", self.max_thickness_m),
            ("max_area_m2", self.max_area_m2),
            ("max_weight_kg", self.max_weight_kg),
        ):
            _optional_positive(value, name)
        if self.technology is not None and any(not item.strip() for item in self.technology):
            raise PVCatalogConfigError("technology filter tokens must be non-empty")
        return self


class InverterFilter(BaseModel):
    """Inverter selection bounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_rated_ac_power_w: float | None = None
    max_rated_ac_power_w: float | None = None
    min_efficiency: float | None = None
    max_efficiency: float | None = None
    include_fixtures: bool = False

    @model_validator(mode="after")
    def _bounds(self) -> InverterFilter:
        _ordered(self.min_rated_ac_power_w, self.max_rated_ac_power_w, "rated_ac_power_w")
        _ordered(self.min_efficiency, self.max_efficiency, "efficiency")
        return self


def _require_finite(value: float, name: str) -> None:
    if value != value or value in (float("inf"), float("-inf")):
        raise PVCatalogConfigError(f"{name} must be finite")


def _require_positive(value: float, name: str) -> None:
    _require_finite(value, name)
    if value <= 0.0:
        raise PVCatalogConfigError(f"{name} must be positive")


def _optional_positive(value: float | None, name: str) -> None:
    if value is not None:
        _require_positive(value, name)


def _optional_unit_interval(value: float | None, name: str) -> None:
    if value is None:
        return
    _require_finite(value, name)
    if value <= 0.0 or value > 1.0:
        raise PVCatalogConfigError(f"{name} must be in (0, 1]")


def _ordered(lo: float | None, hi: float | None, name: str) -> None:
    if lo is not None:
        _require_finite(lo, f"min_{name}")
    if hi is not None:
        _require_finite(hi, f"max_{name}")
    if lo is not None and hi is not None and hi < lo:
        raise PVCatalogConfigError(f"max_{name} must be >= min_{name}")
