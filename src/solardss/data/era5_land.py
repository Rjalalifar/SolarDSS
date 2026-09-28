"""Optional ERA5-Land weather adapter.

Variable names, units, and accumulation rules follow the Copernicus ERA5-Land
documentation (Muñoz-Sabater et al., 2021; CKB ERA5-Land data documentation).
Unknown fields stay unknown. SolarDSS never fills them with defaults.

CDS access is optional. Credentials come from the environment or the
constructor and are never written into source, cache payloads, or provenance.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from solardss.core.arrays import BoolArray, FloatArray
from solardss.core.jsonutil import JsonScalar, JsonValue
from solardss.data.contracts import DatasetRef
from solardss.data.geo import BoundingBox
from solardss.exceptions import InvalidParameterError, MissingExtraError
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.indicators.ids import IndicatorId
from solardss.provenance.coverage import SpatialCoverage, TemporalCoverage
from solardss.provenance.observation import (
    DataProvenance,
    IndicatorObservation,
    ObservationStatus,
    QualityFlag,
    ValueOrigin,
)
from solardss.providers.cache import DatasetCache
from solardss.providers.clock import Clock, UtcClock
from solardss.providers.contracts import (
    AvailabilityStatus,
    CoverageReport,
    HealthStatus,
    ProviderAvailability,
    ProviderHealth,
)
from solardss.providers.errors import CacheError, ERA5LandResponseError
from solardss.providers.region import request_bbox, request_point
from solardss.providers.request import FetchRequest, request_cache_key

ERA5_LAND_DATASET_ID: Final = "reanalysis-era5-land"
ERA5_LAND_RESOLUTION: Final = "0.1deg"
ERA5_LAND_START_YEAR: Final = 1950
ERA5_LAND_SOURCE_NAME: Final = "ERA5-Land"
DEFAULT_CDS_API_URL: Final = "https://cds.climate.copernicus.eu/api"
ERA5_LAND_CATALOGUE_URL: Final = "https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land"

ERA5_LAND_PATH_ENV: Final = "SOLARDSS_ERA5_LAND_PATH"
ERA5_LAND_DATASET_ENV: Final = "SOLARDSS_ERA5_LAND_DATASET"
ERA5_WINDOW_ENV: Final = "SOLARDSS_ERA5_WINDOW"
ERA5_TYPICAL_START_ENV: Final = "SOLARDSS_ERA5_TYPICAL_START_YEAR"
ERA5_TYPICAL_END_ENV: Final = "SOLARDSS_ERA5_TYPICAL_END_YEAR"
ERA5_RECENT_YEARS_ENV: Final = "SOLARDSS_ERA5_RECENT_YEARS"
ERA5_PRECIP_CONV_ENV: Final = "SOLARDSS_ERA5_PRECIPITATION_CONVENTION"
ERA5_CDS_URL_ENV: Final = "SOLARDSS_CDSAPI_URL"
ERA5_CDS_KEY_ENV: Final = "SOLARDSS_CDSAPI_KEY"
CDSAPI_URL_ENV: Final = "CDSAPI_URL"
CDSAPI_KEY_ENV: Final = "CDSAPI_KEY"

WMO_NORMAL_START_YEAR: Final = 1991
WMO_NORMAL_END_YEAR: Final = 2020
DEFAULT_RECENT_YEARS: Final = 10
DEFAULT_SPATIAL_PAD_DEG: Final = 0.2
KELVIN_OFFSET: Final = 273.15
# Alduchov and Eskridge (1996) August-Roche-Magnus coefficients, T in degC.
MAGNUS_A: Final = 17.625
MAGNUS_B: Final = 243.04

ERA5_EXTRA_MODULES: Final = ("xarray", "netCDF4", "cdsapi")
_CACHE_SCHEMA: Final = "solardss.era5_land_obs.v1"

CALC_K_TO_C: Final = "era5_land_kelvin_to_celsius"
CALC_WIND: Final = "era5_land_wind_speed_from_u10_v10"
CALC_RH: Final = "era5_land_magnus_rh_alduchov_eskridge_1996"
CALC_PRECIP: Final = "era5_land_precipitation_m_to_mm_year"
CALC_HEAT: Final = "era5_land_extreme_heat_days_caller_threshold"
CALC_SNOW_COVER: Final = "era5_land_mean_snow_cover_fraction"
CALC_SNOW_DEPTH: Final = "era5_land_snow_presence_fraction_proxy"
CALC_CLOUD: Final = "era5_land_mean_cloud_cover"

CDS_RETRIEVE_VARIABLES: Final = (
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "snow_cover",
)

_VAR_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "t2m": ("t2m", "2t", "2m_temperature"),
    "d2m": ("d2m", "2d", "2m_dewpoint_temperature"),
    "u10": ("u10", "10m_u_component_of_wind"),
    "v10": ("v10", "10m_v_component_of_wind"),
    "tp": ("tp", "total_precipitation"),
    "snowc": ("snowc", "snow_cover"),
    "sde": ("sde", "snow_depth"),
    "sd": ("sd", "snow_depth_water_equivalent"),
    "tcc": ("tcc", "total_cloud_cover", "cloud_cover"),
}

_SERVED: Final = frozenset(
    {
        IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN.value,
        IndicatorId.CLIMATE_AIR_TEMPERATURE_MIN.value,
        IndicatorId.CLIMATE_AIR_TEMPERATURE_MAX.value,
        IndicatorId.CLIMATE_EXTREME_HEAT_DAYS.value,
        IndicatorId.CLIMATE_WIND_SPEED_MEAN.value,
        IndicatorId.CLIMATE_RELATIVE_HUMIDITY.value,
        IndicatorId.CLIMATE_CLOUD_COVER.value,
        IndicatorId.CLIMATE_PRECIPITATION.value,
        IndicatorId.CLIMATE_SNOW_RISK.value,
    }
)

CdsRetrieveFn = Callable[[str, Mapping[str, object], Path], None]


class PrecipitationConvention(str, Enum):
    """How ``tp`` values are stored. Wrong convention would invent annual totals."""

    TIMESTEP_INCREMENT = "timestep_increment"
    FORECAST_ACCUMULATED = "forecast_accumulated"
    MONTHLY_MEAN_DAILY = "monthly_mean_daily"


class ERA5LandSettings(BaseModel):
    """Constructor/environment configuration. The API key is never repr'd."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = ERA5_LAND_DATASET_ID
    dataset_path: str | None = None
    cds_url: str = DEFAULT_CDS_API_URL
    cds_key: SecretStr | None = None
    typical_start_year: int = WMO_NORMAL_START_YEAR
    typical_end_year: int = WMO_NORMAL_END_YEAR
    recent_years: int = DEFAULT_RECENT_YEARS
    spatial_pad_deg: float = DEFAULT_SPATIAL_PAD_DEG
    precipitation_convention: PrecipitationConvention | None = None
    window: str = "typical"

    def cds_key_value(self) -> str | None:
        if self.cds_key is None:
            return None
        return self.cds_key.get_secret_value()

    def has_cds_key(self) -> bool:
        key = self.cds_key_value()
        return key is not None and bool(key.strip())


class AnalysisWindow(BaseModel):
    """Closed calendar window used to subset ERA5-Land before aggregation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str
    start: date
    end: date
    label: str


class ERA5Derivation(BaseModel):
    """Deterministic conversion applied to official ERA5-Land fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    method_id: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    source_variables: tuple[str, ...]
    derived_from: tuple[str, ...]


class ERA5MonthlyClimate(BaseModel):
    """One calendar month of preserved ERA5-Land statistics. Absent fields stay null."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int
    month: int
    t2m_mean_deg_c: float | None = None
    t2m_min_deg_c: float | None = None
    t2m_max_deg_c: float | None = None
    wind_speed_mean_m_s: float | None = None
    relative_humidity_mean_percent: float | None = None
    precipitation_mm: float | None = None
    snow_cover_mean_fraction: float | None = None
    cloud_cover_mean: float | None = None
    present_fields: tuple[str, ...] = ()


class ERA5LandRawMetadata(BaseModel):
    """Raw provider context, kept separate from normalized indicators."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    source_url: str
    variables_present: tuple[str, ...]
    units: dict[str, str]
    grid_latitude: float | None = None
    grid_longitude: float | None = None
    time_start: date | None = None
    time_end: date | None = None
    from_cache: bool = False
    retrieved_at: datetime | None = None
    notes: tuple[str, ...] = ()


class ERA5LandResult(BaseModel):
    """Normalized observations plus monthly series and underived metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: ERA5LandRawMetadata
    window: AnalysisWindow | None
    monthly: tuple[ERA5MonthlyClimate, ...]
    derivations: tuple[ERA5Derivation, ...]
    observations: tuple[IndicatorObservation, ...]


def era5_module_available(module_name: str) -> bool:
    """Return whether an importable module spec exists (no import side effects)."""

    import importlib.util

    return importlib.util.find_spec(module_name) is not None


def era5_extra_status() -> dict[str, bool]:
    """Availability of each declared ERA5 extra module."""

    return {name: era5_module_available(name) for name in ERA5_EXTRA_MODULES}


def public_cds_dataset_url(dataset_id: str = ERA5_LAND_DATASET_ID) -> str:
    """Catalogue URL without credentials or userinfo."""

    return f"https://cds.climate.copernicus.eu/datasets/{dataset_id}"


def settings_from_environ(
    environ: Mapping[str, str] | None = None,
    *,
    dataset_path: str | Path | None = None,
    cds_key: str | None = None,
    cds_url: str | None = None,
    precipitation_convention: PrecipitationConvention | str | None = None,
) -> ERA5LandSettings:
    """Build settings from constructor overrides, then process environment.

    Standard CDS variables ``CDSAPI_URL`` / ``CDSAPI_KEY`` are accepted.
    ``~/.cdsapirc`` is not read: SolarDSS does not scan home-directory secrets.
    """

    source = os.environ if environ is None else environ
    path = _first_text(dataset_path, source.get(ERA5_LAND_PATH_ENV))
    key = _first_text(cds_key, source.get(ERA5_CDS_KEY_ENV), source.get(CDSAPI_KEY_ENV))
    url = _first_text(
        cds_url,
        source.get(ERA5_CDS_URL_ENV),
        source.get(CDSAPI_URL_ENV),
        DEFAULT_CDS_API_URL,
    )
    dataset_id = _first_text(source.get(ERA5_LAND_DATASET_ENV), ERA5_LAND_DATASET_ID)
    window = _first_text(source.get(ERA5_WINDOW_ENV), "typical") or "typical"
    typical_start = _int_text(source.get(ERA5_TYPICAL_START_ENV), WMO_NORMAL_START_YEAR)
    typical_end = _int_text(source.get(ERA5_TYPICAL_END_ENV), WMO_NORMAL_END_YEAR)
    recent_years = _int_text(source.get(ERA5_RECENT_YEARS_ENV), DEFAULT_RECENT_YEARS)
    convention = precipitation_convention
    if convention is None:
        raw_conv = source.get(ERA5_PRECIP_CONV_ENV)
        convention = raw_conv.strip() if isinstance(raw_conv, str) and raw_conv.strip() else None
    parsed_conv: PrecipitationConvention | None
    if convention is None:
        parsed_conv = None
    elif isinstance(convention, PrecipitationConvention):
        parsed_conv = convention
    else:
        parsed_conv = PrecipitationConvention(convention)
    return ERA5LandSettings(
        dataset_id=dataset_id or ERA5_LAND_DATASET_ID,
        dataset_path=path,
        cds_url=url or DEFAULT_CDS_API_URL,
        cds_key=SecretStr(key) if key else None,
        typical_start_year=typical_start,
        typical_end_year=typical_end,
        recent_years=recent_years,
        precipitation_convention=parsed_conv,
        window=window,
    )


def kelvin_to_celsius(values: FloatArray) -> FloatArray:
    """Convert ERA5-Land air temperature from kelvin to degree Celsius."""

    return np.asarray(values, dtype=np.float64) - KELVIN_OFFSET


def wind_speed_from_components(u10: FloatArray, v10: FloatArray) -> FloatArray:
    """10 m wind-speed magnitude from official ``u10`` and ``v10`` components."""

    east = np.asarray(u10, dtype=np.float64)
    north = np.asarray(v10, dtype=np.float64)
    return np.sqrt(east * east + north * north)


def magnus_relative_humidity(t_deg_c: FloatArray, td_deg_c: FloatArray) -> FloatArray:
    """Relative humidity (%) from 2 m temperature and dewpoint.

    Formula (Alduchov and Eskridge 1996):
    ``RH = 100 * exp(a*Td/(b+Td) - a*T/(b+T))`` with ``a=17.625``, ``b=243.04``
    and temperatures in degree Celsius. ERA5-Land does not publish RH.
    """

    temperature = np.asarray(t_deg_c, dtype=np.float64)
    dewpoint = np.asarray(td_deg_c, dtype=np.float64)
    denom_t = MAGNUS_B + temperature
    denom_d = MAGNUS_B + dewpoint
    ratio = np.full(temperature.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(temperature) & np.isfinite(dewpoint) & (denom_t != 0.0) & (denom_d != 0.0)
    log_rh = (MAGNUS_A * dewpoint[valid] / denom_d[valid]) - (
        MAGNUS_A * temperature[valid] / denom_t[valid]
    )
    ratio[valid] = 100.0 * np.exp(log_rh)
    return ratio


def snow_cover_to_fraction(snowc: FloatArray, *, units: str | None) -> FloatArray:
    """Convert official snow cover to a 0-1 fraction.

    ERA5-Land documents ``snow_cover`` / ``snowc`` in percent.
    """

    values = np.asarray(snowc, dtype=np.float64)
    normalized = (units or "").strip().lower().replace(" ", "")
    if normalized in {"(0-1)", "1", "fraction"}:
        return values
    return values / 100.0


def deaccumulate_forecast_hourly(times: pd.DatetimeIndex, accumulated_m: FloatArray) -> FloatArray:
    """Hourly precipitation increments from ERA5-Land forecast accumulations.

    CDS validity time 00 UTC is the 24-hour accumulation of the previous day
    (MARS step 24). Hours 01-23 UTC accumulate from 00 UTC that day.
    Gaps are left as NaN; they are not interpolated.
    """

    acc = np.asarray(accumulated_m, dtype=np.float64)
    out = np.full(acc.shape, np.nan, dtype=np.float64)
    index = _naive_utc_index(times)
    hours = index.hour.to_numpy()
    prev_acc = np.nan
    prev_hour = -1
    for i, hour in enumerate(hours.tolist()):
        value = acc[i]
        if not np.isfinite(value):
            prev_acc = np.nan
            prev_hour = -1
            continue
        if hour == 1:
            out[i] = value
        else:
            consecutive = prev_hour >= 0 and hour == prev_hour + 1 and np.isfinite(prev_acc)
            day_wrap = hour == 0 and prev_hour == 23 and np.isfinite(prev_acc)
            if consecutive or day_wrap:
                out[i] = value - prev_acc
        prev_acc = value
        prev_hour = hour
    return out


def complete_calendar_years(index: pd.DatetimeIndex) -> tuple[int, ...]:
    """Years that contain every calendar month. Incomplete years are omitted."""

    if index.empty:
        return ()
    naive = _naive_utc_index(index)
    years: list[int] = []
    for year in sorted(set(naive.year.tolist())):
        months = set(naive[naive.year == year].month.tolist())
        if months == set(range(1, 13)):
            years.append(int(year))
    return tuple(years)


def resolve_analysis_window(
    request: FetchRequest,
    settings: ERA5LandSettings,
    *,
    now: datetime,
) -> AnalysisWindow:
    """Resolve typical / recent / explicit windows without inventing years."""

    start_year = _int_param(request.parameters, "start_year")
    if start_year is None:
        start_year = _int_param(request.parameters, "startyear")
    end_year = _int_param(request.parameters, "end_year")
    if end_year is None:
        end_year = _int_param(request.parameters, "endyear")
    kind = _text_param(request.parameters, "window") or settings.window
    recent_years = _int_param(request.parameters, "recent_years") or settings.recent_years
    if start_year is not None and end_year is not None:
        return _explicit_window(start_year, end_year)
    if request.temporal is not None and request.temporal.start and request.temporal.end:
        return AnalysisWindow(
            kind="explicit",
            start=request.temporal.start,
            end=request.temporal.end,
            label=(
                f"explicit request period {request.temporal.start.isoformat()} to "
                f"{request.temporal.end.isoformat()}"
            ),
        )
    if kind == "recent":
        last_complete = now.date().year - 1
        first = last_complete - recent_years + 1
        return AnalysisWindow(
            kind="recent",
            start=date(first, 1, 1),
            end=date(last_complete, 12, 31),
            label=f"recent {recent_years} complete calendar years through {last_complete}",
        )
    if kind not in {"typical", "explicit", "recent"}:
        raise InvalidParameterError("ERA5-Land window must be typical, recent, or explicit")
    start = settings.typical_start_year
    end = settings.typical_end_year
    return AnalysisWindow(
        kind="typical",
        start=date(start, 1, 1),
        end=date(end, 12, 31),
        label=(
            f"WMO-style climate-normal window {start}-{end}; "
            "not a generated typical meteorological year"
        ),
    )


def build_cds_retrieve_request(
    *,
    latitude: float,
    longitude: float,
    window: AnalysisWindow,
    pad_deg: float = DEFAULT_SPATIAL_PAD_DEG,
    dataset_id: str = ERA5_LAND_DATASET_ID,
) -> dict[str, JsonValue]:
    """CDS retrieve body with spatial and temporal subsetting already applied."""

    del dataset_id
    north = min(90.0, latitude + pad_deg)
    south = max(-90.0, latitude - pad_deg)
    west = max(-180.0, longitude - pad_deg)
    east = min(180.0, longitude + pad_deg)
    years: list[JsonScalar] = [
        f"{year:04d}" for year in range(window.start.year, window.end.year + 1)
    ]
    months: list[JsonScalar] = [f"{month:02d}" for month in range(1, 13)]
    days: list[JsonScalar] = [f"{day:02d}" for day in range(1, 32)]
    times: list[JsonScalar] = [f"{hour:02d}:00" for hour in range(24)]
    variables: list[JsonScalar] = list(CDS_RETRIEVE_VARIABLES)
    area: list[JsonScalar] = [north, west, south, east]
    return {
        "variable": variables,
        "year": years,
        "month": months,
        "day": days,
        "time": times,
        "area": area,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def _first_text(*values: object) -> str | None:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _int_text(raw: str | None, default: int) -> int:
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _text_param(parameters: Mapping[str, JsonValue], key: str) -> str | None:
    value = parameters.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _int_param(parameters: Mapping[str, JsonValue], key: str) -> int | None:
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return int(value)


def _float_param(parameters: Mapping[str, JsonValue], key: str) -> float | None:
    value = parameters.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _explicit_window(start_year: int, end_year: int) -> AnalysisWindow:
    if end_year < start_year:
        raise InvalidParameterError("ERA5-Land end_year must be >= start_year")
    return AnalysisWindow(
        kind="explicit",
        start=date(start_year, 1, 1),
        end=date(end_year, 12, 31),
        label=f"explicit calendar years {start_year}-{end_year}",
    )


def _naive_utc_index(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if index.tz is None:
        return index
    return index.tz_convert("UTC").tz_localize(None)


def _require_xarray() -> Any:
    try:
        import xarray as xr
    except ImportError as exc:
        msg = "Optional ERA5 extra is not installed. Install with: pip install 'solardss[era5]'."
        raise MissingExtraError(msg) from exc
    return xr


def _encode_observation(observation: IndicatorObservation) -> bytes:
    envelope = {
        "schema_id": _CACHE_SCHEMA,
        "observation": observation.model_dump(mode="json"),
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _decode_observation(payload: bytes) -> IndicatorObservation:
    try:
        raw: object = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CacheError("cache entry is not valid JSON") from exc
    if not isinstance(raw, dict) or raw.get("schema_id") != _CACHE_SCHEMA:
        raise CacheError("cache entry has unknown schema")
    body = raw.get("observation")
    if not isinstance(body, dict):
        raise CacheError("cache entry is missing an observation")
    try:
        return IndicatorObservation.model_validate(body)
    except (InvalidParameterError, ValueError, TypeError) as exc:
        raise CacheError("cache entry is not an IndicatorObservation") from exc


def _finite_mean(values: FloatArray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.mean(finite))


def _finite_min(values: FloatArray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.min(finite))


def _finite_max(values: FloatArray) -> float | None:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def _year_mask(index: pd.DatetimeIndex, years: Sequence[int]) -> BoolArray:
    if not years:
        return np.zeros(len(index), dtype=bool)
    return np.asarray(_naive_utc_index(index).year.isin(list(years)), dtype=bool)


def _median_hours(index: pd.DatetimeIndex) -> float | None:
    if len(index) < 2:
        return None
    stamps = _naive_utc_index(index).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    delta = np.diff(stamps) / 3.6e12
    if delta.size == 0:
        return None
    return float(np.median(delta))


def _units_of(dataset: Any, name: str) -> str | None:
    variable = dataset[name]
    units = variable.attrs.get("units")
    if units is None:
        units = variable.attrs.get("unit")
    if units is None:
        return None
    return str(units)


def _find_var(dataset: Any, canonical: str) -> str | None:
    aliases = _VAR_ALIASES[canonical]
    data_vars = set(dataset.data_vars)
    coords = set(dataset.coords)
    for alias in aliases:
        if alias in data_vars or alias in coords:
            return alias
    lowered = {str(name).lower(): name for name in list(dataset.data_vars)}
    for alias in aliases:
        match = lowered.get(alias.lower())
        if match is not None:
            return str(match)
    return None


def _dim_name(dataset: Any, candidates: tuple[str, ...]) -> str | None:
    dims = set(dataset.dims)
    coords = set(dataset.coords)
    for name in candidates:
        if name in dims or name in coords:
            return name
    return None


def _wrap_longitude(lon: float, lon_values: FloatArray) -> float:
    finite = lon_values[np.isfinite(lon_values)]
    if finite.size == 0:
        return lon
    lon_min = float(np.min(finite))
    lon_max = float(np.max(finite))
    if lon_min >= 0.0 and lon_max > 180.0 and lon < 0.0:
        return lon + 360.0
    if lon_min < 0.0 and lon > 180.0:
        return lon - 360.0
    return lon


def _lat_slice(lat_min: float, lat_max: float, lat_values: FloatArray) -> slice:
    finite = lat_values[np.isfinite(lat_values)]
    if finite.size >= 2 and float(finite[0]) > float(finite[-1]):
        return slice(lat_max, lat_min)
    return slice(lat_min, lat_max)


def _to_1d(values: Any) -> FloatArray:
    return np.ravel(np.asarray(values, dtype=np.float64))


class _PointExtract:
    __slots__ = (
        "cloud_cover",
        "cloud_name",
        "d2m_k",
        "d2m_name",
        "grid_latitude",
        "grid_longitude",
        "snow_cover",
        "snow_cover_name",
        "snow_cover_units",
        "snow_depth",
        "snow_depth_name",
        "t2m_k",
        "t2m_name",
        "times",
        "tp_m",
        "tp_name",
        "u10",
        "u10_name",
        "units",
        "v10",
        "v10_name",
        "variables_present",
    )

    def __init__(self) -> None:
        self.times = pd.DatetimeIndex([])
        self.t2m_k: FloatArray | None = None
        self.d2m_k: FloatArray | None = None
        self.u10: FloatArray | None = None
        self.v10: FloatArray | None = None
        self.tp_m: FloatArray | None = None
        self.snow_cover: FloatArray | None = None
        self.snow_depth: FloatArray | None = None
        self.cloud_cover: FloatArray | None = None
        self.t2m_name: str | None = None
        self.d2m_name: str | None = None
        self.u10_name: str | None = None
        self.v10_name: str | None = None
        self.tp_name: str | None = None
        self.snow_cover_name: str | None = None
        self.snow_depth_name: str | None = None
        self.snow_cover_units: str | None = None
        self.cloud_name: str | None = None
        self.units: dict[str, str] = {}
        self.variables_present: tuple[str, ...] = ()
        self.grid_latitude: float | None = None
        self.grid_longitude: float | None = None


def _subset_before_load(
    dataset: Any,
    *,
    latitude: float | None,
    longitude: float | None,
    bbox: BoundingBox | None,
    window: AnalysisWindow,
    pad_deg: float,
) -> Any:
    """Spatial and temporal selection on a lazy dataset before ``.load()``."""

    lat_name = _dim_name(dataset, ("latitude", "lat", "Latitude"))
    lon_name = _dim_name(dataset, ("longitude", "lon", "Longitude"))
    time_name = _dim_name(dataset, ("time", "valid_time"))
    selected = dataset
    if time_name is not None:
        selected = selected.sel(
            {time_name: slice(np.datetime64(window.start), np.datetime64(window.end))}
        )
    if lat_name is None or lon_name is None:
        return selected
    lat_values = np.asarray(dataset[lat_name].values, dtype=np.float64)
    lon_values = np.asarray(dataset[lon_name].values, dtype=np.float64)
    if bbox is not None:
        lat_min, lat_max = bbox.min_lat, bbox.max_lat
        lon_min = _wrap_longitude(bbox.min_lon, lon_values)
        lon_max = _wrap_longitude(bbox.max_lon, lon_values)
    elif latitude is not None and longitude is not None:
        wrapped = _wrap_longitude(longitude, lon_values)
        lat_min, lat_max = latitude - pad_deg, latitude + pad_deg
        lon_min, lon_max = wrapped - pad_deg, wrapped + pad_deg
    else:
        return selected
    selected = selected.sel(
        {
            lat_name: _lat_slice(lat_min, lat_max, lat_values),
            lon_name: slice(lon_min, lon_max),
        }
    )
    return selected


def _extract_point(
    dataset: Any,
    *,
    latitude: float | None,
    longitude: float | None,
    bbox: BoundingBox | None,
) -> _PointExtract:
    extract = _PointExtract()
    present = [str(name) for name in dataset.data_vars]
    extract.variables_present = tuple(present)
    for name in present:
        units = _units_of(dataset, name)
        if units is not None:
            extract.units[name] = units
    lat_name = _dim_name(dataset, ("latitude", "lat", "Latitude"))
    lon_name = _dim_name(dataset, ("longitude", "lon", "Longitude"))
    time_name = _dim_name(dataset, ("time", "valid_time"))
    point = dataset
    has_point = lat_name is not None and lon_name is not None
    if has_point and latitude is not None and longitude is not None:
        assert lat_name is not None
        assert lon_name is not None
        lon_values = np.asarray(dataset[lon_name].values, dtype=np.float64)
        wrapped = _wrap_longitude(longitude, lon_values)
        point = dataset.sel({lat_name: latitude, lon_name: wrapped}, method="nearest")
        extract.grid_latitude = float(np.asarray(point[lat_name].values).reshape(-1)[0])
        extract.grid_longitude = float(np.asarray(point[lon_name].values).reshape(-1)[0])
    elif bbox is not None and lat_name is not None and lon_name is not None:
        spatial_dims = [name for name in (lat_name, lon_name) if name in dataset.dims]
        point = dataset.mean(dim=spatial_dims)
        extract.grid_latitude = (bbox.min_lat + bbox.max_lat) / 2.0
        extract.grid_longitude = (bbox.min_lon + bbox.max_lon) / 2.0
    if time_name is not None:
        extract.times = pd.DatetimeIndex(np.asarray(point[time_name].values))
    mapping = {
        "t2m": "t2m_k",
        "d2m": "d2m_k",
        "u10": "u10",
        "v10": "v10",
        "tp": "tp_m",
        "snowc": "snow_cover",
        "tcc": "cloud_cover",
    }
    for canonical, attr in mapping.items():
        found = _find_var(point, canonical)
        if found is None:
            continue
        setattr(extract, attr, _to_1d(point[found].values))
        if canonical == "t2m":
            extract.t2m_name = found
        elif canonical == "d2m":
            extract.d2m_name = found
        elif canonical == "u10":
            extract.u10_name = found
        elif canonical == "v10":
            extract.v10_name = found
        elif canonical == "tp":
            extract.tp_name = found
        elif canonical == "snowc":
            extract.snow_cover_name = found
            extract.snow_cover_units = extract.units.get(found)
        elif canonical == "tcc":
            extract.cloud_name = found
    snow_depth_name = _find_var(point, "sde") or _find_var(point, "sd")
    if snow_depth_name is not None:
        extract.snow_depth = _to_1d(point[snow_depth_name].values)
        extract.snow_depth_name = snow_depth_name
    return extract


def _monthly_rows(
    times: pd.DatetimeIndex,
    *,
    t2m_c: FloatArray | None,
    wind: FloatArray | None,
    rh: FloatArray | None,
    precip_mm: FloatArray | None,
    snow_fraction: FloatArray | None,
    cloud: FloatArray | None,
) -> tuple[ERA5MonthlyClimate, ...]:
    if times.empty:
        return ()
    frame = pd.DataFrame({"_t": _naive_utc_index(times)})
    frame["year"] = frame["_t"].dt.year
    frame["month"] = frame["_t"].dt.month
    if t2m_c is not None:
        frame["t2m"] = t2m_c
    if wind is not None:
        frame["wind"] = wind
    if rh is not None:
        frame["rh"] = rh
    if precip_mm is not None:
        frame["precip"] = precip_mm
    if snow_fraction is not None:
        frame["snow"] = snow_fraction
    if cloud is not None:
        frame["cloud"] = cloud
    rows: list[ERA5MonthlyClimate] = []
    grouped = frame.groupby(["year", "month"], sort=True)
    for _key, group in grouped:
        year = int(group["year"].iloc[0])
        month = int(group["month"].iloc[0])
        present: list[str] = []
        t_mean = _finite_mean(group["t2m"].to_numpy()) if "t2m" in group else None
        t_min = _finite_min(group["t2m"].to_numpy()) if "t2m" in group else None
        t_max = _finite_max(group["t2m"].to_numpy()) if "t2m" in group else None
        if t_mean is not None:
            present.append("t2m")
        wind_mean = _finite_mean(group["wind"].to_numpy()) if "wind" in group else None
        if wind_mean is not None:
            present.append("wind")
        rh_mean = _finite_mean(group["rh"].to_numpy()) if "rh" in group else None
        if rh_mean is not None:
            present.append("rh")
        precip = None
        if "precip" in group:
            finite = group["precip"].to_numpy()
            finite = finite[np.isfinite(finite)]
            if finite.size:
                precip = float(np.sum(finite))
                present.append("precip")
        snow = _finite_mean(group["snow"].to_numpy()) if "snow" in group else None
        if snow is not None:
            present.append("snow")
        cloud_mean = _finite_mean(group["cloud"].to_numpy()) if "cloud" in group else None
        if cloud_mean is not None:
            present.append("cloud")
        rows.append(
            ERA5MonthlyClimate(
                year=int(year),
                month=int(month),
                t2m_mean_deg_c=t_mean,
                t2m_min_deg_c=t_min,
                t2m_max_deg_c=t_max,
                wind_speed_mean_m_s=wind_mean,
                relative_humidity_mean_percent=rh_mean,
                precipitation_mm=precip,
                snow_cover_mean_fraction=snow,
                cloud_cover_mean=cloud_mean,
                present_fields=tuple(present),
            )
        )
    return tuple(rows)


class ERA5LandWeatherProvider:
    """Optional ERA5-Land weather provider. Missing credentials do not crash."""

    provider_id_default = "era5_land.weather"
    source_name_default = ERA5_LAND_SOURCE_NAME
    served = _SERVED

    def __init__(
        self,
        *,
        settings: ERA5LandSettings | None = None,
        environ: Mapping[str, str] | None = None,
        dataset: object | None = None,
        dataset_path: str | Path | None = None,
        cds_key: str | None = None,
        cds_url: str | None = None,
        cds_retrieve: CdsRetrieveFn | None = None,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        provider_id: str | None = None,
        extreme_heat_threshold_deg_c: float | None = None,
        precipitation_convention: PrecipitationConvention | str | None = None,
        spatial_pad_deg: float | None = None,
    ) -> None:
        base = settings or settings_from_environ(
            environ,
            dataset_path=dataset_path,
            cds_key=cds_key,
            cds_url=cds_url,
            precipitation_convention=precipitation_convention,
        )
        updates: dict[str, Any] = {}
        if dataset_path is not None:
            updates["dataset_path"] = str(dataset_path)
        if cds_key:
            updates["cds_key"] = SecretStr(cds_key)
        if cds_url:
            updates["cds_url"] = cds_url
        if precipitation_convention is not None:
            if isinstance(precipitation_convention, PrecipitationConvention):
                updates["precipitation_convention"] = precipitation_convention
            else:
                updates["precipitation_convention"] = PrecipitationConvention(
                    precipitation_convention
                )
        if spatial_pad_deg is not None:
            updates["spatial_pad_deg"] = spatial_pad_deg
        self._settings = base.model_copy(update=updates) if updates else base
        self._dataset = dataset
        self._cds_retrieve = cds_retrieve
        self._cache = cache
        self._clock: Clock = UtcClock() if clock is None else clock
        self._provider_id = provider_id or self.provider_id_default
        self._heat_threshold = extreme_heat_threshold_deg_c

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def settings(self) -> ERA5LandSettings:
        return self._settings

    def _configured(self) -> bool:
        if self._dataset is not None:
            return True
        path = self._settings.dataset_path
        if path:
            return True
        return self._settings.has_cds_key() or self._cds_retrieve is not None

    def health(self) -> ProviderHealth:
        checked_at = self._clock.now()
        details: dict[str, JsonValue] = {
            "dataset_id": self._settings.dataset_id,
            "source_url": public_cds_dataset_url(self._settings.dataset_id),
            "has_cds_key": self._settings.has_cds_key(),
            "has_local_dataset": bool(self._settings.dataset_path) or self._dataset is not None,
            "xarray": era5_module_available("xarray"),
        }
        if not era5_module_available("xarray") and self._dataset is None:
            return ProviderHealth(
                status=HealthStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                reason="optional ERA5 extra is not installed (pip install 'solardss[era5]')",
                checked_at=checked_at,
                details=details,
            )
        if not self._configured():
            return ProviderHealth(
                status=HealthStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                reason=(
                    "ERA5-Land is not configured: set SOLARDSS_ERA5_LAND_PATH or "
                    "CDSAPI_KEY / SOLARDSS_CDSAPI_KEY"
                ),
                checked_at=checked_at,
                details=details,
            )
        path = self._settings.dataset_path
        if (
            self._dataset is None
            and path is not None
            and not Path(path).exists()
            and not self._settings.has_cds_key()
        ):
            return ProviderHealth(
                status=HealthStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                reason="ERA5-Land dataset_path does not exist",
                checked_at=checked_at,
                details=details,
            )
        return ProviderHealth(
            status=HealthStatus.OK,
            provider_id=self._provider_id,
            reason="ERA5-Land provider is configured",
            checked_at=checked_at,
            details=details,
        )

    def _heat_threshold_for(self, request: FetchRequest) -> float | None:
        value = _float_param(request.parameters, "extreme_heat_threshold_degC")
        if value is None:
            value = _float_param(request.parameters, "extreme_heat_threshold_deg_c")
        if value is not None:
            return value
        return self._heat_threshold

    def availability(self, request: FetchRequest) -> ProviderAvailability:
        if request.indicator_id not in self.served:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="indicator is not served by the ERA5-Land provider",
            )
        if not era5_module_available("xarray") and self._dataset is None:
            return ProviderAvailability(
                status=AvailabilityStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="optional ERA5 extra is not installed (pip install 'solardss[era5]')",
            )
        if not self._configured():
            return ProviderAvailability(
                status=AvailabilityStatus.NOT_CONFIGURED,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=(
                    "ERA5-Land is not configured: set SOLARDSS_ERA5_LAND_PATH or "
                    "CDSAPI_KEY / SOLARDSS_CDSAPI_KEY"
                ),
            )
        point = request_point(request)
        bbox = request_bbox(request)
        if point is None and bbox is None:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="ERA5-Land requires a latitude/longitude point or a bounding box",
            )
        if (
            request.indicator_id == IndicatorId.CLIMATE_EXTREME_HEAT_DAYS.value
            and self._heat_threshold_for(request) is None
        ):
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=(
                    "ERA5-Land does not publish an extreme-heat threshold; "
                    "supply extreme_heat_threshold_degC"
                ),
            )
        return ProviderAvailability(
            status=AvailabilityStatus.AVAILABLE,
            provider_id=self._provider_id,
            retryable=False,
            indicator_id=request.indicator_id,
            reason="ERA5-Land can be queried for this request",
        )

    def coverage(self, request: FetchRequest) -> CoverageReport:
        availability = self.availability(request)
        point = request_point(request)
        spatial = None
        if point is not None:
            spatial = SpatialCoverage(
                resolution=ERA5_LAND_RESOLUTION,
                crs="EPSG:4326",
                geometry_fingerprint=f"lat={point[1]},lon={point[0]}",
            )
        elif request.bbox is not None:
            spatial = SpatialCoverage(
                resolution=ERA5_LAND_RESOLUTION,
                crs=request.bbox.crs,
                bbox=request.bbox,
            )
        temporal = None
        if availability.status is AvailabilityStatus.AVAILABLE:
            window = resolve_analysis_window(request, self._settings, now=self._clock.now())
            temporal = TemporalCoverage(
                resolution="era5-land-subset",
                start=window.start,
                end=window.end,
                aggregation=window.kind,
            )
        return CoverageReport(
            status=availability.status,
            provider_id=self._provider_id,
            spatial=spatial,
            temporal=temporal,
            reason=availability.reason,
        )

    def fetch(self, request: FetchRequest) -> IndicatorObservation:
        request_hash = request_cache_key(self._provider_id, request)
        availability = self.availability(request)
        if availability.status is AvailabilityStatus.TEMPORARY_FAILURE:
            return self._miss(
                request,
                status=ObservationStatus.TEMPORARY_FAILURE,
                reason=availability.reason,
                retrieved_at=self._clock.now(),
            )
        cached = self._read_cache(request_hash)
        if cached is not None:
            return cached
        bundle = self.fetch_climate(request)
        for item in bundle.observations:
            if item.indicator_id == request.indicator_id:
                if item.status is not ObservationStatus.TEMPORARY_FAILURE:
                    self._write_cache(request_hash, item)
                return item
        miss = self._miss(
            request,
            status=ObservationStatus.UNAVAILABLE,
            reason="indicator was not produced from the ERA5-Land subset",
            retrieved_at=self._clock.now(),
        )
        return miss

    def fetch_climate(self, request: FetchRequest) -> ERA5LandResult:
        availability = self.availability(request)
        source_url = public_cds_dataset_url(self._settings.dataset_id)
        if availability.status is not AvailabilityStatus.AVAILABLE:
            retrieved_at = self._clock.now()
            miss = self._miss(
                request,
                status=ObservationStatus(availability.status.value),
                reason=availability.reason,
                retrieved_at=retrieved_at,
            )
            return ERA5LandResult(
                raw=ERA5LandRawMetadata(
                    dataset_id=self._settings.dataset_id,
                    source_url=source_url,
                    variables_present=(),
                    units={},
                    retrieved_at=retrieved_at,
                    notes=(availability.reason,),
                ),
                window=None,
                monthly=(),
                derivations=(),
                observations=(miss,),
            )
        retrieved_at = self._clock.now()
        try:
            window = resolve_analysis_window(request, self._settings, now=retrieved_at)
            loaded = self._load_subset(request, window)
            point = request_point(request)
            bbox = request_bbox(request)
            latitude = point[1] if point else None
            longitude = point[0] if point else None
            extract = _extract_point(
                loaded,
                latitude=latitude,
                longitude=longitude,
                bbox=bbox if point is None else None,
            )
        except MissingExtraError as exc:
            miss = self._miss(
                request,
                status=ObservationStatus.NOT_CONFIGURED,
                reason=exc.message,
                retrieved_at=retrieved_at,
            )
            return ERA5LandResult(
                raw=ERA5LandRawMetadata(
                    dataset_id=self._settings.dataset_id,
                    source_url=source_url,
                    variables_present=(),
                    units={},
                    retrieved_at=retrieved_at,
                    notes=(exc.message,),
                ),
                window=None,
                monthly=(),
                derivations=(),
                observations=(miss,),
            )
        except (ERA5LandResponseError, OSError, ValueError, KeyError, TypeError) as exc:
            reason = str(exc)
            miss = self._miss(
                request,
                status=ObservationStatus.TEMPORARY_FAILURE,
                reason=reason,
                retrieved_at=retrieved_at,
            )
            return ERA5LandResult(
                raw=ERA5LandRawMetadata(
                    dataset_id=self._settings.dataset_id,
                    source_url=source_url,
                    variables_present=(),
                    units={},
                    retrieved_at=retrieved_at,
                    notes=(reason,),
                ),
                window=None,
                monthly=(),
                derivations=(),
                observations=(miss,),
            )
        if extract.times.empty:
            miss = self._miss(
                request,
                status=ObservationStatus.MISSING_COVERAGE,
                reason="ERA5-Land subset has no timesteps in the requested window",
                retrieved_at=retrieved_at,
                grid_lat=extract.grid_latitude,
                grid_lon=extract.grid_longitude,
            )
            return ERA5LandResult(
                raw=self._raw_meta(extract, source_url, retrieved_at, window),
                window=window,
                monthly=(),
                derivations=(),
                observations=(miss,),
            )
        if self._all_masked(extract):
            miss = self._miss(
                request,
                status=ObservationStatus.UNSUPPORTED_REGION,
                reason="ERA5-Land nearest cell is masked (typically ocean)",
                retrieved_at=retrieved_at,
                grid_lat=extract.grid_latitude,
                grid_lon=extract.grid_longitude,
            )
            return ERA5LandResult(
                raw=self._raw_meta(extract, source_url, retrieved_at, window),
                window=window,
                monthly=(),
                derivations=(),
                observations=(miss,),
            )
        observations, monthly, derivations = self._observations(
            request,
            extract=extract,
            window=window,
            retrieved_at=retrieved_at,
            source_url=source_url,
        )
        return ERA5LandResult(
            raw=self._raw_meta(extract, source_url, retrieved_at, window),
            window=window,
            monthly=monthly,
            derivations=derivations,
            observations=observations,
        )

    def _all_masked(self, extract: _PointExtract) -> bool:
        arrays = [
            extract.t2m_k,
            extract.d2m_k,
            extract.u10,
            extract.v10,
            extract.tp_m,
            extract.snow_cover,
            extract.snow_depth,
            extract.cloud_cover,
        ]
        present = [item for item in arrays if item is not None]
        if not present:
            return True
        return all(not np.any(np.isfinite(item)) for item in present)

    def _raw_meta(
        self,
        extract: _PointExtract,
        source_url: str,
        retrieved_at: datetime,
        window: AnalysisWindow,
    ) -> ERA5LandRawMetadata:
        time_start = None
        time_end = None
        if not extract.times.empty:
            naive = _naive_utc_index(extract.times)
            time_start = naive[0].date()
            time_end = naive[-1].date()
        return ERA5LandRawMetadata(
            dataset_id=self._settings.dataset_id,
            source_url=source_url,
            variables_present=extract.variables_present,
            units=extract.units,
            grid_latitude=extract.grid_latitude,
            grid_longitude=extract.grid_longitude,
            time_start=time_start,
            time_end=time_end,
            retrieved_at=retrieved_at,
            notes=(window.label,),
        )

    def _load_subset(self, request: FetchRequest, window: AnalysisWindow) -> Any:
        point = request_point(request)
        bbox = request_bbox(request)
        latitude = point[1] if point else None
        longitude = point[0] if point else None

        def _select(dataset: Any) -> Any:
            subset = _subset_before_load(
                dataset,
                latitude=latitude,
                longitude=longitude,
                bbox=bbox,
                window=window,
                pad_deg=self._settings.spatial_pad_deg,
            )
            return subset.load()

        if self._dataset is not None:
            return _select(self._dataset)
        path = self._settings.dataset_path
        if path and Path(path).exists():
            xr = _require_xarray()
            with xr.open_dataset(path) as dataset:
                return _select(dataset)
        if point is None:
            raise ERA5LandResponseError("CDS retrieve requires a latitude/longitude point")
        lon, lat = point
        payload = build_cds_retrieve_request(
            latitude=lat,
            longitude=lon,
            window=window,
            pad_deg=self._settings.spatial_pad_deg,
            dataset_id=self._settings.dataset_id,
        )
        retrieve = self._cds_retrieve or self._cdsapi_retrieve
        xr = _require_xarray()
        with tempfile.TemporaryDirectory(prefix="solardss-era5-") as tmp:
            target = Path(tmp) / "era5-land-subset.nc"
            retrieve(self._settings.dataset_id, payload, target)
            if not target.exists():
                raise ERA5LandResponseError("CDS retrieve did not write a dataset file")
            with xr.open_dataset(target) as dataset:
                return _select(dataset)

    def _cdsapi_retrieve(self, dataset: str, request: Mapping[str, object], target: Path) -> None:
        key = self._settings.cds_key_value()
        if not key:
            raise ERA5LandResponseError("CDS API key is not configured")
        try:
            import cdsapi
        except ImportError as exc:
            msg = "cdsapi is not installed. Install with: pip install 'solardss[era5]'."
            raise MissingExtraError(msg) from exc
        client = cdsapi.Client(url=self._settings.cds_url, key=key)
        client.retrieve(dataset, dict(request), str(target))

    def _observations(
        self,
        request: FetchRequest,
        *,
        extract: _PointExtract,
        window: AnalysisWindow,
        retrieved_at: datetime,
        source_url: str,
    ) -> tuple[
        tuple[IndicatorObservation, ...],
        tuple[ERA5MonthlyClimate, ...],
        tuple[ERA5Derivation, ...],
    ]:
        years = complete_calendar_years(extract.times)
        t2m_c = kelvin_to_celsius(extract.t2m_k) if extract.t2m_k is not None else None
        d2m_c = kelvin_to_celsius(extract.d2m_k) if extract.d2m_k is not None else None
        wind = None
        if extract.u10 is not None and extract.v10 is not None:
            wind = wind_speed_from_components(extract.u10, extract.v10)
        rh = None
        if t2m_c is not None and d2m_c is not None:
            rh = magnus_relative_humidity(t2m_c, d2m_c)
        precip_mm, precip_derivation, precip_reason = self._precipitation_mm(request, extract)
        snow_fraction, snow_derivation, snow_proxy = self._snow_fraction(extract)
        derivations: list[ERA5Derivation] = []
        if t2m_c is not None and extract.t2m_name:
            derivations.append(
                ERA5Derivation(
                    method_id=CALC_K_TO_C,
                    formula="T_degC = t2m_K - 273.15",
                    source_variables=(extract.t2m_name,),
                    derived_from=(extract.t2m_name,),
                )
            )
        if wind is not None and extract.u10_name and extract.v10_name:
            derivations.append(
                ERA5Derivation(
                    method_id=CALC_WIND,
                    formula="ws = sqrt(u10**2 + v10**2)",
                    source_variables=(extract.u10_name, extract.v10_name),
                    derived_from=(extract.u10_name, extract.v10_name),
                )
            )
        if rh is not None and extract.t2m_name and extract.d2m_name:
            derivations.append(
                ERA5Derivation(
                    method_id=CALC_RH,
                    formula=(
                        "RH=100*exp(a*Td/(b+Td)-a*T/(b+T)); a=17.625; b=243.04 degC; "
                        "T,Td from t2m,d2m minus 273.15 (Alduchov and Eskridge 1996)"
                    ),
                    source_variables=(extract.t2m_name, extract.d2m_name),
                    derived_from=(extract.t2m_name, extract.d2m_name),
                )
            )
        if precip_derivation is not None:
            derivations.append(precip_derivation)
        if snow_derivation is not None:
            derivations.append(snow_derivation)
        if extract.cloud_cover is not None and extract.cloud_name:
            derivations.append(
                ERA5Derivation(
                    method_id=CALC_CLOUD,
                    formula="mean(total_cloud_cover) over complete years",
                    source_variables=(extract.cloud_name,),
                    derived_from=(extract.cloud_name,),
                )
            )
        monthly = _monthly_rows(
            extract.times,
            t2m_c=t2m_c,
            wind=wind,
            rh=rh,
            precip_mm=precip_mm,
            snow_fraction=snow_fraction,
            cloud=extract.cloud_cover,
        )
        items = [
            self._one_observation(
                request.model_copy(update={"indicator_id": indicator_id}),
                extract=extract,
                window=window,
                retrieved_at=retrieved_at,
                source_url=source_url,
                years=years,
                t2m_c=t2m_c,
                wind=wind,
                rh=rh,
                precip_mm=precip_mm,
                precip_reason=precip_reason,
                snow_fraction=snow_fraction,
                snow_proxy=snow_proxy,
                derivations=derivations,
            )
            for indicator_id in (request.indicator_id,)
        ]
        return tuple(items), monthly, tuple(derivations)

    def _precipitation_mm(
        self, request: FetchRequest, extract: _PointExtract
    ) -> tuple[FloatArray | None, ERA5Derivation | None, str | None]:
        if extract.tp_m is None or extract.tp_name is None:
            return None, None, "ERA5-Land total_precipitation (tp) is not in the subset"
        convention = self._precip_convention(request)
        if convention is None:
            return (
                None,
                None,
                (
                    "precipitation_convention is required "
                    "(timestep_increment, forecast_accumulated, or monthly_mean_daily); "
                    "ERA5-Land tp units are metres and the accumulation rule is not guessed"
                ),
            )
        increments = extract.tp_m
        if convention is PrecipitationConvention.FORECAST_ACCUMULATED:
            spacing = _median_hours(extract.times)
            if spacing is None or spacing > 12.0:
                return (
                    None,
                    None,
                    "forecast_accumulated applies to hourly ERA5-Land accumulations",
                )
            increments = deaccumulate_forecast_hourly(extract.times, extract.tp_m)
        if convention is PrecipitationConvention.MONTHLY_MEAN_DAILY:
            days = _naive_utc_index(extract.times).days_in_month.to_numpy(dtype=np.float64)
            precip_mm = increments * 1000.0 * days
            formula = "P_mm_month = tp_m_per_day * 1000 * days_in_month (ERA5-Land moda)"
        else:
            precip_mm = increments * 1000.0
            formula = "P_mm = tp_m_increment * 1000"
        derivation = ERA5Derivation(
            method_id=CALC_PRECIP,
            formula=f"{convention.value}; {formula}",
            source_variables=(extract.tp_name,),
            derived_from=(extract.tp_name,),
        )
        return precip_mm, derivation, None

    def _precip_convention(self, request: FetchRequest) -> PrecipitationConvention | None:
        raw = _text_param(request.parameters, "precipitation_convention")
        if raw:
            return PrecipitationConvention(raw)
        return self._settings.precipitation_convention

    def _snow_fraction(
        self, extract: _PointExtract
    ) -> tuple[FloatArray | None, ERA5Derivation | None, bool]:
        if extract.snow_cover is not None and extract.snow_cover_name:
            fraction = snow_cover_to_fraction(extract.snow_cover, units=extract.snow_cover_units)
            derivation = ERA5Derivation(
                method_id=CALC_SNOW_COVER,
                formula="snow_risk_proxy = mean(snow_cover_percent / 100)",
                source_variables=(extract.snow_cover_name,),
                derived_from=(extract.snow_cover_name,),
            )
            return fraction, derivation, True
        if extract.snow_depth is not None and extract.snow_depth_name:
            presence = (np.asarray(extract.snow_depth, dtype=np.float64) > 0.0).astype(np.float64)
            presence[~np.isfinite(extract.snow_depth)] = np.nan
            derivation = ERA5Derivation(
                method_id=CALC_SNOW_DEPTH,
                formula="snow_risk_proxy = mean(snow_depth > 0); not a snow-load model",
                source_variables=(extract.snow_depth_name,),
                derived_from=(extract.snow_depth_name,),
            )
            return presence, derivation, True
        return None, None, False

    def _one_observation(
        self,
        request: FetchRequest,
        *,
        extract: _PointExtract,
        window: AnalysisWindow,
        retrieved_at: datetime,
        source_url: str,
        years: tuple[int, ...],
        t2m_c: FloatArray | None,
        wind: FloatArray | None,
        rh: FloatArray | None,
        precip_mm: FloatArray | None,
        precip_reason: str | None,
        snow_fraction: FloatArray | None,
        snow_proxy: bool,
        derivations: Sequence[ERA5Derivation],
    ) -> IndicatorObservation:
        spatial = SpatialCoverage(
            resolution=ERA5_LAND_RESOLUTION,
            crs="EPSG:4326",
            geometry_fingerprint=(
                None
                if extract.grid_latitude is None or extract.grid_longitude is None
                else f"lat={extract.grid_latitude},lon={extract.grid_longitude}"
            ),
        )
        temporal = TemporalCoverage(
            resolution="complete-calendar-year",
            start=window.start,
            end=window.end,
            aggregation=window.kind,
        )
        dataset = DatasetRef(
            name=ERA5_LAND_SOURCE_NAME,
            media_type="application/netcdf",
            uri=source_url,
            notes=(
                f"dataset_id={self._settings.dataset_id}",
                "Muñoz-Sabater et al. 2021, ESSD",
            ),
        )
        indicator = request.indicator_id
        mask = _year_mask(extract.times, years)
        no_year_reason = "no complete calendar year of ERA5-Land values in the analysis window"
        if indicator == IndicatorId.CLIMATE_CLOUD_COVER.value and extract.cloud_cover is None:
            return self._miss(
                request,
                status=ObservationStatus.UNAVAILABLE,
                reason=(
                    "ERA5-Land does not include cloud cover; the field is mapped only "
                    "when the opened dataset actually contains total_cloud_cover/tcc"
                ),
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
            )
        if not years:
            return self._miss(
                request,
                status=ObservationStatus.MISSING_COVERAGE,
                reason=no_year_reason,
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
            )

        def derivation_for(method_id: str) -> ERA5Derivation | None:
            for item in derivations:
                if item.method_id == method_id:
                    return item
            return None

        if indicator == IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN.value:
            return self._numeric(
                request,
                value=_finite_mean(t2m_c[mask]) if t2m_c is not None else None,
                unit="degC",
                origin=ValueOrigin.DERIVED,
                kind=ValueProvenanceKind.DERIVED_METHOD,
                method=CALC_K_TO_C,
                derivation=derivation_for(CALC_K_TO_C),
                missing="ERA5-Land 2m_temperature (t2m) is not in the subset",
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
                extra=(window.label,),
            )
        if indicator == IndicatorId.CLIMATE_AIR_TEMPERATURE_MIN.value:
            return self._numeric(
                request,
                value=_finite_min(t2m_c[mask]) if t2m_c is not None else None,
                unit="degC",
                origin=ValueOrigin.DERIVED,
                kind=ValueProvenanceKind.DERIVED_METHOD,
                method=CALC_K_TO_C,
                derivation=derivation_for(CALC_K_TO_C),
                missing="ERA5-Land 2m_temperature (t2m) is not in the subset",
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
                extra=(window.label, "minimum of converted 2 m air temperature"),
            )
        if indicator == IndicatorId.CLIMATE_AIR_TEMPERATURE_MAX.value:
            return self._numeric(
                request,
                value=_finite_max(t2m_c[mask]) if t2m_c is not None else None,
                unit="degC",
                origin=ValueOrigin.DERIVED,
                kind=ValueProvenanceKind.DERIVED_METHOD,
                method=CALC_K_TO_C,
                derivation=derivation_for(CALC_K_TO_C),
                missing="ERA5-Land 2m_temperature (t2m) is not in the subset",
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
                extra=(window.label, "maximum of converted 2 m air temperature"),
            )
        if indicator == IndicatorId.CLIMATE_WIND_SPEED_MEAN.value:
            return self._numeric(
                request,
                value=_finite_mean(wind[mask]) if wind is not None else None,
                unit="m/s",
                origin=ValueOrigin.DERIVED,
                kind=ValueProvenanceKind.DERIVED_METHOD,
                method=CALC_WIND,
                derivation=derivation_for(CALC_WIND),
                missing="ERA5-Land 10 m wind components u10 and v10 are both required",
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
                extra=(window.label, "ERA5-Land wind is at 10 m"),
            )
        if indicator == IndicatorId.CLIMATE_RELATIVE_HUMIDITY.value:
            return self._numeric(
                request,
                value=_finite_mean(rh[mask]) if rh is not None else None,
                unit="%",
                origin=ValueOrigin.DERIVED,
                kind=ValueProvenanceKind.DERIVED_METHOD,
                method=CALC_RH,
                derivation=derivation_for(CALC_RH),
                missing="relative humidity requires both t2m and d2m",
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
                extra=(
                    window.label,
                    "mean of hourly Magnus RH, not RH of mean temperature",
                ),
            )
        if indicator == IndicatorId.CLIMATE_PRECIPITATION.value:
            annual = None
            if precip_mm is not None:
                annual = self._mean_annual_sum(extract.times, precip_mm, years)
            return self._numeric(
                request,
                value=annual,
                unit="mm/year",
                origin=ValueOrigin.DERIVED,
                kind=ValueProvenanceKind.DERIVED_METHOD,
                method=CALC_PRECIP,
                derivation=derivation_for(CALC_PRECIP),
                missing=precip_reason or "ERA5-Land precipitation could not be converted",
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
                extra=(window.label,),
            )
        if indicator == IndicatorId.CLIMATE_SNOW_RISK.value:
            flags = (QualityFlag.PROXY.value,) if snow_proxy else ()
            return self._numeric(
                request,
                value=_finite_mean(snow_fraction[mask]) if snow_fraction is not None else None,
                unit="1",
                origin=ValueOrigin.DERIVED,
                kind=ValueProvenanceKind.DERIVED_PROXY,
                method=CALC_SNOW_COVER if extract.snow_cover is not None else CALC_SNOW_DEPTH,
                derivation=derivation_for(
                    CALC_SNOW_COVER if extract.snow_cover is not None else CALC_SNOW_DEPTH
                ),
                missing=(
                    "ERA5-Land snow_cover and snow_depth are absent; no snow-load model is applied"
                ),
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
                extra=(
                    window.label,
                    "snow-cover or snow-presence fraction is a proxy, "
                    "not a structural snow-load index",
                ),
                quality_flags=flags,
            )
        if indicator == IndicatorId.CLIMATE_CLOUD_COVER.value:
            return self._numeric(
                request,
                value=_finite_mean(extract.cloud_cover[mask])
                if extract.cloud_cover is not None
                else None,
                unit="1",
                origin=ValueOrigin.DERIVED,
                kind=ValueProvenanceKind.DERIVED_METHOD,
                method=CALC_CLOUD,
                derivation=derivation_for(CALC_CLOUD),
                missing="cloud cover is not an ERA5-Land variable",
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
                extra=(window.label,),
            )
        if indicator == IndicatorId.CLIMATE_EXTREME_HEAT_DAYS.value:
            threshold = self._heat_threshold_for(request)
            assert threshold is not None
            value = self._heat_days_per_year(extract.times, t2m_c, years, threshold)
            heat_derivation = ERA5Derivation(
                method_id=CALC_HEAT,
                formula=(
                    "days/year with daily max 2 m air temperature >= "
                    f"{threshold} degC; threshold is caller-supplied"
                ),
                source_variables=(extract.t2m_name,) if extract.t2m_name else (),
                derived_from=(extract.t2m_name,) if extract.t2m_name else (),
            )
            spacing = _median_hours(extract.times)
            missing = "ERA5-Land 2m_temperature (t2m) is not in the subset"
            if t2m_c is not None and (spacing is None or spacing > 36.0):
                missing = (
                    "extreme-heat days need daily or hourly 2 m temperature; "
                    "monthly ERA5-Land values cannot count days"
                )
                value = None
            return self._numeric(
                request,
                value=value,
                unit="d/year",
                origin=ValueOrigin.DERIVED,
                kind=ValueProvenanceKind.DERIVED_METHOD,
                method=CALC_HEAT,
                derivation=heat_derivation,
                missing=missing,
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
                extra=(
                    window.label,
                    f"extreme_heat_threshold_degC={threshold}",
                ),
            )
        return self._miss(
            request,
            status=ObservationStatus.UNAVAILABLE,
            reason="indicator is not mapped from ERA5-Land",
            retrieved_at=retrieved_at,
            spatial=spatial,
            temporal=temporal,
            dataset=dataset,
        )

    def _mean_annual_sum(
        self,
        times: pd.DatetimeIndex,
        values: FloatArray,
        years: Sequence[int],
    ) -> float | None:
        if not years:
            return None
        yearly: list[float] = []
        index = _naive_utc_index(times)
        for year in years:
            chunk = values[index.year == year]
            finite = chunk[np.isfinite(chunk)]
            if finite.size == 0:
                continue
            yearly.append(float(np.sum(finite)))
        if not yearly:
            return None
        return float(sum(yearly) / len(yearly))

    def _heat_days_per_year(
        self,
        times: pd.DatetimeIndex,
        t2m_c: FloatArray | None,
        years: Sequence[int],
        threshold: float,
    ) -> float | None:
        if t2m_c is None or not years:
            return None
        spacing = _median_hours(times)
        if spacing is None or spacing > 36.0:
            return None
        frame = pd.DataFrame(
            {"t": t2m_c, "time": _naive_utc_index(times)},
        ).dropna()
        if frame.empty:
            return None
        daily = frame.set_index("time")["t"].resample("1D").max()
        daily_index = pd.DatetimeIndex(daily.index)
        counts: list[float] = []
        for year in years:
            series = daily[daily_index.year == year]
            if series.empty:
                continue
            counts.append(float(np.sum(series.to_numpy() >= threshold)))
        if not counts:
            return None
        return float(sum(counts) / len(counts))

    def _numeric(
        self,
        request: FetchRequest,
        *,
        value: float | None,
        unit: str,
        origin: ValueOrigin,
        kind: ValueProvenanceKind,
        method: str | None,
        derivation: ERA5Derivation | None,
        missing: str,
        retrieved_at: datetime,
        spatial: SpatialCoverage,
        temporal: TemporalCoverage,
        dataset: DatasetRef,
        extra: tuple[str, ...] = (),
        quality_flags: tuple[str, ...] = (),
    ) -> IndicatorObservation:
        if value is None:
            return self._miss(
                request,
                status=ObservationStatus.UNAVAILABLE,
                reason=missing,
                retrieved_at=retrieved_at,
                spatial=spatial,
                temporal=temporal,
                dataset=dataset,
            )
        assumptions = list(extra)
        if derivation is not None:
            assumptions.extend(
                (
                    f"derived_from={','.join(derivation.derived_from)}",
                    f"formula={derivation.formula}",
                    f"source_variables={','.join(derivation.source_variables)}",
                )
            )
        flags = quality_flags or (QualityFlag.NOMINAL.value,)
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=self.source_name_default,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=kind,
                value_origin=origin,
                source_version=self._settings.dataset_id,
                calculation_method=method if origin is ValueOrigin.DERIVED else None,
                dataset=dataset,
            ),
            spatial=spatial,
            temporal=temporal,
            value=value,
            unit=unit,
            quality_flags=flags,
            assumptions=tuple(assumptions),
        )

    def _miss(
        self,
        request: FetchRequest,
        *,
        status: ObservationStatus,
        reason: str,
        retrieved_at: datetime,
        spatial: SpatialCoverage | None = None,
        temporal: TemporalCoverage | None = None,
        dataset: DatasetRef | None = None,
        grid_lat: float | None = None,
        grid_lon: float | None = None,
    ) -> IndicatorObservation:
        fingerprint = None
        if grid_lat is not None and grid_lon is not None:
            fingerprint = f"lat={grid_lat},lon={grid_lon}"
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=status,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=self.source_name_default,
                retrieved_at=retrieved_at,
                request_hash=request_cache_key(self._provider_id, request),
                value_kind=ValueProvenanceKind.MEASURED_OR_PROVIDER,
                value_origin=ValueOrigin.RAW,
                source_version=self._settings.dataset_id,
                dataset=dataset
                or DatasetRef(
                    name=ERA5_LAND_SOURCE_NAME,
                    media_type="application/netcdf",
                    uri=public_cds_dataset_url(self._settings.dataset_id),
                ),
            ),
            spatial=spatial
            or SpatialCoverage(
                resolution=ERA5_LAND_RESOLUTION,
                crs="EPSG:4326",
                geometry_fingerprint=fingerprint,
            ),
            temporal=temporal or TemporalCoverage(resolution="unknown"),
            assumptions=(reason,),
        )

    def _read_cache(self, request_hash: str) -> IndicatorObservation | None:
        if self._cache is None:
            return None
        payload = self._cache.get(request_hash)
        if payload is None:
            return None
        observation = _decode_observation(payload)
        if observation.provenance.request_hash != request_hash:
            raise CacheError("cached observation request_hash does not match the request")
        if observation.provenance.provider_id != self._provider_id:
            raise CacheError("cached observation belongs to a different provider")
        return observation

    def _write_cache(self, request_hash: str, observation: IndicatorObservation) -> None:
        if self._cache is None:
            return
        self._cache.put(request_hash, _encode_observation(observation))


__all__ = [
    "CALC_CLOUD",
    "CALC_HEAT",
    "CALC_K_TO_C",
    "CALC_PRECIP",
    "CALC_RH",
    "CALC_SNOW_COVER",
    "CALC_SNOW_DEPTH",
    "CALC_WIND",
    "CDSAPI_KEY_ENV",
    "CDSAPI_URL_ENV",
    "CDS_RETRIEVE_VARIABLES",
    "DEFAULT_CDS_API_URL",
    "ERA5_CDS_KEY_ENV",
    "ERA5_CDS_URL_ENV",
    "ERA5_EXTRA_MODULES",
    "ERA5_LAND_CATALOGUE_URL",
    "ERA5_LAND_DATASET_ID",
    "ERA5_LAND_PATH_ENV",
    "ERA5_PRECIP_CONV_ENV",
    "AnalysisWindow",
    "ERA5Derivation",
    "ERA5LandRawMetadata",
    "ERA5LandResult",
    "ERA5LandSettings",
    "ERA5LandWeatherProvider",
    "ERA5MonthlyClimate",
    "PrecipitationConvention",
    "build_cds_retrieve_request",
    "complete_calendar_years",
    "deaccumulate_forecast_hourly",
    "era5_extra_status",
    "kelvin_to_celsius",
    "magnus_relative_humidity",
    "public_cds_dataset_url",
    "resolve_analysis_window",
    "settings_from_environ",
    "snow_cover_to_fraction",
    "wind_speed_from_components",
]
