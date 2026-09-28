"""ERA5-Land weather adapter: local fixtures, units, provenance, no live CDS."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from solardss.core.arrays import FloatArray
from solardss.data.era5_land import (
    CALC_K_TO_C,
    CALC_RH,
    CALC_WIND,
    CDS_RETRIEVE_VARIABLES,
    ERA5_EXTRA_MODULES,
    ERA5LandWeatherProvider,
    PrecipitationConvention,
    build_cds_retrieve_request,
    complete_calendar_years,
    deaccumulate_forecast_hourly,
    era5_extra_status,
    kelvin_to_celsius,
    magnus_relative_humidity,
    public_cds_dataset_url,
    settings_from_environ,
    snow_cover_to_fraction,
    wind_speed_from_components,
)
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.indicators.ids import IndicatorId
from solardss.provenance.observation import ObservationStatus, QualityFlag, ValueOrigin
from solardss.providers.cache import MemoryCache
from solardss.providers.clock import FixedClock, SequenceClock
from solardss.providers.contracts import AvailabilityStatus, HealthStatus
from solardss.providers.protocols import WeatherProvider
from solardss.providers.request import FetchRequest

T1 = datetime(2022, 3, 1, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2022, 3, 2, 12, 0, tzinfo=timezone.utc)
SITE = GeographicRef(name="Berlin", longitude=13.4, latitude=52.5)
LAT = 52.5
LON = 13.4


def _monthly_times(year: int = 2020) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", periods=12, freq="MS")


def _daily_times(year: int = 2020) -> pd.DatetimeIndex:
    return pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")


def _dataset(
    *,
    times: pd.DatetimeIndex | None = None,
    t2m: FloatArray | float | None = 283.15,
    d2m: FloatArray | float | None = 278.15,
    u10: FloatArray | float | None = 3.0,
    v10: FloatArray | float | None = 4.0,
    tp: FloatArray | float | None = 0.002,
    snowc: FloatArray | float | None = 20.0,
    sde: FloatArray | float | None = None,
    tcc: FloatArray | float | None = None,
    latitude: float = LAT,
    longitude: float = LON,
) -> xr.Dataset:
    index = _monthly_times() if times is None else times
    shape = (len(index), 1, 1)

    def _field(value: FloatArray | float | None) -> FloatArray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return np.asarray(value, dtype=np.float64).reshape(shape)
        return np.full(shape, float(value), dtype=np.float64)

    data: dict[str, tuple[tuple[str, ...], FloatArray]] = {}
    units: dict[str, str] = {}
    mapping = {
        "t2m": (t2m, "K"),
        "d2m": (d2m, "K"),
        "u10": (u10, "m s**-1"),
        "v10": (v10, "m s**-1"),
        "tp": (tp, "m"),
        "snowc": (snowc, "%"),
        "sde": (sde, "m"),
        "tcc": (tcc, "(0 - 1)"),
    }
    for name, (value, unit) in mapping.items():
        array = _field(value)
        if array is None:
            continue
        data[name] = (("time", "latitude", "longitude"), array)
        units[name] = unit
    dataset = xr.Dataset(
        data,
        coords={
            "time": index.to_numpy(),
            "latitude": np.array([latitude], dtype=np.float64),
            "longitude": np.array([longitude], dtype=np.float64),
        },
    )
    for name, unit in units.items():
        dataset[name].attrs["units"] = unit
    return dataset


def _provider(dataset: xr.Dataset | None = None, **kwargs: object) -> ERA5LandWeatherProvider:
    params: dict[str, object] = {
        "dataset": dataset if dataset is not None else _dataset(),
        "clock": FixedClock(T1),
        "cache": MemoryCache(),
        "precipitation_convention": PrecipitationConvention.MONTHLY_MEAN_DAILY,
        "environ": {},
    }
    params.update(kwargs)
    return ERA5LandWeatherProvider(**params)  # type: ignore[arg-type]


def _request(indicator: IndicatorId, **kwargs: object) -> FetchRequest:
    parameters = {"start_year": 2020, "end_year": 2020, "window": "explicit"}
    extra = kwargs.pop("parameters", None)
    if isinstance(extra, dict):
        parameters.update(extra)
    payload: dict[str, object] = {
        "indicator_id": indicator.value,
        "location": SITE,
        "parameters": parameters,
    }
    payload.update(kwargs)
    return FetchRequest.model_validate(payload)


def test_era5_extra_status_keys() -> None:
    status = era5_extra_status()
    assert tuple(status) == ERA5_EXTRA_MODULES
    assert status["xarray"] is True


def test_physics_kelvin_wind_rh_snow() -> None:
    assert kelvin_to_celsius(np.array([273.15, 283.15])).tolist() == [0.0, 10.0]
    assert wind_speed_from_components(np.array([3.0]), np.array([4.0]))[0] == pytest.approx(5.0)
    rh = magnus_relative_humidity(np.array([20.0]), np.array([10.0]))
    assert rh[0] == pytest.approx(52.541, abs=0.02)
    assert snow_cover_to_fraction(np.array([20.0]), units="%")[0] == pytest.approx(0.2)


def test_deaccumulate_forecast_hourly() -> None:
    times = pd.DatetimeIndex(
        [
            "2020-01-01 01:00",
            "2020-01-01 02:00",
            "2020-01-01 03:00",
            "2020-01-01 23:00",
            "2020-01-02 00:00",
        ]
    )
    acc = np.array([0.1, 0.3, 0.4, 0.9, 1.0])
    increments = deaccumulate_forecast_hourly(times, acc)
    assert increments[:3].tolist() == pytest.approx([0.1, 0.2, 0.1])
    assert np.isnan(increments[3])
    assert increments[4] == pytest.approx(0.1)


def test_complete_years_omit_partial() -> None:
    times = pd.date_range("2020-01-01", periods=6, freq="MS")
    assert complete_calendar_years(pd.DatetimeIndex(times)) == ()
    assert complete_calendar_years(_monthly_times(2020)) == (2020,)


def test_missing_credentials_are_not_configured() -> None:
    provider = ERA5LandWeatherProvider(environ={}, clock=FixedClock(T1))
    assert provider.health().status is HealthStatus.NOT_CONFIGURED
    assert "has_cds_key" in provider.health().details
    assert provider.health().details["has_cds_key"] is False
    observation = provider.fetch(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN))
    assert observation.status is ObservationStatus.NOT_CONFIGURED
    assert observation.value is None


def test_secret_key_never_enters_provenance_or_repr() -> None:
    secret = "era5-live-token-do-not-leak"
    settings = settings_from_environ({"CDSAPI_KEY": secret})
    assert secret not in repr(settings)
    provider = _provider(
        cds_key=secret,
        dataset=_dataset(),
    )
    observation = provider.fetch(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN))
    blob = json.dumps(observation.model_dump(mode="json"))
    assert secret not in blob
    health_blob = json.dumps(provider.health().model_dump(mode="json"))
    assert secret not in health_blob
    assert provider.health().details["has_cds_key"] is True


def test_protocol_and_source_url() -> None:
    provider = _provider()
    assert isinstance(provider, WeatherProvider)
    assert public_cds_dataset_url().startswith("https://cds.climate.copernicus.eu/datasets/")
    observation = provider.fetch(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN))
    assert observation.provenance.dataset is not None
    assert observation.provenance.dataset.uri == public_cds_dataset_url()
    assert "user" not in (observation.provenance.dataset.uri or "")


def test_mean_min_max_temperature_units_and_latlon() -> None:
    t2m = np.array([273.15 + month for month in range(1, 13)], dtype=np.float64)
    provider = _provider(_dataset(t2m=t2m))
    mean = provider.fetch(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN))
    minimum = provider.fetch(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MIN))
    maximum = provider.fetch(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MAX))
    assert mean.status is ObservationStatus.OBSERVED
    assert mean.unit == "degC"
    assert mean.value == pytest.approx(6.5)
    assert minimum.value == pytest.approx(1.0)
    assert maximum.value == pytest.approx(12.0)
    assert mean.provenance.calculation_method == CALC_K_TO_C
    assert mean.provenance.value_origin is ValueOrigin.DERIVED
    assert any("derived_from=" in item for item in mean.assumptions)
    assert mean.spatial.geometry_fingerprint == f"lat={LAT},lon={LON}"
    bundle = provider.fetch_climate(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN))
    assert len(bundle.monthly) == 12
    assert bundle.monthly[0].t2m_mean_deg_c == pytest.approx(1.0)
    assert bundle.raw.grid_latitude == pytest.approx(LAT)
    assert bundle.raw.grid_longitude == pytest.approx(LON)


def test_wind_and_humidity_derivations() -> None:
    provider = _provider(_dataset(t2m=293.15, d2m=283.15, u10=3.0, v10=4.0))
    wind = provider.fetch(_request(IndicatorId.CLIMATE_WIND_SPEED_MEAN))
    humidity = provider.fetch(_request(IndicatorId.CLIMATE_RELATIVE_HUMIDITY))
    assert wind.status is ObservationStatus.OBSERVED
    assert wind.unit == "m/s"
    assert wind.value == pytest.approx(5.0)
    assert wind.provenance.calculation_method == CALC_WIND
    assert humidity.unit == "%"
    assert humidity.value == pytest.approx(52.541, abs=0.02)
    assert humidity.provenance.calculation_method == CALC_RH
    assert any("t2m" in item and "d2m" in item for item in humidity.assumptions)


def test_precipitation_monthly_mean_daily() -> None:
    provider = _provider(_dataset(tp=0.002))
    observation = provider.fetch(_request(IndicatorId.CLIMATE_PRECIPITATION))
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.unit == "mm/year"
    assert observation.value == pytest.approx(732.0)
    bundle = provider.fetch_climate(_request(IndicatorId.CLIMATE_PRECIPITATION))
    assert bundle.monthly[0].precipitation_mm == pytest.approx(62.0)


def test_precipitation_without_convention_is_unavailable() -> None:
    provider = _provider(
        _dataset(),
        precipitation_convention=None,
    )
    observation = provider.fetch(_request(IndicatorId.CLIMATE_PRECIPITATION))
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.value is None
    assert "precipitation_convention" in observation.assumptions[0]


def test_snow_cover_is_a_proxy() -> None:
    provider = _provider(_dataset(snowc=20.0))
    observation = provider.fetch(_request(IndicatorId.CLIMATE_SNOW_RISK))
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.unit == "1"
    assert observation.value == pytest.approx(0.2)
    assert QualityFlag.PROXY.value in observation.quality_flags


def test_cloud_cover_not_in_era5_land() -> None:
    provider = _provider(_dataset(tcc=None))
    observation = provider.fetch(_request(IndicatorId.CLIMATE_CLOUD_COVER))
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.value is None
    assert "does not include cloud cover" in observation.assumptions[0]


def test_cloud_cover_mapped_only_when_present() -> None:
    provider = _provider(_dataset(tcc=0.4))
    observation = provider.fetch(_request(IndicatorId.CLIMATE_CLOUD_COVER))
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.value == pytest.approx(0.4)
    assert observation.unit == "1"


def test_soiling_is_not_served() -> None:
    provider = _provider()
    availability = provider.availability(_request(IndicatorId.CLIMATE_SOILING_RISK))
    assert availability.status is AvailabilityStatus.UNAVAILABLE
    observation = provider.fetch(_request(IndicatorId.CLIMATE_SOILING_RISK))
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.value is None


def test_extreme_heat_requires_caller_threshold() -> None:
    provider = _provider()
    availability = provider.availability(_request(IndicatorId.CLIMATE_EXTREME_HEAT_DAYS))
    assert availability.status is AvailabilityStatus.UNAVAILABLE
    assert "threshold" in availability.reason
    monthly = provider.fetch(_request(IndicatorId.CLIMATE_EXTREME_HEAT_DAYS))
    assert monthly.status is ObservationStatus.UNAVAILABLE


def test_extreme_heat_days_from_daily_max() -> None:
    times = _daily_times(2020)
    t2m = np.full(len(times), 293.15)
    t2m[:5] = 308.15
    provider = _provider(_dataset(times=times, t2m=t2m, tp=None, snowc=None))
    observation = provider.fetch(
        _request(
            IndicatorId.CLIMATE_EXTREME_HEAT_DAYS,
            parameters={
                "start_year": 2020,
                "end_year": 2020,
                "window": "explicit",
                "extreme_heat_threshold_degC": 30.0,
            },
        )
    )
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.unit == "d/year"
    assert observation.value == pytest.approx(5.0)
    assert "extreme_heat_threshold_degC=30.0" in observation.assumptions


def test_incomplete_year_is_omitted() -> None:
    times = pd.date_range("2020-01-01", periods=6, freq="MS")
    provider = _provider(_dataset(times=times))
    observation = provider.fetch(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN))
    assert observation.status is ObservationStatus.MISSING_COVERAGE
    assert observation.value is None


def test_missing_dewpoint_does_not_default_humidity() -> None:
    provider = _provider(_dataset(d2m=None))
    observation = provider.fetch(_request(IndicatorId.CLIMATE_RELATIVE_HUMIDITY))
    assert observation.status is ObservationStatus.UNAVAILABLE
    assert observation.value is None


def test_ocean_masked_cell_is_unsupported_region() -> None:
    masked = _dataset(t2m=np.nan, d2m=np.nan, u10=np.nan, v10=np.nan, tp=np.nan, snowc=np.nan)
    provider = _provider(masked)
    observation = provider.fetch(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN))
    assert observation.status is ObservationStatus.UNSUPPORTED_REGION
    assert observation.value is None


def test_cache_is_deterministic() -> None:
    cache = MemoryCache()
    dataset = _dataset()
    first = _provider(dataset, cache=cache, clock=FixedClock(T1))
    request = _request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN)
    observed = first.fetch(request)
    second = _provider(dataset, cache=cache, clock=SequenceClock((T2,)))
    cached = second.fetch(request)
    assert cached.status is ObservationStatus.OBSERVED
    assert cached.value == observed.value
    assert cached.provenance.retrieved_at == T1


def test_cds_request_is_spatially_and_temporally_subset() -> None:
    window = _provider().coverage(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN)).temporal
    assert window is not None
    from solardss.data.era5_land import AnalysisWindow

    payload = build_cds_retrieve_request(
        latitude=LAT,
        longitude=LON,
        window=AnalysisWindow(
            kind="explicit",
            start=window.start or datetime(2020, 1, 1).date(),
            end=window.end or datetime(2020, 12, 31).date(),
            label="test",
        ),
        pad_deg=0.2,
    )
    area = payload["area"]
    assert isinstance(area, list) and len(area) == 4
    coords = [float(item) for item in area if isinstance(item, int | float)]
    assert len(coords) == 4
    north, west, south, east = coords
    assert north - south <= 0.4000001
    assert east - west <= 0.4000001
    variables = payload["variable"]
    assert isinstance(variables, list)
    assert "total_cloud_cover" not in variables
    assert tuple(variables) == CDS_RETRIEVE_VARIABLES
    assert payload["year"] == ["2020"]


def test_mocked_cds_retrieve_writes_local_subset(tmp_path: Path) -> None:
    dataset = _dataset()
    captured: dict[str, object] = {}

    def retrieve(_dataset_id: str, request: object, target: Path) -> None:
        captured["request"] = request
        dataset.to_netcdf(target)

    provider = ERA5LandWeatherProvider(
        environ={"CDSAPI_KEY": "fixture-key"},
        cds_retrieve=retrieve,
        clock=FixedClock(T1),
        precipitation_convention=PrecipitationConvention.MONTHLY_MEAN_DAILY,
    )
    observation = provider.fetch(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN))
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.unit == "degC"
    body = captured["request"]
    assert isinstance(body, dict)
    assert "area" in body


def test_local_netcdf_path(tmp_path: Path) -> None:
    path = tmp_path / "era5-land-fixture.nc"
    _dataset().to_netcdf(path)
    provider = ERA5LandWeatherProvider(
        environ={},
        dataset_path=path,
        clock=FixedClock(T1),
        precipitation_convention=PrecipitationConvention.MONTHLY_MEAN_DAILY,
    )
    observation = provider.fetch(_request(IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN))
    assert observation.status is ObservationStatus.OBSERVED
    assert provider.health().status is HealthStatus.OK


def test_bbox_spatial_mean() -> None:
    dataset = _dataset()
    provider = _provider(dataset)
    request = FetchRequest(
        indicator_id=IndicatorId.CLIMATE_WIND_SPEED_MEAN.value,
        bbox=BoundingBox(min_lon=13.0, min_lat=52.0, max_lon=14.0, max_lat=53.0),
        parameters={"start_year": 2020, "end_year": 2020, "window": "explicit"},
    )
    observation = provider.fetch(request)
    assert observation.status is ObservationStatus.OBSERVED
    assert observation.value == pytest.approx(5.0)


@pytest.mark.era5_live
def test_live_cds_optional() -> None:
    if os.environ.get("SOLARDSS_ERA5_LIVE") != "1":
        pytest.skip("set SOLARDSS_ERA5_LIVE=1 to hit Copernicus CDS")
    provider = ERA5LandWeatherProvider()
    if provider.health().status is HealthStatus.NOT_CONFIGURED:
        pytest.skip("no ERA5-Land path or CDS credentials")
    observation = provider.fetch(
        FetchRequest(
            indicator_id=IndicatorId.CLIMATE_AIR_TEMPERATURE_MEAN.value,
            location=SITE,
            parameters={"start_year": 2020, "end_year": 2020, "window": "explicit"},
        )
    )
    assert observation.status in {
        ObservationStatus.OBSERVED,
        ObservationStatus.MISSING_COVERAGE,
        ObservationStatus.UNSUPPORTED_REGION,
        ObservationStatus.TEMPORARY_FAILURE,
    }
