"""Official PVGIS adapter: fixtures, units, provenance, no live HTTP by default."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from solardss.data.geo import GeographicRef
from solardss.data.pvgis import (
    PVGISSolarResourceProvider,
    PVGISYieldProvider,
    mean_complete_year_sums,
    public_source_url,
)
from solardss.indicators.ids import IndicatorId
from solardss.provenance.observation import ObservationStatus, ValueOrigin
from solardss.providers.cache import MemoryCache
from solardss.providers.clock import FixedClock, SequenceClock
from solardss.providers.request import FetchRequest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "pvgis"
T1 = datetime(2022, 3, 1, 12, 0, tzinfo=timezone.utc)
T2 = datetime(2022, 3, 2, 12, 0, tzinfo=timezone.utc)
SITE = GeographicRef(name="PVGIS sample", longitude=8.0, latitude=45.0)


def _load(name: str) -> dict[str, object]:
    raw: object = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _ghi_request(**kwargs: object) -> FetchRequest:
    payload: dict[str, object] = {
        "indicator_id": IndicatorId.SOLAR_GHI_ANNUAL.value,
        "location": SITE,
        "parameters": {"startyear": 2020, "endyear": 2020},
    }
    payload.update(kwargs)
    return FetchRequest.model_validate(payload)


def _transport(
    *,
    payload: dict[str, object] | None = None,
    status_code: int = 200,
    error: dict[str, object] | None = None,
    calls: list[httpx.Request] | None = None,
    sequence: list[httpx.Response] | None = None,
) -> httpx.MockTransport:
    queued = list(sequence or [])

    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if queued:
            return queued.pop(0)
        if status_code != 200:
            body = error or {"message": "error", "status": status_code}
            return httpx.Response(status_code, json=body)
        assert payload is not None
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def _resource(**kwargs: object) -> PVGISSolarResourceProvider:
    payload = _load("mrcalc.json")
    params: dict[str, object] = {
        "transport": _transport(payload=payload),
        "clock": FixedClock(T1),
        "base_url": "https://pvgis.test/api/v5_3",
        "sleep": lambda _seconds: None,
        "cache": MemoryCache(),
    }
    params.update(kwargs)
    return PVGISSolarResourceProvider(**params)  # type: ignore[arg-type]


def _yield(**kwargs: object) -> PVGISYieldProvider:
    payload = _load("pvcalc.json")
    params: dict[str, object] = {
        "transport": _transport(payload=payload),
        "clock": FixedClock(T1),
        "base_url": "https://pvgis.test/api/v5_3",
        "sleep": lambda _seconds: None,
        "cache": MemoryCache(),
        "peak_power_kw": 1.0,
        "system_loss_percent": 14.0,
    }
    params.update(kwargs)
    return PVGISYieldProvider(**params)  # type: ignore[arg-type]


def test_annual_ghi_dni_dhi_from_mrcalc_fixture() -> None:
    provider = _resource()
    try:
        bundle = provider.fetch_resource(_ghi_request())
        ghi = next(
            item
            for item in bundle.observations
            if item.indicator_id == IndicatorId.SOLAR_GHI_ANNUAL.value
        )
        dni = next(
            item
            for item in bundle.observations
            if item.indicator_id == IndicatorId.SOLAR_DNI_ANNUAL.value
        )
        dhi = next(
            item
            for item in bundle.observations
            if item.indicator_id == IndicatorId.SOLAR_DHI_ANNUAL.value
        )
        rows = bundle.monthly
        assert len(rows) == 12
        assert rows[0].month == 1
        assert rows[0].ghi_kwh_m2 == 62.46
        assert rows[0].dni_kwh_m2 == 131.44
        assert rows[0].kd == 0.32
        assert rows[0].t2m_deg_c == 4.2
        assert rows[0].dhi_kwh_m2 == pytest.approx(62.46 * 0.32)
        assert ghi.status is ObservationStatus.OBSERVED
        assert ghi.unit == "kWh/m2/year"
        assert ghi.value == pytest.approx(sum(row.ghi_kwh_m2 or 0.0 for row in rows))
        assert dni.value == pytest.approx(sum(row.dni_kwh_m2 or 0.0 for row in rows))
        assert dhi.value == pytest.approx(sum(row.dhi_kwh_m2 or 0.0 for row in rows))
        assert mean_complete_year_sums(rows, "ghi_kwh_m2") == ghi.value
    finally:
        provider.close()


def test_lat_lon_and_provenance() -> None:
    provider = _resource()
    try:
        observation = provider.fetch(_ghi_request())
        assert observation.provenance.provider_id == "pvgis.solar_resource"
        assert observation.provenance.source_name == "PVGIS"
        assert observation.provenance.source_version == "v5_3"
        assert observation.provenance.retrieved_at == T1
        assert observation.provenance.value_origin is ValueOrigin.DERIVED
        assert observation.provenance.calculation_method is not None
        assert observation.provenance.calculation_method.startswith("pvgis_mrcalc")
        assert observation.spatial.crs == "EPSG:4326"
        assert observation.spatial.geometry_fingerprint is not None
        assert "lat=45.0" in observation.spatial.geometry_fingerprint
        assert "lon=8.0" in observation.spatial.geometry_fingerprint
        assert observation.temporal.start is not None
        assert observation.temporal.start.year == 2020
        assert observation.provenance.dataset is not None
        assert observation.provenance.dataset.uri is not None
        assert "pvgis.test" in observation.provenance.dataset.uri
        assert "@" not in observation.provenance.dataset.uri
        assert "password" not in observation.provenance.dataset.uri
    finally:
        provider.close()


def test_same_request_same_observation() -> None:
    cache = MemoryCache()
    transport_calls: list[httpx.Request] = []
    provider = _resource(
        cache=cache,
        clock=SequenceClock((T1, T2)),
        transport=_transport(payload=_load("mrcalc.json"), calls=transport_calls),
    )
    try:
        request = _ghi_request()
        first = provider.fetch(request)
        second = provider.fetch(request)
        assert first == second
        assert first.provenance.retrieved_at == T1
        assert len(transport_calls) == 1
        assert transport_calls[0].url.params["lat"] in {"45", "45.0"}
        assert transport_calls[0].url.params["lon"] in {"8", "8.0"}
        assert transport_calls[0].url.params["outputformat"] == "json"
    finally:
        provider.close()


def test_missing_monthly_field_is_not_defaulted() -> None:
    payload = deepcopy(_load("mrcalc.json"))
    outputs = payload["outputs"]
    assert isinstance(outputs, dict)
    monthly = outputs["monthly"]
    assert isinstance(monthly, list)
    for row in monthly:
        assert isinstance(row, dict)
        row.pop("Kd")
    provider = _resource(transport=_transport(payload=payload))
    try:
        bundle = provider.fetch_resource(_ghi_request())
        assert all(row.kd is None and row.dhi_kwh_m2 is None for row in bundle.monthly)
        ghi = next(
            item
            for item in bundle.observations
            if item.indicator_id == IndicatorId.SOLAR_GHI_ANNUAL.value
        )
        dhi = next(
            item
            for item in bundle.observations
            if item.indicator_id == IndicatorId.SOLAR_DHI_ANNUAL.value
        )
        assert ghi.status is ObservationStatus.OBSERVED
        assert dhi.status is ObservationStatus.UNAVAILABLE
        assert dhi.value is None
    finally:
        provider.close()


def test_incomplete_year_is_omitted_not_filled() -> None:
    payload = deepcopy(_load("mrcalc.json"))
    outputs = payload["outputs"]
    assert isinstance(outputs, dict)
    monthly = outputs["monthly"]
    assert isinstance(monthly, list)
    monthly.pop()
    provider = _resource(transport=_transport(payload=payload))
    try:
        observation = provider.fetch(_ghi_request())
        assert observation.status is ObservationStatus.UNAVAILABLE
        assert observation.value is None
    finally:
        provider.close()


def test_location_over_sea_is_unsupported_region() -> None:
    error = _load("error_sea.json")
    provider = _resource(transport=_transport(status_code=400, error=error))
    try:
        observation = provider.fetch(
            _ghi_request(location=GeographicRef(latitude=0.0, longitude=0.0))
        )
        assert observation.status is ObservationStatus.UNSUPPORTED_REGION
        assert observation.value is None
        assert "sea" in observation.assumptions[0].lower()
    finally:
        provider.close()


def test_timeout_is_temporary_failure() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    provider = _resource(
        transport=httpx.MockTransport(handler),
        max_attempts=2,
    )
    try:
        observation = provider.fetch(_ghi_request())
        assert observation.status is ObservationStatus.TEMPORARY_FAILURE
        assert observation.value is None
    finally:
        provider.close()


def test_bounded_retry_then_success() -> None:
    payload = _load("mrcalc.json")
    calls: list[httpx.Request] = []
    sequence = [
        httpx.Response(529, json={"message": "Site is overloaded", "status": 529}),
        httpx.Response(200, json=payload),
    ]
    provider = _resource(
        transport=_transport(sequence=sequence, calls=calls),
        max_attempts=3,
    )
    try:
        observation = provider.fetch(_ghi_request())
        assert observation.status is ObservationStatus.OBSERVED
        assert len(calls) == 2
    finally:
        provider.close()


def test_monthly_variability_is_not_invented() -> None:
    provider = _resource()
    try:
        observation = provider.fetch(
            _ghi_request(indicator_id=IndicatorId.SOLAR_MONTHLY_VARIABILITY.value)
        )
        assert observation.status is ObservationStatus.UNAVAILABLE
        assert observation.value is None
    finally:
        provider.close()


def test_pvcalc_yield_and_monthly_series() -> None:
    provider = _yield()
    try:
        request = FetchRequest(
            indicator_id=IndicatorId.SOLAR_PVOUT.value,
            location=SITE,
        )
        bundle = provider.fetch_yield(request)
        pvout = next(
            item
            for item in bundle.observations
            if item.indicator_id == IndicatorId.SOLAR_PVOUT.value
        )
        specific = next(
            item
            for item in bundle.observations
            if item.indicator_id == IndicatorId.SOLAR_SPECIFIC_YIELD.value
        )
        assert pvout.status is ObservationStatus.OBSERVED
        assert pvout.unit == "kWh/kWp/year"
        assert pvout.value == pytest.approx(1125.46)
        assert specific.value == pvout.value
        assert bundle.totals is not None
        assert bundle.totals.e_y == pytest.approx(1125.46)
        assert len(bundle.monthly) == 12
        assert bundle.monthly[0].energy_kwh_month == pytest.approx(39.09)
        assert bundle.monthly[0].monthly_energy_sd_kwh == pytest.approx(5.05)
        assert bundle.location is not None
        assert bundle.location.latitude == 45.0
        assert bundle.location.longitude == 8.0
        assert "SD_m" in bundle.monthly[0].present_fields
    finally:
        provider.close()


def test_yield_requires_peakpower_and_loss() -> None:
    provider = _yield(peak_power_kw=None, system_loss_percent=None)
    try:
        observation = provider.fetch(
            FetchRequest(indicator_id=IndicatorId.SOLAR_PVOUT.value, location=SITE)
        )
        assert observation.status is ObservationStatus.UNAVAILABLE
        assert observation.value is None
        assert "peakpower" in observation.assumptions[0]
    finally:
        provider.close()


def test_bbox_without_point_is_unavailable() -> None:
    from solardss.data.geo import BoundingBox

    provider = _resource()
    try:
        observation = provider.fetch(
            FetchRequest(
                indicator_id=IndicatorId.SOLAR_GHI_ANNUAL.value,
                bbox=BoundingBox(min_lon=8.0, min_lat=45.0, max_lon=9.0, max_lat=46.0),
            )
        )
        assert observation.status is ObservationStatus.UNAVAILABLE
        assert observation.value is None
    finally:
        provider.close()


def test_public_source_url_strips_userinfo() -> None:
    assert public_source_url("https://user:secret@example.invalid/api/v5_3/MRcalc?lat=1") == (
        "https://example.invalid/api/v5_3/MRcalc?lat=1"
    )


def test_health_does_not_call_pvgis() -> None:
    calls: list[httpx.Request] = []
    provider = _resource(transport=_transport(payload=_load("mrcalc.json"), calls=calls))
    try:
        health = provider.health()
        assert health.status.value == "ok"
        assert calls == []
    finally:
        provider.close()


@pytest.mark.pvgis_live
def test_live_pvgis_mrcalc_optional() -> None:
    if os.environ.get("SOLARDSS_PVGIS_LIVE") != "1":
        pytest.skip("set SOLARDSS_PVGIS_LIVE=1 to hit the public PVGIS API")
    provider = PVGISSolarResourceProvider(
        clock=FixedClock(T1),
        timeout_seconds=60.0,
        max_attempts=2,
        sleep=lambda _seconds: None,
    )
    try:
        observation = provider.fetch(
            FetchRequest(
                indicator_id=IndicatorId.SOLAR_GHI_ANNUAL.value,
                location=SITE,
                parameters={"startyear": 2020, "endyear": 2020},
            )
        )
        assert observation.status is ObservationStatus.OBSERVED
        assert isinstance(observation.value, float)
        assert observation.value > 0
        assert observation.unit == "kWh/m2/year"
        assert observation.provenance.dataset is not None
        assert observation.provenance.dataset.uri is not None
        assert "re.jrc.ec.europa.eu" in observation.provenance.dataset.uri
    finally:
        provider.close()
