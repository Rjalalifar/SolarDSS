"""PVGIS PVcalc annual yield engine: fixtures, origins, cache, no fabricated values."""

from __future__ import annotations

import ast
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from solardss.core.methods import MethodId
from solardss.core.status import EvaluationStatus
from solardss.data.geo import GeographicRef
from solardss.data.pvgis import PVGISSolarResourceProvider, PVGISYieldProvider
from solardss.providers.cache import MemoryCache
from solardss.providers.clock import FixedClock
from solardss.pv import (
    PVGISYieldEngine,
    PVSystemConfiguration,
    PVYieldRequest,
    azimuth_north_to_pvgis_aspect,
    energy_from_capacity_factor,
    yield_request_from_configuration,
)
from solardss.pv.design.contracts import MountingType, TrackingMode
from solardss.pv.errors import PVYieldConfigError
from solardss.pv.fixtures import TEST_INVERTER, TEST_MODULE_MONO
from solardss.pv.pvgis_yield import pvgis_pvtechchoice
from solardss.pv.yield_contracts import YieldValueOrigin

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "pvgis"
SRC_PVGIS_YIELD = Path(__file__).resolve().parents[1] / "src" / "solardss" / "pv" / "pvgis_yield.py"
T1 = datetime(2022, 3, 1, 12, 0, tzinfo=timezone.utc)
SITE = GeographicRef(name="PVGIS sample", longitude=8.0, latitude=45.0)


def _load(name: str) -> dict[str, object]:
    raw: object = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _transport(
    *,
    payload: dict[str, object] | None = None,
    status_code: int = 200,
    error: dict[str, object] | None = None,
    calls: list[httpx.Request] | None = None,
) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if calls is not None:
            calls.append(request)
        if status_code != 200:
            body = error or {"message": "error", "status": status_code}
            return httpx.Response(status_code, json=body)
        assert payload is not None
        return httpx.Response(200, json=payload)

    return httpx.MockTransport(handler)


def _engine(**kwargs: object) -> PVGISYieldEngine:
    payload = _load("pvcalc.json")
    params: dict[str, object] = {
        "transport": _transport(payload=payload),
        "clock": FixedClock(T1),
        "base_url": "https://pvgis.test/api/v5_3",
        "sleep": lambda _seconds: None,
        "cache": MemoryCache(),
    }
    params.update(kwargs)
    return PVGISYieldEngine(**params)  # type: ignore[arg-type]


def _request(**kwargs: object) -> PVYieldRequest:
    payload: dict[str, object] = {
        "location": SITE,
        "installed_capacity_kw": 1.0,
        "tilt_deg": 35.0,
        "azimuth_deg": 180.0,
        "mounting": MountingType.GROUND,
        "tracking": TrackingMode.FIXED,
        "technology": "mono-c-Si",
        "system_loss_percent": 14.0,
    }
    payload.update(kwargs)
    return PVYieldRequest.model_validate(payload)


def test_azimuth_north_to_pvgis_aspect() -> None:
    assert azimuth_north_to_pvgis_aspect(180.0) == 0.0
    assert azimuth_north_to_pvgis_aspect(90.0) == -90.0
    assert azimuth_north_to_pvgis_aspect(270.0) == 90.0
    assert azimuth_north_to_pvgis_aspect(0.0) == -180.0
    assert azimuth_north_to_pvgis_aspect(360.0) == -180.0


def test_technology_mapping_does_not_invent_unknown() -> None:
    assert pvgis_pvtechchoice("mono-c-Si") == (
        "crystSi",
        "mapped module technology 'mono-c-Si' to PVGIS pvtechchoice 'crystSi'",
    )
    assert pvgis_pvtechchoice("CIGS")[0] == "CIS"
    assert pvgis_pvtechchoice("CdTe") == ("CdTe", None)
    choice, reason = pvgis_pvtechchoice("other")
    assert choice is None
    assert reason is not None
    assert "substitute" in reason


def test_annual_yield_from_pvcalc_fixture() -> None:
    engine = _engine()
    try:
        result = engine.evaluate(_request())
        assert result.status is EvaluationStatus.EVALUATED
        assert result.annual_energy_kwh is not None
        assert result.annual_energy_kwh.value == pytest.approx(1125.46)
        assert result.annual_energy_kwh.unit == "kWh"
        assert result.annual_energy_kwh.origin is YieldValueOrigin.PVGIS_PROVIDER
        assert result.annual_energy_kwh.source_field == "E_y"
        assert result.specific_yield_kwh_per_kwp is not None
        assert result.specific_yield_kwh_per_kwp.value == pytest.approx(1125.46)
        assert result.specific_yield_kwh_per_kwp.origin is YieldValueOrigin.SOLARDSS_DERIVED
        assert result.specific_yield_kwh_per_kwp.calculation_method == "E_y / peakpower"
        assert result.annual_irradiation is not None
        assert result.annual_irradiation.value == pytest.approx(1476.53)
        assert result.annual_irradiation.origin is YieldValueOrigin.PVGIS_PROVIDER
        assert result.annual_irradiation.source_field == "H(i)_y"
        assert len(result.monthly_energy_kwh) == 12
        assert result.monthly_energy_kwh[0].month == 1
        assert result.monthly_energy_kwh[0].value.value == pytest.approx(39.09)
        assert result.monthly_energy_kwh[0].value.origin is YieldValueOrigin.PVGIS_PROVIDER
        assert result.monthly_irradiation[0].value.value == pytest.approx(50.28)
        assert result.loss_assumptions.requested_system_loss_percent is not None
        assert result.loss_assumptions.requested_system_loss_percent.origin is (
            YieldValueOrigin.CALLER_SUPPLIED
        )
        assert result.loss_assumptions.pvgis_total_loss_percent is not None
        assert result.loss_assumptions.pvgis_total_loss_percent.value == pytest.approx(-23.78)
        assert result.loss_assumptions.pvgis_total_loss_percent.origin is (
            YieldValueOrigin.PVGIS_PROVIDER
        )
        assert result.provider_metadata.tool == "PVcalc"
        assert result.provider_metadata.provider_id == "pvgis.pv_yield"
        assert result.provider_metadata.api_version == "v5_3"
        assert result.provider_metadata.radiation_db == "PVGIS-SARAH3"
        assert result.provider_metadata.elevation_m == pytest.approx(250.0)
        assert result.provenance.stamp.method_id == MethodId.PVGIS_PV_YIELD.value
        assert result.provenance.parameters["pvgis_tool"] == "PVcalc"
    finally:
        engine.close()


def test_query_maps_aspect_tech_mounting_and_loss() -> None:
    calls: list[httpx.Request] = []
    engine = _engine(
        transport=_transport(payload=_load("pvcalc.json"), calls=calls),
        cache=MemoryCache(),
    )
    try:
        engine.evaluate(
            _request(
                azimuth_deg=90.0,
                mounting=MountingType.ROOFTOP,
                technology="PERC",
                system_loss_percent=10.0,
            )
        )
        assert len(calls) == 1
        request = calls[0]
        assert request.url.path.endswith("/PVcalc")
        assert "MRcalc" not in str(request.url)
        params = request.url.params
        assert float(params["aspect"]) == pytest.approx(-90.0)
        assert params["pvtechchoice"] == "crystSi"
        assert params["mountingplace"] == "building"
        assert float(params["loss"]) == pytest.approx(10.0)
        assert float(params["peakpower"]) == pytest.approx(1.0)
        assert float(params["angle"]) == pytest.approx(35.0)
        assert "trackingtype" not in params
    finally:
        engine.close()


def test_ground_mount_sends_free_and_south_aspect_zero() -> None:
    calls: list[httpx.Request] = []
    engine = _engine(transport=_transport(payload=_load("pvcalc.json"), calls=calls))
    try:
        engine.evaluate(_request())
        params = calls[0].url.params
        assert params["mountingplace"] == "free"
        assert float(params["aspect"]) == pytest.approx(0.0)
    finally:
        engine.close()


def test_same_request_is_cached() -> None:
    calls: list[httpx.Request] = []
    cache = MemoryCache()
    engine = _engine(cache=cache, transport=_transport(payload=_load("pvcalc.json"), calls=calls))
    try:
        first = engine.evaluate(_request())
        second = engine.evaluate(_request())
        assert first.annual_energy_kwh == second.annual_energy_kwh
        assert first.provider_metadata.from_cache is False
        assert second.provider_metadata.from_cache is True
        assert len(calls) == 1
    finally:
        engine.close()


def test_missing_system_loss_does_not_default_or_call_http() -> None:
    calls: list[httpx.Request] = []
    engine = _engine(transport=_transport(payload=_load("pvcalc.json"), calls=calls))
    try:
        result = engine.evaluate(_request(system_loss_percent=None))
        assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
        assert result.annual_energy_kwh is None
        assert result.specific_yield_kwh_per_kwp is None
        assert "14%" in result.warnings[0]
        assert calls == []
    finally:
        engine.close()


def test_missing_point_is_incomplete() -> None:
    calls: list[httpx.Request] = []
    engine = _engine(transport=_transport(payload=_load("pvcalc.json"), calls=calls))
    try:
        result = engine.evaluate(
            _request(location=GeographicRef(name="named only, no coordinates"))
        )
        assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
        assert result.annual_energy_kwh is None
        assert calls == []
    finally:
        engine.close()


def test_single_axis_is_not_inferred() -> None:
    calls: list[httpx.Request] = []
    engine = _engine(transport=_transport(payload=_load("pvcalc.json"), calls=calls))
    try:
        result = engine.evaluate(_request(tracking=TrackingMode.SINGLE_AXIS))
        assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
        assert result.annual_energy_kwh is None
        assert "single-axis" in result.warnings[0]
        assert calls == []
    finally:
        engine.close()


def test_unsupported_technology_is_not_rewritten() -> None:
    calls: list[httpx.Request] = []
    engine = _engine(transport=_transport(payload=_load("pvcalc.json"), calls=calls))
    try:
        result = engine.evaluate(_request(technology="other"))
        assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
        assert result.annual_energy_kwh is None
        assert "pvtechchoice" in result.warnings[0]
        assert calls == []
    finally:
        engine.close()


def test_dual_axis_does_not_send_tilt_or_borrow_fixed_outputs() -> None:
    calls: list[httpx.Request] = []
    engine = _engine(transport=_transport(payload=_load("pvcalc.json"), calls=calls))
    try:
        result = engine.evaluate(_request(tracking=TrackingMode.DUAL_AXIS))
        assert len(calls) == 1
        params = calls[0].url.params
        assert params["trackingtype"] == "2"
        assert "angle" not in params
        assert "aspect" not in params
        assert result.status is EvaluationStatus.FAILED
        assert result.annual_energy_kwh is None
        assert any("two_axis" in item for item in result.warnings)
    finally:
        engine.close()


def test_http_error_does_not_fabricate_energy() -> None:
    engine = _engine(
        transport=_transport(status_code=400, error={"message": "Location over the sea"})
    )
    try:
        result = engine.evaluate(_request())
        assert result.status is EvaluationStatus.FAILED
        assert result.annual_energy_kwh is None
        assert result.specific_yield_kwh_per_kwp is None
        assert "sea" in result.warnings[0].lower()
    finally:
        engine.close()


def test_timeout_is_failed_without_energy() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("slow")

    engine = _engine(transport=httpx.MockTransport(handler), max_attempts=2)
    try:
        result = engine.evaluate(_request())
        assert result.status is EvaluationStatus.FAILED
        assert result.annual_energy_kwh is None
        assert "timeout" in result.warnings[0].lower()
    finally:
        engine.close()


def test_missing_e_y_uses_complete_year_sum_as_derived() -> None:
    payload = deepcopy(_load("pvcalc.json"))
    outputs = payload["outputs"]
    assert isinstance(outputs, dict)
    totals = outputs["totals"]
    assert isinstance(totals, dict)
    fixed = totals["fixed"]
    assert isinstance(fixed, dict)
    del fixed["E_y"]
    engine = _engine(transport=_transport(payload=payload))
    try:
        result = engine.evaluate(_request())
        assert result.status is EvaluationStatus.EVALUATED
        assert result.annual_energy_kwh is not None
        monthly_sum = sum(point.value.value for point in result.monthly_energy_kwh)
        assert result.annual_energy_kwh.value == pytest.approx(monthly_sum)
        assert result.annual_energy_kwh.origin is YieldValueOrigin.SOLARDSS_DERIVED
        assert result.annual_energy_kwh.calculation_method == "complete_year_sum_of_E_m"
        assert result.specific_yield_kwh_per_kwp is not None
        assert result.specific_yield_kwh_per_kwp.origin is YieldValueOrigin.SOLARDSS_DERIVED
    finally:
        engine.close()


def test_partial_year_is_not_summed() -> None:
    payload = deepcopy(_load("pvcalc.json"))
    outputs = payload["outputs"]
    assert isinstance(outputs, dict)
    totals = outputs["totals"]
    assert isinstance(totals, dict)
    fixed_totals = totals["fixed"]
    assert isinstance(fixed_totals, dict)
    del fixed_totals["E_y"]
    del fixed_totals["H(i)_y"]
    monthly = outputs["monthly"]
    assert isinstance(monthly, dict)
    series = monthly["fixed"]
    assert isinstance(series, list)
    series.pop()
    engine = _engine(transport=_transport(payload=payload))
    try:
        result = engine.evaluate(_request())
        assert result.status is EvaluationStatus.EVALUATED
        assert result.annual_energy_kwh is None
        assert result.annual_irradiation is None
        assert result.specific_yield_kwh_per_kwp is None
        assert len(result.monthly_energy_kwh) == 11
    finally:
        engine.close()


def test_missing_monthly_irradiation_is_not_zero_filled() -> None:
    payload = deepcopy(_load("pvcalc.json"))
    outputs = payload["outputs"]
    assert isinstance(outputs, dict)
    monthly = outputs["monthly"]
    assert isinstance(monthly, dict)
    series = monthly["fixed"]
    assert isinstance(series, list)
    first = series[0]
    assert isinstance(first, dict)
    del first["H(i)_m"]
    engine = _engine(transport=_transport(payload=payload))
    try:
        result = engine.evaluate(_request())
        assert result.monthly_energy_kwh[0].month == 1
        january_irrad = [row for row in result.monthly_irradiation if row.month == 1]
        assert january_irrad == []
        assert result.annual_irradiation is not None
        assert result.annual_irradiation.origin is YieldValueOrigin.PVGIS_PROVIDER
    finally:
        engine.close()


def test_invalid_loss_is_rejected() -> None:
    with pytest.raises(PVYieldConfigError):
        _request(system_loss_percent=114.0)


def test_engine_is_not_the_solar_resource_provider() -> None:
    assert not issubclass(PVGISYieldEngine, PVGISSolarResourceProvider)
    assert not issubclass(PVGISYieldEngine, PVGISYieldProvider)


def test_capacity_factor_identity_is_not_used() -> None:
    tree = ast.parse(SRC_PVGIS_YIELD.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "solardss.pv.energy":
            imported.extend(alias.name for alias in node.names)
    assert imported == []
    # The identity exists elsewhere and is not a PVGIS stand-in.
    assert energy_from_capacity_factor.__module__ == "solardss.pv.energy"


def test_yield_request_from_configuration() -> None:
    configuration = PVSystemConfiguration(
        location=SITE,
        module_id=TEST_MODULE_MONO.module_id,
        module_count=25,
        installed_capacity_w=10_000.0,
        tilt_deg=20.0,
        azimuth_deg=180.0,
        mounting_type=MountingType.ROOFTOP,
        tracking_mode=TrackingMode.FIXED,
        inverter_id=TEST_INVERTER.inverter_id,
    )
    request = yield_request_from_configuration(
        configuration,
        system_loss_percent=8.0,
        technology=TEST_MODULE_MONO.technology,
    )
    assert request.installed_capacity_kw == pytest.approx(10.0)
    assert request.mounting is MountingType.ROOFTOP
    assert request.system_loss_percent == pytest.approx(8.0)
    engine = _engine()
    try:
        result = engine.evaluate_from_configuration(
            configuration,
            system_loss_percent=8.0,
            technology="mono-c-Si",
        )
        assert result.status is EvaluationStatus.EVALUATED
        assert result.request.installed_capacity_kw == pytest.approx(10.0)
        assert result.specific_yield_kwh_per_kwp is not None
        assert result.specific_yield_kwh_per_kwp.value == pytest.approx(1125.46 / 10.0)
    finally:
        engine.close()
