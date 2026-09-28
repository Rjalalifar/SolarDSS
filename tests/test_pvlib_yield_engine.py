"""Optional pvlib yield engine: protocol, incomplete inputs, numerical identities."""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from solardss.core.methods import MethodId
from solardss.core.status import EvaluationStatus
from solardss.data.geo import GeographicRef
from solardss.exceptions import MissingExtraError
from solardss.pv import (
    PVGISYieldEngine,
    PvlibModelSelection,
    PvlibYieldEngine,
    PVYieldEngine,
    PVYieldRequest,
    WeatherObservation,
    pvlib_module_available,
    require_pvlib,
)
from solardss.pv.design.contracts import MountingType, TrackingMode
from solardss.pv.yield_contracts import TranspositionModel, YieldValueOrigin

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "pvlib" / "weather_simple.json"
SRC_PVLIB = Path(__file__).resolve().parents[1] / "src" / "solardss" / "pv" / "pvlib_yield.py"

# King / pvlib SAPM open_rack_glass_glass coefficients (caller-supplied in tests).
SAPM_A = -3.47
SAPM_B = -0.0594
SAPM_DELTA_T = 3.0
GAMMA_PDC = -0.003
ETA_INV = 0.96
ETA_INV_REF = 0.9637
PDC0_W = 1000.0


def _load_fixture() -> dict[str, object]:
    raw: object = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _weather_from_fixture() -> tuple[WeatherObservation, ...]:
    payload = _load_fixture()
    rows = payload["rows"]
    assert isinstance(rows, list)
    return tuple(WeatherObservation.model_validate(item) for item in rows)


def _request(**kwargs: object) -> PVYieldRequest:
    payload = _load_fixture()
    location = GeographicRef(
        name="pvlib fixture",
        longitude=float(payload["longitude"]),  # type: ignore[arg-type]
        latitude=float(payload["latitude"]),  # type: ignore[arg-type]
    )
    data: dict[str, object] = {
        "location": location,
        "elevation_m": float(payload["elevation_m"]),  # type: ignore[arg-type]
        "timezone": payload["timezone"],
        "installed_capacity_kw": float(payload["installed_capacity_kw"]),  # type: ignore[arg-type]
        "tilt_deg": 0.0,
        "azimuth_deg": 180.0,
        "mounting": MountingType.GROUND,
        "tracking": TrackingMode.FIXED,
        "system_loss_percent": float(payload["system_loss_percent"]),  # type: ignore[arg-type]
        "gamma_pdc": GAMMA_PDC,
        "inverter_efficiency": ETA_INV,
        "temperature_model_parameters": {
            "a": SAPM_A,
            "b": SAPM_B,
            "deltaT": SAPM_DELTA_T,
        },
        "weather": _weather_from_fixture(),
        "interval_hours": float(payload["interval_hours"]),  # type: ignore[arg-type]
        "weather_source": "test_fixture",
        "models": PvlibModelSelection(transposition=TranspositionModel.ISOTROPIC),
    }
    data.update(kwargs)
    return PVYieldRequest.model_validate(data)


def _sapm_cell(poa: float, tair: float, wind: float) -> float:
    return poa * math.exp(SAPM_A + SAPM_B * wind) + tair + (poa / 1000.0) * SAPM_DELTA_T


def _pvwatts_dc(poa: float, tcell: float) -> float:
    return PDC0_W * (poa / 1000.0) * (1.0 + GAMMA_PDC * (tcell - 25.0))


def _pvwatts_ac(pdc: float, pdc0: float = PDC0_W, eta_nom: float = ETA_INV) -> float:
    if pdc <= 0.0 or pdc0 <= 0.0:
        return 0.0
    zeta = pdc / pdc0
    eta = eta_nom / ETA_INV_REF * (-0.0162 * zeta - 0.0059 / zeta + 0.9858)
    return float(min(pdc0 * eta_nom, max(0.0, eta * pdc)))


def test_protocol_is_shared() -> None:
    assert isinstance(PvlibYieldEngine(), PVYieldEngine)
    assert isinstance(PVGISYieldEngine(), PVYieldEngine)
    assert hasattr(PVGISYieldEngine, "evaluate")
    assert hasattr(PvlibYieldEngine, "evaluate")


def test_require_pvlib_message(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__
    monkeypatch.delitem(sys.modules, "pvlib", raising=False)

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "pvlib" or name.startswith("pvlib."):
            raise ImportError("hidden")
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MissingExtraError, match="solardss\\[pvlib\\]"):
        require_pvlib()


def test_missing_dni_is_incomplete_not_decomposed() -> None:
    weather = list(_weather_from_fixture())
    noon = weather[1]
    weather[1] = WeatherObservation(
        timestamp=noon.timestamp,
        ghi_w_m2=noon.ghi_w_m2,
        dni_w_m2=None,
        dhi_w_m2=noon.dhi_w_m2,
        temp_air_c=noon.temp_air_c,
        wind_speed_m_s=noon.wind_speed_m_s,
    )
    result = PvlibYieldEngine().evaluate(_request(weather=tuple(weather)))
    assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
    assert result.annual_energy_kwh is None
    assert "dni" in result.warnings[0]
    assert "decomposed" in result.warnings[0]


def test_missing_gamma_is_incomplete() -> None:
    result = PvlibYieldEngine().evaluate(_request(gamma_pdc=None, module=None, module_count=None))
    assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
    assert result.annual_energy_kwh is None
    assert "gamma_pdc" in result.warnings[0]


def test_missing_elevation_is_incomplete() -> None:
    result = PvlibYieldEngine().evaluate(_request(elevation_m=None))
    assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
    assert "elevation_m" in result.warnings[0]


def test_single_axis_without_geometry_is_incomplete() -> None:
    result = PvlibYieldEngine().evaluate(_request(tracking=TrackingMode.SINGLE_AXIS, albedo=0.2))
    assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
    assert "axis_tilt_deg" in result.warnings[0]


def test_tilted_array_without_albedo_is_incomplete() -> None:
    result = PvlibYieldEngine().evaluate(_request(tilt_deg=30.0, albedo=None))
    assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
    assert "albedo" in result.warnings[0]


def test_missing_inverter_efficiency_is_incomplete() -> None:
    result = PvlibYieldEngine().evaluate(_request(inverter_efficiency=None, inverter=None))
    assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
    assert "inverter" in result.warnings[0]


def test_missing_weather_is_incomplete() -> None:
    result = PvlibYieldEngine().evaluate(_request(weather=()))
    assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
    assert "weather" in result.warnings[0]


def test_omitted_models_are_recorded_as_defaults() -> None:
    result = PvlibYieldEngine().evaluate(_request(models=None, elevation_m=None))
    assert result.status is EvaluationStatus.INCOMPLETE_INPUTS
    assert result.provider_metadata.models_used["transposition"] == "haydavies"
    assert any("haydavies" in note for note in result.provider_metadata.defaults_applied)


@pytest.mark.skipif(pvlib_module_available(), reason="pvlib extra is installed")
def test_complete_request_requires_pvlib_extra() -> None:
    with pytest.raises(MissingExtraError, match="solardss\\[pvlib\\]"):
        PvlibYieldEngine().evaluate(_request())


def test_import_does_not_load_pvlib() -> None:
    code = (
        "import sys\n"
        "import solardss.pv  # noqa: F401\n"
        "raise SystemExit(0 if 'pvlib' not in sys.modules else 1)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_source_does_not_call_retrieve_sam() -> None:
    tree = ast.parse(SRC_PVLIB.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.append(node.attr)
        if isinstance(node, ast.Name):
            names.append(node.id)
    assert "retrieve_sam" not in names
    assert "ModelChain" not in names
    assert "erbs" not in names
    assert "disc" not in names


@pytest.mark.skipif(not pvlib_module_available(), reason="optional pvlib extra is not installed")
def test_night_hour_is_zero_and_noon_matches_published_identities() -> None:
    result = PvlibYieldEngine().evaluate(_request())
    assert result.status is EvaluationStatus.EVALUATED
    assert len(result.hourly_ac_power) == 2
    night, noon = result.hourly_ac_power
    assert night.ac_power_w.value == pytest.approx(0.0, abs=1e-6)
    assert night.poa_global_w_m2 is not None
    assert night.poa_global_w_m2.value == pytest.approx(0.0, abs=1e-6)

    assert noon.poa_global_w_m2 is not None
    assert noon.dc_power_w is not None
    poa = noon.poa_global_w_m2.value
    assert poa == pytest.approx(1000.0, rel=1e-3, abs=1.0)
    tcell = _sapm_cell(poa, 25.0, 2.0)
    pdc = _pvwatts_dc(poa, tcell)
    assert noon.dc_power_w.value == pytest.approx(pdc, rel=1e-4, abs=0.05)
    pac = _pvwatts_ac(pdc)
    # system_loss is 0, so AC after derate equals inverter AC.
    assert noon.ac_power_w.value == pytest.approx(pac, rel=1e-3, abs=0.5)

    interval = 1.0
    energy = sum(point.ac_power_w.value / 1000.0 * interval for point in result.hourly_ac_power)
    assert result.annual_energy_kwh is not None
    assert result.annual_energy_kwh.value == pytest.approx(energy)
    assert result.annual_energy_kwh.origin is YieldValueOrigin.PVLIB_MODEL
    assert result.specific_yield_kwh_per_kwp is not None
    assert result.specific_yield_kwh_per_kwp.value == pytest.approx(energy / 1.0)
    assert result.specific_yield_kwh_per_kwp.origin is YieldValueOrigin.SOLARDSS_DERIVED
    assert result.capacity_factor is not None
    assert result.capacity_factor.value == pytest.approx(energy / (1.0 * 2.0))
    assert result.capacity_factor.origin is YieldValueOrigin.SOLARDSS_DERIVED
    assert len(result.monthly_energy_kwh) == 1
    assert result.monthly_energy_kwh[0].month == 3
    assert result.monthly_energy_kwh[0].value.value == pytest.approx(energy)
    assert result.provider_metadata.provider_id == "pvlib.pv_yield"
    assert result.provider_metadata.weather_source == "test_fixture"
    assert result.provider_metadata.module_data_source == "installed_capacity_kw"
    assert "transposition" in result.provider_metadata.models_used
    assert result.provenance.stamp.method_id == MethodId.PVLIB_PV_YIELD.value
    assert result.loss_assumptions.applied_system_derate_percent is not None
    assert result.loss_assumptions.applied_system_derate_percent.value == pytest.approx(0.0)
    assert result.loss_assumptions.inverter_loss_percent is not None


@pytest.mark.skipif(not pvlib_module_available(), reason="optional pvlib extra is not installed")
def test_system_loss_derates_ac() -> None:
    full = PvlibYieldEngine().evaluate(_request())
    lost = PvlibYieldEngine().evaluate(_request(system_loss_percent=10.0))
    assert full.annual_energy_kwh is not None
    assert lost.annual_energy_kwh is not None
    assert lost.annual_energy_kwh.value == pytest.approx(
        full.annual_energy_kwh.value * 0.9, rel=1e-6
    )
    assert lost.loss_assumptions.applied_system_derate_percent is not None
    assert lost.loss_assumptions.applied_system_derate_percent.origin is (
        YieldValueOrigin.CALLER_SUPPLIED
    )
