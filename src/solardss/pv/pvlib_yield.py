"""Optional pvlib detailed PV energy-yield engine.

Uses official pvlib APIs for solar position, POA transposition, module
temperature, DC output, and the inverter. Advanced coefficients are not
inferred from mounting type or from a module database. Annual generation is
never estimated by an LLM.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

from solardss.core.arrays import FloatArray
from solardss.core.hashing import fingerprint_canonical, fingerprint_sequence
from solardss.core.jsonutil import JsonScalar
from solardss.core.methods import MethodId
from solardss.core.status import EvaluationStatus
from solardss.exceptions import MissingExtraError
from solardss.provenance.record import provenance_record
from solardss.pv.design.contracts import TrackingMode
from solardss.pv.yield_contracts import (
    AcModel,
    AoiLossModel,
    DcModel,
    HourlyPowerPoint,
    MonthlyYieldValue,
    PvlibModelSelection,
    PVLossAssumptions,
    PVYieldEngineMetadata,
    PVYieldRequest,
    PVYieldResult,
    TaggedYieldValue,
    TemperatureModel,
    TranspositionModel,
    YieldValueOrigin,
)

_CALC_SPECIFIC_YIELD: Final = "E_y / peakpower"
_CALC_CAPACITY_FACTOR: Final = "E_kWh / (P_kW * duration_h)"
_CALC_ENERGY_INTEGRAL: Final = "sum(P_ac_kW * interval_hours)"
_CALC_POA_INTEGRAL: Final = "sum(POA_W_m2 * interval_hours / 1000)"
_CALC_INVERTER_LOSS: Final = "100 * (1 - sum(P_ac) / sum(P_dc))"
_STC_TEMP_C: Final = 25.0
_PVWATTS_IRRAD_REF: Final = 1000.0
_PVWATTS_ETA_INV_REF: Final = 0.9637
_PERCENT_GAMMA_UNITS: Final = frozenset({"%/c", "%/k", "%/degc", "percent/c", "percent/k"})
_FRACTION_GAMMA_UNITS: Final = frozenset({"1/c", "/c", "1/k", "/k", "1/degc"})
_DEFAULT_MODELS: Final = PvlibModelSelection()
_TRANSPOSITION_NEEDS_EXTRA: Final = frozenset(
    {TranspositionModel.HAYDAVIES, TranspositionModel.PEREZ}
)
_INTERVAL_IRREGULAR: Final = 0.01


def pvlib_module_available() -> bool:
    """Return whether pvlib is importable, without importing it."""

    return importlib.util.find_spec("pvlib") is not None


def require_pvlib() -> Any:
    """Import pvlib or raise ``MissingExtraError``."""

    try:
        import pvlib
    except ImportError as exc:
        raise MissingExtraError(
            "Optional pvlib extra is not installed. Install with: pip install 'solardss[pvlib]'."
        ) from exc
    return pvlib


class PvlibYieldEngine:
    """Evaluate hourly-to-annual PV yield with official pvlib physics APIs.

    Implements the same ``PVYieldEngine`` protocol as ``PVGISYieldEngine``.
    This engine does not call PVGIS and does not load CEC/Sandia databases.
    """

    def evaluate(self, request: PVYieldRequest) -> PVYieldResult:
        """Run the pvlib chain. Missing required physics inputs stay incomplete."""

        models, default_notes = _models(request)
        planned = {
            "solar_position": models.solar_position.value,
            "transposition": models.transposition.value,
            "temperature": models.temperature.value,
            "dc": models.dc.value,
            "ac": models.ac.value,
            "aoi": models.aoi.value,
            "tracking": request.tracking.value,
        }
        weather_source = request.weather_source
        if weather_source is None and request.weather:
            weather_source = "caller_supplied"
        loss_assumptions = _requested_loss(request)
        metadata = PVYieldEngineMetadata(
            provider_id="pvlib.pv_yield",
            source_name="pvlib",
            tool="pvlib",
            elevation_m=request.elevation_m,
            models_used=planned,
            defaults_applied=tuple(default_notes),
            weather_source=weather_source,
            module_data_source=request.module_data_source,
        )
        incomplete = _incomplete_physics(request, models)
        if incomplete is not None:
            return _result(
                request,
                status=EvaluationStatus.INCOMPLETE_INPUTS,
                warnings=(incomplete,),
                loss_assumptions=loss_assumptions,
                metadata=metadata,
            )
        pvlib = require_pvlib()
        version = str(getattr(pvlib, "__version__", "") or "") or None
        metadata = metadata.model_copy(update={"api_version": version, "pvlib_version": version})
        try:
            run = _run_chain(request, models, pvlib)
        except (ValueError, TypeError, KeyError) as exc:
            return _result(
                request,
                status=EvaluationStatus.FAILED,
                warnings=(f"pvlib evaluation failed: {exc}",),
                loss_assumptions=loss_assumptions,
                metadata=metadata,
            )
        run.defaults_applied = list(default_notes) + run.defaults_applied
        metadata = metadata.model_copy(
            update={
                "models_used": dict(run.models_used),
                "assumptions": tuple(run.assumptions),
                "defaults_applied": tuple(run.defaults_applied),
                "weather_source": run.weather_source,
                "module_data_source": run.module_data_source,
                "inverter_data_source": run.inverter_data_source,
            }
        )
        if run.incomplete is not None:
            return _result(
                request,
                status=EvaluationStatus.INCOMPLETE_INPUTS,
                warnings=_unique((run.incomplete, *run.warnings)),
                loss_assumptions=loss_assumptions,
                metadata=metadata,
                extra_notes=tuple(run.assumptions),
            )
        return _result(
            request,
            status=EvaluationStatus.EVALUATED,
            warnings=tuple(run.warnings),
            loss_assumptions=_merge_losses(loss_assumptions, request, run),
            metadata=metadata,
            annual_energy_kwh=run.annual_energy_kwh,
            specific_yield_kwh_per_kwp=run.specific_yield,
            monthly_energy_kwh=run.monthly_energy,
            monthly_irradiation=run.monthly_irradiation,
            annual_irradiation=run.annual_irradiation,
            hourly_ac_power=run.hourly,
            capacity_factor=run.capacity_factor,
            extra_notes=(*run.assumptions, *_origin_notes(run)),
            extra_parameters=_run_parameters(request, run),
        )


@dataclass
class _ChainResult:
    incomplete: str | None = None
    warnings: list[str] = field(default_factory=list)
    models_used: dict[str, str] = field(default_factory=dict)
    assumptions: list[str] = field(default_factory=list)
    defaults_applied: list[str] = field(default_factory=list)
    weather_source: str | None = None
    module_data_source: str | None = None
    inverter_data_source: str | None = None
    interval_hours: float | None = None
    pdc0_w: float | None = None
    hourly: tuple[HourlyPowerPoint, ...] = ()
    annual_energy_kwh: TaggedYieldValue | None = None
    specific_yield: TaggedYieldValue | None = None
    monthly_energy: tuple[MonthlyYieldValue, ...] = ()
    monthly_irradiation: tuple[MonthlyYieldValue, ...] = ()
    annual_irradiation: TaggedYieldValue | None = None
    capacity_factor: TaggedYieldValue | None = None
    inverter_loss_percent: float | None = None


def _models(request: PVYieldRequest) -> tuple[PvlibModelSelection, list[str]]:
    if request.models is not None:
        return request.models, []
    return _DEFAULT_MODELS, [
        "models.solar_position default nrel_numpy",
        "models.transposition default haydavies",
        "models.temperature default sapm",
        "models.dc default pvwatts",
        "models.ac default pvwatts",
        "models.aoi default none (POA global used as effective irradiance)",
    ]


def _incomplete_physics(request: PVYieldRequest, models: PvlibModelSelection) -> str | None:
    lat = request.location.latitude
    lon = request.location.longitude
    if lat is None or lon is None:
        return "pvlib requires a point latitude and longitude"
    if request.elevation_m is None:
        return "pvlib requires elevation_m; SolarDSS does not assume sea level"
    if not request.weather:
        return (
            "pvlib requires a caller-supplied weather timeseries (GHI, DNI, DHI, temperature, wind)"
        )
    interval_reason = _interval_incomplete(request)
    if interval_reason is not None:
        return interval_reason
    if request.timezone is not None:
        try:
            ZoneInfo(request.timezone.strip())
        except (ZoneInfoNotFoundError, KeyError, ValueError):
            return f"timezone {request.timezone!r} is not a valid IANA zone"
    if request.installed_capacity_kw is None and (
        request.module is None or request.module_count is None
    ):
        return (
            "pvlib requires installed_capacity_kw or module plus module_count; "
            "CEC module databases are not loaded"
        )
    if request.system_loss_percent is None:
        return "pvlib requires system_loss_percent; SolarDSS does not default 14% or 0%"
    tracking_reason = _tracking_incomplete(request)
    if tracking_reason is not None:
        return tracking_reason
    if models.dc is not DcModel.PVWATTS:
        return f"dc model {models.dc.value!r} is not implemented without a coefficient set"
    if models.ac is not AcModel.PVWATTS:
        return f"ac model {models.ac.value!r} is not implemented without a coefficient set"
    aoi_reason = _aoi_incomplete(request, models)
    if aoi_reason is not None:
        return aoi_reason
    gamma, _note = _gamma_pdc(request)
    if gamma is None:
        return (
            "pvlib PVWatts DC requires gamma_pdc or module.temperature_coefficient_pmax; "
            "-0.003 /C is not assumed"
        )
    eta, _source, _note = _inverter_efficiency(request)
    if eta is None:
        return (
            "pvlib PVWatts AC requires inverter.efficiency or inverter_efficiency; "
            "0.96 is not assumed"
        )
    if request.module is not None and request.module.bifacial:
        return (
            "bifacial modules require rear irradiance inputs; pvlib.bifacial is not "
            "enabled from the bifacial flag alone"
        )
    weather_reason = _weather_incomplete(request)
    if weather_reason is not None:
        return weather_reason
    _params, temp_err = _temperature_params_or_reason(request, models)
    if temp_err is not None:
        return temp_err
    return None


def _tracking_incomplete(request: PVYieldRequest) -> str | None:
    if request.tracking is TrackingMode.FIXED:
        if request.tilt_deg is None or request.azimuth_deg is None:
            return "fixed pvlib tracking requires tilt_deg and azimuth_deg"
        if request.tilt_deg > 0.0 and request.albedo is None:
            return "pvlib POA requires albedo when tilt_deg > 0; it is not defaulted to 0.2"
        return None
    if request.tracking is TrackingMode.SINGLE_AXIS:
        if (
            request.axis_tilt_deg is None
            or request.axis_azimuth_deg is None
            or request.max_angle_deg is None
            or request.backtrack is None
        ):
            return (
                "pvlib single-axis tracking requires axis_tilt_deg, axis_azimuth_deg, "
                "max_angle_deg, and backtrack; those values are not inferred"
            )
        if request.backtrack and request.ground_coverage_ratio is None:
            return "pvlib backtracking requires ground_coverage_ratio; it is not inferred"
        if request.albedo is None:
            return "pvlib POA requires albedo for tracked arrays; it is not defaulted"
        return None
    if request.albedo is None:
        return "pvlib POA requires albedo for dual-axis tracking; it is not defaulted"
    return None


def _aoi_incomplete(request: PVYieldRequest, models: PvlibModelSelection) -> str | None:
    params = request.aoi_model_parameters or {}
    if models.aoi is AoiLossModel.PHYSICAL:
        missing = [key for key in ("n", "k", "l") if key not in params]
        if missing:
            return (
                "physical AOI losses require aoi_model_parameters n, k, and l; "
                f"missing {', '.join(missing)}. pvlib defaults are not applied"
            )
    if models.aoi is AoiLossModel.ASHRAE and "b" not in params:
        return "ASHRAE AOI losses require aoi_model_parameters['b']; b=0.05 is not assumed"
    return None


def _weather_incomplete(request: PVYieldRequest) -> str | None:
    for index, row in enumerate(request.weather):
        missing: list[str] = []
        if row.dni_w_m2 is None:
            missing.append("dni")
        if row.dhi_w_m2 is None:
            missing.append("dhi")
        if row.temp_air_c is None:
            missing.append("temp_air")
        if row.wind_speed_m_s is None:
            missing.append("wind_speed")
        if missing:
            return (
                f"weather[{index}] is missing {', '.join(missing)}; "
                "GHI is not decomposed and temperature/wind are not defaulted"
            )
    return None


def _interval_incomplete(request: PVYieldRequest) -> str | None:
    if request.interval_hours is not None:
        return None
    if len(request.weather) < 2:
        return "a single weather row requires interval_hours; 1 h is not assumed"
    stamps = [row.timestamp.astimezone(timezone.utc) for row in request.weather]
    deltas = np.diff(np.array([stamp.timestamp() for stamp in stamps], dtype=float)) / 3600.0
    if np.any(deltas <= 0.0):
        return "weather timestamps must be strictly increasing"
    median = float(np.median(deltas))
    if median <= 0.0:
        return "weather intervals must be positive"
    spread = float(np.max(np.abs(deltas - median) / median))
    if spread > _INTERVAL_IRREGULAR:
        return "weather intervals are irregular; supply interval_hours rather than assuming 1 h"
    return None


def _run_chain(request: PVYieldRequest, models: PvlibModelSelection, pvlib: Any) -> _ChainResult:
    from pvlib import atmosphere, iam, inverter, irradiance, pvsystem, solarposition, temperature
    from pvlib import tracking as pvtracking

    run = _ChainResult()
    frame, interval, infer_note = _weather_frame(request)
    if infer_note:
        run.defaults_applied.append(infer_note)
    run.interval_hours = interval
    run.weather_source = request.weather_source or "caller_supplied"

    pdc0, module_source, pdc_note = _array_pdc0_w(request)
    run.pdc0_w = pdc0
    run.module_data_source = request.module_data_source or module_source
    if pdc_note:
        run.assumptions.append(pdc_note)
    gamma, gamma_note = _gamma_pdc(request)
    if gamma_note:
        run.assumptions.append(gamma_note)
    eta, eta_source, eta_note = _inverter_efficiency(request)
    run.inverter_data_source = eta_source
    if eta_note:
        run.assumptions.append(eta_note)

    lat = request.location.latitude
    lon = request.location.longitude
    assert lat is not None and lon is not None
    assert request.elevation_m is not None
    assert pdc0 is not None and gamma is not None and eta is not None
    assert request.system_loss_percent is not None

    mean_temp = float(frame["temp_air"].mean())
    pressure = float(atmosphere.alt2pres(request.elevation_m))
    run.models_used["solar_position"] = (
        f"pvlib.solarposition.get_solarposition/{models.solar_position.value}"
    )
    run.assumptions.append(
        f"SPA pressure from pvlib.atmosphere.alt2pres(elevation_m={request.elevation_m})"
    )
    run.assumptions.append(
        f"SPA refraction temperature is the mean of supplied temp_air ({mean_temp:.4g} C)"
    )
    solpos = solarposition.get_solarposition(
        time=frame.index,
        latitude=lat,
        longitude=lon,
        altitude=request.elevation_m,
        pressure=pressure,
        temperature=mean_temp,
        method=models.solar_position.value,
    )

    tilt, azimuth, orient_notes = _orientation(request, frame, solpos, pvtracking, run.warnings)
    run.assumptions.extend(orient_notes)
    run.models_used["tracking"] = request.tracking.value

    dni_extra = None
    airmass = None
    if models.transposition in _TRANSPOSITION_NEEDS_EXTRA:
        dni_extra = irradiance.get_extra_radiation(frame.index)
        run.models_used["dni_extra"] = "pvlib.irradiance.get_extra_radiation"
    if models.transposition is TranspositionModel.PEREZ:
        airmass = atmosphere.get_relative_airmass(solpos["apparent_zenith"])
        run.models_used["airmass"] = "pvlib.atmosphere.get_relative_airmass"

    albedo = _albedo(request)
    run.models_used["transposition"] = (
        f"pvlib.irradiance.get_total_irradiance/{models.transposition.value}"
    )
    poa = irradiance.get_total_irradiance(
        surface_tilt=tilt,
        surface_azimuth=azimuth,
        solar_zenith=solpos["apparent_zenith"],
        solar_azimuth=solpos["azimuth"],
        dni=frame["dni"],
        ghi=frame["ghi"],
        dhi=frame["dhi"],
        dni_extra=dni_extra,
        airmass=airmass,
        albedo=albedo,
        model=models.transposition.value,
    )
    poa_global = np.clip(np.asarray(poa["poa_global"], dtype=float), 0.0, None)

    effective, aoi_notes = _effective_irradiance(
        models, request, tilt, azimuth, solpos, poa_global, irradiance, iam
    )
    run.assumptions.extend(aoi_notes)
    run.models_used["aoi"] = f"pvlib/{models.aoi.value}"
    run.assumptions.append("spectral losses are not applied")

    temp_params, temp_err = _temperature_params_or_reason(request, models, pvlib=pvlib)
    if temp_err is not None or temp_params is None:
        run.incomplete = temp_err or "temperature-model coefficients are incomplete"
        return run
    named = request.sapm_temperature_model
    if named:
        run.assumptions.append(
            f"SAPM temperature coefficients from pvlib named set {named!r} (caller-selected)"
        )
    else:
        run.assumptions.append("temperature-model coefficients were caller-supplied")
    run.models_used["temperature"] = f"pvlib.temperature.{models.temperature.value}"
    temp_air = frame["temp_air"].to_numpy(dtype=float)
    wind = frame["wind_speed"].to_numpy(dtype=float)
    if models.temperature is TemperatureModel.SAPM:
        temp_cell = temperature.sapm_cell(
            poa_global, temp_air, wind, temp_params["a"], temp_params["b"], temp_params["deltaT"]
        )
    else:
        temp_cell = temperature.faiman(
            poa_global, temp_air, wind, u0=temp_params["u0"], u1=temp_params["u1"]
        )

    run.models_used["dc"] = "pvlib.pvsystem.pvwatts_dc"
    run.assumptions.append(
        f"PVWatts DC temp_ref={_STC_TEMP_C} C and G_ref={_PVWATTS_IRRAD_REF} W/m2 (STC)"
    )
    pdc = np.asarray(
        pvsystem.pvwatts_dc(effective, temp_cell, pdc0, gamma, temp_ref=_STC_TEMP_C),
        dtype=float,
    )
    pdc = np.where(np.isfinite(pdc), np.clip(pdc, 0.0, None), np.nan)

    inverter_pdc0, inv_note = _inverter_pdc0(request, pdc0, eta)
    run.assumptions.append(inv_note)
    run.models_used["ac"] = "pvlib.inverter.pvwatts"
    run.assumptions.append(
        f"PVWatts inverter eta_inv_ref={_PVWATTS_ETA_INV_REF} (NREL model constant)"
    )
    pac = np.asarray(
        inverter.pvwatts(pdc, inverter_pdc0, eta_inv_nom=eta, eta_inv_ref=_PVWATTS_ETA_INV_REF),
        dtype=float,
    )
    # The PVWatts curve uses 1/zeta; DC power of 0 is defined as AC power of 0.
    pac = np.where(pdc <= 0.0, 0.0, pac)
    pac = np.where(np.isfinite(pac), np.clip(pac, 0.0, None), np.nan)
    pac_net = pac * (1.0 - request.system_loss_percent / 100.0)
    run.models_used["system_loss"] = "caller_system_loss_derate"

    hourly, energy_kwh, poa_kwh_m2, dc_sum, ac_pre = _hourly_and_energy(
        pd.DatetimeIndex(frame.index), pac_net, pac, pdc, poa_global, interval
    )
    run.hourly = hourly
    if not hourly:
        run.incomplete = "no finite hourly AC values were produced"
        run.warnings.append("POA or power contained NaN at every timestep")
        return run
    if len(hourly) < len(frame.index):
        run.warnings.append("POA or power contained NaN timesteps; those hours were omitted")

    run.annual_energy_kwh = TaggedYieldValue(
        value=energy_kwh,
        unit="kWh",
        origin=YieldValueOrigin.PVLIB_MODEL,
        calculation_method=_CALC_ENERGY_INTEGRAL,
        notes=("integrated from pvlib inverter AC after the caller system-loss derate",),
    )
    run.annual_irradiation = TaggedYieldValue(
        value=poa_kwh_m2,
        unit="kWh/m2",
        origin=YieldValueOrigin.PVLIB_MODEL,
        source_field="poa_global",
        calculation_method=_CALC_POA_INTEGRAL,
    )
    peak_kw = pdc0 / 1000.0
    run.specific_yield = TaggedYieldValue(
        value=energy_kwh / peak_kw,
        unit="kWh/kWp",
        origin=YieldValueOrigin.SOLARDSS_DERIVED,
        calculation_method=_CALC_SPECIFIC_YIELD,
        notes=("identity only; pvlib does not publish specific yield as a separate field",),
    )
    duration_h = interval * float(len(hourly))
    run.capacity_factor = TaggedYieldValue(
        value=energy_kwh / (peak_kw * duration_h),
        unit="1",
        origin=YieldValueOrigin.SOLARDSS_DERIVED,
        calculation_method=_CALC_CAPACITY_FACTOR,
        notes=(f"duration_h is interval_hours * n_timesteps ({duration_h:.6g} h)",),
    )
    run.monthly_energy, run.monthly_irradiation = _monthly_aggregates(hourly, interval)
    if dc_sum > 0.0:
        run.inverter_loss_percent = 100.0 * (1.0 - ac_pre / dc_sum)
    return run


def _weather_frame(request: PVYieldRequest) -> tuple[pd.DataFrame, float, str | None]:
    stamps = [_index_stamp(row.timestamp, request.timezone) for row in request.weather]
    frame = pd.DataFrame(
        {
            "ghi": [row.ghi_w_m2 for row in request.weather],
            "dni": [row.dni_w_m2 for row in request.weather],
            "dhi": [row.dhi_w_m2 for row in request.weather],
            "temp_air": [row.temp_air_c for row in request.weather],
            "wind_speed": [row.wind_speed_m_s for row in request.weather],
        },
        index=pd.DatetimeIndex(stamps),
    )
    if request.interval_hours is not None:
        return frame, request.interval_hours, None
    deltas = np.diff(np.array([stamp.timestamp() for stamp in stamps], dtype=float)) / 3600.0
    median = float(np.median(deltas))
    return frame, median, "interval_hours inferred from median timestamp spacing"


def _index_stamp(stamp: datetime, timezone_name: str | None) -> datetime:
    if timezone_name is None:
        return stamp.astimezone(timezone.utc)
    return stamp.astimezone(ZoneInfo(timezone_name.strip()))


def _albedo(request: PVYieldRequest) -> float:
    if request.tracking is TrackingMode.FIXED and request.tilt_deg == 0.0:
        return 0.0
    assert request.albedo is not None
    return request.albedo


def _orientation(
    request: PVYieldRequest,
    frame: pd.DataFrame,
    solpos: pd.DataFrame,
    pvtracking: Any,
    warnings: list[str],
) -> tuple[Any, Any, list[str]]:
    notes: list[str] = [
        "SolarDSS azimuth is clockwise from north and matches pvlib surface_azimuth"
    ]
    if request.tracking is TrackingMode.FIXED:
        assert request.tilt_deg is not None and request.azimuth_deg is not None
        if request.tilt_deg == 0.0:
            notes.append("fixed tilt=0 so ground-reflected POA is identically 0; albedo is unused")
        return request.tilt_deg, request.azimuth_deg, notes
    zenith = solpos["apparent_zenith"]
    solar_az = solpos["azimuth"]
    if request.tracking is TrackingMode.DUAL_AXIS:
        tilt = zenith.clip(lower=0.0, upper=90.0)
        notes.append(
            "dual-axis orientation is surface_tilt=apparent_zenith, "
            "surface_azimuth=solar azimuth (pvlib has no dual-axis Mount helper)"
        )
        return tilt, solar_az, notes
    assert request.axis_tilt_deg is not None
    assert request.axis_azimuth_deg is not None
    assert request.max_angle_deg is not None
    assert request.backtrack is not None
    kwargs: dict[str, Any] = {
        "apparent_zenith": zenith,
        "apparent_azimuth": solar_az,
        "axis_tilt": request.axis_tilt_deg,
        "axis_azimuth": request.axis_azimuth_deg,
        "max_angle": request.max_angle_deg,
        "backtrack": request.backtrack,
    }
    if request.backtrack:
        kwargs["gcr"] = request.ground_coverage_ratio
    oriented = pvtracking.singleaxis(**kwargs)
    tilt = oriented["surface_tilt"].copy()
    azimuth = oriented["surface_azimuth"].copy()
    below = zenith.to_numpy(dtype=float) >= 90.0
    if bool(np.any(below) and (tilt.isna().to_numpy().any() or azimuth.isna().to_numpy().any())):
        tilt = tilt.fillna(0.0)
        azimuth = azimuth.fillna(180.0)
        warnings.append(
            "single-axis orientation is undefined below the horizon; "
            "tilt=0 azimuth=180 was used for those hours"
        )
        notes.append("below-horizon tracker NaNs replaced with tilt=0, azimuth=180")
    notes.append("single-axis orientation from pvlib.tracking.singleaxis")
    return tilt.reindex(frame.index), azimuth.reindex(frame.index), notes


def _effective_irradiance(
    models: PvlibModelSelection,
    request: PVYieldRequest,
    tilt: Any,
    azimuth: Any,
    solpos: pd.DataFrame,
    poa_global: FloatArray,
    irradiance: Any,
    iam_mod: Any,
) -> tuple[FloatArray, list[str]]:
    if models.aoi is AoiLossModel.NONE:
        return poa_global, ["AOI losses not applied; effective irradiance is POA global"]
    aoi = irradiance.aoi(tilt, azimuth, solpos["apparent_zenith"], solpos["azimuth"])
    params = request.aoi_model_parameters or {}
    if models.aoi is AoiLossModel.PHYSICAL:
        modifier = iam_mod.physical(aoi, n=params["n"], k=params["k"], l=params["l"])
        return poa_global * np.asarray(modifier, dtype=float), [
            "AOI modifier pvlib.iam.physical with caller n, k, and l"
        ]
    modifier = iam_mod.ashrae(aoi, b=params["b"])
    return poa_global * np.asarray(modifier, dtype=float), [
        "AOI modifier pvlib.iam.ashrae with caller b"
    ]


def _temperature_params_or_reason(
    request: PVYieldRequest,
    models: PvlibModelSelection,
    pvlib: Any | None = None,
) -> tuple[dict[str, float] | None, str | None]:
    named = request.sapm_temperature_model
    explicit = request.temperature_model_parameters
    if named and explicit:
        return None, ("supply sapm_temperature_model or temperature_model_parameters, not both")
    if models.temperature is TemperatureModel.SAPM:
        if explicit is not None:
            missing = [key for key in ("a", "b", "deltaT") if key not in explicit]
            if missing:
                return None, (
                    "SAPM temperature_model_parameters must include a, b, and deltaT; "
                    f"missing {', '.join(missing)}"
                )
            return {"a": explicit["a"], "b": explicit["b"], "deltaT": explicit["deltaT"]}, None
        if named is None:
            return None, (
                "SAPM temperature requires sapm_temperature_model or "
                "temperature_model_parameters; open_rack_glass_glass is not inferred "
                "from mounting"
            )
        if pvlib is None:
            return {}, None
        table = getattr(pvlib.temperature, "TEMPERATURE_MODEL_PARAMETERS", {}).get("sapm", {})
        params = table.get(named)
        if not isinstance(params, Mapping) or any(
            key not in params for key in ("a", "b", "deltaT")
        ):
            return None, (
                f"sapm_temperature_model {named!r} is not a known pvlib SAPM parameter set"
            )
        return {
            "a": float(params["a"]),
            "b": float(params["b"]),
            "deltaT": float(params["deltaT"]),
        }, None
    if explicit is None:
        return None, (
            "Faiman temperature requires temperature_model_parameters u0 and u1; "
            "they are not inferred from mounting"
        )
    missing = [key for key in ("u0", "u1") if key not in explicit]
    if missing:
        return None, (
            "Faiman temperature_model_parameters must include u0 and u1; "
            f"missing {', '.join(missing)}"
        )
    return {"u0": explicit["u0"], "u1": explicit["u1"]}, None


def _array_pdc0_w(request: PVYieldRequest) -> tuple[float, str, str | None]:
    if request.installed_capacity_kw is not None:
        source = "installed_capacity_kw"
        if request.module is not None:
            note = "array pdc0 is installed_capacity_kw; module rated power is not substituted"
        else:
            note = "array pdc0 is installed_capacity_kw * 1000 W"
        return request.installed_capacity_kw * 1000.0, source, note
    assert request.module is not None and request.module_count is not None
    pdc0 = request.module.rated_power_w * float(request.module_count)
    source = request.module.provenance.source_name
    note = (
        f"array pdc0 is module.rated_power_w * module_count "
        f"({request.module.rated_power_w} W * {request.module_count})"
    )
    return pdc0, source, note


def _gamma_pdc(request: PVYieldRequest) -> tuple[float | None, str | None]:
    if request.gamma_pdc is not None:
        return request.gamma_pdc, "gamma_pdc was caller-supplied in 1/C"
    module = request.module
    if module is None or module.temperature_coefficient_pmax is None:
        return None, None
    unit = module.temperature_coefficient_pmax_unit.strip().lower().replace(" ", "")
    coeff = module.temperature_coefficient_pmax
    if unit in _PERCENT_GAMMA_UNITS:
        return coeff / 100.0, (
            f"gamma_pdc = temperature_coefficient_pmax/100 from unit "
            f"{module.temperature_coefficient_pmax_unit!r}"
        )
    if unit in _FRACTION_GAMMA_UNITS:
        return coeff, (
            f"gamma_pdc taken as temperature_coefficient_pmax with unit "
            f"{module.temperature_coefficient_pmax_unit!r}"
        )
    return None, None


def _inverter_efficiency(request: PVYieldRequest) -> tuple[float | None, str | None, str | None]:
    if request.inverter_efficiency is not None:
        return (
            request.inverter_efficiency,
            "caller_inverter_efficiency",
            "inverter efficiency was caller-supplied",
        )
    if request.inverter is not None and request.inverter.efficiency is not None:
        return (
            request.inverter.efficiency,
            request.inverter.provenance.source_name,
            "inverter efficiency from PVInverter.efficiency",
        )
    return None, None, None


def _inverter_pdc0(request: PVYieldRequest, array_pdc0_w: float, eta: float) -> tuple[float, str]:
    if request.inverter is not None:
        ac_w = request.inverter.rated_ac_power_w
        pdc0 = ac_w / eta
        return pdc0, (f"inverter.pvwatts pdc0 = rated_ac_power_w / eta ({ac_w} W / {eta})")
    return array_pdc0_w, "inverter.pvwatts pdc0 equals array DC nameplate (no inverter nameplate)"


def _hourly_and_energy(
    index: pd.DatetimeIndex,
    pac_net: FloatArray,
    pac: FloatArray,
    pdc: FloatArray,
    poa_global: FloatArray,
    interval: float,
) -> tuple[tuple[HourlyPowerPoint, ...], float, float, float, float]:
    points: list[HourlyPowerPoint] = []
    energy = 0.0
    poa_kwh = 0.0
    dc_sum = 0.0
    ac_pre = 0.0
    if len(index) != len(pac_net) or len(pac_net) != len(pac) or len(pac) != len(pdc):
        raise ValueError("pvlib power arrays and the weather index have different lengths")
    if len(pdc) != len(poa_global):
        raise ValueError("pvlib POA array and the weather index have different lengths")
    for i in range(len(index)):
        ac_net = float(pac_net[i])
        ac_w = float(pac[i])
        dc_w = float(pdc[i])
        poa = float(poa_global[i])
        if not (
            np.isfinite(ac_net) and np.isfinite(ac_w) and np.isfinite(dc_w) and np.isfinite(poa)
        ):
            continue
        stamp = pd.Timestamp(index[i]).to_pydatetime()
        aware = stamp if stamp.tzinfo is not None else stamp.replace(tzinfo=timezone.utc)
        points.append(
            HourlyPowerPoint(
                timestamp=aware,
                ac_power_w=TaggedYieldValue(
                    value=float(ac_net),
                    unit="W",
                    origin=YieldValueOrigin.PVLIB_MODEL,
                    calculation_method="pvlib.inverter.pvwatts+system_loss_derate",
                ),
                dc_power_w=TaggedYieldValue(
                    value=float(dc_w),
                    unit="W",
                    origin=YieldValueOrigin.PVLIB_MODEL,
                    calculation_method="pvlib.pvsystem.pvwatts_dc",
                ),
                poa_global_w_m2=TaggedYieldValue(
                    value=float(poa),
                    unit="W/m2",
                    origin=YieldValueOrigin.PVLIB_MODEL,
                    source_field="poa_global",
                    calculation_method="pvlib.irradiance.get_total_irradiance",
                ),
            )
        )
        energy += float(ac_net) / 1000.0 * interval
        poa_kwh += float(poa) / 1000.0 * interval
        dc_sum += float(dc_w)
        ac_pre += float(ac_w)
    return tuple(points), energy, poa_kwh, dc_sum, ac_pre


def _monthly_aggregates(
    hourly: Sequence[HourlyPowerPoint], interval: float
) -> tuple[tuple[MonthlyYieldValue, ...], tuple[MonthlyYieldValue, ...]]:
    energy_by_month: dict[int, float] = {}
    poa_by_month: dict[int, float] = {}
    for point in hourly:
        month = point.timestamp.month
        energy_by_month[month] = energy_by_month.get(month, 0.0) + (
            point.ac_power_w.value / 1000.0 * interval
        )
        if point.poa_global_w_m2 is not None:
            poa_by_month[month] = poa_by_month.get(month, 0.0) + (
                point.poa_global_w_m2.value / 1000.0 * interval
            )
    energy = tuple(
        MonthlyYieldValue(
            month=month,
            value=TaggedYieldValue(
                value=energy_by_month[month],
                unit="kWh",
                origin=YieldValueOrigin.SOLARDSS_DERIVED,
                calculation_method=_CALC_ENERGY_INTEGRAL,
            ),
        )
        for month in sorted(energy_by_month)
    )
    irrad = tuple(
        MonthlyYieldValue(
            month=month,
            value=TaggedYieldValue(
                value=poa_by_month[month],
                unit="kWh/m2",
                origin=YieldValueOrigin.SOLARDSS_DERIVED,
                source_field="poa_global",
                calculation_method=_CALC_POA_INTEGRAL,
            ),
        )
        for month in sorted(poa_by_month)
    )
    return energy, irrad


def _requested_loss(request: PVYieldRequest) -> PVLossAssumptions:
    requested = None
    if request.system_loss_percent is not None:
        requested = TaggedYieldValue(
            value=request.system_loss_percent,
            unit="%",
            origin=YieldValueOrigin.CALLER_SUPPLIED,
            source_field="system_loss_percent",
            notes=("caller system-loss derate applied to pvlib AC power",),
        )
    return PVLossAssumptions(requested_system_loss_percent=requested)


def _merge_losses(
    base: PVLossAssumptions, request: PVYieldRequest, run: _ChainResult
) -> PVLossAssumptions:
    derate = None
    if request.system_loss_percent is not None:
        derate = TaggedYieldValue(
            value=request.system_loss_percent,
            unit="%",
            origin=YieldValueOrigin.CALLER_SUPPLIED,
            source_field="system_loss_percent",
            notes=("multiplicative derate (1 - loss/100) applied after inverter AC",),
        )
    inverter_loss = None
    if run.inverter_loss_percent is not None:
        inverter_loss = TaggedYieldValue(
            value=run.inverter_loss_percent,
            unit="%",
            origin=YieldValueOrigin.SOLARDSS_DERIVED,
            calculation_method=_CALC_INVERTER_LOSS,
            notes=("from pre-derate inverter AC versus PVWatts DC; not a PVGIS l_* field",),
        )
    return PVLossAssumptions(
        requested_system_loss_percent=base.requested_system_loss_percent,
        inverter_loss_percent=inverter_loss,
        applied_system_derate_percent=derate,
        notes=("pvlib does not publish PVGIS l_aoi/l_spec/l_tg/l_total fields",),
    )


def _result(
    request: PVYieldRequest,
    *,
    status: EvaluationStatus,
    warnings: tuple[str, ...],
    loss_assumptions: PVLossAssumptions,
    metadata: PVYieldEngineMetadata,
    annual_energy_kwh: TaggedYieldValue | None = None,
    specific_yield_kwh_per_kwp: TaggedYieldValue | None = None,
    monthly_energy_kwh: tuple[MonthlyYieldValue, ...] = (),
    monthly_irradiation: tuple[MonthlyYieldValue, ...] = (),
    annual_irradiation: TaggedYieldValue | None = None,
    hourly_ac_power: tuple[HourlyPowerPoint, ...] = (),
    capacity_factor: TaggedYieldValue | None = None,
    extra_parameters: Mapping[str, JsonScalar] | None = None,
    extra_notes: Sequence[str] = (),
) -> PVYieldResult:
    lat = request.location.latitude
    lon = request.location.longitude
    parameters: dict[str, JsonScalar] = {
        "lat": lat,
        "lon": lon,
        "elevation_m": request.elevation_m,
        "timezone": request.timezone,
        "installed_capacity_kw": request.installed_capacity_kw,
        "tilt_deg": request.tilt_deg,
        "azimuth_deg": request.azimuth_deg,
        "mounting": request.mounting.value,
        "tracking": request.tracking.value,
        "system_loss_percent": request.system_loss_percent,
        "weather_source": metadata.weather_source,
        "module_data_source": metadata.module_data_source,
        "inverter_data_source": metadata.inverter_data_source,
        "annual_energy_origin": None
        if annual_energy_kwh is None
        else annual_energy_kwh.origin.value,
        "specific_yield_origin": None
        if specific_yield_kwh_per_kwp is None
        else specific_yield_kwh_per_kwp.origin.value,
        "capacity_factor_origin": None if capacity_factor is None else capacity_factor.origin.value,
    }
    if extra_parameters:
        parameters.update(extra_parameters)
    notes = [
        "annual generation is pvlib physics, not PVGIS PVcalc and not an LLM estimate",
        *request.notes,
        *extra_notes,
        *warnings,
    ]
    fingerprints = (
        fingerprint_canonical(
            {
                "lat": lat,
                "lon": lon,
                "elevation_m": request.elevation_m,
                "timezone": request.timezone,
                "peakpower": request.installed_capacity_kw,
                "loss": request.system_loss_percent,
                "tilt": request.tilt_deg,
                "azimuth": request.azimuth_deg,
                "tracking": request.tracking.value,
                "weather_ghi": fingerprint_sequence([row.ghi_w_m2 for row in request.weather]),
                "weather_n": len(request.weather),
            }
        ),
    )
    return PVYieldResult(
        status=status,
        request=request,
        annual_energy_kwh=annual_energy_kwh,
        specific_yield_kwh_per_kwp=specific_yield_kwh_per_kwp,
        monthly_energy_kwh=monthly_energy_kwh,
        monthly_irradiation=monthly_irradiation,
        annual_irradiation=annual_irradiation,
        hourly_ac_power=hourly_ac_power,
        capacity_factor=capacity_factor,
        loss_assumptions=loss_assumptions,
        provider_metadata=metadata,
        provenance=provenance_record(
            MethodId.PVLIB_PV_YIELD,
            parameters=parameters,
            input_fingerprints=fingerprints,
            notes=tuple(notes),
        ),
        warnings=warnings,
    )


def _run_parameters(request: PVYieldRequest, run: _ChainResult) -> dict[str, JsonScalar]:
    return {
        "interval_hours": run.interval_hours,
        "pdc0_w": run.pdc0_w,
        "albedo": request.albedo,
        "gamma_pdc": request.gamma_pdc,
        "n_hourly": len(run.hourly),
    }


def _origin_notes(run: _ChainResult) -> tuple[str, ...]:
    notes: list[str] = []
    if run.annual_energy_kwh is not None:
        notes.append(f"annual_energy_kwh origin={run.annual_energy_kwh.origin.value}")
    if run.specific_yield is not None:
        notes.append(f"specific_yield_kwh_per_kwp origin={run.specific_yield.origin.value}")
    if run.capacity_factor is not None:
        notes.append(f"capacity_factor origin={run.capacity_factor.origin.value}")
    return tuple(notes)


def _unique(messages: Sequence[str | None]) -> tuple[str, ...]:
    ordered: list[str] = []
    for item in messages:
        if item and item not in ordered:
            ordered.append(item)
    return tuple(ordered)
