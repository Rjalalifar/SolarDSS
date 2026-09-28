"""PVGIS 5.3 PVcalc annual energy-yield engine.

This is not the solar-resource provider (``PVGISSolarResourceProvider`` /
``MRcalc``). Annual generation is taken from PVcalc; SolarDSS never estimates
it with an LLM or with a capacity-factor identity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from solardss.core.hashing import fingerprint_canonical
from solardss.core.jsonutil import JsonScalar
from solardss.core.methods import MethodId
from solardss.core.status import EvaluationStatus
from solardss.data.geo import GeographicRef
from solardss.data.pvgis import (
    PVGIS_API_VERSION,
    PVGIS_DEFAULT_BASE_URL,
    PVGIS_SOURCE_NAME,
    PVGISClient,
    PVGISHttpResult,
    PVGISMeteoData,
    PVGISPVMonthlyRow,
    PVGISPVTotals,
    parse_location,
    parse_meteo,
    parse_pv_monthly,
    parse_pv_totals,
)
from solardss.provenance.record import provenance_record
from solardss.providers.cache import DatasetCache, MemoryCache
from solardss.providers.clock import Clock, UtcClock
from solardss.providers.errors import PVGISHTTPError, PVGISResponseError
from solardss.pv.contracts import ModuleTechnology
from solardss.pv.design.contracts import MountingType, PVSystemConfiguration, TrackingMode
from solardss.pv.yield_contracts import (
    MonthlyYieldValue,
    PVLossAssumptions,
    PVYieldEngineMetadata,
    PVYieldRequest,
    PVYieldResult,
    TaggedYieldValue,
    YieldValueOrigin,
)

_CALC_SPECIFIC_YIELD: Final = "E_y / peakpower"
_CALC_SUM_ENERGY: Final = "complete_year_sum_of_E_m"
_CALC_SUM_IRRAD: Final = "complete_year_sum_of_H(i)_m"

_ENERGY_YEAR_UNITS: Final = frozenset({"kwh/y", "kwh/year", "kwh"})
_IRRAD_YEAR_UNITS: Final = frozenset({"kwh/m2/y", "kwh/m2/year"})
_ENERGY_MONTH_UNITS: Final = frozenset({"kwh/mo", "kwh/month", "kwh"})
_IRRAD_MONTH_UNITS: Final = frozenset({"kwh/m2/mo", "kwh/m2/month"})
_LOSS_UNITS: Final = frozenset({"%", "percent"})

_TECH_TO_PVGIS: Final[dict[str, str]] = {
    ModuleTechnology.MONO_CSI.value: "crystSi",
    ModuleTechnology.POLY_CSI.value: "crystSi",
    ModuleTechnology.PERC.value: "crystSi",
    ModuleTechnology.TOPCON.value: "crystSi",
    ModuleTechnology.HJT.value: "crystSi",
    ModuleTechnology.CDTE.value: "CdTe",
    ModuleTechnology.CIGS.value: "CIS",
    "crystSi": "crystSi",
    "CIS": "CIS",
    "CdTe": "CdTe",
    "Unknown": "Unknown",
}

_MOUNTINGPLACE: Final[dict[MountingType, str]] = {
    MountingType.ROOFTOP: "building",
    MountingType.GROUND: "free",
}

# PVGIS PVcalc JSON keys documented in the 5.3 meta.choices string.
_TRACKINGTYPE_OUTPUT_KEY: Final[dict[int, str]] = {
    0: "fixed",
    2: "two_axis",
    3: "vertical_axis",
    5: "inclined_axis",
}

_SINGLE_AXIS_DOCUMENTED: Final = frozenset({3, 5})


@dataclass(frozen=True)
class _QueryPlan:
    params: dict[str, JsonScalar]
    output_key: str
    notes: tuple[str, ...]
    incomplete_reason: str | None
    pvgis_aspect: float | None
    pvgis_angle: float | None
    pvgis_trackingtype: int | None
    pvgis_pvtechchoice: str | None
    pvgis_mountingplace: str | None


def azimuth_north_to_pvgis_aspect(azimuth_deg: float) -> float:
    """Convert SolarDSS azimuth (clockwise from north) to PVGIS aspect.

    PVGIS: 0 = south, 90 = west, -90 = east. North is -180.
    """

    wrapped = azimuth_deg % 360.0
    return wrapped - 180.0


def pvgis_pvtechchoice(technology: str | ModuleTechnology | None) -> tuple[str | None, str | None]:
    """Map a SolarDSS technology class to PVGIS ``pvtechchoice`` when supported.

    Unsupported classes are not rewritten as ``Unknown`` or ``crystSi``. The
    second item is a mapping note, or a reason the value cannot be sent.
    """

    if technology is None:
        return None, None
    raw = technology.value if isinstance(technology, ModuleTechnology) else technology
    mapped = _TECH_TO_PVGIS.get(raw)
    if mapped is None:
        return None, (
            f"PVGIS pvtechchoice does not support technology {raw!r}; "
            "SolarDSS will not send a substitute"
        )
    if raw == mapped:
        return mapped, None
    return mapped, f"mapped module technology {raw!r} to PVGIS pvtechchoice {mapped!r}"


def yield_request_from_configuration(
    configuration: PVSystemConfiguration,
    *,
    system_loss_percent: float,
    technology: str | ModuleTechnology | None = None,
    raddatabase: str | None = None,
    usehorizon: bool | None = None,
    pvgis_trackingtype: int | None = None,
) -> PVYieldRequest:
    """Copy geometry and capacity from an evaluated configuration into a yield request.

    ``system_loss_percent`` remains a caller assumption. It is not taken from
    the catalog or from a 14% default.
    """

    capacity_kw = None
    if configuration.installed_capacity_w is not None:
        capacity_kw = configuration.installed_capacity_w / 1000.0
    return PVYieldRequest(
        location=configuration.location,
        installed_capacity_kw=capacity_kw,
        tilt_deg=configuration.tilt_deg,
        azimuth_deg=configuration.azimuth_deg,
        mounting=configuration.mounting_type,
        tracking=configuration.tracking_mode,
        technology=(
            None
            if technology is None
            else (technology.value if isinstance(technology, ModuleTechnology) else technology)
        ),
        system_loss_percent=system_loss_percent,
        raddatabase=raddatabase,
        usehorizon=usehorizon,
        pvgis_trackingtype=pvgis_trackingtype,
        notes=configuration.notes,
    )


class PVGISYieldEngine:
    """Evaluate annual PV yield with official PVGIS 5.3 ``PVcalc``.

    Distinct from ``PVGISSolarResourceProvider`` (``MRcalc``) and from the
    indicator-mapping ``PVGISYieldProvider``. Requests are cached on
    ``PVGISClient``.
    """

    def __init__(
        self,
        *,
        client: PVGISClient | None = None,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        base_url: str = PVGIS_DEFAULT_BASE_URL,
        transport: object | None = None,
        http_client: object | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._clock: Clock = UtcClock() if clock is None else clock
        self._owns_client = client is None
        if client is not None:
            self._pvgis = client
            return
        resolved_cache: DatasetCache = MemoryCache() if cache is None else cache
        self._pvgis = PVGISClient(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            cache=resolved_cache,
            clock=self._clock,
            client=http_client,  # type: ignore[arg-type]
            transport=transport,  # type: ignore[arg-type]
            sleep=sleep,
        )

    @property
    def client(self) -> PVGISClient:
        return self._pvgis

    def close(self) -> None:
        if self._owns_client:
            self._pvgis.close()

    def __enter__(self) -> PVGISYieldEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def evaluate(self, request: PVYieldRequest) -> PVYieldResult:
        """Run PVcalc for ``request``. Missing PVGIS fields stay missing."""

        loss_assumptions = _requested_loss(request)
        incomplete = _incomplete_reason(request)
        if incomplete is not None:
            return self._result(
                request,
                status=EvaluationStatus.INCOMPLETE_INPUTS,
                warnings=(incomplete,),
                loss_assumptions=loss_assumptions,
                metadata=_metadata(),
            )

        plan = _query_plan(request)
        if plan.incomplete_reason is not None:
            return self._result(
                request,
                status=EvaluationStatus.INCOMPLETE_INPUTS,
                warnings=_unique_messages((plan.incomplete_reason, *plan.notes)),
                loss_assumptions=loss_assumptions,
                metadata=_metadata(
                    pvgis_pvtechchoice=plan.pvgis_pvtechchoice,
                    pvgis_mountingplace=plan.pvgis_mountingplace,
                    pvgis_aspect=plan.pvgis_aspect,
                    pvgis_angle=plan.pvgis_angle,
                    pvgis_trackingtype=plan.pvgis_trackingtype,
                    output_mounting_key=plan.output_key or None,
                ),
            )

        try:
            raw = self._pvgis.get_json("PVcalc", plan.params)
        except PVGISHTTPError as exc:
            return self._result(
                request,
                status=EvaluationStatus.FAILED,
                warnings=(exc.message,),
                loss_assumptions=loss_assumptions,
                metadata=_metadata(
                    source_url=self._pvgis.base_url,
                    pvgis_pvtechchoice=plan.pvgis_pvtechchoice,
                    pvgis_mountingplace=plan.pvgis_mountingplace,
                    pvgis_aspect=plan.pvgis_aspect,
                    pvgis_angle=plan.pvgis_angle,
                    pvgis_trackingtype=plan.pvgis_trackingtype,
                    output_mounting_key=plan.output_key,
                ),
                extra_parameters=_query_parameters(plan),
            )

        return self._from_http(request, plan, raw, loss_assumptions)

    def evaluate_from_configuration(
        self,
        configuration: PVSystemConfiguration,
        *,
        system_loss_percent: float,
        technology: str | ModuleTechnology | None = None,
        raddatabase: str | None = None,
        usehorizon: bool | None = None,
        pvgis_trackingtype: int | None = None,
    ) -> PVYieldResult:
        """Evaluate yield for an already-resolved PV configuration."""

        return self.evaluate(
            yield_request_from_configuration(
                configuration,
                system_loss_percent=system_loss_percent,
                technology=technology,
                raddatabase=raddatabase,
                usehorizon=usehorizon,
                pvgis_trackingtype=pvgis_trackingtype,
            )
        )

    def _from_http(
        self,
        request: PVYieldRequest,
        plan: _QueryPlan,
        raw: PVGISHttpResult,
        loss_assumptions: PVLossAssumptions,
    ) -> PVYieldResult:
        metadata = _metadata(
            source_url=raw.source_url,
            from_cache=raw.from_cache,
            status_code=raw.status_code,
            retrieved_at=raw.retrieved_at,
            pvgis_pvtechchoice=plan.pvgis_pvtechchoice,
            pvgis_mountingplace=plan.pvgis_mountingplace,
            pvgis_aspect=plan.pvgis_aspect,
            pvgis_angle=plan.pvgis_angle,
            pvgis_trackingtype=plan.pvgis_trackingtype,
            output_mounting_key=plan.output_key,
        )
        if raw.payload is None:
            reason = raw.message or f"PVGIS HTTP {raw.status_code}"
            return self._result(
                request,
                status=EvaluationStatus.FAILED,
                warnings=(reason, *plan.notes),
                loss_assumptions=loss_assumptions,
                metadata=metadata,
                extra_parameters=_query_parameters(plan),
            )

        location = parse_location(raw.payload)
        meteo = parse_meteo(raw.payload)
        metadata = metadata.model_copy(
            update={
                "radiation_db": None if meteo is None else meteo.radiation_db,
                "meteo_db": None if meteo is None else meteo.meteo_db,
                "year_min": None if meteo is None else meteo.year_min,
                "year_max": None if meteo is None else meteo.year_max,
                "elevation_m": None if location is None else location.elevation,
            }
        )
        warnings = list(plan.notes)
        totals: PVGISPVTotals | None = None
        monthly_rows: tuple[PVGISPVMonthlyRow, ...] = ()
        try:
            totals = parse_pv_totals(raw.payload, plan.output_key)
        except PVGISResponseError as exc:
            warnings.append(str(exc))
        try:
            monthly_rows = parse_pv_monthly(raw.payload, plan.output_key)
        except PVGISResponseError as exc:
            warnings.append(str(exc))
        if totals is None and not monthly_rows:
            return self._result(
                request,
                status=EvaluationStatus.FAILED,
                warnings=tuple(warnings) if warnings else ("PVcalc output could not be parsed",),
                loss_assumptions=loss_assumptions,
                metadata=metadata,
                extra_parameters=_query_parameters(plan),
            )

        meta = raw.payload.meta
        monthly_energy, monthly_irrad, month_warn = _monthly_series(monthly_rows, meta)
        warnings.extend(month_warn)
        annual_energy, energy_warn = _annual_energy(totals, monthly_energy, meta)
        warnings.extend(energy_warn)
        annual_irrad, irrad_warn = _annual_irradiation(totals, monthly_irrad, meta)
        warnings.extend(irrad_warn)
        specific = _specific_yield(annual_energy, request.installed_capacity_kw)
        losses = _merge_losses(loss_assumptions, totals, meta)
        return self._result(
            request,
            status=EvaluationStatus.EVALUATED,
            warnings=tuple(warnings),
            loss_assumptions=losses,
            metadata=metadata,
            annual_energy_kwh=annual_energy,
            specific_yield_kwh_per_kwp=specific,
            monthly_energy_kwh=monthly_energy,
            monthly_irradiation=monthly_irrad,
            annual_irradiation=annual_irrad,
            extra_parameters=_query_parameters(plan),
            extra_notes=_origin_notes(annual_energy, specific, annual_irrad),
            meteo=meteo,
        )

    def _result(
        self,
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
        extra_parameters: Mapping[str, JsonScalar] | None = None,
        extra_notes: Sequence[str] = (),
        meteo: PVGISMeteoData | None = None,
    ) -> PVYieldResult:
        lat, lon = _point(request.location)
        parameters: dict[str, JsonScalar] = {
            "lat": lat,
            "lon": lon,
            "installed_capacity_kw": request.installed_capacity_kw,
            "tilt_deg": request.tilt_deg,
            "azimuth_deg": request.azimuth_deg,
            "mounting": request.mounting.value,
            "tracking": request.tracking.value,
            "technology": request.technology,
            "system_loss_percent": request.system_loss_percent,
            "pvgis_tool": "PVcalc",
            "from_cache": metadata.from_cache,
            "annual_energy_origin": None
            if annual_energy_kwh is None
            else annual_energy_kwh.origin.value,
            "specific_yield_origin": None
            if specific_yield_kwh_per_kwp is None
            else specific_yield_kwh_per_kwp.origin.value,
            "annual_irradiation_origin": None
            if annual_irradiation is None
            else annual_irradiation.origin.value,
        }
        if extra_parameters:
            parameters.update(extra_parameters)
        notes = [
            "annual generation is PVGIS PVcalc, not MRcalc and not an LLM estimate",
            *request.notes,
            *extra_notes,
            *warnings,
        ]
        if meteo is not None and meteo.radiation_db:
            notes.append(f"radiation_db={meteo.radiation_db}")
        fingerprints = (
            fingerprint_canonical(
                {
                    "lat": lat,
                    "lon": lon,
                    "peakpower": request.installed_capacity_kw,
                    "loss": request.system_loss_percent,
                    "tilt": request.tilt_deg,
                    "azimuth": request.azimuth_deg,
                    "mounting": request.mounting.value,
                    "tracking": request.tracking.value,
                    "technology": request.technology,
                    "raddatabase": request.raddatabase,
                    "usehorizon": request.usehorizon,
                    "pvgis_trackingtype": request.pvgis_trackingtype,
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
            loss_assumptions=loss_assumptions,
            provider_metadata=metadata,
            provenance=provenance_record(
                MethodId.PVGIS_PV_YIELD,
                parameters=parameters,
                input_fingerprints=fingerprints,
                notes=tuple(notes),
            ),
            warnings=warnings,
        )


def _unique_messages(messages: Sequence[str | None]) -> tuple[str, ...]:
    ordered: list[str] = []
    for item in messages:
        if item and item not in ordered:
            ordered.append(item)
    return tuple(ordered)


def _point(location: GeographicRef) -> tuple[float | None, float | None]:
    return location.latitude, location.longitude


def _incomplete_reason(request: PVYieldRequest) -> str | None:
    lat, lon = _point(request.location)
    if lat is None or lon is None:
        return "PVGIS PVcalc requires a point latitude and longitude"
    if request.installed_capacity_kw is None:
        return "PVGIS PVcalc requires installed_capacity_kw (peakpower)"
    if request.system_loss_percent is None:
        return "PVGIS PVcalc requires system_loss_percent; SolarDSS does not default 14%"
    return None


def _query_plan(request: PVYieldRequest) -> _QueryPlan:
    notes: list[str] = []
    trackingtype, output_key, tracking_reason = _tracking_plan(request)
    tech, tech_note = pvgis_pvtechchoice(request.technology)
    if tech is None and request.technology is not None:
        return _QueryPlan(
            params={},
            output_key=output_key,
            notes=(),
            incomplete_reason=tech_note
            or (
                f"PVGIS pvtechchoice does not support technology {request.technology!r}; "
                "SolarDSS will not send a substitute"
            ),
            pvgis_aspect=None,
            pvgis_angle=None,
            pvgis_trackingtype=trackingtype,
            pvgis_pvtechchoice=None,
            pvgis_mountingplace=_MOUNTINGPLACE[request.mounting],
        )
    if tech_note is not None:
        notes.append(tech_note)
    if tracking_reason is not None:
        return _QueryPlan(
            params={},
            output_key=output_key,
            notes=tuple(notes),
            incomplete_reason=tracking_reason,
            pvgis_aspect=None,
            pvgis_angle=None,
            pvgis_trackingtype=trackingtype,
            pvgis_pvtechchoice=tech,
            pvgis_mountingplace=_MOUNTINGPLACE[request.mounting],
        )

    two_axis = output_key == "two_axis" or trackingtype == 2
    angle: float | None = None
    aspect: float | None = None
    if two_axis:
        if request.tilt_deg is not None or request.azimuth_deg is not None:
            notes.append("tilt_deg and azimuth_deg are not sent to PVGIS for two-axis tracking")
    else:
        if request.tilt_deg is None or request.azimuth_deg is None:
            return _QueryPlan(
                params={},
                output_key=output_key,
                notes=tuple(notes),
                incomplete_reason=(
                    "fixed and single-axis PVcalc requests require tilt_deg and azimuth_deg; "
                    "SolarDSS does not invent PVGIS slope or aspect"
                ),
                pvgis_aspect=None,
                pvgis_angle=None,
                pvgis_trackingtype=trackingtype,
                pvgis_pvtechchoice=tech,
                pvgis_mountingplace=_MOUNTINGPLACE[request.mounting],
            )
        angle = request.tilt_deg
        aspect = azimuth_north_to_pvgis_aspect(request.azimuth_deg)
        notes.append(
            "converted SolarDSS azimuth (clockwise from north) to PVGIS aspect "
            f"(0=south): {request.azimuth_deg} -> {aspect}"
        )

    lat, lon = _point(request.location)
    assert lat is not None and lon is not None
    assert request.installed_capacity_kw is not None
    assert request.system_loss_percent is not None
    mountingplace = _MOUNTINGPLACE[request.mounting]
    params: dict[str, JsonScalar] = {
        "lat": lat,
        "lon": lon,
        "peakpower": request.installed_capacity_kw,
        "loss": request.system_loss_percent,
        "mountingplace": mountingplace,
    }
    if tech is not None:
        params["pvtechchoice"] = tech
    else:
        notes.append("pvtechchoice omitted; PVGIS applies its default module model")
    if angle is not None:
        params["angle"] = angle
    if aspect is not None:
        params["aspect"] = aspect
    if trackingtype is not None:
        params["trackingtype"] = trackingtype
    if request.raddatabase is not None:
        params["raddatabase"] = request.raddatabase.strip()
    if request.usehorizon is not None:
        params["usehorizon"] = request.usehorizon
    return _QueryPlan(
        params=params,
        output_key=output_key,
        notes=tuple(notes),
        incomplete_reason=None,
        pvgis_aspect=aspect,
        pvgis_angle=angle,
        pvgis_trackingtype=trackingtype,
        pvgis_pvtechchoice=tech,
        pvgis_mountingplace=mountingplace,
    )


def _tracking_plan(request: PVYieldRequest) -> tuple[int | None, str, str | None]:
    if request.pvgis_trackingtype is not None:
        key = _TRACKINGTYPE_OUTPUT_KEY.get(request.pvgis_trackingtype)
        if key is None:
            return (
                request.pvgis_trackingtype,
                "",
                (
                    f"PVGIS trackingtype {request.pvgis_trackingtype} has no documented "
                    "PVcalc output key in SolarDSS; the axis type is not inferred"
                ),
            )
        return request.pvgis_trackingtype, key, None
    if request.tracking is TrackingMode.FIXED:
        return None, "fixed", None
    if request.tracking is TrackingMode.DUAL_AXIS:
        return 2, "two_axis", None
    return (
        None,
        "",
        (
            "PVGIS single-axis trackingtype is not implied; supply pvgis_trackingtype "
            "(3=vertical_axis, 5=inclined_axis). Types 1 and 4 are not mapped."
        ),
    )


def _query_parameters(plan: _QueryPlan) -> dict[str, JsonScalar]:
    return {
        "pvgis_pvtechchoice": plan.pvgis_pvtechchoice,
        "pvgis_mountingplace": plan.pvgis_mountingplace,
        "pvgis_aspect": plan.pvgis_aspect,
        "pvgis_angle": plan.pvgis_angle,
        "pvgis_trackingtype": plan.pvgis_trackingtype,
        "pvgis_output_key": plan.output_key or None,
    }


def _metadata(**kwargs: Any) -> PVYieldEngineMetadata:
    return PVYieldEngineMetadata(
        provider_id="pvgis.pv_yield",
        api_version=PVGIS_API_VERSION,
        source_name=PVGIS_SOURCE_NAME,
        tool="PVcalc",
        **kwargs,
    )


def _requested_loss(request: PVYieldRequest) -> PVLossAssumptions:
    requested = None
    if request.system_loss_percent is not None:
        requested = TaggedYieldValue(
            value=request.system_loss_percent,
            unit="%",
            origin=YieldValueOrigin.CALLER_SUPPLIED,
            source_field="loss",
            notes=("caller system-loss assumption sent as PVcalc loss",),
        )
    return PVLossAssumptions(requested_system_loss_percent=requested)


def _tagged_provider(
    value: float | None,
    *,
    unit: str,
    source_field: str,
    allowed: frozenset[str],
    meta: Mapping[str, Any] | None,
    section: str,
) -> tuple[TaggedYieldValue | None, str | None]:
    if value is None:
        return None, None
    if not _units_ok(meta, section, source_field, allowed):
        return None, (
            f"PVcalc {source_field} is present but uses an unexpected unit; value omitted"
        )
    return (
        TaggedYieldValue(
            value=value,
            unit=unit,
            origin=YieldValueOrigin.PVGIS_PROVIDER,
            source_field=source_field,
        ),
        None,
    )


def _monthly_series(
    rows: Sequence[PVGISPVMonthlyRow],
    meta: Mapping[str, Any] | None,
) -> tuple[tuple[MonthlyYieldValue, ...], tuple[MonthlyYieldValue, ...], list[str]]:
    energy: list[MonthlyYieldValue] = []
    irrad: list[MonthlyYieldValue] = []
    warnings: list[str] = []
    for row in rows:
        tagged_e, warn_e = _tagged_provider(
            row.e_m,
            unit="kWh",
            source_field="E_m",
            allowed=_ENERGY_MONTH_UNITS,
            meta=meta,
            section="monthly",
        )
        tagged_h, warn_h = _tagged_provider(
            row.h_i_m,
            unit="kWh/m2",
            source_field="H(i)_m",
            allowed=_IRRAD_MONTH_UNITS,
            meta=meta,
            section="monthly",
        )
        if warn_e:
            warnings.append(f"month {row.month}: {warn_e}")
        if warn_h:
            warnings.append(f"month {row.month}: {warn_h}")
        if tagged_e is not None:
            energy.append(MonthlyYieldValue(month=row.month, value=tagged_e))
        if tagged_h is not None:
            irrad.append(MonthlyYieldValue(month=row.month, value=tagged_h))
    return tuple(energy), tuple(irrad), warnings


def _annual_energy(
    totals: PVGISPVTotals | None,
    monthly: Sequence[MonthlyYieldValue],
    meta: Mapping[str, Any] | None,
) -> tuple[TaggedYieldValue | None, list[str]]:
    warnings: list[str] = []
    if totals is not None:
        tagged, warn = _tagged_provider(
            totals.e_y,
            unit="kWh",
            source_field="E_y",
            allowed=_ENERGY_YEAR_UNITS,
            meta=meta,
            section="totals",
        )
        if warn:
            warnings.append(warn)
        if tagged is not None:
            return tagged, warnings
    summed = _complete_year_sum(monthly)
    if summed is None:
        return None, warnings
    warnings.append("E_y missing; annual energy is the complete-year sum of E_m")
    return (
        TaggedYieldValue(
            value=summed,
            unit="kWh",
            origin=YieldValueOrigin.SOLARDSS_DERIVED,
            source_field="E_m",
            calculation_method=_CALC_SUM_ENERGY,
            notes=("sum of twelve monthly E_m values; not a PVGIS E_y field",),
        ),
        warnings,
    )


def _annual_irradiation(
    totals: PVGISPVTotals | None,
    monthly: Sequence[MonthlyYieldValue],
    meta: Mapping[str, Any] | None,
) -> tuple[TaggedYieldValue | None, list[str]]:
    warnings: list[str] = []
    if totals is not None:
        tagged, warn = _tagged_provider(
            totals.h_i_y,
            unit="kWh/m2",
            source_field="H(i)_y",
            allowed=_IRRAD_YEAR_UNITS,
            meta=meta,
            section="totals",
        )
        if warn:
            warnings.append(warn)
        if tagged is not None:
            return tagged, warnings
    summed = _complete_year_sum(monthly)
    if summed is None:
        return None, warnings
    warnings.append("H(i)_y missing; annual irradiation is the complete-year sum of H(i)_m")
    return (
        TaggedYieldValue(
            value=summed,
            unit="kWh/m2",
            origin=YieldValueOrigin.SOLARDSS_DERIVED,
            source_field="H(i)_m",
            calculation_method=_CALC_SUM_IRRAD,
            notes=("sum of twelve monthly H(i)_m values; not a PVGIS H(i)_y field",),
        ),
        warnings,
    )


def _specific_yield(
    annual_energy: TaggedYieldValue | None,
    peakpower_kw: float | None,
) -> TaggedYieldValue | None:
    if annual_energy is None or peakpower_kw is None or peakpower_kw == 0.0:
        return None
    notes = (
        "identity only; PVGIS does not publish specific yield as a separate field",
        f"annual energy origin={annual_energy.origin.value}",
    )
    return TaggedYieldValue(
        value=annual_energy.value / peakpower_kw,
        unit="kWh/kWp",
        origin=YieldValueOrigin.SOLARDSS_DERIVED,
        calculation_method=_CALC_SPECIFIC_YIELD,
        notes=notes,
    )


def _merge_losses(
    base: PVLossAssumptions,
    totals: PVGISPVTotals | None,
    meta: Mapping[str, Any] | None,
) -> PVLossAssumptions:
    if totals is None:
        return base
    aoi, _ = _tagged_provider(
        totals.l_aoi,
        unit="%",
        source_field="l_aoi",
        allowed=_LOSS_UNITS,
        meta=meta,
        section="totals",
    )
    spec, _ = _tagged_provider(
        totals.l_spec,
        unit="%",
        source_field="l_spec",
        allowed=_LOSS_UNITS,
        meta=meta,
        section="totals",
    )
    tg, _ = _tagged_provider(
        totals.l_tg, unit="%", source_field="l_tg", allowed=_LOSS_UNITS, meta=meta, section="totals"
    )
    total, _ = _tagged_provider(
        totals.l_total,
        unit="%",
        source_field="l_total",
        allowed=_LOSS_UNITS,
        meta=meta,
        section="totals",
    )
    return PVLossAssumptions(
        requested_system_loss_percent=base.requested_system_loss_percent,
        pvgis_aoi_loss_percent=aoi,
        pvgis_spectral_loss_percent=spec,
        pvgis_temperature_irradiance_loss_percent=tg,
        pvgis_total_loss_percent=total,
        notes=("PVGIS l_* values are provider-calculated; they are not a SolarDSS loss model",),
    )


def _complete_year_sum(points: Sequence[MonthlyYieldValue]) -> float | None:
    by_month: dict[int, float] = {}
    for point in points:
        if point.month in by_month:
            return None
        by_month[point.month] = point.value.value
    if set(by_month) != set(range(1, 13)):
        return None
    return sum(by_month[month] for month in range(1, 13))


def _origin_notes(
    annual_energy: TaggedYieldValue | None,
    specific: TaggedYieldValue | None,
    annual_irrad: TaggedYieldValue | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if annual_energy is not None:
        notes.append(f"annual_energy_kwh origin={annual_energy.origin.value}")
    if specific is not None:
        notes.append(f"specific_yield_kwh_per_kwp origin={specific.origin.value}")
    if annual_irrad is not None:
        notes.append(f"annual_irradiation origin={annual_irrad.origin.value}")
    return tuple(notes)


def _units_ok(
    meta: Mapping[str, Any] | None,
    section: str,
    field: str,
    allowed: frozenset[str],
) -> bool:
    units = _meta_units(meta, section, field)
    if units is None:
        return True
    return units.strip().lower().replace(" ", "") in allowed


def _meta_units(meta: Mapping[str, Any] | None, section: str, field: str) -> str | None:
    if meta is None:
        return None
    outputs = meta.get("outputs")
    if not isinstance(outputs, Mapping):
        return None
    block = outputs.get(section)
    if not isinstance(block, Mapping):
        return None
    variables = block.get("variables")
    if not isinstance(variables, Mapping):
        return None
    spec = variables.get(field)
    if not isinstance(spec, Mapping):
        return None
    units = spec.get("units")
    if not isinstance(units, str):
        return None
    return units
