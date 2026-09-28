"""Official PVGIS 5.3 HTTP adapter.

Field names and units follow the public JSON contract of ``MRcalc`` and
``PVcalc`` (PVGIS 5.3, ``https://re.jrc.ec.europa.eu/api/v5_3``). Unknown or
absent fields stay absent. SolarDSS never fills them with defaults.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import suppress
from datetime import date, datetime
from typing import Annotated, Any, Final, Literal
from urllib.parse import urlsplit, urlunsplit

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, ValidationError

from solardss.core.hashing import fingerprint_canonical
from solardss.core.jsonutil import JsonScalar, JsonValue
from solardss.data.contracts import DatasetRef
from solardss.exceptions import InvalidParameterError
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
from solardss.providers.errors import PVGISHTTPError, PVGISResponseError
from solardss.providers.region import request_point
from solardss.providers.request import FetchRequest, request_cache_key
from solardss.version import __version__

PVGIS_API_VERSION: Final = "v5_3"
PVGIS_DEFAULT_BASE_URL: Final = "https://re.jrc.ec.europa.eu/api/v5_3"
PVGIS_SOURCE_NAME: Final = "PVGIS"
PVGIS_USER_AGENT: Final = f"solardss/{__version__} (PVGIS {PVGIS_API_VERSION} client)"

_RETRY_STATUS: Final = frozenset({429, 500, 502, 503, 504, 529})
_GHI_MONTH_UNITS: Final = frozenset({"kwh/m2/mo", "kwh/m2/month"})
_DNI_MONTH_UNITS: Final = frozenset({"kwh/m2/mo", "kwh/m2/month"})
_KD_UNITS: Final = frozenset({"-", "1", ""})
_TEMP_UNITS: Final = frozenset({"degreecelsius", "degc", "c", "celsius", "°c"})
_ENERGY_YEAR_UNITS: Final = frozenset({"kwh/y", "kwh/year", "kwh"})

_RESOURCE_INDICATORS: Final = frozenset(
    {
        IndicatorId.SOLAR_GHI_ANNUAL.value,
        IndicatorId.SOLAR_DNI_ANNUAL.value,
        IndicatorId.SOLAR_DHI_ANNUAL.value,
        IndicatorId.SOLAR_MONTHLY_VARIABILITY.value,
    }
)
_YIELD_INDICATORS: Final = frozenset(
    {
        IndicatorId.SOLAR_PVOUT.value,
        IndicatorId.SOLAR_SPECIFIC_YIELD.value,
        IndicatorId.SOLAR_MONTHLY_VARIABILITY.value,
    }
)

_CALC_ANNUAL_SUM = "pvgis_mrcalc_mean_of_complete_year_sums"
_CALC_DHI = "pvgis_mrcalc_Kd_times_H(h)_m_mean_of_complete_year_sums"
_CALC_YIELD = "pvgis_pvcalc_E_y_per_peak_power"

ToolName = Literal["MRcalc", "PVcalc"]


def _coerce_optional_float(value: object) -> float | None:
    if value is None or value == "" or value == "-":
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a PVGIS numeric field")
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        number = float(value.strip())
    else:
        raise ValueError(f"unsupported PVGIS numeric type {type(value).__name__}")
    if number != number or number in (float("inf"), float("-inf")):
        raise ValueError("non-finite PVGIS numeric field")
    return number


OptionalNumber = Annotated[float | None, BeforeValidator(_coerce_optional_float)]


def _norm_unit(raw: str) -> str:
    return raw.strip().lower().replace(" ", "")


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


def _units_allowed(
    meta: Mapping[str, Any] | None,
    section: str,
    field: str,
    allowed: frozenset[str],
) -> bool:
    units = _meta_units(meta, section, field)
    if units is None:
        return True
    return _norm_unit(units) in allowed


class PVGISLocation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    latitude: float
    longitude: float
    elevation: OptionalNumber = None


class PVGISMeteoData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore")

    radiation_db: str | None = None
    meteo_db: str | None = None
    year_min: int | None = None
    year_max: int | None = None
    use_horizon: bool | None = None
    horizon_db: str | None = None
    horizon_data: str | None = None


class PVGISMonthlyRadiationRow(BaseModel):
    """One ``MRcalc`` monthly record. Absent keys stay ``None``."""

    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    year: int
    month: int
    h_h_m: OptionalNumber = Field(default=None, alias="H(h)_m")
    hb_n_m: OptionalNumber = Field(default=None, alias="Hb(n)_m")
    kd: OptionalNumber = Field(default=None, alias="Kd")
    t2m: OptionalNumber = Field(default=None, alias="T2m")
    h_i_opt_m: OptionalNumber = Field(default=None, alias="H(i_opt)_m")
    h_i_m: OptionalNumber = Field(default=None, alias="H(i)_m")


class PVGISPVMonthlyRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    month: int
    e_d: OptionalNumber = Field(default=None, alias="E_d")
    e_m: OptionalNumber = Field(default=None, alias="E_m")
    h_i_d: OptionalNumber = Field(default=None, alias="H(i)_d")
    h_i_m: OptionalNumber = Field(default=None, alias="H(i)_m")
    sd_m: OptionalNumber = Field(default=None, alias="SD_m")


class PVGISPVTotals(BaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", populate_by_name=True)

    e_d: OptionalNumber = Field(default=None, alias="E_d")
    e_m: OptionalNumber = Field(default=None, alias="E_m")
    e_y: OptionalNumber = Field(default=None, alias="E_y")
    h_i_d: OptionalNumber = Field(default=None, alias="H(i)_d")
    h_i_m: OptionalNumber = Field(default=None, alias="H(i)_m")
    h_i_y: OptionalNumber = Field(default=None, alias="H(i)_y")
    sd_m: OptionalNumber = Field(default=None, alias="SD_m")
    sd_y: OptionalNumber = Field(default=None, alias="SD_y")
    l_aoi: OptionalNumber = Field(default=None, alias="l_aoi")
    l_spec: OptionalNumber = Field(default=None, alias="l_spec")
    l_tg: OptionalNumber = Field(default=None, alias="l_tg")
    l_total: OptionalNumber = Field(default=None, alias="l_total")


class PVGISRawDocument(BaseModel):
    """Validated PVGIS JSON envelope. Nested bodies are not interpreted here."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    inputs: dict[str, Any]
    outputs: dict[str, Any]
    meta: dict[str, Any] | None = None


class PVGISHttpResult(BaseModel):
    """Raw HTTP outcome. Indicator mapping happens after this object exists."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool: ToolName
    status_code: int
    source_url: str
    from_cache: bool
    payload: PVGISRawDocument | None = None
    message: str | None = None
    retrieved_at: datetime | None = None


class PVGISMonthlyRadiation(BaseModel):
    """Normalized monthly radiation row plus unmapped PVGIS meteorology."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    year: int
    month: int
    ghi_kwh_m2: float | None
    dni_kwh_m2: float | None
    dhi_kwh_m2: float | None
    kd: float | None
    t2m_deg_c: float | None
    poa_kwh_m2: float | None
    poa_optimal_kwh_m2: float | None
    present_fields: tuple[str, ...]


class PVGISMonthlyYield(BaseModel):
    """Normalized monthly PVGIS yield row. ``SD_m`` is kept in kWh, not remapped."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    month: int
    energy_kwh_day: float | None
    energy_kwh_month: float | None
    poa_kwh_m2_day: float | None
    poa_kwh_m2_month: float | None
    monthly_energy_sd_kwh: float | None
    present_fields: tuple[str, ...]


class PVGISResourceResult(BaseModel):
    """Raw ``MRcalc`` document plus monthly series and mapped annual indicators."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: PVGISHttpResult
    location: PVGISLocation | None
    meteo: PVGISMeteoData | None
    monthly: tuple[PVGISMonthlyRadiation, ...]
    observations: tuple[IndicatorObservation, ...]


class PVGISYieldResult(BaseModel):
    """Raw ``PVcalc`` document plus monthly series and mapped yield indicators."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw: PVGISHttpResult
    location: PVGISLocation | None
    meteo: PVGISMeteoData | None
    mounting: str
    peak_power_kw: float | None
    system_loss_percent: float | None
    monthly: tuple[PVGISMonthlyYield, ...]
    totals: PVGISPVTotals | None
    observations: tuple[IndicatorObservation, ...]


def public_source_url(url: str) -> str:
    """Drop userinfo from a URL. PVGIS has no API key; this keeps the contract."""

    parts = urlsplit(url)
    netloc = parts.hostname or ""
    if parts.port:
        netloc = f"{netloc}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))


def _canonical_query(params: Mapping[str, JsonScalar]) -> dict[str, str | int | float]:
    query: dict[str, str | int | float] = {}
    for key in sorted(params):
        value = params[key]
        if value is None:
            continue
        if isinstance(value, bool):
            query[key] = 1 if value else 0
        elif isinstance(value, int | float | str):
            query[key] = value
        else:
            raise InvalidParameterError(f"unsupported PVGIS query value for {key}")
    return query


class PVGISClient:
    """GET-only PVGIS 5.3 client with timeout, bounded retries, and raw caching."""

    def __init__(
        self,
        *,
        base_url: str = PVGIS_DEFAULT_BASE_URL,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        client: httpx.Client | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] | None = None,
        user_agent: str = PVGIS_USER_AGENT,
    ) -> None:
        if timeout_seconds <= 0:
            raise InvalidParameterError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise InvalidParameterError("max_attempts must be >= 1")
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout_seconds
        self._max_attempts = max_attempts
        self._cache = cache
        self._clock: Clock = UtcClock() if clock is None else clock
        self._sleep = sleep
        self._user_agent = user_agent
        self._owns_client = client is None
        if client is not None:
            self._client = client
        else:
            self._client = httpx.Client(
                timeout=httpx.Timeout(timeout_seconds),
                headers={"User-Agent": user_agent, "Accept": "application/json"},
                transport=transport,
                follow_redirects=True,
            )

    @property
    def base_url(self) -> str:
        return self._base_url

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> PVGISClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def get_json(self, tool: ToolName, params: Mapping[str, JsonScalar]) -> PVGISHttpResult:
        query = _canonical_query({**params, "outputformat": "json", "browser": 0})
        cache_key = fingerprint_canonical(
            {"base_url": self._base_url, "tool": tool, "query": query}
        )
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return self._result_from_cache(tool, cached)
        retrieved_at = self._clock.now()
        last_status: int | None = None
        last_message = "PVGIS request failed"
        for attempt in range(self._max_attempts):
            try:
                response = self._client.get(f"{self._base_url}/{tool}", params=query)
            except httpx.TimeoutException as exc:
                last_status = None
                last_message = f"PVGIS timeout after {self._timeout}s"
                if attempt + 1 >= self._max_attempts:
                    raise PVGISHTTPError(last_message) from exc
                self._backoff(attempt, retry_after=None)
                continue
            except httpx.TransportError as exc:
                last_status = None
                last_message = "PVGIS transport error"
                if attempt + 1 >= self._max_attempts:
                    raise PVGISHTTPError(last_message) from exc
                self._backoff(attempt, retry_after=None)
                continue
            source_url = public_source_url(str(response.url))
            last_status = response.status_code
            if response.status_code in _RETRY_STATUS:
                last_message = f"PVGIS HTTP {response.status_code}"
                if attempt + 1 >= self._max_attempts:
                    return PVGISHttpResult(
                        tool=tool,
                        status_code=response.status_code,
                        source_url=source_url,
                        from_cache=False,
                        message=last_message,
                        retrieved_at=retrieved_at,
                    )
                self._backoff(attempt, retry_after=response.headers.get("Retry-After"))
                continue
            if response.status_code != 200:
                message = _pvgis_error_message(response)
                return PVGISHttpResult(
                    tool=tool,
                    status_code=response.status_code,
                    source_url=source_url,
                    from_cache=False,
                    message=message,
                    retrieved_at=retrieved_at,
                )
            payload = _parse_envelope(response.content)
            if self._cache is not None:
                self._cache.put(cache_key, _encode_cached(source_url, payload, retrieved_at))
            return PVGISHttpResult(
                tool=tool,
                status_code=200,
                source_url=source_url,
                from_cache=False,
                payload=payload,
                retrieved_at=retrieved_at,
            )
        return PVGISHttpResult(
            tool=tool,
            status_code=last_status or 0,
            source_url=self._base_url,
            from_cache=False,
            message=last_message,
            retrieved_at=retrieved_at,
        )

    def _result_from_cache(self, tool: ToolName, blob: bytes) -> PVGISHttpResult:
        try:
            raw: object = json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PVGISResponseError("cached PVGIS payload is not JSON") from exc
        if not isinstance(raw, dict) or raw.get("schema_id") != "solardss.pvgis_cache.v1":
            raise PVGISResponseError("cached PVGIS payload has unknown schema")
        source_url = raw.get("source_url")
        body = raw.get("payload")
        if not isinstance(source_url, str) or not isinstance(body, dict):
            raise PVGISResponseError("cached PVGIS payload is incomplete")
        payload = PVGISRawDocument.model_validate(body)
        retrieved_raw = raw.get("retrieved_at")
        retrieved_at = None
        if isinstance(retrieved_raw, str):
            parsed = datetime.fromisoformat(retrieved_raw)
            if parsed.tzinfo is not None:
                retrieved_at = parsed
        return PVGISHttpResult(
            tool=tool,
            status_code=200,
            source_url=public_source_url(source_url),
            from_cache=True,
            payload=payload,
            retrieved_at=retrieved_at,
        )

    def _backoff(self, attempt: int, *, retry_after: str | None) -> None:
        if self._sleep is None:
            return
        seconds = 0.25 * (2**attempt)
        if retry_after is not None:
            with suppress(ValueError):
                seconds = max(seconds, float(retry_after))
        self._sleep(min(seconds, 2.0))


def _encode_cached(source_url: str, payload: PVGISRawDocument, retrieved_at: datetime) -> bytes:
    envelope = {
        "schema_id": "solardss.pvgis_cache.v1",
        "source_url": source_url,
        "retrieved_at": retrieved_at.isoformat(),
        "payload": payload.model_dump(mode="json"),
    }
    return json.dumps(envelope, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    )


def _parse_envelope(content: bytes) -> PVGISRawDocument:
    try:
        raw: object = json.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PVGISResponseError("PVGIS response is not JSON") from exc
    try:
        return PVGISRawDocument.model_validate(raw)
    except (ValidationError, TypeError, ValueError) as exc:
        raise PVGISResponseError("PVGIS JSON is missing inputs/outputs") from exc


def _pvgis_error_message(response: httpx.Response) -> str:
    try:
        body: object = response.json()
    except ValueError:
        text = response.text.strip()
        return text or f"PVGIS HTTP {response.status_code}"
    if isinstance(body, dict):
        message = body.get("message")
        if isinstance(message, str) and message:
            return message
    return f"PVGIS HTTP {response.status_code}"


def parse_location(payload: PVGISRawDocument) -> PVGISLocation | None:
    location = payload.inputs.get("location")
    if not isinstance(location, Mapping):
        return None
    try:
        return PVGISLocation.model_validate(location)
    except (ValidationError, TypeError, ValueError):
        return None


def parse_meteo(payload: PVGISRawDocument) -> PVGISMeteoData | None:
    meteo = payload.inputs.get("meteo_data")
    if not isinstance(meteo, Mapping):
        return None
    try:
        return PVGISMeteoData.model_validate(meteo)
    except (ValidationError, TypeError, ValueError):
        return None


def parse_monthly_radiation(payload: PVGISRawDocument) -> tuple[PVGISMonthlyRadiationRow, ...]:
    monthly = payload.outputs.get("monthly")
    if not isinstance(monthly, list):
        raise PVGISResponseError("MRcalc outputs.monthly must be a JSON array")
    rows: list[PVGISMonthlyRadiationRow] = []
    for item in monthly:
        try:
            rows.append(PVGISMonthlyRadiationRow.model_validate(item))
        except (ValidationError, TypeError, ValueError) as exc:
            raise PVGISResponseError("MRcalc monthly row failed validation") from exc
    return tuple(rows)


def parse_pv_monthly(payload: PVGISRawDocument, mounting: str) -> tuple[PVGISPVMonthlyRow, ...]:
    monthly = payload.outputs.get("monthly")
    if not isinstance(monthly, Mapping):
        raise PVGISResponseError("PVcalc outputs.monthly must be a JSON object")
    series = monthly.get(mounting)
    if not isinstance(series, list):
        raise PVGISResponseError(f"PVcalc outputs.monthly.{mounting} is missing")
    rows: list[PVGISPVMonthlyRow] = []
    for item in series:
        try:
            rows.append(PVGISPVMonthlyRow.model_validate(item))
        except (ValidationError, TypeError, ValueError) as exc:
            raise PVGISResponseError("PVcalc monthly row failed validation") from exc
    return tuple(rows)


def parse_pv_totals(payload: PVGISRawDocument, mounting: str) -> PVGISPVTotals:
    totals = payload.outputs.get("totals")
    if not isinstance(totals, Mapping):
        raise PVGISResponseError("PVcalc outputs.totals must be a JSON object")
    block = totals.get(mounting)
    if not isinstance(block, Mapping):
        raise PVGISResponseError(f"PVcalc outputs.totals.{mounting} is missing")
    try:
        return PVGISPVTotals.model_validate(block)
    except (ValidationError, TypeError, ValueError) as exc:
        raise PVGISResponseError("PVcalc totals failed validation") from exc


def normalize_radiation_rows(
    rows: Sequence[PVGISMonthlyRadiationRow],
    *,
    meta: Mapping[str, Any] | None,
) -> tuple[PVGISMonthlyRadiation, ...]:
    ghi_ok = _units_allowed(meta, "monthly", "H(h)_m", _GHI_MONTH_UNITS)
    dni_ok = _units_allowed(meta, "monthly", "Hb(n)_m", _DNI_MONTH_UNITS)
    kd_ok = _units_allowed(meta, "monthly", "Kd", _KD_UNITS)
    t2m_ok = _units_allowed(meta, "monthly", "T2m", _TEMP_UNITS)
    normalized: list[PVGISMonthlyRadiation] = []
    for row in rows:
        present: list[str] = []
        ghi = row.h_h_m if ghi_ok else None
        dni = row.hb_n_m if dni_ok else None
        kd = row.kd if kd_ok else None
        t2m = row.t2m if t2m_ok else None
        if row.h_h_m is not None:
            present.append("H(h)_m")
        if row.hb_n_m is not None:
            present.append("Hb(n)_m")
        if row.kd is not None:
            present.append("Kd")
        if row.t2m is not None:
            present.append("T2m")
        if row.h_i_m is not None:
            present.append("H(i)_m")
        if row.h_i_opt_m is not None:
            present.append("H(i_opt)_m")
        dhi = None
        if ghi is not None and kd is not None:
            dhi = ghi * kd
        normalized.append(
            PVGISMonthlyRadiation(
                year=row.year,
                month=row.month,
                ghi_kwh_m2=ghi,
                dni_kwh_m2=dni,
                dhi_kwh_m2=dhi,
                kd=kd,
                t2m_deg_c=t2m,
                poa_kwh_m2=row.h_i_m,
                poa_optimal_kwh_m2=row.h_i_opt_m,
                present_fields=tuple(present),
            )
        )
    return tuple(normalized)


def mean_complete_year_sums(
    monthly: Sequence[PVGISMonthlyRadiation],
    attr: Literal["ghi_kwh_m2", "dni_kwh_m2", "dhi_kwh_m2"],
) -> float | None:
    """Mean of calendar-year totals. Incomplete years are omitted, not filled."""

    by_year: dict[int, dict[int, float]] = {}
    for row in monthly:
        value = getattr(row, attr)
        if value is None:
            continue
        by_year.setdefault(row.year, {})[row.month] = value
    yearly: list[float] = []
    for months in by_year.values():
        if set(months) != set(range(1, 13)):
            continue
        yearly.append(sum(months[month] for month in range(1, 13)))
    if not yearly:
        return None
    return sum(yearly) / float(len(yearly))


def _point_or_none(request: FetchRequest) -> tuple[float, float] | None:
    point = request_point(request)
    if point is None:
        return None
    lon, lat = point
    if not -90.0 <= lat <= 90.0 or not -180.0 <= lon <= 180.0:
        return None
    return (lon, lat)


def _years_from_request(request: FetchRequest) -> tuple[int | None, int | None]:
    start = request.parameters.get("startyear")
    end = request.parameters.get("endyear")
    start_i = int(start) if isinstance(start, int | float) and not isinstance(start, bool) else None
    end_i = int(end) if isinstance(end, int | float) and not isinstance(end, bool) else None
    if request.temporal is not None:
        if start_i is None and isinstance(request.temporal.start, date):
            start_i = request.temporal.start.year
        if end_i is None and isinstance(request.temporal.end, date):
            end_i = request.temporal.end.year
    return start_i, end_i


def _scalar_param(parameters: Mapping[str, JsonValue], key: str) -> JsonScalar:
    value = parameters.get(key)
    if isinstance(value, list | dict):
        return None
    return value


def _http_status_to_observation(result: PVGISHttpResult) -> ObservationStatus:
    message = (result.message or "").lower()
    if result.status_code in _RETRY_STATUS or result.status_code == 0:
        return ObservationStatus.TEMPORARY_FAILURE
    if result.status_code == 400 and "sea" in message:
        return ObservationStatus.UNSUPPORTED_REGION
    if result.status_code == 400:
        return ObservationStatus.UNSUPPORTED_REGION
    if result.status_code == 404:
        return ObservationStatus.UNAVAILABLE
    return ObservationStatus.TEMPORARY_FAILURE


def _spatial(location: PVGISLocation | None, request: FetchRequest) -> SpatialCoverage:
    point = _point_or_none(request)
    lat = location.latitude if location is not None else (point[1] if point else None)
    lon = location.longitude if location is not None else (point[0] if point else None)
    notes = []
    if lat is not None and lon is not None:
        notes.append(f"lat={lat}")
        notes.append(f"lon={lon}")
    return SpatialCoverage(
        resolution="pvgis-point",
        crs="EPSG:4326",
        geometry_fingerprint=",".join(notes) if notes else None,
    )


def _temporal(
    meteo: PVGISMeteoData | None,
    *,
    resolution: str,
    aggregation: str | None,
) -> TemporalCoverage:
    start = date(meteo.year_min, 1, 1) if meteo is not None and meteo.year_min is not None else None
    end = date(meteo.year_max, 12, 31) if meteo is not None and meteo.year_max is not None else None
    return TemporalCoverage(
        resolution=resolution,
        start=start,
        end=end,
        aggregation=aggregation,
    )


class _PVGISProviderBase:
    provider_id_default = "pvgis"
    source_name_default = PVGIS_SOURCE_NAME
    served: frozenset[str] = frozenset()

    def __init__(
        self,
        *,
        client: PVGISClient | None = None,
        cache: DatasetCache | None = None,
        clock: Clock | None = None,
        timeout_seconds: float = 30.0,
        max_attempts: int = 3,
        base_url: str = PVGIS_DEFAULT_BASE_URL,
        transport: httpx.BaseTransport | None = None,
        http_client: httpx.Client | None = None,
        sleep: Callable[[float], None] | None = None,
        provider_id: str | None = None,
    ) -> None:
        self._provider_id = provider_id or self.provider_id_default
        self._clock: Clock = UtcClock() if clock is None else clock
        self._owns_pvgis_client = client is None
        self._pvgis = client or PVGISClient(
            base_url=base_url,
            timeout_seconds=timeout_seconds,
            max_attempts=max_attempts,
            cache=cache,
            clock=self._clock,
            client=http_client,
            transport=transport,
            sleep=sleep,
        )

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def client(self) -> PVGISClient:
        return self._pvgis

    def close(self) -> None:
        if self._owns_pvgis_client:
            self._pvgis.close()

    def health(self) -> ProviderHealth:
        return ProviderHealth(
            status=HealthStatus.OK,
            provider_id=self._provider_id,
            reason=f"PVGIS {PVGIS_API_VERSION} client configured",
            checked_at=self._clock.now(),
            details={"base_url": self._pvgis.base_url, "api_version": PVGIS_API_VERSION},
        )

    def availability(self, request: FetchRequest) -> ProviderAvailability:
        if request.indicator_id not in self.served:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="indicator is not served by this PVGIS provider",
            )
        if _point_or_none(request) is None:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="PVGIS requires a point latitude and longitude",
            )
        extra = self._availability_extra(request)
        if extra is not None:
            return extra
        return ProviderAvailability(
            status=AvailabilityStatus.AVAILABLE,
            provider_id=self._provider_id,
            retryable=False,
            indicator_id=request.indicator_id,
            reason="PVGIS can be queried for this point",
        )

    def _availability_extra(self, request: FetchRequest) -> ProviderAvailability | None:
        if request.indicator_id == IndicatorId.SOLAR_MONTHLY_VARIABILITY.value:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=(
                    "PVGIS does not publish a unitless monthly variability index; "
                    "monthly series are returned with SOLAR_GHI_ANNUAL"
                ),
            )
        return None

    def coverage(self, request: FetchRequest) -> CoverageReport:
        availability = self.availability(request)
        point = _point_or_none(request)
        spatial = None
        if point is not None:
            spatial = SpatialCoverage(
                resolution="pvgis-point",
                crs="EPSG:4326",
                geometry_fingerprint=f"lat={point[1]},lon={point[0]}",
            )
        return CoverageReport(
            status=availability.status,
            provider_id=self._provider_id,
            spatial=spatial,
            temporal=request.temporal,
            reason=availability.reason,
        )

    def _dataset(self, meteo: PVGISMeteoData | None, source_url: str) -> DatasetRef:
        name = PVGIS_SOURCE_NAME
        if meteo is not None and meteo.radiation_db:
            name = meteo.radiation_db
        return DatasetRef(
            name=name,
            media_type="application/json",
            uri=source_url,
            notes=(f"pvgis_api={PVGIS_API_VERSION}",),
        )

    def _miss(
        self,
        request: FetchRequest,
        *,
        status: ObservationStatus,
        reason: str,
        source_url: str,
        location: PVGISLocation | None = None,
        meteo: PVGISMeteoData | None = None,
        retrieved_at: datetime,
    ) -> IndicatorObservation:
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
                source_version=PVGIS_API_VERSION,
                dataset=self._dataset(meteo, source_url),
            ),
            spatial=_spatial(location, request),
            temporal=_temporal(meteo, resolution="unknown", aggregation=None),
            quality_flags=(),
            assumptions=(reason,),
        )

    def _from_http_miss(
        self, request: FetchRequest, result: PVGISHttpResult, retrieved_at: datetime
    ) -> IndicatorObservation:
        status = _http_status_to_observation(result)
        reason = result.message or f"PVGIS HTTP {result.status_code}"
        return self._miss(
            request,
            status=status,
            reason=reason,
            source_url=result.source_url,
            retrieved_at=retrieved_at,
        )


class PVGISSolarResourceProvider(_PVGISProviderBase):
    """Map ``MRcalc`` monthly radiation onto SolarDSS solar-resource indicators."""

    provider_id_default = "pvgis.solar_resource"
    served = _RESOURCE_INDICATORS

    def fetch_resource(self, request: FetchRequest) -> PVGISResourceResult:
        availability = self.availability(request)
        if availability.status is not AvailabilityStatus.AVAILABLE:
            retrieved_at = self._clock.now()
            miss = self._miss(
                request,
                status=ObservationStatus(availability.status.value),
                reason=availability.reason,
                source_url=self._pvgis.base_url,
                retrieved_at=retrieved_at,
            )
            return PVGISResourceResult(
                raw=PVGISHttpResult(
                    tool="MRcalc",
                    status_code=0,
                    source_url=self._pvgis.base_url,
                    from_cache=False,
                    message=availability.reason,
                ),
                location=None,
                meteo=None,
                monthly=(),
                observations=(miss,),
            )
        point = _point_or_none(request)
        assert point is not None
        lon, lat = point
        startyear, endyear = _years_from_request(request)
        params: dict[str, JsonScalar] = {
            "lat": lat,
            "lon": lon,
            "horirrad": 1,
            "mr_dni": 1,
            "d2g": 1,
            "avtemp": 1,
        }
        raddatabase = _scalar_param(request.parameters, "raddatabase")
        if isinstance(raddatabase, str):
            params["raddatabase"] = raddatabase
        usehorizon = _scalar_param(request.parameters, "usehorizon")
        if usehorizon is not None:
            params["usehorizon"] = usehorizon
        if startyear is not None:
            params["startyear"] = startyear
        if endyear is not None:
            params["endyear"] = endyear
        try:
            raw = self._pvgis.get_json("MRcalc", params)
        except PVGISHTTPError as exc:
            retrieved_at = self._clock.now()
            miss = self._miss(
                request,
                status=ObservationStatus.TEMPORARY_FAILURE,
                reason=exc.message,
                source_url=self._pvgis.base_url,
                retrieved_at=retrieved_at,
            )
            return PVGISResourceResult(
                raw=PVGISHttpResult(
                    tool="MRcalc",
                    status_code=0,
                    source_url=self._pvgis.base_url,
                    from_cache=False,
                    message=exc.message,
                    retrieved_at=retrieved_at,
                ),
                location=None,
                meteo=None,
                monthly=(),
                observations=(miss,),
            )
        retrieved_at = raw.retrieved_at or self._clock.now()
        if raw.payload is None:
            miss = self._from_http_miss(request, raw, retrieved_at)
            return PVGISResourceResult(
                raw=raw, location=None, meteo=None, monthly=(), observations=(miss,)
            )
        location = parse_location(raw.payload)
        meteo = parse_meteo(raw.payload)
        rows = parse_monthly_radiation(raw.payload)
        monthly = normalize_radiation_rows(rows, meta=raw.payload.meta)
        observations = tuple(
            self._observation_from_monthly(
                FetchRequest(
                    indicator_id=indicator_id,
                    location=request.location,
                    bbox=request.bbox,
                    temporal=request.temporal,
                    parameters=request.parameters,
                ),
                raw=raw,
                location=location,
                meteo=meteo,
                monthly=monthly,
                retrieved_at=retrieved_at,
            )
            for indicator_id in sorted(_RESOURCE_INDICATORS)
        )
        return PVGISResourceResult(
            raw=raw,
            location=location,
            meteo=meteo,
            monthly=monthly,
            observations=observations,
        )

    def fetch(self, request: FetchRequest) -> IndicatorObservation:
        bundle = self.fetch_resource(request)
        for item in bundle.observations:
            if item.indicator_id == request.indicator_id:
                return item
        retrieved_at = self._clock.now()
        return self._miss(
            request,
            status=ObservationStatus.UNAVAILABLE,
            reason="indicator was not produced from the PVGIS response",
            source_url=bundle.raw.source_url,
            location=bundle.location,
            meteo=bundle.meteo,
            retrieved_at=retrieved_at,
        )

    def _observation_from_monthly(
        self,
        request: FetchRequest,
        *,
        raw: PVGISHttpResult,
        location: PVGISLocation | None,
        meteo: PVGISMeteoData | None,
        monthly: tuple[PVGISMonthlyRadiation, ...],
        retrieved_at: datetime,
    ) -> IndicatorObservation:
        request_hash = request_cache_key(self._provider_id, request)
        dataset = self._dataset(meteo, raw.source_url)
        spatial = _spatial(location, request)
        temporal = _temporal(meteo, resolution="annual", aggregation="mean_of_complete_year_sums")
        assumptions = [
            "MRcalc monthly fields H(h)_m, Hb(n)_m, Kd, T2m as returned by PVGIS",
            "annual totals use complete Jan-Dec years only",
            f"source_url={raw.source_url}",
        ]
        if request.indicator_id == IndicatorId.SOLAR_MONTHLY_VARIABILITY.value:
            return IndicatorObservation(
                indicator_id=request.indicator_id,
                status=ObservationStatus.UNAVAILABLE,
                provenance=DataProvenance(
                    provider_id=self._provider_id,
                    source_name=self.source_name_default,
                    retrieved_at=retrieved_at,
                    request_hash=request_hash,
                    value_kind=ValueProvenanceKind.MEASURED_OR_PROVIDER,
                    value_origin=ValueOrigin.RAW,
                    source_version=PVGIS_API_VERSION,
                    dataset=dataset,
                ),
                spatial=spatial,
                temporal=temporal,
                assumptions=(
                    "PVGIS MRcalc has no unitless monthly variability index; "
                    "monthly H(h)_m is preserved on PVGISResourceResult.monthly",
                ),
            )
        attr: Literal["ghi_kwh_m2", "dni_kwh_m2", "dhi_kwh_m2"]
        method: str
        origin: ValueOrigin
        kind: ValueProvenanceKind
        extra: str
        if request.indicator_id == IndicatorId.SOLAR_GHI_ANNUAL.value:
            attr, method, origin, kind, extra = (
                "ghi_kwh_m2",
                _CALC_ANNUAL_SUM + ":H(h)_m",
                ValueOrigin.DERIVED,
                ValueProvenanceKind.DERIVED_METHOD,
                "H(h)_m is monthly GHI in kWh/m2/mo",
            )
        elif request.indicator_id == IndicatorId.SOLAR_DNI_ANNUAL.value:
            attr, method, origin, kind, extra = (
                "dni_kwh_m2",
                _CALC_ANNUAL_SUM + ":Hb(n)_m",
                ValueOrigin.DERIVED,
                ValueProvenanceKind.DERIVED_METHOD,
                "Hb(n)_m is monthly DNI in kWh/m2/mo",
            )
        else:
            attr, method, origin, kind, extra = (
                "dhi_kwh_m2",
                _CALC_DHI,
                ValueOrigin.DERIVED,
                ValueProvenanceKind.DERIVED_METHOD,
                "monthly DHI is Kd * H(h)_m only when both fields are present",
            )
        value = mean_complete_year_sums(monthly, attr)
        if value is None:
            return IndicatorObservation(
                indicator_id=request.indicator_id,
                status=ObservationStatus.UNAVAILABLE,
                provenance=DataProvenance(
                    provider_id=self._provider_id,
                    source_name=self.source_name_default,
                    retrieved_at=retrieved_at,
                    request_hash=request_hash,
                    value_kind=kind,
                    value_origin=origin,
                    source_version=PVGIS_API_VERSION,
                    calculation_method=method if origin is ValueOrigin.DERIVED else None,
                    dataset=dataset,
                ),
                spatial=spatial,
                temporal=temporal,
                assumptions=(extra, "no complete calendar year of PVGIS monthly values"),
            )
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=self.source_name_default,
                retrieved_at=retrieved_at,
                request_hash=request_hash,
                value_kind=kind,
                value_origin=origin,
                source_version=PVGIS_API_VERSION,
                calculation_method=method,
                dataset=dataset,
            ),
            spatial=spatial,
            temporal=temporal,
            value=value,
            unit="kWh/m2/year",
            quality_flags=(QualityFlag.NOMINAL.value,),
            assumptions=tuple([*assumptions, extra]),
        )


class PVGISYieldProvider(_PVGISProviderBase):
    """Map ``PVcalc`` energy output onto PVOUT / specific yield.

    ``peakpower`` and ``loss`` are obligatory PVGIS inputs. They must be supplied
    on the provider or the fetch request; SolarDSS does not invent 14% losses.
    """

    provider_id_default = "pvgis.yield"
    served = _YIELD_INDICATORS

    def __init__(
        self,
        *,
        peak_power_kw: float | None = None,
        system_loss_percent: float | None = None,
        mounting: str = "fixed",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._peak_power_kw = peak_power_kw
        self._system_loss_percent = system_loss_percent
        self._mounting = mounting

    def _availability_extra(self, request: FetchRequest) -> ProviderAvailability | None:
        if request.indicator_id == IndicatorId.SOLAR_MONTHLY_VARIABILITY.value:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason=(
                    "PVGIS SD_m/SD_y are in kWh, not a unitless variability index; "
                    "they are preserved on PVGISYieldResult"
                ),
            )
        peak, loss = self._system_params(request)
        if peak is None or loss is None:
            return ProviderAvailability(
                status=AvailabilityStatus.UNAVAILABLE,
                provider_id=self._provider_id,
                retryable=False,
                indicator_id=request.indicator_id,
                reason="PVcalc requires peakpower (kW) and loss (percent)",
            )
        return None

    def _system_params(self, request: FetchRequest) -> tuple[float | None, float | None]:
        peak = self._peak_power_kw
        loss = self._system_loss_percent
        raw_peak = _scalar_param(request.parameters, "peakpower")
        raw_loss = _scalar_param(request.parameters, "loss")
        if isinstance(raw_peak, int | float) and not isinstance(raw_peak, bool):
            peak = float(raw_peak)
        if isinstance(raw_loss, int | float) and not isinstance(raw_loss, bool):
            loss = float(raw_loss)
        return peak, loss

    def fetch_yield(self, request: FetchRequest) -> PVGISYieldResult:
        availability = self.availability(request)
        if availability.status is not AvailabilityStatus.AVAILABLE:
            retrieved_at = self._clock.now()
            miss = self._miss(
                request,
                status=ObservationStatus(availability.status.value),
                reason=availability.reason,
                source_url=self._pvgis.base_url,
                retrieved_at=retrieved_at,
            )
            return PVGISYieldResult(
                raw=PVGISHttpResult(
                    tool="PVcalc",
                    status_code=0,
                    source_url=self._pvgis.base_url,
                    from_cache=False,
                    message=availability.reason,
                ),
                location=None,
                meteo=None,
                mounting=self._mounting,
                peak_power_kw=self._peak_power_kw,
                system_loss_percent=self._system_loss_percent,
                monthly=(),
                totals=None,
                observations=(miss,),
            )
        point = _point_or_none(request)
        assert point is not None
        lon, lat = point
        peak, loss = self._system_params(request)
        assert peak is not None and loss is not None
        params: dict[str, JsonScalar] = {
            "lat": lat,
            "lon": lon,
            "peakpower": peak,
            "loss": loss,
        }
        for key in ("raddatabase", "pvtechchoice", "mountingplace", "angle", "aspect"):
            value = _scalar_param(request.parameters, key)
            if value is not None:
                params[key] = value
        try:
            raw = self._pvgis.get_json("PVcalc", params)
        except PVGISHTTPError as exc:
            retrieved_at = self._clock.now()
            miss = self._miss(
                request,
                status=ObservationStatus.TEMPORARY_FAILURE,
                reason=exc.message,
                source_url=self._pvgis.base_url,
                retrieved_at=retrieved_at,
            )
            return PVGISYieldResult(
                raw=PVGISHttpResult(
                    tool="PVcalc",
                    status_code=0,
                    source_url=self._pvgis.base_url,
                    from_cache=False,
                    message=exc.message,
                    retrieved_at=retrieved_at,
                ),
                location=None,
                meteo=None,
                mounting=self._mounting,
                peak_power_kw=peak,
                system_loss_percent=loss,
                monthly=(),
                totals=None,
                observations=(miss,),
            )
        retrieved_at = raw.retrieved_at or self._clock.now()
        if raw.payload is None:
            miss = self._from_http_miss(request, raw, retrieved_at)
            return PVGISYieldResult(
                raw=raw,
                location=None,
                meteo=None,
                mounting=self._mounting,
                peak_power_kw=peak,
                system_loss_percent=loss,
                monthly=(),
                totals=None,
                observations=(miss,),
            )
        location = parse_location(raw.payload)
        meteo = parse_meteo(raw.payload)
        pv_rows = parse_pv_monthly(raw.payload, self._mounting)
        totals = parse_pv_totals(raw.payload, self._mounting)
        monthly = _normalize_yield_rows(pv_rows)
        observations = tuple(
            self._observation_from_yield(
                FetchRequest(
                    indicator_id=indicator_id,
                    location=request.location,
                    bbox=request.bbox,
                    temporal=request.temporal,
                    parameters=request.parameters,
                ),
                raw=raw,
                location=location,
                meteo=meteo,
                totals=totals,
                peak_power_kw=peak,
                retrieved_at=retrieved_at,
            )
            for indicator_id in sorted(_YIELD_INDICATORS)
        )
        return PVGISYieldResult(
            raw=raw,
            location=location,
            meteo=meteo,
            mounting=self._mounting,
            peak_power_kw=peak,
            system_loss_percent=loss,
            monthly=monthly,
            totals=totals,
            observations=observations,
        )

    def fetch(self, request: FetchRequest) -> IndicatorObservation:
        bundle = self.fetch_yield(request)
        for item in bundle.observations:
            if item.indicator_id == request.indicator_id:
                return item
        retrieved_at = self._clock.now()
        return self._miss(
            request,
            status=ObservationStatus.UNAVAILABLE,
            reason="indicator was not produced from the PVGIS response",
            source_url=bundle.raw.source_url,
            location=bundle.location,
            meteo=bundle.meteo,
            retrieved_at=retrieved_at,
        )

    def _observation_from_yield(
        self,
        request: FetchRequest,
        *,
        raw: PVGISHttpResult,
        location: PVGISLocation | None,
        meteo: PVGISMeteoData | None,
        totals: PVGISPVTotals,
        peak_power_kw: float,
        retrieved_at: datetime,
    ) -> IndicatorObservation:
        request_hash = request_cache_key(self._provider_id, request)
        dataset = self._dataset(meteo, raw.source_url)
        spatial = _spatial(location, request)
        temporal = _temporal(meteo, resolution="annual", aggregation="pvgis_pvcalc_totals")
        if request.indicator_id == IndicatorId.SOLAR_MONTHLY_VARIABILITY.value:
            return IndicatorObservation(
                indicator_id=request.indicator_id,
                status=ObservationStatus.UNAVAILABLE,
                provenance=DataProvenance(
                    provider_id=self._provider_id,
                    source_name=self.source_name_default,
                    retrieved_at=retrieved_at,
                    request_hash=request_hash,
                    value_kind=ValueProvenanceKind.MEASURED_OR_PROVIDER,
                    value_origin=ValueOrigin.RAW,
                    source_version=PVGIS_API_VERSION,
                    dataset=dataset,
                ),
                spatial=spatial,
                temporal=temporal,
                assumptions=(
                    "PVGIS SD_m/SD_y are in kWh, not a unitless variability index; "
                    "they are preserved on PVGISYieldResult",
                ),
            )
        energy_units_ok = True
        if raw.payload is not None:
            energy_units_ok = _units_allowed(raw.payload.meta, "totals", "E_y", _ENERGY_YEAR_UNITS)
        if not energy_units_ok or totals.e_y is None:
            return IndicatorObservation(
                indicator_id=request.indicator_id,
                status=ObservationStatus.UNAVAILABLE,
                provenance=DataProvenance(
                    provider_id=self._provider_id,
                    source_name=self.source_name_default,
                    retrieved_at=retrieved_at,
                    request_hash=request_hash,
                    value_kind=ValueProvenanceKind.DERIVED_METHOD,
                    value_origin=ValueOrigin.DERIVED,
                    source_version=PVGIS_API_VERSION,
                    calculation_method=_CALC_YIELD,
                    dataset=dataset,
                ),
                spatial=spatial,
                temporal=temporal,
                assumptions=("PVcalc totals.E_y is missing or uses an unexpected unit",),
            )
        if peak_power_kw == 0:
            raise InvalidParameterError("peakpower must be non-zero")
        specific = totals.e_y / peak_power_kw
        return IndicatorObservation(
            indicator_id=request.indicator_id,
            status=ObservationStatus.OBSERVED,
            provenance=DataProvenance(
                provider_id=self._provider_id,
                source_name=self.source_name_default,
                retrieved_at=retrieved_at,
                request_hash=request_hash,
                value_kind=ValueProvenanceKind.DERIVED_METHOD,
                value_origin=ValueOrigin.DERIVED,
                source_version=PVGIS_API_VERSION,
                calculation_method=_CALC_YIELD,
                dataset=dataset,
            ),
            spatial=spatial,
            temporal=temporal,
            value=specific,
            unit="kWh/kWp/year",
            quality_flags=(QualityFlag.NOMINAL.value,),
            assumptions=(
                "E_y is PVGIS annual AC energy for the requested peakpower and loss",
                "specific yield is E_y / peakpower",
                "PVOUT and specific yield share this PVGIS quantity",
                f"source_url={raw.source_url}",
            ),
        )


def _normalize_yield_rows(rows: Sequence[PVGISPVMonthlyRow]) -> tuple[PVGISMonthlyYield, ...]:
    normalized: list[PVGISMonthlyYield] = []
    for row in rows:
        present: list[str] = []
        if row.e_d is not None:
            present.append("E_d")
        if row.e_m is not None:
            present.append("E_m")
        if row.h_i_d is not None:
            present.append("H(i)_d")
        if row.h_i_m is not None:
            present.append("H(i)_m")
        if row.sd_m is not None:
            present.append("SD_m")
        normalized.append(
            PVGISMonthlyYield(
                month=row.month,
                energy_kwh_day=row.e_d,
                energy_kwh_month=row.e_m,
                poa_kwh_m2_day=row.h_i_d,
                poa_kwh_m2_month=row.h_i_m,
                monthly_energy_sd_kwh=row.sd_m,
                present_fields=tuple(present),
            )
        )
    return tuple(normalized)
