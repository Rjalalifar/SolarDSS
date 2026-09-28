"""Load a caller-supplied PV catalog from local JSON or CSV.

Files are validated. SolarDSS does not fetch a remote manufacturer database.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from solardss.core.methods import MethodId
from solardss.provenance.record import ProvenanceRecord, provenance_record
from solardss.pv.catalog import PVComponentCatalog
from solardss.pv.contracts import (
    BatterySystem,
    ComponentProvenance,
    ComponentSourceKind,
    PVInverter,
    PVModule,
)
from solardss.pv.errors import PVCatalogConfigError, PVCatalogSchemaError

_TRUE = {"1", "true", "yes", "y"}
_FALSE = {"0", "false", "no", "n"}
_Row = TypeVar("_Row", PVModule, PVInverter, BatterySystem)


def load_pv_catalog(
    path: Path | str,
    *,
    kind: str | None = None,
) -> PVComponentCatalog:
    """Load and validate a local ``.json`` or ``.csv`` catalog.

    ``kind`` is required for CSV (``module``, ``inverter``, or ``battery``) and
    ignored for JSON objects that already nest those arrays.
    """

    file_path = Path(path)
    if not file_path.is_file():
        raise PVCatalogSchemaError(f"PV catalog file not found: {file_path.name}")
    suffix = file_path.suffix.casefold()
    stamp = _file_stamp(file_path)
    if suffix == ".json":
        return _load_json(file_path, stamp)
    if suffix == ".csv":
        if kind is None:
            kind = _infer_csv_kind(file_path)
        return _load_csv(file_path, kind=kind, stamp=stamp)
    raise PVCatalogSchemaError(
        f"unsupported catalog suffix {file_path.suffix!r}; use .json or .csv"
    )


def _file_stamp(path: Path) -> ProvenanceRecord:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return provenance_record(
        MethodId.PV_CATALOG_LOAD,
        parameters={"filename": path.name, "sha256": digest},
        input_fingerprints=(digest,),
        notes=("local file import; contents are caller-supplied",),
    )


def _load_json(path: Path, stamp: ProvenanceRecord) -> PVComponentCatalog:
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        payload = {"modules": payload}
    if not isinstance(payload, dict):
        raise PVCatalogSchemaError("catalog JSON must be an object or a module array")
    modules = _parse_list(payload.get("modules", []), PVModule, stamp)
    inverters = _parse_list(payload.get("inverters", []), PVInverter, stamp)
    batteries = _parse_list(payload.get("batteries", []), BatterySystem, stamp)
    extra = set(payload) - {"modules", "inverters", "batteries", "notes"}
    if extra:
        raise PVCatalogSchemaError(f"unexpected catalog JSON keys: {sorted(extra)}")
    return PVComponentCatalog(
        modules=modules,
        inverters=inverters,
        batteries=batteries,
        source_name=path.name,
    )


def _parse_list(raw: object, model: type[_Row], stamp: ProvenanceRecord) -> tuple[_Row, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise PVCatalogSchemaError(f"{model.__name__} catalog field must be an array")
    out: list[_Row] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise PVCatalogSchemaError(f"{model.__name__} row {index} must be an object")
        try:
            parsed = model.model_validate(_with_defaults(item))
        except (ValidationError, PVCatalogConfigError) as exc:
            raise PVCatalogSchemaError(f"{model.__name__} row {index} is invalid: {exc}") from exc
        out.append(_attach_stamp(parsed, stamp))
    return tuple(out)


def _with_defaults(item: Mapping[str, object]) -> dict[str, object]:
    payload = dict(item)
    if "provenance" not in payload:
        raise PVCatalogSchemaError("each catalog row requires provenance")
    return payload


def _attach_stamp(row: _Row, stamp: ProvenanceRecord) -> _Row:
    if row.provenance.record is not None:
        return row
    provenance = row.provenance.model_copy(update={"record": stamp})
    return row.model_copy(update={"provenance": provenance})


def _infer_csv_kind(path: Path) -> str:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = {name.casefold() for name in (reader.fieldnames or [])}
    if "module_id" in fields:
        return "module"
    if "inverter_id" in fields:
        return "inverter"
    if "battery_id" in fields:
        return "battery"
    raise PVCatalogSchemaError(
        "CSV catalog kind could not be inferred; pass kind='module', 'inverter', or 'battery'"
    )


def _load_csv(path: Path, *, kind: str, stamp: ProvenanceRecord) -> PVComponentCatalog:
    rows = _read_csv_rows(path)
    if kind == "module":
        modules = tuple(_validate_csv_row(PVModule, row, index) for index, row in enumerate(rows))
        modules = tuple(_attach_stamp(item, stamp) for item in modules)
        return PVComponentCatalog(modules=modules, source_name=path.name)
    if kind == "inverter":
        inverters = tuple(
            _validate_csv_row(PVInverter, row, index) for index, row in enumerate(rows)
        )
        inverters = tuple(_attach_stamp(item, stamp) for item in inverters)
        return PVComponentCatalog(inverters=inverters, source_name=path.name)
    if kind == "battery":
        batteries = tuple(
            _validate_csv_row(BatterySystem, row, index) for index, row in enumerate(rows)
        )
        batteries = tuple(_attach_stamp(item, stamp) for item in batteries)
        return PVComponentCatalog(batteries=batteries, source_name=path.name)
    raise PVCatalogSchemaError(f"unknown CSV kind {kind!r}")


def _validate_csv_row(model: type[_Row], row: dict[str, object], index: int) -> _Row:
    if "provenance" not in row:
        raise PVCatalogSchemaError("each catalog row requires provenance")
    try:
        return model.model_validate(row)
    except (ValidationError, PVCatalogConfigError) as exc:
        raise PVCatalogSchemaError(f"{model.__name__} CSV row {index} is invalid: {exc}") from exc


def _read_csv_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise PVCatalogSchemaError("CSV catalog is missing a header row")
        rows = []
        for raw in reader:
            rows.append(_csv_row(raw))
    return rows


def _csv_row(raw: Mapping[str, str | None]) -> dict[str, object]:
    nested: dict[str, object] = {}
    top: dict[str, object] = {}
    for key, value in raw.items():
        if key is None:
            continue
        token = key.strip()
        text = None if value is None else value.strip()
        parsed: object = None if text == "" or text is None else _csv_value(token, text)
        if token.startswith("provenance."):
            nested[token.split(".", 1)[1]] = parsed
        elif token.startswith("metadata."):
            meta = top.setdefault("metadata", {})
            if not isinstance(meta, dict):
                raise PVCatalogSchemaError("metadata must be a mapping")
            if parsed is not None:
                meta[token.split(".", 1)[1]] = str(parsed)
        else:
            top[token] = parsed
    if nested:
        if "source_kind" in nested and isinstance(nested["source_kind"], str):
            try:
                nested["source_kind"] = ComponentSourceKind(nested["source_kind"])
            except ValueError as exc:
                raise PVCatalogSchemaError(
                    f"unknown provenance.source_kind {nested['source_kind']!r}"
                ) from exc
        if "fixture" in nested:
            nested["fixture"] = _as_bool(nested["fixture"], "provenance.fixture")
        if "notes" in nested and isinstance(nested["notes"], str):
            nested["notes"] = tuple(
                part.strip() for part in nested["notes"].split("|") if part.strip()
            )
        try:
            top["provenance"] = ComponentProvenance.model_validate(nested)
        except (ValidationError, PVCatalogConfigError) as exc:
            raise PVCatalogSchemaError(f"CSV provenance is invalid: {exc}") from exc
    return {key: value for key, value in top.items() if value is not None}


def _csv_value(field: str, text: str) -> object:
    boolean_fields = {"bifacial", "fixture", "provenance.fixture"}
    if field in boolean_fields or field.endswith(".fixture"):
        return _as_bool(text, field)
    try:
        if "." in text or "e" in text.casefold():
            return float(text)
        return int(text)
    except ValueError:
        return text


def _as_bool(value: object, field: str) -> bool:
    if isinstance(value, bool):
        return value
    token = str(value).strip().casefold()
    if token in _TRUE:
        return True
    if token in _FALSE:
        return False
    raise PVCatalogSchemaError(f"{field} must be a boolean, received {value!r}")
