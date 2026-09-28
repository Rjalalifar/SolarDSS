"""JSON-safe parameter encoding for provenance records.

Values are one level deep so the types stay valid on Python 3.10 with Pydantic.
Nested JSON trees are not stored; callers flatten method parameters first.
"""

from __future__ import annotations

from collections.abc import Mapping

from solardss.exceptions import InvalidParameterError

JsonScalar = str | int | float | bool | None
JsonValue = JsonScalar | list[JsonScalar] | dict[str, JsonScalar]


def json_scalar(value: object) -> JsonScalar:
    """Convert a scalar to a JSON-safe value, or raise."""

    if value is None or isinstance(value, str | int | bool):
        return value
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return value
    msg = f"Cannot store {type(value).__name__} as a provenance scalar"
    raise InvalidParameterError(msg)


def json_value(value: object) -> JsonValue:
    """Convert a scalar, list of scalars, or string-keyed scalar map."""

    if isinstance(value, list | tuple):
        return [json_scalar(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_scalar(item) for key, item in value.items()}
    return json_scalar(value)


def json_params(parameters: Mapping[str, object]) -> dict[str, JsonValue]:
    """Copy caller parameters into a JSON-safe dict with string keys."""

    return {str(key): json_value(value) for key, value in parameters.items()}
