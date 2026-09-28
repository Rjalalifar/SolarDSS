"""Content fingerprints. These hash values, never filesystem paths."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from solardss.core.arrays import AnyArray

_HASH_LENGTH = 16


def _digest(payload: bytes, *, length: int | None = _HASH_LENGTH) -> str:
    digest = hashlib.sha256(payload).hexdigest()
    if length is None:
        return digest
    return digest[:length]


def fingerprint_canonical(mapping: Mapping[str, object]) -> str:
    """Full SHA-256 of a canonical JSON object. Used as cache keys."""

    payload = json.dumps(
        mapping,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _digest(payload, length=None)


def fingerprint_array(values: AnyArray) -> str:
    """Stable fingerprint of an array's shape, dtype, and bytes."""

    contiguous = np.ascontiguousarray(values)
    payload = (
        str(contiguous.shape).encode("utf-8")
        + b"|"
        + str(contiguous.dtype).encode("utf-8")
        + b"|"
        + contiguous.tobytes()
    )
    return _digest(payload)


def fingerprint_frame(frame: pd.DataFrame, *, id_column: str | None = None) -> str:
    """Stable fingerprint of a table. Rows are ordered by ``id_column`` when given."""

    ordered = frame.copy()
    if id_column is not None:
        ordered = ordered.sort_values(id_column, kind="mergesort", ignore_index=True)
    columns = list(ordered.columns)
    payload = (",".join(columns) + "\n").encode("utf-8")
    payload += ordered.to_csv(index=False).encode("utf-8")
    return _digest(payload)


def fingerprint_sequence(values: Sequence[object]) -> str:
    """Stable fingerprint of a sequence of JSON-like scalars."""

    payload = repr(tuple(values)).encode("utf-8")
    return _digest(payload)
