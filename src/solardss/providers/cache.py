"""Deterministic local cache for provider payloads.

Keys are SHA-256 hex digests from ``request_cache_key``. The cache stores opaque
bytes; it never interprets scientific values. There is no default machine path:
``DirectoryCache`` requires an explicit root, or ``cache_root_from_config``
reads ``SOLARDSS_CACHE_DIR`` when the caller asks it to.
"""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, runtime_checkable

from solardss.providers.errors import CacheError

_CACHE_KEY = re.compile(r"^[0-9a-f]{64}$")
CACHE_DIR_ENV = "SOLARDSS_CACHE_DIR"


def cache_root_from_config(
    *,
    explicit: Path | str | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path | None:
    """Resolve a cache directory without guessing a home or system path.

    Preference order: ``explicit``, then ``SOLARDSS_CACHE_DIR``. A missing or
    blank environment value returns ``None``; the caller must then use
    ``MemoryCache`` or pass a path.
    """

    if explicit is not None:
        return Path(explicit)
    source = os.environ if environ is None else environ
    raw = source.get(CACHE_DIR_ENV)
    if raw is None or not raw.strip():
        return None
    return Path(raw)


def require_cache_key(key: str) -> str:
    """Reject anything that is not a full SHA-256 hex digest."""

    if not _CACHE_KEY.fullmatch(key):
        raise CacheError("cache key must be a 64-character SHA-256 hex digest")
    return key


@runtime_checkable
class DatasetCache(Protocol):
    """Byte store keyed by a stable request hash."""

    def get(self, key: str) -> bytes | None:
        """Return the stored payload, or ``None`` on a miss."""

    def put(self, key: str, payload: bytes) -> None:
        """Write ``payload`` under ``key``. Overwrites an existing entry."""


class MemoryCache:
    """Process-local cache. Suitable for tests."""

    def __init__(self) -> None:
        self._entries: dict[str, bytes] = {}

    def get(self, key: str) -> bytes | None:
        return self._entries.get(require_cache_key(key))

    def put(self, key: str, payload: bytes) -> None:
        self._entries[require_cache_key(key)] = payload


class DirectoryCache:
    """Filesystem cache under a caller-supplied directory.

    Entries are ``{root}/{key[:2]}/{key}.bin``. Writes are atomic (tempfile
    plus ``os.replace``). The constructor does not fall back to ``~`` or
    platform cache directories.
    """

    def __init__(self, root: Path | str) -> None:
        self._root = Path(root)
        if self._root.exists() and not self._root.is_dir():
            raise CacheError("cache root exists and is not a directory")

    @property
    def root(self) -> Path:
        return self._root

    def _path(self, key: str) -> Path:
        digest = require_cache_key(key)
        return self._root / digest[:2] / f"{digest}.bin"

    def get(self, key: str) -> bytes | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError as exc:
            raise CacheError(f"cannot read cache entry {key}") from exc

    def put(self, key: str, payload: bytes) -> None:
        path = self._path(key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{key[:16]}.",
                suffix=".tmp",
                dir=path.parent,
            )
        except OSError as exc:
            raise CacheError(f"cannot write cache under {self._root}") from exc
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, path)
        except OSError as exc:
            tmp_path.unlink(missing_ok=True)
            raise CacheError(f"cannot write cache entry {key}") from exc
