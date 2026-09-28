"""GIS extra is optional and is not imported by the core package."""

from __future__ import annotations

import subprocess
import sys

from solardss.data import GIS_EXTRA_MODULES, gis_extra_status, require_gis
from solardss.exceptions import MissingExtraError


def test_gis_status_keys() -> None:
    status = gis_extra_status()
    assert tuple(status) == GIS_EXTRA_MODULES
    assert all(isinstance(value, bool) for value in status.values())


def test_require_gis_when_missing() -> None:
    status = gis_extra_status()
    if all(status.values()):
        require_gis()
        return
    try:
        require_gis()
    except MissingExtraError as exc:
        assert "solardss[gis]" in exc.message
    else:
        raise AssertionError("expected MissingExtraError when GIS extra is incomplete")


def test_core_import_does_not_load_gis_stack() -> None:
    modules = ", ".join(repr(name) for name in GIS_EXTRA_MODULES)
    code = (
        "import sys\n"
        "import solardss.data  # noqa: F401\n"
        "import solardss.candidates  # noqa: F401\n"
        f"loaded = [name for name in ({modules}) if name in sys.modules]\n"
        "raise SystemExit(','.join(loaded) if loaded else 0)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
