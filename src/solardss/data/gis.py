"""Optional GIS extra. Nothing here opens files or loads spatial libraries at import."""

from __future__ import annotations

import importlib.util

from solardss.exceptions import MissingExtraError

GIS_EXTRA_MODULES: tuple[str, ...] = ("shapely", "geopandas", "pyproj", "rasterio")


def gis_module_available(module_name: str) -> bool:
    """Return whether an importable module spec exists (no import side effects)."""

    return importlib.util.find_spec(module_name) is not None


def gis_extra_status() -> dict[str, bool]:
    """Availability of each declared GIS extra module."""

    return {name: gis_module_available(name) for name in GIS_EXTRA_MODULES}


def require_gis() -> None:
    """Raise if the optional GIS extra is incomplete.

    GeoTIFF DEM/land-cover windowing, CRS transforms, GeoPackage, and shapefile
    I/O call this (or import the extra lazily) so the error stays explicit.
    """

    missing = [name for name, found in gis_extra_status().items() if not found]
    if missing:
        joined = ", ".join(missing)
        msg = (
            "Optional GIS extra is not fully installed. "
            "Install with: pip install 'solardss[gis]'. "
            f"Missing: {joined}"
        )
        raise MissingExtraError(msg)
