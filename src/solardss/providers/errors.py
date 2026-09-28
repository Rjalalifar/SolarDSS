"""Provider access errors. Missing data is an observation status, not an exception."""

from __future__ import annotations

from solardss.exceptions import SolarDSSError


class ProviderError(SolarDSSError):
    """A provider call was malformed or the cache could not be used."""

    code = "provider_error"


class CacheError(ProviderError):
    """The dataset cache could not read or write an entry."""

    code = "provider_cache_error"


class PVGISError(ProviderError):
    """A PVGIS HTTP or payload problem. Missing values stay observations."""

    code = "pvgis_error"


class PVGISHTTPError(PVGISError):
    """The PVGIS HTTP call failed after retries or returned an unexpected status."""

    code = "pvgis_http_error"

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class PVGISResponseError(PVGISError):
    """The PVGIS JSON body could not be validated against the official schema."""

    code = "pvgis_response_error"


class ERA5LandError(ProviderError):
    """An ERA5-Land configuration or dataset problem. Missing values stay observations."""

    code = "era5_land_error"


class ERA5LandResponseError(ERA5LandError):
    """The local or CDS ERA5-Land payload could not be interpreted."""

    code = "era5_land_response_error"


class DEMError(ProviderError):
    """A DEM windowing or configuration problem. Missing values stay observations."""

    code = "dem_error"


class DEMCoverageError(DEMError):
    """The requested window does not intersect the configured DEM."""

    code = "dem_coverage_error"


class LandCoverError(ProviderError):
    """A land-cover windowing or configuration problem. Missing values stay observations."""

    code = "land_cover_error"


class LandCoverCoverageError(LandCoverError):
    """The requested window does not intersect the configured land-cover raster."""

    code = "land_cover_coverage_error"


class OSMError(ProviderError):
    """An OpenStreetMap / Overpass configuration or query problem."""

    code = "osm_error"


class OSMHTTPError(OSMError):
    """The Overpass HTTP call failed after retries or returned an unexpected status."""

    code = "osm_http_error"

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class OSMResponseError(OSMError):
    """The Overpass JSON body could not be interpreted."""

    code = "osm_response_error"


class OSMQueryBoundError(OSMError):
    """The Overpass request was rejected because it was not spatially bounded."""

    code = "osm_query_bound_error"


class ProtectedAreaError(ProviderError):
    """A protected-area cadastre configuration or I/O problem."""

    code = "protected_area_error"


class ProtectedAreaCoverageError(ProtectedAreaError):
    """The requested window does not intersect the configured cadastre."""

    code = "protected_area_coverage_error"


class HazardError(ProviderError):
    """A hazard-map configuration or I/O problem. Missing values stay observations."""

    code = "hazard_error"


class HazardCoverageError(HazardError):
    """The requested window does not intersect the configured hazard layer."""

    code = "hazard_coverage_error"
