"""Data-provider protocols, fakes, PVGIS, hazard maps, and optional ERA5-Land adapters.

Missing data is a structured miss. Implementations must not invent values.
HTTP/xarray adapters are imported lazily so ``providers.cache`` can load
without a cycle.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from solardss.providers.cache import (
    CACHE_DIR_ENV,
    DirectoryCache,
    MemoryCache,
    cache_root_from_config,
)
from solardss.providers.clock import FixedClock, SequenceClock, UtcClock
from solardss.providers.contracts import (
    AvailabilityStatus,
    CoverageReport,
    HealthStatus,
    ProviderAvailability,
    ProviderHealth,
)
from solardss.providers.errors import (
    CacheError,
    DEMCoverageError,
    DEMError,
    ERA5LandError,
    ERA5LandResponseError,
    HazardCoverageError,
    HazardError,
    LandCoverCoverageError,
    LandCoverError,
    OSMError,
    OSMHTTPError,
    OSMQueryBoundError,
    OSMResponseError,
    ProtectedAreaCoverageError,
    ProtectedAreaError,
    ProviderError,
    PVGISError,
    PVGISHTTPError,
    PVGISResponseError,
)
from solardss.providers.fake import (
    FakeBuildingProvider,
    FakeDataProvider,
    FakeGridCapacityProvider,
    FakeHazardProvider,
    FakeInfrastructureProvider,
    FakeLandCoverProvider,
    FakeProtectedAreaProvider,
    FakeSeed,
    FakeSolarResourceProvider,
    FakeTerrainProvider,
    FakeWeatherProvider,
)
from solardss.providers.protocols import (
    BuildingProvider,
    DataProvider,
    GridCapacityProvider,
    HazardProvider,
    InfrastructureProvider,
    LandCoverProvider,
    ProtectedAreaProvider,
    SolarResourceProvider,
    TerrainProvider,
    WeatherProvider,
)
from solardss.providers.request import FetchRequest, request_cache_key

if TYPE_CHECKING:
    from solardss.data.dem import CopernicusDEMProvider, DEMTerrainProvider
    from solardss.data.era5_land import (
        ERA5_LAND_DATASET_ID,
        ERA5LandWeatherProvider,
    )
    from solardss.data.hazard import (
        FloodRiskProvider,
        HailRiskProvider,
        HazardCatalog,
        LandslideRiskProvider,
        RasterHazardProvider,
        VectorHazardProvider,
        WildfireRiskProvider,
        hazard_available,
        hazard_for_candidates,
        hazard_for_site,
    )
    from solardss.data.osm import (
        OSMBuildingProvider,
        OSMInfrastructureProvider,
        OSMOverpassProvider,
        OverpassClient,
    )
    from solardss.data.protected import (
        GenericVectorProtectedAreaProvider,
        RegionalProtectedAreaProvider,
        VectorProtectedAreaProvider,
    )
    from solardss.data.pvgis import (
        PVGIS_API_VERSION,
        PVGIS_DEFAULT_BASE_URL,
        PVGISClient,
        PVGISSolarResourceProvider,
        PVGISYieldProvider,
    )
    from solardss.data.worldcover import (
        ESAWorldCoverProvider,
        LandCoverRasterProvider,
        WorldCoverLandCoverProvider,
    )

__all__ = [
    "CACHE_DIR_ENV",
    "ERA5_LAND_DATASET_ID",
    "PVGIS_API_VERSION",
    "PVGIS_DEFAULT_BASE_URL",
    "AvailabilityStatus",
    "BuildingProvider",
    "CacheError",
    "CopernicusDEMProvider",
    "CoverageReport",
    "DEMCoverageError",
    "DEMError",
    "DEMTerrainProvider",
    "DataProvider",
    "DirectoryCache",
    "ERA5LandError",
    "ERA5LandResponseError",
    "ERA5LandWeatherProvider",
    "ESAWorldCoverProvider",
    "FakeBuildingProvider",
    "FakeDataProvider",
    "FakeGridCapacityProvider",
    "FakeHazardProvider",
    "FakeInfrastructureProvider",
    "FakeLandCoverProvider",
    "FakeProtectedAreaProvider",
    "FakeSeed",
    "FakeSolarResourceProvider",
    "FakeTerrainProvider",
    "FakeWeatherProvider",
    "FetchRequest",
    "FixedClock",
    "FloodRiskProvider",
    "GenericVectorProtectedAreaProvider",
    "GridCapacityProvider",
    "HailRiskProvider",
    "HazardCatalog",
    "HazardCoverageError",
    "HazardError",
    "HazardProvider",
    "HealthStatus",
    "InfrastructureProvider",
    "LandCoverCoverageError",
    "LandCoverError",
    "LandCoverProvider",
    "LandCoverRasterProvider",
    "LandslideRiskProvider",
    "MemoryCache",
    "OSMBuildingProvider",
    "OSMError",
    "OSMHTTPError",
    "OSMInfrastructureProvider",
    "OSMOverpassProvider",
    "OSMQueryBoundError",
    "OSMResponseError",
    "OverpassClient",
    "PVGISClient",
    "PVGISError",
    "PVGISHTTPError",
    "PVGISResponseError",
    "PVGISSolarResourceProvider",
    "PVGISYieldProvider",
    "ProtectedAreaCoverageError",
    "ProtectedAreaError",
    "ProtectedAreaProvider",
    "ProviderAvailability",
    "ProviderError",
    "ProviderHealth",
    "RasterHazardProvider",
    "RegionalProtectedAreaProvider",
    "SequenceClock",
    "SolarResourceProvider",
    "TerrainProvider",
    "UtcClock",
    "VectorHazardProvider",
    "VectorProtectedAreaProvider",
    "WeatherProvider",
    "WildfireRiskProvider",
    "WorldCoverLandCoverProvider",
    "cache_root_from_config",
    "hazard_available",
    "hazard_for_candidates",
    "hazard_for_site",
    "request_cache_key",
]

_LAZY_ATTRS = {
    "CopernicusDEMProvider": ("solardss.data.dem", "CopernicusDEMProvider"),
    "DEMTerrainProvider": ("solardss.data.dem", "DEMTerrainProvider"),
    "ESAWorldCoverProvider": ("solardss.data.worldcover", "ESAWorldCoverProvider"),
    "ERA5_LAND_DATASET_ID": ("solardss.data.era5_land", "ERA5_LAND_DATASET_ID"),
    "ERA5LandWeatherProvider": ("solardss.data.era5_land", "ERA5LandWeatherProvider"),
    "LandCoverRasterProvider": ("solardss.data.worldcover", "LandCoverRasterProvider"),
    "OSMBuildingProvider": ("solardss.data.osm", "OSMBuildingProvider"),
    "OSMInfrastructureProvider": ("solardss.data.osm", "OSMInfrastructureProvider"),
    "OSMOverpassProvider": ("solardss.data.osm", "OSMOverpassProvider"),
    "OverpassClient": ("solardss.data.osm", "OverpassClient"),
    "PVGIS_API_VERSION": ("solardss.data.pvgis", "PVGIS_API_VERSION"),
    "PVGIS_DEFAULT_BASE_URL": ("solardss.data.pvgis", "PVGIS_DEFAULT_BASE_URL"),
    "PVGISClient": ("solardss.data.pvgis", "PVGISClient"),
    "PVGISSolarResourceProvider": ("solardss.data.pvgis", "PVGISSolarResourceProvider"),
    "PVGISYieldProvider": ("solardss.data.pvgis", "PVGISYieldProvider"),
    "WorldCoverLandCoverProvider": ("solardss.data.worldcover", "WorldCoverLandCoverProvider"),
    "GenericVectorProtectedAreaProvider": (
        "solardss.data.protected",
        "GenericVectorProtectedAreaProvider",
    ),
    "RegionalProtectedAreaProvider": (
        "solardss.data.protected",
        "RegionalProtectedAreaProvider",
    ),
    "VectorProtectedAreaProvider": (
        "solardss.data.protected",
        "VectorProtectedAreaProvider",
    ),
    "FloodRiskProvider": ("solardss.data.hazard", "FloodRiskProvider"),
    "HailRiskProvider": ("solardss.data.hazard", "HailRiskProvider"),
    "HazardCatalog": ("solardss.data.hazard", "HazardCatalog"),
    "LandslideRiskProvider": ("solardss.data.hazard", "LandslideRiskProvider"),
    "RasterHazardProvider": ("solardss.data.hazard", "RasterHazardProvider"),
    "VectorHazardProvider": ("solardss.data.hazard", "VectorHazardProvider"),
    "WildfireRiskProvider": ("solardss.data.hazard", "WildfireRiskProvider"),
    "hazard_available": ("solardss.data.hazard", "hazard_available"),
    "hazard_for_candidates": ("solardss.data.hazard", "hazard_for_candidates"),
    "hazard_for_site": ("solardss.data.hazard", "hazard_for_site"),
}


def __getattr__(name: str) -> object:
    target = _LAZY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = target
    value = getattr(import_module(module_name), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
