"""Provider protocols. Implementations must not invent missing values."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from solardss.provenance.observation import IndicatorObservation
from solardss.providers.contracts import CoverageReport, ProviderAvailability, ProviderHealth
from solardss.providers.request import FetchRequest


@runtime_checkable
class DataProvider(Protocol):
    """Common retrieval surface for every SolarDSS data provider."""

    @property
    def provider_id(self) -> str:
        """Stable provider identity recorded in provenance."""

    def health(self) -> ProviderHealth:
        """Configuration and reachability. Does not fetch indicator values."""

    def availability(self, request: FetchRequest) -> ProviderAvailability:
        """Whether this provider can serve ``request`` in principle."""

    def coverage(self, request: FetchRequest) -> CoverageReport:
        """Whether the dataset covers the requested place and time."""

    def fetch(self, request: FetchRequest) -> IndicatorObservation:
        """Return an observation or a structured miss. Never fabricate a value."""


@runtime_checkable
class SolarResourceProvider(DataProvider, Protocol):
    """GHI, DNI, DHI, and related solar-resource layers."""


@runtime_checkable
class WeatherProvider(DataProvider, Protocol):
    """Air temperature, wind, humidity, cloud, precipitation, and similar."""


@runtime_checkable
class TerrainProvider(DataProvider, Protocol):
    """Elevation and other DEM-derived terrain layers supplied by a dataset."""


@runtime_checkable
class LandCoverProvider(DataProvider, Protocol):
    """Land-cover classes and class-area fractions."""


@runtime_checkable
class InfrastructureProvider(DataProvider, Protocol):
    """Distances to mapped infrastructure. Not grid hosting capacity."""


@runtime_checkable
class ProtectedAreaProvider(DataProvider, Protocol):
    """Designated protected-area overlap from a cadastre."""


@runtime_checkable
class HazardProvider(DataProvider, Protocol):
    """Flood, landslide, wildfire, hail, and similar hazard layers."""


@runtime_checkable
class BuildingProvider(DataProvider, Protocol):
    """Building footprints and rooftop attributes. Not structural safety."""


@runtime_checkable
class GridCapacityProvider(DataProvider, Protocol):
    """Interconnection / hosting capacity from a grid operator.

    Presence of a power line or substation is not capacity.
    """
