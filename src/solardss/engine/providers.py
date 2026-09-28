"""Registered provider bindings. Providers do not look up other providers."""

from __future__ import annotations

from collections.abc import Sequence

from solardss.engine.errors import IndicatorEngineError
from solardss.providers.contracts import AvailabilityStatus, ProviderAvailability
from solardss.providers.protocols import DataProvider
from solardss.providers.request import FetchRequest


class ProviderBinding:
    """One provider and the indicator ids it may be asked to serve."""

    def __init__(
        self,
        provider: DataProvider,
        *,
        indicators: frozenset[str] | None = None,
        priority: int = 0,
    ) -> None:
        self.provider = provider
        self.indicators = indicators
        self.priority = priority

    def serves(self, indicator_id: str) -> bool:
        if self.indicators is None:
            return True
        return indicator_id in self.indicators


class ProviderRegistry:
    """Ordered catalog used by the engine. Selection is availability, not guessing."""

    def __init__(self, bindings: Sequence[ProviderBinding] = ()) -> None:
        self._bindings = list(bindings)

    def register(
        self,
        provider: DataProvider,
        *,
        indicators: Sequence[str] | None = None,
        priority: int = 0,
    ) -> None:
        """Append a provider. Lower ``priority`` is tried first."""

        served = _served_indicators(provider) if indicators is None else frozenset(indicators)
        self._bindings.append(ProviderBinding(provider, indicators=served, priority=priority))
        self._bindings.sort(key=lambda item: (item.priority, item.provider.provider_id))

    def providers_for(self, indicator_id: str) -> tuple[DataProvider, ...]:
        return tuple(item.provider for item in self._bindings if item.serves(indicator_id))

    def select(
        self, indicator_id: str, request: FetchRequest
    ) -> tuple[DataProvider, ProviderAvailability] | None:
        """First provider that reports AVAILABLE for ``request``.

        If none are available, return the first binding that serves the indicator
        together with its availability report, or ``None`` when nobody serves it.
        """

        fallback: tuple[DataProvider, ProviderAvailability] | None = None
        for provider in self.providers_for(indicator_id):
            availability = provider.availability(request)
            if availability.status is AvailabilityStatus.AVAILABLE:
                return provider, availability
            if fallback is None:
                fallback = (provider, availability)
        return fallback

    def __len__(self) -> int:
        return len(self._bindings)

    def provider_ids(self) -> tuple[str, ...]:
        return tuple(item.provider.provider_id for item in self._bindings)


def _served_indicators(provider: DataProvider) -> frozenset[str] | None:
    served = getattr(provider, "served_indicators", None)
    if served is None:
        return None
    if isinstance(served, set | frozenset | tuple | list):
        return frozenset(str(item) for item in served)
    raise IndicatorEngineError("served_indicators must be a collection of indicator ids")


def registry_from_providers(
    providers: Sequence[DataProvider | tuple[DataProvider, Sequence[str]]],
) -> ProviderRegistry:
    """Build a registry from providers or ``(provider, indicator ids)`` pairs."""

    catalog = ProviderRegistry()
    for item in providers:
        if isinstance(item, tuple):
            provider, indicators = item
            catalog.register(provider, indicators=indicators)
        else:
            catalog.register(item)
    return catalog
