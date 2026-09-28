"""Default registry and backward-compatible catalog helpers."""

from __future__ import annotations

from solardss.indicators.contracts import IndicatorSpec
from solardss.indicators.definitions import ALL_INDICATORS
from solardss.indicators.ids import IndicatorId
from solardss.indicators.registry import IndicatorRegistry, mapping_from_registry

registry = IndicatorRegistry(ALL_INDICATORS)
INDICATOR_CATALOG: dict[str, IndicatorSpec] = dict(mapping_from_registry(registry))


def get_indicator(indicator_id: str | IndicatorId) -> IndicatorSpec:
    """Return a registry entry by canonical id or documented alias."""

    return registry.get(indicator_id)


def catalog_ids() -> tuple[str, ...]:
    """Sorted canonical indicator ids."""

    return tuple(spec.id for spec in registry.list_all())
