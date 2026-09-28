"""In-memory PV component catalog. The default catalog is empty."""

from __future__ import annotations

from collections.abc import Sequence

from solardss.core.hashing import fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.provenance.record import ProvenanceRecord, provenance_record
from solardss.pv.contracts import (
    BatterySystem,
    InverterFilter,
    ModuleFilter,
    PVInverter,
    PVModule,
)
from solardss.pv.errors import PVCatalogConfigError, UnknownComponentError


class PVComponentCatalog:
    """Validated modules, inverters, and batteries.

    Construction does not download a manufacturer database. An empty catalog
    stays empty. Test fixtures are omitted from listings unless explicitly
    requested.
    """

    def __init__(
        self,
        *,
        modules: Sequence[PVModule] = (),
        inverters: Sequence[PVInverter] = (),
        batteries: Sequence[BatterySystem] = (),
        source_name: str = "empty",
    ) -> None:
        self._modules = _unique_modules(modules)
        self._inverters = _unique_inverters(inverters)
        self._batteries = _unique_batteries(batteries)
        self._source_name = source_name
        self._provenance = provenance_record(
            MethodId.PV_CATALOG_LOAD,
            parameters={
                "source_name": source_name,
                "module_count": len(self._modules),
                "inverter_count": len(self._inverters),
                "battery_count": len(self._batteries),
                "fixture_module_count": sum(
                    1 for item in self._modules.values() if item.provenance.fixture
                ),
            },
            input_fingerprints=(
                fingerprint_canonical(
                    {
                        "modules": sorted(self._modules),
                        "inverters": sorted(self._inverters),
                        "batteries": sorted(self._batteries),
                    }
                ),
            ),
            notes=(
                "SolarDSS does not ship manufacturer datasheets",
                "test fixtures are not commercial products",
            ),
        )

    @classmethod
    def empty(cls) -> PVComponentCatalog:
        """Return a catalog with no components."""

        return cls(source_name="empty")

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def provenance(self) -> ProvenanceRecord:
        return self._provenance

    def __len__(self) -> int:
        return len(self._modules) + len(self._inverters) + len(self._batteries)

    def get_module(self, module_id: str) -> PVModule:
        try:
            return self._modules[module_id]
        except KeyError as exc:
            raise UnknownComponentError(f"Unknown PV module {module_id!r}") from exc

    def get_inverter(self, inverter_id: str) -> PVInverter:
        try:
            return self._inverters[inverter_id]
        except KeyError as exc:
            raise UnknownComponentError(f"Unknown PV inverter {inverter_id!r}") from exc

    def get_battery(self, battery_id: str) -> BatterySystem:
        try:
            return self._batteries[battery_id]
        except KeyError as exc:
            raise UnknownComponentError(f"Unknown battery system {battery_id!r}") from exc

    def list_modules(self, query: ModuleFilter | None = None) -> tuple[PVModule, ...]:
        include_fixtures = False if query is None else query.include_fixtures
        items = tuple(
            item
            for item in sorted(self._modules.values(), key=lambda row: row.module_id)
            if include_fixtures or not item.provenance.fixture
        )
        if query is None:
            return items
        return tuple(item for item in items if _module_matches(item, query))

    def list_inverters(self, query: InverterFilter | None = None) -> tuple[PVInverter, ...]:
        include_fixtures = False if query is None else query.include_fixtures
        items = tuple(
            item
            for item in sorted(self._inverters.values(), key=lambda row: row.inverter_id)
            if include_fixtures or not item.provenance.fixture
        )
        if query is None:
            return items
        return tuple(item for item in items if _inverter_matches(item, query))

    def list_batteries(self, *, include_fixtures: bool = False) -> tuple[BatterySystem, ...]:
        return tuple(
            item
            for item in sorted(self._batteries.values(), key=lambda row: row.battery_id)
            if include_fixtures or not item.provenance.fixture
        )

    def merged(self, other: PVComponentCatalog) -> PVComponentCatalog:
        """Return a new catalog with ``other`` appended. Duplicate ids are rejected."""

        return PVComponentCatalog(
            modules=(*self._modules.values(), *other._modules.values()),
            inverters=(*self._inverters.values(), *other._inverters.values()),
            batteries=(*self._batteries.values(), *other._batteries.values()),
            source_name=f"{self._source_name}+{other._source_name}",
        )


def empty_pv_catalog() -> PVComponentCatalog:
    """Public constructor for an empty catalog."""

    return PVComponentCatalog.empty()


def _unique_modules(rows: Sequence[PVModule]) -> dict[str, PVModule]:
    out: dict[str, PVModule] = {}
    for item in rows:
        if item.module_id in out:
            raise PVCatalogConfigError(f"duplicate module_id {item.module_id!r}")
        out[item.module_id] = item
    return out


def _unique_inverters(rows: Sequence[PVInverter]) -> dict[str, PVInverter]:
    out: dict[str, PVInverter] = {}
    for item in rows:
        if item.inverter_id in out:
            raise PVCatalogConfigError(f"duplicate inverter_id {item.inverter_id!r}")
        out[item.inverter_id] = item
    return out


def _unique_batteries(rows: Sequence[BatterySystem]) -> dict[str, BatterySystem]:
    out: dict[str, BatterySystem] = {}
    for item in rows:
        if item.battery_id in out:
            raise PVCatalogConfigError(f"duplicate battery_id {item.battery_id!r}")
        out[item.battery_id] = item
    return out


def _module_matches(item: PVModule, query: ModuleFilter) -> bool:
    if query.technology is not None:
        wanted = {token.casefold() for token in query.technology}
        if item.technology.casefold() not in wanted:
            return False
    if not _in_range(item.efficiency, query.min_efficiency, query.max_efficiency):
        return False
    if not _in_range(item.rated_power_w, query.min_rated_power_w, query.max_rated_power_w):
        return False
    if not _in_range(
        item.temperature_coefficient_pmax,
        query.min_temperature_coefficient_pmax,
        query.max_temperature_coefficient_pmax,
    ):
        return False
    if not _at_most(item.length_m, query.max_length_m):
        return False
    if not _at_most(item.width_m, query.max_width_m):
        return False
    if not _at_most(item.thickness_m, query.max_thickness_m):
        return False
    if not _at_most(item.area_m2, query.max_area_m2):
        return False
    if not _at_most(item.weight_kg, query.max_weight_kg):
        return False
    return query.bifacial is None or item.bifacial is query.bifacial


def _inverter_matches(item: PVInverter, query: InverterFilter) -> bool:
    if not _in_range(item.rated_ac_power_w, query.min_rated_ac_power_w, query.max_rated_ac_power_w):
        return False
    return _in_range(item.efficiency, query.min_efficiency, query.max_efficiency)


def _in_range(value: float | None, lo: float | None, hi: float | None) -> bool:
    if lo is None and hi is None:
        return True
    if value is None:
        return False
    if lo is not None and value < lo:
        return False
    return hi is None or value <= hi


def _at_most(value: float | None, limit: float | None) -> bool:
    if limit is None:
        return True
    if value is None:
        return False
    return value <= limit
