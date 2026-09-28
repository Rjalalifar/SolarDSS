"""PV identities, catalog, design evaluation, and yield engines."""

from __future__ import annotations

from solardss.pv.adapters import (
    KNOWN_FUTURE_ADAPTERS,
    AdapterAvailability,
    LocalFileAdapter,
    PVCatalogAdapter,
    UnimplementedDatabaseAdapter,
)
from solardss.pv.catalog import PVComponentCatalog, empty_pv_catalog
from solardss.pv.contracts import (
    BatterySystem,
    ComponentProvenance,
    ComponentSourceKind,
    InverterFilter,
    ModuleFilter,
    ModuleTechnology,
    PVInverter,
    PVModule,
)
from solardss.pv.design import (
    BoundedEnumerationGenerator,
    ConfigurationSearchSpace,
    PVDesignEngine,
    PVDesignProblem,
    PVDesignResult,
    PVSystemConfiguration,
    UnimplementedOptimizer,
    design_problem_from_decision,
)
from solardss.pv.energy import energy_from_capacity_factor, energy_from_irradiation
from solardss.pv.errors import (
    PVCatalogConfigError,
    PVCatalogError,
    PVCatalogSchemaError,
    PVYieldConfigError,
    PVYieldError,
    UnknownComponentError,
)
from solardss.pv.io import load_pv_catalog
from solardss.pv.pvgis_yield import (
    PVGISYieldEngine,
    azimuth_north_to_pvgis_aspect,
    yield_request_from_configuration,
)
from solardss.pv.pvlib_yield import PvlibYieldEngine, pvlib_module_available, require_pvlib
from solardss.pv.yield_contracts import (
    PvlibModelSelection,
    PVYieldEngine,
    PVYieldRequest,
    PVYieldResult,
    WeatherObservation,
    YieldValueOrigin,
)

__all__ = [
    "KNOWN_FUTURE_ADAPTERS",
    "AdapterAvailability",
    "BatterySystem",
    "BoundedEnumerationGenerator",
    "ComponentProvenance",
    "ComponentSourceKind",
    "ConfigurationSearchSpace",
    "InverterFilter",
    "LocalFileAdapter",
    "ModuleFilter",
    "ModuleTechnology",
    "PVCatalogAdapter",
    "PVCatalogConfigError",
    "PVCatalogError",
    "PVCatalogSchemaError",
    "PVComponentCatalog",
    "PVDesignEngine",
    "PVDesignProblem",
    "PVDesignResult",
    "PVGISYieldEngine",
    "PVInverter",
    "PVModule",
    "PVSystemConfiguration",
    "PVYieldConfigError",
    "PVYieldEngine",
    "PVYieldError",
    "PVYieldRequest",
    "PVYieldResult",
    "PvlibModelSelection",
    "PvlibYieldEngine",
    "UnimplementedDatabaseAdapter",
    "UnimplementedOptimizer",
    "UnknownComponentError",
    "WeatherObservation",
    "YieldValueOrigin",
    "azimuth_north_to_pvgis_aspect",
    "design_problem_from_decision",
    "empty_pv_catalog",
    "energy_from_capacity_factor",
    "energy_from_irradiation",
    "load_pv_catalog",
    "pvlib_module_available",
    "require_pvlib",
    "yield_request_from_configuration",
]
