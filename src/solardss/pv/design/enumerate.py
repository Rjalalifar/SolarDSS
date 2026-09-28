"""Bounded cartesian enumeration of PV configurations. Not a global optimizer."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence

from solardss.pv.catalog import PVComponentCatalog
from solardss.pv.contracts import InverterFilter, ModuleFilter, PVInverter, PVModule
from solardss.pv.design.contracts import (
    ConfigurationSearchSpace,
    MountingType,
    PVDesignProblem,
    PVSystemConfiguration,
    TrackingMode,
    almost_equal,
    ceil_count,
    is_rooftop_project,
    mounting_for,
)
from solardss.pv.design.errors import PVDesignConfigError


def enumerate_configurations(
    problem: PVDesignProblem,
    catalog: PVComponentCatalog,
) -> tuple[tuple[PVSystemConfiguration, ...], bool, tuple[str, ...]]:
    """Emit up to ``search_space.max_candidates`` configurations.

    Rooftop tilt/azimuth come from the roof plane when those fields are set.
    Ground-mount tilt, azimuth, and row spacing stay caller-supplied decision
    variables. No LLM and no external solver is used.
    """

    space = problem.search_space
    modules = _modules(problem, catalog)
    inverters = _inverters(problem, catalog)
    warnings: list[str] = []
    if not modules or not inverters:
        warnings.append("catalog has no modules or inverters to enumerate")
        return (), False, tuple(warnings)

    mounting = mounting_for(problem.project_type)
    tilts = _tilts(problem, warnings)
    azimuths = _azimuths(problem, warnings)
    trackings = _trackings(problem)
    if is_rooftop_project(problem.project_type) and space.tracking_modes and not trackings:
        warnings.append("rooftop tracking modes other than fixed were not enumerated")
    if not tilts or not azimuths or not trackings:
        return (), False, tuple(warnings)

    spacings, gcrs = _ground_slots(problem, mounting)
    batteries = _battery_slots(problem, catalog)
    groups: list[Sequence[object]] = [
        modules,
        inverters,
        space.inverter_counts,
        tilts,
        azimuths,
        trackings,
        spacings,
        gcrs,
        batteries,
    ]

    emitted: list[PVSystemConfiguration] = []
    truncated = False
    seen: set[str] = set()
    for combo in itertools.product(*groups):
        module = combo[0]
        inverter = combo[1]
        inverter_count = combo[2]
        tilt = combo[3]
        azimuth = combo[4]
        tracking = combo[5]
        spacing = combo[6]
        gcr = combo[7]
        battery_slot = combo[8]
        if not isinstance(module, PVModule) or not isinstance(inverter, PVInverter):
            raise PVDesignConfigError("enumeration produced a non-component catalog row")
        if not isinstance(inverter_count, int) or isinstance(inverter_count, bool):
            raise PVDesignConfigError("inverter_count slots must be integers")
        tilt_f = _as_float(tilt)
        azimuth_f = _as_float(azimuth)
        if tilt_f is None or azimuth_f is None:
            raise PVDesignConfigError("tilt and azimuth slots must be numeric")
        if not isinstance(tracking, TrackingMode):
            raise PVDesignConfigError("tracking slots must be TrackingMode values")
        if not isinstance(battery_slot, tuple) or len(battery_slot) != 2:
            raise PVDesignConfigError("battery slots must be (id, kWh) pairs")
        battery_id = battery_slot[0] if isinstance(battery_slot[0], str) else None
        battery_kwh = _as_float(battery_slot[1])
        counts = _module_counts(space, module, problem)
        for count in counts:
            config = _build(
                problem,
                mounting=mounting,
                module=module,
                module_count=count,
                inverter=inverter,
                inverter_count=inverter_count,
                tilt=tilt_f,
                azimuth=azimuth_f,
                tracking=tracking,
                spacing=_as_float(spacing),
                gcr=_as_float(gcr),
                battery_id=battery_id,
                battery_kwh=battery_kwh,
            )
            if config.configuration_id in seen:
                continue
            seen.add(config.configuration_id)
            emitted.append(config)
            if len(emitted) >= space.max_candidates:
                truncated = True
                break
        if truncated:
            break

    if truncated:
        warnings.append(
            f"enumeration stopped at max_candidates={space.max_candidates}; "
            "this is not a globally optimal design"
        )
    return tuple(emitted), truncated, tuple(warnings)


def _modules(problem: PVDesignProblem, catalog: PVComponentCatalog) -> tuple[PVModule, ...]:
    space = problem.search_space
    prefs = problem.module_preferences
    technology = None if prefs is None else prefs.technology
    listed = catalog.list_modules(
        ModuleFilter(include_fixtures=problem.include_fixtures, technology=technology)
    )
    if space.module_ids:
        wanted = []
        for module_id in space.module_ids:
            item = catalog.get_module(module_id)
            if not problem.include_fixtures and item.provenance.fixture:
                raise PVDesignConfigError(
                    f"search-space module {module_id!r} is a test fixture; "
                    "set include_fixtures=True to use it"
                )
            wanted.append(item)
        listed_ids = {item.module_id for item in listed}
        return tuple(item for item in wanted if item.module_id in listed_ids)
    if prefs is not None and prefs.model is not None:
        listed = tuple(item for item in listed if item.model.casefold() == prefs.model.casefold())
    return listed


def _inverters(problem: PVDesignProblem, catalog: PVComponentCatalog) -> tuple[PVInverter, ...]:
    space = problem.search_space
    listed = catalog.list_inverters(InverterFilter(include_fixtures=problem.include_fixtures))
    if space.inverter_ids:
        wanted = []
        for inverter_id in space.inverter_ids:
            item = catalog.get_inverter(inverter_id)
            if not problem.include_fixtures and item.provenance.fixture:
                raise PVDesignConfigError(
                    f"search-space inverter {inverter_id!r} is a test fixture; "
                    "set include_fixtures=True to use it"
                )
            wanted.append(item)
        listed_ids = {item.inverter_id for item in listed}
        return tuple(item for item in wanted if item.inverter_id in listed_ids)
    return listed


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _contains_angle(values: tuple[float, ...], angle: float) -> bool:
    return any(almost_equal(angle, item, atol=1e-6) for item in values)


def _tilts(problem: PVDesignProblem, warnings: list[str]) -> tuple[float, ...]:
    space = problem.search_space
    listed = tuple(float(item) for item in space.tilt_deg)
    if problem.roof_tilt_deg is not None:
        if listed and not _contains_angle(listed, problem.roof_tilt_deg):
            warnings.append(
                "roof_tilt_deg is a constraint; search_space.tilt_deg values that "
                "disagree with the roof plane were not enumerated"
            )
            return ()
        return (float(problem.roof_tilt_deg),)
    if listed:
        return listed
    raise PVDesignConfigError(
        "search_space.tilt_deg is required when the roof tilt is not a fixed constraint"
    )


def _azimuths(problem: PVDesignProblem, warnings: list[str]) -> tuple[float, ...]:
    space = problem.search_space
    listed = tuple(float(item) for item in space.azimuth_deg)
    if problem.roof_azimuth_deg is not None:
        if listed and not _contains_angle(listed, problem.roof_azimuth_deg):
            warnings.append(
                "roof_azimuth_deg is a constraint; search_space.azimuth_deg values that "
                "disagree with the roof plane were not enumerated"
            )
            return ()
        return (float(problem.roof_azimuth_deg),)
    if listed:
        return listed
    raise PVDesignConfigError(
        "search_space.azimuth_deg is required when the roof orientation is not fixed"
    )


def _trackings(problem: PVDesignProblem) -> tuple[TrackingMode, ...]:
    space = problem.search_space
    prefs = problem.module_preferences
    if space.tracking_modes:
        modes = space.tracking_modes
    elif prefs is not None and prefs.tracking_mode:
        modes = (TrackingMode(prefs.tracking_mode),)
    else:
        modes = (TrackingMode.FIXED,)
    if is_rooftop_project(problem.project_type):
        return tuple(mode for mode in modes if mode is TrackingMode.FIXED) or ()
    return modes


def _ground_slots(
    problem: PVDesignProblem,
    mounting: MountingType,
) -> tuple[tuple[float | None, ...], tuple[float | None, ...]]:
    if mounting is MountingType.ROOFTOP:
        return (None,), (None,)
    space = problem.search_space
    spacings: tuple[float | None, ...] = space.row_spacing_m or (None,)
    gcrs: tuple[float | None, ...] = space.ground_coverage_ratio or (None,)
    return spacings, gcrs


def _battery_slots(
    problem: PVDesignProblem,
    catalog: PVComponentCatalog,
) -> tuple[tuple[str | None, float | None], ...]:
    space = problem.search_space
    if space.battery_ids:
        slots: list[tuple[str | None, float | None]] = []
        for battery_id in space.battery_ids:
            item = catalog.get_battery(battery_id)
            if not problem.include_fixtures and item.provenance.fixture:
                raise PVDesignConfigError(
                    f"search-space battery {battery_id!r} is a test fixture; "
                    "set include_fixtures=True to use it"
                )
            slots.append((item.battery_id, item.usable_capacity_kwh))
        return tuple(slots)
    if space.battery_capacity_kwh:
        return tuple((None, capacity) for capacity in space.battery_capacity_kwh)
    return ((None, None),)


def _module_counts(
    space: ConfigurationSearchSpace,
    module: PVModule,
    problem: PVDesignProblem,
) -> tuple[int, ...]:
    if space.module_counts:
        return space.module_counts
    counts: set[int] = set()
    floor_w = problem.capacity_floor_w
    if floor_w is not None:
        counts.add(ceil_count(floor_w, module.rated_power_w))
    if problem.max_capacity_w is not None:
        n_max = math.floor(problem.max_capacity_w / module.rated_power_w + 1e-12)
        if n_max >= 1:
            counts.add(n_max)
    if is_rooftop_project(problem.project_type):
        area = problem.roof_area_m2
    else:
        area = problem.available_area_m2
    module_area = module.area_m2
    if module_area is None and module.length_m is not None and module.width_m is not None:
        module_area = module.length_m * module.width_m
    if area is not None and module_area is not None and module_area > 0.0:
        n_area = math.floor(area / module_area + 1e-12)
        if n_area >= 1:
            counts.add(n_area)
    if not counts:
        raise PVDesignConfigError(
            "search_space.module_counts is required when capacity and area bounds "
            "cannot size the array"
        )
    return tuple(sorted(counts))


def _build(
    problem: PVDesignProblem,
    *,
    mounting: MountingType,
    module: PVModule,
    module_count: int,
    inverter: PVInverter,
    inverter_count: int,
    tilt: float,
    azimuth: float,
    tracking: TrackingMode,
    spacing: float | None,
    gcr: float | None,
    battery_id: str | None,
    battery_kwh: float | None,
) -> PVSystemConfiguration:
    installed = float(module_count) * module.rated_power_w
    ac_w = float(inverter_count) * inverter.rated_ac_power_w
    dc_ac = installed / ac_w if ac_w > 0.0 else None
    return PVSystemConfiguration(
        location=problem.location,
        module_id=module.module_id,
        module_count=module_count,
        installed_capacity_w=installed,
        tilt_deg=tilt,
        azimuth_deg=azimuth,
        mounting_type=mounting,
        tracking_mode=tracking,
        inverter_id=inverter.inverter_id,
        inverter_count=inverter_count,
        dc_ac_ratio=dc_ac,
        row_spacing_m=spacing,
        ground_coverage_ratio=gcr,
        battery_id=battery_id,
        battery_capacity_kwh=battery_kwh,
    )
