"""Evaluate a PVSystemConfiguration against an explicit PVDesignProblem.

Missing datasheet fields fail a bound or are reported unknown. They are never
treated as zero. This is not a PV performance model.
"""

from __future__ import annotations

from solardss.core.hashing import fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.core.status import FeasibilityStatus
from solardss.decision.contracts import ConstraintOperator, UserConstraint
from solardss.indicators.ids import IndicatorId
from solardss.provenance.record import provenance_record
from solardss.pv.catalog import PVComponentCatalog
from solardss.pv.contracts import BatterySystem, PVInverter, PVModule
from solardss.pv.design.contracts import (
    DerivedDesignQuantities,
    DesignConstraintFailure,
    DesignConstraintKind,
    MountingType,
    PVDesignProblem,
    PVDesignResult,
    PVSystemConfiguration,
    TrackingMode,
    almost_equal,
    azimuth_delta,
    is_rooftop_project,
    mounting_for,
)
from solardss.pv.design.errors import PVDesignConfigError

_ANGLE_ATOL = 1e-6
_POWER_ATOL = 1e-6

_INDICATOR_FIELDS: dict[str, str] = {
    IndicatorId.PV_TILT.value: "tilt_deg",
    IndicatorId.PV_AZIMUTH.value: "azimuth_deg",
    IndicatorId.PV_DC_AC_RATIO.value: "dc_ac_ratio",
    IndicatorId.PV_ROW_SPACING.value: "row_spacing_m",
    IndicatorId.PV_GROUND_COVERAGE_RATIO.value: "ground_coverage_ratio",
    IndicatorId.PV_BATTERY_CAPACITY.value: "battery_capacity_kwh",
    IndicatorId.PV_SYSTEM_CAPACITY.value: "installed_capacity_kw",
    IndicatorId.PV_MODULE_MODEL.value: "module_id",
    IndicatorId.PV_INVERTER.value: "inverter_id",
    IndicatorId.PV_TRACKING_MODE.value: "tracking_mode",
    IndicatorId.PV_MODULE_TECHNOLOGY.value: "technology",
}


def evaluate_configuration(
    configuration: PVSystemConfiguration,
    problem: PVDesignProblem,
    catalog: PVComponentCatalog,
) -> PVDesignResult:
    """Screen one configuration. ``feasible`` is true only when every check passed."""

    module = catalog.get_module(configuration.module_id)
    inverter = catalog.get_inverter(configuration.inverter_id)
    battery: BatterySystem | None = None
    if configuration.battery_id is not None:
        battery = catalog.get_battery(configuration.battery_id)

    derived = _derive(configuration, module, inverter, battery)
    failed: list[DesignConstraintFailure] = []
    unknown: list[DesignConstraintFailure] = []
    warnings: list[str] = []
    if is_rooftop_project(problem.project_type) and problem.roof_area_m2 is None:
        warnings.append("roof_area_m2 was not supplied; roof area was not checked")
    if (not is_rooftop_project(problem.project_type)) and problem.available_area_m2 is None:
        warnings.append("available_area_m2 was not supplied; land area was not checked")

    def record(item: DesignConstraintFailure) -> None:
        if item.unknown:
            unknown.append(item)
        else:
            failed.append(item)

    for item in _location_and_mounting(configuration, problem):
        record(item)
    for item in _roof_geometry(configuration, problem):
        record(item)
    for item in _preferences(configuration, problem, module):
        record(item)
    for item in _module_dimensions(module, problem):
        record(item)
    for item in _capacity(configuration, derived, problem):
        record(item)
    for item in _area(derived, problem):
        record(item)
    for item in _electrical(configuration, derived, problem, module, inverter):
        record(item)
    for item in _battery(configuration, derived, problem, battery):
        record(item)
    for item in _user_constraints(configuration, derived, problem, module):
        record(item)

    if failed:
        status = FeasibilityStatus.EXCLUDED
    elif unknown:
        status = FeasibilityStatus.UNKNOWN
    else:
        status = FeasibilityStatus.FEASIBLE
    return PVDesignResult(
        configuration=configuration,
        status=status,
        feasible=status is FeasibilityStatus.FEASIBLE,
        derived=derived,
        failed_constraints=tuple(failed),
        unknown_constraints=tuple(unknown),
        warnings=tuple(warnings),
        provenance=provenance_record(
            MethodId.PV_DESIGN_EVALUATE,
            parameters={
                "configuration_id": configuration.configuration_id,
                "project_type": problem.project_type.value,
                "decision_mode": problem.decision_mode.value,
                "failed_count": len(failed),
                "unknown_count": len(unknown),
            },
            input_fingerprints=(
                fingerprint_canonical(
                    {
                        "configuration_id": configuration.configuration_id,
                        "module_id": configuration.module_id,
                        "inverter_id": configuration.inverter_id,
                    }
                ),
            ),
            notes=(
                "PV design evaluation does not run a performance model",
                "location is not optimized at this stage",
            ),
        ),
    )


def _derive(
    configuration: PVSystemConfiguration,
    module: PVModule,
    inverter: PVInverter,
    battery: BatterySystem | None,
) -> DerivedDesignQuantities:
    installed = float(configuration.module_count) * module.rated_power_w
    ac_w = float(configuration.inverter_count) * inverter.rated_ac_power_w
    dc_ac = installed / ac_w if ac_w > 0.0 else None
    module_area = _module_area(module)
    array_area = None if module_area is None else module_area * float(configuration.module_count)
    gcr = configuration.ground_coverage_ratio
    spacing = configuration.row_spacing_m
    if gcr is None and spacing is not None and module.length_m is not None:
        gcr = module.length_m / spacing if spacing > 0.0 else None
    occupied = _occupied_area(configuration.mounting_type, array_area, gcr)
    battery_kwh = configuration.battery_capacity_kwh
    if battery_kwh is None and battery is not None:
        battery_kwh = battery.usable_capacity_kwh
    return DerivedDesignQuantities(
        installed_capacity_w=installed,
        dc_ac_ratio=dc_ac,
        module_area_m2=module_area,
        array_area_m2=array_area,
        occupied_area_m2=occupied,
        ground_coverage_ratio=gcr,
        row_spacing_m=spacing,
        battery_capacity_kwh=battery_kwh,
    )


def _module_area(module: PVModule) -> float | None:
    if module.area_m2 is not None:
        return module.area_m2
    if module.length_m is not None and module.width_m is not None:
        return module.length_m * module.width_m
    return None


def _occupied_area(
    mounting: MountingType,
    array_area: float | None,
    gcr: float | None,
) -> float | None:
    if array_area is None:
        return None
    if mounting is MountingType.ROOFTOP:
        return array_area
    if gcr is None:
        return None
    return array_area / gcr


def _location_and_mounting(
    configuration: PVSystemConfiguration,
    problem: PVDesignProblem,
) -> tuple[DesignConstraintFailure, ...]:
    out: list[DesignConstraintFailure] = []
    if configuration.location != problem.location:
        out.append(
            _failure(
                "location_resolved",
                DesignConstraintKind.LOCATION,
                "configuration location must be the resolved design-problem site",
                actual=configuration.location.name or configuration.location.place_id,
                limit=problem.location.name or problem.location.place_id,
            )
        )
    expected = mounting_for(problem.project_type)
    if configuration.mounting_type is not expected:
        out.append(
            _failure(
                "mounting_type",
                DesignConstraintKind.MOUNTING,
                "mounting_type must match project_type",
                actual=configuration.mounting_type.value,
                limit=expected.value,
            )
        )
    if (
        is_rooftop_project(problem.project_type)
        and configuration.tracking_mode is not TrackingMode.FIXED
    ):
        out.append(
            _failure(
                "rooftop_tracking",
                DesignConstraintKind.MOUNTING,
                "rooftop tracking is not implied; tracking_mode must be fixed",
                actual=configuration.tracking_mode.value,
                limit=TrackingMode.FIXED.value,
            )
        )
    return tuple(out)


def _roof_geometry(
    configuration: PVSystemConfiguration,
    problem: PVDesignProblem,
) -> tuple[DesignConstraintFailure, ...]:
    out: list[DesignConstraintFailure] = []
    if (
        problem.roof_tilt_deg is not None
        and abs(configuration.tilt_deg - problem.roof_tilt_deg) > _ANGLE_ATOL
    ):
        out.append(
            _failure(
                "roof_tilt",
                DesignConstraintKind.ROOF_TILT,
                "tilt_deg is constrained by the roof plane and is not a free variable",
                actual=configuration.tilt_deg,
                limit=problem.roof_tilt_deg,
            )
        )
    if (
        problem.roof_azimuth_deg is not None
        and azimuth_delta(configuration.azimuth_deg, problem.roof_azimuth_deg) > _ANGLE_ATOL
    ):
        out.append(
            _failure(
                "roof_orientation",
                DesignConstraintKind.ROOF_ORIENTATION,
                "azimuth_deg is constrained by the roof plane and is not a free variable",
                actual=configuration.azimuth_deg,
                limit=problem.roof_azimuth_deg,
            )
        )
    return tuple(out)


def _preferences(
    configuration: PVSystemConfiguration,
    problem: PVDesignProblem,
    module: PVModule,
) -> tuple[DesignConstraintFailure, ...]:
    prefs = problem.module_preferences
    if prefs is None:
        return ()
    out: list[DesignConstraintFailure] = []
    if prefs.technology is not None and prefs.technology.casefold() != module.technology.casefold():
        out.append(
            _failure(
                "module_technology",
                DesignConstraintKind.USER_PREFERENCE,
                "module technology does not match module_preferences",
                actual=module.technology,
                limit=prefs.technology,
            )
        )
    if prefs.model is not None and prefs.model.casefold() != module.model.casefold():
        out.append(
            _failure(
                "module_model",
                DesignConstraintKind.USER_PREFERENCE,
                "module model does not match module_preferences",
                actual=module.model,
                limit=prefs.model,
            )
        )
    if (
        prefs.tracking_mode is not None
        and prefs.tracking_mode.casefold() != configuration.tracking_mode.value.casefold()
    ):
        out.append(
            _failure(
                "tracking_mode",
                DesignConstraintKind.USER_PREFERENCE,
                "tracking_mode does not match module_preferences",
                actual=configuration.tracking_mode.value,
                limit=prefs.tracking_mode,
            )
        )
    return tuple(out)


def _module_dimensions(
    module: PVModule,
    problem: PVDesignProblem,
) -> tuple[DesignConstraintFailure, ...]:
    checks = (
        ("max_module_length_m", module.length_m, problem.max_module_length_m, "length_m"),
        ("max_module_width_m", module.width_m, problem.max_module_width_m, "width_m"),
        (
            "max_module_thickness_m",
            module.thickness_m,
            problem.max_module_thickness_m,
            "thickness_m",
        ),
        ("max_module_weight_kg", module.weight_kg, problem.max_module_weight_kg, "weight_kg"),
    )
    out: list[DesignConstraintFailure] = []
    for constraint_id, actual, limit, label in checks:
        if limit is None:
            continue
        if actual is None:
            out.append(
                _failure(
                    constraint_id,
                    DesignConstraintKind.MODULE_DIMENSION,
                    f"module {label} is missing; the dimensional bound is not treated as passed",
                    limit=limit,
                    unknown=True,
                )
            )
        elif actual - limit > _POWER_ATOL:
            out.append(
                _failure(
                    constraint_id,
                    DesignConstraintKind.MODULE_DIMENSION,
                    f"module {label} exceeds the configured maximum",
                    actual=actual,
                    limit=limit,
                )
            )
    return tuple(out)


def _capacity(
    configuration: PVSystemConfiguration,
    derived: DerivedDesignQuantities,
    problem: PVDesignProblem,
) -> tuple[DesignConstraintFailure, ...]:
    out: list[DesignConstraintFailure] = []
    if configuration.installed_capacity_w is not None and not almost_equal(
        configuration.installed_capacity_w,
        derived.installed_capacity_w,
        atol=_POWER_ATOL,
    ):
        out.append(
            _failure(
                "installed_capacity_identity",
                DesignConstraintKind.REQUESTED_CAPACITY,
                "installed_capacity_w must equal module_count * rated_power_w",
                actual=configuration.installed_capacity_w,
                limit=derived.installed_capacity_w,
            )
        )
    floor_w = problem.capacity_floor_w
    if floor_w is not None and derived.installed_capacity_w + _POWER_ATOL < floor_w:
        out.append(
            _failure(
                "min_capacity",
                DesignConstraintKind.REQUESTED_CAPACITY,
                "installed DC capacity is below the requested minimum",
                actual=derived.installed_capacity_w,
                limit=floor_w,
            )
        )
    if (
        problem.max_capacity_w is not None
        and derived.installed_capacity_w - problem.max_capacity_w > _POWER_ATOL
    ):
        out.append(
            _failure(
                "max_capacity",
                DesignConstraintKind.REQUESTED_CAPACITY,
                "installed DC capacity exceeds the requested maximum",
                actual=derived.installed_capacity_w,
                limit=problem.max_capacity_w,
            )
        )
    return tuple(out)


def _area(
    derived: DerivedDesignQuantities,
    problem: PVDesignProblem,
) -> tuple[DesignConstraintFailure, ...]:
    out: list[DesignConstraintFailure] = []
    if is_rooftop_project(problem.project_type):
        limit = problem.roof_area_m2
        kind = DesignConstraintKind.ROOF_AREA
        constraint_id = "roof_area"
        label = "roof area"
    else:
        limit = problem.available_area_m2
        kind = DesignConstraintKind.AVAILABLE_AREA
        constraint_id = "available_area"
        label = "available area"
    if limit is None:
        return ()
    occupied = derived.occupied_area_m2
    if occupied is None:
        missing = "module area" if derived.array_area_m2 is None else "row spacing or GCR"
        out.append(
            _failure(
                constraint_id,
                kind,
                f"{missing} is missing; {label} cannot be checked and is not treated as passed",
                limit=limit,
                unknown=True,
            )
        )
        return tuple(out)
    if occupied - limit > _POWER_ATOL:
        out.append(
            _failure(
                constraint_id,
                kind,
                f"array occupied area exceeds {label}",
                actual=occupied,
                limit=limit,
            )
        )
    return tuple(out)


def _electrical(
    configuration: PVSystemConfiguration,
    derived: DerivedDesignQuantities,
    problem: PVDesignProblem,
    module: PVModule,
    inverter: PVInverter,
) -> tuple[DesignConstraintFailure, ...]:
    out: list[DesignConstraintFailure] = []
    if configuration.dc_ac_ratio is not None:
        if derived.dc_ac_ratio is None:
            out.append(
                _failure(
                    "dc_ac_identity",
                    DesignConstraintKind.ELECTRICAL_LIMIT,
                    "dc_ac_ratio could not be derived from inverter AC rating",
                    actual=configuration.dc_ac_ratio,
                    unknown=True,
                )
            )
        elif not almost_equal(configuration.dc_ac_ratio, derived.dc_ac_ratio, atol=_POWER_ATOL):
            out.append(
                _failure(
                    "dc_ac_identity",
                    DesignConstraintKind.ELECTRICAL_LIMIT,
                    "dc_ac_ratio must equal installed DC / inverter AC",
                    actual=configuration.dc_ac_ratio,
                    limit=derived.dc_ac_ratio,
                )
            )
    ratio = derived.dc_ac_ratio
    if problem.min_dc_ac_ratio is not None:
        if ratio is None:
            out.append(_unknown_ratio("min_dc_ac_ratio", problem.min_dc_ac_ratio))
        elif ratio + _POWER_ATOL < problem.min_dc_ac_ratio:
            out.append(
                _failure(
                    "min_dc_ac_ratio",
                    DesignConstraintKind.ELECTRICAL_LIMIT,
                    "DC/AC ratio is below the configured minimum",
                    actual=ratio,
                    limit=problem.min_dc_ac_ratio,
                )
            )
    if problem.max_dc_ac_ratio is not None:
        if ratio is None:
            out.append(_unknown_ratio("max_dc_ac_ratio", problem.max_dc_ac_ratio))
        elif ratio - problem.max_dc_ac_ratio > _POWER_ATOL:
            out.append(
                _failure(
                    "max_dc_ac_ratio",
                    DesignConstraintKind.ELECTRICAL_LIMIT,
                    "DC/AC ratio exceeds the configured maximum",
                    actual=ratio,
                    limit=problem.max_dc_ac_ratio,
                )
            )
    dc_cap = inverter.max_dc_input_w
    if dc_cap is not None:
        limit = dc_cap * float(configuration.inverter_count)
        if derived.installed_capacity_w - limit > _POWER_ATOL:
            out.append(
                _failure(
                    "inverter_max_dc_input",
                    DesignConstraintKind.INVERTER_LIMIT,
                    "installed DC capacity exceeds inverter max DC input",
                    actual=derived.installed_capacity_w,
                    limit=limit,
                )
            )
    if inverter.max_dc_voltage is not None:
        if module.voc is None:
            out.append(
                _failure(
                    "inverter_max_dc_voltage",
                    DesignConstraintKind.INVERTER_LIMIT,
                    "module voc is missing; inverter max DC voltage cannot be checked",
                    limit=inverter.max_dc_voltage,
                    unknown=True,
                )
            )
        elif module.voc - inverter.max_dc_voltage > _POWER_ATOL:
            out.append(
                _failure(
                    "inverter_max_dc_voltage",
                    DesignConstraintKind.INVERTER_LIMIT,
                    "module voc exceeds inverter max DC voltage even for a single module",
                    actual=module.voc,
                    limit=inverter.max_dc_voltage,
                )
            )
    if (
        module.max_system_voltage is not None
        and module.voc is not None
        and module.voc - module.max_system_voltage > _POWER_ATOL
    ):
        out.append(
            _failure(
                "module_max_system_voltage",
                DesignConstraintKind.ELECTRICAL_LIMIT,
                "module voc exceeds module max_system_voltage",
                actual=module.voc,
                limit=module.max_system_voltage,
            )
        )
    return tuple(out)


def _battery(
    configuration: PVSystemConfiguration,
    derived: DerivedDesignQuantities,
    problem: PVDesignProblem,
    battery: BatterySystem | None,
) -> tuple[DesignConstraintFailure, ...]:
    out: list[DesignConstraintFailure] = []
    capacity = derived.battery_capacity_kwh
    has_storage = (capacity is not None and capacity > _POWER_ATOL) or battery is not None
    if problem.battery_required and not has_storage:
        out.append(
            _failure(
                "battery_required",
                DesignConstraintKind.BATTERY,
                "battery_required is set but no battery size was supplied",
                actual=0.0,
            )
        )
    if (
        battery is not None
        and configuration.battery_capacity_kwh is not None
        and not almost_equal(
            configuration.battery_capacity_kwh,
            battery.usable_capacity_kwh,
            atol=_POWER_ATOL,
        )
    ):
        out.append(
            _failure(
                "battery_capacity_identity",
                DesignConstraintKind.BATTERY,
                "battery_capacity_kwh must equal the catalogued usable_capacity_kwh",
                actual=configuration.battery_capacity_kwh,
                limit=battery.usable_capacity_kwh,
            )
        )
    return tuple(out)


def _user_constraints(
    configuration: PVSystemConfiguration,
    derived: DerivedDesignQuantities,
    problem: PVDesignProblem,
    module: PVModule,
) -> tuple[DesignConstraintFailure, ...]:
    out: list[DesignConstraintFailure] = []
    for index, constraint in enumerate(problem.user_constraints):
        field = _constraint_field(constraint)
        actual = _field_value(field, configuration, derived, module)
        if actual is None:
            out.append(
                _failure(
                    f"user_constraint_{index}_{field}",
                    DesignConstraintKind.USER_PREFERENCE,
                    f"user constraint field {field!r} is missing and is not treated as passed",
                    unknown=True,
                )
            )
            continue
        if not _compare(constraint.operator, actual, constraint.value):
            out.append(
                _failure(
                    f"user_constraint_{index}_{field}",
                    DesignConstraintKind.USER_PREFERENCE,
                    "configuration does not satisfy a caller-supplied user constraint",
                    actual=_as_scalar(actual),
                    limit=_as_scalar(constraint.value),
                )
            )
    return tuple(out)


def _constraint_field(constraint: UserConstraint) -> str:
    if constraint.field:
        return constraint.field
    indicator = constraint.indicator_id
    if indicator is None:
        raise PVDesignConfigError("UserConstraint requires field or indicator_id")
    try:
        return _INDICATOR_FIELDS[indicator]
    except KeyError as exc:
        raise PVDesignConfigError(
            f"user constraint indicator {indicator!r} is not a PV design variable"
        ) from exc


def _field_value(
    field: str,
    configuration: PVSystemConfiguration,
    derived: DerivedDesignQuantities,
    module: PVModule,
) -> object:
    if field == "installed_capacity_kw":
        return derived.installed_capacity_w / 1000.0
    if field == "installed_capacity_w":
        return derived.installed_capacity_w
    if field == "dc_ac_ratio":
        return derived.dc_ac_ratio
    if field == "occupied_area_m2":
        return derived.occupied_area_m2
    if field == "technology":
        return module.technology
    if field == "tracking_mode":
        return configuration.tracking_mode.value
    if field == "mounting_type":
        return configuration.mounting_type.value
    if hasattr(configuration, field):
        return getattr(configuration, field)
    if hasattr(derived, field):
        return getattr(derived, field)
    raise PVDesignConfigError(f"unknown user constraint field {field!r}")


def _compare(operator: ConstraintOperator, actual: object, expected: object) -> bool:
    if operator is ConstraintOperator.EQ:
        return _equal(actual, expected)
    if operator is ConstraintOperator.NE:
        return not _equal(actual, expected)
    if operator is ConstraintOperator.INCLUDE:
        return _equal(actual, expected)
    if operator is ConstraintOperator.EXCLUDE:
        return not _equal(actual, expected)
    left = _as_float(actual)
    right = _as_float(expected)
    if left is None or right is None:
        return False
    if operator is ConstraintOperator.LT:
        return left < right
    if operator is ConstraintOperator.LE:
        return left <= right + _POWER_ATOL
    if operator is ConstraintOperator.GT:
        return left > right
    if operator is ConstraintOperator.GE:
        return left + _POWER_ATOL >= right
    return False


def _equal(actual: object, expected: object) -> bool:
    if isinstance(actual, str) or isinstance(expected, str):
        return str(actual).casefold() == str(expected).casefold()
    left = _as_float(actual)
    right = _as_float(expected)
    if left is not None and right is not None:
        return almost_equal(left, right, atol=_POWER_ATOL)
    return actual == expected


def _as_float(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _as_scalar(value: object) -> float | str | bool | None:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int | float):
        return float(value)
    return str(value)


def _unknown_ratio(constraint_id: str, limit: float) -> DesignConstraintFailure:
    return _failure(
        constraint_id,
        DesignConstraintKind.ELECTRICAL_LIMIT,
        "DC/AC ratio could not be derived; the bound is not treated as passed",
        limit=limit,
        unknown=True,
    )


def _failure(
    constraint_id: str,
    kind: DesignConstraintKind,
    message: str,
    *,
    actual: float | str | bool | None = None,
    limit: float | str | bool | None = None,
    unknown: bool = False,
) -> DesignConstraintFailure:
    return DesignConstraintFailure(
        constraint_id=constraint_id,
        kind=kind,
        message=message,
        actual=actual,
        limit=limit,
        unknown=unknown,
    )
