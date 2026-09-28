"""PV configuration generation and evaluation after a site is resolved."""

from __future__ import annotations

from pathlib import Path

import pytest

from solardss.core.status import FeasibilityStatus
from solardss.data.geo import GeographicRef
from solardss.decision import SolarDecisionProblem
from solardss.decision.contracts import ConstraintOperator, ModulePreferences, UserConstraint
from solardss.exceptions import UnsupportedMethodError
from solardss.indicators import Controllability, DecisionMode, ProjectType, SiteSelectionScope
from solardss.pv import (
    PVDesignEngine,
    PVDesignProblem,
    PVSystemConfiguration,
    UnimplementedOptimizer,
    design_problem_from_decision,
    empty_pv_catalog,
)
from solardss.pv.catalog import PVComponentCatalog
from solardss.pv.design.contracts import (
    ConfigurationSearchSpace,
    DesignConstraintKind,
    MountingType,
    TrackingMode,
)
from solardss.pv.design.errors import PVDesignConfigError
from solardss.pv.fixtures import TEST_INVERTER, TEST_MODULE_MONO, labelled_fixture_catalog

SITE = GeographicRef(name="Resolved Test Site", longitude=13.4, latitude=52.52)


def _space(**kwargs: object) -> ConfigurationSearchSpace:
    payload: dict[str, object] = {
        "module_ids": (TEST_MODULE_MONO.module_id,),
        "inverter_ids": (TEST_INVERTER.inverter_id,),
        "module_counts": (25,),
        "tilt_deg": (20.0,),
        "azimuth_deg": (180.0,),
    }
    payload.update(kwargs)
    return ConfigurationSearchSpace.model_validate(payload)


def _roof_problem(**kwargs: object) -> PVDesignProblem:
    payload: dict[str, object] = {
        "location": SITE,
        "project_type": ProjectType.RESIDENTIAL_ROOFTOP,
        "roof_area_m2": 60.0,
        "roof_tilt_deg": 20.0,
        "roof_azimuth_deg": 180.0,
        "target_capacity_w": 10_000.0,
        "include_fixtures": True,
        "search_space": _space(),
    }
    payload.update(kwargs)
    return PVDesignProblem.model_validate(payload)


def _ground_problem(**kwargs: object) -> PVDesignProblem:
    payload: dict[str, object] = {
        "location": SITE,
        "project_type": ProjectType.GROUND_MOUNTED,
        "available_area_m2": 200.0,
        "target_capacity_w": 10_000.0,
        "include_fixtures": True,
        "search_space": _space(row_spacing_m=(4.0,), tilt_deg=(15.0, 25.0)),
    }
    payload.update(kwargs)
    return PVDesignProblem.model_validate(payload)


def _config(**kwargs: object) -> PVSystemConfiguration:
    payload: dict[str, object] = {
        "location": SITE,
        "module_id": TEST_MODULE_MONO.module_id,
        "module_count": 25,
        "installed_capacity_w": 10_000.0,
        "tilt_deg": 20.0,
        "azimuth_deg": 180.0,
        "mounting_type": MountingType.ROOFTOP,
        "tracking_mode": TrackingMode.FIXED,
        "inverter_id": TEST_INVERTER.inverter_id,
        "inverter_count": 1,
        "dc_ac_ratio": 1.0,
    }
    payload.update(kwargs)
    return PVSystemConfiguration.model_validate(payload)


def _engine() -> PVDesignEngine:
    return PVDesignEngine(labelled_fixture_catalog())


def test_feasible_rooftop_configuration_meets_area_and_capacity() -> None:
    result = _engine().evaluate(_config(), _roof_problem())
    assert result.feasible is True
    assert result.status is FeasibilityStatus.FEASIBLE
    assert result.derived.installed_capacity_w == 10_000.0
    assert result.derived.occupied_area_m2 == 50.0
    assert result.provenance.stamp.method_id == "pv_design_evaluate"


def test_roof_tilt_and_azimuth_are_constraints_not_free_variables() -> None:
    engine = _engine()
    problem = _roof_problem()
    bad_tilt = engine.evaluate(_config(tilt_deg=35.0), problem)
    assert bad_tilt.feasible is False
    assert any(item.kind is DesignConstraintKind.ROOF_TILT for item in bad_tilt.failed_constraints)
    bad_az = engine.evaluate(_config(azimuth_deg=90.0), problem)
    assert any(
        item.kind is DesignConstraintKind.ROOF_ORIENTATION for item in bad_az.failed_constraints
    )


def test_rooftop_enumeration_does_not_search_free_tilt() -> None:
    result = _engine().generate(_roof_problem())
    assert result.results
    assert {item.configuration.tilt_deg for item in result.results} == {20.0}
    assert {item.configuration.azimuth_deg for item in result.results} == {180.0}
    assert all(item.configuration.tracking_mode is TrackingMode.FIXED for item in result.results)


def test_rooftop_tracking_is_rejected() -> None:
    result = _engine().evaluate(_config(tracking_mode=TrackingMode.SINGLE_AXIS), _roof_problem())
    assert result.feasible is False
    assert any(item.kind is DesignConstraintKind.MOUNTING for item in result.failed_constraints)


def test_roof_area_constraint() -> None:
    result = _engine().evaluate(
        _config(module_count=31, installed_capacity_w=12_400.0, dc_ac_ratio=1.24),
        _roof_problem(),
    )
    assert result.feasible is False
    kinds = {item.kind for item in result.failed_constraints}
    assert DesignConstraintKind.ROOF_AREA in kinds


def test_requested_capacity_constraint() -> None:
    result = _engine().evaluate(
        _config(module_count=20, installed_capacity_w=8_000.0, dc_ac_ratio=0.8),
        _roof_problem(),
    )
    assert result.feasible is False
    assert any(
        item.kind is DesignConstraintKind.REQUESTED_CAPACITY for item in result.failed_constraints
    )


def test_inverter_dc_input_limit() -> None:
    result = _engine().evaluate(
        _config(module_count=38, installed_capacity_w=15_200.0, dc_ac_ratio=1.52),
        _roof_problem(roof_area_m2=80.0, max_capacity_w=20_000.0, target_capacity_w=None),
    )
    assert result.feasible is False
    assert any(item.constraint_id == "inverter_max_dc_input" for item in result.failed_constraints)


def test_inverter_voltage_limit() -> None:
    inverter = TEST_INVERTER.model_validate(
        {**TEST_INVERTER.model_dump(), "max_dc_voltage": 40.0, "min_dc_voltage": 20.0}
    )
    catalog = PVComponentCatalog(
        modules=(TEST_MODULE_MONO,),
        inverters=(inverter,),
        source_name="voltage-limit",
    )
    result = PVDesignEngine(catalog).evaluate(_config(), _roof_problem())
    assert result.feasible is False
    assert any(
        item.constraint_id == "inverter_max_dc_voltage" for item in result.failed_constraints
    )


def test_module_dimension_constraint() -> None:
    result = _engine().evaluate(_config(), _roof_problem(max_module_length_m=1.5))
    assert result.feasible is False
    assert any(
        item.kind is DesignConstraintKind.MODULE_DIMENSION for item in result.failed_constraints
    )


def test_module_preference_technology() -> None:
    problem = _roof_problem(
        module_preferences=ModulePreferences(technology="CdTe"),
    )
    result = _engine().evaluate(_config(), problem)
    assert result.feasible is False
    assert any(
        item.kind is DesignConstraintKind.USER_PREFERENCE for item in result.failed_constraints
    )


def test_user_constraint_dc_ac_ratio() -> None:
    problem = _roof_problem(
        user_constraints=(
            UserConstraint(field="dc_ac_ratio", operator=ConstraintOperator.LE, value=1.05),
        )
    )
    ok = _engine().evaluate(_config(), problem)
    assert ok.feasible is True
    high = _engine().evaluate(
        _config(module_count=30, installed_capacity_w=12_000.0, dc_ac_ratio=1.2),
        problem,
    )
    assert high.feasible is False


def test_missing_module_area_is_unknown_not_passed() -> None:
    module = TEST_MODULE_MONO.model_validate(
        {**TEST_MODULE_MONO.model_dump(), "area_m2": None, "length_m": None, "width_m": None}
    )
    catalog = PVComponentCatalog(
        modules=(module,),
        inverters=(TEST_INVERTER,),
        source_name="no-area",
    )
    result = PVDesignEngine(catalog).evaluate(_config(), _roof_problem())
    assert result.feasible is False
    assert result.status is FeasibilityStatus.UNKNOWN
    assert any(item.unknown for item in result.unknown_constraints)


def test_ground_mount_enumerates_tilt_azimuth_and_row_spacing() -> None:
    result = _engine().generate(_ground_problem())
    tilts = {item.configuration.tilt_deg for item in result.results}
    assert tilts == {15.0, 25.0}
    assert all(item.configuration.mounting_type is MountingType.GROUND for item in result.results)
    assert all(item.configuration.row_spacing_m == 4.0 for item in result.results)
    assert result.generator_id == "bounded_enumeration"


def test_ground_occupied_area_uses_gcr_from_row_spacing() -> None:
    config = _config(
        mounting_type=MountingType.GROUND,
        row_spacing_m=4.0,
        tilt_deg=15.0,
    )
    result = _engine().evaluate(config, _ground_problem(available_area_m2=40.0))
    length_m = TEST_MODULE_MONO.length_m
    assert length_m is not None
    assert result.derived.ground_coverage_ratio == length_m / 4.0
    assert result.feasible is False
    assert any(
        item.kind is DesignConstraintKind.AVAILABLE_AREA for item in result.failed_constraints
    )


def test_battery_required_constraint() -> None:
    result = _engine().evaluate(_config(), _roof_problem(battery_required=True))
    assert result.feasible is False
    assert any(item.kind is DesignConstraintKind.BATTERY for item in result.failed_constraints)


def test_empty_catalog_enumerates_nothing() -> None:
    result = PVDesignEngine(empty_pv_catalog()).generate(
        _roof_problem(
            include_fixtures=False,
            search_space=_space(module_ids=(), inverter_ids=()),
        )
    )
    assert result.results == ()


def test_fixtures_are_not_enumerated_as_products() -> None:
    result = PVDesignEngine(labelled_fixture_catalog()).generate(
        _roof_problem(include_fixtures=False, search_space=_space(module_ids=(), inverter_ids=()))
    )
    assert result.results == ()


def test_enumeration_is_bounded() -> None:
    space = _space(tilt_deg=(10.0, 20.0, 30.0), azimuth_deg=(90.0, 180.0), max_candidates=2)
    result = _engine().generate(
        _ground_problem(search_space=space.model_copy(update={"row_spacing_m": (4.0,)}))
    )
    assert result.truncated is True
    assert len(result.results) == 2


def test_unimplemented_optimizer() -> None:
    engine = PVDesignEngine(
        labelled_fixture_catalog(),
        generator=UnimplementedOptimizer("mip_layout"),
    )
    with pytest.raises(UnsupportedMethodError, match="not implemented"):
        engine.generate(_roof_problem())


def test_site_selection_requires_resolved_location() -> None:
    decision = SolarDecisionProblem(
        mode=DecisionMode.SITE_SELECTION,
        site_selection_scope=SiteSelectionScope.REGION_SEARCH,
        project_type=ProjectType.GROUND_MOUNTED,
        region=SITE,
    )
    assert decision.location_controllability is Controllability.CONTROLLABLE
    with pytest.raises(PVDesignConfigError, match="resolved to a location"):
        design_problem_from_decision(decision)


def test_fixed_site_design_uses_decision_location() -> None:
    decision = SolarDecisionProblem(
        mode=DecisionMode.FIXED_SITE,
        project_type=ProjectType.RESIDENTIAL_ROOFTOP,
        location=SITE,
        target_capacity_kw=10.0,
    )
    assert decision.location_controllability is Controllability.FIXED
    problem = design_problem_from_decision(
        decision,
        roof_area_m2=60.0,
        roof_tilt_deg=20.0,
        roof_azimuth_deg=180.0,
        search_space=_space(),
        include_fixtures=True,
    )
    assert problem.location == SITE
    assert problem.target_capacity_w == 10_000.0
    result = _engine().generate(problem)
    assert result.results
    assert all(item.configuration.location == SITE for item in result.results)


def test_rooftop_rejects_row_spacing_on_configuration() -> None:
    with pytest.raises(PVDesignConfigError, match="ground-mount variables"):
        _config(row_spacing_m=4.0)


def test_location_mismatch_is_rejected() -> None:
    elsewhere = GeographicRef(name="Other Site", longitude=8.0, latitude=50.0)
    result = _engine().evaluate(_config(location=elsewhere), _roof_problem())
    assert result.feasible is False
    assert any(item.kind is DesignConstraintKind.LOCATION for item in result.failed_constraints)


def test_design_package_does_not_import_optimizers() -> None:
    package = Path(__file__).resolve().parents[1] / "src" / "solardss" / "pv" / "design"
    forbidden = ("scipy", "cvxpy", "pulp", "pyomo", "optuna")
    found: list[str] = []
    for path in package.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            if f"import {name}" in text or f"from {name}" in text:
                found.append(f"{path.name}:{name}")
    assert found == []
