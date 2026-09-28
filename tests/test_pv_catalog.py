"""PV component catalog: empty default, local import, filters, test fixtures."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from solardss.exceptions import UnsupportedMethodError
from solardss.pv import (
    KNOWN_FUTURE_ADAPTERS,
    InverterFilter,
    LocalFileAdapter,
    ModuleFilter,
    ModuleTechnology,
    PVCatalogConfigError,
    PVCatalogSchemaError,
    PVComponentCatalog,
    PVInverter,
    PVModule,
    UnimplementedDatabaseAdapter,
    UnknownComponentError,
    empty_pv_catalog,
    load_pv_catalog,
)
from solardss.pv.contracts import ComponentProvenance, ComponentSourceKind
from solardss.pv.fixtures import (
    TEST_INVERTER,
    TEST_MODULE_BIFACIAL,
    TEST_MODULE_CDTE,
    TEST_MODULE_MONO,
    labelled_fixture_catalog,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "pv"


def _module(**updates: Any) -> PVModule:
    payload = TEST_MODULE_MONO.model_dump()
    payload.update(updates)
    return PVModule.model_validate(payload)


def test_empty_catalog_stays_empty() -> None:
    catalog = empty_pv_catalog()
    assert len(catalog) == 0
    assert catalog.list_modules() == ()
    assert catalog.list_inverters() == ()
    assert catalog.list_batteries() == ()
    with pytest.raises(UnknownComponentError):
        catalog.get_module("anything")


def test_caller_supplied_row_is_listed_without_fixture_flag() -> None:
    row = PVModule(
        module_id="caller.local.1",
        manufacturer="Caller Import",
        model="Local-400",
        technology=ModuleTechnology.MONO_CSI.value,
        rated_power_w=400.0,
        provenance=ComponentProvenance(
            source_name="caller.json",
            source_kind=ComponentSourceKind.CATALOG_FILE,
            notes=("Caller-supplied datasheet row imported from a local file.",),
        ),
    )
    catalog = PVComponentCatalog(modules=(row,), source_name="caller")
    listed = catalog.list_modules()
    assert [item.module_id for item in listed] == ["caller.local.1"]
    assert listed[0].provenance.fixture is False


def test_fixtures_are_labelled_and_not_listed_as_products() -> None:
    catalog = labelled_fixture_catalog()
    assert catalog.list_modules() == ()
    assert catalog.list_inverters() == ()
    assert catalog.list_batteries() == ()
    shown = catalog.list_modules(ModuleFilter(include_fixtures=True))
    assert {item.module_id for item in shown} == {
        TEST_MODULE_MONO.module_id,
        TEST_MODULE_CDTE.module_id,
        TEST_MODULE_BIFACIAL.module_id,
    }
    for item in shown:
        assert item.provenance.fixture is True
        assert item.provenance.source_kind is ComponentSourceKind.TEST_FIXTURE
        assert "test fixture" in " ".join(item.provenance.notes).casefold()
        assert "test" in item.manufacturer.casefold() or "fixture" in item.model.casefold()
    assert TEST_MODULE_MONO.manufacturer == "SolarDSS Test Fixture"


def test_fixtures_not_in_public_pv_exports() -> None:
    import solardss
    import solardss.pv as pv_pkg

    assert "TEST_MODULE_MONO" not in pv_pkg.__all__
    assert "labelled_fixture_catalog" not in pv_pkg.__all__
    assert "fixtures" not in pv_pkg.__all__
    assert not hasattr(solardss, "TEST_MODULE_MONO")


def test_load_json_catalog() -> None:
    catalog = load_pv_catalog(FIXTURE_DIR / "catalog.json")
    modules = catalog.list_modules(ModuleFilter(include_fixtures=True))
    assert len(modules) == 2
    inverter = catalog.get_inverter("test.file.inv.10k")
    assert inverter.rated_ac_power_w == 10_000.0
    assert inverter.max_dc_input_w == 15_000.0
    assert inverter.mppt_voltage_min == 200.0
    assert inverter.mppt_voltage_max == 800.0
    assert inverter.efficiency == pytest.approx(0.98)
    assert inverter.max_dc_voltage == 1000.0
    assert inverter.max_dc_current == 20.0
    assert catalog.get_battery("test.file.bat.5").usable_capacity_kwh == 5.0
    assert catalog.list_modules() == ()
    assert catalog.provenance.stamp.method_id == "pv_catalog_load"
    assert modules[0].provenance.record is not None


def test_load_empty_json() -> None:
    catalog = load_pv_catalog(FIXTURE_DIR / "empty.json")
    assert len(catalog) == 0


def test_load_csv_modules() -> None:
    catalog = load_pv_catalog(FIXTURE_DIR / "modules.csv")
    rows = catalog.list_modules(ModuleFilter(include_fixtures=True))
    assert {item.module_id for item in rows} == {
        "test.csv.mono.400",
        "test.csv.cdte.110",
    }
    assert catalog.get_module("test.csv.mono.400").efficiency == pytest.approx(0.20)


def test_load_rejects_invalid_electrical_row(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        """
        {"modules": [{
          "module_id": "test.bad",
          "manufacturer": "SolarDSS Test Fixture",
          "model": "TEST-BAD",
          "technology": "mono-c-Si",
          "rated_power_w": 400,
          "voc": 40.0,
          "vmp": 41.0,
          "provenance": {
            "source_name": "bad.json",
            "source_kind": "test_fixture",
            "fixture": true,
            "notes": ["TEST FIXTURE. Not a commercial product."]
          }
        }]}
        """,
        encoding="utf-8",
    )
    with pytest.raises(PVCatalogSchemaError, match="vmp must not exceed voc"):
        load_pv_catalog(path)


def test_filter_by_technology_efficiency_power_temp_coeff_and_size() -> None:
    catalog = labelled_fixture_catalog()
    cdte = catalog.list_modules(
        ModuleFilter(include_fixtures=True, technology=ModuleTechnology.CDTE)
    )
    assert [item.module_id for item in cdte] == [TEST_MODULE_CDTE.module_id]
    efficient = catalog.list_modules(ModuleFilter(include_fixtures=True, min_efficiency=0.19))
    ids = {item.module_id for item in efficient}
    assert TEST_MODULE_MONO.module_id in ids
    assert TEST_MODULE_BIFACIAL.module_id in ids
    assert TEST_MODULE_CDTE.module_id not in ids
    power = catalog.list_modules(
        ModuleFilter(include_fixtures=True, min_rated_power_w=350.0, max_rated_power_w=410.0)
    )
    assert [item.module_id for item in power] == [TEST_MODULE_MONO.module_id]
    milder = catalog.list_modules(
        ModuleFilter(include_fixtures=True, min_temperature_coefficient_pmax=-0.30)
    )
    milder_ids = {item.module_id for item in milder}
    assert TEST_MODULE_CDTE.module_id in milder_ids
    assert TEST_MODULE_MONO.module_id not in milder_ids
    compact = catalog.list_modules(
        ModuleFilter(include_fixtures=True, max_length_m=1.5, max_width_m=0.8, max_weight_kg=15.0)
    )
    assert [item.module_id for item in compact] == [TEST_MODULE_CDTE.module_id]
    bifacial = catalog.list_modules(ModuleFilter(include_fixtures=True, bifacial=True))
    assert [item.module_id for item in bifacial] == [TEST_MODULE_BIFACIAL.module_id]


def test_inverter_power_and_efficiency_filter() -> None:
    catalog = labelled_fixture_catalog()
    assert catalog.list_inverters() == ()
    matched = catalog.list_inverters(
        InverterFilter(include_fixtures=True, min_rated_ac_power_w=9000.0, min_efficiency=0.97)
    )
    assert [item.inverter_id for item in matched] == [TEST_INVERTER.inverter_id]
    missed = catalog.list_inverters(
        InverterFilter(include_fixtures=True, min_rated_ac_power_w=20_000.0)
    )
    assert missed == ()


def test_missing_dimension_fails_dimensional_filter() -> None:
    row = _module(length_m=None, width_m=None)
    catalog = PVComponentCatalog(modules=(row,), source_name="partial")
    assert catalog.list_modules(ModuleFilter(include_fixtures=True, max_length_m=2.0)) == ()


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(PVCatalogConfigError, match="duplicate module_id"):
        PVComponentCatalog(modules=(TEST_MODULE_MONO, TEST_MODULE_MONO))


def test_inconsistent_iv_curve_rejected() -> None:
    with pytest.raises(PVCatalogConfigError, match="vmp must not exceed voc"):
        _module(voc=40.0, vmp=41.0)


def test_commercial_looking_fixture_rejected() -> None:
    with pytest.raises(PVCatalogConfigError, match="TEST or FIXTURE"):
        PVModule(
            module_id="sneaky",
            manufacturer="Acme Solar",
            model="Silent-400",
            technology="mono-c-Si",
            rated_power_w=400.0,
            provenance=ComponentProvenance(
                source_name="invented",
                source_kind=ComponentSourceKind.TEST_FIXTURE,
                fixture=True,
                notes=("TEST FIXTURE. Not a commercial product.",),
            ),
        )


def test_row_requires_provenance() -> None:
    with pytest.raises(ValidationError):
        PVModule(  # type: ignore[call-arg]
            module_id="x",
            manufacturer="SolarDSS Test Fixture",
            model="TEST-X",
            technology="mono-c-Si",
            rated_power_w=1.0,
        )


def test_unimplemented_database_adapter() -> None:
    adapter = UnimplementedDatabaseAdapter("nrel_cec")
    assert adapter.availability().configured is False
    with pytest.raises(UnsupportedMethodError, match="not implemented"):
        adapter.load()
    assert "nrel_cec" in KNOWN_FUTURE_ADAPTERS
    assert "iec_61853_library" in KNOWN_FUTURE_ADAPTERS


def test_local_file_adapter() -> None:
    adapter = LocalFileAdapter(FIXTURE_DIR / "empty.json")
    assert adapter.availability().configured is True
    assert len(adapter.load()) == 0


def test_bifacial_requires_factor() -> None:
    with pytest.raises(PVCatalogConfigError, match="bifaciality_factor"):
        _module(bifacial=True)


def test_inverter_mppt_range_must_be_ordered() -> None:
    payload = TEST_INVERTER.model_dump()
    payload["mppt_voltage_min"] = 900.0
    payload["mppt_voltage_max"] = 200.0
    with pytest.raises(PVCatalogConfigError, match="mppt_voltage_max"):
        PVInverter.model_validate(payload)
