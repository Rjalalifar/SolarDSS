"""Synthetic PV rows for unit tests.

These are **not** commercial products. They must not be treated as a default
catalog. Manufacturer and model names are labelled TEST/FIXTURE.
"""

from __future__ import annotations

from solardss.pv.catalog import PVComponentCatalog
from solardss.pv.contracts import (
    BatterySystem,
    ComponentProvenance,
    ComponentSourceKind,
    ModuleTechnology,
    PVInverter,
    PVModule,
)

_FIXTURE_NOTES = (
    "TEST FIXTURE. Not a commercial product. Values are synthetic for unit tests only.",
)

_PROVENANCE = ComponentProvenance(
    source_name="solardss.test_fixture",
    source_kind=ComponentSourceKind.TEST_FIXTURE,
    fixture=True,
    notes=_FIXTURE_NOTES,
)


TEST_MODULE_MONO = PVModule(
    module_id="test.module.mono.400",
    manufacturer="SolarDSS Test Fixture",
    model="TEST-M-MONO-400",
    technology=ModuleTechnology.MONO_CSI.value,
    rated_power_w=400.0,
    efficiency=0.20,
    area_m2=2.0,
    length_m=1.72,
    width_m=1.13,
    thickness_m=0.035,
    temperature_coefficient_pmax=-0.35,
    voc=49.0,
    isc=10.5,
    vmp=41.0,
    imp=9.76,
    bifacial=False,
    degradation_rate=0.005,
    max_system_voltage=1500.0,
    weight_kg=22.0,
    provenance=_PROVENANCE,
)

TEST_MODULE_CDTE = PVModule(
    module_id="test.module.cdte.110",
    manufacturer="SolarDSS Test Fixture",
    model="TEST-M-CDTE-110",
    technology=ModuleTechnology.CDTE.value,
    rated_power_w=110.0,
    efficiency=0.18,
    area_m2=0.72,
    length_m=1.20,
    width_m=0.60,
    thickness_m=0.007,
    temperature_coefficient_pmax=-0.25,
    voc=90.0,
    isc=1.70,
    vmp=73.0,
    imp=1.51,
    bifacial=False,
    degradation_rate=0.004,
    max_system_voltage=1000.0,
    weight_kg=12.0,
    provenance=_PROVENANCE,
)

TEST_MODULE_BIFACIAL = PVModule(
    module_id="test.module.bifacial.430",
    manufacturer="SolarDSS Test Fixture",
    model="TEST-M-BIFI-430",
    technology=ModuleTechnology.TOPCON.value,
    rated_power_w=430.0,
    efficiency=0.215,
    area_m2=2.0,
    length_m=2.10,
    width_m=1.13,
    thickness_m=0.030,
    temperature_coefficient_pmax=-0.29,
    voc=52.0,
    isc=10.8,
    vmp=44.0,
    imp=9.78,
    bifacial=True,
    bifaciality_factor=0.70,
    degradation_rate=0.004,
    max_system_voltage=1500.0,
    weight_kg=24.5,
    provenance=_PROVENANCE,
)

TEST_INVERTER = PVInverter(
    inverter_id="test.inverter.ac10k",
    manufacturer="SolarDSS Test Fixture",
    model="TEST-INV-10K",
    rated_ac_power_w=10_000.0,
    max_dc_input_w=15_000.0,
    mppt_voltage_min=200.0,
    mppt_voltage_max=800.0,
    efficiency=0.98,
    max_dc_voltage=1000.0,
    max_dc_current=20.0,
    min_dc_voltage=150.0,
    max_ac_current=16.0,
    mppt_count=2,
    provenance=_PROVENANCE,
)

TEST_BATTERY = BatterySystem(
    battery_id="test.battery.5kwh",
    manufacturer="SolarDSS Test Fixture",
    model="TEST-BAT-5",
    chemistry="TEST-LFP",
    usable_capacity_kwh=5.0,
    rated_power_w=3000.0,
    roundtrip_efficiency=0.90,
    voltage_nominal=48.0,
    provenance=_PROVENANCE,
)

TEST_MODULES = (TEST_MODULE_MONO, TEST_MODULE_CDTE, TEST_MODULE_BIFACIAL)
TEST_INVERTERS = (TEST_INVERTER,)
TEST_BATTERIES = (TEST_BATTERY,)


def labelled_fixture_catalog() -> PVComponentCatalog:
    """Catalog of labelled TEST fixtures. Listings omit them unless requested."""

    return PVComponentCatalog(
        modules=TEST_MODULES,
        inverters=TEST_INVERTERS,
        batteries=TEST_BATTERIES,
        source_name="solardss.test_fixture",
    )
