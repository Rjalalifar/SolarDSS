"""Indicator observations and data-source provenance."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from pydantic import ValidationError

from solardss.exceptions import InvalidParameterError
from solardss.indicators.contracts import ValueProvenanceKind
from solardss.provenance import (
    DataProvenance,
    IndicatorObservation,
    ObservationStatus,
    QualityFlag,
    SpatialCoverage,
    TemporalCoverage,
    ValueOrigin,
)


def _provenance(**kwargs: object) -> DataProvenance:
    payload: dict[str, object] = {
        "provider_id": "fake.solar_resource",
        "source_name": "unit test catalog",
        "retrieved_at": datetime(2020, 1, 1, tzinfo=timezone.utc),
        "request_hash": "a" * 64,
        "value_kind": ValueProvenanceKind.MEASURED_OR_PROVIDER,
        "value_origin": ValueOrigin.RAW,
        "source_version": "v1",
    }
    payload.update(kwargs)
    return DataProvenance.model_validate(payload)


def _spatial() -> SpatialCoverage:
    return SpatialCoverage(resolution="0.25deg", crs="EPSG:4326", native_cell_size_m=25000.0)


def _temporal() -> TemporalCoverage:
    return TemporalCoverage(
        resolution="annual",
        start=date(2010, 1, 1),
        end=date(2020, 12, 31),
        aggregation="mean",
    )


def test_retrieved_at_must_be_timezone_aware() -> None:
    with pytest.raises((InvalidParameterError, ValidationError)):
        _provenance(retrieved_at=datetime(2020, 1, 1))


def test_derived_requires_calculation_method() -> None:
    with pytest.raises((InvalidParameterError, ValidationError)):
        _provenance(
            value_origin=ValueOrigin.DERIVED,
            value_kind=ValueProvenanceKind.DERIVED_METHOD,
        )


def test_raw_forbids_calculation_method() -> None:
    with pytest.raises((InvalidParameterError, ValidationError)):
        _provenance(calculation_method="not-allowed-on-raw")


def test_observed_requires_value_and_unit() -> None:
    provenance = _provenance()
    with pytest.raises((InvalidParameterError, ValidationError)):
        IndicatorObservation(
            indicator_id="SOLAR_GHI_ANNUAL",
            status=ObservationStatus.OBSERVED,
            provenance=provenance,
            spatial=_spatial(),
            temporal=_temporal(),
            value=1800.0,
        )
    with pytest.raises((InvalidParameterError, ValidationError)):
        IndicatorObservation(
            indicator_id="SOLAR_GHI_ANNUAL",
            status=ObservationStatus.OBSERVED,
            provenance=provenance,
            spatial=_spatial(),
            temporal=_temporal(),
            unit="kWh/m2/year",
        )


def test_non_observed_forbids_value() -> None:
    provenance = _provenance()
    with pytest.raises((InvalidParameterError, ValidationError)):
        IndicatorObservation(
            indicator_id="SOLAR_GHI_ANNUAL",
            status=ObservationStatus.MISSING_COVERAGE,
            provenance=provenance,
            spatial=_spatial(),
            temporal=_temporal(),
            value=1800.0,
            unit="kWh/m2/year",
        )


def test_observed_carries_quality_flags_and_assumptions() -> None:
    observation = IndicatorObservation(
        indicator_id="SOLAR_GHI_ANNUAL",
        status=ObservationStatus.OBSERVED,
        provenance=_provenance(),
        spatial=_spatial(),
        temporal=_temporal(),
        value=1625.0,
        unit="kWh/m2/year",
        quality_flags=(QualityFlag.NOMINAL.value,),
        assumptions=("annual mean of daily GHI",),
    )
    assert observation.value == 1625.0
    assert observation.provenance.value_origin is ValueOrigin.RAW
    assert observation.spatial.resolution == "0.25deg"
    assert observation.temporal.aggregation == "mean"
