"""PV energy identities. No irradiation or performance model is implied."""

from __future__ import annotations

from solardss.core.methods import MethodId
from solardss.core.quantity import Quantity
from solardss.core.results import LayerResult
from solardss.core.status import EvaluationStatus
from solardss.exceptions import InvalidParameterError
from solardss.provenance.record import provenance_record


def energy_from_capacity_factor(
    rated_power: Quantity,
    capacity_factor: float,
    duration: Quantity,
) -> tuple[Quantity, LayerResult]:
    """``E_kWh = P_kW * CF * t_h``.

    ``capacity_factor`` must be supplied in ``[0, 1]``. SolarDSS does not
    estimate CF from climate or equipment.
    """

    power_kw = rated_power.require_unit("kW")
    hours = duration.require_unit("h")
    if power_kw < 0.0 or hours < 0.0:
        raise InvalidParameterError("rated_power and duration must be non-negative")
    if not 0.0 <= capacity_factor <= 1.0:
        raise InvalidParameterError("capacity_factor must be in [0, 1]")

    energy = Quantity(value=power_kw * capacity_factor * hours, unit="kWh")
    layer = LayerResult(
        name="energy_kwh",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.CAPACITY_FACTOR_ENERGY.value,
        provenance=provenance_record(
            MethodId.CAPACITY_FACTOR_ENERGY,
            parameters={
                "rated_power_kw": power_kw,
                "capacity_factor": capacity_factor,
                "duration_h": hours,
                "identity": "E = P_rated * CF * t",
            },
        ),
    )
    return energy, layer


def energy_from_irradiation(
    irradiation: Quantity,
    area: Quantity,
    efficiency: float,
) -> tuple[Quantity, LayerResult]:
    """``E_kWh = G_kWh_m2 * A_m2 * η``.

    ``efficiency`` is a dimensionless factor in ``[0, 1]`` supplied by the
    caller (module, system, or otherwise). SolarDSS does not derive it.
    """

    ghi = irradiation.require_unit("kWh/m2")
    area_m2 = area.require_unit("m2")
    if ghi < 0.0 or area_m2 < 0.0:
        raise InvalidParameterError("irradiation and area must be non-negative")
    if not 0.0 <= efficiency <= 1.0:
        raise InvalidParameterError("efficiency must be in [0, 1]")

    energy = Quantity(value=ghi * area_m2 * efficiency, unit="kWh")
    layer = LayerResult(
        name="energy_kwh",
        status=EvaluationStatus.EVALUATED,
        method_id=MethodId.IRRADIATION_AREA_EFFICIENCY_ENERGY.value,
        provenance=provenance_record(
            MethodId.IRRADIATION_AREA_EFFICIENCY_ENERGY,
            parameters={
                "irradiation_kwh_m2": ghi,
                "area_m2": area_m2,
                "efficiency": efficiency,
                "identity": "E = G * A * eta",
            },
        ),
    )
    return energy, layer
