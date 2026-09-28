"""Extensible PV configuration generators. The first implementation enumerates."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from solardss.exceptions import UnsupportedMethodError
from solardss.pv.catalog import PVComponentCatalog
from solardss.pv.design.contracts import PVDesignProblem, PVSystemConfiguration
from solardss.pv.design.enumerate import enumerate_configurations


@runtime_checkable
class PVConfigurationGenerator(Protocol):
    """Source of candidate configurations. Implementations must not call an LLM."""

    @property
    def generator_id(self) -> str:
        """Stable identity recorded in provenance."""

        ...

    def generate(
        self,
        problem: PVDesignProblem,
        catalog: PVComponentCatalog,
    ) -> tuple[tuple[PVSystemConfiguration, ...], bool, tuple[str, ...]]:
        """Return configurations, a truncation flag, and warnings."""

        ...


class BoundedEnumerationGenerator:
    """Cartesian product of an explicit discrete search space, then stop.

    This is not a globally optimal layout solver.
    """

    @property
    def generator_id(self) -> str:
        return "bounded_enumeration"

    def generate(
        self,
        problem: PVDesignProblem,
        catalog: PVComponentCatalog,
    ) -> tuple[tuple[PVSystemConfiguration, ...], bool, tuple[str, ...]]:
        return enumerate_configurations(problem, catalog)


class UnimplementedOptimizer:
    """Placeholder for a future numerical optimizer.

    SolarDSS does not pull in a MIP/NLP library for PV layout.
    """

    def __init__(self, optimizer_id: str) -> None:
        if not optimizer_id.strip():
            raise UnsupportedMethodError("optimizer_id must be a non-empty adapter name")
        self._optimizer_id = optimizer_id.strip()

    @property
    def generator_id(self) -> str:
        return self._optimizer_id

    def generate(
        self,
        problem: PVDesignProblem,
        catalog: PVComponentCatalog,
    ) -> tuple[tuple[PVSystemConfiguration, ...], bool, tuple[str, ...]]:
        _ = (problem, catalog)
        raise UnsupportedMethodError(
            f"{self._optimizer_id} PV layout optimizer is not implemented. "
            "Use BoundedEnumerationGenerator with an explicit discrete search space."
        )


KNOWN_FUTURE_OPTIMIZERS: tuple[str, ...] = ("mip_layout", "nlp_layout")
