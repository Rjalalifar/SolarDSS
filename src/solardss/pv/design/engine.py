"""PV design engine: evaluate configurations and enumerate a bounded search space."""

from __future__ import annotations

from solardss.core.hashing import fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.data.geo import GeographicRef
from solardss.decision.contracts import DecisionOperation
from solardss.decision.problem import SolarDecisionProblem
from solardss.indicators.contracts import DecisionMode
from solardss.provenance.record import provenance_record
from solardss.pv.catalog import PVComponentCatalog
from solardss.pv.design.contracts import (
    ConfigurationSearchSpace,
    PVDesignProblem,
    PVDesignResult,
    PVDesignSet,
    PVSystemConfiguration,
)
from solardss.pv.design.errors import PVDesignConfigError
from solardss.pv.design.evaluate import evaluate_configuration
from solardss.pv.design.generator import BoundedEnumerationGenerator, PVConfigurationGenerator


class PVDesignEngine:
    """Generate and evaluate PV configurations against a resolved-site problem.

    Location is an input. In site selection it must already have been chosen.
    """

    def __init__(
        self,
        catalog: PVComponentCatalog,
        *,
        generator: PVConfigurationGenerator | None = None,
    ) -> None:
        self._catalog = catalog
        self._generator = BoundedEnumerationGenerator() if generator is None else generator

    @property
    def catalog(self) -> PVComponentCatalog:
        return self._catalog

    def evaluate(
        self,
        configuration: PVSystemConfiguration,
        problem: PVDesignProblem,
    ) -> PVDesignResult:
        """Evaluate one configuration. Does not rank or invent a design."""

        return evaluate_configuration(configuration, problem, self._catalog)

    def generate(self, problem: PVDesignProblem) -> PVDesignSet:
        """Enumerate, then evaluate. Truncation is not optimality."""

        configs, truncated, generate_warnings = self._generator.generate(problem, self._catalog)
        results = tuple(self.evaluate(item, problem) for item in configs)
        warnings = list(generate_warnings)
        if truncated:
            warnings.append("truncated enumeration is not a globally optimal PV design")
        return PVDesignSet(
            results=results,
            generator_id=self._generator.generator_id,
            truncated=truncated,
            warnings=tuple(dict.fromkeys(warnings)),
            provenance=provenance_record(
                MethodId.PV_DESIGN_ENUMERATE,
                parameters={
                    "generator_id": self._generator.generator_id,
                    "max_candidates": problem.search_space.max_candidates,
                    "emitted": len(results),
                    "feasible": sum(1 for item in results if item.feasible),
                    "truncated": truncated,
                    "project_type": problem.project_type.value,
                    "decision_mode": problem.decision_mode.value,
                    "catalog_source": self._catalog.source_name,
                },
                input_fingerprints=(
                    fingerprint_canonical(
                        {
                            "location": problem.location.model_dump(mode="json").get("name"),
                            "project_type": problem.project_type.value,
                            "max_candidates": problem.search_space.max_candidates,
                        }
                    ),
                ),
                notes=(
                    "bounded discrete enumeration; not an LLM optimizer",
                    "location is not a design variable after site resolution",
                ),
            ),
        )


def design_problem_from_decision(
    decision: SolarDecisionProblem,
    *,
    location: GeographicRef | None = None,
    available_area_m2: float | None = None,
    roof_area_m2: float | None = None,
    roof_tilt_deg: float | None = None,
    roof_azimuth_deg: float | None = None,
    min_capacity_w: float | None = None,
    max_capacity_w: float | None = None,
    min_dc_ac_ratio: float | None = None,
    max_dc_ac_ratio: float | None = None,
    max_module_length_m: float | None = None,
    max_module_width_m: float | None = None,
    max_module_thickness_m: float | None = None,
    max_module_weight_kg: float | None = None,
    search_space: ConfigurationSearchSpace | None = None,
    include_fixtures: bool = False,
    notes: tuple[str, ...] = (),
) -> PVDesignProblem:
    """Build a design problem from a decision problem after the site is known.

    ``SYSTEM_DESIGN`` must be an allowed operation. Site selection does not
    choose location here: pass the resolved candidate as ``location``.
    """

    decision.require_operation(DecisionOperation.SYSTEM_DESIGN)
    resolved = _resolved_location(decision, location)
    target_w = None
    if decision.target_capacity_kw_resolved is not None:
        target_w = decision.target_capacity_kw_resolved * 1000.0
    battery_required = bool(decision.battery_required)
    extra_notes = notes
    if decision.mode is DecisionMode.SITE_SELECTION:
        extra_notes = (
            *notes,
            "location was resolved from site selection and is not re-optimized",
        )
    return PVDesignProblem(
        location=resolved,
        project_type=decision.project_type,
        decision_mode=decision.mode,
        available_area_m2=available_area_m2,
        roof_area_m2=roof_area_m2,
        roof_tilt_deg=roof_tilt_deg,
        roof_azimuth_deg=roof_azimuth_deg,
        target_capacity_w=target_w,
        min_capacity_w=min_capacity_w,
        max_capacity_w=max_capacity_w,
        min_dc_ac_ratio=min_dc_ac_ratio,
        max_dc_ac_ratio=max_dc_ac_ratio,
        max_module_length_m=max_module_length_m,
        max_module_width_m=max_module_width_m,
        max_module_thickness_m=max_module_thickness_m,
        max_module_weight_kg=max_module_weight_kg,
        module_preferences=decision.module_preferences,
        user_constraints=decision.user_constraints,
        battery_required=battery_required,
        include_fixtures=include_fixtures,
        search_space=search_space or ConfigurationSearchSpace(),
        notes=extra_notes,
    )


def _resolved_location(
    decision: SolarDecisionProblem,
    location: GeographicRef | None,
) -> GeographicRef:
    if decision.mode is DecisionMode.FIXED_SITE:
        resolved = location if location is not None else decision.location
        if resolved is None:
            raise PVDesignConfigError("FIXED_SITE PV design requires a location")
        return resolved
    if location is None:
        raise PVDesignConfigError(
            "SITE_SELECTION must be resolved to a location before PV design; "
            "location is controllable only during site selection"
        )
    return location
