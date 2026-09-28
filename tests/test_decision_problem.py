"""Decision-problem modeling: modes, scopes, and structured gaps."""

from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from solardss.core.quantity import Quantity
from solardss.data.tables import validate_candidate_table
from solardss.decision import (
    BoundingBox,
    CandidateSet,
    DecisionOperation,
    GeographicRef,
    SolarDecisionProblem,
    get_missing_requirements,
    missing_requirement_records,
)
from solardss.decision.contracts import ConstraintOperator, ModulePreferences, UserConstraint
from solardss.decision.errors import OperationNotAllowedError
from solardss.exceptions import InvalidParameterError, UnknownIndicatorError
from solardss.indicators import (
    Controllability,
    DecisionMode,
    IndicatorRole,
    ProjectType,
    SiteSelectionScope,
)


def _region(**kwargs: object) -> SolarDecisionProblem:
    payload: dict[str, object] = {
        "mode": DecisionMode.SITE_SELECTION,
        "site_selection_scope": SiteSelectionScope.REGION_SEARCH,
        "project_type": ProjectType.ROOFTOP,
    }
    payload.update(kwargs)
    return SolarDecisionProblem.model_validate(payload)


def _fixed(**kwargs: object) -> SolarDecisionProblem:
    payload: dict[str, object] = {
        "mode": DecisionMode.FIXED_SITE,
        "project_type": ProjectType.RESIDENTIAL_ROOFTOP,
    }
    payload.update(kwargs)
    return SolarDecisionProblem.model_validate(payload)


def test_region_search_requires_region() -> None:
    problem = _region()
    missing = get_missing_requirements(problem)
    required = [item for item in missing if item.required]
    assert [item.field for item in required] == ["region"]
    assert required[0].reason == "region_search requires region"


def test_aoi_search_requires_aoi() -> None:
    problem = SolarDecisionProblem(
        mode=DecisionMode.SITE_SELECTION,
        site_selection_scope=SiteSelectionScope.AOI_SEARCH,
        project_type=ProjectType.GROUND_MOUNTED,
    )
    required = [item for item in get_missing_requirements(problem) if item.required]
    assert [item.field for item in required] == ["aoi"]


def test_candidate_ranking_requires_candidates() -> None:
    problem = SolarDecisionProblem(
        mode=DecisionMode.SITE_SELECTION,
        site_selection_scope=SiteSelectionScope.CANDIDATE_RANKING,
        project_type=ProjectType.COMMERCIAL_ROOFTOP,
    )
    required = [item for item in get_missing_requirements(problem) if item.required]
    assert [item.field for item in required] == ["candidates"]


def test_fixed_site_requires_location() -> None:
    problem = _fixed()
    required = [item for item in get_missing_requirements(problem) if item.required]
    assert [item.field for item in required] == ["location"]
    assert "candidate" not in required[0].reason


def test_complete_region_search_has_no_required_gaps() -> None:
    problem = _region(region=GeographicRef(name="Vienna"))
    required = [item for item in get_missing_requirements(problem) if item.required]
    assert required == []
    optional = {item.field: item for item in get_missing_requirements(problem) if not item.required}
    assert optional["target_capacity_kw"].reason == "needed only for capacity-specific sizing"


def test_complete_fixed_site_can_be_fully_specified() -> None:
    problem = _fixed(
        location=GeographicRef(latitude=48.2, longitude=16.37, name="owned-house"),
        target_capacity_kw=8.0,
        budget=Quantity(value=15000.0, unit="EUR"),
        objective="OBJECTIVE_MAX_SELF_CONSUMPTION",
        grid_connection_required=False,
        battery_required=False,
        module_preferences=ModulePreferences(technology="c-Si"),
    )
    assert get_missing_requirements(problem) == ()


def test_missing_requirement_records_are_dicts() -> None:
    records = missing_requirement_records(_region())
    assert records[0] == {
        "field": "region",
        "required": True,
        "reason": "region_search requires region",
    }
    capacity = next(item for item in records if item["field"] == "target_capacity_kw")
    assert capacity == {
        "field": "target_capacity_kw",
        "required": False,
        "reason": "needed only for capacity-specific sizing",
    }


def test_reasons_are_not_conversational() -> None:
    for item in get_missing_requirements(_fixed()):
        reason = item.reason.lower()
        assert "?" not in item.reason
        assert "please" not in reason
        assert "could you" not in reason
        assert "tell me" not in reason


def test_mw_satisfies_capacity_gap() -> None:
    problem = _region(region=GeographicRef(name="Vienna"), target_capacity_mw=2.0)
    fields = [item.field for item in get_missing_requirements(problem)]
    assert "target_capacity_kw" not in fields
    assert problem.target_capacity_kw_resolved == pytest.approx(2000.0)


def test_inconsistent_capacity_rejected() -> None:
    with pytest.raises((InvalidParameterError, ValidationError)):
        _region(
            region=GeographicRef(name="Vienna"),
            target_capacity_kw=1000.0,
            target_capacity_mw=2.0,
        )


def test_site_selection_without_scope_rejected() -> None:
    with pytest.raises((InvalidParameterError, ValidationError)):
        SolarDecisionProblem(
            mode=DecisionMode.SITE_SELECTION,
            project_type=ProjectType.UTILITY_SCALE,
        )


def test_fixed_site_rejects_scope_and_candidates() -> None:
    with pytest.raises((InvalidParameterError, ValidationError)):
        _fixed(site_selection_scope=SiteSelectionScope.REGION_SEARCH)
    with pytest.raises((InvalidParameterError, ValidationError)):
        _fixed(
            location=GeographicRef(name="house"),
            candidates=CandidateSet(candidate_ids=("a",)),
        )


def test_region_search_rejects_aoi_and_candidates() -> None:
    with pytest.raises((InvalidParameterError, ValidationError)):
        _region(region=GeographicRef(name="Vienna"), aoi=GeographicRef(name="poly"))
    with pytest.raises((InvalidParameterError, ValidationError)):
        _region(
            region=GeographicRef(name="Vienna"),
            candidates=CandidateSet(candidate_ids=("b1",)),
        )


def test_pv_location_semantics_follow_registry() -> None:
    site = _region(region=GeographicRef(name="Vienna"))
    fixed = _fixed(location=GeographicRef(name="house"))
    assert site.location_role is IndicatorRole.DECISION_VARIABLE
    assert site.location_controllability is Controllability.CONTROLLABLE
    assert fixed.location_role is IndicatorRole.CONTEXT
    assert fixed.location_controllability is Controllability.FIXED


def test_fixed_site_forbids_generation_and_ranking() -> None:
    problem = _fixed(location=GeographicRef(name="house"))
    assert problem.operation_allowed(DecisionOperation.CANDIDATE_GENERATION) is False
    assert problem.operation_allowed(DecisionOperation.SITE_RANKING) is False
    assert problem.operation_allowed(DecisionOperation.SYSTEM_DESIGN) is True
    with pytest.raises(OperationNotAllowedError):
        problem.require_operation(DecisionOperation.CANDIDATE_GENERATION)
    with pytest.raises(OperationNotAllowedError):
        problem.require_operation("site_ranking")


def test_region_search_allows_generation_and_ranking() -> None:
    problem = _region(region=GeographicRef(name="Vienna"))
    assert problem.operation_allowed(DecisionOperation.CANDIDATE_GENERATION) is True
    assert problem.operation_allowed(DecisionOperation.SITE_RANKING) is True
    problem.require_operation(DecisionOperation.CANDIDATE_GENERATION)


def test_candidate_ranking_does_not_generate() -> None:
    problem = SolarDecisionProblem(
        mode=DecisionMode.SITE_SELECTION,
        site_selection_scope=SiteSelectionScope.CANDIDATE_RANKING,
        project_type=ProjectType.COMMERCIAL_ROOFTOP,
        candidates=CandidateSet(candidate_ids=("b1", "b2")),
    )
    assert problem.operation_allowed(DecisionOperation.CANDIDATE_GENERATION) is False
    assert problem.operation_allowed(DecisionOperation.SITE_RANKING) is True
    with pytest.raises(OperationNotAllowedError):
        problem.require_operation(DecisionOperation.CANDIDATE_GENERATION)


def test_aoi_search_complete() -> None:
    problem = SolarDecisionProblem(
        mode=DecisionMode.SITE_SELECTION,
        site_selection_scope=SiteSelectionScope.AOI_SEARCH,
        project_type=ProjectType.UTILITY_SCALE,
        aoi=GeographicRef(
            name="study-aoi",
            bbox=BoundingBox(min_lon=16.2, min_lat=48.1, max_lon=16.5, max_lat=48.3),
        ),
    )
    assert [item.field for item in get_missing_requirements(problem) if item.required] == []


def test_candidate_set_from_table() -> None:
    table = validate_candidate_table(
        pd.DataFrame({"candidate_id": ["r0c0", "r1c1"]}),
    )
    candidates = CandidateSet.from_table(table, label="industrial-roofs")
    assert candidates.candidate_ids == ("r0c0", "r1c1")
    problem = SolarDecisionProblem(
        mode=DecisionMode.SITE_SELECTION,
        site_selection_scope=SiteSelectionScope.CANDIDATE_RANKING,
        project_type=ProjectType.COMMERCIAL_ROOFTOP,
        candidates=candidates,
    )
    assert "candidates" not in [
        item.field for item in get_missing_requirements(problem) if item.required
    ]


def test_grid_distance_required_when_connection_required() -> None:
    problem = _fixed(
        location=GeographicRef(name="house"),
        grid_connection_required=True,
    )
    required = [item for item in get_missing_requirements(problem) if item.required]
    assert any(item.field == "maximum_grid_distance" and item.required for item in required)


def test_objective_must_be_registered_objective() -> None:
    with pytest.raises((InvalidParameterError, ValidationError)):
        _region(region=GeographicRef(name="Vienna"), objective="SOLAR_GHI_ANNUAL")


def test_user_constraint_unknown_indicator() -> None:
    with pytest.raises(UnknownIndicatorError):
        _region(
            region=GeographicRef(name="Vienna"),
            user_constraints=(
                UserConstraint(
                    indicator_id="not_an_indicator",
                    operator=ConstraintOperator.LE,
                    value=10.0,
                ),
            ),
        )


def test_geographic_ref_requires_identity() -> None:
    with pytest.raises((InvalidParameterError, ValidationError)):
        GeographicRef()


def test_missing_requirements_method_matches_function() -> None:
    problem = _region()
    assert problem.missing_requirements() == get_missing_requirements(problem)
