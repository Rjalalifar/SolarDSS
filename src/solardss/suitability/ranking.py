"""Weighted Sum Model ranking with a full per-criterion trace."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from solardss.core.hashing import fingerprint_canonical
from solardss.core.methods import MethodId
from solardss.core.quantity import Quantity
from solardss.core.status import EvaluationStatus, FeasibilityStatus
from solardss.exceptions import InvalidParameterError
from solardss.indicators.catalog import registry as default_registry
from solardss.indicators.ids import IndicatorId
from solardss.indicators.registry import IndicatorRegistry
from solardss.provenance.record import provenance_record
from solardss.suitability.contracts import (
    CriterionContribution,
    CriterionWeight,
    DecisionProfile,
    MultiObjectiveResult,
    NormalizationStats,
    ObjectiveSpec,
    OmissionReason,
    OmittedCandidate,
    RankedCandidate,
    RankingResult,
    validate_decision_profile,
)
from solardss.suitability.errors import RankingConfigError
from solardss.suitability.inputs import (
    CandidateRows,
    ConstraintStatusInput,
    ScoreValue,
    map_indicator_values,
    require_unique_ids,
    resolve_constraint_statuses,
)
from solardss.suitability.normalize import scores_from_optional


class SuitabilityEngine:
    """Rank feasible candidates with an explicit ``DecisionProfile``.

    A profile is never inferred. Missing indicator values stay missing.
    """

    def __init__(self, catalog: IndicatorRegistry | None = None) -> None:
        self._catalog = default_registry if catalog is None else catalog

    def rank(
        self,
        rows: CandidateRows,
        profile: DecisionProfile,
        *,
        constraint_status: ConstraintStatusInput = None,
    ) -> RankingResult:
        """Weighted Sum Model. Rank 1 is best; ties break on ``candidate_id``."""

        validate_decision_profile(profile, self._catalog)
        require_unique_ids(rows)
        statuses = resolve_constraint_statuses(rows, constraint_status)
        parsed = [_parse_row(ident, values, profile) for ident, values in rows]
        omitted: list[OmittedCandidate] = []
        eligible: list[_Eligible] = []
        for item in parsed:
            status = statuses[item.candidate_id]
            if status is FeasibilityStatus.EXCLUDED:
                omitted.append(
                    OmittedCandidate(
                        candidate_id=item.candidate_id,
                        reason=OmissionReason.EXCLUDED,
                        constraint_status=status,
                    )
                )
                continue
            if status is FeasibilityStatus.UNKNOWN:
                omitted.append(
                    OmittedCandidate(
                        candidate_id=item.candidate_id,
                        reason=OmissionReason.UNKNOWN_FEASIBILITY,
                        constraint_status=status,
                    )
                )
                continue
            if item.missing:
                omitted.append(
                    OmittedCandidate(
                        candidate_id=item.candidate_id,
                        reason=OmissionReason.INCOMPLETE_INDICATORS,
                        constraint_status=status,
                        missing_indicators=item.missing,
                    )
                )
                continue
            eligible.append(_Eligible(item.candidate_id, status, item.values))

        used_weights = profile.resolved_weights()
        columns: dict[str, list[float | None]] = {
            criterion.indicator_id: [] for criterion in profile.criteria
        }
        for row in eligible:
            for criterion in profile.criteria:
                columns[criterion.indicator_id].append(row.values.get(criterion.indicator_id))

        normalized: dict[str, tuple[float | None, ...]] = {}
        stats_list: list[NormalizationStats] = []
        if eligible:
            for criterion in profile.criteria:
                scores, stats = scores_from_optional(
                    columns[criterion.indicator_id],
                    kind=criterion.normalization,
                    range_min=criterion.range_min,
                    range_max=criterion.range_max,
                    target=criterion.target_value,
                    indicator_id=criterion.indicator_id,
                    unit=criterion.unit,
                )
                normalized[criterion.indicator_id] = scores
                stats_list.append(stats)

        ranked_rows: list[RankedCandidate] = []
        for index, row in enumerate(eligible):
            contributions: list[CriterionContribution] = []
            score = 0.0
            for criterion in profile.criteria:
                raw = row.values[criterion.indicator_id]
                scaled = normalized[criterion.indicator_id][index]
                if scaled is None:
                    raise RankingConfigError(
                        f"{row.candidate_id}: {criterion.indicator_id} normalized to missing"
                    )
                weight = used_weights[criterion.indicator_id]
                spec_stats = next(
                    item for item in stats_list if item.indicator_id == criterion.indicator_id
                )
                contributions.append(_contribution(criterion, raw, scaled, weight, spec_stats))
                score += weight * scaled
            ranked_rows.append(
                RankedCandidate(
                    candidate_id=row.candidate_id,
                    score=score,
                    rank=1,
                    constraint_status=row.status,
                    contributions=tuple(contributions),
                )
            )

        ranked_rows.sort(key=lambda item: (-item.score, item.candidate_id))
        ranked = tuple(
            item.model_copy(update={"rank": rank}) for rank, item in enumerate(ranked_rows, start=1)
        )
        status_note = (
            "constraint status not supplied; candidates ranked as provided"
            if constraint_status is None
            else "excluded and unproven candidates omitted from the ranking"
        )
        value_payload: dict[str, dict[str, float]] = {}
        for item in parsed:
            value_payload[item.candidate_id] = {
                key: item.values[key] for key in sorted(item.values)
            }
        record = provenance_record(
            MethodId.WEIGHTED_SUM_RANKING,
            parameters={
                "profile_id": profile.profile_id,
                "example_profile": profile.example,
                "method": "weighted_sum_model",
                "ranked_count": len(ranked),
                "omitted_count": len(omitted),
                "criterion_ids": [item.indicator_id for item in profile.criteria],
                "used_weights": used_weights,
                "weight_mode": "normalized" if profile.normalize_weights else "as_supplied",
            },
            input_fingerprints=(
                fingerprint_canonical(
                    {
                        "profile_id": profile.profile_id,
                        "ids": [ident for ident, _ in rows],
                        "values": value_payload,
                    }
                ),
            ),
            notes=(
                "weights are configuration, not universal truth, and are not LLM-generated",
                status_note,
                *profile.notes[:2],
            ),
        )
        return RankingResult(
            profile_id=profile.profile_id,
            ranked=ranked,
            omitted=tuple(omitted),
            criteria=profile.criteria,
            used_weights=used_weights,
            normalization=tuple(stats_list),
            example_profile=profile.example,
            provenance=record,
            status=EvaluationStatus.EVALUATED,
        )

    def pareto(
        self,
        rows: CandidateRows,
        objectives: Sequence[ObjectiveSpec],
        *,
        constraint_status: ConstraintStatusInput = None,
    ) -> MultiObjectiveResult:
        """Non-dominated set from raw objective values. Not a WSM score."""

        from solardss.suitability.pareto import pareto_candidates

        return pareto_candidates(
            rows, objectives, constraint_status=constraint_status, catalog=self._catalog
        )


def rank_candidates(
    rows: CandidateRows,
    profile: DecisionProfile,
    *,
    constraint_status: ConstraintStatusInput = None,
    catalog: IndicatorRegistry | None = None,
) -> RankingResult:
    """Rank ``rows`` with the Weighted Sum Model."""

    return SuitabilityEngine(catalog).rank(rows, profile, constraint_status=constraint_status)


class _ParsedRow:
    def __init__(
        self,
        candidate_id: str,
        values: dict[str, float],
        missing: tuple[str, ...],
    ) -> None:
        self.candidate_id = candidate_id
        self.values = values
        self.missing = missing


class _Eligible:
    def __init__(
        self, candidate_id: str, status: FeasibilityStatus, values: dict[str, float]
    ) -> None:
        self.candidate_id = candidate_id
        self.status = status
        self.values = values


def _parse_row(
    candidate_id: str,
    values: Mapping[str | IndicatorId, ScoreValue],
    profile: DecisionProfile,
) -> _ParsedRow:
    mapped = map_indicator_values(values)
    parsed: dict[str, float] = {}
    missing: list[str] = []
    for criterion in profile.criteria:
        raw = mapped.get(criterion.indicator_id)
        if raw is None:
            missing.append(criterion.indicator_id)
            continue
        parsed[criterion.indicator_id] = _as_float(raw, criterion, candidate_id)
    return _ParsedRow(candidate_id, parsed, tuple(missing))


def _as_float(raw: ScoreValue, criterion: CriterionWeight, candidate_id: str) -> float:
    if isinstance(raw, Quantity):
        value = raw.require_unit(criterion.unit)
    else:
        value = Quantity(value=float(raw), unit=criterion.unit).require_unit(criterion.unit)
    if value != value or value in (float("inf"), float("-inf")):
        raise InvalidParameterError(
            f"{candidate_id}: {criterion.indicator_id} must be finite; missing stays missing"
        )
    return value


def _contribution(
    criterion: CriterionWeight,
    raw: float,
    scaled: float,
    weight: float,
    stats: NormalizationStats,
) -> CriterionContribution:
    return CriterionContribution(
        indicator_id=criterion.indicator_id,
        raw_value=raw,
        unit=criterion.unit,
        normalized_value=scaled,
        weight=weight,
        weighted_contribution=weight * scaled,
        normalization=criterion.normalization,
        range_min=stats.range_min,
        range_max=stats.range_max,
        range_source=stats.range_source,
        target_value=criterion.target_value,
        source=criterion.source,
        profile=criterion.profile,
        rationale=criterion.rationale,
    )
