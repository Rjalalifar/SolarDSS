"""Typed pieces of a solar PV decision problem.

These models declare *what is being asked*. They do not geocode, fetch maps,
generate candidates, or rank sites.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from solardss.core.jsonutil import JsonScalar
from solardss.data.contracts import DatasetRef
from solardss.data.tables import CandidateTable
from solardss.exceptions import InvalidParameterError


class DecisionOperation(str, Enum):
    """Operations a problem may authorize. Authorization is not execution."""

    CANDIDATE_GENERATION = "candidate_generation"
    SITE_RANKING = "site_ranking"
    SYSTEM_EVALUATION = "system_evaluation"
    SYSTEM_DESIGN = "system_design"


class ConstraintOperator(str, Enum):
    """Comparison applied to a caller-supplied constraint value."""

    EQ = "eq"
    NE = "ne"
    LT = "lt"
    LE = "le"
    GT = "gt"
    GE = "ge"
    EXCLUDE = "exclude"
    INCLUDE = "include"


class CandidateSet(BaseModel):
    """Identified candidate sites or buildings. Geometry stays with the caller."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    candidate_ids: tuple[str, ...] = Field(min_length=1)
    label: str | None = None
    dataset: DatasetRef | None = None

    @model_validator(mode="after")
    def _unique_ids(self) -> CandidateSet:
        if any(not ident for ident in self.candidate_ids):
            raise InvalidParameterError("Candidate ids must be non-empty strings")
        if len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise InvalidParameterError("Candidate ids must be unique")
        return self

    @classmethod
    def from_table(cls, table: CandidateTable, *, label: str | None = None) -> CandidateSet:
        """Copy ids from a ``CandidateTable`` without interpreting geometry."""

        return cls(candidate_ids=table.ids(), label=label)

    def __len__(self) -> int:
        return len(self.candidate_ids)


class EnvironmentalPreferences(BaseModel):
    """Caller policy over registered environmental/land indicators.

    SolarDSS does not invent impact weights.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    avoid_protected_areas: bool | None = None
    minimize_indicator_ids: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


class ModulePreferences(BaseModel):
    """Optional technology hints. Not a performance model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    technology: str | None = None
    model: str | None = None
    tracking_mode: str | None = None


class UserConstraint(BaseModel):
    """A single caller-stated bound. Values are not computed here."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operator: ConstraintOperator
    value: JsonScalar = None
    field: str | None = None
    indicator_id: str | None = None

    @model_validator(mode="after")
    def _target(self) -> UserConstraint:
        if not self.field and not self.indicator_id:
            raise InvalidParameterError("UserConstraint requires field or indicator_id")
        return self


class MissingRequirement(BaseModel):
    """Structured gap. Adapters decide whether to ask a human; this is not dialogue."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    field: str = Field(min_length=1)
    required: bool
    reason: str = Field(min_length=1)
