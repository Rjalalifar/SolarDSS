"""Errors raised while materialising spatial candidates."""

from __future__ import annotations

from solardss.exceptions import InvalidGridError, InvalidParameterError, SolarDSSError


class CandidateGenerationError(SolarDSSError):
    """Candidate generation could not produce a valid set."""

    code = "candidate_generation_error"


class CandidateGridError(CandidateGenerationError, InvalidGridError):
    """A grid strategy cannot be realised as stated."""

    code = "candidate_grid_error"


class CandidateClipError(CandidateGenerationError, InvalidParameterError):
    """A boundary clip cannot be performed correctly with the available geometry."""

    code = "candidate_clip_error"
