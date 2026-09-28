"""Injectable clocks so retrieval timestamps stay deterministic in tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Protocol


class Clock(Protocol):
    """Returns timezone-aware instants."""

    def now(self) -> datetime:
        """Current instant used as ``retrieved_at`` on a cache miss."""


class UtcClock:
    """Wall-clock UTC. Tests should inject ``FixedClock`` or ``SequenceClock``."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)


def _aware_utc(instant: datetime, *, clock_name: str) -> datetime:
    if instant.tzinfo is None:
        raise ValueError(f"{clock_name} instants must be timezone-aware")
    return instant.astimezone(timezone.utc)


class FixedClock:
    """Always returns the same instant."""

    def __init__(self, instant: datetime) -> None:
        self._instant = _aware_utc(instant, clock_name="FixedClock")

    def now(self) -> datetime:
        return self._instant


class SequenceClock:
    """Returns successive instants, then raises when the sequence is exhausted."""

    def __init__(self, instants: Sequence[datetime]) -> None:
        if not instants:
            raise ValueError("SequenceClock requires at least one instant")
        self._instants = tuple(_aware_utc(item, clock_name="SequenceClock") for item in instants)
        self._index = 0

    def now(self) -> datetime:
        if self._index >= len(self._instants):
            raise RuntimeError("SequenceClock has no remaining instants")
        instant = self._instants[self._index]
        self._index += 1
        return instant
