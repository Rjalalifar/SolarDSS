"""Adapters that can fill a PVComponentCatalog.

Local files are implemented. Remote manufacturer databases are not: those
adapters exist so a future trusted source can be plugged in without inventing
product rows here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from solardss.exceptions import UnsupportedMethodError
from solardss.pv.catalog import PVComponentCatalog
from solardss.pv.io import load_pv_catalog


class AdapterAvailability(BaseModel):
    """Whether an adapter is configured. This is not a product list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    configured: bool
    reason: str = Field(min_length=1)


@runtime_checkable
class PVCatalogAdapter(Protocol):
    """Source of catalog rows. Implementations must not invent datasheets."""

    @property
    def adapter_id(self) -> str:
        """Stable adapter identity recorded in provenance."""

        ...

    def availability(self) -> AdapterAvailability:
        """Whether ``load`` can run. ``not_configured`` is not an empty catalog."""

        ...

    def load(self) -> PVComponentCatalog:
        """Return a validated catalog. Empty is allowed; fabricated rows are not."""

        ...


class LocalFileAdapter:
    """Load a caller-supplied JSON or CSV file."""

    def __init__(self, path: Path | str, *, kind: str | None = None) -> None:
        self._path = Path(path)
        self._kind = kind

    @property
    def adapter_id(self) -> str:
        return "local_file"

    def availability(self) -> AdapterAvailability:
        if self._path.is_file():
            return AdapterAvailability(configured=True, reason=f"local file {self._path.name}")
        return AdapterAvailability(
            configured=False, reason=f"catalog file {self._path.name} is not present"
        )

    def load(self) -> PVComponentCatalog:
        return load_pv_catalog(self._path, kind=self._kind)


class UnimplementedDatabaseAdapter:
    """Placeholder for a trusted external module database (for example CEC).

    SolarDSS does not download or fabricate that database.
    """

    def __init__(self, database_id: str) -> None:
        if not database_id.strip():
            raise UnsupportedMethodError("database_id must be a non-empty adapter name")
        self._database_id = database_id.strip()

    @property
    def adapter_id(self) -> str:
        return self._database_id

    def availability(self) -> AdapterAvailability:
        return AdapterAvailability(
            configured=False,
            reason=(
                f"{self._database_id} is not implemented in SolarDSS; "
                "load a local CSV/JSON catalog or plug in a future trusted adapter"
            ),
        )

    def load(self) -> PVComponentCatalog:
        raise UnsupportedMethodError(
            f"{self._database_id} catalog adapter is not implemented. "
            "SolarDSS does not invent manufacturer data; supply a local file."
        )


KNOWN_FUTURE_ADAPTERS: tuple[str, ...] = (
    "nrel_cec",
    "iec_61853_library",
)
