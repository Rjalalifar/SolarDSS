"""Input tables and dataset references. No builtin paths or GIS I/O."""

from __future__ import annotations

from solardss.data.contracts import DatasetRef
from solardss.data.geo import BoundingBox, GeographicRef
from solardss.data.gis import GIS_EXTRA_MODULES, gis_extra_status, require_gis
from solardss.data.tables import DEFAULT_ID_COLUMN, CandidateTable, validate_candidate_table

__all__ = [
    "DEFAULT_ID_COLUMN",
    "GIS_EXTRA_MODULES",
    "BoundingBox",
    "CandidateTable",
    "DatasetRef",
    "GeographicRef",
    "gis_extra_status",
    "require_gis",
    "validate_candidate_table",
]
