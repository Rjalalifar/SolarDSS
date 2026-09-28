"""Provenance records hash content, not paths."""

from __future__ import annotations

from solardss.core.methods import MethodId
from solardss.data import DatasetRef
from solardss.provenance import provenance_record, software_stamp
from solardss.version import __version__


def test_software_stamp() -> None:
    stamp = software_stamp(MethodId.MINMAX_NORMALIZE)
    assert stamp.package == "solardss"
    assert stamp.version == __version__
    assert stamp.method_id == "minmax_normalize"
    assert stamp.method_version == "1"


def test_provenance_does_not_require_paths() -> None:
    record = provenance_record(
        MethodId.BOOLEAN_EXCLUSION,
        parameters={"layer_names": ["water"]},
        input_fingerprints=("abc123",),
    )
    dumped = record.model_dump()
    assert "path" not in dumped
    assert "uri" not in dumped
    assert record.input_fingerprints == ("abc123",)


def test_dataset_ref_uri_is_optional() -> None:
    ref = DatasetRef(name="ghi", media_type="application/x-numpy")
    assert ref.uri is None
    assert ref.fingerprint is None
