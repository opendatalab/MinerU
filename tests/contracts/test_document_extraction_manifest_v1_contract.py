from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "schemas" / "document-extraction-manifest-v1.schema.json"
FIXTURE = (
    ROOT
    / "examples"
    / "contracts"
    / "document-extraction-manifest-v1"
    / "example.json"
)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_document_extraction_manifest_schema_and_fixture_are_source_owned() -> None:
    schema = _load(SCHEMA)
    fixture = _load(FIXTURE)

    assert schema["title"] == "MinerU document extraction manifest v1"
    assert schema["properties"]["contract_version"]["const"] == (
        "document-extraction-manifest-v1"
    )
    assert fixture["contract_version"] == "document-extraction-manifest-v1"
    assert fixture["derivative_not_native_source_evidence"] is True
    assert fixture["does_not_establish_source_sufficiency"] is True
    assert "evidence_status" not in fixture
    assert "readiness_status" not in fixture
    assert "regulatory_conclusion" not in fixture


def test_document_extraction_manifest_schema_requires_identity_and_outputs() -> None:
    schema = _load(SCHEMA)
    required = schema["required"]

    assert {
        "contract_version",
        "extraction_run_id",
        "source_id",
        "source_version_id",
        "input_sha256",
        "mineru_identity",
        "runtime_identity",
        "outputs",
    } <= set(required)
