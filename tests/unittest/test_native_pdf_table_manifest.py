# Copyright (c) Opendatalab. All rights reserved.
"""验证仓库内 Native PDF Table 真实语料、隐私和结构发布门。"""

from __future__ import annotations

import json
import hashlib
import subprocess
import sys
from pathlib import Path

from pypdf import PdfReader


_PROJECT_ROOT = Path(__file__).parents[2]
_FIXTURE_ROOT = Path(__file__).parent / "pdfs" / "native_pdf_tables"
_EVALUATOR = _PROJECT_ROOT / "tests" / "fixtures" / "evaluate_native_pdf_table_manifest.py"
_EXPECTED_PAGE_COUNTS = {
    "annual_report_fundraising_projects_table.pdf": 2,
    "annual_report_management_roles_table.pdf": 2,
    "annual_report_research_projects_table.pdf": 2,
    "conference_schedule_tables.pdf": 12,
    "engineering_process_restrictions_table.pdf": 9,
    "fund_asset_and_transaction_tables.pdf": 2,
    "fund_manager_profile_table.pdf": 2,
    "manufacturing_facilities_cross_page_table.pdf": 2,
    "pollutant_discharge_tables.pdf": 89,
    "procurement_contract_tables.pdf": 2,
    "procurement_document_blank_page_tables.pdf": 4,
    "quarterly_report_financial_tables.pdf": 2,
}
_SENSITIVE_PROBE_HASHES = (
    (14, "f467375e9c732abd34da5916ef67d09601fe5a160fc07e34eae45b40b92a66fd"),
    (12, "14627c288df62a412eca1a27ed52b6126bf2958b4be4a67694dd7d1144937f11"),
    (10, "9b2e4ae1fe4b64def12ddb3e1096c542a47a8278c789d321641dbef69e9711cd"),
    (18, "df3f9bd04eb98a918d39eb50ff84d22d16296aa39297e2b6c93bc27d5ebd301f"),
    (2, "a03f26ca7f4771eecaf858e1dced39d5213c3423084bafdb62d1086c382ae906"),
    (3, "99bc11448af2388b7cdba8f712eab541b868931286bce3162703de572b27d97f"),
    (11, "ed0491aea67e74b66b983ee86ee32f067693f7603812b66127fd8e4be79ebbdc"),
)


def _contains_sensitive_probe_hash(text: str) -> bool:
    """使用不可逆摘要检查已知敏感片段，避免把原文写入仓库。"""

    for length, expected_digest in _SENSITIVE_PROBE_HASHES:
        if any(
            hashlib.sha256(text[start : start + length].encode()).hexdigest() == expected_digest
            for start in range(max(0, len(text) - length + 1))
        ):
            return True
    return False


def test_repository_native_table_manifest_matches_all_fixtures() -> None:
    """验证仓库内 133 表真值及六个 Flash 精确目标全部匹配。"""

    completed = subprocess.run(
        [
            sys.executable,
            str(_EVALUATOR),
            "--skip-performance-gate",
        ],
        cwd=_PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["summary"] | {
        "p95_milliseconds": 0.0,
    } == {
        "accuracy_scope": "targeted_only",
        "tables": 133,
        "html": 133,
        "coverage": 1.0,
        "target_tables": 41,
        "target_html": 41,
        "target_mismatches": 0,
        "target_precision": 1.0,
        "regression_mismatches": 0,
        "flash_targets": 6,
        "flash_target_html": 6,
        "flash_target_mismatches": 0,
        "p95_milliseconds": 0.0,
    }


def test_repository_native_table_fixtures_are_sanitized() -> None:
    """验证文件清单、页数、文档信息和污染物敏感字段均符合提交边界。"""

    fixture_paths = sorted(_FIXTURE_ROOT.glob("*.pdf"))
    assert {path.name for path in fixture_paths} == set(_EXPECTED_PAGE_COUNTS)
    for path in fixture_paths:
        reader = PdfReader(path)
        assert len(reader.pages) == _EXPECTED_PAGE_COUNTS[path.name]
        assert not reader.metadata
        assert reader.xmp_metadata is None
        assert reader.trailer.get("/Info") is None
        assert "/Metadata" not in reader.root_object
        assert all("/Metadata" not in page for page in reader.pages)

    pollutant_reader = PdfReader(_FIXTURE_ROOT / "pollutant_discharge_tables.pdf")
    visible_text = "".join("".join((page.extract_text() or "").split()) for page in pollutant_reader.pages)
    assert not _contains_sensitive_probe_hash(visible_text)
