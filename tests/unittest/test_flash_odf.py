from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest
from _odf_test_utils import build_odf_package, build_odp_fixture, build_ods_fixture, build_odt_fixture
from docvortex.analyzers.native import OdpModel, OdsModel, OdtModel
from docvortex.analyzers.native.office.odf.metadata import MAX_ODT_METADATA_PAGE_COUNT, extract_odf_metadata
from docvortex.document.detection import guess_suffix_by_path

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.file_io import extract_metadata
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.parse_svc import ParseService
from mineru.errors import InvalidRequestError
from mineru.parser import api_server, parse, parse_async
from mineru.parser.api_server import CreateJobRequest, FileStore


@pytest.mark.parametrize(
    ("suffix", "model_class", "payload", "page_count"),
    [
        ("odt", OdtModel, build_odt_fixture(), 1),
        ("ods", OdsModel, build_ods_fixture(), 2),
        ("odp", OdpModel, build_odp_fixture(), 3),
    ],
    ids=["odt", "ods", "odp"],
)
def test_odf_models_and_analyze_keep_flash_contract(
    suffix: str,
    model_class: type[Any],
    payload: bytes,
    page_count: int,
) -> None:
    """验证三个 ODF 模型、同步/异步入口及输入流所有权。"""
    stream = BytesIO(payload)
    model_pages = model_class().predict(stream)
    assert not stream.closed
    assert len(model_pages) == page_count

    middle, model = doc_analyze(payload, effort="xhigh", parse_mode="ocr", file_suffix=suffix)  # type: ignore[arg-type]
    async_middle, async_model = asyncio.run(
        aio_doc_analyze(payload, effort="medium", parse_mode="auto", file_suffix=suffix)  # type: ignore[arg-type]
    )
    assert model.pages == async_model.pages == model_pages
    assert middle.model_dump() == async_middle.model_dump()
    assert model.metadata.file_suffix == middle.metadata.file_suffix == suffix
    assert model.extensions["mineru"]["tier"] == middle.extensions["mineru"]["tier"] == "flash"
    assert model.extensions["mineru"]["parse_mode"] == middle.extensions["mineru"]["parse_mode"] == "txt"


def test_plain_text_renamed_to_odf_is_not_accepted(tmp_path: Path) -> None:
    """验证 ODF 扩展名本身不能把普通文本升级为结构化文档。"""
    source = tmp_path / "fake.odt"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    assert guess_suffix_by_path(source) not in {"odt", "ods", "odp"}
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse(source)


@pytest.mark.parametrize(
    ("suffix", "payload", "expected"),
    [
        ("odt", build_odt_fixture(), {"page_count": 3, "title": "ODT Meta", "author": "Alice", "keywords": "one"}),
        ("ods", build_ods_fixture(), {"page_count": 2}),
        ("odp", build_odp_fixture(), {"page_count": 3}),
    ],
    ids=["odt", "ods", "odp"],
)
def test_doclib_extracts_odf_metadata(
    tmp_path: Path,
    suffix: str,
    payload: bytes,
    expected: dict[str, object],
) -> None:
    """验证 doclib ODF 元数据分支不复用 CSV 或 RTF 逻辑。"""
    source = tmp_path / f"sample.{suffix}"
    source.write_bytes(payload)
    metadata = asyncio.run(extract_metadata(str(source)))
    for key, value in expected.items():
        assert metadata[key] == value


def test_odt_metadata_page_count_is_bounded_before_doclib_range_expansion() -> None:
    """验证 producer page-count 不会把微小 ODT 扩张为无界任务范围。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0">
 <office:body><office:text/></office:body>
</office:document-content>"""
    meta = """<office:document-meta
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:meta="urn:oasis:names:tc:opendocument:xmlns:meta:1.0">
 <office:meta><meta:document-statistic meta:page-count="999999999999"/></office:meta>
</office:document-meta>"""

    metadata = extract_odf_metadata(BytesIO(build_odf_package("odt", content, meta_xml=meta)), "odt")

    assert MAX_ODT_METADATA_PAGE_COUNT == 10_000
    assert metadata["page_count"] == MAX_ODT_METADATA_PAGE_COUNT


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [("odt", build_odt_fixture()), ("ods", build_ods_fixture()), ("odp", build_odp_fixture())],
    ids=["odt", "ods", "odp"],
)
def test_public_parser_handles_odf_sync_and_async(
    tmp_path: Path,
    suffix: str,
    payload: bytes,
) -> None:
    """验证路径解析器依靠内容识别进入 ODF，并保留原始后缀元数据。"""
    source = tmp_path / f"sample.{suffix}"
    source.write_bytes(payload)
    result = parse(source)
    async_result = asyncio.run(parse_async(source))
    assert result.middle_json.metadata.file_suffix == async_result.middle_json.metadata.file_suffix == suffix
    assert result.middle_json.model_dump() == async_result.middle_json.model_dump()


def test_odf_parse_server_job_emits_flash_outputs(tmp_path: Path) -> None:
    """验证 local parse server 接受 ODF 并输出严格 Middle JSON 与结构化内容。"""
    source = tmp_path / "sample.odt"
    source.write_bytes(build_odt_fixture())
    file_store = FileStore(tmp_path / "api-files")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}}],
            "tier": "standard",
            "output_formats": ["markdown", "middle_json", "structured_content"],
        }
    )
    record = api_server.JobStore().create(request, file_store)
    asyncio.run(
        api_server._run_job(
            record,
            request,
            file_store,
            image_analysis=True,
            allow_local_source=True,
        )
    )
    parsed_file = record.files[0]
    assert parsed_file.status == "completed"
    assert parsed_file.output_files is not None
    middle_record = file_store.get_file(parsed_file.output_files.middle_json.file_id)  # type: ignore[union-attr]
    assert middle_record.sha256sum is not None
    payload = json.loads(file_store.read_blob(middle_record.sha256sum))
    assert payload["metadata"]["file_suffix"] == "odt"
    assert payload["extensions"]["mineru"]["tier"] == "flash"
    assert payload["extensions"]["mineru"]["parse_mode"] == "txt"


@pytest.mark.parametrize(
    ("suffix", "payload", "page_count"),
    [("odt", build_odt_fixture(), 3), ("ods", build_ods_fixture(), 2), ("odp", build_odp_fixture(), 3)],
    ids=["odt", "ods", "odp"],
)
def test_doclib_ingests_odf_as_local_flash(
    tmp_path: Path,
    suffix: str,
    payload: bytes,
    page_count: int,
) -> None:
    """验证 doclib 为 ODF 建立本地 flash parse row 和正确页数。"""

    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, object]]:
            """关闭 parsing rules，让测试只观察默认 ODF 行为。"""
            return []

    async def run() -> None:
        """执行隔离 SQLite 入库并检查文档与解析任务。"""
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),  # type: ignore[arg-type]
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / f"sample.{suffix}"
        source.write_bytes(payload)
        response = await service.request_parse(str(source), tier="flash")
        doc = await db.fetchone(
            "SELECT file_type, page_count FROM docs WHERE sha256=?",
            (response.sha256,),
        )
        parses = await db.fetchall(
            "SELECT tier, status, privacy FROM parses WHERE sha256=?",
            (response.sha256,),
        )
        assert response.tier == "flash"
        assert doc == {"file_type": suffix, "page_count": page_count}
        assert parses == [{"tier": "flash", "status": "pending", "privacy": "local"}]

    asyncio.run(run())


def test_doclib_rejects_odf_remote_parse(tmp_path: Path) -> None:
    """验证 ODF 继承非 PDF/image 的严格 remote 拒绝语义。"""

    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, object]]:
            """关闭 parsing rules，让测试只观察主动请求校验。"""
            return []

    async def run() -> None:
        """创建隔离 doclib 并断言稳定错误码。"""
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),  # type: ignore[arg-type]
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "sample.odt"
        source.write_bytes(build_odt_fixture())
        with pytest.raises(InvalidRequestError) as exc_info:
            await service.request_parse(str(source), tier="flash", remote=True)
        assert exc_info.value.code == "remote_unsupported_for_file_type"
        assert exc_info.value.param == "remote"

    asyncio.run(run())
