from __future__ import annotations

import asyncio
import json
from io import BytesIO
from pathlib import Path

import pytest
from _ofd_test_utils import build_ofd_package, page_xml, text_object
from docvortex.analyzers.native import OfdModel
from docvortex.analyzers.native.ofd import detect_ofd
from docvortex.document.detection import guess_suffix_by_bytes, guess_suffix_by_path

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.file_io import extract_metadata
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.parse_svc import ParseService
from mineru.errors import InvalidRequestError
from mineru.parser import MinerUParser, api_server
from mineru.parser.api_server import CreateJobRequest, FileStore
from mineru.render import render_docx, render_html, render_markdown, render_structured_content
from mineru.types import BlockType


def _minimal_payload(*, namespace: str = "http://www.ofdspec.org/2016", version: str = "1.0") -> bytes:
    """构造包含单行文字的最小 OFD。"""
    content = text_object(3, "你好，OFD！", boundary="10 10 50 12", delta_x="g 6 5")
    return build_ofd_package(
        [("Pages/Page_0/Content.xml", page_xml(content, namespace=namespace))],
        namespace=namespace,
        version=version,
    )


def test_ofd_model_analyze_detection_and_renderers(tmp_path: Path) -> None:
    """验证 OFD 从内容识别到统一四类 renderer 的完整链路。"""
    payload = _minimal_payload()
    source = tmp_path / "disguised.csv"
    source.write_bytes(payload)

    assert detect_ofd(payload)
    assert guess_suffix_by_bytes(payload, str(source)) == "ofd"
    assert guess_suffix_by_path(source) == "ofd"
    model_pages = OfdModel().predict(BytesIO(payload))
    middle, model = doc_analyze(payload, effort="xhigh", parse_mode="ocr", file_suffix="ofd")
    async_middle, async_model = asyncio.run(aio_doc_analyze(payload, effort="medium", file_suffix="ofd"))

    assert model.pages == async_model.pages == model_pages
    assert middle.model_dump() == async_middle.model_dump()
    assert model.metadata.file_suffix == middle.metadata.file_suffix == "ofd"
    assert model.extensions["mineru"]["tier"] == middle.extensions["mineru"]["tier"] == "flash"
    assert model.extensions["mineru"]["parse_mode"] == middle.extensions["mineru"]["parse_mode"] == "txt"
    assert middle.is_full_document is True
    assert middle.pages[0].blocks[0].type == BlockType.TEXT
    assert middle.pages[0].blocks[0].bbox is not None
    assert "你好，OFD" in render_markdown(middle)
    assert "你好，OFD" in render_html(middle)
    assert render_structured_content(middle)["metadata"]["file_suffix"] == "ofd"
    assert render_docx(middle).startswith(b"PK")


def test_ofd_parser_rejects_page_range_and_doclib_reads_metadata(tmp_path: Path) -> None:
    """验证 OFD 只支持整本解析且 Doclib 读取原生页数和标题。"""
    payload = _minimal_payload()
    source = tmp_path / "sample.ofd"
    source.write_bytes(payload)

    with pytest.raises(InvalidRequestError) as exc_info:
        MinerUParser(tier="flash").parse(source, page_range="1")
    assert exc_info.value.code == "page_range_invalid"

    metadata = asyncio.run(extract_metadata(str(source)))
    assert metadata["page_count"] == 1
    assert metadata["title"] == "Fixture Title"
    assert metadata["author"] == "Fixture Author"


def test_ofd_parse_server_job_emits_flash_outputs(tmp_path: Path) -> None:
    """验证本地 Parse Jobs 接受 OFD 并输出严格 Middle JSON。"""
    source = tmp_path / "sample.ofd"
    source.write_bytes(_minimal_payload())
    file_store = FileStore(tmp_path / "api-files")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}}],
            "tier": "standard",
            "output_formats": ["markdown", "middle_json", "structured_content", "zip"],
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
    assert payload["metadata"]["file_suffix"] == "ofd"
    assert payload["extensions"]["mineru"]["tier"] == "flash"
    assert payload["extensions"]["mineru"]["parse_mode"] == "txt"
    assert payload["pages"][0]["blocks"][0]["bbox"]


def test_doclib_ingests_ofd_as_local_full_document_flash(tmp_path: Path) -> None:
    """验证 Doclib 为 OFD 建立本地整本 flash 解析任务。"""

    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, object]]:
            """关闭 parsing rules，让测试只观察 OFD 默认行为。"""
            return []

    async def run() -> None:
        """执行隔离 SQLite 入库并检查 OFD 文档与解析任务。"""
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),  # type: ignore[arg-type]
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "sample.ofd"
        source.write_bytes(_minimal_payload())
        response = await service.request_parse(str(source), tier="flash")
        doc = await db.fetchone(
            "SELECT file_type, page_count, title FROM docs WHERE sha256=?",
            (response.sha256,),
        )
        parses = await db.fetchall(
            "SELECT tier, status, privacy FROM parses WHERE sha256=?",
            (response.sha256,),
        )
        assert response.tier == "flash"
        assert doc == {"file_type": "ofd", "page_count": 1, "title": "Fixture Title"}
        assert parses == [{"tier": "flash", "status": "pending", "privacy": "local"}]

    asyncio.run(run())
