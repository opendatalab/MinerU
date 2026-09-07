from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest
from _epub_test_utils import (
    build_epub_fixture,
)
from docvortex.analyzers.native import EpubModel
from docvortex.document.detection import guess_suffix_by_bytes, guess_suffix_by_path

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.file_io import extract_metadata
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.parse_svc import ParseService
from mineru.errors import InvalidRequestError
from mineru.parser import MinerUParser, api_server, parse, parse_async
from mineru.parser.api_server import CreateJobRequest, FileStore
from mineru.render import render_docx, render_html, render_markdown, render_structured_content
from mineru.types import BlockType


def test_epub_model_analyze_and_renderers_preserve_structured_content() -> None:
    """验证 EPUB 模型、统一 Analyze 及四种 renderer 保留核心结构。"""
    payload = build_epub_fixture()
    stream = BytesIO(payload)
    model_pages = EpubModel().predict(stream)
    assert not stream.closed
    assert len(model_pages) == 3

    middle, model = doc_analyze(payload, effort="xhigh", parse_mode="ocr", file_suffix="epub")
    explicit_full_middle, explicit_full_model = doc_analyze(payload, page_index_map=[], file_suffix="epub")
    async_middle, async_model = asyncio.run(aio_doc_analyze(payload, effort="medium", parse_mode="auto", file_suffix="epub"))
    assert model.pages == async_model.pages == model_pages
    assert explicit_full_model.pages == model_pages
    assert explicit_full_middle.is_full_document is True
    assert middle.model_dump() == async_middle.model_dump()
    assert model.file_suffix == middle.file_suffix == "epub"
    assert model.extensions["mineru"]["effort"] == middle.extensions["mineru"]["effort"] == "flash"
    assert model.extensions["mineru"]["parse_mode"] == middle.extensions["mineru"]["parse_mode"] == "txt"
    assert [page.page_idx for page in middle.pages] == [0, 1, 2]
    assert middle.pages[0].blocks[0].type == BlockType.DOC_TITLE

    raw_blocks = [block for page in model.pages for block in page]
    raw_types = [block["type"] for block in raw_blocks]
    for expected_type in (
        BlockType.DOC_TITLE,
        BlockType.PARAGRAPH_TITLE,
        BlockType.TEXT,
        BlockType.LIST,
        BlockType.TABLE,
        BlockType.CODE,
        BlockType.EQUATION,
        BlockType.IMAGE,
        BlockType.PAGE_FOOTNOTE,
    ):
        assert expected_type in raw_types
    assert "hidden secret" not in str(raw_blocks)
    assert "alert('active')" not in str(raw_blocks)
    assert "Remote image" in str(raw_blocks)
    assert any(block.get("content") == "y^2" for block in raw_blocks)
    assert any(block.get("image_base64", "").startswith("data:image/png;base64,") for block in raw_blocks)

    markdown = render_markdown(middle)
    html_output = render_html(middle)
    structured = render_structured_content(middle)
    docx = render_docx(middle)
    assert "Chapter One" in markdown and "Section Two" in markdown
    assert "hidden secret" not in markdown
    assert "Data table" in markdown
    assert "<table" in html_output
    assert structured["file_suffix"] == "epub"
    assert docx.startswith(b"PK")


def test_public_parser_rejects_epub_page_range(tmp_path: Path) -> None:
    """验证 EPUB 公共 Parser 只接受整本解析。"""
    source = tmp_path / "book.epub"
    source.write_bytes(build_epub_fixture())

    with pytest.raises(InvalidRequestError, match="only supported for PDF") as exc_info:
        parse(source, page_range="2-3")
    assert exc_info.value.code == "page_range_invalid"

    with pytest.raises(InvalidRequestError, match="only supported for PDF"):
        asyncio.run(parse_async(source, page_range="2-3"))


@pytest.mark.parametrize(
    "suffix",
    ["doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf", "csv", "epub", "html", "ofd", "odt", "ods", "odp"],
)
def test_non_pdf_analyze_rejects_non_empty_page_index_map(suffix: str) -> None:
    """验证所有非 PDF Analyze 分支拒绝伪造 partial page mapping。"""
    with pytest.raises(ValueError, match="only supported for PDF"):
        doc_analyze(b"not-read", page_index_map=[0], file_suffix=suffix)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "suffix",
    ["doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf", "csv", "epub", "html", "ofd", "odt", "ods", "odp"],
)
def test_non_pdf_public_parser_rejects_page_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suffix: str,
) -> None:
    """验证所有非 PDF 路径入口在初始化具体模型前拒绝 page_range。"""
    source = tmp_path / f"sample.{suffix}"
    source.write_bytes(b"not-read")
    monkeypatch.setattr("docvortex.document.detection.guess_suffix_by_path", lambda _path: suffix)
    parser = MinerUParser(tier="flash")
    with pytest.raises(InvalidRequestError, match="full-document parsing") as exc_info:
        parser.parse(source, page_range="1")
    assert exc_info.value.code == "page_range_invalid"


def test_epub_content_detection_precedes_extension_and_rejects_fake_packages(tmp_path: Path) -> None:
    """验证 EPUB 强内容身份覆盖伪装扩展名，而普通 ZIP/文本不能依赖扩展名通过。"""
    payload = build_epub_fixture()
    disguised = tmp_path / "book.csv"
    disguised.write_bytes(payload)
    assert guess_suffix_by_bytes(payload, str(disguised)) == "epub"
    assert guess_suffix_by_path(disguised) == "epub"

    fake = tmp_path / "fake.epub"
    fake.write_text("not an epub", encoding="utf-8")
    assert guess_suffix_by_path(fake) != "epub"
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse(fake)


def test_doclib_extracts_epub_metadata(tmp_path: Path) -> None:
    """验证 doclib 从 OPF 读取元数据和 spine 逻辑页数。"""
    source = tmp_path / "book.epub"
    source.write_bytes(build_epub_fixture())
    metadata = asyncio.run(extract_metadata(str(source)))
    assert metadata == {
        "page_count": 3,
        "title": "EPUB Fixture",
        "author": "Alice",
        "subject": "Testing",
        "keywords": "Testing, epub, mineru",
        "is_image_based": 0,
    }


def test_epub_local_parse_job_emits_spine_aligned_flash_outputs(tmp_path: Path) -> None:
    """验证本地 Parse Jobs 接受 EPUB，并按 spine 输出全部正文。"""
    source = tmp_path / "book.epub"
    source.write_bytes(build_epub_fixture())
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
    assert parsed_file.output_files.zip is not None
    middle_record = file_store.get_file(parsed_file.output_files.middle_json.file_id)  # type: ignore[union-attr]
    assert middle_record.sha256sum is not None
    payload = json.loads(file_store.read_blob(middle_record.sha256sum))
    assert payload["file_suffix"] == "epub"
    assert payload["effort"] == "flash"
    assert payload["parse_mode"] == "txt"
    assert [page["page_idx"] for page in payload["pages"]] == [0, 1, 2]
    assert payload["pages"][0]["blocks"][0]["type"] == "doc_title"


def test_epub_local_parse_job_preserves_page_range_error_code(tmp_path: Path) -> None:
    """验证 Parse Jobs 对 EPUB 显式范围返回 page_range_invalid。"""
    source = tmp_path / "book.epub"
    source.write_bytes(build_epub_fixture())
    file_store = FileStore(tmp_path / "api-files")
    request = CreateJobRequest.model_validate(
        {
            "files": [{"source": {"type": "local", "path": str(source)}, "page_range": "2"}],
            "tier": "flash",
            "output_formats": ["middle_json"],
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
    assert record.files[0].status == "failed"
    assert record.files[0].error is not None
    assert record.files[0].error.code == "page_range_invalid"


def test_doclib_ingests_epub_as_local_flash_with_spine_page_count(tmp_path: Path) -> None:
    """验证 doclib 为 EPUB 建立整本 flash row，并记录 spine 页数。"""

    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, object]]:
            """关闭 parsing rules，让测试只观察 EPUB 默认行为。"""
            return []

    async def run() -> None:
        """执行隔离 SQLite 入库并检查 EPUB 文档与解析任务。"""
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),  # type: ignore[arg-type]
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "book.epub"
        source.write_bytes(build_epub_fixture())
        response = await service.request_parse(str(source), tier="flash")
        doc = await db.fetchone(
            "SELECT file_type, page_count FROM docs WHERE sha256=?",
            (response.sha256,),
        )
        parses = await db.fetchall(
            "SELECT tier, status, privacy, page_range FROM parses WHERE sha256=?",
            (response.sha256,),
        )
        assert response.tier == "flash"
        assert doc == {"file_type": "epub", "page_count": 3}
        assert parses == [{"tier": "flash", "status": "pending", "privacy": "local", "page_range": "1-3"}]

        with pytest.raises(InvalidRequestError) as range_exc:
            await service.request_parse(str(source), tier="flash", page_range="2")
        assert range_exc.value.code == "page_range_invalid"

        with pytest.raises(InvalidRequestError) as exc_info:
            await service.request_parse(str(source), tier="flash", remote=True)
        assert exc_info.value.code == "remote_unsupported_for_file_type"

    asyncio.run(run())


def test_doclib_local_epub_task_clears_persisted_full_page_range(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 doclib 缓存仍记完整覆盖范围，但调用非 PDF Parser 时不传 page_range。"""
    observed: dict[str, object] = {}
    expected = object()

    def fake_parse(path: str, *, tier: str, page_range: str) -> object:
        """记录 doclib 传给本地 Parser 的整本参数。"""
        observed.update(path=path, tier=tier, page_range=page_range)
        return expected

    monkeypatch.setattr("mineru.parser.parse", fake_parse)
    service = object.__new__(ParseService)
    result = asyncio.run(
        service._parse_via_local(  # type: ignore[arg-type]
            {"path": "/tmp/book.epub", "ext": "epub"},
            "flash",
            "1-3",
        )
    )
    assert result is expected
    assert observed == {"path": "/tmp/book.epub", "tier": "flash", "page_range": ""}


def test_epub_public_import_does_not_load_heavy_models() -> None:
    """验证公开 Parser 导入不会提前加载 EPUB、Torch、OpenCV 或 VLM。"""
    code = """
import sys
import mineru.parser
blocked = ('torch', 'cv2', 'docvortex.analyzers.native.epub', 'mineru_vl_utils')
assert not any(name == prefix or name.startswith(prefix + '.') for prefix in blocked for name in sys.modules)
print('ok')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_epub_middle_json_roundtrip_remains_schema_2() -> None:
    """验证 EPUB 只扩展 file_suffix，不引入新的 schema 或 Block 字段。"""
    middle, _ = doc_analyze(build_epub_fixture(), file_suffix="epub")
    payload = middle.to_dict(skip_defaults=False)
    assert payload["file_suffix"] == "epub"
    assert json.loads(middle.to_json(skip_defaults=False))["file_suffix"] == "epub"
