"""验证原始文档属性跨 MinerU 输入处理、JSON 和 Doclib 的传递。"""

from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

from docvortex.schema import DocumentMetadata, DocumentProperties, MiddleJson, PageInfo, Producer
from pypdf import PdfWriter
import pytest

from mineru.backend.analysis.contracts import AnalysisResult
from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.parse_svc import ParseService, ensure_doc_record
from mineru.parser import MinerUParser, ParseResult


def source_pdf() -> bytes:
    """构造两页带标题和作者的 PDF，用于验证抽页不丢原始属性。"""
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    writer.add_blank_page(width=100, height=100)
    writer.add_metadata({"/Title": "Original title", "/Author": "Source author"})
    stream = BytesIO()
    writer.write(stream)
    return stream.getvalue()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_parser_preserves_source_metadata_before_selection(tmp_path: Path, asynchronous: bool) -> None:
    """同步和异步解析均保存完整源文档计数，而非重写后的一页。"""
    source = tmp_path / "document.pdf"
    source.write_bytes(source_pdf())
    parser = MinerUParser(tier="flash", parse_mode="txt")
    result = asyncio.run(parser.parse_async(source, page_range="2")) if asynchronous else parser.parse(source, page_range="2")
    props = result.middle_json.metadata.document
    assert (props.title, props.authors, props.page_count) == ("Original title", ["Source author"], 2)
    assert result.middle_json.pages[0].page_idx == 1
    assert result.middle_json.metadata.producer.name == "mineru"
    assert result._model_output.metadata.document == props
    assert ParseResult.from_json(result.to_json()).middle_json.metadata.document == props
    assert result.structured_content()["metadata"]["document"]["title"] == "Original title"


@pytest.mark.parametrize("effort", ["flash", "medium", "high", "xhigh"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_backend_all_tiers_receive_same_source_metadata(
    monkeypatch: pytest.MonkeyPatch,
    effort: str,
    asynchronous: bool,
) -> None:
    """直接分析入口不依赖 parser 预处理，也不把源属性归入某个推理档位。"""
    from mineru.backend.analysis.pdf import pipeline

    def analyze(data: bytes, **kwargs: object) -> AnalysisResult:
        """替代模型推理，只保留真实统一分析门面的元数据行为。"""
        return AnalysisResult(model_list=[[]], effort=effort, parse_mode="txt", elapsed=0.0)

    monkeypatch.setattr(pipeline, "analyze_pdf", analyze)
    result = (
        asyncio.run(aio_doc_analyze(source_pdf(), effort=effort)) if asynchronous else doc_analyze(source_pdf(), effort=effort)
    )
    middle, model = result
    assert middle.metadata.document == model.metadata.document
    assert model.metadata.document.page_count == 2
    assert model.metadata.document.title == "Original title"


def test_doclib_source_columns_and_fts_are_independent_of_tier(tmp_path: Path) -> None:
    """真实 SQLite 验证属性补齐不降级高档正文索引，空字段不清除旧信息。"""

    async def scenario() -> None:
        """创建入库记录，使用新源属性刷新现有列和已有索引。"""
        db = DatabaseManager(str(tmp_path / "db.sqlite"))
        await db.initialize()
        fts = FTSManager(db)
        service = ParseService(db, fts, ConfigService(db), str(tmp_path), parse_lock_timeout_sec=30)
        sha = "a" * 64
        await ensure_doc_record(
            db,
            sha256=sha,
            size_bytes=10,
            file_type="pdf",
            page_count=1,
            title="Old",
            author="Old author",
            subject="Keep subject",
            keywords="Keep keywords",
            language="en",
            error_code=None,
            error_msg=None,
            first_seen_at=1,
            updated_at=1,
        )
        await fts.replace(sha256=sha, tier="advanced", text="Preserved body", title="Old", author="Old author")
        metadata = DocumentMetadata(
            file_suffix="pdf",
            producer=Producer(name="mineru", version="test"),
            document=DocumentProperties(
                title="New title", authors=["Alice", "Bob"], languages=["zh-CN"], page_count=9, page_count_kind="physical"
            ),
        )
        result = ParseResult(MiddleJson(pages=[PageInfo(page_idx=0)], is_full_document=False, metadata=metadata))
        await service._update_source_metadata(sha, result)
        row = await db.fetchone("SELECT * FROM docs WHERE sha256=?", (sha,))
        assert (row["title"], row["author"], row["language"], row["page_count"]) == ("New title", "Alice; Bob", "zh-CN", 9)
        assert row["subject"] == "Keep subject" and row["keywords"] == "Keep keywords"
        index = await db.fetchone("SELECT * FROM fts_contents WHERE sha256=?", (sha,))
        assert index["tier"] == "advanced" and index["title"] == "New title" and index["author"] == "Alice; Bob"
        assert "Preserved" in index["text"]
        result.middle_json.metadata.document = None
        await service._update_source_metadata(sha, result)
        assert (await db.fetchone("SELECT title FROM docs WHERE sha256=?", (sha,)))["title"] == "New title"

    asyncio.run(scenario())


def test_doclib_ingest_gets_html_metadata_without_parsing(tmp_path: Path) -> None:
    """真实入库流程无需正文任务完成，即可保存 HTML 标题作者和语言。"""

    async def scenario() -> None:
        """使用临时数据库与源 HTML 走入库服务。"""
        db = DatabaseManager(str(tmp_path / "db.sqlite"))
        await db.initialize()
        service = ParseService(db, FTSManager(db), ConfigService(db), str(tmp_path), parse_lock_timeout_sec=30)
        source = tmp_path / "source.html"
        source.write_text(
            '<html lang="zh-CN"><head><title>入库标题</title><meta name="author" content="作者"></head><body>正文</body></html>'
        )
        refresh = await service.refresh_file(str(source), ensure_ingested=True)
        assert refresh.file is not None
        row = await db.fetchone("SELECT * FROM docs WHERE sha256=?", (refresh.file.sha256,))
        assert (row["title"], row["author"], row["language"]) == ("入库标题", "作者", "zh-CN")

    asyncio.run(scenario())
