from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from docvortex.analyzers.native.office.rtf.converter import extract_rtf_metadata
from docvortex.export.middle import export_middle_json

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.doclib.core import file_io
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.parse_svc import ParseService
from mineru.errors import InvalidRequestError
from mineru.parser import parse
from mineru.render import RenderMode, render_docx, render_html, render_markdown, render_structured_content

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


_SEMANTIC_RTF = _PROJECT_ROOT / "tests" / "fixtures" / "rtf" / "semantic.rtf"


def _complex_rtf() -> bytes:
    """读取覆盖标题、列表、表格、公式、注释、链接和图片的确定性 RTF。"""
    return _SEMANTIC_RTF.read_bytes()


def test_rtf_doc_analyze_and_renderers_share_strict_metadata() -> None:
    """验证 RTF 同步/异步入口和四类 renderer 使用同一严格 Middle JSON。"""
    middle, model = doc_analyze(_complex_rtf(), effort="xhigh", parse_mode="ocr", file_suffix="rtf")
    async_middle, async_model = asyncio.run(
        aio_doc_analyze(_complex_rtf(), effort="medium", parse_mode="auto", file_suffix="rtf")
    )

    assert middle.file_suffix == model.file_suffix == "rtf"
    assert middle.extensions["mineru"]["effort"] == model.extensions["mineru"]["effort"] == "flash"
    assert middle.extensions["mineru"]["parse_mode"] == model.extensions["mineru"]["parse_mode"] == "txt"
    assert middle.is_full_document is model.is_full_document is True
    assert len(middle.pages) == len(model.pages) == 1
    assert async_middle == middle
    assert async_model == model

    markdown = render_markdown(middle, mode=RenderMode.FULL)
    html = render_html(middle, mode=RenderMode.FULL, standalone=False)
    docx = render_docx(middle)
    structured = render_structured_content(middle)
    assert "Inherited heading" in markdown
    assert "Foot body" in markdown
    assert "<table" in html
    assert docx.startswith(b"PK\x03\x04")
    assert structured["file_suffix"] == "rtf"


def test_public_parser_detects_rtf_content_before_extension(tmp_path: Path) -> None:
    """验证路径 parser 用强签名路由，并把图片导出到 ParseResult cache。"""
    source = tmp_path / "disguised.csv"
    source.write_bytes(b"\xef\xbb\xbf \r\n" + _complex_rtf())

    result = parse(source, tier="flash")

    with pytest.raises(InvalidRequestError, match="only supported for PDF") as exc_info:
        parse(source, tier="flash", page_range="99")
    assert exc_info.value.code == "page_range_invalid"

    assert result.middle_json.file_suffix == "rtf"
    assert result.middle_json.is_full_document is True
    assert len(result.pages) == 1
    exported = export_middle_json(result.middle_json, tmp_path / "export")
    assert exported.image_paths
    assert all(path.exists() for path in exported.image_paths)
    assert "image_base64" not in exported.middle_json.to_json()


def test_rtf_metadata_is_reused_by_doclib(tmp_path: Path) -> None:
    """验证 info destination 元数据和固定单页计数进入 doclib。"""
    source = tmp_path / "sample.rtf"
    source.write_bytes(_complex_rtf())

    direct = extract_rtf_metadata(BytesIO(_complex_rtf()))
    metadata = asyncio.run(file_io.extract_metadata(str(source)))

    assert direct == {
        "title": "Café",
        "author": "MinerU",
        "subject": "RTF",
        "keywords": "alpha;beta",
    }
    assert metadata["page_count"] == 1
    assert metadata["title"] == "Café"


def test_doclib_ingests_rtf_as_one_page_flash_document(tmp_path: Path) -> None:
    """验证 doclib 为 RTF 建立 flash parse row、独立文件类型和固定页数。"""

    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, object]]:
            """让测试只观察默认 RTF 入库与 tier 归一行为。"""
            return []

    async def run() -> None:
        """执行真实 SQLite doclib 入库并检查稳定行数据。"""
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),  # type: ignore[arg-type]
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "sample.rtf"
        source.write_bytes(_complex_rtf())

        response = await service.request_parse(str(source), tier="flash")
        doc = await db.fetchone(
            "SELECT file_type, page_count, title FROM docs WHERE sha256=?",
            (response.sha256,),
        )
        parses = await db.fetchall(
            "SELECT tier, page_range, status, privacy FROM parses WHERE sha256=?",
            (response.sha256,),
        )

        assert response.status == "pending"
        assert response.tier == "flash"
        assert doc == {"file_type": "rtf", "page_count": 1, "title": "Café"}
        assert parses == [{"tier": "flash", "page_range": "1", "status": "pending", "privacy": "local"}]

    asyncio.run(run())


@pytest.mark.parametrize(
    ("request_kwargs", "expected_code", "expected_param"),
    [
        ({"tier": "standard"}, "tier_unsupported_for_file_type", "tier"),
        ({"tier": "flash", "remote": True}, "remote_unsupported_for_file_type", "remote"),
    ],
)
def test_doclib_rejects_rtf_quality_tier_and_remote(
    tmp_path: Path,
    request_kwargs: dict[str, object],
    expected_code: str,
    expected_param: str,
) -> None:
    """验证 RTF 继承非 PDF/image 的严格 tier 与 remote 边界。"""

    class _NoRulesConfig:
        async def match_rules(self, path: str, rule_type: str) -> list[dict[str, object]]:
            """让测试只观察主动单文件请求的参数校验。"""
            return []

    async def run() -> None:
        """创建隔离 doclib 并断言稳定错误契约。"""
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        service = ParseService(
            db=db,
            fts=FTSManager(db),
            config_svc=_NoRulesConfig(),  # type: ignore[arg-type]
            data_dir=str(tmp_path / "data"),
            parse_lock_timeout_sec=1800,
        )
        source = tmp_path / "sample.rtf"
        source.write_bytes(_complex_rtf())

        with pytest.raises(InvalidRequestError) as exc_info:
            await service.request_parse(str(source), **request_kwargs)  # type: ignore[arg-type]

        assert exc_info.value.code == expected_code
        assert exc_info.value.param == expected_param

    asyncio.run(run())
