from __future__ import annotations

import asyncio
from io import BytesIO
import subprocess
import sys
from pathlib import Path

import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.doclib.core import file_io
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.parse_svc import ParseService
from mineru.errors import InvalidRequestError
from mineru.model.flash import RtfModel
from mineru.model.flash.office.legacy import (
    LegacyOfficeMalformedError,
    LegacyOfficeResourceLimitError,
)
from mineru.model.flash.office.rtf import lexer as lexer_module
from mineru.model.flash.office.rtf import parser as parser_module
from mineru.model.flash.office.rtf.converter import extract_rtf_metadata
from mineru.model.flash.office.rtf.lexer import RtfBinary, RtfLexer
from mineru.parser import parse
from mineru.render import RenderMode, render_docx, render_html, render_markdown, render_structured_content
from mineru.types import BlockType


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REAL_RTF = _PROJECT_ROOT / "demo" / "office_docs" / "rtf_01.rtf"
_SEMANTIC_RTF = _PROJECT_ROOT / "tests" / "fixtures" / "rtf" / "semantic.rtf"
_PNG_HEX = (
    b"89504e470d0a1a0a0000000d494844520000000200000002080600000072b60d24"
    b"0000001549444154789c6394acb8f39f81818181094480300024350270cd4262d30000000049454e44ae426082"
)


def _complex_rtf() -> bytes:
    """读取覆盖标题、列表、表格、公式、注释、链接和图片的确定性 RTF。"""
    return _SEMANTIC_RTF.read_bytes()


def test_rtf_lexer_bin_payload_is_position_explicit() -> None:
    """验证 bin 中的花括号和反斜杠不改变 lexer group 结构。"""
    tokens = list(RtfLexer(br"{\rtf1 before\bin5 }}{\x after}"))
    binaries = [token for token in tokens if isinstance(token, RtfBinary)]

    assert [token.data for token in binaries] == [b"}}{\\x"]


def test_rtf_lexer_rejects_truncated_bin_payload() -> None:
    """验证截断 bin 在游标失去可信边界前立即失败。"""
    with pytest.raises(LegacyOfficeMalformedError, match="truncated"):
        list(RtfLexer(br"{\rtf1\bin9 ab}"))


def test_rtf_font_codepages_and_unicode_surrogates_are_exact() -> None:
    """验证 per-font CP1251/CP932、多字节 hex 和 UTF-16 代理对恢复。"""
    russian = b"".join(f"\\'{value:02x}".encode("ascii") for value in "Привет".encode("cp1251"))
    japanese = b"".join(f"\\'{value:02x}".encode("ascii") for value in "こんにちは".encode("cp932"))
    source = b"".join(
        [
            br"{\rtf1\ansi{\fonttbl{\f0\fcharset204 Arial;}{\f1\fcharset128 MS Gothic;}}",
            br"\pard\f0 ",
            russian,
            br" \f1 ",
            japanese,
            br" \u-10179?\u-8704?\par}",
        ]
    )

    pages = RtfModel().predict(BytesIO(source))

    assert pages == [[{"type": BlockType.TEXT, "content": "Привет こんにちは 😀"}]]


def test_rtf_model_recovers_unicode_styles_and_structures() -> None:
    """验证完整 typed RTF 链路生成稳定单页 raw blocks。"""
    stream = BytesIO(_complex_rtf())

    pages = RtfModel().predict(stream)

    assert not stream.closed
    assert len(pages) == 1
    blocks = pages[0]
    assert [block["type"] for block in blocks].count(BlockType.TABLE) == 1
    assert [block["type"] for block in blocks].count(BlockType.IMAGE) == 1
    assert [block["type"] for block in blocks].count(BlockType.CODE) == 1
    title = blocks[0]
    assert title["type"] == BlockType.PARAGRAPH_TITLE
    assert title["level"] == 2
    assert title["anchor"] == "heading"
    assert "bold,italic" in title["content"]
    body = next(block for block in blocks if block.get("type") == BlockType.TEXT and "Unicode" in block.get("content", ""))
    assert "中文" in body["content"]
    assert "superscript" in body["content"]
    link = next(block for block in blocks if block.get("type") == BlockType.TEXT and "Jump heading" in block.get("content", ""))
    assert "<url>#heading</url>" in link["content"]
    assert "javascript:" not in link["content"]
    assert blocks[-3:] == [
        {"type": BlockType.HEADER, "content": "Header text"},
        {"type": BlockType.FOOTER, "content": "Footer text"},
        {"type": BlockType.PAGE_FOOTNOTE, "content": "[1] Foot body."},
    ]
    code = next(block for block in blocks if block.get("type") == BlockType.CODE)
    assert code["content"] == "first()\nsecond()"
    table = next(block for block in blocks if block.get("type") == BlockType.TABLE)
    assert 'colspan="2"' in table["content"]
    assert "<th" in table["content"]


def test_rtf_model_parses_real_libreoffice_fixture() -> None:
    """验证真实 LibreOffice RTF 在纯 Python 路径中保留全部可见段落。"""
    with _REAL_RTF.open("rb") as stream:
        pages = RtfModel().predict(stream)

    assert len(pages) == 1
    assert len(pages[0]) == 9
    content = "\n".join(str(block.get("content", "")) for block in pages[0])
    assert "KVCache-centric Scheduling Algorithm" in content
    assert "Prefill Global Scheduling" in content
    assert "Conductor estimates" in content


def test_rtf_page_controls_remain_inside_one_semantic_page() -> None:
    """验证 page/column 只保留换行，sect 只结束段落。"""
    pages = RtfModel().predict(BytesIO(br"{\rtf1\ansi A\page B\column C\sect D\par}"))

    assert len(pages) == 1
    assert pages[0] == [
        {"type": BlockType.TEXT, "content": "A\nB\nC"},
        {"type": BlockType.TEXT, "content": "D"},
    ]


def test_rtf_nested_and_merged_tables_preserve_content() -> None:
    """验证 nested table、cellx 投影及纵向 continuation 不丢可见文本。"""
    source = br"""{\rtf1\ansi
\trowd\clvmgf\cellx1000\cellx2000 A\cell B\cell\row
\trowd\clvmrg\cellx1000\cellx2000 \cell C\cell\row
\trowd\cellx1000\cellx2000
\pard\intbl\itap2 N1\nestcell N2\nestcell{\*\nesttableprops\trowd\cellx500\cellx1000\nestrow}
\pard\intbl outer\cell\row}"""

    pages = RtfModel().predict(BytesIO(source))

    tables = [block for block in pages[0] if block["type"] == BlockType.TABLE]
    assert len(tables) == 1
    html = tables[0]["content"]
    assert 'rowspan="2"' in html
    assert html.count("<table>") == 2
    assert all(text in html for text in ("A", "B", "C", "N1", "N2", "outer"))


def test_rtf_math_picture_is_only_used_when_formula_conversion_fails() -> None:
    """验证 Office Math 成功时去重预览，失败时保留首个安全 fallback 图片。"""
    valid = b"".join(
        [
            br"{\rtf1{\mmath{\*\moMath{\mr x}}{\mmathPict{\pict\pngblip ",
            _PNG_HEX,
            br"}}}}",
        ]
    )
    invalid = b"".join(
        [
            br"{\rtf1{\mmath{\*\moMath{\munknown x}}{\mmathPict{\pict\pngblip ",
            _PNG_HEX,
            br"}}}}",
        ]
    )

    valid_blocks = RtfModel().predict(BytesIO(valid))[0]
    invalid_blocks = RtfModel().predict(BytesIO(invalid))[0]

    assert [block["type"] for block in valid_blocks] == [BlockType.EQUATION]
    assert [block["type"] for block in invalid_blocks] == [BlockType.IMAGE]


def test_rtf_doc_analyze_and_renderers_share_strict_metadata() -> None:
    """验证 RTF 同步/异步入口和四类 renderer 使用同一严格 Middle JSON。"""
    middle, model = doc_analyze(_complex_rtf(), effort="xhigh", parse_mode="ocr", file_suffix="rtf")
    async_middle, async_model = asyncio.run(
        aio_doc_analyze(_complex_rtf(), effort="medium", parse_mode="auto", file_suffix="rtf")
    )

    assert middle.file_suffix == model.file_suffix == "rtf"
    assert middle.effort == model.effort == "flash"
    assert middle.parse_mode == model.parse_mode == "txt"
    assert middle.is_full_document is model.is_full_document is True
    assert len(middle.pages) == len(model.pages) == 1
    assert async_middle == middle
    assert async_model == model

    markdown = render_markdown(middle, mode=RenderMode.FULL)
    html = render_html(middle, mode=RenderMode.FULL, standalone=False)
    docx = render_docx(middle, mode=RenderMode.FULL)
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
    exported = result.middle_json.export(tmp_path / "export")
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


def test_rtf_unknown_destination_and_unbalanced_tail_recover_visible_text() -> None:
    """验证未知 ignorable destination 不泄漏，缺尾括号仍保留已恢复正文。"""
    source = br"{\rtf1\ansi{\*\unknown hidden}\pard visible\par"

    pages = RtfModel().predict(BytesIO(source))

    assert pages == [[{"type": BlockType.TEXT, "content": "visible"}]]


def test_rtf_source_html_is_rendered_as_inert_text() -> None:
    """验证源文档中的活动 HTML 外观不会进入 HTML renderer DOM。"""
    middle, _ = doc_analyze(
        br"{\rtf1\ansi visible <script>alert(1)</script> and x < y\par}",
        file_suffix="rtf",
    )

    html = render_html(middle, standalone=False)
    markdown = render_markdown(middle)

    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "script" in markdown and "alert(1)" in markdown


def test_rtf_object_keeps_safe_result_and_suppresses_objdata() -> None:
    """验证对象载荷不执行不泄漏，但显式 result 文本仍可恢复。"""
    pages = RtfModel().predict(
        BytesIO(br"{\rtf1 before {\object{\*\objdata 41424344}{\result Visible object}} after\par}")
    )

    assert pages == [[{"type": BlockType.TEXT, "content": "before Visible object after"}]]


def test_rtf_malformed_vector_picture_is_locally_dropped() -> None:
    """验证伪装 WMF 的 bin 字节不会生成占位图或破坏周围正文。"""
    pages = RtfModel().predict(
        BytesIO(br"{\rtf1 before {\pict\wmetafile8\bin5 abcde} after\par}")
    )

    assert pages == [[
        {"type": BlockType.TEXT, "content": "before"},
        {"type": BlockType.TEXT, "content": "after"},
    ]]


def test_rtf_valid_empty_and_invalid_header_have_distinct_results() -> None:
    """合法空 RTF 返回空逻辑页，非 RTF 输入返回稳定 malformed 错误。"""
    assert RtfModel().predict(BytesIO(br"{\rtf1}")) == [[]]
    with pytest.raises(LegacyOfficeMalformedError, match="not an RTF"):
        RtfModel().predict(BytesIO(b"plain text"))


def test_rtf_resource_limits_are_fixed_and_non_configurable(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 token、group、input、asset、grid 和 nested-table 上限均硬失败。"""
    monkeypatch.setattr(lexer_module, "MAX_RECORDS", 3)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_tokens"):
        list(RtfLexer(br"{\rtf1 text}"))

    monkeypatch.setattr(lexer_module, "MAX_RECORDS", 16_000_000)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_group_depth"):
        list(RtfLexer(b"{" * 257 + b"}" * 257))

    monkeypatch.setattr(parser_module, "MAX_RTF_BYTES", 8)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_bytes"):
        RtfModel().predict(BytesIO(br"{\rtf1 too long}"))

    monkeypatch.setattr(parser_module, "MAX_RTF_BYTES", 128 * 1024 * 1024)
    monkeypatch.setattr(parser_module, "MAX_ASSET_TOTAL_BYTES", 1)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_asset_total_bytes"):
        RtfModel().predict(BytesIO(br"{\rtf1{\pict\pngblip 89504e47}}"))

    monkeypatch.setattr(parser_module, "MAX_ASSET_TOTAL_BYTES", 128 * 1024 * 1024)
    monkeypatch.setattr(parser_module, "MAX_GRID_SLOTS", 1)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_grid_slots"):
        RtfModel().predict(BytesIO(br"{\rtf1\trowd\cellx1\cellx2 A\cell B\cell\row}"))

    monkeypatch.setattr(parser_module, "MAX_GRID_SLOTS", 4_000_000)
    with pytest.raises(LegacyOfficeResourceLimitError, match="max_table_depth"):
        RtfModel().predict(BytesIO(br"{\rtf1\pard\intbl\itap5 nested\par}"))


def test_rtf_runtime_has_no_anydoc_dependency() -> None:
    """验证依赖清单、源码导入和惰性模型加载均不包含 anydoc。"""
    assert "firecrawl-anydoc" not in (_PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert all(
        "import anydoc" not in path.read_text(encoding="utf-8")
        for path in (_PROJECT_ROOT / "mineru").rglob("*.py")
    )
    script = "\n".join(
        [
            "import sys",
            "from mineru.model.flash import RtfModel",
            "assert 'mineru.model.flash.office.rtf.converter' not in sys.modules",
            "pages = RtfModel().predict(__import__('io').BytesIO(b'{\\\\rtf1 ok}'))",
            "assert pages == [[{'type': 'text', 'content': 'ok'}]]",
            "assert 'anydoc' not in sys.modules",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
