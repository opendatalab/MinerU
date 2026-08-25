from __future__ import annotations

import asyncio
from collections import Counter
from io import BytesIO
from pathlib import Path

from bs4 import BeautifulSoup
import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.backend.postprocess.lists import fix_office_list_blocks
from mineru.model.flash import DocModel
from mineru.model.flash.office.doc.fields import sanitize_hyperlink_target
from mineru.model.flash.office.doc.models import DocCharStyle, DocTableCell
from mineru.model.flash.office.doc.parser import _RawTableRow, _materialize_table_rows
from mineru.model.flash.office.doc.records import DocBudget
from mineru.model.flash.office.doc.sprm import apply_character_sprms
from mineru.model.flash.office.legacy import (
    LegacyOfficeEncryptedError,
    LegacyOfficeMalformedError,
    LegacyOfficeMissingPartError,
    LegacyOfficeResourceLimitError,
)
from mineru.parser import parse
from mineru.types import BlockType, ChartBlock, MiddleJson, ModelJson, TableBlock

from _legacy_doc_test_utils import build_doc, utf16_cp
from _legacy_ppt_test_utils import _build_cfb


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_REAL_DOC = _PROJECT_ROOT / "demo" / "office_docs" / "docx_01.doc"


def test_doc_model_preserves_empty_sections_and_ignores_page_breaks() -> None:
    """验证 section 是唯一分页来源，空 section 保留且分页符不额外切页。"""

    text = "\fFirst\rSecond\r"
    first_end = utf16_cp("\fFirst\r")
    stream = BytesIO(build_doc(text, section_ends=[0, first_end, utf16_cp(text)]))

    pages = DocModel().predict(stream)

    assert not stream.closed
    assert pages == [[], [{"type": BlockType.TEXT, "content": "First"}], [{"type": BlockType.TEXT, "content": "Second"}]]


def test_doc_analyze_sync_and_async_return_strict_doc_contract() -> None:
    """验证同步和异步 Analyze 均保留 DOC 严格后缀和 Flash/TXT 元数据。"""

    file_bytes = build_doc("Hello\r")
    middle, model = doc_analyze(file_bytes, file_suffix="doc")
    async_middle, async_model = asyncio.run(aio_doc_analyze(file_bytes, file_suffix="doc"))

    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert model.file_suffix == middle.file_suffix == "doc"
    assert model.effort == middle.effort == "flash"
    assert model.parse_mode == middle.parse_mode == "txt"
    assert async_model == model
    assert async_middle == middle


@pytest.mark.parametrize(
    ("text", "compressed", "kwargs"),
    [
        ("A😀B\r", False, {}),
        ("café\r", True, {}),
        ("Привет, мир!\r", True, {"codec": "cp1251", "lid": 0x0419}),
        ("こんにちは世界。\r", True, {"codec": "cp932", "lid": 0x0411, "flags_extra": 0x4000}),
    ],
)
def test_doc_piece_text_preserves_non_bmp_and_compressed_codepage(
    text: str,
    compressed: bool,
    kwargs: dict[str, object],
) -> None:
    """验证 UTF-16 非 BMP CP 和压缩 ANSI piece 均恢复原文本。"""

    pages = DocModel().predict(BytesIO(build_doc(text, compressed=compressed, **kwargs)))

    assert pages == [[{"type": BlockType.TEXT, "content": text.rstrip("\r")}]]


def test_doc_footnote_reference_and_body_bind_to_reference_section() -> None:
    """验证脚注引用使用上标样式，脚注正文追加到引用所在 section。"""

    pages = DocModel().predict(
        BytesIO(
            build_doc(
                "Body\x02\rTail\r",
                section_ends=[6, 11],
                footnote_text="Foot note\r",
            )
        )
    )

    assert '<text style="superscript">[1]</text>' in pages[0][0]["content"]
    assert pages[0][-1] == {"type": BlockType.PAGE_FOOTNOTE, "content": "[1] Foot note"}
    assert all(block.get("type") != BlockType.PAGE_FOOTNOTE for block in pages[1])


def test_doc_hyperlink_field_keeps_safe_target_and_drops_dangerous_target() -> None:
    """验证字段缓存结果保留，危险 URL 只降级为普通文本。"""

    text = '\x13 HYPERLINK "https://example.test/a" \x14Safe\x15\r\x13 HYPERLINK "javascript:alert(1)" \x14Danger\x15\r'
    pages = DocModel().predict(BytesIO(build_doc(text)))

    assert "<url>https://example.test/a</url>" in pages[0][0]["content"]
    assert pages[0][1] == {"type": BlockType.TEXT, "content": "Danger"}


@pytest.mark.parametrize(
    ("target", "expected"),
    [
        ("https://example.test", "https://example.test"),
        ("#bookmark", "#bookmark"),
        ("docs/readme.doc", "docs/readme.doc"),
        ("file:///tmp/a.doc", None),
        (r"C:\\docs\\a.doc", None),
        (r"\\server\\share\\a.doc", None),
        ("data:text/plain,x", None),
    ],
)
def test_doc_hyperlink_security_policy(target: str, expected: str | None) -> None:
    """验证 DOC 链接白名单拒绝本地和可执行目标。"""

    assert sanitize_hyperlink_target(target) == expected


def test_doc_character_sprms_preserve_visible_styles_and_hide_revisions() -> None:
    """验证常见 CHPX 样式、上下标、隐藏和删除修订状态。"""

    grpprl = (
        b"\x35\x08\x01"  # bold
        b"\x36\x08\x01"  # italic
        b"\x3e\x2a\x01"  # underline
        b"\x37\x08\x01"  # strike
        b"\x48\x2a\x01"  # superscript
        b"\x3c\x08\x01"  # hidden
        b"\x00\x08\x01"  # deleted revision
    )

    style = apply_character_sprms(grpprl, DocCharStyle(), DocCharStyle())

    assert style.bold and style.italic and style.underline and style.strike
    assert style.superscript and not style.subscript
    assert style.hidden and style.deleted


def test_doc_exact_list_label_is_consumed_before_strict_projection() -> None:
    """验证 Roman/复合列表标签优先于通用十进制编号且私有字段被删除。"""

    blocks = [
        {
            "type": BlockType.LIST,
            "attribute": "ordered",
            "start": 4,
            "ilevel": 0,
            "content": [
                {"type": BlockType.TEXT, "content": "item", "list_label": "IV."},
            ],
        }
    ]

    assert fix_office_list_blocks(blocks)[0]["content"] == [{"type": BlockType.TEXT, "content": "IV. item"}]


def test_doc_table_grid_materializes_colspan_and_rowspan() -> None:
    """验证 Word table edge 网格能同时恢复横向和纵向合并。"""

    from mineru.model.flash.office.doc.models import DocTableCellFormat, DocTableFormat

    first = DocTableCell(blocks=[])
    raw_rows = [
        _RawTableRow(
            [first, DocTableCell()],
            DocTableFormat(
                boundaries=(0, 100, 200),
                cells=(
                    DocTableCellFormat(100, horizontal_first=True, vertical_first=True),
                    DocTableCellFormat(200, horizontal_continue=True),
                ),
            ),
        ),
        _RawTableRow(
            [DocTableCell(), DocTableCell()],
            DocTableFormat(
                boundaries=(0, 100, 200),
                cells=(
                    DocTableCellFormat(100, vertical_continue=True),
                    DocTableCellFormat(200),
                ),
            ),
        ),
    ]

    rows = _materialize_table_rows(raw_rows, DocBudget())

    assert rows[0].cells[0].col_span == 2
    assert rows[0].cells[0].row_span == 2


def test_doc_rejects_word95_encryption_rtf_and_missing_word_stream() -> None:
    """验证不支持版本、加密、RTF 冒充和缺失核心 stream 使用稳定错误。"""

    with pytest.raises(LegacyOfficeMalformedError, match="Word 95"):
        DocModel().predict(BytesIO(build_doc("old\r", n_fib=0x0065)))
    with pytest.raises(LegacyOfficeEncryptedError):
        DocModel().predict(BytesIO(build_doc("secret\r", flags_extra=0x0100)))
    with pytest.raises(LegacyOfficeMalformedError):
        DocModel().predict(BytesIO(b"{\\rtf1 renamed}"))
    with pytest.raises(LegacyOfficeMissingPartError):
        DocModel().predict(BytesIO(_build_cfb([("1Table", b"table")])))


def test_doc_budget_uses_stable_resource_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 DOC 记录预算超过固定上限时使用共享错误类型。"""

    import mineru.model.flash.office.doc.records as records

    monkeypatch.setattr(records, "MAX_RECORDS", 1)
    budget = records.DocBudget()
    budget.charge()
    with pytest.raises(LegacyOfficeResourceLimitError):
        budget.charge()


@pytest.mark.skipif(not _REAL_DOC.exists(), reason="real Office roundtrip fixture is local-only")
def test_real_doc_recovers_sections_structure_and_sidecars(tmp_path: Path) -> None:
    """验证真实 DOC 的 section、目录、表格、图片和严格 export 闭包。"""

    middle, model = doc_analyze(_REAL_DOC.read_bytes(), file_suffix="doc")
    counts = Counter(block.get("type") for page in model.pages for block in page)

    assert len(model.pages) == len(middle.pages) == 3
    assert counts[BlockType.DOC_TITLE] == 1
    assert counts[BlockType.PARAGRAPH_TITLE] == 37
    assert counts[BlockType.INDEX] == 1
    assert counts[BlockType.LIST] == 5
    assert counts[BlockType.TABLE] == 8
    assert counts[BlockType.HEADER] == 4
    assert counts[BlockType.FOOTER] == 1
    assert counts[BlockType.IMAGE] >= 44
    assert counts[BlockType.CHART] == 1

    table_blocks = [block for page in middle.pages for block in page.blocks if isinstance(block, TableBlock)]
    assert len(table_blocks) == 8
    soups = [BeautifulSoup(block.content[0].content, "html.parser") for block in table_blocks]
    assert sum(max(len(soup.find_all("table")) - 1, 0) for soup in soups) == 3
    assert sum(len(soup.find_all("img")) for soup in soups) == 5
    assert any(len(soup.find_all("tr")) == 39 for soup in soups)
    assert any(
        len([cell for cell in soup.find_all(["td", "th"]) if cell.has_attr("rowspan") or cell.has_attr("colspan")]) == 141
        for soup in soups
    )
    chart = next(block for page in middle.pages for block in page.blocks if isinstance(block, ChartBlock))
    chart_soup = BeautifulSoup(chart.content[0].content, "html.parser")
    assert [
        [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"], recursive=False)]
        for row in chart_soup.find_all("tr")
    ] == [
        ["列1", "系列 1", "系列 2", "系列 3"],
        ["类别 1", "4.3", "2.4", "2"],
        ["类别 2", "2.5", "4.4", "2"],
        ["类别 3", "3.5", "1.8", "3"],
        ["类别 4", "4.5", "2.8", "5"],
    ]
    assert chart.content[0].image_base64 is not None

    export = middle.export(tmp_path / "export")
    payload = export.middle_json.model_dump_json(exclude_none=True)
    assert export.json_path.exists()
    assert len(export.image_paths) >= 49
    assert all(path.exists() and path.stat().st_size > 0 for path in export.image_paths)
    assert "image_base64" not in payload
    assert "data:image/" not in payload


def test_doc_is_supported_by_public_parser(tmp_path: Path) -> None:
    """验证公共 parser 通过统一 MinerUParser 路由 DOC。"""

    path = tmp_path / "sample.doc"
    path.write_bytes(build_doc("Hello\r"))

    result = parse(path, tier="flash")

    assert result.middle_json.file_suffix == "doc"
    assert result.pages[0].blocks[0].content == "Hello"
