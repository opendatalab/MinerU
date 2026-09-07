from __future__ import annotations

import asyncio
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock

import pytest

from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from docvortex.postprocess.lists import fix_office_list_blocks
from docvortex.analyzers.native import DocModel
from docvortex.analyzers.native._shared.hyperlink import OFFICE_EXTERNAL_HYPERLINK_SCHEMES
from docvortex.analyzers.native._shared.hyperlink import sanitize_hyperlink_target
from docvortex.analyzers.native.office.doc.models import DocCharStyle
from docvortex.analyzers.native.office.doc.models import DocTableCell
from docvortex.analyzers.native.office.doc.images import ImageStore
from docvortex.analyzers.native.office.doc.parser import _RawTableRow
from docvortex.analyzers.native.office.doc.parser import _materialize_table_rows
from docvortex.analyzers.native.office.doc.records import DocBudget
from docvortex.analyzers.native.office.doc.sprm import apply_character_sprms
from docvortex.analyzers.native.office.errors import LegacyOfficeEncryptedError
from docvortex.analyzers.native.office.errors import LegacyOfficeMalformedError
from docvortex.analyzers.native.office.errors import LegacyOfficeMissingPartError
from docvortex.analyzers.native.office.errors import LegacyOfficeResourceLimitError
from docvortex.analyzers.native.office.legacy.officeart import OfficeImagePayload
from mineru.parser import parse
from mineru.types import BlockType, MiddleJson, ModelJson

from _legacy_doc_test_utils import build_doc, utf16_cp
from _legacy_ppt_test_utils import _build_cfb
from _span_test_utils import inline, inline_items, inline_text, inline_urls


def test_doc_image_store_distinguishes_render_size_without_double_accounting() -> None:
    """验证相同图片的不同 ptSize 独立缓存，但原始 bytes 只计费和解码一次。"""
    decoder = Mock()
    decoder.decode.return_value = None
    store = ImageStore(equation_decoder=decoder)
    data = b"same-standard-wmf"

    landscape = store.add(OfficeImagePayload(data, "wmf", "image/wmf", (914_400, 457_200)))
    portrait = store.add(OfficeImagePayload(data, "wmf", "image/wmf", (457_200, 914_400)))
    landscape_again = store.add(OfficeImagePayload(data, "wmf", "image/wmf", (914_400, 457_200)))

    assert landscape.render_size_emu == (914_400, 457_200)
    assert portrait.render_size_emu == (457_200, 914_400)
    assert portrait is not landscape
    assert landscape_again is landscape
    assert store.total == len(data)
    decoder.decode.assert_called_once()


def test_doc_model_preserves_empty_sections_and_ignores_page_breaks() -> None:
    """验证 section 是唯一分页来源，空 section 保留且分页符不额外切页。"""

    text = "\fFirst\rSecond\r"
    first_end = utf16_cp("\fFirst\r")
    stream = BytesIO(build_doc(text, section_ends=[0, first_end, utf16_cp(text)]))

    pages = DocModel().predict(stream)

    assert not stream.closed
    assert pages == [
        [],
        [{"type": BlockType.TEXT, "content": inline("First")}],
        [{"type": BlockType.TEXT, "content": inline("Second")}],
    ]


def test_doc_analyze_sync_and_async_return_strict_doc_contract() -> None:
    """验证同步和异步 Analyze 均保留 DOC 严格后缀和 Flash/TXT 元数据。"""

    file_bytes = build_doc("Hello\r")
    middle, model = doc_analyze(file_bytes, file_suffix="doc")
    async_middle, async_model = asyncio.run(aio_doc_analyze(file_bytes, file_suffix="doc"))

    assert isinstance(model, ModelJson)
    assert isinstance(middle, MiddleJson)
    assert model.file_suffix == middle.file_suffix == "doc"
    assert model.extensions["mineru"]["effort"] == middle.extensions["mineru"]["effort"] == "flash"
    assert model.extensions["mineru"]["parse_mode"] == middle.extensions["mineru"]["parse_mode"] == "txt"
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

    assert pages == [[{"type": BlockType.TEXT, "content": inline(text.rstrip("\r"))}]]


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

    assert any(
        isinstance(span, dict) and span.get("content") == "[1]" and "superscript" in span.get("styles", [])
        for span in inline_items(pages[0][0]["content"])
    )
    assert pages[0][-1] == {"type": BlockType.PAGE_FOOTNOTE, "content": inline("[1] Foot note")}
    assert all(block.get("type") != BlockType.PAGE_FOOTNOTE for block in pages[1])


def test_doc_hyperlink_field_keeps_safe_target_and_drops_dangerous_target() -> None:
    """验证字段缓存结果保留，危险 URL 只降级为普通文本。"""

    text = '\x13 HYPERLINK "https://example.test/a" \x14Safe\x15\r\x13 HYPERLINK "javascript:alert(1)" \x14Danger\x15\r'
    pages = DocModel().predict(BytesIO(build_doc(text)))

    assert inline_urls(pages[0][0]["content"]) == ["https://example.test/a"]
    assert pages[0][1] == {"type": BlockType.TEXT, "content": inline("Danger")}


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

    assert (
        sanitize_hyperlink_target(
            target,
            allowed_schemes=OFFICE_EXTERNAL_HYPERLINK_SCHEMES,
            allow_relative=True,
            allow_fragment=True,
        )
        == expected
    )


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
                {"type": BlockType.TEXT, "content": inline("item"), "list_label": "IV."},
            ],
        }
    ]

    assert inline_text(fix_office_list_blocks(blocks)[0]["content"][0]["content"]) == "IV. item"


def test_doc_table_grid_materializes_colspan_and_rowspan() -> None:
    """验证 Word table edge 网格能同时恢复横向和纵向合并。"""

    from docvortex.analyzers.native.office.doc.models import DocTableCellFormat
    from docvortex.analyzers.native.office.doc.models import DocTableFormat

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

    import docvortex.analyzers.native.office.doc.records as records

    monkeypatch.setattr(records, "MAX_RECORDS", 1)
    budget = records.DocBudget()
    budget.charge()
    with pytest.raises(LegacyOfficeResourceLimitError):
        budget.charge()


def test_doc_is_supported_by_public_parser(tmp_path: Path) -> None:
    """验证公共 parser 通过统一 MinerUParser 路由 DOC。"""

    path = tmp_path / "sample.doc"
    path.write_bytes(build_doc("Hello\r"))

    result = parse(path, tier="flash")

    assert result.middle_json.file_suffix == "doc"
    assert inline_text(result.pages[0].blocks[0].content) == "Hello"
