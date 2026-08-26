from __future__ import annotations

import asyncio
import importlib.util
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from lxml import etree

import mineru.model.flash.office.odf.table as odf_table_module
from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.doclib.core.file_io import extract_metadata
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.parse_svc import ParseService
from mineru.errors import InvalidRequestError
from mineru.model.flash import OdpModel, OdsModel, OdtModel
from mineru.model.flash.office.odf.errors import OdfEncryptedError, OdfParseError, OdfResourceLimitError
from mineru.model.flash.office.odf.package import OdfPackage
from mineru.parser import parse, parse_async
from mineru.parser import api_server
from mineru.parser.api_server import CreateJobRequest, FileStore
from mineru.parser.file_type import guess_suffix_by_bytes, guess_suffix_by_path
from mineru.render import render_docx, render_html, render_markdown, render_structured_content
from mineru.types import BlockType

from _odf_test_utils import _PIXEL_PNG, build_odf_package, build_odp_fixture, build_ods_fixture, build_odt_fixture


@pytest.mark.parametrize(
    ("suffix", "model_class", "payload", "page_count"),
    [
        ("odt", OdtModel, build_odt_fixture(), 3),
        ("ods", OdsModel, build_ods_fixture(), 2),
        ("odp", OdpModel, build_odp_fixture(), 3),
    ],
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
    assert model.file_suffix == middle.file_suffix == suffix
    assert model.effort == middle.effort == "flash"
    assert model.parse_mode == middle.parse_mode == "txt"


def test_odt_recovers_structure_and_all_renderers() -> None:
    """验证 ODT 标题、富文本、列表、合并表、分页、脚注、公式和图片。"""
    middle, model = doc_analyze(build_odt_fixture(), file_suffix="odt")
    raw_blocks = [block for page in model.pages for block in page]
    raw_types = [block["type"] for block in raw_blocks]
    assert len(model.pages) == 3
    assert BlockType.DOC_TITLE in raw_types
    assert BlockType.PARAGRAPH_TITLE in raw_types
    assert BlockType.LIST in raw_types
    assert BlockType.TABLE in raw_types
    assert BlockType.EQUATION in raw_types
    assert BlockType.IMAGE in raw_types
    assert BlockType.PAGE_FOOTNOTE in raw_types
    assert BlockType.HEADER in raw_types
    assert BlockType.FOOTER in raw_types
    assert any('style="bold"' in str(block.get("content")) for block in raw_blocks)
    assert any(
        "rowspan" not in str(block.get("content")) and 'colspan="2"' in str(block.get("content")) for block in raw_blocks
    )
    assert any(block.get("content") == r"\frac{x}{2}" for block in raw_blocks)
    assert any(block.get("type") == BlockType.PAGE_FOOTNOTE and "Note body" in block.get("content", "") for block in raw_blocks)

    markdown = render_markdown(middle)
    html_output = render_html(middle)
    structured = render_structured_content(middle)
    assert "ODT Title" in markdown
    assert "<script>alert(1)</script>" not in markdown
    assert "<script>alert(1)</script>" not in html_output
    assert "<table" in html_output
    assert structured["pages"][0]["blocks"][0]["type"] == "doc_title"


def test_odt_promotes_numbered_heading_inside_list() -> None:
    """验证 LibreOffice 编码在 list-item 中的 text:h 恢复为编号章节标题。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0">
 <office:automatic-styles><text:list-style style:name="L1">
  <text:list-level-style-number text:level="1" style:num-format="1"/>
 </text:list-style></office:automatic-styles>
 <office:body><office:text><text:list text:style-name="L1"><text:list-item>
  <text:h text:outline-level="1">Chapter</text:h>
 </text:list-item></text:list></office:text></office:body>
</office:document-content>"""
    middle, _ = doc_analyze(build_odf_package("odt", content), file_suffix="odt")
    assert middle.pages[0].blocks[0].type == BlockType.PARAGRAPH_TITLE
    assert middle.pages[0].blocks[0].content == "1 Chapter"  # type: ignore[union-attr]


def test_odt_list_lifts_visual_blocks_outside_strict_list() -> None:
    """验证列表段落图片保留原类型并提升为 LIST 的有序兄弟块。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink">
 <office:automatic-styles><text:list-style style:name="L1">
  <text:list-level-style-number text:level="1" style:num-format="1"/>
 </text:list-style></office:automatic-styles>
 <office:body><office:text><text:list text:style-name="L1"><text:list-item text:start-value="3">
  <text:p>Illustrated item<draw:frame><draw:image xlink:href="Pictures/pixel.png"/></draw:frame></text:p>
 </text:list-item><text:list-item><text:p>Next item</text:p></text:list-item></text:list></office:text></office:body>
</office:document-content>"""

    middle, _ = doc_analyze(
        build_odf_package("odt", content, extra_parts={"Pictures/pixel.png": _PIXEL_PNG}),
        file_suffix="odt",
    )

    assert [block.type for block in middle.pages[0].blocks] == [BlockType.LIST, BlockType.IMAGE, BlockType.LIST]
    assert [child.content for child in middle.pages[0].blocks[0].content] == ["3. Illustrated item"]  # type: ignore[union-attr]
    assert [child.content for child in middle.pages[0].blocks[2].content] == ["4. Next item"]  # type: ignore[union-attr]


def test_odt_list_item_joins_multiple_paragraphs_before_markers() -> None:
    """验证一个源 list-item 的多个段落只生成一个 LIST 文本叶子和一个 marker。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
 <office:body><office:text><text:list>
  <text:list-item><text:p>First paragraph</text:p><text:p>Second paragraph</text:p></text:list-item>
  <text:list-item><text:p>Next item</text:p></text:list-item>
 </text:list></office:text></office:body>
</office:document-content>"""

    middle, _ = doc_analyze(build_odf_package("odt", content), file_suffix="odt")
    list_block = middle.pages[0].blocks[0]

    assert list_block.type == BlockType.LIST
    assert [child.content for child in list_block.content] == ["- First paragraph\nSecond paragraph", "- Next item"]  # type: ignore[union-attr]
    assert render_markdown(middle).count("- ") == 2


def test_odt_table_cell_renders_inline_image_once() -> None:
    """验证 ODT/ODP 单元格内联图片不会再被对应段外 image block 重复输出。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink">
 <office:body><office:text><table:table><table:table-row><table:table-cell>
  <text:p>Cell<draw:frame><draw:image xlink:href="Pictures/pixel.png"/></draw:frame></text:p>
 </table:table-cell></table:table-row></table:table></office:text></office:body>
</office:document-content>"""

    pages = OdtModel().predict(BytesIO(build_odf_package("odt", content, extra_parts={"Pictures/pixel.png": _PIXEL_PNG})))

    assert pages[0][0]["type"] == BlockType.TABLE
    assert pages[0][0]["content"].count("<img") == 1


def test_odt_inline_visual_stays_before_soft_page_break() -> None:
    """验证 soft-page-break 前遇到的段内视觉块仍留在来源逻辑页。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink">
 <office:body><office:text><text:p>
  Before<draw:frame><draw:image xlink:href="Pictures/pixel.png"/></draw:frame><text:soft-page-break/>After
 </text:p></office:text></office:body>
</office:document-content>"""

    pages = OdtModel().predict(BytesIO(build_odf_package("odt", content, extra_parts={"Pictures/pixel.png": _PIXEL_PNG})))

    assert [[block["type"] for block in page] for page in pages] == [
        [BlockType.TEXT, BlockType.IMAGE],
        [BlockType.TEXT],
    ]


def test_odt_child_style_can_reset_inherited_page_breaks() -> None:
    """验证 break-before/after=auto 显式关闭父样式的逻辑分页。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0"
 xmlns:fo="urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0">
 <office:automatic-styles>
  <style:style style:name="Parent" style:family="paragraph">
   <style:paragraph-properties fo:break-before="page" fo:break-after="page"/>
  </style:style>
  <style:style style:name="Inherited" style:family="paragraph" style:parent-style-name="Parent"/>
  <style:style style:name="Reset" style:family="paragraph" style:parent-style-name="Parent">
   <style:paragraph-properties fo:break-before="auto" fo:break-after="auto"/>
  </style:style>
 </office:automatic-styles>
 <office:body><office:text>
  <text:p>Before</text:p><text:p text:style-name="Reset">Reset</text:p><text:p>After</text:p>
 </office:text></office:body>
</office:document-content>"""
    inherited_content = content.replace('text:style-name="Reset"', 'text:style-name="Inherited"')

    reset_pages = OdtModel().predict(BytesIO(build_odf_package("odt", content)))
    inherited_pages = OdtModel().predict(BytesIO(build_odf_package("odt", inherited_content)))

    assert [[block.get("content") for block in page] for page in reset_pages] == [["Before", "Reset", "After"]]
    assert [[block.get("content") for block in page] for page in inherited_pages] == [
        ["Before"],
        ["Reset"],
        ["After"],
    ]


def test_odf_covered_placeholder_reuses_colspan_coordinate() -> None:
    """验证 colspan 后的 covered placeholder 不会额外扩宽表格。"""
    table = etree.fromstring(
        """<table:table xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
 <table:table-row>
  <table:table-cell table:number-columns-spanned="2"><text:p>Merged</text:p></table:table-cell>
  <table:covered-table-cell/><table:table-cell><text:p>Tail</text:p></table:table-cell>
 </table:table-row>
 <table:table-row>
  <table:table-cell><text:p>A</text:p></table:table-cell>
  <table:table-cell><text:p>B</text:p></table:table-cell>
  <table:table-cell><text:p>C</text:p></table:table-cell>
 </table:table-row>
</table:table>""".encode()
    )

    grid = odf_table_module.parse_table_grid(table, lambda cell: "".join(cell.itertext()).strip())

    assert grid.width == 3
    assert grid.covered == {(0, 1)}
    assert grid.rows[0][0] is not None and grid.rows[0][0].col_span == 2
    assert grid.rows[0][2] is not None and grid.rows[0][2].html == "Tail"
    assert [cell.html if cell is not None else None for cell in grid.rows[1]] == ["A", "B", "C"]


def test_odt_note_after_soft_page_break_stays_on_new_page() -> None:
    """验证 soft-page-break 后的 note reference 和正文归属同一新页。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
 <office:body><office:text><text:p>Before<text:soft-page-break/>After
  <text:note><text:note-citation>1</text:note-citation><text:note-body><text:p>After note</text:p></text:note-body></text:note>
 </text:p></office:text></office:body>
</office:document-content>"""

    pages = OdtModel().predict(BytesIO(build_odf_package("odt", content)))

    assert pages == [
        [{"type": BlockType.TEXT, "content": "Before"}],
        [
            {"type": BlockType.TEXT, "content": "After [1]"},
            {"type": BlockType.PAGE_FOOTNOTE, "content": "[1] After note"},
        ],
    ]


def test_ods_cell_note_emits_page_footnote() -> None:
    """验证 ODS cell citation 对应的 note body 在当前 sheet 页末输出。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">
 <office:body><office:spreadsheet><table:table table:name="Sheet1"><table:table-row><table:table-cell><text:p>Cell
  <text:note><text:note-citation>1</text:note-citation><text:note-body><text:p>Cell note</text:p></text:note-body></text:note>
 </text:p></table:table-cell></table:table-row></table:table></office:spreadsheet></office:body>
</office:document-content>"""

    pages = OdsModel().predict(BytesIO(build_odf_package("ods", content)))

    assert [block["type"] for block in pages[0]] == [BlockType.TABLE, BlockType.PAGE_FOOTNOTE]
    assert "Cell [1]" in pages[0][0]["content"]
    assert pages[0][1]["content"] == "[1] Cell note"


def test_odp_slide_inline_note_emits_page_footnote() -> None:
    """验证 ODP slide 正文中的 note body 不会被 presentation notes 路径遗漏。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:presentation="urn:oasis:names:tc:opendocument:xmlns:presentation:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">
 <office:body><office:presentation><draw:page draw:name="Slide1"><draw:frame><draw:text-box><text:p>Slide
  <text:note><text:note-citation>1</text:note-citation><text:note-body><text:p>Inline note</text:p></text:note-body></text:note>
 </text:p></draw:text-box></draw:frame></draw:page></office:presentation></office:body>
</office:document-content>"""

    pages = OdpModel().predict(BytesIO(build_odf_package("odp", content)))

    assert pages == [
        [
            {"type": BlockType.TEXT, "content": "Slide [1]"},
            {"type": BlockType.PAGE_FOOTNOTE, "content": "[1] Inline note"},
        ]
    ]


def test_odp_preserves_empty_slide_chart_preview_and_notes() -> None:
    """验证 ODP 空 slide 不丢失，图表同时保留数据和预览，备注归属原页。"""
    middle, model = doc_analyze(build_odp_fixture(), file_suffix="odp")
    assert len(model.pages) == 3
    assert model.pages[1] == []
    assert model.pages[0][0]["type"] == BlockType.DOC_TITLE
    chart = next(block for block in model.pages[2] if block["type"] == BlockType.CHART)
    assert "Category" in chart["content"]
    assert "Value" in chart["content"]
    assert chart["image_base64"].startswith("data:image/")
    assert any(block["type"] == BlockType.PAGE_FOOTNOTE and "Speaker note" in block["content"] for block in model.pages[2])
    assert len(middle.pages) == 3


def test_ods_skips_hidden_sheet_and_emits_tables_images_and_charts() -> None:
    """验证 ODS 可见 sheet 边界、typed value、合并结构和图表对象。"""
    middle, model = doc_analyze(build_ods_fixture(), file_suffix="ods")
    assert len(model.pages) == 2
    assert [page[0]["content"] for page in model.pages] == ["Visible A", "Visible B"]
    flattened = [block for page in model.pages for block in page]
    assert "secret" not in str(flattened)
    assert "50%" in str(flattened)
    assert 'colspan="2"' in str(flattened)
    assert any(block["type"] == BlockType.CHART for block in flattened)
    assert len(middle.pages) == 2


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [("odt", build_odt_fixture()), ("ods", build_ods_fixture()), ("odp", build_odp_fixture())],
)
def test_odf_content_detection_precedes_csv_extension(
    tmp_path: Path,
    suffix: str,
    payload: bytes,
) -> None:
    """验证 ODF 强内容身份覆盖伪装扩展名和 CSV 无签名兜底。"""
    disguised = tmp_path / "disguised.csv"
    disguised.write_bytes(payload)
    assert guess_suffix_by_bytes(payload, str(disguised)) == suffix
    assert guess_suffix_by_path(disguised) == suffix


def test_rtf_signature_still_precedes_odf_extension(tmp_path: Path) -> None:
    """验证新增 ZIP 探测不改变 RTF 强签名的最高优先级。"""
    source = tmp_path / "disguised.odt"
    source.write_bytes(rb"{\rtf1\ansi visible}")
    assert guess_suffix_by_path(source) == "rtf"
    assert guess_suffix_by_bytes(source.read_bytes(), str(source)) == "rtf"


def test_plain_text_renamed_to_odf_is_not_accepted(tmp_path: Path) -> None:
    """验证 ODF 扩展名本身不能把普通文本升级为结构化文档。"""
    source = tmp_path / "fake.odt"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    assert guess_suffix_by_path(source) not in {"odt", "ods", "odp"}
    with pytest.raises(ValueError, match="Unsupported file type"):
        parse(source)


def test_odf_rejects_mismatched_encrypted_and_expanding_packages() -> None:
    """验证格式错配、manifest 加密和超大重复行在分配前稳定失败。"""
    with pytest.raises(OdfParseError, match="expected"):
        OdtModel().predict(BytesIO(build_ods_fixture()))

    encrypted_content = (
        '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0">'
        "<office:body><office:text/></office:body></office:document-content>"
    )
    encrypted = build_odf_package("odt", encrypted_content, encrypted=True)
    with pytest.raises(OdfEncryptedError, match="Encrypted ODF"):
        OdtModel().predict(BytesIO(encrypted))

    expanding_content = (
        '<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0" '
        'xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0" '
        'xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0">'
        '<office:body><office:spreadsheet><table:table><table:table-row table:number-rows-repeated="4000001">'
        "<table:table-cell><text:p>x</text:p></table:table-cell>"
        "</table:table-row></table:table></office:spreadsheet></office:body></office:document-content>"
    )
    expanding = build_odf_package("ods", expanding_content)
    with pytest.raises(OdfResourceLimitError, match="max_grid_slots"):
        OdsModel().predict(BytesIO(expanding))


@pytest.mark.parametrize("span_attribute", ["number-rows-spanned", "number-columns-spanned"])
def test_odf_rejects_oversized_cell_spans_before_grid_materialization(
    monkeypatch: pytest.MonkeyPatch,
    span_attribute: str,
) -> None:
    """验证超大行列跨度在渲染单元格或扩容网格前立即失败。"""
    monkeypatch.setattr(odf_table_module, "MAX_GRID_SLOTS", 4)

    def unexpected_materialization(*_args: object, **_kwargs: object) -> None:
        """超限 span 不得进入单元格渲染或网格扩容。"""
        pytest.fail("oversized span reached grid materialization")

    monkeypatch.setattr(odf_table_module, "_ensure_row", unexpected_materialization)
    table = etree.fromstring(
        f'<table:table xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">'
        f'<table:table-row><table:table-cell table:{span_attribute}="5"/></table:table-row>'
        "</table:table>"
    )

    with pytest.raises(OdfResourceLimitError, match="max_grid_slots"):
        odf_table_module.parse_table_grid(table, unexpected_materialization)


def test_odf_rejects_projected_span_extent_before_extending_existing_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证单个 span 合法但累计宽度超限时不会先扩展现有行。"""
    monkeypatch.setattr(odf_table_module, "MAX_GRID_SLOTS", 4)
    original_ensure_row = odf_table_module._ensure_row
    observed_widths: list[int] = []

    def tracking_ensure_row(grid: object, row_index: int, width: int = 0) -> object:
        """记录实际扩容宽度，确保失败前未越过共享预算。"""
        observed_widths.append(width)
        return original_ensure_row(grid, row_index, width)  # type: ignore[arg-type]

    monkeypatch.setattr(odf_table_module, "_ensure_row", tracking_ensure_row)
    table = etree.fromstring(
        '<table:table xmlns:table="urn:oasis:names:tc:opendocument:xmlns:table:1.0">'
        '<table:table-row><table:table-cell table:number-columns-spanned="3"/>'
        '<table:table-cell table:number-columns-spanned="2"/></table:table-row>'
        "</table:table>"
    )

    with pytest.raises(OdfResourceLimitError, match="max_grid_slots"):
        odf_table_module.parse_table_grid(table, lambda _cell: "x")
    assert observed_widths and max(observed_widths) <= 3


@pytest.mark.parametrize(
    "target",
    [
        "javascript:alert(1)",
        "JaVaScRiPt:alert(1)",
        "data:text/plain,unsafe",
        "vbscript:msgbox(1)",
        "file:///tmp/unsafe",
        "ftp://example.com/file",
        "//example.com/path",
        "\\\\server\\share",
    ],
)
def test_odf_rejects_unsafe_hyperlinks_before_shared_renderers(target: str) -> None:
    """验证危险 ODF 链接在 Raw 阶段降级，Markdown 与 DOCX 不再携带目标。"""
    content = f'''<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink">
 <office:body><office:text><text:p>before <text:a xlink:href="{target}">click</text:a> after</text:p>
 </office:text></office:body></office:document-content>'''
    payload = build_odf_package("odt", content)
    pages = OdtModel().predict(BytesIO(payload))
    middle, _ = doc_analyze(payload, file_suffix="odt")
    markdown = render_markdown(middle)
    docx = render_docx(middle)
    with ZipFile(BytesIO(docx)) as package:
        relationships = package.read("word/_rels/document.xml.rels").decode("utf-8")

    assert pages == [[{"type": BlockType.TEXT, "content": "before click after"}]]
    assert markdown == "before click after"
    assert target not in relationships


@pytest.mark.parametrize(
    "target",
    [
        "https://example.com/path",
        "mailto:reader@example.com",
        "tel:+123456",
        "chapter.odt#section-one",
    ],
)
def test_odf_preserves_allowed_external_and_relative_hyperlinks(target: str) -> None:
    """验证允许协议、相对地址和 fragment 继续进入共享 hyperlink 协议。"""
    content = f'''<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink">
 <office:body><office:text><text:p><text:a xlink:href="{target}">click</text:a></text:p>
 </office:text></office:body></office:document-content>'''
    pages = OdtModel().predict(BytesIO(build_odf_package("odt", content)))

    assert pages == [
        [
            {
                "type": BlockType.TEXT,
                "content": f"<hyperlink><text>click</text><url>{target}</url></hyperlink>",
            }
        ]
    ]


def test_odf_preserves_title_fragment_and_drops_unemittable_text_fragment() -> None:
    """验证本地 fragment 仅链接到标题类 block 实际公开的 bookmark。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink">
 <office:body><office:text>
  <text:p><text:a xlink:href="#title-target">Title jump</text:a> / <text:a xlink:href="#text-target">Text jump</text:a></text:p>
  <text:h text:outline-level="1"><text:bookmark-start text:name="title-target"/>Heading</text:h>
  <text:p><text:bookmark text:name="text-target"/>Ordinary target</text:p>
 </office:text></office:body>
</office:document-content>"""

    pages = OdtModel().predict(BytesIO(build_odf_package("odt", content)))

    assert pages[0][0]["content"] == ("<hyperlink><text>Title jump</text><url>#title-target</url></hyperlink> / Text jump")
    assert pages[0][1]["type"] == BlockType.PARAGRAPH_TITLE
    assert pages[0][1]["anchor"] == "title-target"
    assert pages[0][2] == {"type": BlockType.TEXT, "content": "Ordinary target"}


def test_odf_corrupt_optional_styles_and_external_image_degrade_locally() -> None:
    """验证可选样式损坏和外部图片不会阻断正文或触发网络读取。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:draw="urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"
 xmlns:xlink="http://www.w3.org/1999/xlink">
 <office:body><office:text><text:p>visible</text:p>
  <draw:frame><draw:image xlink:href="https://example.com/external.png"/></draw:frame>
 </office:text></office:body></office:document-content>"""
    pages = OdtModel().predict(BytesIO(build_odf_package("odt", content, styles_xml="<broken")))
    assert pages == [[{"type": BlockType.TEXT, "content": "visible"}]]


def test_odf_style_cycle_is_bounded_and_preserves_text() -> None:
    """验证循环 parent-style-name 在有限链路内降级，不阻塞正文解析。"""
    content = """<office:document-content
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:text="urn:oasis:names:tc:opendocument:xmlns:text:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0">
 <office:body><office:text><text:p text:style-name="A">visible</text:p></office:text></office:body>
</office:document-content>"""
    styles = """<office:document-styles
 xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0"
 xmlns:style="urn:oasis:names:tc:opendocument:xmlns:style:1.0">
 <office:styles><style:style style:name="A" style:family="paragraph" style:parent-style-name="B"/>
  <style:style style:name="B" style:family="paragraph" style:parent-style-name="A"/>
 </office:styles></office:document-styles>"""
    assert OdtModel().predict(BytesIO(build_odf_package("odt", content, styles_xml=styles))) == [
        [{"type": BlockType.TEXT, "content": "visible"}]
    ]


def test_odf_package_rejects_unsafe_member_paths_and_dtd() -> None:
    """验证 ZIP 上跳成员和 XML DTD 在进入语义解析前失败。"""
    output = BytesIO()
    with ZipFile(output, "w", ZIP_DEFLATED) as package:
        package.writestr("mimetype", "application/vnd.oasis.opendocument.text")
        package.writestr("../escape", b"unsafe")
    with pytest.raises(OdfParseError, match="unsafe member path"):
        OdfPackage(output.getvalue())

    dtd_content = """<!DOCTYPE doc [<!ENTITY x "hidden">]>
<office:document-content xmlns:office="urn:oasis:names:tc:opendocument:xmlns:office:1.0">
 <office:body><office:text/></office:body></office:document-content>"""
    with pytest.raises(OdfParseError, match="DTD is not allowed"):
        OdtModel().predict(BytesIO(build_odf_package("odt", dtd_content)))


@pytest.mark.parametrize(
    ("suffix", "payload", "expected"),
    [
        ("odt", build_odt_fixture(), {"page_count": 3, "title": "ODT Meta", "author": "Alice", "keywords": "one"}),
        ("ods", build_ods_fixture(), {"page_count": 2}),
        ("odp", build_odp_fixture(), {"page_count": 3}),
    ],
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


@pytest.mark.parametrize(
    ("suffix", "payload"),
    [("odt", build_odt_fixture()), ("ods", build_ods_fixture()), ("odp", build_odp_fixture())],
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
    assert result.middle_json.file_suffix == async_result.middle_json.file_suffix == suffix
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
            ocr_mode="auto",
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
    assert payload["file_suffix"] == "odt"
    assert payload["effort"] == "flash"
    assert payload["parse_mode"] == "txt"


@pytest.mark.parametrize(
    ("suffix", "payload", "page_count"),
    [("odt", build_odt_fixture(), 3), ("ods", build_ods_fixture(), 2), ("odp", build_odp_fixture(), 3)],
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


def test_csv_and_rtf_runtime_do_not_load_odf_modules() -> None:
    """验证新增 ODF converter 不进入既有 CSV/RTF 的惰性导入边界。"""
    script = "\n".join(
        [
            "import io, sys",
            "from mineru.model.flash import CsvModel, RtfModel",
            "CsvModel().predict(io.BytesIO(b'a,b\\n1,2\\n'))",
            "RtfModel().predict(io.BytesIO(b'{\\\\rtf1 ok}'))",
            "assert not any(name.startswith('mineru.model.flash.office.odf') for name in sys.modules)",
        ]
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_odf_subpackage_does_not_export_models() -> None:
    """验证 ODF 模型只从 Flash 根包公开，不形成第二套公共路径。"""
    assert importlib.util.find_spec("mineru.model.flash.office.odf.model") is None
    package = __import__("mineru.model.flash.office.odf", fromlist=["__all__"])
    assert package.__all__ == []
