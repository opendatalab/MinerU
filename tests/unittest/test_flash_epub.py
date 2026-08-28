from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest
from bs4 import BeautifulSoup
from lxml import etree

import mineru.model.flash.epub.package as epub_package_module
from mineru.backend.analyze import aio_doc_analyze, doc_analyze
from mineru.backend.postprocess.lists import fix_office_list_blocks
from mineru.doclib.core.file_io import extract_metadata
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.parse_svc import ParseService
from mineru.errors import InvalidRequestError
from mineru.model.flash import EpubModel
from mineru.model.flash.epub import EpubEncryptedError, EpubPackage, EpubParseError, EpubResourceLimitError, detect_epub
from mineru.model.flash._shared.markup import MarkupStylesheet, TextStyle
from mineru.model.flash.epub.xhtml import EpubChapterConverter, build_anchor_registry, convert_svg_spine
from mineru.parser import MinerUParser, parse, parse_async
from mineru.parser import api_server
from mineru.parser.api_server import CreateJobRequest, FileStore
from mineru.parser.file_type import guess_suffix_by_bytes, guess_suffix_by_path
from mineru.render import RenderMode, render_docx, render_html, render_markdown, render_structured_content
from mineru.types import BlockType, PageFootnoteBlock

from _epub_test_utils import (
    build_epub2_fixture,
    build_epub_fixture,
    build_epub_notes_fixture,
    build_epub_table_toc_fixture,
)


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
    assert model.effort == middle.effort == "flash"
    assert model.parse_mode == middle.parse_mode == "txt"
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


def test_epub_notes_use_page_footnote_and_document_wide_anchors() -> None:
    """验证 Footnote/Endnote、ARIA role、重复 ID 与跨章节 noteref 的统一语义。"""
    middle, model = doc_analyze(build_epub_notes_fixture(), file_suffix="epub")
    blocks = [block for page in middle.pages for block in page.blocks]
    footnotes = [block for block in blocks if block.type == BlockType.PAGE_FOOTNOTE]
    assert all(isinstance(block, PageFootnoteBlock) for block in footnotes)
    footnote_by_text = {block.content: block for block in footnotes}  # type: ignore[union-attr]

    first = footnote_by_text["First footnote paragraph back."]
    second = footnote_by_text["Second footnote paragraph."]
    endnote = footnote_by_text["First endnote paragraph."]
    duplicate_first = footnote_by_text["First duplicate note."]
    duplicate_second = footnote_by_text["Second duplicate note."]
    assert first.anchor and endnote.anchor and duplicate_first.anchor  # type: ignore[union-attr]
    assert first.anchor == "epub-5da65ec1e273b731b44c"  # type: ignore[union-attr]
    assert endnote.anchor == "epub-b7ab2fcf9373c09c71d6"  # type: ignore[union-attr]
    assert duplicate_first.anchor == "epub-2bd2576dd7c5d3cf58f4"  # type: ignore[union-attr]
    assert duplicate_second.anchor == "epub-2b154ee53d6a59bb32f7"  # type: ignore[union-attr]
    assert second.anchor is None  # type: ignore[union-attr]
    assert duplicate_second.anchor and duplicate_second.anchor != duplicate_first.anchor  # type: ignore[union-attr]
    assert footnote_by_text["ARIA footnote."].anchor  # type: ignore[union-attr]
    assert footnote_by_text["ARIA endnote."].anchor  # type: ignore[union-attr]
    assert footnote_by_text["Legacy rearnote."].anchor  # type: ignore[union-attr]
    assert footnote_by_text["Anonymous endnote."].anchor  # type: ignore[union-attr]

    body_text = next(
        block
        for block in middle.pages[0].blocks
        if block.type == BlockType.TEXT and "Same-page" in block.content  # type: ignore[union-attr]
    )
    assert f"<url>#{first.anchor}</url>" in body_text.content  # type: ignore[union-attr]
    assert f"<url>#{endnote.anchor}</url>" in body_text.content  # type: ignore[union-attr]
    assert f"<url>#{duplicate_first.anchor}</url>" in body_text.content  # type: ignore[union-attr]
    assert "empty [4]" in body_text.content and "empty <hyperlink>" not in body_text.content  # type: ignore[union-attr]
    assert "<hyperlink>back" not in first.content  # type: ignore[union-attr]

    assert any(block.type == BlockType.TEXT and block.content == "Ordinary aside." for block in blocks)  # type: ignore[union-attr]
    assert any(block.type == BlockType.TEXT and block.content == "Footnotes collection label." for block in blocks)  # type: ignore[union-attr]
    assert any(block.type == BlockType.LIST and "Footnote sibling list" in str(block.content) for block in blocks)  # type: ignore[union-attr]
    assert any(block.type == BlockType.TABLE and "Complex only" in str(block.content) for block in blocks)  # type: ignore[union-attr]
    assert all(
        block.get("type") != BlockType.REF_TEXT
        for page in model.pages
        for block in page
        if "footnote" in str(block.get("content", "")).casefold() or "endnote" in str(block.get("content", "")).casefold()
    )

    default_markdown = render_markdown(middle)
    full_markdown = render_markdown(middle, mode=RenderMode.FULL)
    assert "First footnote paragraph" in default_markdown
    assert f"](#{first.anchor})" in default_markdown
    assert f'id="{first.anchor}" class="mineru-page-footnote"' in default_markdown
    assert f'id="{first.anchor}" class="mineru-page-footnote"' in full_markdown
    assert "Page footnote:" not in default_markdown

    for html_output in (
        render_html(middle, standalone=False),
        render_html(middle, mode=RenderMode.FULL, standalone=False),
    ):
        assert f'href="#{first.anchor}"' in html_output
        assert f'id="{first.anchor}"' in html_output
        assert 'class="mineru-page-footnote"' in html_output

    structured = render_structured_content(middle)
    structured_blocks = [block for page in structured["pages"] for block in page["blocks"]]
    structured_footnote = next(block for block in structured_blocks if block.get("anchor") == first.anchor)
    assert structured_footnote["type"] == BlockType.PAGE_FOOTNOTE
    assert structured_footnote["content"] == "First footnote paragraph back."
    assert any(f"](#{first.anchor})" in str(block.get("content", "")) for block in structured_blocks)


def test_epub_internal_links_and_lists_use_cross_renderer_projection() -> None:
    """验证 spine 链接保留，列表规范为连续阿拉伯序号且不注入目录页。"""
    middle, _ = doc_analyze(build_epub_fixture(), file_suffix="epub")
    first_page = middle.pages[0]
    second_page = middle.pages[1]
    first_title = first_page.blocks[0]
    second_title = second_page.blocks[0]
    assert first_title.anchor and second_title.anchor  # type: ignore[union-attr]
    assert first_title.anchor == "epub-1daacccca5bf43833643"  # type: ignore[union-attr]
    assert second_title.anchor == "epub-8d7fe965e2d714cf08ae"  # type: ignore[union-attr]
    markdown = render_markdown(middle)
    assert markdown.startswith(f'<a id="{first_title.anchor}"></a>\n# Chapter One')  # type: ignore[union-attr]
    assert "NAV Chapter One" not in markdown
    assert "NCX Chapter One" not in markdown
    assert "Landmark" not in markdown
    assert f"](#{second_title.anchor})" in markdown  # type: ignore[union-attr]
    assert f"](#{first_title.anchor})" in markdown  # type: ignore[union-attr]
    assert "3. Three" in markdown
    assert "4. Two" in markdown
    list_block = next(block for block in first_page.blocks if block.type == BlockType.LIST)
    assert [child.content for child in list_block.content] == ["3. Three", "4. Two"]
    assert all(label in render_html(middle, standalone=False) for label in ("Three", "Two"))
    structured = render_structured_content(middle)
    assert any(block.get("content") == "3. Three\n4. Two" for page in structured["pages"] for block in page["blocks"])
    assert render_docx(middle).startswith(b"PK")


def test_epub_hidden_list_items_do_not_consume_normalized_numbers() -> None:
    """验证隐藏条目先被过滤，统一列表后处理再为可见内容连续编号。"""
    package = EpubPackage(build_epub_fixture())
    chapter_path = "EPUB/text/ch1.xhtml"
    try:
        root = package.xml_part(chapter_path, allow_external_doctype=True)
        ordered_list = next(element for element in root.iter() if isinstance(element.tag, str) and element.tag.endswith("}ol"))
        ordered_list.attrib.pop("start", None)
        ordered_list.attrib.pop("reversed", None)
        ordered_list.attrib.pop("type", None)
        items = [child for child in ordered_list if isinstance(child.tag, str) and child.tag.endswith("}li")]
        items[0].set("hidden", "hidden")
        anchors = build_anchor_registry([(chapter_path, root)], package)
        blocks = EpubChapterConverter(package, chapter_path, root, anchors).convert()
        list_block = next(block for block in blocks if block["type"] == BlockType.LIST)

        assert list_block["start"] == 1
        assert list_block["content"] == [{"type": BlockType.TEXT, "content": "Two"}]
        fix_office_list_blocks([list_block])
        assert list_block["content"] == [{"type": BlockType.TEXT, "content": "1. Two"}]
    finally:
        package.close()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("hidden", "hidden"),
        ("aria-hidden", "true"),
        ("style", "display: none"),
        ("class", "hidden"),
    ],
)
def test_epub_anchor_registry_excludes_hidden_headings(attribute: str, value: str) -> None:
    """验证 registry 不为 converter 会丢弃的隐藏标题生成悬空 anchor。"""
    package = EpubPackage(build_epub_fixture())
    chapter_path = "EPUB/text/ch1.xhtml"
    try:
        root = package.xml_part(chapter_path, allow_external_doctype=True)
        heading = next(element for element in root.iter() if isinstance(element.tag, str) and element.tag.endswith("}h1"))
        heading.set(attribute, value)
        heading.text = "Hidden target"
        body = next(element for element in root.iter() if isinstance(element.tag, str) and element.tag.endswith("}body"))
        namespace = etree.QName(body).namespace
        paragraph = etree.SubElement(body, f"{{{namespace}}}p")
        link = etree.SubElement(paragraph, f"{{{namespace}}}a", href="#chapter-one")
        link.text = "Hidden link"

        anchors = build_anchor_registry([(chapter_path, root)], package)
        blocks = EpubChapterConverter(package, chapter_path, root, anchors).convert()

        assert anchors.resolve_link("#chapter-one", base_part=chapter_path) is None
        assert "Hidden target" not in str(blocks)
        assert "Hidden link" in str(blocks)
        assert "<hyperlink>Hidden link" not in str(blocks)
    finally:
        package.close()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("hidden", "hidden"),
        ("aria-hidden", "true"),
        ("style", "display: none"),
        ("class", "hidden"),
    ],
)
def test_epub_anchor_registry_excludes_hidden_notes(attribute: str, value: str) -> None:
    """验证隐藏 note 不生成 converter 无法兑现的 fragment target。"""
    package = EpubPackage(build_epub_notes_fixture())
    try:
        chapters: list[tuple[str, etree._Element]] = []
        note_path = ""
        hidden_note: etree._Element | None = None
        for item in package.spine:
            if not item.path or not item.media_type or "xhtml" not in item.media_type:
                continue
            root = package.xml_part(item.path, allow_external_doctype=True)
            chapters.append((item.path, root))
            for element in root.iter():
                if element.get("id") == "fn-one":
                    hidden_note = element
                    note_path = item.path
        assert hidden_note is not None
        hidden_note.set(attribute, value)
        if attribute == "class":
            namespace = etree.QName(hidden_note.getroottree().getroot()).namespace
            style = etree.SubElement(hidden_note.getroottree().getroot(), f"{{{namespace}}}style")
            style.text = ".hidden { display: none; }"

        anchors = build_anchor_registry(chapters, package)
        blocks = [
            block
            for chapter_path, root in chapters
            for block in EpubChapterConverter(package, chapter_path, root, anchors).convert()
        ]

        assert anchors.resolve_link("#fn-one", base_part=note_path) is None
        assert "First footnote paragraph" not in str(blocks)
        assert "Same-page [1]" in str(blocks)
        assert "<hyperlink>[1]" not in str(blocks)
    finally:
        package.close()


def test_epub_table_contents_resolve_title_marker_and_expand_only_exact_single_target_rows() -> None:
    """验证标题内部 marker 可跳转，且目录表格只扩展严格匹配的单目标行。"""
    middle, model = doc_analyze(build_epub_table_toc_fixture(), file_suffix="epub")
    assert len(middle.pages) == 2
    assert all(block.type != BlockType.INDEX for page in middle.pages for block in page.blocks)
    table = next(block for block in model.pages[0] if block["type"] == BlockType.TABLE)
    soup = BeautifulSoup(str(table["content"]), "html.parser")
    rows = soup.find_all("tr")

    positive_links = rows[0].find_all("a", href=True)
    assert [link.get_text(strip=True) for link in positive_links] == ["CHAPTER I.", "Target Chapter"]
    assert len({link["href"] for link in positive_links}) == 1
    assert rows[1].find_all("a", href=True)[0].get_text(strip=True) == "Mismatch"
    assert len(rows[1].find_all("a", href=True)) == 1
    assert rows[2].find("a", href=True)["href"] == "https://example.com"  # type: ignore[index]
    assert len(rows[2].find_all("a", href=True)) == 1
    assert len(rows[3].find_all("a", href=True)) == 2

    rendered = BeautifulSoup(render_html(middle, standalone=False), "html.parser")
    chapter_link = rendered.find("a", string="Target Chapter")
    chapter_title = rendered.find(id=positive_links[0]["href"].removeprefix("#"))
    assert chapter_link is not None and chapter_title is not None
    assert chapter_link["href"] == f"#{chapter_title['id']}"


def test_epub_table_spans_are_bounded_before_downstream_grid_parsing() -> None:
    """验证超大 EPUB colspan 在进入 DOCX 占位网格前被移除。"""
    package = EpubPackage(build_epub_fixture())
    chapter_path = "EPUB/text/ch2.xhtml"
    try:
        root = package.xml_part(chapter_path, allow_external_doctype=True)
        cell = next(element for element in root.iter() if isinstance(element.tag, str) and element.tag.endswith("}td"))
        cell.set("colspan", "100000000")

        anchors = build_anchor_registry([(chapter_path, root)], package)
        blocks = EpubChapterConverter(package, chapter_path, root, anchors).convert()
        table = next(block for block in blocks if block["type"] == BlockType.TABLE)
        parsed_cell = BeautifulSoup(str(table["content"]), "html.parser").find("td", string="1")

        assert parsed_cell is not None
        assert parsed_cell["rowspan"] == "2"
        assert not parsed_cell.has_attr("colspan")
    finally:
        package.close()


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("hidden", "hidden"),
        ("aria-hidden", "true"),
        ("style", "display: none"),
        ("class", "hidden"),
    ],
)
def test_epub_figure_skips_hidden_direct_images(attribute: str, value: str) -> None:
    """验证 figure 的直接图片同样遵守元素属性、内联样式和 CSS 隐藏规则。"""
    package = EpubPackage(build_epub_fixture())
    chapter_path = "EPUB/text/ch1.xhtml"
    try:
        root = package.xml_part(chapter_path, allow_external_doctype=True)
        figure = next(element for element in root.iter() if isinstance(element.tag, str) and element.tag.endswith("}figure"))
        image = next(child for child in figure if isinstance(child.tag, str) and child.tag.endswith("}img"))
        image.set(attribute, value)

        anchors = build_anchor_registry([(chapter_path, root)], package)
        blocks = EpubChapterConverter(package, chapter_path, root, anchors).convert()

        assert all(block["type"] != BlockType.IMAGE for block in blocks)
        assert "Dot caption" in str(blocks)
    finally:
        package.close()


def test_epub_figure_preserves_direct_text_and_child_tails() -> None:
    """验证共享 projector 不丢弃 XHTML figure 的直属文本及图片、caption tail。"""
    package = EpubPackage(build_epub_fixture())
    chapter_path = "EPUB/text/ch1.xhtml"
    try:
        root = package.xml_part(chapter_path, allow_external_doctype=True)
        figure = next(element for element in root.iter() if isinstance(element.tag, str) and element.tag.endswith("}figure"))
        image = next(child for child in figure if isinstance(child.tag, str) and child.tag.endswith("}img"))
        caption = next(child for child in figure if isinstance(child.tag, str) and child.tag.endswith("}figcaption"))
        figure.text = "Before"
        image.tail = "After"
        caption.tail = "Tail"

        anchors = build_anchor_registry([(chapter_path, root)], package)
        blocks = EpubChapterConverter(package, chapter_path, root, anchors).convert()
        relevant = [
            block
            for block in blocks
            if block.get("type") in {BlockType.IMAGE, BlockType.IMAGE_CAPTION}
            or (isinstance(block.get("content"), str) and block.get("content") in {"Before", "AfterTail"})
        ]

        assert [(block["type"], block.get("content")) for block in relevant] == [
            (BlockType.TEXT, "Before"),
            (BlockType.IMAGE, ""),
            (BlockType.IMAGE_CAPTION, "Dot caption"),
            (BlockType.TEXT, "AfterTail"),
        ]
    finally:
        package.close()


def test_public_parser_rejects_epub_page_range(tmp_path: Path) -> None:
    """验证 EPUB 公共 Parser 只接受整本解析。"""
    source = tmp_path / "book.epub"
    source.write_bytes(build_epub_fixture())

    with pytest.raises(InvalidRequestError, match="only supported for PDF") as exc_info:
        parse(source, page_range="2~3")
    assert exc_info.value.code == "page_range_invalid"

    with pytest.raises(InvalidRequestError, match="only supported for PDF"):
        asyncio.run(parse_async(source, page_range="2~3"))


@pytest.mark.parametrize(
    "suffix",
    ["doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf", "csv", "epub", "html", "odt", "ods", "odp"],
)
def test_non_pdf_analyze_rejects_non_empty_page_index_map(suffix: str) -> None:
    """验证所有非 PDF Analyze 分支拒绝伪造 partial page mapping。"""
    with pytest.raises(ValueError, match="only supported for PDF"):
        doc_analyze(b"not-read", page_index_map=[0], file_suffix=suffix)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "suffix",
    ["doc", "docx", "ppt", "pptx", "xls", "xlsx", "rtf", "csv", "epub", "html", "odt", "ods", "odp"],
)
def test_non_pdf_public_parser_rejects_page_range(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    suffix: str,
) -> None:
    """验证所有非 PDF 路径入口在初始化具体模型前拒绝 page_range。"""
    source = tmp_path / f"sample.{suffix}"
    source.write_bytes(b"not-read")
    monkeypatch.setattr("mineru.parser.file_type.guess_suffix_by_path", lambda _path: suffix)
    parser = MinerUParser(tier="flash")
    with pytest.raises(InvalidRequestError, match="full-document parsing") as exc_info:
        parser.parse(source, page_range="1")
    assert exc_info.value.code == "page_range_invalid"


def test_epub2_and_mimetype_less_compatibility_packages_parse() -> None:
    """验证 EPUB 2 common subset 和缺 mimetype 的 container 兼容分支。"""
    epub2 = build_epub2_fixture()
    compatibility = build_epub_fixture(omit_mimetype=True)
    assert detect_epub(epub2)
    assert detect_epub(compatibility)
    epub2_middle = doc_analyze(epub2, file_suffix="epub")[0]
    assert epub2_middle.pages[0].blocks[0].content == "EPUB Two Chapter"  # type: ignore[union-attr]
    assert epub2_middle.pages[0].blocks[1].content == "Legacy\u00a0body"  # type: ignore[union-attr]
    assert len(doc_analyze(compatibility, file_suffix="epub")[0].pages) == 3


def test_epub_spine_foreign_resource_uses_xhtml_fallback_chain() -> None:
    """验证 foreign spine item 缺失时仍沿 manifest fallback 到 XHTML。"""
    middle, _ = doc_analyze(build_epub_fixture(use_foreign_fallback=True), file_suffix="epub")
    assert middle.pages[1].blocks[0].content == "Section Two"  # type: ignore[union-attr]


def test_epub_spine_missing_supported_resource_uses_xhtml_fallback_chain() -> None:
    """验证缺失的 XHTML 主资源不会阻断可用 fallback 链。"""
    middle, _ = doc_analyze(build_epub_fixture(use_missing_supported_fallback=True), file_suffix="epub")
    assert middle.pages[1].blocks[0].content == "Section Two"  # type: ignore[union-attr]


def test_epub_nav_and_ncx_outside_spine_do_not_create_synthetic_page() -> None:
    """验证 spine 外的损坏 nav 与有效 NCX 都不会生成合成目录页。"""
    middle, _ = doc_analyze(build_epub_fixture(corrupt_nav=True), file_suffix="epub")
    markdown = render_markdown(middle)
    assert len(middle.pages) == 3
    assert all(block.type != BlockType.INDEX for page in middle.pages for block in page.blocks)
    assert "NCX Chapter One" not in markdown
    assert "NAV Chapter One" not in markdown


def test_epub_without_authored_toc_keeps_only_spine_pages_and_titles() -> None:
    """验证缺失 nav/NCX 时不从正文标题生成额外目录页。"""
    middle, _ = doc_analyze(
        build_epub_fixture(include_nav=False, include_ncx=False),
        file_suffix="epub",
    )
    assert len(middle.pages) == 3
    assert middle.pages[0].blocks[0].type == BlockType.DOC_TITLE
    assert middle.pages[1].blocks[0].type == BlockType.PARAGRAPH_TITLE
    assert all(block.type != BlockType.INDEX for page in middle.pages for block in page.blocks)


def test_epub_without_toc_or_headings_does_not_add_empty_page() -> None:
    """验证没有任何有效目录条目时不生成空 IndexBlock 专页。"""
    middle, _ = doc_analyze(
        build_epub_fixture(
            include_nav=False,
            include_ncx=False,
            strip_headings=True,
        ),
        file_suffix="epub",
    )
    assert len(middle.pages) == 3
    assert middle.pages[0].blocks[0].type != BlockType.INDEX


def test_epub_navigation_in_spine_preserves_page_order_and_extra_body_content() -> None:
    """验证 spine 中的 nav 作为普通 XHTML 页保留目录和额外正文。"""
    middle, _ = doc_analyze(
        build_epub_fixture(nav_in_spine=True, nav_extra_body_text="Publisher front matter"),
        file_suffix="epub",
    )
    assert len(middle.pages) == 4
    assert all(block.type != BlockType.INDEX for page in middle.pages for block in page.blocks)
    assert any(getattr(block, "content", None) == "Publisher front matter" for block in middle.pages[0].blocks)
    assert middle.pages[1].blocks[0].content == "Chapter One"  # type: ignore[union-attr]


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


def test_epub_corrupt_chapter_keeps_empty_spine_placeholder() -> None:
    """验证局部 XHTML 损坏不移动后续 spine 页号。"""
    middle, model = doc_analyze(build_epub_fixture(corrupt_second_chapter=True), file_suffix="epub")
    assert len(model.pages) == 3
    assert model.pages[1] == []
    assert [page.page_idx for page in middle.pages] == [0, 1, 2]
    assert middle.pages[1].blocks == []
    assert "SVG text" in middle.pages[2].blocks[0].content  # type: ignore[union-attr]


def test_epub_svg_extraction_skips_hidden_descendants_and_hidden_root() -> None:
    """验证 standalone SVG 不提取隐藏文本、隐藏图片或隐藏祖先子树。"""
    package = EpubPackage(build_epub_fixture())
    path = "EPUB/fixed/page.svg"
    try:
        root = package.xml_part(path)
        assert root is not None
        namespace = etree.QName(root).namespace
        original_text = next(root.iter(f"{{{namespace}}}text"))
        original_text.set("style", "display: none")
        original_image = next(root.iter(f"{{{namespace}}}image"))
        original_image.set("style", "visibility: hidden")
        hidden_group = etree.SubElement(root, f"{{{namespace}}}g", style="display: none")
        etree.SubElement(hidden_group, f"{{{namespace}}}text").text = "hidden group text"
        etree.SubElement(root, f"{{{namespace}}}text").text = "visible graphic label"

        blocks = convert_svg_spine(package, path, root)

        assert "visible graphic label" in str(blocks)
        assert "SVG text" not in str(blocks)
        assert "hidden group text" not in str(blocks)
        assert all(block["type"] != BlockType.IMAGE for block in blocks)

        root.set("style", "display: none")
        assert convert_svg_spine(package, path, root) == []
    finally:
        package.close()


def test_epub_malformed_resource_and_link_references_degrade_locally() -> None:
    """验证非法 URI 只丢弃样式、图片或链接目标，不阻断章节正文。"""
    package = EpubPackage(build_epub_fixture())
    chapter_path = "EPUB/text/ch1.xhtml"
    try:
        root = package.xml_part(chapter_path, allow_external_doctype=True)
        assert root is not None
        stylesheet = next(element for element in root.iter() if etree.QName(element).localname == "link")
        hyperlink = next(element for element in root.iter() if etree.QName(element).localname == "a")
        image = next(element for element in root.iter() if etree.QName(element).localname == "img")
        stylesheet.set("href", "http://[")
        hyperlink.set("href", "http://[")
        image.set("src", "http://[")

        anchors = build_anchor_registry([(chapter_path, root)], package)
        blocks = EpubChapterConverter(package, chapter_path, root, anchors).convert()

        assert "Chapter One" in str(blocks)
        assert "chapter two" in str(blocks)
        assert "Remote image" in str(blocks)
        assert "<hyperlink>chapter two" not in str(blocks)
    finally:
        package.close()


def test_epub_rejects_encrypted_unsafe_and_dtd_inputs() -> None:
    """验证选中正文加密、上跳成员和 DTD 在语义解析前稳定失败。"""
    encrypted = build_epub_fixture(encrypted_paths=("EPUB/text/ch1.xhtml",))
    with pytest.raises(EpubEncryptedError, match="Encrypted EPUB resource"):
        EpubModel().predict(BytesIO(encrypted))

    unsafe = build_epub_fixture(unsafe_member="../escape")
    with pytest.raises(EpubParseError, match="unsafe member path"):
        EpubPackage(unsafe)

    dtd_package = EpubPackage(build_epub_fixture(dtd_first_chapter=True))
    try:
        with pytest.raises(EpubParseError, match="DTD declarations are not allowed"):
            dtd_package.xml_part("EPUB/text/ch1.xhtml")
    finally:
        dtd_package.close()


def test_epub_resource_limits_fail_before_semantic_conversion(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 ZIP 条目与 XML 深度预算在正文遍历前生效。"""
    payload = build_epub_fixture()
    monkeypatch.setattr(epub_package_module, "MAX_ENTRY_BYTES", 32)
    with pytest.raises(EpubResourceLimitError, match="max_entry_bytes"):
        EpubPackage(payload)

    monkeypatch.setattr(epub_package_module, "MAX_ENTRY_BYTES", 128 * 1024 * 1024)
    monkeypatch.setattr(epub_package_module, "MAX_XML_DEPTH", 2)
    with pytest.raises(EpubResourceLimitError, match="max_xml_depth"):
        EpubPackage(payload)


def test_epub_stylesheet_indexes_repeated_selectors_and_preserves_cascade_order() -> None:
    """验证重复 selector 按属性聚合，交错同优先级规则仍遵守源码顺序。"""
    stylesheet = MarkupStylesheet()
    stylesheet.add(
        ".x { font-weight: bold; display: none; }"
        ".y { font-weight: normal; display: block; }"
        ".x { font-style: italic; }" + ".unused { text-decoration: underline; }" * 10_000
    )
    element = etree.fromstring(b'<span class="x y"/>')

    resolved = stylesheet.resolve(element, TextStyle())

    assert resolved.text.bold is False
    assert resolved.text.italic is True
    assert resolved.hidden is False
    assert len(stylesheet._class_cascades) == 3


def test_epub_stylesheet_honors_important_before_specificity_and_source_order() -> None:
    """验证 important 声明优先于后续普通规则和普通 inline，并允许 inline important 覆盖。"""
    stylesheet = MarkupStylesheet()
    stylesheet.add(
        ".secret { display: none !important; font-weight: bold !important; }.secret { display: block; font-weight: normal; }"
    )

    normal_inline = stylesheet.resolve(
        etree.fromstring(b'<span class="secret" style="display: block; font-weight: normal"/>'),
        TextStyle(),
    )
    important_inline = stylesheet.resolve(
        etree.fromstring(b'<span class="secret" style="display: block !important; font-weight: normal !important"/>'),
        TextStyle(),
    )

    assert normal_inline.hidden is True
    assert normal_inline.text.bold is True
    assert important_inline.hidden is False
    assert important_inline.text.bold is False


@pytest.mark.parametrize(
    ("css", "expected_hidden"),
    [
        (".secret { display: none; visibility: visible; }", True),
        (".secret { display: block; visibility: hidden; }", True),
        (".secret { display: block; visibility: collapse; }", True),
        (
            ".secret { display: none !important; visibility: hidden; }"
            ".secret { display: block; visibility: visible !important; }",
            True,
        ),
        (".secret { display: block; visibility: visible; }", False),
    ],
)
def test_epub_stylesheet_tracks_display_and_visibility_independently(css: str, expected_hidden: bool) -> None:
    """验证 display 与 visibility 各自级联，任一计算结果隐藏时都不输出元素。"""
    stylesheet = MarkupStylesheet()
    stylesheet.add(css)

    resolved = stylesheet.resolve(etree.fromstring(b'<span class="secret"/>'), TextStyle())

    assert resolved.hidden is expected_hidden


def test_epub_combined_visibility_rule_does_not_export_hidden_content() -> None:
    """验证真实 EPUB 中 visibility:visible 不会覆盖同规则的 display:none。"""
    source = BytesIO(build_epub_fixture())
    rewritten = BytesIO()
    with ZipFile(source) as archive, ZipFile(rewritten, "w", ZIP_DEFLATED) as output:
        for info in archive.infolist():
            data = archive.read(info.filename)
            if info.filename == "EPUB/styles/book.css":
                data = b".hidden { display: none; visibility: visible; }"
            output.writestr(info, data)

    pages = EpubModel().predict(BytesIO(rewritten.getvalue()))

    assert "hidden secret" not in str(pages)


def test_epub_stylesheet_allows_visible_descendant_to_override_inherited_visibility() -> None:
    """验证 visibility 可继承且显式 visible 后代能够恢复自身输出。"""
    stylesheet = MarkupStylesheet()
    stylesheet.add(".parent { visibility: hidden; } .child { visibility: visible; }")
    parent = etree.fromstring(b'<div class="parent"><span class="child"/></div>')
    child = parent[0]

    parent_style = stylesheet.resolve(parent, TextStyle())
    child_style = stylesheet.resolve(child, parent_style.text, parent_style.visibility_hidden)

    assert parent_style.subtree_hidden is False
    assert parent_style.visibility_hidden is True
    assert child_style.subtree_hidden is False
    assert child_style.visibility_hidden is False


def test_epub_visibility_visible_descendants_survive_hidden_containers() -> None:
    """验证真实 EPUB 的块、行内、列表、表格、SVG 和标题锚点均可从 visibility:hidden 恢复。"""
    source = BytesIO(build_epub_fixture())
    rewritten = BytesIO()
    replacement = b"""<div class="visibility-parent">
  hidden parent text
  <h2 id="visibility-heading">hidden heading text <span class="visibility-child">Visible descendant heading</span></h2>
  <p>hidden paragraph</p>
  <p class="visibility-child">visible block text</p>
  <p>hidden before <span class="visibility-child">visible inline text</span> hidden after</p>
  <ol><li class="visibility-child">visible list item</li><li>hidden list item</li></ol>
  <table><tbody><tr><td class="visibility-child">visible table cell</td><td>hidden table cell</td></tr></tbody></table>
  <svg xmlns="http://www.w3.org/2000/svg"><g><text class="visibility-child">visible SVG text</text>
    <text>hidden SVG text</text></g></svg>
</div>
<div class="display-parent"><p class="visibility-child">display-hidden descendant</p></div>
<div hidden="hidden"><p class="visibility-child">attribute-hidden descendant</p></div>
<p><a href="#visibility-heading">jump to visible heading</a></p>"""
    with ZipFile(source) as archive, ZipFile(rewritten, "w", ZIP_DEFLATED) as output:
        for info in archive.infolist():
            data = archive.read(info.filename)
            if info.filename == "EPUB/text/ch1.xhtml":
                data = data.replace(b'<p class="hidden">hidden secret</p>', replacement)
            elif info.filename == "EPUB/styles/book.css":
                data = (
                    b".visibility-parent { visibility: hidden; }"
                    b".visibility-child { visibility: visible; }"
                    b".display-parent { display: none; }"
                )
            output.writestr(info, data)

    pages = EpubModel().predict(BytesIO(rewritten.getvalue()))
    blocks = [block for page in pages for block in page]
    flattened = str(blocks)

    for visible_text in (
        "Visible descendant heading",
        "visible block text",
        "visible inline text",
        "visible list item",
        "visible table cell",
        "visible SVG text",
    ):
        assert visible_text in flattened
    for hidden_text in (
        "hidden parent text",
        "hidden heading text",
        "hidden paragraph",
        "hidden before",
        "hidden after",
        "hidden list item",
        "hidden table cell",
        "hidden SVG text",
        "display-hidden descendant",
        "attribute-hidden descendant",
    ):
        assert hidden_text not in flattened

    heading = next(block for block in blocks if block.get("content") == "Visible descendant heading")
    assert f"<url>#{heading['anchor']}</url>" in flattened


def test_epub_visibility_hidden_body_still_visits_visible_children() -> None:
    """验证 body 自身 visibility:hidden 时不会在显式可见子节点之前剪枝。"""
    source = BytesIO(build_epub_fixture())
    rewritten = BytesIO()
    with ZipFile(source) as archive, ZipFile(rewritten, "w", ZIP_DEFLATED) as output:
        for info in archive.infolist():
            data = archive.read(info.filename)
            if info.filename == "EPUB/text/ch1.xhtml":
                data = data.replace(b"<body>", b'<body class="visibility-parent">', 1)
                data = data.replace(b'<h1 id="chapter-one">', b'<h1 id="chapter-one" class="visibility-child">', 1)
            elif info.filename == "EPUB/styles/book.css":
                data = (
                    b".visibility-parent { visibility: hidden; }"
                    b".visibility-child { visibility: visible; }"
                    b".hidden { display: none; }"
                )
            output.writestr(info, data)

    first_page = str(EpubModel().predict(BytesIO(rewritten.getvalue()))[0])

    assert "Chapter One" in first_page
    assert "Hello" not in first_page


def test_epub_stylesheet_rejects_overlong_numeric_font_weight_before_int() -> None:
    """验证超长或越界数字字重作为无效声明忽略且不会覆盖继承样式。"""
    stylesheet = MarkupStylesheet()
    stylesheet.add(f".x {{ font-weight: {'9' * 100_000}; }} .y {{ font-weight: 1001; }}")

    inherited = TextStyle(bold=True)
    overlong = stylesheet.resolve(etree.fromstring(b'<span class="x"/>'), inherited)
    out_of_range = stylesheet.resolve(etree.fromstring(b'<span class="y"/>'), inherited)

    assert overlong.text.bold is True
    assert out_of_range.text.bold is True


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
            ocr_mode="auto",
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
            ocr_mode="auto",
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
        assert parses == [{"tier": "flash", "status": "pending", "privacy": "local", "page_range": "1~3"}]

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
            "1~3",
        )
    )
    assert result is expected
    assert observed == {"path": "/tmp/book.epub", "tier": "flash", "page_range": ""}


def test_epub_public_import_does_not_load_heavy_models() -> None:
    """验证公开 Parser 导入不会提前加载 EPUB、Torch、OpenCV 或 VLM。"""
    code = """
import sys
import mineru.parser
blocked = ('torch', 'cv2', 'mineru.model.flash.epub', 'mineru_vl_utils')
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
