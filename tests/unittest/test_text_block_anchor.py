from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from zipfile import ZipFile

from bs4 import BeautifulSoup
from lxml import etree
from pydantic import ValidationError
from pypdf import PdfReader
import pytest

from mineru.backend.analyze import doc_analyze
from mineru.render import (
    render_content_list,
    render_content_list_v2,
    render_docx,
    render_epub,
    render_html,
    render_markdown,
    render_pdf,
    render_structured_content,
)
from mineru.render._internal.common.planner import build_render_plan
from mineru.types import IndexBlock, MiddleJson, PageInfo, RefTextBlock, TextBlock


def _inline(text: str) -> list[dict[str, str]]:
    """构造最小结构化文本 span。"""
    return [{"type": "text", "content": text}]


def _middle_with_text_anchor() -> MiddleJson:
    """构造目录前向引用顶层 TextBlock 的严格文档。"""
    return MiddleJson(
        pages=[
            PageInfo(
                page_idx=0,
                blocks=[
                    IndexBlock(
                        type="index",
                        index=0,
                        content=[TextBlock(type="text", anchor="body target", content=_inline("Body target\t3"))],
                    )
                ],
            ),
            PageInfo(
                page_idx=1,
                blocks=[TextBlock(type="text", index=0, anchor="body target", content=_inline("Body paragraph"))],
            ),
        ],
        is_full_document=True,
        file_suffix="docx",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def _zip_text(payload: bytes, name: str) -> str:
    """读取 ZIP 容器中的 UTF-8 XML 文本。"""
    with ZipFile(BytesIO(payload)) as archive:
        return archive.read(name).decode("utf-8")


def test_text_anchor_is_strict_and_blocks_continuation_merge() -> None:
    """验证 TextBlock 接受 anchor，而带目标的续段不会被规划器吸收。"""
    anchored = TextBlock(type="text", anchor="target", content=_inline("continued"), continues_prev=True)
    middle = MiddleJson(
        pages=[
            PageInfo(page_idx=0, blocks=[TextBlock(type="text", index=0, content=_inline("first"))]),
            PageInfo(page_idx=1, blocks=[anchored.model_copy(update={"index": 0})]),
        ],
        is_full_document=True,
        file_suffix="docx",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )

    planned = build_render_plan(middle)
    assert planned[1][0].removed is False
    assert TextBlock.model_validate(anchored.model_dump()).anchor == "target"
    with pytest.raises(ValidationError):
        RefTextBlock.model_validate({"type": "ref_text", "anchor": "target", "content": _inline("ref")})


def test_duplicate_and_empty_text_anchors_emit_only_the_first_visible_target() -> None:
    """验证重复正文 anchor 仅首项生效，空正文不会产生悬空目标。"""
    middle = MiddleJson(
        pages=[
            PageInfo(
                page_idx=0,
                blocks=[
                    TextBlock(type="text", index=0, anchor="same", content=_inline("First")),
                    TextBlock(type="text", index=1, anchor="same", content=_inline("Second")),
                    TextBlock(type="text", index=2, anchor="empty", content=[]),
                ],
            )
        ],
        is_full_document=True,
        file_suffix="docx",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )

    assert render_markdown(middle).count('<a id="same"></a>') == 1
    rendered_html = BeautifulSoup(render_html(middle), "html.parser")
    assert len(rendered_html.select('[id="same"]')) == 1
    assert rendered_html.select_one('[id="empty"]') is None


def test_text_anchor_markdown_html_and_html_wire_roundtrip() -> None:
    """验证 Markdown、HTML 目标与 canonical HTML v1 往返保持 TextBlock anchor。"""
    middle = _middle_with_text_anchor()

    markdown = render_markdown(middle)
    assert "[Body target](#body%20target)" in markdown
    assert '<a id="body target"></a>\nBody paragraph' in markdown

    rendered_html = render_html(middle)
    soup = BeautifulSoup(rendered_html, "html.parser")
    target_wrapper = soup.select_one('.mineru-block[data-block-type="text"][data-anchor="body target"]')
    assert target_wrapper is not None
    assert target_wrapper.find("p")["id"] == "body-target"
    assert soup.select_one('[data-block-type="index"] a')["href"] == "#body-target"

    decoded, _ = doc_analyze(rendered_html.encode("utf-8"), effort="flash", parse_mode="auto", file_suffix="html")
    decoded_target = next(
        block
        for page in decoded.pages
        for block in page.blocks
        if isinstance(block, TextBlock) and block.anchor == "body target"
    )
    decoded_index = next(block for page in decoded.pages for block in page.blocks if isinstance(block, IndexBlock))
    assert decoded_target.anchor == "body target"
    assert isinstance(decoded_index.content[0], TextBlock)
    assert decoded_index.content[0].anchor == "body target"

    missing = MiddleJson.model_validate(
        {
            **middle.model_dump(mode="python", exclude={"pages"}),
            "pages": [
                {
                    "page_idx": 0,
                    "blocks": [
                        {
                            "type": "index",
                            "index": 0,
                            "content": [{"type": "text", "anchor": "missing", "content": _inline("Missing\t1")}],
                        }
                    ],
                }
            ],
        }
    )
    assert "#missing" not in render_markdown(missing)
    assert BeautifulSoup(render_html(missing), "html.parser").select_one('[data-block-type="index"] a') is None


def test_text_anchor_docx_pdf_and_epub_targets() -> None:
    """验证 DOCX bookmark、PDF 内链和 EPUB XHTML 目标均指向 TextBlock。"""
    middle = _middle_with_text_anchor()

    document_xml = _zip_text(render_docx(middle), "word/document.xml")
    assert 'w:bookmarkStart w:id="0" w:name="body_target"' in document_xml
    assert 'w:hyperlink w:anchor="body_target"' in document_xml

    pdf_reader = PdfReader(BytesIO(render_pdf(middle)))
    annotations = [annotation.get_object() for page in pdf_reader.pages for annotation in (page.get("/Annots") or [])]
    assert any("/Dest" in annotation for annotation in annotations)

    epub = render_epub(middle, modified_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc))
    content = etree.fromstring(_zip_text(epub, "EPUB/text/content.xhtml").encode("utf-8"))
    namespaces = {"xhtml": "http://www.w3.org/1999/xhtml"}
    assert content.xpath("//xhtml:p[@id='body-target']/text()", namespaces=namespaces) == ["Body paragraph"]
    navigation = etree.fromstring(_zip_text(epub, "EPUB/nav.xhtml").encode("utf-8"))
    assert "text/content.xhtml#body-target" in navigation.xpath("//xhtml:a/@href", namespaces=namespaces)


def test_text_anchor_structured_and_content_list_metadata() -> None:
    """验证结构化输出只保留 anchor 元数据，不向正文内容注入目标标签。"""
    middle = _middle_with_text_anchor()

    structured = render_structured_content(middle)
    structured_target = structured["pages"][1]["blocks"][0]
    assert structured_target["anchor"] == "body target"
    assert structured_target["content"] == "Body paragraph"
    assert "<a " not in structured_target["content"]

    content_list = render_content_list(middle)
    target_v1 = next(item for item in content_list if item.get("text") == "Body paragraph")
    assert target_v1["anchor"] == "body target"
    assert content_list[0]["list_items"] == ["- [Body target](#body%20target)"]

    content_list_v2 = render_content_list_v2(middle)
    assert content_list_v2[1][0]["anchor"] == "body target"
    assert content_list_v2[0][0]["content"]["list_items"][0]["anchor"] == "body target"
