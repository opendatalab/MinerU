# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, Tag

from .base import DocumentParser, ParseResult
from ..types import EMPTY_BBOX, Block, Line, PageInfo, Span


class HtmlParser(DocumentParser):
    """Parse HTML files into structured ``ParseResult``.

    HTML elements are mapped to typed blocks inside a single page.
    """

    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        html_text = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_text, "html.parser")

        body = soup.body or soup
        blocks: list[Block] = []
        _walk_elements(body, blocks)
        _assign_block_indexes(blocks)

        return ParseResult(
            pages=[PageInfo(page_idx=0, para_blocks=blocks)],
        )


_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCK_CONTAINER_TAGS = frozenset({
    "div", "section", "article", "aside", "main", "header", "footer", "nav",
})


def _heading_level(tag_name: str) -> int:
    return int(tag_name[1])


def _extract_text(element: Tag) -> str:
    return element.get_text(separator=" ", strip=True)


def _assign_block_indexes(blocks: list[Block]) -> None:
    """递归补齐 HTML 解析产生的局部顺序号，保证 middle_json block 契约稳定。"""
    for index, block in enumerate(blocks):
        block.index = index
        _assign_block_indexes(block.blocks)


def _line_with_spans(spans: list[Span]) -> Line:
    """HTML 没有页面坐标，统一使用 EMPTY_BBOX 表示未知行框。"""
    return Line(bbox=EMPTY_BBOX, spans=spans)


def _build_span(element: Tag) -> Span:
    content = _extract_text(element)
    img = element.find("img")
    if img is not None:
        src: str = img.get("src", "") or ""
        if src.startswith("data:image"):
            return Span(type="image", bbox=EMPTY_BBOX, image_base64=src, content=content)
        return Span(type="image", bbox=EMPTY_BBOX, image_path=src, content=content)
    return Span(type="text", bbox=EMPTY_BBOX, content=content)


def _walk_elements(parent: Tag, blocks: list[Block]) -> None:  # type: ignore[type-arg]
    for child in parent.children:
        if not isinstance(child, Tag):
            text = str(child).strip()
            if text:
                blocks.append(Block(
                    index=0,
                    type="text",
                    bbox=EMPTY_BBOX,
                    lines=[_line_with_spans([Span(type="text", bbox=EMPTY_BBOX, content=text)])],
                ))
            continue

        tag_name = child.name

        if tag_name in _HEADING_TAGS:
            blocks.append(_build_heading(child))
        elif tag_name in ("p", "span"):
            blocks.append(_build_text_block(child))
        elif tag_name in _BLOCK_CONTAINER_TAGS:
            _walk_elements(child, blocks)
        elif tag_name in ("ul", "ol"):
            blocks.append(_build_list(child))
        elif tag_name == "table":
            blocks.append(_build_table(child))
        elif tag_name == "pre":
            blocks.append(_build_code(child))
        elif tag_name in ("img",):
            blocks.append(_build_image_block(child))
        elif tag_name in ("br",):
            pass
        else:
            # fallback: treat unknown inline tags as text
            if child.get_text(strip=True):
                blocks.append(_build_text_block(child))


def _build_heading(element: Tag) -> Block:
    level = _heading_level(element.name)
    return Block(
        index=0,
        type="title",
        bbox=EMPTY_BBOX,
        level=level,
        lines=[_line_with_spans([_build_span(element)])],
    )


def _build_text_block(element: Tag) -> Block:
    # Collapse nested inline children into a single span.
    return Block(
        index=0,
        type="text",
        bbox=EMPTY_BBOX,
        lines=[_line_with_spans([_build_span(element)])],
    )


def _build_image_block(element: Tag) -> Block:
    span = _build_span(element)
    return Block(
        index=0,
        type="image",
        bbox=EMPTY_BBOX,
        lines=[_line_with_spans([span])],
    )


def _build_list(element: Tag) -> Block:
    items: list[Block] = []
    for li in element.find_all("li", recursive=False):
        nested_blocks: list[Block] = []
        _walk_elements(li, nested_blocks)
        items.append(Block(
            index=0,
            type="list_item",
            bbox=EMPTY_BBOX,
            lines=[_line_with_spans([Span(type="text", bbox=EMPTY_BBOX, content=_extract_text(li))])],
            blocks=nested_blocks,
        ))
    return Block(index=0, type="list", bbox=EMPTY_BBOX, blocks=items)


def _build_table(element: Tag) -> Block:
    return Block(
        index=0,
        type="table",
        bbox=EMPTY_BBOX,
        lines=[_line_with_spans([Span(type="table", bbox=EMPTY_BBOX, content=str(element))])],
    )


def _build_code(element: Tag) -> Block:
    code = element.find("code")
    content = _extract_text(code) if code else _extract_text(element)
    return Block(
        index=0,
        type="code",
        bbox=EMPTY_BBOX,
        lines=[_line_with_spans([Span(type="code", bbox=EMPTY_BBOX, content=content)])],
    )
