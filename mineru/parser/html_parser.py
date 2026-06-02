# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup, Tag

from .base import DocumentParser
from .parse_result import ParseResult
from .types import Block, Line, PageInfo, Span


class HtmlParser(DocumentParser):
    """Parse HTML files into structured ``ParseResult``.

    HTML elements are mapped to typed blocks inside a single page.
    """

    def parse(self, path: str | Path) -> ParseResult:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)

        file_name = path.stem
        html_text = path.read_text(encoding="utf-8")
        soup = BeautifulSoup(html_text, "html.parser")

        body = soup.body or soup
        blocks: list[Block] = []
        _walk_elements(body, blocks)

        return ParseResult(
            pages=[PageInfo(page_idx=0, para_blocks=blocks)],
            _backend="html",
            _version_name=_version(),
            _file_name=file_name,
        )


_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_BLOCK_CONTAINER_TAGS = frozenset({
    "div", "section", "article", "aside", "main", "header", "footer", "nav",
})


def _heading_level(tag_name: str) -> int:
    return int(tag_name[1])


def _extract_text(element: Tag) -> str:
    return element.get_text(separator=" ", strip=True)


def _build_span(element: Tag) -> Span:
    content = _extract_text(element)
    img = element.find("img")
    if img is not None:
        src: str = img.get("src", "") or ""
        if src.startswith("data:image"):
            return Span(type="image", image_base64=src, content=content)
        return Span(type="image", image_path=src, content=content)
    return Span(type="text", content=content)


def _walk_elements(parent: Tag, blocks: list[Block]) -> None:  # type: ignore[type-arg]
    for child in parent.children:
        if not isinstance(child, Tag):
            text = str(child).strip()
            if text:
                blocks.append(Block(
                    type="text",
                    lines=[Line(spans=[Span(type="text", content=text)])],
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
        type="title",
        level=level,
        lines=[Line(spans=[_build_span(element)])],
    )


def _build_text_block(element: Tag) -> Block:
    # Collapse nested inline children into a single span.
    return Block(
        type="text",
        lines=[Line(spans=[_build_span(element)])],
    )


def _build_image_block(element: Tag) -> Block:
    span = _build_span(element)
    return Block(
        type="image",
        lines=[Line(spans=[span])],
    )


def _build_list(element: Tag) -> Block:
    items: list[Block] = []
    for li in element.find_all("li", recursive=False):
        nested_blocks: list[Block] = []
        _walk_elements(li, nested_blocks)
        items.append(Block(
            type="list_item",
            lines=[Line(spans=[Span(type="text", content=_extract_text(li))])],
            blocks=nested_blocks,
        ))
    return Block(type="list", blocks=items)


def _build_table(element: Tag) -> Block:
    return Block(
        type="table",
        html=str(element),
        lines=[Line(spans=[Span(type="table", html=str(element))])],
    )


def _build_code(element: Tag) -> Block:
    code = element.find("code")
    content = _extract_text(code) if code else _extract_text(element)
    return Block(
        type="code",
        lines=[Line(spans=[Span(type="code", content=content)])],
    )


def _version() -> str:
    from ..version import __version__
    return __version__
