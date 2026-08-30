# Copyright (c) Opendatalab. All rights reserved.
"""ODT、ODS、ODP 到 MinerU raw model-list 的原生 converter。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, BinaryIO, Iterator

from lxml import etree  # type: ignore[reportMissingImports]

from .....types import BlockType
from ..._shared.spans import inline_span_plain_text, text_spans
from .constants import OdfSuffix, qname
from .models import InlineNote
from .package import OdfPackage
from .styles import OdfStyles
from .table import OdfTableExpansionBudget, parse_table_grid, split_table_regions, table_grid_to_html
from .text import (
    OdfBlockParser,
    OdfMasterPageChange,
    OdfTextExpansionBudget,
    collect_emittable_anchor_targets,
    flatten_block_text,
)


_LENGTH_RE = re.compile(r"^\s*(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+))(?P<unit>cm|mm|in|pt|pc|px)?\s*$")
_LENGTH_TO_PT = {"": 1.0, "pt": 1.0, "pc": 12.0, "in": 72.0, "cm": 72.0 / 2.54, "mm": 72.0 / 25.4, "px": 0.75}


@dataclass(frozen=True, slots=True)
class _OdfContext:
    """保存 converter 一次调用内共享的包、内容树、样式和正文。"""

    package: OdfPackage
    content_root: etree._Element
    styles: OdfStyles
    body: etree._Element


@dataclass(frozen=True, slots=True)
class _PositionedBlocks:
    """保存幻灯片对象的阅读顺序坐标、XML 序号和 raw blocks。"""

    y: float
    x: float
    order: int
    title: bool
    blocks: list[dict[str, Any]]


def _open_context(file_binary: BinaryIO, suffix: OdfSuffix) -> _OdfContext:
    """读取调用方流并建立已验证 ODF 包、样式和正文上下文。"""
    package = OdfPackage(file_binary.read())
    try:
        content_root = package.validate_document(suffix)
        styles_root = package.xml_part("styles.xml")
        styles = OdfStyles(styles_root, content_root)
        return _OdfContext(
            package=package,
            content_root=content_root,
            styles=styles,
            body=package.body_element(content_root, suffix),
        )
    except Exception:
        package.close()
        raise


def _new_page(
    pages: list[list[dict[str, Any]]],
    page_masters: list[str | None],
    master_name: str | None,
) -> None:
    """追加一个新逻辑页及其 master-page 归属。"""
    pages.append([])
    page_masters.append(master_name)


def _flush_notes(parser: OdfBlockParser, page: list[dict[str, Any]]) -> None:
    """把解析器累计脚注追加为当前页面的 PAGE_FOOTNOTE blocks。"""
    for note in parser.drain_notes():
        page.append({"type": BlockType.PAGE_FOOTNOTE, "content": text_spans(note)})


def _append_flow_items(
    items: list[dict[str, Any] | InlineNote],
    *,
    parser: OdfBlockParser,
    page: list[dict[str, Any]],
) -> None:
    """把段落结果和脚注按统一流语义追加到当前章节页。"""
    for item in items:
        if isinstance(item, InlineNote):
            parser.notes.append(item.content)
        else:
            page.append(item)


def _master_auxiliary_blocks(
    master_page: etree._Element | None,
    *,
    package: OdfPackage,
    styles: OdfStyles,
    anchor_targets: frozenset[str],
    text_expansion_budget: OdfTextExpansionBudget,
    table_expansion_budget: OdfTableExpansionBudget,
) -> list[dict[str, Any]]:
    """从 master-page 的 header/footer 中提取页面辅助文本。"""
    if master_page is None:
        return []
    parser = OdfBlockParser(
        package,
        styles,
        anchor_targets=anchor_targets,
        text_expansion_budget=text_expansion_budget,
        table_expansion_budget=table_expansion_budget,
    )
    result: list[dict[str, Any]] = []
    for tag_name, block_type in (("header", BlockType.HEADER), ("footer", BlockType.FOOTER)):
        element = master_page.find(qname("style", tag_name))
        if element is None:
            element = master_page.find(qname("style", f"{tag_name}-left"))
        if element is None:
            continue
        for block in parser.parse_container(element):
            content = block.get("content")
            if isinstance(content, list) and inline_span_plain_text(span for span in content if isinstance(span, dict)).strip():
                result.append({"type": block_type, "content": content})
    return result


def _parse_odt_pages(context: _OdfContext) -> list[list[dict[str, Any]]]:
    """仅按 master-page 章节变化递归构造 ODT 逻辑页。"""
    anchor_targets = collect_emittable_anchor_targets(context.content_root, context.styles)
    text_expansion_budget = OdfTextExpansionBudget()
    parser = OdfBlockParser(
        context.package,
        context.styles,
        anchor_targets=anchor_targets,
        text_expansion_budget=text_expansion_budget,
    )
    pages: list[list[dict[str, Any]]] = [[]]
    page_masters: list[str | None] = [None]
    current_master: str | None = None

    def apply_master_page(requested_master: str | None) -> None:
        """按段落或列表事件切换 ODT 虚拟页及其 master-page。"""
        nonlocal current_master
        master_changed = requested_master is not None and current_master is not None and requested_master != current_master
        if master_changed and pages[-1]:
            _flush_notes(parser, pages[-1])
            _new_page(pages, page_masters, requested_master)
        if requested_master is not None:
            current_master = requested_master
            page_masters[-1] = current_master

    def walk(parent: etree._Element) -> None:
        """递归遍历 ODT block 容器并维护当前页与 master-page。"""
        for child in parent:
            if not isinstance(child.tag, str):
                continue
            if child.tag in {qname("text", "p"), qname("text", "h")}:
                requested_master = context.styles.paragraph_master_page_name(child.get(qname("text", "style-name")))
                apply_master_page(requested_master)
                _append_flow_items(
                    parser.parse_paragraph(child),
                    parser=parser,
                    page=pages[-1],
                )
            elif child.tag in {
                qname("text", "section"),
                qname("text", "index-body"),
                qname("text", "index-title"),
            }:
                walk(child)
            elif child.tag == qname("text", "list"):
                for item in parser.parse_list_blocks(child, emit_master_page_changes=True):
                    if isinstance(item, OdfMasterPageChange):
                        apply_master_page(item.master_page_name)
                    elif isinstance(item, InlineNote):
                        parser.notes.append(item.content)
                    else:
                        pages[-1].append(item)
            else:
                pages[-1].extend(parser.parse_element(child))

    walk(context.body)
    _flush_notes(parser, pages[-1])
    while len(pages) > 1 and not pages[-1]:
        pages.pop()
        page_masters.pop()
    for page, master_name in zip(pages, page_masters, strict=True):
        page.extend(
            _master_auxiliary_blocks(
                context.styles.master_page(master_name),
                package=context.package,
                styles=context.styles,
                anchor_targets=anchor_targets,
                text_expansion_budget=text_expansion_budget,
                table_expansion_budget=parser.table_expansion_budget,
            )
        )
    return pages or [[]]


def _length_to_points(value: str | None) -> float:
    """把 ODF SVG 长度转换为用于阅读顺序比较的 point。"""
    match = _LENGTH_RE.match(value or "")
    if match is None:
        return 0.0
    return float(match.group("value")) * _LENGTH_TO_PT.get(match.group("unit") or "", 1.0)


def _iter_slide_shapes(
    parent: etree._Element,
    *,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
) -> Iterator[tuple[etree._Element, float, float]]:
    """递归展开幻灯片 group，并产出可见 shape 及近似绝对坐标。"""
    for child in parent:
        if not isinstance(child.tag, str) or child.tag == qname("presentation", "notes"):
            continue
        if child.tag == qname("draw", "g"):
            group_x = x_offset + _length_to_points(child.get(qname("svg", "x")))
            group_y = y_offset + _length_to_points(child.get(qname("svg", "y")))
            yield from _iter_slide_shapes(child, x_offset=group_x, y_offset=group_y)
            continue
        if child.tag in {
            qname("draw", "frame"),
            qname("draw", "custom-shape"),
            qname("draw", "rect"),
            qname("draw", "ellipse"),
            qname("draw", "caption"),
        }:
            yield (
                child,
                x_offset + _length_to_points(child.get(qname("svg", "x"))),
                y_offset + _length_to_points(child.get(qname("svg", "y"))),
            )


def _shape_blocks(shape: etree._Element, parser: OdfBlockParser) -> list[dict[str, Any]]:
    """把 frame 或带文本 custom-shape 转为页面 raw blocks。"""
    if shape.tag == qname("draw", "frame"):
        return parser.parse_frame_blocks(shape)
    return parser.parse_container(shape)


def _notes_blocks(page: etree._Element, parser: OdfBlockParser) -> list[dict[str, Any]]:
    """提取 ODP speaker notes，并聚合为页面脚注。"""
    notes = page.find(qname("presentation", "notes"))
    if notes is None:
        return []
    blocks: list[dict[str, Any]] = []
    for frame in notes.iter(qname("draw", "frame")):
        blocks.extend(parser.parse_frame_blocks(frame))
    visible = flatten_block_text(blocks)
    return [{"type": BlockType.PAGE_FOOTNOTE, "content": text_spans(visible)}] if visible else []


def _parse_odp_pages(context: _OdfContext) -> list[list[dict[str, Any]]]:
    """保持一页一 slide，并按坐标和 XML 顺序构造 ODP model-list。"""
    parser = OdfBlockParser(context.package, context.styles)
    pages: list[list[dict[str, Any]]] = []
    document_title_emitted = False
    for page in context.body:
        if page.tag != qname("draw", "page"):
            continue
        if not context.styles.drawing_page_is_visible(page):
            continue
        positioned: list[_PositionedBlocks] = []
        for order, (shape, x, y) in enumerate(_iter_slide_shapes(page)):
            presentation_class = shape.get(qname("presentation", "class"), "")
            if presentation_class in {"page-number", "date-time", "footer", "header"}:
                continue
            blocks = _shape_blocks(shape, parser)
            if not blocks:
                continue
            positioned.append(
                _PositionedBlocks(
                    y=y,
                    x=x,
                    order=order,
                    title=presentation_class in {"title", "subtitle"},
                    blocks=blocks,
                )
            )
        title_entries = sorted((item for item in positioned if item.title), key=lambda item: (item.y, item.x, item.order))
        body_entries = sorted((item for item in positioned if not item.title), key=lambda item: (item.y, item.x, item.order))
        output: list[dict[str, Any]] = []
        for entry in title_entries:
            visible = flatten_block_text(entry.blocks)
            if visible:
                title_type = BlockType.DOC_TITLE if not document_title_emitted else BlockType.PARAGRAPH_TITLE
                output.append(
                    {
                        "type": title_type,
                        "level": 1 if title_type == BlockType.DOC_TITLE else 2,
                        "content": text_spans(visible.replace("\n", " ")),
                    }
                )
                document_title_emitted = True
            output.extend(
                block for block in entry.blocks if block.get("type") in {BlockType.IMAGE, BlockType.TABLE, BlockType.CHART}
            )
        for entry in body_entries:
            output.extend(entry.blocks)
        _flush_notes(parser, output)
        output.extend(_notes_blocks(page, parser))
        _flush_notes(parser, output)
        pages.append(output)
    return pages or [[]]


def _sheet_blocks(sheet: etree._Element, parser: OdfBlockParser) -> list[dict[str, Any]]:
    """把一个可见 ODS sheet 拆为数据区域和锚定视觉对象。"""
    grid = parse_table_grid(sheet, parser.render_cell_html, expansion_budget=parser.table_expansion_budget)
    blocks: list[dict[str, Any]] = []
    for region in split_table_regions(grid):
        content = table_grid_to_html(region)
        if content:
            blocks.append({"type": BlockType.TABLE, "content": content})
    blocks.extend(parser.drain_cell_visuals())
    for shapes in sheet.iter(qname("table", "shapes")):
        for frame in shapes.iter(qname("draw", "frame")):
            for block in parser.parse_frame_blocks(frame):
                if block.get("type") in {BlockType.IMAGE, BlockType.CHART, BlockType.EQUATION}:
                    blocks.append(block)
    _flush_notes(parser, blocks)
    return blocks


def _parse_ods_pages(context: _OdfContext) -> list[list[dict[str, Any]]]:
    """保持一页一可见 sheet，并在多表时添加工作表标题。"""
    parser = OdfBlockParser(context.package, context.styles, collect_cell_visuals=True)
    sheet_pages: list[tuple[str, list[dict[str, Any]]]] = []
    for sheet in context.body:
        if sheet.tag != qname("table", "table"):
            continue
        if sheet.get(qname("table", "display"), "true").casefold() == "false":
            continue
        if not context.styles.table_is_visible(sheet.get(qname("table", "style-name"))):
            continue
        name = sheet.get(qname("table", "name"), "Sheet")
        sheet_pages.append((name, _sheet_blocks(sheet, parser)))
    if sum(bool(blocks) for _, blocks in sheet_pages) > 1:
        for name, blocks in sheet_pages:
            if blocks:
                blocks.insert(0, {"type": BlockType.PARAGRAPH_TITLE, "level": 2, "content": text_spans(name)})
    return [blocks for _, blocks in sheet_pages] or [[]]


class OdtConverter:
    """把 OpenDocument Text 转换为 MinerU 分页 raw blocks。"""

    def __init__(self) -> None:
        """初始化空分页结果，等待 convert 填充。"""
        self.pages: list[list[dict[str, Any]]] = []

    def convert(self, file_binary: BinaryIO) -> None:
        """解析调用方持有的 ODT 流，并保持调用方流打开。"""
        context = _open_context(file_binary, "odt")
        try:
            self.pages = _parse_odt_pages(context)
        finally:
            context.package.close()


class OdpConverter:
    """把 OpenDocument Presentation 转换为逐幻灯片 raw blocks。"""

    def __init__(self) -> None:
        """初始化空幻灯片结果，等待 convert 填充。"""
        self.pages: list[list[dict[str, Any]]] = []

    def convert(self, file_binary: BinaryIO) -> None:
        """解析调用方持有的 ODP 流，并保持调用方流打开。"""
        context = _open_context(file_binary, "odp")
        try:
            self.pages = _parse_odp_pages(context)
        finally:
            context.package.close()


class OdsConverter:
    """把 OpenDocument Spreadsheet 转换为逐可见工作表 raw blocks。"""

    def __init__(self) -> None:
        """初始化空工作表结果，等待 convert 填充。"""
        self.pages: list[list[dict[str, Any]]] = []

    def convert(self, file_binary: BinaryIO) -> None:
        """解析调用方持有的 ODS 流，并保持调用方流打开。"""
        context = _open_context(file_binary, "ods")
        try:
            self.pages = _parse_ods_pages(context)
        finally:
            context.package.close()


__all__ = ["OdpConverter", "OdsConverter", "OdtConverter"]
