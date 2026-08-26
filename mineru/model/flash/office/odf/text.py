# Copyright (c) Opendatalab. All rights reserved.
"""把 ODF 文本、列表、表格和嵌入对象投影为 MinerU raw blocks。"""

from __future__ import annotations

import base64
import html
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from lxml import etree  # type: ignore[reportMissingImports]

from .....types import BlockType
from ..._shared.hyperlink import (
    escape_inline_protocol_text,
    sanitize_hyperlink_target,
)
from ..._shared.mathml import mathml_to_latex
from ..._shared.image import image_to_b64str
from ..image import create_text_placeholder, serialize_office_image
from ..rich_text import OfficeRichTextSegment, build_rich_text_from_segments
from .chart import parse_chart_block
from .constants import MAX_EXPANSION_TEXT_BYTES, qname
from .errors import OdfResourceLimitError
from .models import (
    InlineAtom,
    InlineBlockGroup,
    InlineBreak,
    InlineImage,
    InlineMath,
    InlineNote,
    InlineText,
    TextStyle,
)
from .package import OdfPackage
from .styles import OdfStyles
from .table import OdfTableExpansionBudget, parse_table_grid, table_grid_to_html


_WHITESPACE_RE = re.compile(r"[\t\r\n ]+")
_MAX_EXPLICIT_SPACE_COUNT = 10_000


@dataclass(slots=True)
class OdfTextExpansionBudget:
    """记录单个 ODF 文档显式文本膨胀的累计字节数。"""

    used_bytes: int = 0

    def charge(self, byte_count: int) -> None:
        """在分配膨胀文本前计费，超过固定上限时立即失败。"""
        if byte_count < 0 or self.used_bytes > MAX_EXPANSION_TEXT_BYTES - byte_count:
            raise OdfResourceLimitError(f"ODF resource limit exceeded: max_expansion_text_bytes={MAX_EXPANSION_TEXT_BYTES}")
        self.used_bytes += byte_count


@dataclass(frozen=True, slots=True)
class OdfMasterPageChange:
    """表示列表流中由段落样式请求的 master-page 变化。"""

    master_page_name: str


RawFlowItem = dict[str, Any] | InlineNote
OdfListFlowItem = dict[str, Any] | InlineNote | OdfMasterPageChange


def _clean_xml_text(value: str | None) -> str:
    """折叠 XML 排版空白，显式多空格由 text:s 单独恢复。"""
    if not value:
        return ""
    return _WHITESPACE_RE.sub(" ", value)


def _paragraph_anchor(paragraph: etree._Element) -> str | None:
    """返回段落最终能够挂载到输出 block 的首个 bookmark 名称。"""
    for tag in (qname("text", "bookmark"), qname("text", "bookmark-start")):
        bookmark = next(paragraph.iter(tag), None)
        if bookmark is not None and (name := bookmark.get(qname("text", "name"))):
            return name
    return None


def collect_emittable_anchor_targets(root: etree._Element, styles: OdfStyles) -> frozenset[str]:
    """收集 ODT 标题类 block 实际能够公开的 bookmark target。"""
    targets: set[str] = set()
    for paragraph in root.iter():
        if paragraph.tag not in {qname("text", "p"), qname("text", "h")}:
            continue
        style_name = paragraph.get(qname("text", "style-name"))
        if paragraph.tag != qname("text", "h") and not styles.is_document_title(style_name):
            continue
        if anchor := _paragraph_anchor(paragraph):
            targets.add(anchor)
    return frozenset(targets)


def _style_html(text: str, style: TextStyle) -> str:
    """按稳定顺序把已转义文本包裹为 HTML 行内样式。"""
    rendered = text
    wrappers = [
        (style.bold, "strong"),
        (style.italic, "em"),
        (style.underline, "u"),
        (style.strikethrough, "s"),
        (style.superscript, "sup"),
        (style.subscript, "sub"),
    ]
    for enabled, tag in wrappers:
        if enabled:
            rendered = f"<{tag}>{rendered}</{tag}>"
    return rendered


def _serialize_odf_image(
    image_bytes: bytes,
    *,
    part_name: str | None,
    content_type: str | None,
) -> str | None:
    """序列化 ODF 图片；SVG、SVM 和 GDIMeta 使用安全占位图保留对象位置。"""
    normalized_type = (content_type or "").split(";", 1)[0].strip().casefold()
    suffix = (part_name or "").rsplit(".", 1)[-1].casefold() if "." in (part_name or "") else ""
    if normalized_type == "image/svg+xml" or suffix == "svg":
        placeholder = create_text_placeholder((320, 180), ["SVG image", "Preview unavailable"])
        return image_to_b64str(placeholder, image_format="JPEG")
    if suffix == "svm" or "gdimetafile" in normalized_type or image_bytes.startswith(b"VCLMTF"):
        placeholder = create_text_placeholder((320, 180), ["ODF vector image", "Preview unavailable"])
        return image_to_b64str(placeholder, image_format="JPEG")
    return serialize_office_image(image_bytes, part_name=part_name, content_type=content_type)


def render_atoms_to_html(atoms: Sequence[InlineAtom]) -> str:
    """把 ODF 行内语义序列安全渲染为表格单元格 HTML。"""
    parts: list[str] = []
    for atom in atoms:
        if isinstance(atom, InlineText):
            rendered = _style_html(html.escape(atom.text), atom.style)
            if atom.hyperlink:
                rendered = f'<a href="{html.escape(atom.hyperlink, quote=True)}">{rendered}</a>'
            parts.append(rendered)
        elif isinstance(atom, InlineMath):
            parts.append(f"<eq>{html.escape(atom.latex)}</eq>")
        elif isinstance(atom, InlineBreak):
            parts.append("<br/>")
        elif isinstance(atom, InlineImage):
            parts.append(f'<img src="{html.escape(atom.data_uri, quote=True)}" alt="{html.escape(atom.alt, quote=True)}"/>')
        elif isinstance(atom, (InlineBlockGroup, InlineNote)):
            continue
    return "".join(parts)


def render_atoms_to_model(atoms: Sequence[InlineAtom], *, trim_edges: bool = False) -> str:
    """把 ODF 行内语义序列转换为现有 Office 富文本协议。"""
    parts: list[str] = []
    segments: list[OfficeRichTextSegment] = []
    text_fragments: list[str] = []
    fragment_style: tuple[str, ...] | None = None
    fragment_hyperlink: str | None = None

    def flush_text_fragments() -> None:
        """线性合并连续同样式文本，避免逐片段重复复制前缀。"""
        nonlocal fragment_style, fragment_hyperlink
        if not text_fragments:
            return
        segments.append(OfficeRichTextSegment("".join(text_fragments), fragment_style, fragment_hyperlink))
        text_fragments.clear()
        fragment_style = None
        fragment_hyperlink = None

    def flush_segments() -> None:
        """把连续文本片段批量写入富文本结果。"""
        flush_text_fragments()
        if not segments:
            return
        parts.append(build_rich_text_from_segments(list(segments), trim_plain_edges=trim_edges and not parts))
        segments.clear()

    for atom in atoms:
        if isinstance(atom, InlineText):
            style_names = atom.style.names()
            hyperlink = atom.hyperlink
            if text_fragments and (style_names != fragment_style or hyperlink != fragment_hyperlink):
                flush_text_fragments()
            if not text_fragments:
                fragment_style = style_names
                fragment_hyperlink = hyperlink
            text_fragments.append(atom.text)
            continue
        flush_segments()
        if isinstance(atom, InlineMath):
            parts.append(f"<eq>{html.escape(atom.latex, quote=False)}</eq>")
        elif isinstance(atom, InlineBreak):
            parts.append("\n")
        elif isinstance(atom, InlineImage) and atom.alt:
            parts.append(escape_inline_protocol_text(atom.alt))
        elif isinstance(atom, (InlineBlockGroup, InlineNote)):
            continue
    flush_segments()
    return "".join(parts).strip() if trim_edges else "".join(parts)


class OdfBlockParser:
    """在单个 ODF 包上下文中解析正文、表格和嵌入资源。"""

    def __init__(
        self,
        package: OdfPackage,
        styles: OdfStyles,
        *,
        base_part: str = "content.xml",
        shared_notes: list[str] | None = None,
        list_counters: dict[tuple[str, int], int] | None = None,
        list_ids: dict[str, int] | None = None,
        collect_cell_visuals: bool = False,
        shared_cell_visuals: list[dict[str, Any]] | None = None,
        anchor_targets: frozenset[str] | None = None,
        text_expansion_budget: OdfTextExpansionBudget | None = None,
        table_expansion_budget: OdfTableExpansionBudget | None = None,
    ) -> None:
        """绑定单次解析包、样式、子文档路径及可跨 parser 共享的状态。"""
        self.package = package
        self.styles = styles
        self.base_part = base_part
        self.notes = shared_notes if shared_notes is not None else []
        self._list_counters = list_counters if list_counters is not None else {}
        self._list_ids = list_ids if list_ids is not None else {}
        self._collect_cell_visuals = collect_cell_visuals
        self._cell_visuals = shared_cell_visuals if shared_cell_visuals is not None else []
        self._anchor_targets = anchor_targets or frozenset()
        self._text_expansion_budget = text_expansion_budget or OdfTextExpansionBudget()
        self.table_expansion_budget = table_expansion_budget or OdfTableExpansionBudget()

    def _append_text_atom(
        self,
        atoms: list[InlineAtom],
        value: str | None,
        *,
        style: TextStyle,
        hyperlink: str | None,
        preserve_whitespace: bool = False,
    ) -> None:
        """清理并追加文本节点；显式 ODF 空格可跳过普通 XML 空白折叠。"""
        text = value if preserve_whitespace else _clean_xml_text(value)
        if not text:
            return
        atoms.append(InlineText(text=text, style=style, hyperlink=hyperlink))

    def _walk_inlines(
        self,
        element: etree._Element,
        *,
        style: TextStyle,
        hyperlink: str | None,
        atoms: list[InlineAtom],
    ) -> None:
        """递归遍历段落行内节点，并把 frame 视觉对象旁路为 block。"""
        self._append_text_atom(atoms, element.text, style=style, hyperlink=hyperlink)
        for child in element:
            if not isinstance(child.tag, str):
                self._append_text_atom(atoms, child.tail, style=style, hyperlink=hyperlink)
                continue
            if child.tag == qname("text", "span"):
                span_style = self.styles.text_style(
                    child.get(qname("text", "style-name")),
                    family="text",
                    inherited=style,
                )
                self._walk_inlines(
                    child,
                    style=span_style,
                    hyperlink=hyperlink,
                    atoms=atoms,
                )
            elif child.tag == qname("text", "a"):
                target = sanitize_hyperlink_target(
                    child.get(qname("xlink", "href")),
                    allow_relative=True,
                    allow_fragment=True,
                )
                if target is not None and target.startswith("#") and target[1:] not in self._anchor_targets:
                    target = None
                self._walk_inlines(
                    child,
                    style=style,
                    hyperlink=target or hyperlink,
                    atoms=atoms,
                )
            elif child.tag == qname("text", "s"):
                count = _positive_space_count(child.get(qname("text", "c")))
                self._text_expansion_budget.charge(count)
                self._append_text_atom(
                    atoms,
                    " " * count,
                    style=style,
                    hyperlink=hyperlink,
                    preserve_whitespace=True,
                )
            elif child.tag == qname("text", "tab"):
                self._append_text_atom(atoms, " ", style=style, hyperlink=hyperlink)
            elif child.tag == qname("text", "line-break"):
                atoms.append(InlineBreak())
            elif child.tag == qname("text", "soft-page-break"):
                pass
            elif child.tag == qname("text", "note"):
                self._parse_note(child, style=style, hyperlink=hyperlink, atoms=atoms)
            elif child.tag == qname("office", "annotation"):
                if annotation_text := self._annotation_text(child):
                    atoms.append(InlineNote(annotation_text))
            elif child.tag == qname("office", "annotation-end"):
                pass
            elif child.tag == qname("draw", "frame"):
                inline_atom, blocks = self._parse_frame(child)
                if inline_atom is not None:
                    atoms.append(inline_atom)
                if blocks:
                    atoms.append(
                        InlineBlockGroup(
                            tuple(blocks),
                            inline_image_rendered=isinstance(inline_atom, InlineImage),
                        )
                    )
            elif child.tag == qname("math", "math"):
                if latex := mathml_to_latex(child):
                    atoms.append(InlineMath(latex))
            elif child.tag in {
                qname("text", "bookmark"),
                qname("text", "bookmark-start"),
                qname("text", "bookmark-end"),
            }:
                pass
            else:
                self._walk_inlines(
                    child,
                    style=style,
                    hyperlink=hyperlink,
                    atoms=atoms,
                )
            self._append_text_atom(atoms, child.tail, style=style, hyperlink=hyperlink)

    def _annotation_text(self, annotation: etree._Element) -> str:
        """只提取 ODF annotation 的正文段落与列表，不混入作者日期元数据。"""
        return flatten_block_text(self.parse_container(annotation)).strip()

    def _parse_note(
        self,
        note: etree._Element,
        *,
        style: TextStyle,
        hyperlink: str | None,
        atoms: list[InlineAtom],
    ) -> None:
        """保留脚注标记，并把 note-body 内容排入当前逻辑页脚注队列。"""
        citation = note.find(qname("text", "note-citation"))
        citation_text = (
            "".join(citation.itertext()).strip()
            if citation is not None
            else str(len(self.notes) + sum(isinstance(atom, InlineNote) for atom in atoms) + 1)
        )
        self._append_text_atom(atoms, f"[{citation_text}]", style=style, hyperlink=hyperlink)
        body = note.find(qname("text", "note-body"))
        if body is None:
            return
        blocks = self.parse_container(body)
        visible = flatten_block_text(blocks)
        if visible:
            atoms.append(InlineNote(f"[{citation_text}] {visible}"))

    def parse_inline_atoms(self, paragraph: etree._Element) -> list[InlineAtom]:
        """解析一个段落的行内语义，并用原位 marker 保留段外 block。"""
        paragraph_style = self.styles.text_style(
            paragraph.get(qname("text", "style-name")),
            family="paragraph",
        )
        atoms: list[InlineAtom] = []
        self._walk_inlines(
            paragraph,
            style=paragraph_style,
            hyperlink=None,
            atoms=atoms,
        )
        return atoms

    def parse_paragraph(self, paragraph: etree._Element) -> list[RawFlowItem]:
        """把 text:p/text:h 转为标题、正文、公式和段外内容。"""
        atoms = self.parse_inline_atoms(paragraph)
        results: list[RawFlowItem] = []
        is_heading = paragraph.tag == qname("text", "h")
        style_name = paragraph.get(qname("text", "style-name"))
        content_atoms = [atom for atom in atoms if not isinstance(atom, (InlineBlockGroup, InlineNote))]
        content = render_atoms_to_model(content_atoms, trim_edges=True)
        math_atoms = [atom for atom in content_atoms if isinstance(atom, InlineMath)]
        visible_text = "".join(atom.text for atom in content_atoms if isinstance(atom, InlineText)).strip()
        if content:
            if math_atoms and not visible_text and len(math_atoms) == 1 and len(content_atoms) == 1:
                results.append({"type": BlockType.EQUATION, "content": math_atoms[0].latex})
            elif self.styles.is_document_title(style_name):
                block: dict[str, Any] = {"type": BlockType.DOC_TITLE, "level": 1, "content": content}
                if anchor := _paragraph_anchor(paragraph):
                    block["anchor"] = anchor
                results.append(block)
            elif is_heading:
                try:
                    outline_level = int(paragraph.get(qname("text", "outline-level"), "1"))
                except ValueError:
                    outline_level = 1
                block = {
                    "type": BlockType.PARAGRAPH_TITLE,
                    "level": min(max(outline_level + 1, 2), 6),
                    "is_numbered_style": False,
                    "content": content,
                }
                if anchor := _paragraph_anchor(paragraph):
                    block["anchor"] = anchor
                results.append(block)
            else:
                results.append({"type": BlockType.TEXT, "content": content})
        for atom in atoms:
            if isinstance(atom, InlineBlockGroup):
                results.extend(atom.blocks)
            elif isinstance(atom, InlineNote):
                results.append(atom)
        return results

    def parse_list(
        self,
        element: etree._Element,
        *,
        depth: int = 0,
        inherited_style: str | None = None,
        emit_master_page_changes: bool = False,
    ) -> list[OdfListFlowItem]:
        """递归构造严格 LIST 分片，并把不允许嵌套的 block 提升为有序兄弟。"""
        items = [
            item
            for item in element
            if isinstance(item.tag, str) and item.tag in {qname("text", "list-item"), qname("text", "list-header")}
        ]
        return self._parse_list_items(
            element,
            items,
            depth=depth,
            inherited_style=inherited_style,
            emit_master_page_changes=emit_master_page_changes,
        )

    def _parse_list_items(
        self,
        element: etree._Element,
        items: Sequence[etree._Element],
        *,
        depth: int,
        inherited_style: str | None,
        emit_master_page_changes: bool,
    ) -> list[OdfListFlowItem]:
        """按源条目构造 LIST 分片，每个条目只保留一个文本叶子和一个 marker。"""
        style_name = element.get(qname("text", "style-name")) or inherited_style
        level = self.styles.list_level(style_name, depth)
        key = (style_name or "", depth)
        start = level.start
        continue_list = element.get(qname("text", "continue-list"))
        if continue_list and continue_list in self._list_ids:
            start = self._list_ids[continue_list]
        elif element.get(qname("text", "continue-numbering")) == "true" and key in self._list_counters:
            start = self._list_counters[key]
        results: list[OdfListFlowItem] = []
        content: list[dict[str, Any]] = []
        fragment_notes: list[InlineNote] = []
        item_count = 0
        active_master: str | None = None

        def flush_content(fragment_start: int) -> None:
            """把当前合法子块冻结为一个 LIST 分片。"""
            if content:
                block: dict[str, Any] = {
                    "type": BlockType.LIST,
                    "attribute": "ordered" if level.ordered else "unordered",
                    "ilevel": depth,
                    "content": list(content),
                }
                if level.ordered:
                    block["start"] = fragment_start
                results.append(block)
                content.clear()
            if fragment_notes:
                results.extend(fragment_notes)
                fragment_notes.clear()

        fragment_start = start
        for item in items:
            is_header = item.tag == qname("text", "list-header")
            if not is_header:
                if item_count == 0:
                    try:
                        item_start = int(item.get(qname("text", "start-value"), str(start)))
                        start = max(0, item_start)
                        fragment_start = start
                    except ValueError:
                        pass
                # 统一 LIST 只支持列表级起始值，后续逐项重启按连续序号投影。
                item_count += 1
            first_paragraph = next(
                (
                    child
                    for child in item
                    if isinstance(child.tag, str) and child.tag in {qname("text", "p"), qname("text", "h")}
                ),
                None,
            )
            requested_master = (
                self.styles.paragraph_master_page_name(first_paragraph.get(qname("text", "style-name")))
                if first_paragraph is not None
                else None
            )
            if emit_master_page_changes and requested_master is not None and requested_master != active_master:
                flush_content(fragment_start)
                results.append(OdfMasterPageChange(requested_master))
                active_master = requested_master
                fragment_start = start + item_count - (0 if is_header else 1)
            text_parts: list[str] = []
            nested_blocks: list[dict[str, Any]] = []
            lifted_blocks: list[OdfListFlowItem] = []

            def consume_flow(flow: Sequence[RawFlowItem]) -> None:
                """把段落子流投影到列表文本、嵌套列表或提升块。"""
                for block in flow:
                    if isinstance(block, InlineNote):
                        if emit_master_page_changes:
                            fragment_notes.append(block)
                        else:
                            self.notes.append(block.content)
                        continue
                    block_type = block.get("type")
                    block_content = block.get("content")
                    if block_type in {
                        BlockType.TEXT,
                        BlockType.REF_TEXT,
                        BlockType.DOC_TITLE,
                        BlockType.PARAGRAPH_TITLE,
                    } and isinstance(block_content, str):
                        if block_content:
                            text_parts.append(block_content)
                    elif block_type == BlockType.LIST:
                        nested_blocks.append(block)
                    else:
                        lifted_blocks.append(block)

            for child in item:
                if not isinstance(child.tag, str):
                    continue
                if child.tag in {qname("text", "p"), qname("text", "h")}:
                    consume_flow(self.parse_paragraph(child))
                elif child.tag == qname("text", "list"):
                    nested_flow = self.parse_list_blocks(
                        child,
                        depth=depth + 1,
                        inherited_style=style_name,
                        emit_master_page_changes=emit_master_page_changes,
                    )
                    if not any(isinstance(block, OdfMasterPageChange) for block in nested_flow) and all(
                        isinstance(block, InlineNote) or block.get("type") == BlockType.LIST for block in nested_flow
                    ):
                        nested_blocks.extend(block for block in nested_flow if isinstance(block, dict))
                        fragment_notes.extend(block for block in nested_flow if isinstance(block, InlineNote))
                    else:
                        lifted_blocks.extend(nested_flow)
                else:
                    consume_flow(self.parse_container(child))
            if text_parts:
                content.append({"type": BlockType.TEXT, "content": "\n".join(text_parts)})
            content.extend(nested_blocks)
            if lifted_blocks:
                flush_content(fragment_start)
                results.extend(lifted_blocks)
                fragment_start = start + item_count

        flush_content(fragment_start)
        next_value = start + item_count
        self._list_counters[key] = next_value
        if list_id := element.get(qname("xml", "id")):
            self._list_ids[list_id] = next_value
        return results

    def parse_list_blocks(
        self,
        element: etree._Element,
        *,
        depth: int = 0,
        inherited_style: str | None = None,
        emit_master_page_changes: bool = False,
    ) -> list[OdfListFlowItem]:
        """把含 text:h 的编号章节提升为标题，并保留其余连续列表。"""
        if next(element.iter(qname("text", "h")), None) is None:
            return self.parse_list(
                element,
                depth=depth,
                inherited_style=inherited_style,
                emit_master_page_changes=emit_master_page_changes,
            )
        results: list[OdfListFlowItem] = []
        pending_items: list[etree._Element] = []
        active_master: str | None = None

        def flush_pending() -> None:
            """把标题之间积累的普通列表项写为独立连续 LIST block。"""
            if not pending_items:
                return
            blocks = self._parse_list_items(
                element,
                list(pending_items),
                depth=depth,
                inherited_style=inherited_style,
                emit_master_page_changes=emit_master_page_changes,
            )
            pending_items.clear()
            results.extend(blocks)

        for item in element:
            if not isinstance(item.tag, str) or item.tag not in {
                qname("text", "list-item"),
                qname("text", "list-header"),
            }:
                continue
            if next(item.iter(qname("text", "h")), None) is None:
                pending_items.append(item)
                continue
            flush_pending()
            for child in item:
                if not isinstance(child.tag, str):
                    continue
                if child.tag in {qname("text", "p"), qname("text", "h")}:
                    requested_master = self.styles.paragraph_master_page_name(child.get(qname("text", "style-name")))
                    if emit_master_page_changes and requested_master is not None and requested_master != active_master:
                        results.append(OdfMasterPageChange(requested_master))
                        active_master = requested_master
                    for parsed in self.parse_paragraph(child):
                        if isinstance(parsed, InlineNote):
                            if emit_master_page_changes:
                                results.append(parsed)
                            else:
                                self.notes.append(parsed.content)
                            continue
                        if child.tag == qname("text", "h") and parsed.get("type") == BlockType.PARAGRAPH_TITLE:
                            parsed["is_numbered_style"] = True
                        results.append(parsed)
                elif child.tag == qname("text", "list"):
                    results.extend(
                        self.parse_list_blocks(
                            child,
                            depth=depth + 1,
                            inherited_style=element.get(qname("text", "style-name")) or inherited_style,
                            emit_master_page_changes=emit_master_page_changes,
                        )
                    )
                else:
                    results.extend(self.parse_element(child))
        flush_pending()
        return results

    def _parse_index(self, element: etree._Element) -> dict[str, Any] | None:
        """把 ODF 已存储目录正文转换为扁平 INDEX 子项。"""
        leaves: list[dict[str, Any]] = []
        for paragraph in element.iter():
            if paragraph.tag not in {qname("text", "p"), qname("text", "h")}:
                continue
            for block in self.parse_paragraph(paragraph):
                if isinstance(block, InlineNote):
                    self.notes.append(block.content)
                    continue
                if isinstance(block, dict) and block.get("content"):
                    leaves.append({"type": BlockType.TEXT, "content": block["content"]})
        if not leaves:
            return None
        return {"type": BlockType.INDEX, "ilevel": 0, "content": leaves}

    def parse_table(self, element: etree._Element) -> dict[str, Any] | None:
        """把一个 ODF table 转为包含合并语义的 TABLE raw block。"""
        grid = parse_table_grid(element, self.render_cell_html, expansion_budget=self.table_expansion_budget)
        content = table_grid_to_html(grid)
        return {"type": BlockType.TABLE, "content": content} if content else None

    def _load_image(self, image: etree._Element) -> tuple[str | None, str]:
        """读取 draw:image 的包内或内联载荷并复用 Office 图片序列化。"""
        href = image.get(qname("xlink", "href"), "")
        part_name = self.package.resolve_reference(href, base_part=self.base_part) if href else None
        image_bytes: bytes | None = None
        content_type: str | None = None
        if part_name:
            image_bytes = self.package.read_part(part_name, asset=True)
            content_type = self.package.content_type_for(part_name)
        if image_bytes is None:
            binary = image.find(f".//{qname('office', 'binary-data')}")
            if binary is not None and (binary.text or "").strip():
                try:
                    image_bytes = base64.b64decode("".join((binary.text or "").split()), validate=True)
                except (ValueError, TypeError):
                    image_bytes = None
        alt = ""
        parent = image.getparent()
        if parent is not None:
            title = parent.find(qname("svg", "title"))
            description = parent.find(qname("svg", "desc"))
            alt = " ".join(
                text.strip()
                for text in (
                    "".join(title.itertext()) if title is not None else "",
                    "".join(description.itertext()) if description is not None else "",
                )
                if text.strip()
            )
        if not image_bytes:
            return None, alt
        return _serialize_odf_image(image_bytes, part_name=part_name, content_type=content_type), alt

    def _object_root(self, object_element: etree._Element) -> tuple[etree._Element | None, str | None]:
        """读取 draw:object 指向的子文档内容树和成员路径。"""
        inline_math = next(object_element.iter(qname("math", "math")), None)
        if inline_math is not None:
            return inline_math, self.base_part
        href = object_element.get(qname("xlink", "href"), "")
        part_name = self.package.resolve_object_content(href, base_part=self.base_part)
        if part_name is None:
            return None, None
        return self.package.xml_part(part_name), part_name

    def _parse_frame(self, frame: etree._Element) -> tuple[InlineAtom | None, list[dict[str, Any]]]:
        """按公式、图表、文本框、表格、图片优先级解析一个 draw:frame。"""
        image_element = next(frame.iter(qname("draw", "image")), None)
        preview_uri: str | None = None
        preview_alt = ""

        def load_preview() -> tuple[str | None, str]:
            """只在对象需要图片回退或图表预览时读取 sibling draw:image。"""
            nonlocal preview_uri, preview_alt
            if image_element is not None and preview_uri is None:
                preview_uri, preview_alt = self._load_image(image_element)
            return preview_uri, preview_alt

        object_element = next(frame.iter(qname("draw", "object")), None)
        if object_element is not None:
            object_root, object_part = self._object_root(object_element)
            if object_root is not None:
                math_element = (
                    object_root
                    if object_root.tag == qname("math", "math")
                    else next(
                        object_root.iter(qname("math", "math")),
                        None,
                    )
                )
                if math_element is not None and (latex := mathml_to_latex(math_element)):
                    return InlineMath(latex), []
                load_preview()
                object_parser = OdfBlockParser(
                    self.package,
                    self.styles,
                    base_part=object_part or self.base_part,
                    shared_notes=self.notes,
                    list_counters=self._list_counters,
                    list_ids=self._list_ids,
                    collect_cell_visuals=self._collect_cell_visuals,
                    shared_cell_visuals=self._cell_visuals,
                    anchor_targets=self._anchor_targets,
                    text_expansion_budget=self._text_expansion_budget,
                    table_expansion_budget=self.table_expansion_budget,
                )
                chart = parse_chart_block(
                    object_root,
                    render_cell=object_parser.render_cell_html,
                    preview_data_uri=preview_uri,
                    table_expansion_budget=self.table_expansion_budget,
                )
                if chart is not None:
                    return None, [chart]
        text_box = next(frame.iter(qname("draw", "text-box")), None)
        if text_box is not None:
            return None, self.parse_container(text_box)
        table = next(frame.iter(qname("table", "table")), None)
        if table is not None and (table_block := self.parse_table(table)) is not None:
            return None, [table_block]
        load_preview()
        if preview_uri:
            return InlineImage(preview_uri, preview_alt), [{"type": BlockType.IMAGE, "image_base64": preview_uri}]
        if preview_alt:
            return InlineText(preview_alt), []
        return None, []

    def parse_frame_blocks(self, frame: etree._Element) -> list[dict[str, Any]]:
        """把 frame 的内联结果提升为页面级 block，避免正文重复图片。"""
        inline, blocks = self._parse_frame(frame)
        if blocks:
            return blocks
        if isinstance(inline, InlineMath):
            return [{"type": BlockType.EQUATION, "content": inline.latex}]
        if isinstance(inline, InlineImage):
            return [{"type": BlockType.IMAGE, "image_base64": inline.data_uri}]
        if isinstance(inline, InlineText):
            content = render_atoms_to_model([inline], trim_edges=True)
            return [{"type": BlockType.TEXT, "content": content}] if content else []
        return []

    def parse_element(self, element: etree._Element) -> list[dict[str, Any]]:
        """解析一个 ODF block 元素，不移动或修改原始 XML 节点。"""
        if element.tag in {qname("text", "p"), qname("text", "h")}:
            blocks: list[dict[str, Any]] = []
            for item in self.parse_paragraph(element):
                if isinstance(item, dict):
                    blocks.append(item)
                elif isinstance(item, InlineNote):
                    self.notes.append(item.content)
            return blocks
        if element.tag == qname("text", "list"):
            return [item for item in self.parse_list_blocks(element) if isinstance(item, dict)]
        if element.tag == qname("office", "annotation"):
            if annotation_text := self._annotation_text(element):
                self.notes.append(annotation_text)
            return []
        if element.tag == qname("office", "annotation-end"):
            return []
        if element.tag == qname("table", "table"):
            table_block = self.parse_table(element)
            return [table_block] if table_block is not None else []
        if element.tag == qname("draw", "frame"):
            return self.parse_frame_blocks(element)
        if element.tag in {
            qname("text", "section"),
            qname("text", "index-body"),
            qname("text", "index-title"),
            qname("draw", "g"),
            qname("draw", "custom-shape"),
        }:
            return self.parse_container(element)
        if element.tag in {
            qname("text", "table-of-content"),
            qname("text", "alphabetical-index"),
            qname("text", "bibliography"),
            qname("text", "illustration-index"),
        }:
            index = self._parse_index(element)
            return [index] if index is not None else []
        return []

    def parse_container(self, parent: etree._Element) -> list[dict[str, Any]]:
        """按文档顺序解析普通 ODF block 容器，不建立页面边界。"""
        blocks: list[dict[str, Any]] = []
        for child in parent:
            if isinstance(child.tag, str):
                blocks.extend(self.parse_element(child))
        return blocks

    def _append_cell_blocks(
        self,
        parts: list[str],
        blocks: Sequence[dict[str, Any]],
        *,
        inline_image_rendered: bool,
    ) -> None:
        """按单元格视觉策略收集或内联 block，并避免重复输出配对图片。"""
        for block in blocks:
            block_type = block.get("type")
            if self._collect_cell_visuals and block_type in {
                BlockType.IMAGE,
                BlockType.CHART,
                BlockType.EQUATION,
            }:
                self._cell_visuals.append(block)
                continue
            if inline_image_rendered and block_type == BlockType.IMAGE:
                continue
            if block_type in {BlockType.TABLE, BlockType.CHART} and block.get("content"):
                parts.append(str(block["content"]))
            elif block.get("image_base64"):
                parts.append(f'<img src="{html.escape(str(block["image_base64"]), quote=True)}"/>')

    def _queue_inline_notes(self, atoms: Sequence[InlineAtom]) -> None:
        """把单元格行内流中的 note marker 排入当前逻辑页队列。"""
        self.notes.extend(atom.content for atom in atoms if isinstance(atom, InlineNote))

    def render_cell_html(self, cell: etree._Element) -> str:
        """把表格单元格中的段落、列表、嵌套表和 frame 转为 HTML。"""
        parts: list[str] = []
        for child in cell:
            if not isinstance(child.tag, str):
                continue
            if child.tag in {qname("text", "p"), qname("text", "h")}:
                atoms = self.parse_inline_atoms(child)
                self._queue_inline_notes(atoms)
                rendered_atoms = (
                    [atom for atom in atoms if not isinstance(atom, InlineImage)] if self._collect_cell_visuals else atoms
                )
                parts.append(f"<p>{render_atoms_to_html(rendered_atoms)}</p>")
                for atom in atoms:
                    if isinstance(atom, InlineBlockGroup):
                        self._append_cell_blocks(
                            parts,
                            atom.blocks,
                            inline_image_rendered=atom.inline_image_rendered and not self._collect_cell_visuals,
                        )
            elif child.tag == qname("text", "list"):
                parts.append(self._render_list_html(child))
            elif child.tag == qname("table", "table"):
                nested = parse_table_grid(child, self.render_cell_html, expansion_budget=self.table_expansion_budget)
                parts.append(table_grid_to_html(nested))
            elif child.tag == qname("draw", "frame"):
                inline, blocks = self._parse_frame(child)
                if inline is not None and not (self._collect_cell_visuals and isinstance(inline, InlineImage)):
                    parts.append(render_atoms_to_html([inline]))
                self._append_cell_blocks(
                    parts,
                    blocks,
                    inline_image_rendered=isinstance(inline, InlineImage) and not self._collect_cell_visuals,
                )
        return "".join(part for part in parts if part)

    def _render_list_html(self, element: etree._Element, *, depth: int = 0, inherited_style: str | None = None) -> str:
        """把单元格内 ODF 列表递归渲染为 ol/ul HTML。"""
        style_name = element.get(qname("text", "style-name")) or inherited_style
        level = self.styles.list_level(style_name, depth)
        tag = "ol" if level.ordered else "ul"
        start = f' start="{level.start}"' if level.ordered and level.start != 1 else ""
        parts = [f"<{tag}{start}>"]
        for item in element:
            if item.tag not in {qname("text", "list-item"), qname("text", "list-header")}:
                continue
            if item.tag == qname("text", "list-header"):
                for child in item:
                    if child.tag in {qname("text", "p"), qname("text", "h")}:
                        atoms = self.parse_inline_atoms(child)
                        self._queue_inline_notes(atoms)
                        parts.append(f"<li>{render_atoms_to_html(atoms)}</li>")
                continue
            parts.append("<li>")
            for child in item:
                if child.tag in {qname("text", "p"), qname("text", "h")}:
                    atoms = self.parse_inline_atoms(child)
                    self._queue_inline_notes(atoms)
                    parts.append(render_atoms_to_html(atoms))
                elif child.tag == qname("text", "list"):
                    parts.append(self._render_list_html(child, depth=depth + 1, inherited_style=style_name))
            parts.append("</li>")
        parts.append(f"</{tag}>")
        return "".join(parts)

    def drain_notes(self) -> list[str]:
        """取出当前累计脚注并清空共享队列。"""
        values = list(self.notes)
        self.notes.clear()
        return values

    def drain_cell_visuals(self) -> list[dict[str, Any]]:
        """取出 ODS 单元格解析期间收集的视觉对象并清空队列。"""
        values = list(self._cell_visuals)
        self._cell_visuals.clear()
        return values


def _positive_space_count(value: str | None) -> int:
    """在整数转换前校验并限制 text:s 重复空格数，非法值按一处理。"""
    normalized = (value or "").strip()
    if normalized.startswith("+"):
        normalized = normalized[1:]
    if not normalized or not normalized.isascii() or not normalized.isdigit():
        return 1
    significant = normalized.lstrip("0")
    if not significant:
        return 1
    max_digits = len(str(_MAX_EXPLICIT_SPACE_COUNT))
    if len(significant) > max_digits:
        return _MAX_EXPLICIT_SPACE_COUNT
    return min(max(1, int(significant)), _MAX_EXPLICIT_SPACE_COUNT)


def flatten_block_text(blocks: list[dict[str, Any]]) -> str:
    """递归提取 raw block 的可见字符串，供标题和备注聚合。"""
    parts: list[str] = []
    for block in blocks:
        content = block.get("content")
        if isinstance(content, str):
            visible = re.sub(r"<url>.*?</url>", "", content, flags=re.DOTALL)
            visible = re.sub(r"<[^>]+>", "", visible)
            if visible.strip():
                decoded = html.unescape(visible).strip()
                parts.append(decoded.replace("<", "&lt;").replace(">", "&gt;"))
        elif isinstance(content, list):
            children = [child for child in content if isinstance(child, dict)]
            nested = flatten_block_text(children)
            if nested:
                parts.append(nested)
    return "\n".join(parts)


__all__ = [
    "OdfBlockParser",
    "OdfMasterPageChange",
    "OdfTextExpansionBudget",
    "RawFlowItem",
    "flatten_block_text",
    "render_atoms_to_html",
    "render_atoms_to_model",
]
