# Copyright (c) Opendatalab. All rights reserved.
"""把 typed RTF 语义文档转换为 MinerU 单逻辑页 raw model-list。"""

from __future__ import annotations

from dataclasses import dataclass, field
from html import escape
from typing import Any, BinaryIO, Iterable

from .....types import BlockType
from ..image import (
    ensure_bmp_header,
    is_valid_vector_image_payload,
    is_vector_image_part,
    serialize_office_image,
)
from ..equation.image import OfficeImageEquationDecoder
from ..rich_text import OfficeRichTextSegment, build_rich_text_from_segments
from .models import (
    RtfAnchor,
    RtfBlock,
    RtfDisplayEquation,
    RtfDocument,
    RtfImage,
    RtfInline,
    RtfInlineEquation,
    RtfLineBreak,
    RtfNoteReference,
    RtfParagraph,
    RtfTable,
    RtfTableCell,
    RtfTextRun,
)
from .parser import MAX_RTF_LIST_DEPTH, parse_rtf, parse_rtf_prelude, read_rtf_bytes


@dataclass(slots=True)
class _GridOrigin:
    """保存 HTML table 网格中一个 origin cell 的 span 状态。"""

    row: int
    col: int
    cell: RtfTableCell
    row_span: int = 1
    col_span: int = 1


@dataclass(slots=True)
class _HtmlListNode:
    """保存表格单元格内列表段落的临时层级树。"""

    paragraph: RtfParagraph
    children: list[_HtmlListNode] = field(default_factory=list)


def _style_names(run: RtfTextRun) -> list[str]:
    """把 RTF 字符属性映射为现有 Office 富文本样式名。"""
    result: list[str] = []
    if run.style.bold:
        result.append("bold")
    if run.style.italic:
        result.append("italic")
    if run.style.underline:
        result.append("underline")
    if run.style.strike:
        result.append("strikethrough")
    if run.style.superscript:
        result.append("superscript")
    elif run.style.subscript:
        result.append("subscript")
    return result


def _paragraph_anchor(paragraph: RtfParagraph) -> str | None:
    """返回段落内第一个非空 bookmark 名。"""
    for inline in paragraph.inlines:
        if isinstance(inline, RtfAnchor) and inline.name.strip():
            return inline.name.strip()
    return None


def _iter_nested_blocks(blocks: Iterable[RtfBlock]) -> Iterable[RtfBlock]:
    """按深度优先顺序遍历正文及嵌套表格单元格块。"""
    pending = list(reversed(list(blocks)))
    while pending:
        block = pending.pop()
        yield block
        if isinstance(block, RtfTable):
            for row in reversed(block.rows):
                for cell in reversed(row.cells):
                    pending.extend(reversed(cell.blocks))


def _title_anchors(document: RtfDocument) -> set[str]:
    """收集 schema 能公开的标题 bookmark，普通段落 bookmark 不生成悬空链接。"""
    result: set[str] = set()
    for block in _iter_nested_blocks(document.blocks):
        if not isinstance(block, RtfParagraph):
            continue
        if not block.is_title and block.outline_level is None:
            continue
        anchor = _paragraph_anchor(block)
        if anchor:
            result.add(anchor)
    return result


def _note_numbers(document: RtfDocument) -> dict[str, int]:
    """按第一次引用顺序编号 note，未引用 note 稳定追加在末尾。"""
    numbers: dict[str, int] = {}

    def visit_blocks(blocks: Iterable[RtfBlock]) -> None:
        """扫描一组块中的 note reference。"""
        for block in _iter_nested_blocks(blocks):
            if not isinstance(block, RtfParagraph):
                continue
            for inline in block.inlines:
                if isinstance(inline, RtfNoteReference) and inline.note_id not in numbers:
                    numbers[inline.note_id] = len(numbers) + 1

    visit_blocks(document.blocks)
    visit_blocks(document.headers)
    visit_blocks(document.footers)
    for note in document.notes:
        visit_blocks(note.blocks)
    for note in document.notes:
        numbers.setdefault(note.id, len(numbers) + 1)
    return numbers


def _plain_inlines(inlines: Iterable[RtfInline], note_numbers: dict[str, int]) -> str:
    """提取行内节点可见文本，供代码块、alt 和脚注降级使用。"""
    parts: list[str] = []
    for inline in inlines:
        if isinstance(inline, RtfTextRun):
            parts.append(inline.text)
        elif isinstance(inline, RtfInlineEquation):
            parts.append(inline.latex)
        elif isinstance(inline, RtfImage):
            parts.append(inline.alt)
        elif isinstance(inline, RtfNoteReference):
            number = note_numbers.get(inline.note_id)
            if number is not None:
                parts.append(f"[{number}]")
        elif isinstance(inline, RtfLineBreak):
            parts.append("\n")
    return "".join(parts)


class RtfConverter:
    """把 RTF typed IR 投影为现有 Office raw-block 协议。"""

    def __init__(self) -> None:
        """初始化空输出和每文档图片公式 decoder。"""
        self.pages: list[list[dict[str, Any]]] = []
        self.document: RtfDocument | None = None
        self._note_numbers: dict[str, int] = {}
        self._title_anchors: set[str] = set()
        self._image_equations = OfficeImageEquationDecoder()

    def convert(self, file_binary: BinaryIO) -> None:
        """解析 RTF 二进制流并生成固定单逻辑页 model-list。"""
        document = parse_rtf(file_binary)
        self.document = document
        self._note_numbers = _note_numbers(document)
        self._title_anchors = _title_anchors(document)
        page = self._document_blocks(document.blocks)
        page.extend(self._auxiliary_blocks(document.headers, BlockType.HEADER))
        page.extend(self._auxiliary_blocks(document.footers, BlockType.FOOTER))
        page.extend(self._note_blocks(document))
        self.pages = [page]

    def _resolved_hyperlink(self, target: str | None) -> str | None:
        """只保留指向可公开标题 bookmark 的内部链接。"""
        if not target:
            return None
        if target.startswith("#") and target[1:] not in self._title_anchors:
            return None
        return target

    def _rich_text(self, inlines: Iterable[RtfInline]) -> str:
        """把非图片行内节点转换为现有 Office 富文本和公式协议。"""
        parts: list[str] = []
        segments: list[OfficeRichTextSegment] = []

        def flush() -> None:
            """在公式边界前输出累计普通富文本。"""
            if not segments:
                return
            parts.append(build_rich_text_from_segments(segments, trim_plain_edges=not parts))
            segments.clear()

        for inline in inlines:
            if isinstance(inline, RtfTextRun):
                hyperlink = self._resolved_hyperlink(inline.hyperlink)
                segments.append(
                    OfficeRichTextSegment(
                        text=escape(inline.text, quote=False),
                        style=_style_names(inline),
                        hyperlink=escape(hyperlink, quote=False) if hyperlink is not None else None,
                    )
                )
            elif isinstance(inline, RtfInlineEquation):
                flush()
                parts.append(f"<eq>{escape(inline.latex, quote=False)}</eq>")
            elif isinstance(inline, RtfNoteReference):
                number = self._note_numbers.get(inline.note_id)
                if number is not None:
                    segments.append(
                        OfficeRichTextSegment(
                            text=f"[{number}]",
                            style="superscript",
                        )
                    )
            elif isinstance(inline, RtfLineBreak):
                segments.append(OfficeRichTextSegment(text="\n"))
        flush()
        return "".join(parts).strip()

    def _image_payload(self, image: RtfImage) -> tuple[bytes, str, str] | None:
        """规范 DIB 载荷并返回图片数据、part name 和 content type。"""
        if image.part_name.lower().endswith(".dib"):
            return ensure_bmp_header(image.data), "pict.bmp", "image/bmp"
        if is_vector_image_part(image.part_name, image.content_type) and not is_valid_vector_image_payload(
            image.data,
            part_name=image.part_name,
            content_type=image.content_type,
        ):
            return None
        return image.data, image.part_name, image.content_type

    def _image_block(self, image: RtfImage) -> dict[str, Any] | None:
        """优先恢复图片 MTEF 公式，否则序列化为安全图片 data URI。"""
        normalized = self._image_payload(image)
        if normalized is None:
            if image.alt.strip():
                return {"type": BlockType.TEXT, "content": image.alt.strip()}
            return None
        payload, part_name, content_type = normalized
        latex = self._image_equations.decode(
            payload,
            part_name=part_name,
            content_type=content_type,
        )
        if latex:
            return {"type": BlockType.EQUATION, "content": latex}
        image_base64 = serialize_office_image(
            payload,
            part_name=part_name,
            content_type=content_type,
        )
        if image_base64 is None:
            if image.alt.strip():
                return {"type": BlockType.TEXT, "content": image.alt.strip()}
            return None
        block: dict[str, Any] = {
            "type": BlockType.IMAGE,
            "content": "",
            "image_base64": image_base64,
        }
        if image.alt.strip():
            block["sub_type"] = image.alt.strip()
        return block

    def _paragraph_text_block(
        self,
        paragraph: RtfParagraph,
        inlines: list[RtfInline],
        *,
        allow_title: bool,
    ) -> dict[str, Any] | None:
        """把一个不含图片的段落片段投影为标题、代码或正文 raw block。"""
        if paragraph.block_style == "code":
            content = _plain_inlines(inlines, self._note_numbers).strip("\n")
            return {"type": BlockType.CODE, "content": content} if content else None
        content = self._rich_text(inlines)
        if not content:
            return None
        if allow_title and paragraph.is_title:
            block: dict[str, Any] = {
                "type": BlockType.DOC_TITLE,
                "level": 1,
                "content": content,
            }
        elif allow_title and paragraph.outline_level is not None:
            block = {
                "type": BlockType.PARAGRAPH_TITLE,
                "level": min(max(paragraph.outline_level + 2, 2), 6),
                "is_numbered_style": False,
                "content": content,
            }
        else:
            block = {"type": BlockType.TEXT, "content": content}
        if block["type"] in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE}:
            anchor = _paragraph_anchor(paragraph)
            if anchor:
                block["anchor"] = anchor
        return block

    def _paragraph_blocks(self, paragraph: RtfParagraph) -> list[dict[str, Any]]:
        """按行内图片位置拆分段落，并只让首个文本片段继承标题类型。"""
        if paragraph.list_info is not None and (paragraph.is_title or paragraph.outline_level is not None):
            label = paragraph.list_info.label if paragraph.list_info.ordered else None
            if label:
                paragraph = RtfParagraph(
                    inlines=[RtfTextRun(f"{label} "), *paragraph.inlines],
                    style_name=paragraph.style_name,
                    outline_level=paragraph.outline_level,
                    is_title=paragraph.is_title,
                    block_style=paragraph.block_style,
                )

        non_image = [inline for inline in paragraph.inlines if not isinstance(inline, (RtfImage, RtfAnchor))]
        equations = [inline for inline in non_image if isinstance(inline, RtfInlineEquation)]
        ordinary = [
            inline
            for inline in non_image
            if not isinstance(inline, (RtfInlineEquation, RtfLineBreak, RtfNoteReference))
            and (not isinstance(inline, RtfTextRun) or bool(inline.text.strip()))
        ]
        if equations and not ordinary:
            return [{"type": BlockType.EQUATION, "content": equation.latex} for equation in equations]

        blocks: list[dict[str, Any]] = []
        current: list[RtfInline] = []
        text_emitted = False

        def flush() -> None:
            """输出当前图片边界前累计的段落片段。"""
            nonlocal text_emitted
            block = self._paragraph_text_block(
                paragraph,
                current,
                allow_title=not text_emitted,
            )
            current.clear()
            if block is not None:
                blocks.append(block)
                text_emitted = True

        for inline in paragraph.inlines:
            if isinstance(inline, RtfImage):
                flush()
                image_block = self._image_block(inline)
                if image_block is not None:
                    blocks.append(image_block)
                continue
            current.append(inline)
        flush()
        return blocks

    def _append_list_item(
        self,
        page: list[dict[str, Any]],
        stack: list[dict[str, Any]],
        identity: int | None,
        paragraph: RtfParagraph,
    ) -> int:
        """把一个 RTF 列表段落追加到嵌套 raw list 树。"""
        info = paragraph.list_info
        if info is None:
            return identity or -1
        content = self._rich_text(
            inline for inline in paragraph.inlines if not isinstance(inline, (RtfImage, RtfAnchor))
        )
        if identity != info.identity:
            stack.clear()
            identity = info.identity
        level = min(max(info.level, 0), MAX_RTF_LIST_DEPTH)
        while len(stack) > level + 1:
            stack.pop()
        while len(stack) < level + 1:
            list_block: dict[str, Any] = {
                "type": BlockType.LIST,
                "attribute": "ordered" if info.ordered else "unordered",
                "ilevel": len(stack),
                "content": [],
            }
            if info.ordered:
                list_block["start"] = info.start
            if stack:
                stack[-1]["content"].append(list_block)
            else:
                page.append(list_block)
            stack.append(list_block)
        current = stack[level]
        expected = "ordered" if info.ordered else "unordered"
        if current.get("attribute") != expected:
            del stack[level:]
            return self._append_list_item(page, stack, None, paragraph)
        if content:
            leaf: dict[str, Any] = {"type": BlockType.TEXT, "content": content}
            if info.ordered and info.label:
                leaf["list_label"] = info.label
            current["content"].append(leaf)
        return info.identity

    def _document_blocks(self, blocks: Iterable[RtfBlock]) -> list[dict[str, Any]]:
        """按源顺序转换正文块，并维护顶层列表连续性。"""
        page: list[dict[str, Any]] = []
        list_stack: list[dict[str, Any]] = []
        list_identity: int | None = None
        source_blocks = list(blocks)
        index = 0
        while index < len(source_blocks):
            block = source_blocks[index]
            if isinstance(block, RtfParagraph) and block.block_style == "code":
                list_stack.clear()
                list_identity = None
                lines: list[str] = []
                images: list[RtfImage] = []
                while index < len(source_blocks):
                    candidate = source_blocks[index]
                    if not isinstance(candidate, RtfParagraph) or candidate.block_style != "code":
                        break
                    lines.append(_plain_inlines(candidate.inlines, self._note_numbers).rstrip("\n"))
                    images.extend(inline for inline in candidate.inlines if isinstance(inline, RtfImage))
                    index += 1
                content = "\n".join(lines).strip("\n")
                if content:
                    page.append({"type": BlockType.CODE, "content": content})
                for image in images:
                    image_block = self._image_block(image)
                    if image_block is not None:
                        page.append(image_block)
                continue
            if isinstance(block, RtfParagraph) and block.list_info is not None and not (
                block.is_title or block.outline_level is not None
            ):
                list_identity = self._append_list_item(page, list_stack, list_identity, block)
                images = [inline for inline in block.inlines if isinstance(inline, RtfImage)]
                if images:
                    list_stack.clear()
                    list_identity = None
                    for image in images:
                        image_block = self._image_block(image)
                        if image_block is not None:
                            page.append(image_block)
                index += 1
                continue
            list_stack.clear()
            list_identity = None
            if isinstance(block, RtfParagraph):
                page.extend(self._paragraph_blocks(block))
            elif isinstance(block, RtfDisplayEquation):
                if block.latex.strip():
                    page.append({"type": BlockType.EQUATION, "content": block.latex.strip()})
            elif isinstance(block, RtfTable):
                page.append({"type": BlockType.TABLE, "content": self._table_html(block)})
            index += 1
        return page

    def _inline_html(self, inlines: Iterable[RtfInline]) -> str:
        """把表格单元格行内节点转换为白名单 HTML。"""
        parts: list[str] = []
        for inline in inlines:
            if isinstance(inline, RtfTextRun):
                content = escape(inline.text, quote=False).replace("\n", "<br>")
                if inline.style.code:
                    content = f"<code>{content}</code>"
                if inline.style.superscript:
                    content = f"<sup>{content}</sup>"
                elif inline.style.subscript:
                    content = f"<sub>{content}</sub>"
                if inline.style.underline:
                    content = f"<u>{content}</u>"
                if inline.style.bold:
                    content = f"<strong>{content}</strong>"
                if inline.style.italic:
                    content = f"<em>{content}</em>"
                if inline.style.strike:
                    content = f"<s>{content}</s>"
                target = self._resolved_hyperlink(inline.hyperlink)
                if target:
                    content = f'<a href="{escape(target, quote=True)}">{content}</a>'
                parts.append(content)
            elif isinstance(inline, RtfInlineEquation):
                parts.append(f"<eq>{escape(inline.latex, quote=False)}</eq>")
            elif isinstance(inline, RtfLineBreak):
                parts.append("<br>")
            elif isinstance(inline, RtfNoteReference):
                number = self._note_numbers.get(inline.note_id)
                if number is not None:
                    parts.append(f"<sup>[{number}]</sup>")
            elif isinstance(inline, RtfImage):
                normalized = self._image_payload(inline)
                if normalized is None:
                    if inline.alt:
                        parts.append(escape(inline.alt, quote=False))
                    continue
                payload, part_name, content_type = normalized
                latex = self._image_equations.decode(
                    payload,
                    part_name=part_name,
                    content_type=content_type,
                )
                if latex:
                    parts.append(f"<eq>{escape(latex, quote=False)}</eq>")
                    continue
                source = serialize_office_image(
                    payload,
                    part_name=part_name,
                    content_type=content_type,
                )
                if source:
                    parts.append(
                        f'<img src="{escape(source, quote=True)}" alt="{escape(inline.alt, quote=True)}">'
                    )
                elif inline.alt:
                    parts.append(escape(inline.alt, quote=False))
        return "".join(parts)

    def _list_tree(self, paragraphs: list[RtfParagraph]) -> list[_HtmlListNode]:
        """把连续列表段落构造成单元格 HTML 使用的嵌套树。"""
        roots: list[_HtmlListNode] = []
        stack: list[_HtmlListNode] = []
        for paragraph in paragraphs:
            level = min(paragraph.list_info.level if paragraph.list_info else 0, len(stack))
            while len(stack) > level:
                stack.pop()
            node = _HtmlListNode(paragraph)
            if level > 0 and stack:
                stack[-1].children.append(node)
            else:
                roots.append(node)
            stack.append(node)
        return roots

    def _list_nodes_html(self, nodes: list[_HtmlListNode]) -> str:
        """递归序列化一层单元格列表节点。"""
        if not nodes:
            return ""
        info = nodes[0].paragraph.list_info
        ordered = bool(info and info.ordered)
        tag = "ol" if ordered else "ul"
        start = f' start="{max(info.start, 0)}"' if ordered and info else ""
        items: list[str] = []
        for node in nodes:
            content = self._inline_html(node.paragraph.inlines)
            nested = self._list_nodes_html(node.children)
            items.append(f"<li>{content}{nested}</li>")
        return f"<{tag}{start}>{''.join(items)}</{tag}>"

    def _blocks_html(self, blocks: list[RtfBlock]) -> str:
        """序列化 table cell 内允许的段落、列表、代码、引用和嵌套表格。"""
        parts: list[str] = []
        index = 0
        while index < len(blocks):
            block = blocks[index]
            if isinstance(block, RtfParagraph) and block.list_info is not None:
                run: list[RtfParagraph] = []
                identity = block.list_info.identity
                while index < len(blocks):
                    candidate = blocks[index]
                    if not isinstance(candidate, RtfParagraph) or candidate.list_info is None:
                        break
                    if candidate.list_info.identity != identity:
                        break
                    run.append(candidate)
                    index += 1
                parts.append(self._list_nodes_html(self._list_tree(run)))
                continue
            if isinstance(block, RtfParagraph):
                content = self._inline_html(block.inlines)
                if block.block_style == "code":
                    parts.append(f"<pre><code>{escape(_plain_inlines(block.inlines, self._note_numbers))}</code></pre>")
                elif block.block_style == "quote":
                    parts.append(f"<blockquote>{content}</blockquote>")
                else:
                    parts.append(f"<p>{content}</p>")
            elif isinstance(block, RtfDisplayEquation):
                parts.append(f"<p><eq>{escape(block.latex, quote=False)}</eq></p>")
            elif isinstance(block, RtfTable):
                parts.append(self._table_html(block))
            index += 1
        return "".join(parts)

    def _table_grid(self, table: RtfTable) -> list[list[_GridOrigin | None]]:
        """解析横向与纵向 merge continuation，生成 exactly-once origin 网格。"""
        boundaries = sorted(
            {
                cell.right_boundary
                for row in table.rows
                for cell in row.cells
                if cell.right_boundary is not None
            }
        )
        width = len(boundaries) or max((len(row.cells) for row in table.rows), default=0)
        boundary_index = {value: index for index, value in enumerate(boundaries)}
        grid: list[list[_GridOrigin | None]] = []
        for row_index, row in enumerate(table.rows):
            slots: list[_GridOrigin | None] = [None] * width
            previous_end = -1
            for fallback_col, cell in enumerate(row.cells):
                right_boundary = cell.right_boundary
                end_col = boundary_index.get(right_boundary, fallback_col) if right_boundary is not None else fallback_col
                start_col = previous_end + 1
                if end_col < start_col:
                    end_col = start_col
                end_col = min(end_col, width - 1)
                previous_end = end_col
                above = grid[row_index - 1][start_col] if row_index > 0 and start_col < width else None
                left = slots[start_col - 1] if start_col > 0 else None
                has_content = bool(self._blocks_plain_text(cell.blocks).strip())
                if cell.vertical_merge == "continue" and above is not None and not has_content:
                    origin = above
                    origin.row_span = max(origin.row_span, row_index - origin.row + 1)
                elif cell.horizontal_merge == "continue" and left is not None and not has_content:
                    origin = left
                    origin.col_span = max(origin.col_span, end_col - origin.col + 1)
                else:
                    origin = _GridOrigin(
                        row_index,
                        start_col,
                        cell,
                        col_span=max(end_col - start_col + 1, 1),
                    )
                for col_index in range(start_col, end_col + 1):
                    slots[col_index] = origin
            for col_index, origin in enumerate(slots):
                if origin is None:
                    slots[col_index] = _GridOrigin(row_index, col_index, RtfTableCell())
            grid.append(slots)
        return grid

    def _table_html(self, table: RtfTable) -> str:
        """把 RTF 表格输出为带 rowspan/colspan 的安全 HTML。"""
        grid = self._table_grid(table)
        rows: list[str] = []
        for row_index, slots in enumerate(grid):
            tag = "th" if row_index < len(table.rows) and table.rows[row_index].header else "td"
            cells: list[str] = []
            for col_index, origin in enumerate(slots):
                if origin is None or origin.row != row_index or origin.col != col_index:
                    continue
                attributes: list[str] = []
                if origin.row_span > 1:
                    attributes.append(f'rowspan="{origin.row_span}"')
                if origin.col_span > 1:
                    attributes.append(f'colspan="{origin.col_span}"')
                suffix = f" {' '.join(attributes)}" if attributes else ""
                cells.append(f"<{tag}{suffix}>{self._blocks_html(origin.cell.blocks)}</{tag}>")
            rows.append(f"<tr>{''.join(cells)}</tr>")
        return f"<table>{''.join(rows)}</table>"

    def _auxiliary_blocks(self, blocks: list[RtfBlock], block_type: BlockType) -> list[dict[str, Any]]:
        """把页眉页脚段落去重后投影为页面辅助块。"""
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for block in blocks:
            if isinstance(block, RtfParagraph):
                content = self._rich_text(block.inlines)
            elif isinstance(block, RtfDisplayEquation):
                content = f"<eq>{escape(block.latex, quote=False)}</eq>"
            elif isinstance(block, RtfTable):
                content = escape(self._table_plain_text(block), quote=False)
            else:
                continue
            if content and not content.isdigit() and content not in seen:
                seen.add(content)
                result.append({"type": block_type, "content": content})
        return result

    def _table_plain_text(self, table: RtfTable) -> str:
        """把表格可见文本压平，供注释和辅助块无损降级。"""
        rows: list[str] = []
        for row in table.rows:
            cells = [self._blocks_plain_text(cell.blocks) for cell in row.cells]
            rows.append(" | ".join(cells))
        return "\n".join(rows)

    def _blocks_plain_text(self, blocks: Iterable[RtfBlock]) -> str:
        """提取块列表可见文本，保持段落和表格行边界。"""
        parts: list[str] = []
        for block in blocks:
            if isinstance(block, RtfParagraph):
                parts.append(_plain_inlines(block.inlines, self._note_numbers))
            elif isinstance(block, RtfDisplayEquation):
                parts.append(block.latex)
            elif isinstance(block, RtfTable):
                parts.append(self._table_plain_text(block))
        return "\n".join(part for part in parts if part.strip())

    def _note_blocks(self, document: RtfDocument) -> list[dict[str, Any]]:
        """按公开编号输出脚注与尾注正文。"""
        result: list[dict[str, Any]] = []
        ordered = sorted(
            document.notes,
            key=lambda note: self._note_numbers[note.id] if note.id in self._note_numbers else 2**31 - 1,
        )
        for note in ordered:
            number = self._note_numbers.get(note.id)
            content = self._blocks_plain_text(note.blocks).strip()
            if number is None or not content:
                continue
            result.append(
                {
                    "type": BlockType.PAGE_FOOTNOTE,
                    "content": f"[{number}] {content}",
                }
            )
        return result


def extract_rtf_metadata(file_binary: BinaryIO) -> dict[str, str | None]:
    """有界读取 RTF，仅解析 info destination 并返回 doclib 字段。"""
    data = read_rtf_bytes(file_binary)
    metadata = parse_rtf_prelude(data).metadata
    return {
        "title": metadata.title,
        "author": metadata.author,
        "subject": metadata.subject,
        "keywords": metadata.keywords,
    }


__all__ = ["RtfConverter", "extract_rtf_metadata"]
