# Copyright (c) Opendatalab. All rights reserved.

"""把旧版 PPT 内部语义模型转换为 MinerU 分页 model-list。"""

from __future__ import annotations

from html import escape
from typing import Any, BinaryIO

from mineru.model.flash.legacy_office import BoundedOleReader
from mineru.model.flash.xycut import sort_entries
from mineru.model.office_stream import read_stream_bytes_from_start
from mineru.types import BlockType
from mineru.utils.office_rich_text import OfficeRichTextSegment, build_rich_text_from_segments

from .models import (
    PptEquationElement,
    PptImageElement,
    PptParagraph,
    PptPresentation,
    PptSlide,
    PptTableCell,
    PptTableElement,
    PptTextElement,
    PptTextRun,
)
from .parser import parse_ppt_document

PPT_XYCUT_BETA = 2.0
PPT_XYCUT_DENSITY_THRESHOLD = 0.9


class PptConverter:
    """将 PowerPoint 97–2003 二进制流转换为分页 raw blocks。"""

    def __init__(self) -> None:
        """初始化无状态转换器输出。"""

        self.pages: list[list[dict[str, Any]]] = []

    def convert(self, file_binary: BinaryIO) -> None:
        """读取输入流、解析三个核心 OLE streams 并生成 model-list。"""

        file_bytes = read_stream_bytes_from_start(file_binary)
        with BoundedOleReader(file_bytes) as ole:
            presentation = parse_ppt_document(
                ole.read_stream("PowerPoint Document"),
                current_user=ole.read_stream("Current User", required=False),
                pictures=ole.read_stream("Pictures", required=False),
            )
        self.pages = self._presentation_to_pages(presentation)

    @staticmethod
    def _run_styles(run: PptTextRun) -> list[str]:
        """把内部字符属性转换为 MinerU 富文本样式名。"""

        styles: list[str] = []
        if run.bold:
            styles.append("bold")
        if run.italic:
            styles.append("italic")
        if run.underline:
            styles.append("underline")
        if run.strike:
            styles.append("strikethrough")
        if run.baseline is not None and run.baseline > 0:
            styles.append("superscript")
        elif run.baseline is not None and run.baseline < 0:
            styles.append("subscript")
        return styles

    @classmethod
    def _paragraph_content(cls, paragraph: PptParagraph) -> str:
        """把段落 run 构建为已转义的 MinerU Office 富文本。"""

        segments = [
            OfficeRichTextSegment(
                text=escape(run.text, quote=False).replace("\n", " "),
                style=cls._run_styles(run),
                hyperlink=escape(run.hyperlink, quote=False) if run.hyperlink else None,
            )
            for run in paragraph.runs
            if run.text
        ]
        return build_rich_text_from_segments(segments, trim_plain_edges=True)

    @classmethod
    def _append_list_paragraph(
        cls,
        root_blocks: list[dict[str, Any]],
        stack: list[dict[str, Any]],
        paragraph: PptParagraph,
    ) -> None:
        """把一个列表段落放入对应层级，并按需创建中间列表。"""

        depth = max(0, int(paragraph.depth))
        while len(stack) > depth + 1:
            stack.pop()
        while len(stack) < depth + 1:
            attribute = paragraph.list_kind or "unordered"
            new_list: dict[str, Any] = {
                "type": BlockType.LIST,
                "attribute": attribute,
                "ilevel": len(stack),
                "content": [],
            }
            if attribute == "ordered" and paragraph.start is not None:
                new_list["start"] = paragraph.start
            if stack:
                stack[-1]["content"].append(new_list)
            else:
                root_blocks.append(new_list)
            stack.append(new_list)
        current = stack[depth]
        expected_attribute = paragraph.list_kind or "unordered"
        if current.get("attribute") != expected_attribute:
            del stack[depth:]
            cls._append_list_paragraph(root_blocks, stack, paragraph)
            return
        content = cls._paragraph_content(paragraph)
        if content:
            current["content"].append({"type": BlockType.TEXT, "content": content})

    @classmethod
    def _text_element_blocks(
        cls,
        element: PptTextElement,
        *,
        title_candidate: bool,
    ) -> list[dict[str, Any]]:
        """把文本形状转换为标题、正文和嵌套列表 raw blocks。"""

        blocks: list[dict[str, Any]] = []
        list_stack: list[dict[str, Any]] = []
        title_consumed = False
        for paragraph in element.paragraphs:
            content = cls._paragraph_content(paragraph)
            if not content:
                continue
            if title_candidate and not title_consumed and paragraph.list_kind is None:
                blocks.append(
                    {
                        "type": BlockType.PARAGRAPH_TITLE,
                        "content": content,
                        "level": 2,
                        "_ppt_title_candidate": True,
                    }
                )
                title_consumed = True
                list_stack.clear()
                continue
            if paragraph.list_kind is not None:
                cls._append_list_paragraph(blocks, list_stack, paragraph)
                continue
            list_stack.clear()
            blocks.append({"type": BlockType.TEXT, "content": content})
        return blocks

    @classmethod
    def _table_cell_content(cls, cell: PptTableCell) -> str:
        """把表格单元格内的多个段落连接为 HTML 内容。"""

        return "<br/>".join(
            content
            for paragraph in cell.paragraphs
            if (content := cls._paragraph_content(paragraph))
        )

    @classmethod
    def _table_html(cls, table: PptTableElement) -> str:
        """按原点单元格生成带 rowspan/colspan 的稳定 HTML 表格。"""

        origins = {(cell.row, cell.col): cell for cell in table.cells}
        covered: set[tuple[int, int]] = set()
        rows: list[str] = []
        for row in range(table.rows):
            cells: list[str] = []
            for col in range(table.cols):
                if (row, col) in covered:
                    continue
                cell = origins.get((row, col))
                if cell is None:
                    cells.append("<td></td>")
                    continue
                attributes: list[str] = []
                if cell.row_span > 1:
                    attributes.append(f'rowspan="{cell.row_span}"')
                if cell.col_span > 1:
                    attributes.append(f'colspan="{cell.col_span}"')
                attribute_text = f" {' '.join(attributes)}" if attributes else ""
                cells.append(f"<td{attribute_text}>{cls._table_cell_content(cell)}</td>")
                for covered_row in range(row, row + cell.row_span):
                    for covered_col in range(col, col + cell.col_span):
                        if (covered_row, covered_col) != (row, col):
                            covered.add((covered_row, covered_col))
            rows.append(f"<tr>{''.join(cells)}</tr>")
        return f'<table border="1">{"".join(rows)}</table>'

    @classmethod
    def _element_blocks(
        cls,
        element: PptTextElement | PptImageElement | PptEquationElement | PptTableElement,
        *,
        slide_height: int,
        is_first_text_element: bool,
    ) -> list[dict[str, Any]]:
        """把一个语义元素转换为 raw blocks。"""

        if isinstance(element, PptImageElement):
            return [{"type": BlockType.IMAGE, "image_base64": element.image_base64}]
        if isinstance(element, PptEquationElement):
            return [{"type": BlockType.EQUATION, "content": element.latex}]
        if isinstance(element, PptTableElement):
            return [{"type": BlockType.TABLE, "content": cls._table_html(element)}]

        authoritative_title = element.text_type in {0, 6}
        first_paragraph = element.paragraphs[0] if element.paragraphs else None
        heuristic_title = bool(
            is_first_text_element
            and element.is_placeholder
            and first_paragraph is not None
            and first_paragraph.list_kind is None
            and element.bbox[1] <= slide_height * 0.35
            and len("".join(run.text for run in first_paragraph.runs).strip()) <= 200
        )
        return cls._text_element_blocks(
            element,
            title_candidate=authoritative_title or heuristic_title,
        )

    @classmethod
    def _slide_to_page(cls, slide: PptSlide, presentation: PptPresentation) -> list[dict[str, Any]]:
        """按 XYCut++ 排序一张幻灯片，并把备注稳定追加到末尾。"""

        entries: list[dict[str, Any]] = []
        text_seen = False
        for element in slide.elements:
            is_first_text = isinstance(element, PptTextElement) and not text_seen
            blocks = cls._element_blocks(
                element,
                slide_height=presentation.height,
                is_first_text_element=is_first_text,
            )
            if isinstance(element, PptTextElement) and blocks:
                text_seen = True
            if blocks:
                entries.append({"bbox": element.bbox, "blocks": blocks, "order": element.order})
        ordered_entries = sort_entries(
            entries,
            beta=PPT_XYCUT_BETA,
            density_threshold=PPT_XYCUT_DENSITY_THRESHOLD,
        )
        page = [block for entry in ordered_entries for block in entry["blocks"]]
        for paragraph in slide.notes:
            content = cls._paragraph_content(paragraph)
            if content:
                page.append({"type": BlockType.PAGE_FOOTNOTE, "content": content})
        return page

    @classmethod
    def _presentation_to_pages(cls, presentation: PptPresentation) -> list[list[dict[str, Any]]]:
        """转换整份演示文稿，并把首个有效标题提升为文档标题。"""

        pages = [cls._slide_to_page(slide, presentation) for slide in presentation.slides]
        document_title_promoted = False
        for page in pages:
            candidates = [block for block in page if block.pop("_ppt_title_candidate", False)]
            if not candidates:
                continue
            first = candidates[0]
            if not document_title_promoted:
                first["type"] = BlockType.DOC_TITLE
                first["level"] = 1
                document_title_promoted = True
            for candidate in candidates[1:] if first["type"] == BlockType.DOC_TITLE else candidates:
                candidate["type"] = BlockType.PARAGRAPH_TITLE
                candidate["level"] = 2
        return pages
