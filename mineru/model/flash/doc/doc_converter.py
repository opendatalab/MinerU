# Copyright (c) Opendatalab. All rights reserved.

"""把 Word 97–2003 语义模型转换为 MinerU 分页 model-list。"""

from __future__ import annotations

from html import escape
import re
from typing import Any, BinaryIO, Iterable

from mineru.model.flash.legacy_office import BoundedOleReader, LegacyOfficeEncryptedError
from mineru.model.flash.office.image import serialize_office_image
from mineru.model.office_stream import read_stream_bytes_from_start
from mineru.types import RAW_CAPTION, BlockType
from mineru.utils.office_rich_text import OfficeRichTextSegment, build_rich_text_from_segments

from .fib import parse_fib
from mineru.model.flash.legacy_office.mtef import read_object_pool_equations
from .models import (
    DocCharStyle,
    DocDocument,
    DocElement,
    DocImage,
    DocImagePayload,
    DocParagraph,
    DocSection,
    DocTable,
    DocTableCell,
    DocTextRun,
)
from .parser import parse_doc_document


class DocConverter:
    """将 Word 97–2003 OLE 二进制流转换为逐 section raw blocks。"""

    def __init__(self) -> None:
        """初始化空输出。"""

        self.pages: list[list[dict[str, Any]]] = []

    def convert(self, file_binary: BinaryIO) -> None:
        """读取输入 OLE streams，解析 DOC 并生成 model-list。"""

        file_bytes = read_stream_bytes_from_start(file_binary)
        with BoundedOleReader(file_bytes) as ole:
            if ole.has_stream("EncryptedPackage") or ole.has_stream("EncryptionInfo"):
                raise LegacyOfficeEncryptedError("encrypted OOXML package is not a binary DOC")
            word_document = ole.read_stream("WordDocument")
            fib = parse_fib(word_document)
            if fib.base.encrypted or fib.base.obfuscated:
                raise LegacyOfficeEncryptedError("password-protected DOC is unsupported")
            preferred = "1Table" if fib.base.uses_1table else "0Table"
            alternate = "0Table" if preferred == "1Table" else "1Table"
            table_stream = ole.read_stream(preferred, required=False)
            if not table_stream:
                table_stream = ole.read_stream(alternate, required=bool(fib.base.complex))
            data_stream = ole.read_stream("Data", required=False)
            native_equations = read_object_pool_equations(ole)
            document = parse_doc_document(
                word_document,
                table_stream,
                data_stream,
                fib,
                native_equations=native_equations,
            )
        self.pages = self._document_pages(document)

    @staticmethod
    def _style_names(style: DocCharStyle) -> list[str]:
        """把 DOC 字符属性转换为 MinerU 富文本样式名。"""

        names: list[str] = []
        if style.bold:
            names.append("bold")
        if style.italic:
            names.append("italic")
        if style.underline:
            names.append("underline")
        if style.emphasis:
            names.append("emphasis")
        if style.strike:
            names.append("strikethrough")
        if style.superscript:
            names.append("superscript")
        elif style.subscript:
            names.append("subscript")
        return names

    @classmethod
    def _rich_text(cls, runs: Iterable[DocTextRun], *, trim: bool = True) -> str:
        """把 DOC runs 转换为已转义的 Office 富文本协议。"""

        parts: list[str] = []
        segments: list[OfficeRichTextSegment] = []

        def flush_segments() -> None:
            """输出公式边界前累计的普通富文本。"""

            if not segments:
                return
            parts.append(
                build_rich_text_from_segments(
                    segments,
                    trim_plain_edges=trim and not parts,
                )
            )
            segments.clear()

        for run in runs:
            if not run.text:
                continue
            if run.formula:
                flush_segments()
                latex = run.text.replace("<", r"\lt ").replace(">", r"\gt ")
                parts.append(f"<eq>{latex}</eq>")
                continue
            segments.append(
                OfficeRichTextSegment(
                    text=escape(run.text, quote=False).replace("\n", "<br/>"),
                    style=cls._style_names(run.style),
                    hyperlink=escape(run.hyperlink, quote=True) if run.hyperlink else None,
                )
            )
        flush_segments()
        return "".join(parts).strip() if trim else "".join(parts)

    @staticmethod
    def _plain_text(paragraph: DocParagraph) -> str:
        """返回段落不含内部标记的可见文本。"""

        return "".join(run.text for run in paragraph.runs)

    @staticmethod
    def _serialize_image(payload: DocImagePayload) -> str | None:
        """复用 Office 图片序列化与矢量占位策略。"""

        return serialize_office_image(
            payload.data,
            part_name=f"image.{payload.extension}",
            content_type=payload.content_type,
        )

    @classmethod
    def _image_block(cls, payload: DocImagePayload) -> dict[str, Any] | None:
        """把原始图片载荷转换为 equation 或 image block。"""

        if payload.equation_latex:
            return {
                "type": BlockType.EQUATION,
                "content": payload.equation_latex,
            }

        image_base64 = cls._serialize_image(payload)
        if image_base64 is None:
            return None
        return {"type": BlockType.IMAGE, "image_base64": image_base64}

    @classmethod
    def _toc_runs(cls, paragraph: DocParagraph) -> list[DocTextRun]:
        """从 TOC 段落末尾移除仅用于排版的 tab/page number。"""

        runs = list(paragraph.runs)
        while runs and re.fullmatch(r"[\s\t]*\d+[\s\t]*", runs[-1].text):
            runs.pop()
        if runs:
            cleaned = re.sub(r"\t+\s*\d+\s*$", "", runs[-1].text)
            if cleaned != runs[-1].text:
                last = runs[-1]
                runs[-1] = DocTextRun(cleaned, last.style, last.hyperlink)
        return runs

    @classmethod
    def _toc_anchor(cls, paragraph: DocParagraph) -> str | None:
        """读取 TOC 超链接指向的内部书签。"""

        for run in paragraph.runs:
            if run.hyperlink and run.hyperlink.startswith("#") and len(run.hyperlink) > 1:
                return run.hyperlink[1:]
        return paragraph.anchor

    @classmethod
    def _append_index_item(
        cls,
        page: list[dict[str, Any]],
        stack: list[dict[str, Any]],
        paragraph: DocParagraph,
    ) -> None:
        """把一个 TOC 段落追加到对应层级的 index 树。"""

        content = cls._rich_text(cls._toc_runs(paragraph))
        content += "".join(
            f"<eq>{escape(payload.equation_latex, quote=False)}</eq>"
            for payload in paragraph.images
            if payload.equation_latex
        )
        if not content:
            return
        level = min(max(paragraph.toc_level or 0, 0), 8)
        while len(stack) > level + 1:
            stack.pop()
        while len(stack) < level + 1:
            index_block: dict[str, Any] = {
                "type": BlockType.INDEX,
                "ilevel": len(stack),
                "content": [],
            }
            if stack:
                stack[-1]["content"].append(index_block)
            else:
                page.append(index_block)
            stack.append(index_block)
        leaf: dict[str, Any] = {"type": BlockType.TEXT, "content": content}
        anchor = cls._toc_anchor(paragraph)
        if anchor:
            leaf["anchor"] = anchor
        stack[level]["content"].append(leaf)

    @classmethod
    def _append_list_item(
        cls,
        page: list[dict[str, Any]],
        stack: list[dict[str, Any]],
        identity: int | None,
        paragraph: DocParagraph,
    ) -> int:
        """把一个 DOC 列表段落追加到嵌套 raw list 树。"""

        info = paragraph.list_info
        if info is None:
            return identity or -1
        content = cls._rich_text(paragraph.runs)
        content += "".join(
            f"<eq>{escape(payload.equation_latex, quote=False)}</eq>"
            for payload in paragraph.images
            if payload.equation_latex
        )
        if not content and not paragraph.images:
            return identity or info.identity
        if identity != info.identity:
            stack.clear()
            identity = info.identity
        level = min(max(info.level, 0), 8)
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
            return cls._append_list_item(page, stack, None, paragraph)
        if content:
            leaf: dict[str, Any] = {"type": BlockType.TEXT, "content": content}
            if info.label:
                leaf["list_label"] = info.label
            current["content"].append(leaf)
        return info.identity

    @classmethod
    def _paragraph_blocks(cls, paragraph: DocParagraph) -> list[dict[str, Any]]:
        """把非目录、非普通列表段落投影为 raw blocks。"""

        content = cls._rich_text(paragraph.runs)
        blocks: list[dict[str, Any]] = []
        formula_runs = [run for run in paragraph.runs if run.formula and run.text]
        ordinary_text = "".join(run.text for run in paragraph.runs if not run.formula).strip()
        list_info = paragraph.list_info
        exact_label = list_info.label if list_info is not None and list_info.ordered else None
        if exact_label and (paragraph.is_title or paragraph.heading_level is not None):
            content = f"{escape(exact_label, quote=False)} {content}".strip()
        if formula_runs and not ordinary_text and not (
            paragraph.is_title
            or paragraph.heading_level is not None
            or paragraph.is_caption
            or paragraph.is_code
        ):
            blocks.extend(
                {"type": BlockType.EQUATION, "content": run.text}
                for run in formula_runs
            )
        elif content:
            if paragraph.is_title:
                block: dict[str, Any] = {
                    "type": BlockType.DOC_TITLE,
                    "level": 1,
                    "content": content,
                }
            elif paragraph.heading_level is not None:
                block = {
                    "type": BlockType.PARAGRAPH_TITLE,
                    "level": min(max(paragraph.heading_level + 1, 2), 6),
                    "is_numbered_style": False,
                    "content": content,
                }
            elif paragraph.is_caption:
                block = {"type": RAW_CAPTION, "content": content}
            elif paragraph.is_code:
                block = {"type": BlockType.CODE, "content": cls._plain_text(paragraph)}
            else:
                block = {"type": BlockType.TEXT, "content": content}
            if paragraph.anchor and block["type"] in {BlockType.DOC_TITLE, BlockType.PARAGRAPH_TITLE}:
                block["anchor"] = paragraph.anchor
            blocks.append(block)
        for payload in paragraph.images:
            image = cls._image_block(payload)
            if image is not None:
                blocks.append(image)
        return blocks

    @classmethod
    def _cell_list_html(cls, paragraphs: list[DocParagraph]) -> str:
        """把连续表格单元格列表段落序列化为嵌套 HTML list。"""

        result: list[str] = []
        stack: list[str] = []
        for paragraph in paragraphs:
            info = paragraph.list_info
            if info is None:
                while stack:
                    result.append(f"</{stack.pop()}>")
                result.append(cls._cell_paragraph_html(paragraph))
                continue
            level = min(max(info.level, 0), 8)
            tag = "ol" if info.ordered else "ul"
            while len(stack) > level + 1:
                result.append(f"</{stack.pop()}>")
            while len(stack) < level + 1:
                result.append(f"<{tag}>")
                stack.append(tag)
            if stack[-1] != tag:
                result.append(f"</{stack.pop()}><{tag}>")
                stack.append(tag)
            label = f"{escape(info.label)} " if info.label and info.ordered else ""
            result.append(
                f"<li>{label}{cls._rich_text(paragraph.runs)}"
                f"{cls._cell_images_html(paragraph.images)}</li>"
            )
        while stack:
            result.append(f"</{stack.pop()}>")
        return "".join(result)

    @classmethod
    def _cell_paragraph_html(cls, paragraph: DocParagraph) -> str:
        """序列化一个表格单元格段落及其内联图片。"""

        parts: list[str] = []
        content = cls._rich_text(paragraph.runs, trim=False)
        if content:
            parts.append(f"<p>{content}</p>")
        parts.append(cls._cell_images_html(paragraph.images))
        return "".join(parts)

    @classmethod
    def _cell_images_html(
        cls,
        payloads: list[DocImagePayload],
    ) -> str:
        """把表格段落中的图片或 comment 公式序列化为内联 HTML。"""

        parts: list[str] = []
        for payload in payloads:
            if payload.equation_latex:
                parts.append(
                    f"<eq>{escape(payload.equation_latex, quote=False)}</eq>"
                )
                continue
            image = cls._serialize_image(payload)
            if image:
                parts.append(f'<img src="{escape(image, quote=True)}"/>')
        return "".join(parts)

    @classmethod
    def _cell_html(cls, cell: DocTableCell) -> str:
        """递归序列化单元格中的段落和嵌套表格。"""

        parts: list[str] = []
        paragraph_buffer: list[DocParagraph] = []

        def flush() -> None:
            """输出当前连续段落缓冲。"""

            nonlocal paragraph_buffer
            if paragraph_buffer:
                parts.append(cls._cell_list_html(paragraph_buffer))
                paragraph_buffer = []

        for block in cell.blocks:
            if isinstance(block, DocParagraph):
                paragraph_buffer.append(block)
            elif isinstance(block, DocTable):
                flush()
                parts.append(cls._table_html(block))
            elif isinstance(block, DocImage):
                flush()
                if block.payload.equation_latex:
                    parts.append(
                        "<eq>"
                        f"{escape(block.payload.equation_latex, quote=False)}"
                        "</eq>"
                    )
                    continue
                image = cls._serialize_image(block.payload)
                if image:
                    parts.append(f'<img src="{escape(image, quote=True)}"/>')
        flush()
        return "".join(parts)

    @classmethod
    def _table_html(cls, table: DocTable) -> str:
        """序列化带 rowspan/colspan 及嵌套内容的 Word 表格。"""

        rows: list[str] = []
        for row in table.rows:
            tag = "th" if row.header else "td"
            cells: list[str] = []
            for cell in row.cells:
                attributes: list[str] = []
                if cell.row_span > 1:
                    attributes.append(f'rowspan="{cell.row_span}"')
                if cell.col_span > 1:
                    attributes.append(f'colspan="{cell.col_span}"')
                suffix = f" {' '.join(attributes)}" if attributes else ""
                cells.append(f"<{tag}{suffix}>{cls._cell_html(cell)}</{tag}>")
            rows.append(f"<tr>{''.join(cells)}</tr>")
        return f"<table>{''.join(rows)}</table>"

    @classmethod
    def _element_blocks(cls, element: DocElement) -> list[dict[str, Any]]:
        """把正文语义元素转换为 raw block。"""

        if isinstance(element, DocImage):
            image = cls._image_block(element.payload)
            return [image] if image is not None else []
        if isinstance(element, DocTable):
            return [{"type": BlockType.TABLE, "content": cls._table_html(element)}]
        return cls._paragraph_blocks(element)

    @classmethod
    def _auxiliary_contents(cls, paragraphs: list[DocParagraph]) -> list[str]:
        """去重页眉页脚段落并过滤纯页码。"""

        result: list[str] = []
        seen: set[str] = set()
        for paragraph in paragraphs:
            content = cls._rich_text(paragraph.runs)
            plain = cls._plain_text(paragraph).strip()
            if not content or plain.isdigit() or content in seen:
                continue
            seen.add(content)
            result.append(content)
        return result

    @classmethod
    def _section_page(cls, section: DocSection) -> list[dict[str, Any]]:
        """转换一个 section，并将页面辅助文本稳定追加到末尾。"""

        page: list[dict[str, Any]] = []
        list_stack: list[dict[str, Any]] = []
        list_identity: int | None = None
        index_stack: list[dict[str, Any]] = []
        for element in section.elements:
            if isinstance(element, DocParagraph) and element.is_toc:
                list_stack.clear()
                list_identity = None
                cls._append_index_item(page, index_stack, element)
                for payload in element.images:
                    if payload.equation_latex:
                        continue
                    image = cls._image_block(payload)
                    if image is not None:
                        page.append(image)
                continue
            index_stack.clear()
            if isinstance(element, DocParagraph) and element.list_info is not None and not (
                element.is_title or element.heading_level is not None
            ):
                list_identity = cls._append_list_item(page, list_stack, list_identity, element)
                for payload in element.images:
                    if payload.equation_latex:
                        continue
                    image = cls._image_block(payload)
                    if image is not None:
                        page.append(image)
                continue
            list_stack.clear()
            list_identity = None
            page.extend(cls._element_blocks(element))
        page.extend(
            {"type": BlockType.HEADER, "content": content}
            for content in cls._auxiliary_contents(section.headers)
        )
        page.extend(
            {"type": BlockType.FOOTER, "content": content}
            for content in cls._auxiliary_contents(section.footers)
        )
        page.extend(
            {"type": BlockType.PAGE_FOOTNOTE, "content": content}
            for content in cls._auxiliary_contents(section.footnotes)
        )
        return page

    @classmethod
    def _document_pages(cls, document: DocDocument) -> list[list[dict[str, Any]]]:
        """转换整份 DOC，并至少保留一个空 section page。"""

        pages = [cls._section_page(section) for section in document.sections]
        return pages or [[]]
