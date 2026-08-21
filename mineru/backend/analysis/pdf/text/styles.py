# Copyright (c) Opendatalab. All rights reserved.
"""Hybrid TXT 路径复用的 PDF 原生文本富化准备入口。"""

from __future__ import annotations

from typing import Any, Sequence

from pdftext.schema import Char

from mineru.utils.pdf_document import PDFPage, get_lines_from_chars
from mineru.utils.pdf_text_styles import (
    PDFTextLinkLine,
    PDFTextStyleLine,
    detect_pdf_text_link_lines,
    detect_pdf_text_style_lines,
)


def build_pdf_native_visual_lines_and_styles(
    pdf_page: PDFPage,
    *,
    page_chars: list[Char] | None = None,
    supported_angles: Sequence[float] = (0.0,),
) -> tuple[
    list[Char],
    list[Any],
    list[PDFTextStyleLine],
    list[PDFTextLinkLine],
]:
    """一次读取当前页字符，构造视觉 run、文本样式和超链接证据。"""

    # 延迟导入避免 Hybrid 模块初始化时提前加载完整 Flash PDF 流水线。
    from mineru.model.flash.native_pdf.native_text import _build_native_line_items

    chars = page_chars if page_chars is not None else pdf_page.get_chars()
    line_items = _build_native_line_items(
        get_lines_from_chars(chars),
        tuple(float(value) for value in pdf_page.size),
        page_rotation=pdf_page.rotation,
        supported_angles=supported_angles,
    )
    style_lines = detect_pdf_text_style_lines(
        line_items,
        pdf_page.get_drawing_lines(),
    )
    link_lines = detect_pdf_text_link_lines(
        line_items,
        pdf_page.get_link_annotations(),
    )
    return chars, line_items, style_lines, link_lines


__all__ = ["build_pdf_native_visual_lines_and_styles"]
