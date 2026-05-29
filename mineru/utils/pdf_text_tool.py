# Copyright (c) Opendatalab. All rights reserved.
import math
from typing import List

import pypdfium2 as pdfium
from pdftext.pdf.chars import deduplicate_chars, get_chars
from pdftext.pdf.pages import assign_scripts, get_blocks, get_lines, get_spans

from mineru.utils.pdfium_guard import pdfium_guard


def get_page(
    page: pdfium.PdfPage,
    quote_loosebox: bool = True,
    superscript_height_threshold: float = 0.7,
    line_distance_threshold: float = 0.1,
) -> dict:
    page_chars = get_page_chars(page, quote_loosebox=quote_loosebox)
    lines = get_lines_from_chars(
        page_chars["chars"],
        superscript_height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    blocks = get_blocks(lines)

    return {
        "bbox": page_chars["bbox"],
        "width": page_chars["width"],
        "height": page_chars["height"],
        "rotation": page_chars["rotation"],
        "blocks": blocks,
    }


def get_page_chars(
    page: pdfium.PdfPage,
    textpage=None,
    quote_loosebox: bool = True,
    page_char_count: int | None = None,
) -> dict:
    """轻量读取页面字符坐标，供只需要 char 级信息的路径复用。"""
    with pdfium_guard():
        if textpage is None:
            textpage = page.get_textpage()
        page_bbox: List[float] = page.get_bbox()
        page_width = math.ceil(abs(page_bbox[2] - page_bbox[0]))
        page_height = math.ceil(abs(page_bbox[1] - page_bbox[3]))

        page_rotation = 0
        try:
            page_rotation = page.get_rotation()
        except Exception:
            pass

        if page_char_count is None:
            page_char_count = textpage.count_chars()

        chars = deduplicate_chars(
            get_chars(textpage, page_bbox, page_rotation, quote_loosebox)
        )

    return {
        "bbox": page_bbox,
        "width": page_width,
        "height": page_height,
        "rotation": page_rotation,
        "char_count": page_char_count,
        "chars": chars,
    }


def get_lines_from_chars(
    chars,
    superscript_height_threshold: float = 0.7,
    line_distance_threshold: float = 0.1,
):
    """从已提取的字符构建 pdftext lines，避免重复读取 PDFium textpage。"""
    spans = get_spans(
        chars,
        superscript_height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    lines = get_lines(spans)
    assign_scripts(
        lines,
        height_threshold=superscript_height_threshold,
        line_distance_threshold=line_distance_threshold,
    )
    return lines
