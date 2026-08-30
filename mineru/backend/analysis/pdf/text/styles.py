# Copyright (c) Opendatalab. All rights reserved.
"""Hybrid TXT 路径复用的 PDF 原生文本富化准备入口。"""

from __future__ import annotations

from typing import Any, Sequence

from .....model.flash.pdf.document import PDFPage, PDFPageTextGeometry, get_lines_from_chars
from .....model.flash.pdf.text_styles import (
    PDFTextLinkLine,
    PDFTextScriptLine,
    PDFTextStyleLine,
    detect_pdf_text_link_lines,
    detect_pdf_text_script_lines,
    detect_pdf_text_style_lines,
)
from .....types import BBox


def _script_range_center_in_regions(
    bbox: BBox,
    regions: Sequence[BBox],
) -> bool:
    """判断脚本证据中心是否落入由 layout 确认的行内公式区域。"""

    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    return any(region[0] <= center_x <= region[2] and region[1] <= center_y <= region[3] for region in regions)


def _exclude_layout_formula_script_ranges(
    lines: list[PDFTextScriptLine],
    regions: Sequence[BBox],
) -> list[PDFTextScriptLine]:
    """从 Hybrid TXT 脚本 sidecar 中剔除已由 layout/MFR 拥有的公式字符。"""

    if not regions:
        return lines
    return [
        PDFTextScriptLine(
            bbox=line.bbox,
            text=line.text,
            script_ranges=tuple(
                script_range
                for script_range in line.script_ranges
                if not _script_range_center_in_regions(script_range.bbox, regions)
            ),
            source_index=line.source_index,
            angle=line.angle,
        )
        for line in lines
    ]


def build_pdf_native_visual_lines_and_styles(
    pdf_page: PDFPage,
    *,
    page_text_geometry: PDFPageTextGeometry | None = None,
    supported_angles: Sequence[float] = (0.0,),
    inline_math_regions: Sequence[BBox] = (),
    table_regions: Sequence[BBox] = (),
) -> tuple[
    PDFPageTextGeometry,
    list[Any],
    list[PDFTextStyleLine],
    list[PDFTextLinkLine],
    list[PDFTextScriptLine],
]:
    """一次读取当前页字符，构造视觉 run、普通样式、链接和脚本证据。"""

    # 延迟导入避免 Hybrid 模块初始化时提前加载完整 Flash PDF 流水线。
    from .....model.flash.pdf.native_text import _build_native_line_items
    from .....model.flash.pdf.line_merging import (
        _merge_overlapping_inline_text_clusters,
        _merge_same_baseline_text_lines,
    )

    geometry = page_text_geometry if page_text_geometry is not None else pdf_page.get_chars_with_geometry()
    chars = geometry.chars
    line_items = _build_native_line_items(
        get_lines_from_chars(chars),
        tuple(float(value) for value in pdf_page.size),
        page_rotation=pdf_page.rotation,
        supported_angles=supported_angles,
    )
    resolved_inline_math_regions = [tuple(float(value) for value in region) for region in inline_math_regions]
    resolved_table_regions = [tuple(float(value) for value in region) for region in table_regions]
    drawing_lines = pdf_page.get_drawing_lines()
    style_lines = detect_pdf_text_style_lines(
        line_items,
        drawing_lines,
    )
    link_lines = detect_pdf_text_link_lines(
        line_items,
        pdf_page.get_link_annotations(),
    )
    script_line_items = _merge_same_baseline_text_lines(
        list(line_items),
        tuple(float(value) for value in pdf_page.size),
        resolved_table_regions,
    )
    script_line_items = _merge_overlapping_inline_text_clusters(
        script_line_items,
        tuple(float(value) for value in pdf_page.size),
        resolved_table_regions,
    )
    script_line_items = _merge_same_baseline_text_lines(
        script_line_items,
        tuple(float(value) for value in pdf_page.size),
        resolved_table_regions,
    )
    script_lines = detect_pdf_text_script_lines(
        script_line_items,
        tuple(float(value) for value in pdf_page.size),
        geometry.tight_bboxes,
        geometry.origins,
        all_chars=chars,
        drawing_lines=drawing_lines,
    )
    script_lines = _exclude_layout_formula_script_ranges(
        script_lines,
        resolved_inline_math_regions,
    )
    return geometry, line_items, style_lines, link_lines, script_lines


__all__ = ["build_pdf_native_visual_lines_and_styles"]
