# Copyright (c) Opendatalab. All rights reserved.
"""将 AnalyzeSpan 按横排或竖排规则聚合为 AnalyzeLine。"""

from __future__ import annotations

from .....types import ContentType
from .....model.ocr.geometry import _is_overlaps_x_exceeds_threshold, _is_overlaps_y_exceeds_threshold
from .models import _AnalyzeLine, _AnalyzeSpan

VERTICAL_SPAN_HEIGHT_TO_WIDTH_RATIO_THRESHOLD = 2
VERTICAL_SPAN_IN_BLOCK_THRESHOLD = 0.8


def is_vertical_text_block_by_spans(spans: list[_AnalyzeSpan]) -> bool:
    """根据块内文本 span 的高宽比判断文本块是否更像竖排文本。"""
    valid_span_count = 0
    vertical_span_count = 0
    for span in spans:
        bbox = span.bbox
        if not bbox or len(bbox) < 4:
            continue
        span_width = bbox[2] - bbox[0]
        span_height = bbox[3] - bbox[1]
        if span_width <= 0 or span_height <= 0:
            continue
        valid_span_count += 1
        if span_height / span_width > VERTICAL_SPAN_HEIGHT_TO_WIDTH_RATIO_THRESHOLD:
            vertical_span_count += 1
    if valid_span_count == 0:
        return False
    return vertical_span_count / valid_span_count > VERTICAL_SPAN_IN_BLOCK_THRESHOLD


def group_spans_to_lines(spans: list[_AnalyzeSpan]) -> list[_AnalyzeLine]:
    """直接把私有 span 分组成有序行，不再构造带临时字段的公开 Block。"""
    for span in spans:
        if span.type == ContentType.INTERLINE_EQUATION:
            span.type = ContentType.INLINE_EQUATION

    if is_vertical_text_block_by_spans(spans):
        return vertical_line_sort_spans_from_top_to_bottom(merge_spans_to_vertical_line(spans))
    return line_sort_spans_by_left_to_right(merge_spans_to_line(spans))


def merge_spans_to_line(
    spans: list[_AnalyzeSpan],
    threshold: float = 0.6,
) -> list[list[_AnalyzeSpan]]:
    """按 y 轴重叠关系把横排 span 分组成行。"""
    if not spans:
        return []
    spans.sort(key=lambda span: span.bbox[1])
    lines: list[list[_AnalyzeSpan]] = []
    current_line = [spans[0]]
    special_types = {ContentType.INTERLINE_EQUATION, ContentType.IMAGE, ContentType.TABLE}
    for span in spans[1:]:
        if span.type in special_types or any(item.type in special_types for item in current_line):
            lines.append(current_line)
            current_line = [span]
        elif _is_overlaps_y_exceeds_threshold(span.bbox, current_line[-1].bbox, threshold):
            current_line.append(span)
        else:
            lines.append(current_line)
            current_line = [span]
    lines.append(current_line)
    return lines


def merge_spans_to_vertical_line(
    spans: list[_AnalyzeSpan],
    threshold: float = 0.6,
) -> list[list[_AnalyzeSpan]]:
    """按 x 轴重叠关系把竖排 span 从右向左分组成列。"""
    if not spans:
        return []
    spans.sort(key=lambda span: span.bbox[2], reverse=True)
    lines: list[list[_AnalyzeSpan]] = []
    current_line = [spans[0]]
    special_types = {ContentType.INTERLINE_EQUATION, ContentType.IMAGE, ContentType.TABLE}
    for span in spans[1:]:
        if span.type in special_types or any(item.type in special_types for item in current_line):
            lines.append(current_line)
            current_line = [span]
        elif _is_overlaps_x_exceeds_threshold(span.bbox, current_line[-1].bbox, threshold):
            current_line.append(span)
        else:
            lines.append(current_line)
            current_line = [span]
    lines.append(current_line)
    return lines


def line_sort_spans_by_left_to_right(
    lines: list[list[_AnalyzeSpan]],
) -> list[_AnalyzeLine]:
    """将横排行内 span 从左到右排序并计算行框。"""
    line_objects: list[_AnalyzeLine] = []
    for line in lines:
        line.sort(key=lambda span: span.bbox[0])
        line_objects.append(_line_from_spans(line))
    return line_objects


def vertical_line_sort_spans_from_top_to_bottom(
    vertical_lines: list[list[_AnalyzeSpan]],
) -> list[_AnalyzeLine]:
    """将竖排列内 span 从上到下排序并计算列框。"""
    line_objects: list[_AnalyzeLine] = []
    for line in vertical_lines:
        line.sort(key=lambda span: span.bbox[1])
        line_objects.append(_line_from_spans(line))
    return line_objects


def _line_from_spans(spans: list[_AnalyzeSpan]) -> _AnalyzeLine:
    """聚合一组 span 的外接矩形并构造私有行对象。"""
    return _AnalyzeLine(
        bbox=(
            min(span.bbox[0] for span in spans),
            min(span.bbox[1] for span in spans),
            max(span.bbox[2] for span in spans),
            max(span.bbox[3] for span in spans),
        ),
        spans=spans,
    )
