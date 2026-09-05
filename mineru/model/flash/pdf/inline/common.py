# Copyright (c) Opendatalab. All rights reserved.
"""提供行内证据共享的字符和几何规范化原语。"""

from __future__ import annotations

import math
from typing import Any, Iterable, cast

from .....types import BBox
from .types import (
    _LIGATURE_REPLACEMENTS,
    _PDF_CONTROL_CHAR_RE,
    _PDF_SEPARATOR_SPACE_CHARS,
    _PDF_ZERO_WIDTH_CHARS,
    PDF_TEXT_STYLE_ORDER,
    PDFTextStyle,
    PDFTextStyleLine,
)


def _style_line_reading_order_key(
    line: PDFTextStyleLine,
) -> tuple[int, float, float]:
    """优先使用原生 source_index 排序，重复索引时再以 bbox 保持稳定。"""

    return line.source_index, line.bbox[1], line.bbox[0]


def _coerce_bbox(value: Any) -> BBox | None:
    """把 list、tuple 或 pdftext bbox 对象收敛为合法有限 bbox。"""

    raw_bbox = getattr(value, "bbox", value)
    try:
        if raw_bbox is None or len(raw_bbox) != 4:
            return None
        bbox = tuple(float(item) for item in raw_bbox)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in bbox):
        return None
    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None
    return bbox  # type: ignore[return-value]


def _ordered_line_chars(line: Any) -> list[dict[str, Any]]:
    """按 char_idx 修复异常乱序字符，同时保留缺少索引时的来源顺序。"""

    chars = [char for char in getattr(line, "chars", []) if isinstance(char, dict)]
    indexed_chars = [char.get("char_idx") for char in chars]
    if (
        chars
        and all(isinstance(index, int) for index in indexed_chars)
        and any(first > second for first, second in zip(indexed_chars, indexed_chars[1:]))
    ):
        return sorted(chars, key=lambda char: int(char["char_idx"]))
    return chars


def _normalize_match_fragment(value: Any) -> str:
    """把单个字符片段规范为忽略排版空白的确定性匹配文本。"""

    output: list[str] = []
    for char in str(value or ""):
        if char in _PDF_ZERO_WIDTH_CHARS or char == "\u00ad":
            continue
        if char == "\x02":
            output.append("-")
            continue
        if char.isspace() or char in _PDF_SEPARATOR_SPACE_CHARS:
            continue
        if _PDF_CONTROL_CHAR_RE.fullmatch(char):
            continue
        output.append(_LIGATURE_REPLACEMENTS.get(char, char))
    return "".join(output)


def _canonical_styles(styles: Iterable[str]) -> tuple[PDFTextStyle, ...]:
    """按公开富文本协议顺序过滤、去重并规范样式集合。"""

    style_set = set(styles)
    return cast(
        tuple[PDFTextStyle, ...],
        tuple(style for style in PDF_TEXT_STYLE_ORDER if style in style_set),
    )


def _bbox_intersection_area(first: BBox, second: BBox) -> float:
    """返回两个合法 bbox 的相交面积。"""

    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _bbox_overlap_ratio(first: BBox, second: BBox) -> float:
    """返回 first 面积中落入 second 的比例。"""

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = max(0.01, (first[2] - first[0]) * (first[3] - first[1]))
    return intersection_width * intersection_height / first_area


__all__ = [
    "_style_line_reading_order_key",
    "_coerce_bbox",
    "_ordered_line_chars",
    "_normalize_match_fragment",
    "_canonical_styles",
    "_bbox_intersection_area",
    "_bbox_overlap_ratio",
]
