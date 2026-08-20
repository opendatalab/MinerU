# Copyright (c) Opendatalab. All rights reserved.
"""从 PDF 原生字符与绘图线中恢复文本删除线样式。"""

from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass
from typing import Any, Sequence

from loguru import logger

from mineru.types import BBox, BlockType, RAW_CAPTION, RAW_FOOTNOTE


STRIKETHROUGH_MIN_LENGTH_HEIGHT_RATIO = 2.0
STRIKETHROUGH_CENTER_TOLERANCE_HEIGHT_RATIO = 0.2
STRIKETHROUGH_MAX_WIDTH_HEIGHT_RATIO = 0.2
STRIKETHROUGH_MIN_TEXT_COVERAGE_RATIO = 0.55
STRIKETHROUGH_ENDPOINT_TOLERANCE_HEIGHT_RATIO = 0.5

PDF_NATURAL_TEXT_STYLE_BLOCK_TYPES = frozenset(
    {
        BlockType.TEXT,
        BlockType.REF_TEXT,
        BlockType.DOC_TITLE,
        BlockType.PARAGRAPH_TITLE,
        BlockType.LIST,
        BlockType.INDEX,
        BlockType.HEADER,
        BlockType.FOOTER,
        BlockType.PAGE_NUMBER,
        BlockType.ASIDE_TEXT,
        BlockType.PAGE_FOOTNOTE,
        RAW_CAPTION,
        RAW_FOOTNOTE,
    }
)

_KNOWN_INLINE_TAG_RE = re.compile(
    r"<(?P<close>/)?(?P<tag>eq|text|hyperlink|url|sup|sub|strong|b|em|i|s|u)(?P<attrs>\s[^<>]*?)?>",
    re.IGNORECASE,
)
_STYLE_ATTR_RE = re.compile(r"\bstyle\s*=\s*([\"'])(?P<style>.*?)\1", re.IGNORECASE | re.DOTALL)
_PDF_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_PDF_SEPARATOR_SPACE_CHARS = frozenset(
    "\u00a0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006"
    "\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
)
_PDF_ZERO_WIDTH_CHARS = frozenset({"\u200b", "\u2060", "\ufeff"})
_LIGATURE_REPLACEMENTS = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}


@dataclass(frozen=True, slots=True)
class PDFTextStyleRange:
    """保存可比较文本中的一个半开删除线区间。"""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class PDFTextStyleLine:
    """保存一个视觉文本 run 的几何、可比较文本和删除线区间。"""

    bbox: BBox
    text: str
    strikethrough_ranges: tuple[PDFTextStyleRange, ...]
    source_index: int


@dataclass(frozen=True, slots=True)
class _VisibleChar:
    """保存参与删除线几何判断的可见字符。"""

    source_index: int
    bbox: BBox


@dataclass(slots=True)
class _LineCandidate:
    """保存删除线匹配阶段使用的视觉文本行指标。"""

    bbox: BBox
    chars: list[dict[str, Any]]
    visible_chars: list[_VisibleChar]
    median_height: float
    center_y: float
    source_index: int
    ranges: list[tuple[int, int]]


@dataclass(frozen=True, slots=True)
class _DrawingMatch:
    """保存单条 drawing 对单个文本行的匹配结果和排序指标。"""

    start_index: int
    end_index: int
    center_distance_ratio: float
    horizontal_overlap_ratio: float


@dataclass(frozen=True, slots=True)
class _ProjectedChar:
    """保存 model content 可比较字符到原字符串位置的映射。"""

    value: str
    raw_start: int
    raw_end: int
    styleable: bool


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


def _build_line_candidate(line: Any) -> _LineCandidate | None:
    """从视觉水平 line 构造字符几何候选，旋转文字和退化行返回空。"""

    if int(getattr(line, "angle", 0) or 0) % 360 != 0:
        return None
    line_bbox = _coerce_bbox(getattr(line, "bbox", None))
    if line_bbox is None:
        return None
    chars = _ordered_line_chars(line)
    visible_chars: list[_VisibleChar] = []
    for char_index, char in enumerate(chars):
        text = str(char.get("char") or "")
        bbox = _coerce_bbox(char.get("bbox"))
        if bbox is None or not text.isprintable() or text.isspace():
            continue
        visible_chars.append(_VisibleChar(source_index=char_index, bbox=bbox))
    if not visible_chars:
        return None

    heights = [char.bbox[3] - char.bbox[1] for char in visible_chars]
    median_height = statistics.median(heights)
    if median_height <= 0:
        return None
    body_centers = [
        (char.bbox[1] + char.bbox[3]) / 2
        for char, height in zip(visible_chars, heights)
        if height >= 0.8 * median_height
    ]
    if not body_centers:
        return None
    return _LineCandidate(
        bbox=line_bbox,
        chars=chars,
        visible_chars=visible_chars,
        median_height=median_height,
        center_y=statistics.median(body_centers),
        source_index=int(getattr(line, "source_index", 0) or 0),
        ranges=[],
    )


def _drawing_match_for_line(line: _LineCandidate, drawing: Any) -> _DrawingMatch | None:
    """按长度、中带、线宽、字符覆盖和端点贴合校验单条删除线候选。"""

    if getattr(drawing, "orientation", None) != "horizontal":
        return None
    drawing_bbox = _coerce_bbox(getattr(drawing, "bbox", None))
    if drawing_bbox is None:
        return None
    drawing_length = drawing_bbox[2] - drawing_bbox[0]
    if drawing_length < STRIKETHROUGH_MIN_LENGTH_HEIGHT_RATIO * line.median_height:
        return None
    drawing_center_y = (drawing_bbox[1] + drawing_bbox[3]) / 2
    center_distance_ratio = abs(drawing_center_y - line.center_y) / line.median_height
    if center_distance_ratio > STRIKETHROUGH_CENTER_TOLERANCE_HEIGHT_RATIO:
        return None
    try:
        drawing_width = max(0.0, float(getattr(drawing, "width", 0.0) or 0.0))
    except (TypeError, ValueError):
        return None
    if drawing_width > STRIKETHROUGH_MAX_WIDTH_HEIGHT_RATIO * line.median_height:
        return None

    hit_chars = [
        char
        for char in line.visible_chars
        if drawing_bbox[0] <= (char.bbox[0] + char.bbox[2]) / 2 <= drawing_bbox[2]
    ]
    if not hit_chars:
        return None
    hit_left = min(char.bbox[0] for char in hit_chars)
    hit_right = max(char.bbox[2] for char in hit_chars)
    if (hit_right - hit_left) / drawing_length < STRIKETHROUGH_MIN_TEXT_COVERAGE_RATIO:
        return None
    endpoint_distance = min(
        abs(drawing_bbox[0] - hit_left),
        abs(drawing_bbox[2] - hit_right),
    )
    if endpoint_distance > STRIKETHROUGH_ENDPOINT_TOLERANCE_HEIGHT_RATIO * line.median_height:
        return None

    overlap = max(
        0.0,
        min(line.bbox[2], drawing_bbox[2]) - max(line.bbox[0], drawing_bbox[0]),
    )
    horizontal_overlap_ratio = overlap / max(
        0.01,
        min(line.bbox[2] - line.bbox[0], drawing_length),
    )
    return _DrawingMatch(
        start_index=min(char.source_index for char in hit_chars),
        end_index=max(char.source_index for char in hit_chars) + 1,
        center_distance_ratio=center_distance_ratio,
        horizontal_overlap_ratio=horizontal_overlap_ratio,
    )


def _merge_source_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并重叠或相邻的来源字符区间。"""

    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _line_style_payload(line: _LineCandidate) -> PDFTextStyleLine | None:
    """把来源字符和删除线索引转换为紧凑文本及其半开样式区间。"""

    source_ranges = _merge_source_ranges(line.ranges)
    compact_parts: list[str] = []
    compact_ranges: list[PDFTextStyleRange] = []
    compact_length = 0
    active_start: int | None = None
    active_end = 0
    for char_index, char in enumerate(line.chars):
        fragment = _normalize_match_fragment(char.get("char"))
        if not fragment:
            continue
        fragment_start = compact_length
        compact_parts.append(fragment)
        fragment_end = fragment_start + len(fragment)
        compact_length = fragment_end
        is_struck = any(start <= char_index < end for start, end in source_ranges)
        if is_struck:
            if active_start is None:
                active_start = fragment_start
            active_end = fragment_end
        elif active_start is not None:
            compact_ranges.append(PDFTextStyleRange(active_start, active_end))
            active_start = None
    if active_start is not None:
        compact_ranges.append(PDFTextStyleRange(active_start, active_end))

    text = "".join(compact_parts)
    if not text:
        return None
    return PDFTextStyleLine(
        bbox=line.bbox,
        text=text,
        strikethrough_ranges=tuple(compact_ranges),
        source_index=line.source_index,
    )


def _build_line_center_grid(
    candidates: Sequence[_LineCandidate],
) -> tuple[float, dict[int, list[int]]]:
    """按删除线允许的纵向中带建立行索引，避免逐 drawing 扫描整页文本行。"""

    grid_size = max(
        1.0,
        statistics.median(line.median_height for line in candidates),
    )
    grid: dict[int, list[int]] = {}
    for line_index, line in enumerate(candidates):
        tolerance = STRIKETHROUGH_CENTER_TOLERANCE_HEIGHT_RATIO * line.median_height
        start_cell = math.floor((line.center_y - tolerance) / grid_size)
        end_cell = math.floor((line.center_y + tolerance) / grid_size)
        for cell in range(start_cell, end_cell + 1):
            grid.setdefault(cell, []).append(line_index)
    return grid_size, grid


def detect_pdf_strikethrough_lines(
    lines: Sequence[Any],
    drawing_lines: Sequence[Any],
) -> list[PDFTextStyleLine]:
    """从视觉文本 run 与页面 drawing 中生成全部水平行及其删除线证据。"""

    horizontal_drawings = [
        drawing
        for drawing in drawing_lines
        if getattr(drawing, "orientation", None) == "horizontal"
    ]
    if not horizontal_drawings:
        return []
    candidates = [candidate for line in lines if (candidate := _build_line_candidate(line)) is not None]
    if not candidates:
        return []
    grid_size, line_grid = _build_line_center_grid(candidates)
    for drawing in horizontal_drawings:
        drawing_bbox = _coerce_bbox(getattr(drawing, "bbox", None))
        if drawing_bbox is None:
            continue
        drawing_center_y = (drawing_bbox[1] + drawing_bbox[3]) / 2
        candidate_indices = line_grid.get(math.floor(drawing_center_y / grid_size), [])
        matches = [
            (line_index, match)
            for line_index in candidate_indices
            if (match := _drawing_match_for_line(candidates[line_index], drawing)) is not None
        ]
        if not matches:
            continue
        line_index, best_match = min(
            matches,
            key=lambda item: (
                item[1].center_distance_ratio,
                -item[1].horizontal_overlap_ratio,
                candidates[item[0]].source_index,
            ),
        )
        candidates[line_index].ranges.append(
            (best_match.start_index, best_match.end_index)
        )

    if not any(line.ranges for line in candidates):
        return []
    payloads = [payload for line in candidates if (payload := _line_style_payload(line)) is not None]
    return sorted(payloads, key=lambda line: (line.bbox[1], line.bbox[0], line.source_index))


def _block_bbox_to_page_bbox(value: Any, page_size: tuple[float, float]) -> BBox | None:
    """把 model-list 的归一化 bbox 转回页面 point，同时兼容已是绝对坐标的内部输入。"""

    bbox = _coerce_bbox(value)
    if bbox is None:
        return None
    if all(0.0 <= coordinate <= 1.0 for coordinate in bbox):
        return (
            bbox[0] * page_size[0],
            bbox[1] * page_size[1],
            bbox[2] * page_size[0],
            bbox[3] * page_size[1],
        )
    return bbox


def _bbox_overlap_ratio(first: BBox, second: BBox) -> float:
    """返回 first 面积中落入 second 的比例。"""

    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    first_area = max(0.01, (first[2] - first[0]) * (first[3] - first[1]))
    return intersection_width * intersection_height / first_area


def _line_block_score(line_bbox: BBox, block_bbox: BBox) -> tuple[float, float, float]:
    """计算文本行归属 block 的中心包含、重叠率与紧致度评分。"""

    center_x = (line_bbox[0] + line_bbox[2]) / 2
    center_y = (line_bbox[1] + line_bbox[3]) / 2
    center_inside = float(
        block_bbox[0] <= center_x <= block_bbox[2]
        and block_bbox[1] <= center_y <= block_bbox[3]
    )
    overlap_ratio = _bbox_overlap_ratio(line_bbox, block_bbox)
    block_area = (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1])
    return center_inside, overlap_ratio, -block_area


def _assign_lines_to_blocks(
    blocks: list[dict[str, Any]],
    lines: Sequence[PDFTextStyleLine],
    page_size: tuple[float, float],
) -> dict[int, list[PDFTextStyleLine]]:
    """把每个视觉文本行唯一分配给最匹配的自然语言 block。"""

    target_bboxes = {
        block_index: block_bbox
        for block_index, block in enumerate(blocks)
        if block.get("type") in PDF_NATURAL_TEXT_STYLE_BLOCK_TYPES
        and isinstance(block.get("content"), str)
        and (block_bbox := _block_bbox_to_page_bbox(block.get("bbox"), page_size)) is not None
    }
    assignments: dict[int, list[PDFTextStyleLine]] = {}
    for line in lines:
        matches = [
            (block_index, _line_block_score(line.bbox, block_bbox))
            for block_index, block_bbox in target_bboxes.items()
            if (
                block_bbox[0] <= (line.bbox[0] + line.bbox[2]) / 2 <= block_bbox[2]
                and block_bbox[1] <= (line.bbox[1] + line.bbox[3]) / 2 <= block_bbox[3]
            )
            or _bbox_overlap_ratio(line.bbox, block_bbox) >= 0.5
        ]
        if not matches:
            continue
        block_index, _score = max(matches, key=lambda item: (*item[1], -item[0]))
        assignments.setdefault(block_index, []).append(line)
    for block_lines in assignments.values():
        block_lines.sort(key=lambda line: (line.bbox[1], line.bbox[0], line.source_index))
    return assignments


def _text_tag_has_strikethrough(attrs: str) -> bool:
    """判断 text 标签是否已经显式携带删除线样式。"""

    match = _STYLE_ATTR_RE.search(attrs)
    if match is None:
        return False
    return "strikethrough" in {
        style.strip().lower()
        for style in match.group("style").split(",")
    }


def _pop_inline_tag(stack: list[tuple[str, bool, bool]], tag: str) -> None:
    """从行内标签栈弹出最近的同名元素，容忍损坏嵌套。"""

    for index in range(len(stack) - 1, -1, -1):
        if stack[index][0] == tag:
            del stack[index:]
            return


def _project_content_chars(content: str) -> list[_ProjectedChar]:
    """把 model content 投影为忽略空白和公式的可比较字符，并保留原始 offset。"""

    projected: list[_ProjectedChar] = []
    stack: list[tuple[str, bool, bool]] = []
    cursor = 0
    while cursor < len(content):
        if not any(item[1] for item in stack) and content.startswith(r"\(", cursor):
            formula_end = content.find(r"\)", cursor + 2)
            if formula_end >= 0:
                cursor = formula_end + 2
                continue
        tag_match = _KNOWN_INLINE_TAG_RE.match(content, cursor)
        if tag_match is not None:
            tag = tag_match.group("tag").lower()
            if tag_match.group("close"):
                _pop_inline_tag(stack, tag)
            else:
                attrs = tag_match.group("attrs") or ""
                stack.append(
                    (
                        tag,
                        tag in {"eq", "url"},
                        tag == "s" or (tag == "text" and _text_tag_has_strikethrough(attrs)),
                    )
                )
            cursor = tag_match.end()
            continue

        raw_char = content[cursor]
        if not any(item[1] for item in stack):
            fragment = _normalize_match_fragment(raw_char)
            for value in fragment:
                projected.append(
                    _ProjectedChar(
                        value=value,
                        raw_start=cursor,
                        raw_end=cursor + 1,
                        styleable=not any(item[2] for item in stack),
                    )
                )
        cursor += 1
    return projected


def _all_occurrences(content: str, target: str, start: int) -> list[int]:
    """返回 target 在 content 指定位置后的全部精确匹配起点。"""

    output: list[int] = []
    cursor = start
    while target and (match := content.find(target, cursor)) >= 0:
        output.append(match)
        cursor = match + 1
    return output


def _resolve_fallback_occurrence(
    content: str,
    line: PDFTextStyleLine,
    style_range: PDFTextStyleRange,
    start: int,
) -> int | None:
    """在整行无法对齐时，用唯一删除片段及两侧精确上下文选择位置。"""

    target = line.text[style_range.start : style_range.end]
    occurrences = _all_occurrences(content, target, start)
    if len(occurrences) == 1:
        return occurrences[0]
    if not occurrences:
        return None
    left_context = line.text[max(0, style_range.start - 12) : style_range.start]
    right_context = line.text[style_range.end : style_range.end + 12]
    scored = [
        (
            int(bool(left_context) and content[max(0, position - len(left_context)) : position] == left_context)
            + int(
                bool(right_context)
                and content[
                    position + len(target) : position + len(target) + len(right_context)
                ]
                == right_context
            ),
            position,
        )
        for position in occurrences
    ]
    best_score = max(score for score, _position in scored)
    best_positions = [position for score, position in scored if score == best_score]
    return best_positions[0] if best_score > 0 and len(best_positions) == 1 else None


def _match_style_ranges(
    projected_text: str,
    lines: Sequence[PDFTextStyleLine],
) -> list[PDFTextStyleRange]:
    """按物理行顺序把删除线证据确定性对齐到 block 的可比较文本。"""

    output: list[PDFTextStyleRange] = []
    cursor = 0
    for line in lines:
        line_start = projected_text.find(line.text, cursor)
        if line_start >= 0:
            output.extend(
                PDFTextStyleRange(
                    line_start + style_range.start,
                    line_start + style_range.end,
                )
                for style_range in line.strikethrough_ranges
            )
            cursor = line_start + len(line.text)
            continue
        for style_range in line.strikethrough_ranges:
            position = _resolve_fallback_occurrence(
                projected_text,
                line,
                style_range,
                cursor,
            )
            if position is None:
                logger.debug(
                    "Skip ambiguous PDF strikethrough mapping: "
                    f"line={line.text!r}, target={line.text[style_range.start:style_range.end]!r}"
                )
                continue
            output.append(
                PDFTextStyleRange(
                    position,
                    position + style_range.end - style_range.start,
                )
            )
            cursor = position + style_range.end - style_range.start
    return _merge_style_ranges(output)


def _merge_style_ranges(ranges: Sequence[PDFTextStyleRange]) -> list[PDFTextStyleRange]:
    """合并可比较文本中重叠或直接相邻的样式区间。"""

    merged: list[PDFTextStyleRange] = []
    for style_range in sorted(ranges, key=lambda item: (item.start, item.end)):
        if style_range.start >= style_range.end:
            continue
        if merged and style_range.start <= merged[-1].end:
            merged[-1] = PDFTextStyleRange(
                merged[-1].start,
                max(merged[-1].end, style_range.end),
            )
        else:
            merged.append(style_range)
    return merged


def _raw_style_intervals(
    content: str,
    projected: Sequence[_ProjectedChar],
    ranges: Sequence[PDFTextStyleRange],
) -> list[tuple[int, int]]:
    """把可比较文本区间转换为不跨公式或现有标签的原字符串区间。"""

    intervals: list[tuple[int, int]] = []
    for style_range in ranges:
        current_start: int | None = None
        current_end = 0
        for token in projected[style_range.start : style_range.end]:
            if not token.styleable:
                if current_start is not None:
                    intervals.append((current_start, current_end))
                    current_start = None
                continue
            if current_start is None:
                current_start = token.raw_start
                current_end = token.raw_end
                continue
            gap = content[current_end : token.raw_start]
            if token.raw_start <= current_end or not gap or gap.isspace():
                current_end = max(current_end, token.raw_end)
            else:
                intervals.append((current_start, current_end))
                current_start = token.raw_start
                current_end = token.raw_end
        if current_start is not None:
            intervals.append((current_start, current_end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if start >= end:
            continue
        if merged and (start <= merged[-1][1] or content[merged[-1][1] : start].isspace()):
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _wrap_strikethrough_intervals(content: str, intervals: Sequence[tuple[int, int]]) -> str:
    """从右向左包装删除线区间，确保原 offset 在插入标签时保持有效。"""

    output = content
    for start, end in reversed(intervals):
        output = (
            f'{output[:start]}<text style="strikethrough">'
            f"{output[start:end]}</text>{output[end:]}"
        )
    return output


def apply_pdf_strikethrough_styles(
    blocks: list[dict[str, Any]],
    lines: Sequence[PDFTextStyleLine],
    page_size: tuple[float, float],
) -> None:
    """把页面删除线证据写入自然语言 block content，无法唯一对齐时保持原文。"""

    assignments = _assign_lines_to_blocks(blocks, lines, page_size)
    for block_index, block_lines in assignments.items():
        block = blocks[block_index]
        content = block.get("content")
        if not isinstance(content, str) or not content or not any(
            line.strikethrough_ranges for line in block_lines
        ):
            continue
        projected = _project_content_chars(content)
        projected_text = "".join(token.value for token in projected)
        style_ranges = _match_style_ranges(projected_text, block_lines)
        if not style_ranges:
            continue
        intervals = _raw_style_intervals(content, projected, style_ranges)
        if intervals:
            block["content"] = _wrap_strikethrough_intervals(content, intervals)


__all__ = [
    "PDFTextStyleLine",
    "PDFTextStyleRange",
    "apply_pdf_strikethrough_styles",
    "detect_pdf_strikethrough_lines",
]
