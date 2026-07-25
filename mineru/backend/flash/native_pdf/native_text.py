# Copyright (c) Opendatalab. All rights reserved.

"""将 PDF 原生字符重建为带排版信息的视觉文本行。"""

from __future__ import annotations

import math
import re
import statistics
from typing import Any, Literal, Sequence

from pdftext.schema import Char

from mineru.types import BBox
from mineru.utils.pdf_document import PDFDocument

from .models import (
    _AxisLine,
    _LineItem,
)
from .geometry import (
    _bbox_center_y,
    _bbox_union,
    _bbox_union_many,
    _clip_bbox,
    _coerce_bbox,
    _horizontal_bbox_gap,
    _rotate_bbox_to_upright,
)


_PDF_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _build_native_line_items(
    pdf_lines: Sequence[dict[str, Any]],
    page_size: tuple[float, float],
    *,
    page_rotation: int = 0,
) -> list[_LineItem]:
    """将 pdftext 粗行按字符间隙精修成 Flash 视觉 run。"""

    items: list[_LineItem] = []
    for visual_row_id, pdf_line in enumerate(pdf_lines):
        bbox = _clip_bbox(_coerce_bbox(pdf_line.get("bbox")), page_size)
        if bbox is None:
            continue
        spans = pdf_line.get("spans") or []
        chars = [char for span in spans for char in (span.get("chars") or []) if isinstance(char, dict)]
        coarse_item = _LineItem(
            text="".join(str(span.get("text") or "") for span in spans),
            bbox=bbox,
            angle=(_normalize_pdftext_angle(pdf_line.get("rotation")) + int(page_rotation or 0)) % 360,
            source_index=-1,
            chars=chars,
            visual_row_id=visual_row_id,
        )
        items.extend(_split_native_visual_runs(coarse_item, page_size))

    merged_items = _merge_native_inline_scripts(items, page_size)
    for source_index, item in enumerate(merged_items):
        # source_index 必须在页内唯一，表格投影和失败回滚都依赖该精确成员标识。
        item.source_index = source_index
    return merged_items


def _split_native_visual_runs(
    line: _LineItem,
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """保留字符源顺序与空白信息，并将一个 pdftext 粗行拆成远距视觉 run。"""

    tokens: list[tuple[Char, str, BBox | None, BBox | None]] = []
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        if raw_char in {"\r", "\n"}:
            continue
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle) if bbox is not None else None
        tokens.append((char, raw_char, bbox, local_bbox))

    visible_indices = [
        index
        for index, (_char, raw_char, _bbox, local_bbox) in enumerate(tokens)
        if raw_char.isprintable() and not raw_char.isspace() and local_bbox is not None
    ]
    if not visible_indices:
        text = _normalize_native_run_text(line.text)
        if not text:
            return []
        local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
        line.text = text
        line.effective_height = max(0.1, local_bbox[3] - local_bbox[1])
        return [line]

    glyph_widths = [
        max(0.1, tokens[index][3][2] - tokens[index][3][0])  # type: ignore[index]
        for index in visible_indices
    ]
    median_glyph_width = statistics.median(glyph_widths)
    local_page_width = page_size[1] if line.angle in {90, 270} else page_size[0]
    hard_gap_threshold = max(15.0, 3.0 * median_glyph_width, 0.02 * local_page_width)
    adjacent_gaps: list[float] = []
    for previous, current in zip(visible_indices, visible_indices[1:]):
        previous_bbox = tokens[previous][3]
        current_bbox = tokens[current][3]
        if previous_bbox is None or current_bbox is None:
            continue
        gap = _horizontal_bbox_gap(previous_bbox, current_bbox)
        if gap < hard_gap_threshold:
            adjacent_gaps.append(gap)
    # 少于三个相邻样本无法可靠代表“常规”字距；零间隙也是真实的紧排字距，
    # 必须纳入统计，避免唯一的 15pt cell gap 反过来抬高软拆阈值。
    median_regular_gap = statistics.median(adjacent_gaps) if len(adjacent_gaps) >= 3 else 0.0

    split_indices: list[int] = []
    for previous, current in zip(visible_indices, visible_indices[1:]):
        previous_bbox = tokens[previous][3]
        current_bbox = tokens[current][3]
        if previous_bbox is None or current_bbox is None:
            continue
        gap = _horizontal_bbox_gap(previous_bbox, current_bbox)
        has_source_whitespace = any(tokens[index][1].isspace() for index in range(previous + 1, current))
        soft_gap_threshold = max(
            8.0,
            2.2 * median_glyph_width,
            3.0 * median_regular_gap,
        )
        if gap >= hard_gap_threshold or (has_source_whitespace and gap >= soft_gap_threshold):
            split_indices.append(current)

    ranges: list[tuple[int, int]] = []
    start = 0
    for split_index in split_indices:
        ranges.append((start, split_index))
        start = split_index
    ranges.append((start, len(tokens)))

    output: list[_LineItem] = []
    for run_index, (start, end) in enumerate(ranges):
        run_tokens = tokens[start:end]
        run_text = _normalize_native_run_text("".join(token[1] for token in run_tokens))
        run_bboxes = [
            token[2]
            for token in run_tokens
            if token[2] is not None and token[1].isprintable() and not token[1].isspace()
        ]
        if not run_text or not run_bboxes:
            continue
        run_chars = [token[0] for token in run_tokens]
        run_item = _LineItem(
            text=run_text,
            bbox=_bbox_union_many(run_bboxes),
            angle=line.angle,
            source_index=-1,
            chars=run_chars,
            visual_row_id=line.visual_row_id,
            run_index=run_index,
            split_from_row=len(ranges) > 1,
        )
        _fill_native_typography(run_item, page_size)
        output.append(run_item)
    return output


def _normalize_native_run_text(text: str) -> str:
    """清理原生 run 文本，并把字母后的 PDF 软断词标记转换成 ASCII hyphen。"""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=[A-Za-z])\x02(?=\s*$)", "-", normalized)
    normalized = _sanitize_pdf_control_text(normalized, preserve_newlines=False)
    normalized = re.sub(r"[\t\f\v ]+", " ", normalized)
    return normalized.strip()


def _sanitize_pdf_control_text(text: str, *, preserve_newlines: bool) -> str:
    """删除 PDF 字体编码残留控制字符，并按调用场景决定是否保留物理换行。"""

    normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"(?<=[A-Za-z])\x02(?=[\t ]*(?:\n|$))", "-", normalized)
    normalized = normalized.replace("\t", " ")
    if not preserve_newlines:
        normalized = normalized.replace("\n", "")
    return _PDF_CONTROL_CHAR_RE.sub("", normalized)


def _fill_native_typography(line: _LineItem, page_size: tuple[float, float]) -> None:
    """使用非空字符 bbox 高度和 dominant font 填充原生行排版特征。"""

    heights: list[float] = []
    glyph_widths: list[float] = []
    font_counts: dict[tuple[str, int], int] = {}
    font_weights: dict[tuple[str, int], list[float]] = {}
    valid_font_chars = 0
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        if not raw_char.isprintable() or raw_char.isspace():
            continue
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        if bbox is None:
            continue
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle)
        heights.append(max(0.1, local_bbox[3] - local_bbox[1]))
        glyph_widths.append(max(0.1, local_bbox[2] - local_bbox[0]))
        font = char.get("font") or {}
        font_name = str(font.get("name") or "")
        if not font_name:
            continue
        try:
            font_flags = int(font.get("flags") or 0)
        except (TypeError, ValueError):
            font_flags = 0
        signature = (font_name, font_flags)
        font_counts[signature] = font_counts.get(signature, 0) + 1
        try:
            font_weight = float(font.get("weight"))
        except (TypeError, ValueError):
            font_weight = math.nan
        if math.isfinite(font_weight):
            font_weights.setdefault(signature, []).append(font_weight)
        valid_font_chars += 1

    local_bbox = _rotate_bbox_to_upright(line.bbox, page_size, line.angle)
    line.effective_height = statistics.median(heights) if heights else max(0.1, local_bbox[3] - local_bbox[1])
    line.median_glyph_width = statistics.median(glyph_widths) if glyph_widths else None
    if font_counts and valid_font_chars:
        line.font_signature, dominant_count = max(font_counts.items(), key=lambda item: item[1])
        line.font_coverage = dominant_count / valid_font_chars
        dominant_weights = font_weights.get(line.font_signature, [])
        line.dominant_font_weight = statistics.median(dominant_weights) if dominant_weights else None
    else:
        line.font_signature = None
        line.font_coverage = 0.0
        line.dominant_font_weight = None


def _merge_native_inline_scripts(
    lines: list[_LineItem],
    page_size: tuple[float, float],
) -> list[_LineItem]:
    """以 mutual-nearest 规则把跨粗行的小字号前后置标记合入主体视觉行。"""

    candidates: list[tuple[float, int, int, Literal["prefix", "suffix"]]] = []
    for small_index, small in enumerate(lines):
        compact_text = "".join(char for char in small.text if not char.isspace())
        if not compact_text:
            continue
        small_local_bbox = _rotate_bbox_to_upright(small.bbox, page_size, small.angle)
        for base_index, base in enumerate(lines):
            if small_index == base_index or small.angle != base.angle or small.visual_row_id == base.visual_row_id:
                continue
            if small.effective_height <= 0 or base.effective_height <= 0:
                continue
            height_ratio = small.effective_height / base.effective_height
            if not 0.35 <= height_ratio <= 0.8:
                continue
            if len(compact_text) > 8 and small_local_bbox[2] - small_local_bbox[0] > 3.0 * base.effective_height:
                continue
            base_local_bbox = _rotate_bbox_to_upright(base.bbox, page_size, base.angle)
            vertical_overlap = max(
                0.0,
                min(small_local_bbox[3], base_local_bbox[3]) - max(small_local_bbox[1], base_local_bbox[1]),
            )
            small_height = max(0.1, small_local_bbox[3] - small_local_bbox[1])
            overlap_ratio = vertical_overlap / small_height
            if overlap_ratio < 0.5:
                continue
            center_offset = abs(_bbox_center_y(small_local_bbox) - _bbox_center_y(base_local_bbox))
            if center_offset < max(0.5, 0.12 * base.effective_height):
                # 同基线居中的小字号文本更可能是表格相邻 cell，而不是上下标。
                continue
            gap_limit = max(1.5, 0.35 * base.effective_height)
            edge_options: list[tuple[float, Literal["prefix", "suffix"], float]] = []
            for position, gap in (
                ("prefix", base_local_bbox[0] - small_local_bbox[2]),
                ("suffix", small_local_bbox[0] - base_local_bbox[2]),
            ):
                if -0.35 * base.effective_height <= gap <= gap_limit:
                    edge_options.append((abs(gap), position, gap))
            if not edge_options:
                continue
            _edge_distance, position, gap = min(edge_options, key=lambda item: item[0])

            outside_offset = max(
                base_local_bbox[1] - small_local_bbox[1],
                small_local_bbox[3] - base_local_bbox[3],
            )
            tightly_attached = abs(gap) <= max(1.0, 0.1 * base.effective_height)
            if outside_offset < max(0.5, 0.08 * base.effective_height) and not tightly_attached:
                # 紧贴边缘的小字号上下标可能完全落入高字形 bbox；其余内嵌小字仍按普通 cell 排除。
                continue
            metric = abs(gap) + (1.0 - overlap_ratio) * base.effective_height
            candidates.append((metric, small_index, base_index, position))

    best_base_for_small: dict[int, tuple[float, int, Literal["prefix", "suffix"]]] = {}
    best_small_for_base: dict[tuple[int, str], tuple[float, int]] = {}
    for metric, small_index, base_index, position in candidates:
        if small_index not in best_base_for_small or metric < best_base_for_small[small_index][0]:
            best_base_for_small[small_index] = (metric, base_index, position)
        base_key = (base_index, position)
        if base_key not in best_small_for_base or metric < best_small_for_base[base_key][0]:
            best_small_for_base[base_key] = (metric, small_index)

    matches: dict[int, dict[str, int]] = {}
    for small_index, (_metric, base_index, position) in best_base_for_small.items():
        if best_small_for_base.get((base_index, position), (math.inf, -1))[1] == small_index:
            matches.setdefault(base_index, {})[position] = small_index

    consumed_small_indices = {small_index for positions in matches.values() for small_index in positions.values()}
    merged_base_indices: set[int] = set()

    def merge_children(base_index: int, visiting: set[int]) -> None:
        """先合并更小的依赖标记，再把当前完整节点递归合入更大的主体行。"""

        if base_index in merged_base_indices or base_index in visiting:
            return
        visiting.add(base_index)
        positions = matches.get(base_index, {})
        for child_index in positions.values():
            merge_children(child_index, visiting)
        base = lines[base_index]
        merged_bbox = base.bbox
        merged_chars = list(base.chars)
        if "prefix" in positions:
            prefix_index = positions["prefix"]
            prefix = lines[prefix_index]
            base.text = f"{prefix.text.strip()} {base.text.lstrip()}"
            merged_bbox = _bbox_union(merged_bbox, prefix.bbox)
            merged_chars = [*prefix.chars, *merged_chars]
            base.split_from_row = base.split_from_row or prefix.split_from_row
        if "suffix" in positions:
            suffix_index = positions["suffix"]
            suffix = lines[suffix_index]
            base.text = f"{base.text.rstrip()}{suffix.text.strip()}"
            merged_bbox = _bbox_union(merged_bbox, suffix.bbox)
            merged_chars.extend(suffix.chars)
            base.split_from_row = base.split_from_row or suffix.split_from_row
        base.bbox = merged_bbox
        base.chars = merged_chars
        _fill_native_typography(base, page_size)
        visiting.remove(base_index)
        merged_base_indices.add(base_index)

    # 从最终不会被消费的根主体开始，确保 small -> medium -> large 链不会丢失最小节点。
    root_base_indices = [base_index for base_index in matches if base_index not in consumed_small_indices]
    for base_index in root_base_indices:
        merge_children(base_index, set())
    for base_index in matches:
        merge_children(base_index, set())

    output = [line for index, line in enumerate(lines) if index not in consumed_small_indices]
    output.sort(key=lambda item: (item.visual_row_id if item.visual_row_id is not None else math.inf, item.run_index))
    return output


def _normalize_pdftext_angle(value: Any) -> int:
    """将 pdftext 弧度方向就近归一到四个标准角度。"""

    try:
        angle_radians = float(value or 0.0)
    except (TypeError, ValueError):
        return 0
    angle_degrees = math.degrees(angle_radians)
    normalized = int(round(angle_degrees / 90.0) * 90) % 360
    return normalized if normalized in {0, 90, 180, 270} else 0


def _get_pdf_drawing_lines(pdf_doc: PDFDocument, page_idx: int) -> list[_AxisLine]:
    """读取 PDFDocument 的公共绘图线结果，并隔离具体 PDFium 类型。"""

    output: list[_AxisLine] = []
    for drawing_line in pdf_doc.get_page_drawing_lines(page_idx):
        bbox = _coerce_bbox(drawing_line.bbox)
        if bbox is None:
            continue
        output.append(
            _AxisLine(
                bbox=bbox,
                width=max(0.0, float(drawing_line.width)),
                orientation=drawing_line.orientation,
            )
        )
    return output


def _median_native_glyph_width(line: _LineItem, page_size: tuple[float, float]) -> float | None:
    """返回单个原生 run 的可见字符中位宽度，缺少字符时返回空。"""

    if line.median_glyph_width is not None:
        return line.median_glyph_width
    widths: list[float] = []
    for char in line.chars:
        raw_char = str(char.get("char") or "")
        bbox = _clip_bbox(_coerce_bbox(char.get("bbox")), page_size)
        if not raw_char.isprintable() or raw_char.isspace() or bbox is None:
            continue
        local_bbox = _rotate_bbox_to_upright(bbox, page_size, line.angle)
        widths.append(max(0.1, local_bbox[2] - local_bbox[0]))
    line.median_glyph_width = statistics.median(widths) if widths else None
    return line.median_glyph_width

