# Copyright (c) Opendatalab. All rights reserved.
"""Native PDF 表格字符选择、视觉组行和单元格文本重建。"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from pdftext.schema import Char

from mineru.types import BBox
from mineru.utils.spatial_text import _normalize_table_text

from .contracts import (
    NativeTableGlyph,
    NativeTableInput,
    NativeTableText,
    NativeTableTextRow,
    NativeTableToken,
)
from .geometry import (
    bbox_center,
    bbox_intersection,
    bbox_union,
    normalize_angle,
    normalize_bbox,
    page_bbox_to_table_local,
)


@dataclass(frozen=True, slots=True)
class _PendingGlyph:
    """保存尚未分配视觉行的局部 PDF 字符。"""

    glyph_id: int
    source_index: int
    text: str
    bbox: BBox
    explicit_space_before: bool = False
    explicit_break_before: bool = False


def _select_pending_glyphs(table_input: NativeTableInput) -> list[_PendingGlyph]:
    """按字符中心选择表格内可见字符，并转换到正向局部坐标。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return []
    selected: list[tuple[int, int, Char]] = []
    for fallback_index, char in enumerate(table_input.chars):
        char_bbox = normalize_bbox(char.get("bbox"))
        if char_bbox is None:
            continue
        center_x, center_y = bbox_center(char_bbox)
        if not (table_bbox[0] <= center_x <= table_bbox[2] and table_bbox[1] <= center_y <= table_bbox[3]):
            continue
        try:
            source_index = int(char.get("char_idx", fallback_index))
        except (TypeError, ValueError):
            source_index = fallback_index
        selected.append((source_index, fallback_index, char))

    selected.sort(key=lambda item: (item[0], item[1]))
    output: list[_PendingGlyph] = []
    angle = normalize_angle(table_input.angle)
    pending_space = False
    pending_break = False
    for _source_sort, fallback_index, char in selected:
        raw_text = str(char.get("char") or "")
        if not raw_text:
            continue
        if raw_text in {"\r", "\n"}:
            pending_break = True
            pending_space = False
            continue
        if raw_text.isspace():
            if not pending_break:
                pending_space = True
            continue
        text = _normalize_table_text(raw_text)
        if not text or text.isspace():
            continue
        absolute_bbox = normalize_bbox(char.get("bbox"))
        if absolute_bbox is None:
            continue
        local_bbox = page_bbox_to_table_local(
            absolute_bbox,
            table_bbox,
            angle,
        )
        if local_bbox is None or local_bbox[3] - local_bbox[1] < 0.5:
            continue
        try:
            source_index = int(char.get("char_idx", fallback_index))
        except (TypeError, ValueError):
            source_index = fallback_index
        output.append(
            _PendingGlyph(
                glyph_id=len(output),
                source_index=source_index,
                text=text,
                bbox=local_bbox,
                explicit_space_before=pending_space,
                explicit_break_before=pending_break,
            )
        )
        pending_space = False
        pending_break = False
    return output


def _vertical_overlap_ratio(first: BBox, second: BBox) -> float:
    """返回两个字符框相对较小高度的垂直交叠比例。"""

    overlap = min(first[3], second[3]) - max(first[1], second[1])
    minimum_height = min(first[3] - first[1], second[3] - second[1])
    if overlap <= 0 or minimum_height <= 0:
        return 0.0
    return min(1.0, overlap / minimum_height)


def _assign_visual_rows(
    pending: list[_PendingGlyph],
    median_height: float,
) -> list[list[_PendingGlyph]]:
    """按垂直交叠和中心距离把字符聚成稳定视觉行。"""

    sorted_glyphs = sorted(
        pending,
        key=lambda glyph: (
            (glyph.bbox[1] + glyph.bbox[3]) / 2.0,
            glyph.bbox[0],
            glyph.glyph_id,
        ),
    )
    rows: list[list[_PendingGlyph]] = []
    row_bboxes: list[BBox] = []
    for glyph in sorted_glyphs:
        glyph_center_y = (glyph.bbox[1] + glyph.bbox[3]) / 2.0
        best_index: int | None = None
        best_distance = float("inf")
        for row_index, row_bbox in enumerate(row_bboxes):
            row_center_y = (row_bbox[1] + row_bbox[3]) / 2.0
            distance = abs(glyph_center_y - row_center_y)
            if (
                _vertical_overlap_ratio(glyph.bbox, row_bbox) >= 0.45 or distance <= max(0.75, median_height * 0.40)
            ) and distance < best_distance:
                best_index = row_index
                best_distance = distance
        if best_index is None:
            rows.append([glyph])
            row_bboxes.append(glyph.bbox)
        else:
            rows[best_index].append(glyph)
            row_bboxes[best_index] = bbox_union([row_bboxes[best_index], glyph.bbox])

    ordered = sorted(
        zip(rows, row_bboxes),
        key=lambda item: (item[1][1], item[1][0]),
    )
    return [sorted(row, key=lambda glyph: (glyph.bbox[0], glyph.bbox[1], glyph.glyph_id)) for row, _bbox in ordered]


def _contains_cjk(text: str) -> bool:
    """判断文本是否包含常见中日韩统一表意字符。"""

    return any("\u3400" <= char <= "\u9fff" for char in text)


def _join_glyph_line(
    glyphs: list[NativeTableGlyph] | list[_PendingGlyph],
    median_height: float,
) -> str:
    """按字符间距重建单行文本，避免在中文字符之间强插空格。"""

    if not glyphs:
        return ""
    ordered = sorted(glyphs, key=lambda glyph: (glyph.bbox[0], glyph.bbox[1], glyph.glyph_id))
    parts = [ordered[0].text]
    previous = ordered[0]
    for glyph in ordered[1:]:
        gap = glyph.bbox[0] - previous.bbox[2]
        previous_tail = previous.text[-1:] or ""
        current_head = glyph.text[:1] or ""
        needs_space = glyph.explicit_space_before or (
            not glyph.explicit_break_before
            and gap > max(0.6, median_height * 0.20)
            and not _contains_cjk(previous_tail + current_head)
            and previous_tail.isprintable()
            and current_head.isprintable()
        )
        if needs_space and not parts[-1].endswith(" "):
            parts.append(" ")
        parts.append(glyph.text)
        previous = glyph
    return "".join(parts).strip()


def _tokenize_row(
    glyphs: list[NativeTableGlyph],
    median_width: float,
    median_height: float,
) -> tuple[NativeTableToken, ...]:
    """按显著水平间隙把视觉行拆成供列推断使用的文本项。"""

    if not glyphs:
        return ()
    split_gap = max(2.0 * median_width, 0.70 * median_height, 2.0)
    groups: list[list[NativeTableGlyph]] = [[glyphs[0]]]
    for glyph in glyphs[1:]:
        if glyph.bbox[0] - groups[-1][-1].bbox[2] > split_gap:
            groups.append([glyph])
        else:
            groups[-1].append(glyph)
    tokens: list[NativeTableToken] = []
    for group in groups:
        content = _join_glyph_line(group, median_height)
        if not content:
            continue
        tokens.append(
            NativeTableToken(
                text=content,
                bbox=bbox_union(item.bbox for item in group),
                glyph_ids=tuple(item.glyph_id for item in group),
                source_char_indices=tuple(item.source_index for item in group),
            )
        )
    return tuple(tokens)


def build_native_table_text(table_input: NativeTableInput) -> NativeTableText | None:
    """把原生 PDF 字符转换为正向表格字形、视觉行和文本项。"""

    pending = _select_pending_glyphs(table_input)
    if not pending:
        return None
    widths = [glyph.bbox[2] - glyph.bbox[0] for glyph in pending if glyph.bbox[2] > glyph.bbox[0]]
    heights = [glyph.bbox[3] - glyph.bbox[1] for glyph in pending if glyph.bbox[3] > glyph.bbox[1]]
    median_width = max(0.1, float(statistics.median(widths)) if widths else 1.0)
    median_height = max(0.1, float(statistics.median(heights)) if heights else 1.0)
    pending_rows = _assign_visual_rows(pending, median_height)

    glyphs: list[NativeTableGlyph] = []
    rows: list[NativeTableTextRow] = []
    for row_index, pending_row in enumerate(pending_rows):
        row_glyphs = [
            NativeTableGlyph(
                glyph_id=item.glyph_id,
                source_index=item.source_index,
                text=item.text,
                bbox=item.bbox,
                visual_row=row_index,
                explicit_space_before=item.explicit_space_before,
                explicit_break_before=item.explicit_break_before,
            )
            for item in pending_row
        ]
        glyphs.extend(row_glyphs)
        rows.append(
            NativeTableTextRow(
                row_index=row_index,
                bbox=bbox_union(item.bbox for item in row_glyphs),
                tokens=_tokenize_row(row_glyphs, median_width, median_height),
                glyph_ids=tuple(item.glyph_id for item in row_glyphs),
            )
        )
    glyphs.sort(key=lambda item: item.glyph_id)
    return NativeTableText(
        glyphs=tuple(glyphs),
        rows=tuple(rows),
        median_glyph_width=median_width,
        median_glyph_height=median_height,
    )


def build_cell_text(
    glyphs: list[NativeTableGlyph],
    median_height: float,
) -> str:
    """按视觉行和局部横向顺序重建文本，并将同一单元格内的多行直接拼接。"""

    if not glyphs:
        return ""
    grouped: dict[int, list[NativeTableGlyph]] = {}
    for glyph in glyphs:
        grouped.setdefault(glyph.visual_row, []).append(glyph)
    return "".join(line for row_index in sorted(grouped) if (line := _join_glyph_line(grouped[row_index], median_height)))


def glyph_overlap_ratio(glyph: NativeTableGlyph, bbox: BBox) -> float:
    """返回字符框面积被目标单元格覆盖的比例。"""

    intersection = bbox_intersection(glyph.bbox, bbox)
    glyph_area = (glyph.bbox[2] - glyph.bbox[0]) * (glyph.bbox[3] - glyph.bbox[1])
    if intersection is None or glyph_area <= 0:
        return 0.0
    intersection_area = (intersection[2] - intersection[0]) * (intersection[3] - intersection[1])
    return min(1.0, intersection_area / glyph_area)


__all__ = [
    "build_cell_text",
    "build_native_table_text",
    "glyph_overlap_ratio",
]
