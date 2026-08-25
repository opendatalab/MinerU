# Copyright (c) Opendatalab. All rights reserved.
"""Native PDF 表格候选的网格校验、字符落格和 HTML 序列化。"""

from __future__ import annotations

import html
from dataclasses import dataclass

from mineru.types import BBox

from .contracts import (
    NativeTableCandidate,
    NativeTableCandidateSource,
    NativeTableCell,
    NativeTableGlyph,
    NativeTableText,
)
from .geometry import bbox_center
from .text import build_cell_text, glyph_overlap_ratio

MIN_TEXT_CAPTURE = 0.98


@dataclass(frozen=True, slots=True)
class GridCellSpec:
    """保存尚未回填文本的逻辑网格单元格。"""

    row: int
    col: int
    rowspan: int
    colspan: int
    bbox: BBox


def _validate_grid_specs(
    rows: int,
    cols: int,
    specs: tuple[GridCellSpec, ...],
    *,
    allow_single_row: bool = False,
    allow_single_column: bool = False,
) -> bool:
    """校验逻辑单元格完整覆盖矩形网格且没有重叠。"""

    minimum_rows = 1 if allow_single_row else 2
    minimum_cols = 1 if allow_single_column else 2
    if rows < minimum_rows or cols < minimum_cols or (rows == 1 and cols == 1) or not specs:
        return False
    occupied = [[False for _ in range(cols)] for _ in range(rows)]
    for spec in specs:
        if (
            spec.row < 0
            or spec.col < 0
            or spec.rowspan < 1
            or spec.colspan < 1
            or spec.row + spec.rowspan > rows
            or spec.col + spec.colspan > cols
            or spec.bbox[2] <= spec.bbox[0]
            or spec.bbox[3] <= spec.bbox[1]
        ):
            return False
        for row_index in range(spec.row, spec.row + spec.rowspan):
            for col_index in range(spec.col, spec.col + spec.colspan):
                if occupied[row_index][col_index]:
                    return False
                occupied[row_index][col_index] = True
    return all(all(row) for row in occupied)


def _choose_cell_for_glyph(
    glyph: NativeTableGlyph,
    specs: tuple[GridCellSpec, ...],
) -> tuple[int | None, bool]:
    """按字符面积交叠选择唯一单元格，并标记近似平局的边界歧义。"""

    overlaps = [(glyph_overlap_ratio(glyph, spec.bbox), index) for index, spec in enumerate(specs)]
    overlaps.sort(reverse=True)
    best_ratio, best_index = overlaps[0]
    if best_ratio <= 0:
        center_x, center_y = bbox_center(glyph.bbox)
        containing = [
            index
            for index, spec in enumerate(specs)
            if spec.bbox[0] <= center_x <= spec.bbox[2] and spec.bbox[1] <= center_y <= spec.bbox[3]
        ]
        return (containing[0], False) if len(containing) == 1 else (None, False)
    ambiguous = len(overlaps) > 1 and overlaps[1][0] >= 0.45 and abs(best_ratio - overlaps[1][0]) <= 0.02
    return best_index, ambiguous


def _order_consistency(
    text: NativeTableText,
    assignments: dict[int, int],
    specs: tuple[GridCellSpec, ...],
) -> float:
    """衡量每条视觉行中的字符单元格序号是否保持从左到右单调。"""

    comparable = 0
    ordered_pairs = 0
    glyph_by_id = {glyph.glyph_id: glyph for glyph in text.glyphs}
    for row in text.rows:
        glyphs = sorted(
            (glyph_by_id[glyph_id] for glyph_id in row.glyph_ids if glyph_id in assignments),
            key=lambda glyph: (glyph.bbox[0], glyph.glyph_id),
        )
        previous_position: tuple[int, int] | None = None
        for glyph in glyphs:
            spec = specs[assignments[glyph.glyph_id]]
            position = spec.row, spec.col
            if previous_position is not None:
                comparable += 1
                if position >= previous_position:
                    ordered_pairs += 1
            previous_position = position
    return ordered_pairs / comparable if comparable else 1.0


def build_candidate(
    *,
    source: NativeTableCandidateSource,
    rows: int,
    cols: int,
    specs: tuple[GridCellSpec, ...],
    text: NativeTableText,
    structure_support: float,
    row_stability: float,
    column_stability: float,
    issues: tuple[str, ...] = (),
    allow_single_row: bool = False,
    allow_single_column: bool = False,
) -> NativeTableCandidate | None:
    """校验网格、唯一分配字符并计算统一质量分。"""

    if not _validate_grid_specs(
        rows,
        cols,
        specs,
        allow_single_row=allow_single_row,
        allow_single_column=allow_single_column,
    ):
        return None
    assignments: dict[int, int] = {}
    cell_glyphs: list[list[NativeTableGlyph]] = [[] for _ in specs]
    ambiguous_count = 0
    for glyph in text.glyphs:
        cell_index, ambiguous = _choose_cell_for_glyph(glyph, specs)
        if cell_index is None:
            continue
        assignments[glyph.glyph_id] = cell_index
        cell_glyphs[cell_index].append(glyph)
        ambiguous_count += int(ambiguous)

    glyph_count = len(text.glyphs)
    text_capture = len(assignments) / glyph_count if glyph_count else 0.0
    ambiguous_ratio = ambiguous_count / glyph_count if glyph_count else 0.0
    if text_capture < MIN_TEXT_CAPTURE or ambiguous_ratio > 0.02:
        return None
    cells = tuple(
        NativeTableCell(
            row=spec.row,
            col=spec.col,
            rowspan=spec.rowspan,
            colspan=spec.colspan,
            bbox=spec.bbox,
            content=build_cell_text(
                cell_glyphs[index],
                text.median_glyph_height,
            ),
            source_char_indices=tuple(
                glyph.source_index
                for glyph in sorted(
                    cell_glyphs[index],
                    key=lambda item: (item.visual_row, item.bbox[0], item.glyph_id),
                )
            ),
        )
        for index, spec in enumerate(specs)
    )
    order_consistency = _order_consistency(text, assignments, specs)
    normalized_structure = max(0.0, min(1.0, structure_support))
    normalized_row = max(0.0, min(1.0, row_stability))
    normalized_column = max(0.0, min(1.0, column_stability))
    score = min(
        text_capture,
        normalized_structure,
        normalized_row,
        normalized_column,
        order_consistency,
    )
    return NativeTableCandidate(
        source=source,
        rows=rows,
        cols=cols,
        cells=cells,
        score=score,
        text_capture=text_capture,
        structure_support=normalized_structure,
        row_stability=normalized_row,
        column_stability=normalized_column,
        order_consistency=order_consistency,
        issues=issues,
    )


def serialize_candidate_html(candidate: NativeTableCandidate) -> str:
    """把合法候选序列化为稳定、转义且不猜测表头语义的 HTML。"""

    cells_by_row: dict[int, list[NativeTableCell]] = {}
    for cell in candidate.cells:
        cells_by_row.setdefault(cell.row, []).append(cell)
    rendered_rows: list[str] = []
    for row_index in range(candidate.rows):
        rendered_cells: list[str] = []
        for cell in sorted(cells_by_row.get(row_index, []), key=lambda item: item.col):
            attributes = []
            if cell.rowspan > 1:
                attributes.append(f' rowspan="{cell.rowspan}"')
            if cell.colspan > 1:
                attributes.append(f' colspan="{cell.colspan}"')
            escaped_content = html.escape(cell.content, quote=False)
            rendered_cells.append(f"<td{''.join(attributes)}>{escaped_content}</td>")
        rendered_rows.append(f"<tr>{''.join(rendered_cells)}</tr>")
    return f"<table><tbody>{''.join(rendered_rows)}</tbody></table>"


__all__ = [
    "GridCellSpec",
    "build_candidate",
    "serialize_candidate_html",
]
