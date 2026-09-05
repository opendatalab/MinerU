# Copyright (c) Opendatalab. All rights reserved.

"""使用 PDF 原生字符几何为高置信表格 HTML 恢复上下标。"""

from __future__ import annotations

import html
from collections import defaultdict
from typing import Any

from pdftext.schema import Char

from ....types import BBox
from .geometry import _bbox_union_many, _coerce_bbox
from .line_merging import _merge_overlapping_inline_text_clusters
from .models import _LineItem
from .native_text import _fill_native_typography
from .script_geometry import ScriptRole
from .table_recovery.candidate import serialize_native_table_html
from .table_recovery.contracts import (
    NativeTableCell,
    NativeTableGlyph,
    NativeTableInput,
    NativeTableResult,
    NativeTableRule,
    NativeTableText,
)
from .table_recovery.geometry import page_bbox_to_table_local
from .table_recovery.text import build_cell_text_parts
from .inline.scripts import _fraction_member_indices, _script_line_char_roles


def _table_char_map(chars: tuple[Char, ...]) -> dict[int, Char]:
    """按合法 char_idx 建立页面字符查询表，重复索引保留首项。"""

    output: dict[int, Char] = {}
    for fallback_index, char in enumerate(chars):
        char_idx = char.get("char_idx", fallback_index)
        if isinstance(char_idx, bool) or not isinstance(char_idx, int):
            continue
        output.setdefault(char_idx, char)
    return output


def _cell_glyphs(
    result: NativeTableResult,
    cell: NativeTableCell,
) -> list[NativeTableGlyph]:
    """按 cell 的稳定字符来源收集并恢复视觉行内顺序。"""

    glyph_by_source = {glyph.source_index: glyph for glyph in result.text.glyphs}
    return sorted(
        (glyph_by_source[source_index] for source_index in cell.source_char_indices if source_index in glyph_by_source),
        key=lambda glyph: (glyph.visual_row, glyph.bbox[0], glyph.bbox[1], glyph.glyph_id),
    )


def _cell_visual_lines(
    glyphs: list[NativeTableGlyph],
    chars_by_source: dict[int, Char],
    page_size: tuple[float, float],
    angle: int,
) -> list[_LineItem]:
    """把一个 cell 的 glyph 按 visual row 转换成正文公式分段使用的行。"""

    grouped: dict[int, list[NativeTableGlyph]] = defaultdict(list)
    for glyph in glyphs:
        grouped[glyph.visual_row].append(glyph)

    lines: list[_LineItem] = []
    for visual_row, row_glyphs in sorted(grouped.items()):
        ordered = sorted(row_glyphs, key=lambda glyph: (glyph.bbox[0], glyph.bbox[1], glyph.glyph_id))
        chars = [chars_by_source[glyph.source_index] for glyph in ordered if glyph.source_index in chars_by_source]
        bboxes = [bbox for char in chars if (bbox := _coerce_bbox(char.get("bbox"))) is not None]
        if not chars or not bboxes:
            continue
        line = _LineItem(
            text="".join(glyph.text for glyph in ordered),
            bbox=_bbox_union_many(bboxes),
            angle=angle,
            source_index=visual_row,
            chars=chars,
            visual_row_id=visual_row,
        )
        _fill_native_typography(line, page_size)
        lines.append(line)
    return lines


def _rule_overlaps_boundary(
    rule_bbox: BBox,
    boundary_y: float,
    boundary_left: float,
    boundary_right: float,
    tolerance: float,
) -> bool:
    """判断局部横线是否与一个逻辑 cell 的水平边界重合。"""

    rule_y = (rule_bbox[1] + rule_bbox[3]) / 2.0
    if abs(rule_y - boundary_y) > tolerance:
        return False
    overlap = min(rule_bbox[2], boundary_right) - max(rule_bbox[0], boundary_left)
    rule_width = rule_bbox[2] - rule_bbox[0]
    boundary_width = boundary_right - boundary_left
    return overlap > 0 and overlap / max(1e-6, min(rule_width, boundary_width)) >= 0.5


def _non_grid_fraction_rules(
    table_input: NativeTableInput,
    result: NativeTableResult,
) -> list[NativeTableRule]:
    """移除与恢复后 cell 边界重合的横线，仅保留可能的分式线。"""

    tolerance = max(0.75, 0.12 * result.text.median_glyph_height)
    output: list[NativeTableRule] = []
    for rule in table_input.drawing_lines:
        if rule.orientation != "horizontal":
            continue
        local_bbox = page_bbox_to_table_local(rule.bbox, table_input.table_bbox, table_input.angle)
        if local_bbox is None:
            continue
        is_grid_rule = any(
            _rule_overlaps_boundary(
                local_bbox,
                boundary_y,
                cell.bbox[0],
                cell.bbox[2],
                max(tolerance, rule.width),
            )
            for cell in result.cells
            for boundary_y in (cell.bbox[1], cell.bbox[3])
        )
        if not is_grid_rule:
            output.append(rule)
    return output


def _drop_shifted_word_prefixes(chars: list[dict[str, Any]], roles: list[ScriptRole]) -> None:
    """拒绝普通字母词仅首字母偏移的弱候选，完整同基线词由精炼层闭合。"""

    start = 0
    while start < len(chars):
        if not str(chars[start].get("char", "")).isalpha():
            start += 1
            continue
        end = start + 1
        while end < len(chars) and str(chars[end].get("char", "")).isalpha():
            end += 1
        if end - start >= 2 and roles[start] != "body" and all(role == "body" for role in roles[start + 1 : end]):
            roles[start] = "body"
        start = end


def _cell_script_roles(
    glyphs: list[NativeTableGlyph],
    chars_by_source: dict[int, Char],
    page_size: tuple[float, float],
    angle: int,
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
    fraction_members: set[int],
) -> dict[int, ScriptRole]:
    """在 cell 内按正文同款二维公式分段和字符几何返回稳定角色。"""

    lines = _cell_visual_lines(glyphs, chars_by_source, page_size, angle)
    if not lines:
        return {}
    segmented_lines = _merge_overlapping_inline_text_clusters(lines, page_size, [])
    roles_by_source: dict[int, ScriptRole] = {}
    for line in segmented_lines:
        chars, roles, _body_counts, _formula_flags = _script_line_char_roles(
            line,
            page_size,
            tight_bboxes,
            origins,
            fraction_members,
        )
        _drop_shifted_word_prefixes(chars, roles)
        for char, role in zip(chars, roles, strict=True):
            char_idx = char.get("char_idx")
            if not isinstance(char_idx, int) or role == "body":
                continue
            previous = roles_by_source.get(char_idx)
            roles_by_source[char_idx] = role if previous in {None, role} else "body"
    return {source_index: role for source_index, role in roles_by_source.items() if role != "body"}


def _render_styled_cell(
    cell: NativeTableCell,
    glyphs: list[NativeTableGlyph],
    median_height: float,
    roles: dict[int, ScriptRole],
) -> str:
    """按旧文本重建规则安全插入平坦的 sup/sub 标签。"""

    parts = build_cell_text_parts(glyphs, median_height)
    if "".join(text for text, _source_index in parts) != cell.content:
        return html.escape(cell.content, quote=False)

    rendered: list[str] = []
    active_role: ScriptRole = "body"
    active_parts: list[str] = []

    def flush() -> None:
        """提交当前同角色片段，并只生成受信的上下标标签。"""

        nonlocal active_parts
        if not active_parts:
            return
        content = html.escape("".join(active_parts), quote=False)
        if active_role == "sup":
            rendered.append(f"<sup>{content}</sup>")
        elif active_role == "sub":
            rendered.append(f"<sub>{content}</sub>")
        else:
            rendered.append(content)
        active_parts = []

    for text, source_index in parts:
        role = roles.get(source_index, "body") if source_index is not None else "body"
        if role != active_role:
            flush()
            active_role = role
        active_parts.append(text)
    flush()
    return "".join(rendered)


def render_native_table_html_with_scripts(
    result: NativeTableResult,
    table_input: NativeTableInput,
    tight_bboxes: dict[int, BBox],
    origins: dict[int, tuple[float, float]],
) -> str:
    """为高置信原生表格恢复上下标，证据不足时返回原始 HTML。"""

    if not tight_bboxes or not origins:
        return result.html
    if not isinstance(getattr(result, "text", None), NativeTableText):
        return result.html
    chars_by_source = _table_char_map(table_input.chars)
    fraction_rules = _non_grid_fraction_rules(table_input, result)
    cell_glyphs = {(cell.row, cell.col): _cell_glyphs(result, cell) for cell in result.cells}
    cell_roles: dict[tuple[int, int], dict[int, ScriptRole]] = {}
    for cell in result.cells:
        key = (cell.row, cell.col)
        glyphs = cell_glyphs[key]
        cell_chars = [chars_by_source[glyph.source_index] for glyph in glyphs if glyph.source_index in chars_by_source]
        fraction_members = _fraction_member_indices(
            table_input.page_size,
            cell_chars,
            tight_bboxes,
            fraction_rules,
            table_input.angle,
        )
        roles = _cell_script_roles(
            glyphs,
            chars_by_source,
            table_input.page_size,
            table_input.angle,
            tight_bboxes,
            origins,
            fraction_members,
        )
        if roles:
            cell_roles[key] = roles
    if not cell_roles:
        return result.html
    return serialize_native_table_html(
        result.rows,
        result.cells,
        render_cell=lambda cell: _render_styled_cell(
            cell,
            cell_glyphs[(cell.row, cell.col)],
            result.text.median_glyph_height,
            cell_roles.get((cell.row, cell.col), {}),
        ),
    )


__all__ = ["render_native_table_html_with_scripts"]
