# Copyright (c) Opendatalab. All rights reserved.
"""基于稀疏规则、行底纹和原生文本对齐恢复少线或无线表格。"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Any

from .candidate import GridCellSpec, build_candidate
from .contracts import (
    NativeTableCandidate,
    NativeTableCandidateSource,
    NativeTableInput,
    NativeTableText,
    NativeTableTextRow,
)
from .geometry import (
    bbox_union,
    normalize_angle,
    normalize_bbox,
    page_bbox_to_table_local,
    table_local_size,
)

MIN_COLUMN_ANCHOR_SUPPORT = 0.60


@dataclass(frozen=True, slots=True)
class _LogicalRowGrouping:
    """保存视觉行分组结果及不宜自动合并的稠密行对。"""

    rows: tuple[NativeTableTextRow, ...]
    dense_ambiguities: tuple[tuple[int, int], ...]
    subset_merges: tuple[tuple[int, int], ...]


def _infer_target_column_count(text: NativeTableText) -> int | None:
    """从多行文本项数量中选择有重复证据的最大叶子列数。"""

    counts = [len(row.tokens) for row in text.rows if len(row.tokens) >= 2]
    if not counts:
        return None
    occurrences = Counter(counts)
    for count in sorted(occurrences, reverse=True):
        if occurrences[count] >= 2:
            return count
    return max(counts) if len(text.rows) <= 3 else None


def _infer_column_tracks(
    text: NativeTableText,
    width: float,
    target_cols: int,
) -> tuple[list[float], float, float] | None:
    """用最稠密视觉行的相邻文本间隙推断全局叶子列边界。"""

    dense_rows = [row for row in text.rows if len(row.tokens) == target_cols]
    if len(dense_rows) < 2 and len(text.rows) > 3:
        return None
    if not dense_rows:
        return None
    boundaries: list[float] = []
    gap_supports: list[float] = []
    minimum_gap = max(1.0, 0.15 * text.median_glyph_height)
    for col_index in range(target_cols - 1):
        midpoints: list[float] = []
        valid_gap_count = 0
        for row in dense_rows:
            left_token = row.tokens[col_index]
            right_token = row.tokens[col_index + 1]
            gap = right_token.bbox[0] - left_token.bbox[2]
            if gap >= minimum_gap:
                valid_gap_count += 1
            midpoints.append((left_token.bbox[2] + right_token.bbox[0]) / 2.0)
        if valid_gap_count / len(dense_rows) < MIN_COLUMN_ANCHOR_SUPPORT:
            return None
        boundaries.append(float(statistics.median(midpoints)))
        gap_supports.append(valid_gap_count / len(dense_rows))
    tracks = [0.0, *boundaries, width]
    if any(current <= previous for previous, current in zip(tracks, tracks[1:])):
        return None

    first_dense_row = min(row.row_index for row in dense_rows)
    boundary_margin = max(0.5, 0.08 * text.median_glyph_height)
    for row in text.rows:
        if row.row_index < first_dense_row:
            continue
        if any(
            token.bbox[0] + boundary_margin < boundary < token.bbox[2] - boundary_margin
            for token in row.tokens
            for boundary in tracks[1:-1]
        ):
            return None

    supported_rows_by_col: list[set[int]] = [set() for _ in range(target_cols)]
    aligned_tokens = 0
    total_tokens = 0
    for row in text.rows:
        previous_col = -1
        for token in row.tokens:
            center_x = (token.bbox[0] + token.bbox[2]) / 2.0
            col = next(
                (index for index, (left, right) in enumerate(zip(tracks, tracks[1:])) if left <= center_x <= right),
                None,
            )
            total_tokens += 1
            if col is None or col < previous_col:
                continue
            supported_rows_by_col[col].add(row.row_index)
            aligned_tokens += 1
            previous_col = col
    row_count = max(1, len(text.rows))
    anchor_support = sum(len(row_indices) / row_count for row_indices in supported_rows_by_col) / target_cols
    alignment_support = aligned_tokens / total_tokens if total_tokens else 0.0
    if anchor_support < MIN_COLUMN_ANCHOR_SUPPORT:
        return None
    gap_support = float(statistics.mean(gap_supports)) if gap_supports else 1.0
    return tracks, anchor_support, min(alignment_support, gap_support)


def _infer_row_tracks(
    rows: tuple[NativeTableTextRow, ...],
    height: float,
) -> list[float] | None:
    """用相邻逻辑行中心的中点构造完整行边界。"""

    if len(rows) < 2:
        return None
    centers = [(row.bbox[1] + row.bbox[3]) / 2.0 for row in rows]
    boundaries = [(previous + current) / 2.0 for previous, current in zip(centers, centers[1:])]
    tracks = [0.0, *boundaries, height]
    if any(current <= previous for previous, current in zip(tracks, tracks[1:])):
        return None
    return tracks


def _token_columns(
    row: NativeTableTextRow,
    x_tracks: list[float],
) -> list[int] | None:
    """把一行文本项按中心点单调映射到叶子列。"""

    output: list[int] = []
    for token in row.tokens:
        center_x = (token.bbox[0] + token.bbox[2]) / 2.0
        col = next(
            (index for index, (left, right) in enumerate(zip(x_tracks, x_tracks[1:])) if left <= center_x <= right),
            None,
        )
        if col is None or (output and col < output[-1]):
            return None
        output.append(col)
    return output


def _has_horizontal_rule_between(
    table_input: NativeTableInput,
    previous: NativeTableTextRow,
    current: NativeTableTextRow,
) -> bool:
    """判断两条视觉文本行之间是否存在贯穿多数表宽的物理横线。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return True
    angle = normalize_angle(table_input.angle)
    width, _height = table_local_size(table_bbox, angle)
    upper = previous.bbox[3]
    lower = current.bbox[1]
    for rule in table_input.drawing_lines:
        rule_bbox = normalize_bbox(rule.bbox)
        if rule_bbox is None:
            continue
        local_bbox = page_bbox_to_table_local(rule_bbox, table_bbox, angle)
        if local_bbox is None:
            continue
        local_width = local_bbox[2] - local_bbox[0]
        local_height = local_bbox[3] - local_bbox[1]
        center_y = (local_bbox[1] + local_bbox[3]) / 2.0
        if local_width >= 0.60 * width and local_width >= 4.0 * max(local_height, 0.1) and upper <= center_y <= lower:
            return True
    return False


def _group_logical_rows(
    table_input: NativeTableInput,
    text: NativeTableText,
    x_tracks: list[float],
) -> _LogicalRowGrouping:
    """保守合并同格内紧邻 continuation，避免 baseline 直接等同逻辑行。"""

    groups: list[list[NativeTableTextRow]] = []
    group_occupancy: list[set[int]] = []
    dense_ambiguities: list[tuple[int, int]] = []
    subset_merges: list[tuple[int, int]] = []
    maximum_gap = max(0.75, 0.40 * text.median_glyph_height)
    target_cols = len(x_tracks) - 1
    dense_column_count = max(2, math.ceil(MIN_COLUMN_ANCHOR_SUPPORT * target_cols))
    for row in text.rows:
        columns = _token_columns(row, x_tracks)
        occupancy = set(columns or [])
        if groups:
            previous = groups[-1][-1]
            gap = row.bbox[1] - previous.bbox[3]
            has_horizontal_rule = _has_horizontal_rule_between(
                table_input,
                previous,
                row,
            )
            if (
                occupancy == group_occupancy[-1]
                and len(occupancy) >= dense_column_count
                and gap <= maximum_gap
                and not has_horizontal_rule
            ):
                dense_ambiguities.append((previous.row_index, row.row_index))
            can_continue = (
                bool(occupancy) and occupancy < group_occupancy[-1] and gap <= maximum_gap and not has_horizontal_rule
            )
            if can_continue:
                groups[-1].append(row)
                subset_merges.append((previous.row_index, row.row_index))
                continue
        groups.append([row])
        group_occupancy.append(occupancy)

    logical_rows: list[NativeTableTextRow] = []
    for row_index, group in enumerate(groups):
        logical_rows.append(
            NativeTableTextRow(
                row_index=row_index,
                bbox=bbox_union(row.bbox for row in group),
                tokens=tuple(
                    sorted(
                        (token for row in group for token in row.tokens),
                        key=lambda token: (token.bbox[0], token.bbox[1]),
                    )
                ),
                glyph_ids=tuple(glyph_id for row in group for glyph_id in row.glyph_ids),
            )
        )
    return _LogicalRowGrouping(
        rows=tuple(logical_rows),
        dense_ambiguities=tuple(dense_ambiguities),
        subset_merges=tuple(subset_merges),
    )


def _header_rows_are_representable(
    rows: tuple[NativeTableTextRow, ...],
    x_tracks: list[float],
    first_dense_row: int,
) -> bool:
    """校验前导多层表头能否仅用当前 colspan 逻辑完整表达。"""

    if first_dense_row < 1:
        return True
    all_columns = set(range(len(x_tracks) - 1))
    for row_index in range(first_dense_row):
        row = rows[row_index]
        spans = _grouped_header_spans(row, rows[row_index + 1], x_tracks) if row_index + 1 < len(rows) else None
        if spans is not None:
            coverage = {col for start_col, end_col in spans for col in range(start_col, end_col + 1)}
        else:
            token_columns = _token_columns(row, x_tracks)
            if token_columns is None:
                return False
            coverage = set(token_columns)
        if coverage != all_columns:
            return False
    return True


def _single_header_span(
    row: NativeTableTextRow,
    x_tracks: list[float],
) -> tuple[int, int] | None:
    """仅在单文本项确实横跨连续叶子列时返回保守表头 colspan。"""

    if len(row.tokens) != 1:
        return None
    token = row.tokens[0]
    token_width = token.bbox[2] - token.bbox[0]
    if token_width <= 0:
        return None
    covered_cols = []
    for col, (left, right) in enumerate(zip(x_tracks, x_tracks[1:])):
        overlap = max(0.0, min(token.bbox[2], right) - max(token.bbox[0], left))
        if overlap / token_width >= 0.20:
            covered_cols.append(col)
    if len(covered_cols) < 2 or covered_cols != list(range(covered_cols[0], covered_cols[-1] + 1)):
        return None
    return covered_cols[0], covered_cols[-1]


def _grouped_header_spans(
    row: NativeTableTextRow,
    next_row: NativeTableTextRow,
    x_tracks: list[float],
) -> list[tuple[int, int]] | None:
    """用下一层表头叶子列为多个分组标题推断连续 colspan。"""

    if len(row.tokens) < 2:
        single_span = _single_header_span(row, x_tracks)
        return [single_span] if single_span is not None else None
    child_cols = _token_columns(next_row, x_tracks)
    if child_cols is None:
        return None
    unique_child_cols = sorted(set(child_cols))
    if len(unique_child_cols) < len(row.tokens):
        return None
    header_centers = [(token.bbox[0] + token.bbox[2]) / 2.0 for token in row.tokens]
    header_gaps = [current - previous for previous, current in zip(header_centers, header_centers[1:]) if current > previous]
    maximum_child_distance = 1.5 * float(statistics.median(header_gaps)) if header_gaps else float("inf")
    groups: list[list[int]] = [[] for _ in row.tokens]
    for col in unique_child_cols:
        col_center = (x_tracks[col] + x_tracks[col + 1]) / 2.0
        owner = min(
            range(len(header_centers)),
            key=lambda index: abs(col_center - header_centers[index]),
        )
        if abs(col_center - header_centers[owner]) > maximum_child_distance:
            continue
        groups[owner].append(col)
    spans: list[tuple[int, int]] = []
    for token, cols in zip(row.tokens, groups, strict=True):
        if not cols or cols != list(range(cols[0], cols[-1] + 1)):
            return None
        token_center = (token.bbox[0] + token.bbox[2]) / 2.0
        if not x_tracks[cols[0]] <= token_center <= x_tracks[cols[-1] + 1]:
            return None
        spans.append((cols[0], cols[-1]))
    return spans


def _build_text_grid_specs(
    rows: tuple[NativeTableTextRow, ...],
    x_tracks: list[float],
    y_tracks: list[float],
    first_dense_row: int,
) -> tuple[tuple[GridCellSpec, ...], float] | None:
    """为每个逻辑行构造单元格，并允许前导分组表头产生 colspan。"""

    cols = len(x_tracks) - 1
    specs: list[GridCellSpec] = []
    stable_rows = 0
    for row_index, row in enumerate(rows):
        occupied: set[int] = set()
        header_spans = (
            _grouped_header_spans(
                row,
                rows[row_index + 1],
                x_tracks,
            )
            if row_index < first_dense_row and row_index + 1 < len(rows)
            else None
        )
        if header_spans is not None:
            for start_col, end_col in header_spans:
                specs.append(
                    GridCellSpec(
                        row=row_index,
                        col=start_col,
                        rowspan=1,
                        colspan=end_col - start_col + 1,
                        bbox=(
                            x_tracks[start_col],
                            y_tracks[row_index],
                            x_tracks[end_col + 1],
                            y_tracks[row_index + 1],
                        ),
                    )
                )
                occupied.update(range(start_col, end_col + 1))
            stable_rows += 1
        else:
            token_cols = _token_columns(row, x_tracks)
            if token_cols is None:
                return None
            unique_cols = set(token_cols)
            occupied.update(unique_cols)
            if len(unique_cols) >= 2:
                stable_rows += 1
        for col in range(cols):
            if col in occupied:
                if header_spans is not None:
                    continue
            specs.append(
                GridCellSpec(
                    row=row_index,
                    col=col,
                    rowspan=1,
                    colspan=1,
                    bbox=(
                        x_tracks[col],
                        y_tracks[row_index],
                        x_tracks[col + 1],
                        y_tracks[row_index + 1],
                    ),
                )
            )
    row_stability = stable_rows / len(rows) if rows else 0.0
    return tuple(specs), row_stability


def _physical_sparse_evidence(
    table_input: NativeTableInput,
    text: NativeTableText,
) -> float:
    """统计长横线和重复行底纹，为少线候选提供独立物理证据。"""

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return 0.0
    angle = normalize_angle(table_input.angle)
    width, height = table_local_size(table_bbox, angle)
    long_horizontal_count = 0
    for rule in table_input.drawing_lines:
        bbox = normalize_bbox(rule.bbox)
        if bbox is None:
            continue
        local_bbox = page_bbox_to_table_local(bbox, table_bbox, angle)
        if local_bbox is None:
            continue
        local_width = local_bbox[2] - local_bbox[0]
        local_height = local_bbox[3] - local_bbox[1]
        if local_width >= 0.50 * width and local_width >= 4.0 * local_height:
            long_horizontal_count += 1

    stripe_count = 0
    for rectangle in table_input.rectangles:
        if rectangle.segment_count != 5 or not rectangle.fill_visible:
            continue
        bbox = normalize_bbox(rectangle.bbox)
        if bbox is None:
            continue
        local_bbox = page_bbox_to_table_local(bbox, table_bbox, angle)
        if local_bbox is None:
            continue
        local_width = local_bbox[2] - local_bbox[0]
        local_height = local_bbox[3] - local_bbox[1]
        if local_width >= 0.60 * width and 0.30 * text.median_glyph_height <= local_height <= min(
            3.0 * text.median_glyph_height, 0.50 * height
        ):
            stripe_count += 1
    # 贯穿竖线只说明 vector 候选应被优先验证，不能关闭其拓扑失败后的
    # 横线加文本兜底；异构候选最终由 verified 物理网格仲裁。
    return min(
        1.0,
        (long_horizontal_count + stripe_count) / 3.0,
    )


def _build_aligned_candidate(
    *,
    table_input: NativeTableInput,
    text: NativeTableText,
    source: NativeTableCandidateSource,
    require_three_rows: bool,
    require_three_cols: bool,
    physical_support: float,
    diagnostics: dict[str, Any] | None = None,
) -> NativeTableCandidate | None:
    """按统一文本轨道构造 sparse、wireless 或 key-value 候选。"""

    if diagnostics is not None:
        diagnostics["source"] = source
        diagnostics["raw_visual_rows"] = len(text.rows)

    def reject(gate: str) -> None:
        """记录文本候选的首个拒绝门。"""

        if diagnostics is not None:
            diagnostics["first_rejection_gate"] = gate
        return None

    target_cols = _infer_target_column_count(text)
    if target_cols is None:
        return reject("column_count")
    if require_three_cols and target_cols < 3:
        return reject("column_count")
    if source == "key_value" and target_cols != 2:
        return reject("column_count")
    if require_three_rows and len(text.rows) < 3:
        return reject("row_count")

    table_bbox = normalize_bbox(table_input.table_bbox)
    if table_bbox is None:
        return reject("table_geometry")
    width, height = table_local_size(table_bbox, normalize_angle(table_input.angle))
    column_result = _infer_column_tracks(text, width, target_cols)
    if column_result is None:
        return reject("column_tracks")
    x_tracks, anchor_support, alignment_support = column_result
    grouping = _group_logical_rows(
        table_input,
        text,
        x_tracks,
    )
    logical_rows = grouping.rows
    if diagnostics is not None:
        diagnostics["logical_rows"] = len(logical_rows)
        diagnostics["dense_row_ambiguities"] = [list(pair) for pair in grouping.dense_ambiguities]
        diagnostics["subset_continuation_merges"] = [list(pair) for pair in grouping.subset_merges]
    if grouping.dense_ambiguities:
        return reject("dense_row_ambiguity")
    y_tracks = _infer_row_tracks(logical_rows, height)
    if y_tracks is None:
        return reject("row_tracks")
    dense_row_indices = [row.row_index for row in logical_rows if len(set(_token_columns(row, x_tracks) or [])) == target_cols]
    first_dense_row = min(dense_row_indices) if dense_row_indices else 0
    header_representable = _header_rows_are_representable(
        logical_rows,
        x_tracks,
        first_dense_row,
    )
    if diagnostics is not None:
        diagnostics["first_dense_row"] = first_dense_row
        diagnostics["header_representable"] = header_representable
    if not header_representable:
        return reject("header_requires_rowspan")
    specs_result = _build_text_grid_specs(
        logical_rows,
        x_tracks,
        y_tracks,
        first_dense_row,
    )
    if specs_result is None:
        return reject("grid_specs")
    specs, row_stability = specs_result
    if row_stability < MIN_COLUMN_ANCHOR_SUPPORT:
        return reject("row_stability")
    structure_support = max(anchor_support, physical_support) if source == "sparse_grid" else alignment_support
    candidate = build_candidate(
        source=source,
        rows=len(logical_rows),
        cols=target_cols,
        specs=specs,
        text=text,
        structure_support=structure_support,
        row_stability=row_stability,
        column_stability=alignment_support,
        require_atomic_tokens=True,
        diagnostics=diagnostics,
    )
    if candidate is None:
        if diagnostics is not None and diagnostics.get("first_rejection_gate") is None:
            diagnostics["first_rejection_gate"] = diagnostics.get("candidate_rejection_gate", "candidate_hard_gate")
        return None
    if diagnostics is not None:
        diagnostics["first_rejection_gate"] = None
        diagnostics["score"] = candidate.score
    return candidate


def build_text_candidates(
    table_input: NativeTableInput,
    text: NativeTableText,
    diagnostics: list[dict[str, Any]] | None = None,
) -> list[NativeTableCandidate]:
    """同时生成少线、三列以上无线表和两列 key-value 候选。"""

    candidates: list[NativeTableCandidate] = []
    physical_support = _physical_sparse_evidence(table_input, text)
    if physical_support > 0:
        sparse_diagnostics: dict[str, Any] | None = {} if diagnostics is not None else None
        sparse = _build_aligned_candidate(
            table_input=table_input,
            text=text,
            source="sparse_grid",
            require_three_rows=False,
            require_three_cols=False,
            physical_support=physical_support,
            diagnostics=sparse_diagnostics,
        )
        if diagnostics is not None and sparse_diagnostics is not None:
            diagnostics.append(sparse_diagnostics)
        if sparse is not None:
            candidates.append(sparse)

    text_diagnostics: dict[str, Any] | None = {} if diagnostics is not None else None
    text_grid = _build_aligned_candidate(
        table_input=table_input,
        text=text,
        source="text_grid",
        require_three_rows=True,
        require_three_cols=True,
        physical_support=0.0,
        diagnostics=text_diagnostics,
    )
    if diagnostics is not None and text_diagnostics is not None:
        diagnostics.append(text_diagnostics)
    if text_grid is not None:
        candidates.append(text_grid)

    key_value_diagnostics: dict[str, Any] | None = {} if diagnostics is not None else None
    key_value = _build_aligned_candidate(
        table_input=table_input,
        text=text,
        source="key_value",
        require_three_rows=True,
        require_three_cols=False,
        physical_support=0.0,
        diagnostics=key_value_diagnostics,
    )
    if diagnostics is not None and key_value_diagnostics is not None:
        diagnostics.append(key_value_diagnostics)
    if key_value is not None:
        candidates.append(key_value)
    return candidates


def diagnose_text_candidate_builds(
    table_input: NativeTableInput,
    text: NativeTableText,
) -> tuple[dict[str, Any], ...]:
    """重放文本候选构造并返回不含单元格全文的诊断。"""

    diagnostics: list[dict[str, Any]] = []
    build_text_candidates(
        table_input,
        text,
        diagnostics=diagnostics,
    )
    return tuple(diagnostics)


__all__ = ["build_text_candidates"]
