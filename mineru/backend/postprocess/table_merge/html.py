# Copyright (c) Opendatalab. All rights reserved.
"""HTML 表格解析、行列扫描和结构状态缓存。"""

from __future__ import annotations

from typing import Any

from bs4 import BeautifulSoup, Tag

from mineru.utils.text_utils import full_to_half

from .models import (
    MAX_HEADER_ROWS,
    RenderedCellSegment,
    RowMetrics,
    RowScanResult,
    RowSignature,
    TableMergeState,
)


def _colspan(cell: Any) -> int:
    """读取 HTML 单元格 colspan，非法值交由上层安全降级。"""
    val = cell.get("colspan", "1")
    assert isinstance(val, str)
    return int(val)


def _rowspan(cell: Any) -> int:
    """读取 HTML 单元格 rowspan，非法值交由上层安全降级。"""
    val = cell.get("rowspan", "1")
    assert isinstance(val, str)
    return int(val)


def _normalize_cell_text(cell: Tag) -> str:
    """生成表头匹配使用的半角无空白文本。"""
    return "".join(full_to_half(cell.get_text()).split())


def _display_cell_text(cell: Tag) -> str:
    """生成保留内部空白的半角展示文本。"""
    return full_to_half(cell.get_text().strip())


def _scan_rows(rows: list[Tag], initial_occupied: dict[int, set[int]] | None = None, start_row_idx: int = 0) -> RowScanResult:
    """单次扫描 HTML 行并缓存有效列、显式列和跨行占位指标。

    ``initial_occupied`` 使用相对首行的偏移记录未来行占位，从而保留跨越
    前后表边界的 rowspan 结构。
    """
    occupied: dict[int, dict[int, bool]] = {}
    max_cols = 0

    for row_offset, cols in (initial_occupied or {}).items():
        if not cols:
            continue
        occupied[row_offset] = dict.fromkeys(cols, True)
        max_cols = max(max_cols, max(cols) + 1)

    row_effective_cols: list[int] = []
    row_metrics: list[RowMetrics] = []
    last_nonempty_row_metrics: RowMetrics | None = None

    for local_idx, row in enumerate(rows):
        occupied_row = occupied.setdefault(local_idx, {})
        col_idx = 0
        cells = row.find_all(["td", "th"])
        actual_cols = 0

        for cell in cells:
            while col_idx in occupied_row:
                col_idx += 1

            colspan = _colspan(cell)
            rowspan = _rowspan(cell)
            actual_cols += colspan

            for row_offset in range(rowspan):
                target_idx = local_idx + row_offset
                occupied_target = occupied.setdefault(target_idx, {})
                for col in range(col_idx, col_idx + colspan):
                    occupied_target[col] = True

            col_idx += colspan
            max_cols = max(max_cols, col_idx)

        effective_cols = max(occupied_row.keys()) + 1 if occupied_row else 0
        row_effective_cols.append(effective_cols)
        max_cols = max(max_cols, effective_cols)

        metrics = RowMetrics(
            row_idx=start_row_idx + local_idx,
            effective_cols=effective_cols,
            actual_cols=actual_cols,
            visual_cols=len(cells),
        )
        row_metrics.append(metrics)
        if cells:
            last_nonempty_row_metrics = metrics

    tail_occupied = {
        row_idx - len(rows): set(cols.keys()) for row_idx, cols in occupied.items() if row_idx >= len(rows) and cols
    }

    return RowScanResult(
        row_effective_cols=row_effective_cols,
        row_metrics=row_metrics,
        total_cols=max_cols,
        last_nonempty_row_metrics=last_nonempty_row_metrics,
        tail_occupied=tail_occupied,
    )


def _build_row_signature(row: Tag, effective_cols: int) -> RowSignature:
    """构建表头检测使用的行结构与文本签名。"""
    cells = row.find_all(["td", "th"])
    return RowSignature(
        effective_cols=effective_cols,
        colspans=tuple(_colspan(cell) for cell in cells),
        rowspans=tuple(_rowspan(cell) for cell in cells),
        normalized_texts=tuple(_normalize_cell_text(cell) for cell in cells),
        display_texts=tuple(_display_cell_text(cell) for cell in cells),
    )


def _build_front_cache(
    rows: list[Tag], max_header_rows: int = MAX_HEADER_ROWS
) -> tuple[list[RowSignature], dict[int, RowMetrics]]:
    """缓存表格前部表头签名和首批数据行指标。"""
    front_limit = min(len(rows), max_header_rows + 1)
    front_rows = rows[:front_limit]
    front_scan = _scan_rows(front_rows)

    front_header_info = [
        _build_row_signature(front_rows[idx], front_scan.row_effective_cols[idx])
        for idx in range(min(len(front_rows), max_header_rows))
    ]
    front_first_data_row_metrics = dict(enumerate(front_scan.row_metrics))
    return front_header_info, front_first_data_row_metrics


def _refresh_table_state_metrics(state: TableMergeState) -> None:
    """HTML 结构调整后重新计算表格状态指标。"""
    scan = _scan_rows(state.rows)
    state.row_effective_cols = scan.row_effective_cols
    state.total_cols = scan.total_cols
    state.last_data_row_metrics = scan.last_nonempty_row_metrics
    state.tail_occupied = scan.tail_occupied
    state.front_header_info, state.front_first_data_row_metrics = _build_front_cache(state.rows)


def build_table_state_from_html(
    html: str,
    max_header_rows: int = MAX_HEADER_ROWS,
) -> TableMergeState | None:
    """从原始 HTML 构建 TableMergeState，不依赖 MinerU block 结构。

    供外部工具（如 mineru-vl-utils）调用，用于跨页表格结构检测。
    返回的 state 供 HTML-only 结构 helper 使用，不包含 MinerU block 所有者。
    """
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody") or soup.find("table")
    rows = soup.find_all("tr")
    if tbody is None or not rows:
        return None

    try:
        scan = _scan_rows(rows)
        front_header_info, front_first_data_row_metrics = _build_front_cache(
            rows,
            max_header_rows=max_header_rows,
        )
    except (AssertionError, TypeError, ValueError):
        return None
    if scan.total_cols <= 0 or scan.last_nonempty_row_metrics is None:
        return None

    return TableMergeState(
        owner_block=None,
        body_block=None,
        soup=soup,
        tbody=tbody,
        rows=rows,
        total_cols=scan.total_cols,
        front_header_info=front_header_info,
        front_first_data_row_metrics=front_first_data_row_metrics,
        last_data_row_metrics=scan.last_nonempty_row_metrics,
        row_effective_cols=scan.row_effective_cols,
        tail_occupied=scan.tail_occupied,
    )


def _serialize_table_state_html(state: TableMergeState) -> bool:
    """将合并后的 BeautifulSoup 写回克隆表体，缺失表体时返回失败。"""
    if state.body_block is None:
        return False
    state.body_block["content"] = str(state.soup)
    state.dirty = False
    return True


def calculate_table_total_columns(soup: BeautifulSoup) -> int:
    """计算表格的总列数，通过分析整个表格结构来处理rowspan和colspan."""
    rows = soup.find_all("tr")
    return _scan_rows(rows).total_cols if rows else 0


def build_table_occupied_matrix(soup: BeautifulSoup) -> dict[int, int]:
    """构建表格的占用矩阵，返回每行的有效列数."""
    rows = soup.find_all("tr")
    if not rows:
        return {}

    scan = _scan_rows(rows)
    return dict(enumerate(scan.row_effective_cols))


def calculate_row_effective_columns(soup: BeautifulSoup, row_idx: int) -> int:
    """计算指定行的有效列数（考虑rowspan占用）."""
    row_effective_cols = build_table_occupied_matrix(soup)
    return row_effective_cols.get(row_idx, 0)


def calculate_row_columns(row: Tag) -> int:
    """计算表格行的实际列数，考虑colspan属性."""
    cells = row.find_all(["td", "th"])
    column_count = 0

    for cell in cells:
        colspan = _colspan(cell)
        column_count += colspan

    return column_count


def calculate_visual_columns(row: Tag) -> int:
    """计算表格行的视觉列数（实际td/th单元格数量，不考虑colspan）."""
    cells = row.find_all(["td", "th"])
    return len(cells)


def _scan_row_visual_sources(
    rows: list[Tag],
    target_row_index: int,
    initial_occupied: dict[int, set[int]] | None = None,
) -> tuple[dict[int, tuple[int, int]], int]:
    """扫描到目标行，记录每个视觉列当前由哪个源单元格占据。

    initial_occupied 表示从上一页延续过来的 rowspan 占位，行号相对
    rows[0] 计算。它只作为虚拟源单元格参与列定位，不对应当前页真实
    <td>/<th> 元素。
    """
    if target_row_index < 0:
        target_row_index += len(rows)
    if target_row_index < 0 or target_row_index >= len(rows):
        return {}, 0

    # occupied[row_idx][col_idx] = (source_row_idx, source_cell_idx)
    occupied: dict[int, dict[int, tuple[int, int]]] = {}
    total_cols = 0
    for row_offset, cols in (initial_occupied or {}).items():
        if not cols:
            continue
        occupied[row_offset] = {col: (-1, col) for col in cols}
        total_cols = max(total_cols, max(cols) + 1)

    for r_idx in range(target_row_index + 1):
        occupied_row = occupied.setdefault(r_idx, {})
        col_idx = 0
        cells = rows[r_idx].find_all(["td", "th"])
        for cell_idx, cell in enumerate(cells):
            while col_idx in occupied_row:
                col_idx += 1
            colspan = _colspan(cell)
            rowspan = _rowspan(cell)
            source_marker = (r_idx, cell_idx)
            for ro in range(rowspan):
                target_idx = r_idx + ro
                occ = occupied.setdefault(target_idx, {})
                for c in range(col_idx, col_idx + colspan):
                    occ[c] = source_marker
            col_idx += colspan
            total_cols = max(total_cols, col_idx)

    return occupied.get(target_row_index, {}), total_cols


def build_visual_col_mapping(
    rows: list[Tag],
    target_row_index: int,
    initial_occupied: dict[int, set[int]] | None = None,
) -> list[int]:
    """构建目标行中每个显式 <td>/<th> 元素到视觉列位置的映射。

    该映射会正确考虑从前序行继承而来的 rowspan 占位。
    initial_occupied 可额外传入上一页延续到当前切片的 rowspan 占位。
    """
    if target_row_index < 0:
        target_row_index += len(rows)
    if target_row_index < 0 or target_row_index >= len(rows):
        return []

    target_occupied, _ = _scan_row_visual_sources(
        rows,
        target_row_index,
        initial_occupied=initial_occupied,
    )

    col_idx = 0
    mapping = []
    target_cells = rows[target_row_index].find_all(["td", "th"])
    for cell in target_cells:
        while col_idx in target_occupied and target_occupied[col_idx][0] < target_row_index:
            col_idx += 1
        mapping.append(col_idx)
        colspan = _colspan(cell)
        col_idx += colspan
    return mapping


def build_row_rendered_cell_segments(
    rows: list[Tag],
    target_row_index: int,
    initial_occupied: dict[int, set[int]] | None = None,
) -> list[RenderedCellSegment]:
    """构建目标行的渲染单元格段，保留每段覆盖的视觉列范围。

    该函数复用表格行视觉来源扫描结果，语义与 calculate_row_rendered_segments()
    保持一致：colspan 只算一个渲染段，rowspan 延续下来的单元格也会作为
    目标行的渲染段返回。
    """
    if target_row_index < 0:
        target_row_index += len(rows)
    if target_row_index < 0 or target_row_index >= len(rows):
        return []

    target_occupied, total_cols = _scan_row_visual_sources(
        rows,
        target_row_index,
        initial_occupied=initial_occupied,
    )
    if total_cols == 0:
        return []

    segments: list[RenderedCellSegment] = []
    current_marker: tuple[int, int] | None = None
    current_start_col: int | None = None
    current_text = ""

    # 连续视觉列来自同一个源单元格时，合并为一个渲染段。
    for col_idx in range(total_cols):
        marker = target_occupied.get(col_idx)
        if marker is None:
            if current_marker is not None and current_start_col is not None:
                segments.append(RenderedCellSegment(text=current_text, start_col=current_start_col, end_col=col_idx))
            current_marker = None
            current_start_col = None
            current_text = ""
            continue

        if marker != current_marker:
            if current_marker is not None and current_start_col is not None:
                segments.append(RenderedCellSegment(text=current_text, start_col=current_start_col, end_col=col_idx))
            current_marker = marker
            current_start_col = col_idx
            source_row_idx, source_cell_idx = marker
            current_text = ""
            if source_row_idx >= 0:
                source_cells = rows[source_row_idx].find_all(["td", "th"])
                if source_cell_idx < len(source_cells):
                    current_text = _display_cell_text(source_cells[source_cell_idx])

    if current_marker is not None and current_start_col is not None:
        segments.append(RenderedCellSegment(text=current_text, start_col=current_start_col, end_col=total_cols))

    return segments


def calculate_row_rendered_segments(rows: list[Tag], target_row_index: int) -> int:
    """计算目标行渲染后的视觉段数。

    段数按“渲染出来的单元格块”统计：
    - 当前行显式单元格各算一段，不展开 colspan
    - 从前序行继承而来的 rowspan 占位也算段
    - 只有连续列且来自同一个源单元格时才算同一段
    """
    target_occupied, total_cols = _scan_row_visual_sources(rows, target_row_index)
    if total_cols == 0:
        return 0

    segment_count = 0
    previous_marker: tuple[int, int] | None = None

    for col_idx in range(total_cols):
        marker = target_occupied.get(col_idx)
        if marker is None:
            previous_marker = None
            continue
        if marker != previous_marker:
            segment_count += 1
            previous_marker = marker

    return segment_count
