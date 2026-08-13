# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, TypeAlias

from bs4 import BeautifulSoup, Tag

from ...types import BlockType
from .char_utils import full_to_half
from .table_continuation import is_table_continuation_text

MAX_HEADER_ROWS = 5

# 页边界扫描时可忽略的版面噪声块；其余非表格块都会阻断跨页关系。
TABLE_BOUNDARY_IGNORED_TYPES = {
    BlockType.HEADER,
    BlockType.FOOTER,
    BlockType.PAGE_NUMBER,
    BlockType.PAGE_FOOTNOTE,
    BlockType.ASIDE_TEXT,
}

BlockDict: TypeAlias = dict[str, Any]
PageInfoDict: TypeAlias = dict[str, Any]
CalculationBBox: TypeAlias = tuple[int, int, int, int]


@dataclass
class RowMetrics:
    row_idx: int
    effective_cols: int
    actual_cols: int
    visual_cols: int


@dataclass
class RowSignature:
    effective_cols: int
    colspans: tuple[int, ...]
    rowspans: tuple[int, ...]
    normalized_texts: tuple[str, ...]
    display_texts: tuple[str, ...]

    @property
    def cell_count(self) -> int:
        return len(self.colspans)


@dataclass
class RenderedCellSegment:
    text: str
    start_col: int
    end_col: int


@dataclass
class RowScanResult:
    row_effective_cols: list[int]
    row_metrics: list[RowMetrics]
    total_cols: int
    last_nonempty_row_metrics: RowMetrics | None
    tail_occupied: dict[int, set[int]]


@dataclass
class TableMergeState:
    owner_block: BlockDict | None
    body_block: BlockDict | None
    soup: Any
    tbody: Any
    rows: list[Any]
    total_cols: int
    front_header_info: list[RowSignature]
    front_first_data_row_metrics: dict[int, RowMetrics]
    last_data_row_metrics: RowMetrics | None
    row_effective_cols: list[int]
    tail_occupied: dict[int, set[int]]
    dirty: bool = False


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
    """Scan rows once and cache effective-column metrics.

    initial_occupied stores future-row occupancy relative to the first scanned row
    and preserves rowspans that cross a merge boundary.
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


def _bbox_for_calculation(bbox: Any) -> CalculationBBox | None:
    """复制归一化 bbox 并放大为千分位整数，原始字段保持不变。"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in bbox):
        return None

    values = tuple(float(value) for value in bbox)
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        return None

    x0, y0, x1, y1 = (int(round(value * 1000)) for value in values)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _table_children(table_block: BlockDict) -> list[BlockDict]:
    """读取 table 根块下的合法 dict 子块。"""
    content = table_block.get("content")
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def _find_table_body_block(table_block: BlockDict) -> BlockDict | None:
    """查找 dict table block 中的主体子块。"""
    for block in _table_children(table_block):
        if block.get("type") == BlockType.TABLE_BODY:
            return block
    return None


def _build_post_body_child_index(table_block: BlockDict, offset: int) -> int | None:
    """为复制到前表的 footnote 生成表体后的安全 index。"""
    body_block = _find_table_body_block(table_block)
    if body_block is None:
        return None
    body_index = body_block.get("index")
    if not isinstance(body_index, int):
        return None

    child_indices = [block.get("index") for block in _table_children(table_block) if isinstance(block.get("index"), int)]
    return max([body_index, *child_indices]) + offset


def _block_text(block: BlockDict) -> str:
    """递归读取 dict block 的文本内容，供续表标记判断使用。"""
    content = block.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "".join(_block_text(child) for child in content if isinstance(child, dict))


def _is_continuation_caption(caption_block: BlockDict) -> bool:
    """判断 dict caption 文本是否带有续表标记。"""
    return is_table_continuation_text(_block_text(caption_block))


def _is_post_table_non_continuation_caption(table_block: BlockDict, caption_block: BlockDict) -> bool:
    """判断 caption 是否是误挂到表格下方的新段落标题。

    这类 caption 位于 table body 下方，且不含续表标记；它不应作为
    当前表的新标题阻断跨页关系判断。
    """
    if _is_continuation_caption(caption_block):
        return False

    body_block = _find_table_body_block(table_block)
    if body_block is None:
        return False

    body_bbox = _bbox_for_calculation(body_block.get("bbox"))
    caption_bbox = _bbox_for_calculation(caption_block.get("bbox"))
    if body_bbox is None or caption_bbox is None:
        return False

    return caption_bbox[1] >= body_bbox[3]


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


def _build_table_state(table_block: BlockDict, max_header_rows: int = MAX_HEADER_ROWS) -> TableMergeState | None:
    """从 dict table block 构建结构缓存，非法主体安全返回空。"""
    body_block = _find_table_body_block(table_block)
    if body_block is None:
        return None

    html = body_block.get("content")
    if not isinstance(html, str) or not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody") or soup.find("table")
    rows = soup.find_all("tr")
    if tbody is None or not rows:
        return None

    scan = _scan_rows(rows)
    if scan.total_cols <= 0 or scan.last_nonempty_row_metrics is None:
        return None
    front_header_info, front_first_data_row_metrics = _build_front_cache(rows, max_header_rows=max_header_rows)

    return TableMergeState(
        owner_block=table_block,
        body_block=body_block,
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


def _get_or_create_table_state(
    table_block: BlockDict,
    state_cache: dict[int, TableMergeState],
    max_header_rows: int = MAX_HEADER_ROWS,
) -> TableMergeState | None:
    """按 table dict 对象身份复用 HTML 结构扫描结果。"""
    cache_key = id(table_block)
    state = state_cache.get(cache_key)
    if state is not None:
        return state

    try:
        state = _build_table_state(table_block, max_header_rows=max_header_rows)
    except (AssertionError, TypeError, ValueError):
        return None
    if state is not None:
        state_cache[cache_key] = state
    return state


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


def detect_table_headers(
    state1: TableMergeState, state2: TableMergeState, max_header_rows: int = MAX_HEADER_ROWS
) -> tuple[int, bool, list[list[str]]]:
    """检测并比较两个表格的表头，仅扫描前几行."""
    front_rows1 = state1.front_header_info[:max_header_rows]
    front_rows2 = state2.front_header_info[:max_header_rows]

    min_rows = min(len(front_rows1), len(front_rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for row_idx in range(min_rows):
        row1 = front_rows1[row_idx]
        row2 = front_rows2[row_idx]
        structure_match = (
            row1.cell_count == row2.cell_count
            and row1.effective_cols == row2.effective_cols
            and row1.colspans == row2.colspans
            and row1.rowspans == row2.rowspans
            and row1.normalized_texts == row2.normalized_texts
        )

        if structure_match:
            header_rows += 1
            header_texts.append(list(row1.display_texts))
        else:
            headers_match = header_rows > 0
            break

    if header_rows == 0:
        header_rows, headers_match, header_texts = _detect_table_headers_visual(state1, state2, max_header_rows=max_header_rows)

    return header_rows, headers_match, header_texts


def _detect_table_headers_visual(
    state1: TableMergeState,
    state2: TableMergeState,
    max_header_rows: int = MAX_HEADER_ROWS,
) -> tuple[int, bool, list[list[str]]]:
    """基于视觉一致性检测表头（只比较文本内容，忽略colspan/rowspan差异）."""
    front_rows1 = state1.front_header_info[:max_header_rows]
    front_rows2 = state2.front_header_info[:max_header_rows]

    min_rows = min(len(front_rows1), len(front_rows2), max_header_rows)
    header_rows = 0
    headers_match = True
    header_texts = []

    for row_idx in range(min_rows):
        row1 = front_rows1[row_idx]
        row2 = front_rows2[row_idx]
        # OCR 识别表头时可能丢失 colspan/rowspan，这里用渲染段数约束视觉一致性。
        rendered_segments1 = calculate_row_rendered_segments(state1.rows, row_idx)
        rendered_segments2 = calculate_row_rendered_segments(state2.rows, row_idx)
        if row1.normalized_texts == row2.normalized_texts and rendered_segments1 == rendered_segments2:
            header_rows += 1
            header_texts.append(list(row1.display_texts))
        else:
            headers_match = header_rows > 0
            break

    if header_rows == 0:
        headers_match = False

    return header_rows, headers_match, header_texts


def _expand_header_count_by_rowspan(rows: list[Tag], header_count: int) -> int:
    """按表头 rowspan 覆盖范围扩展跳过行数。

    跨页续表的第一行表头可能包含 rowspan。如果只跳过已匹配的首行，
    被该 rowspan 覆盖的后续表头行会失去占位来源，合并后形成半截表头。
    因此跳过重复表头时，需要覆盖所有由已跳过表头行跨行占据的行。
    """
    if header_count <= 0 or not rows:
        return header_count

    expanded_header_count = min(header_count, len(rows))
    row_idx = 0
    while row_idx < expanded_header_count:
        row = rows[row_idx]
        for cell in row.find_all(["td", "th"]):
            rowspan = _rowspan(cell)
            if rowspan > 1:
                expanded_header_count = max(expanded_header_count, row_idx + rowspan)
                expanded_header_count = min(expanded_header_count, len(rows))
        row_idx += 1

    return expanded_header_count


def can_merge_by_structure(
    current_state: TableMergeState,
    previous_state: TableMergeState,
    current_bbox: Any = None,
    previous_bbox: Any = None,
) -> bool:
    """仅基于表格结构判断是否可合并（不检查 caption/footnote）。

    供外部工具调用，忽略 caption 和 footnote 检查。
    """
    if (
        current_bbox is not None
        and previous_bbox is not None
        and not _table_widths_are_compatible(
            current_bbox,
            previous_bbox,
        )
    ):
        return False

    if (
        previous_state.total_cols <= 0
        or current_state.total_cols <= 0
        or previous_state.last_data_row_metrics is None
        or current_state.last_data_row_metrics is None
    ):
        return False

    if previous_state.total_cols == current_state.total_cols:
        return True

    return check_rows_match(previous_state, current_state)


def _table_widths_are_compatible(current_bbox: Any, previous_bbox: Any) -> bool:
    """使用千分位 bbox 判断两张表的宽度相对差是否小于百分之十。"""
    current_calc_bbox = _bbox_for_calculation(current_bbox)
    previous_calc_bbox = _bbox_for_calculation(previous_bbox)
    if current_calc_bbox is None or previous_calc_bbox is None:
        return False

    current_width = current_calc_bbox[2] - current_calc_bbox[0]
    previous_width = previous_calc_bbox[2] - previous_calc_bbox[0]
    min_width = min(current_width, previous_width)
    return min_width > 0 and abs(current_width - previous_width) / min_width < 0.1


def can_merge_tables(current_state: TableMergeState, previous_state: TableMergeState) -> bool:
    """根据 dict 表格的辅助文本、宽度和 HTML 结构判断是否可合并。"""
    current_table_block = current_state.owner_block
    previous_table_block = previous_state.owner_block

    if not isinstance(previous_table_block, dict) or not isinstance(current_table_block, dict):
        return False

    previous_children = _table_children(previous_table_block)
    current_children = _table_children(current_table_block)
    footnote_count = sum(1 for block in previous_children if block.get("type") == BlockType.TABLE_FOOTNOTE)
    caption_blocks = [block for block in current_children if block.get("type") == BlockType.TABLE_CAPTION]
    merge_caption_blocks = [
        block for block in caption_blocks if not _is_post_table_non_continuation_caption(current_table_block, block)
    ]
    if merge_caption_blocks:
        has_continuation_marker = any(_is_continuation_caption(block) for block in merge_caption_blocks)

        if not has_continuation_marker:
            return False

        if footnote_count > 1:
            return False
    elif footnote_count > 0:
        return False

    if not _table_widths_are_compatible(current_table_block.get("bbox"), previous_table_block.get("bbox")):
        return False

    return can_merge_by_structure(current_state, previous_state)


def check_rows_match(previous_state: TableMergeState, current_state: TableMergeState) -> bool:
    """检查表格边界行是否匹配."""
    last_row_metrics = previous_state.last_data_row_metrics
    if last_row_metrics is None:
        return False

    header_count, _, _ = detect_table_headers(previous_state, current_state)
    header_count = _expand_header_count_by_rowspan(current_state.rows, header_count)
    first_data_row_metrics = current_state.front_first_data_row_metrics.get(header_count)
    if first_data_row_metrics is None:
        return False

    previous_rendered_segments = calculate_row_rendered_segments(previous_state.rows, last_row_metrics.row_idx)
    current_rendered_segments = calculate_row_rendered_segments(current_state.rows, first_data_row_metrics.row_idx)

    return (
        last_row_metrics.effective_cols == first_data_row_metrics.effective_cols
        or last_row_metrics.actual_cols == first_data_row_metrics.actual_cols
        or previous_rendered_segments == current_rendered_segments
    )


def check_row_columns_match(row1: Tag, row2: Tag) -> bool:
    """判断两行显式单元格数量与 colspan 结构是否一致。"""
    cells1 = row1.find_all(["td", "th"])
    cells2 = row2.find_all(["td", "th"])
    if len(cells1) != len(cells2):
        return False
    for cell1, cell2 in zip(cells1, cells2):
        colspan1 = _colspan(cell1)
        colspan2 = _colspan(cell2)
        if colspan1 != colspan2:
            return False
    return True


def adjust_table_rows_colspan(
    rows: list[Tag],
    start_idx: int,
    end_idx: int,
    row_effective_cols: list[int],
    reference_structure: list[int],
    reference_visual_cols: int,
    target_cols: int,
    match_reference_row: Tag,
) -> None:
    """调整表格行的colspan属性以匹配目标列数."""
    reference_row_copy = deepcopy(match_reference_row)

    for row_idx in range(start_idx, end_idx):
        row = rows[row_idx]
        cells = row.find_all(["td", "th"])
        if not cells:
            continue

        current_row_effective_cols = row_effective_cols[row_idx]
        current_row_cols = calculate_row_columns(row)

        if current_row_effective_cols >= target_cols or current_row_cols >= target_cols:
            continue

        if calculate_visual_columns(row) == reference_visual_cols and check_row_columns_match(row, reference_row_copy):
            if len(cells) <= len(reference_structure):
                for cell_idx, cell in enumerate(cells):
                    if cell_idx < len(reference_structure) and reference_structure[cell_idx] > 1:
                        cell["colspan"] = str(reference_structure[cell_idx])
        else:
            cols_diff = target_cols - current_row_effective_cols
            if cols_diff > 0:
                last_cell = cells[-1]
                current_last_span = _colspan(last_cell)
                last_cell["colspan"] = str(current_last_span + cols_diff)


def _cell_has_semantic_content(cell: Tag) -> bool:
    """判断单元格是否仍包含用户可见的语义内容。"""
    if cell.get_text(strip=True):
        return True

    return cell.find(["img", "svg", "math", "eq", "table", "figure", "object", "embed", "canvas"]) is not None


def _row_has_semantic_content(row: Tag) -> bool:
    """判断整行是否仍保留未并回的语义内容。"""
    return any(_cell_has_semantic_content(cell) for cell in row.find_all(["td", "th"]))


def _insert_cell_before_visual_column(rows: list[Tag], target_row_index: int, start_vcol: int, cell: Tag) -> None:
    """将单元格插入到目标行中对应视觉列之前。"""
    target_row = rows[target_row_index]
    target_cells = target_row.find_all(["td", "th"])
    target_vcol_map = build_visual_col_mapping(rows, target_row_index)

    for idx, target_start_vcol in enumerate(target_vcol_map):
        if target_start_vcol >= start_vcol:
            target_cells[idx].insert_before(cell)
            return

    target_row.append(cell)


def _carry_rowspan_structure_to_next_row(rows: list[Tag], row_idx: int) -> None:
    """下沉空白结构占位单元格，避免删除当前行后破坏后续列对齐。"""
    next_row_idx = row_idx + 1
    if next_row_idx >= len(rows):
        return

    current_row = rows[row_idx]
    current_cells = current_row.find_all(["td", "th"])
    current_vcol_map = build_visual_col_mapping(rows, row_idx)
    carried_cells = []

    for cell, start_vcol in zip(current_cells, current_vcol_map):
        rowspan = _rowspan(cell)
        if rowspan <= 1 or _cell_has_semantic_content(cell):
            continue

        carried_cell = deepcopy(cell)
        new_rowspan = rowspan - 1
        if new_rowspan > 1:
            carried_cell["rowspan"] = str(new_rowspan)
        else:
            carried_cell.attrs.pop("rowspan", None)
        carried_cells.append((start_vcol, carried_cell))

    for start_vcol, carried_cell in sorted(carried_cells, key=lambda item: item[0], reverse=True):
        _insert_cell_before_visual_column(rows, next_row_idx, start_vcol, carried_cell)


def _clip_overlapped_blank_rowspan_cells(
    rows: list[Tag],
    initial_occupied: dict[int, set[int]],
) -> bool:
    """裁剪被上页 rowspan 覆盖的当前页空白结构占位。

    跨页表格中，上一页未结束的 rowspan 会通过 initial_occupied 占住
    当前页开头的视觉列。如果当前页表格识别又生成了同位置的空白
    rowspan 单元格，这个单元格只是结构占位；直接拼接会把同一视觉列
    当成两列。这里仅裁剪无语义内容的空白占位，真实内容单元格不处理。
    """
    if not rows or not initial_occupied:
        return False

    cells_to_remove = []
    cells_to_move = []

    for row_idx, row in enumerate(rows):
        cells = row.find_all(["td", "th"])
        visual_col_map = build_visual_col_mapping(rows, row_idx)
        for cell, start_vcol in zip(cells, visual_col_map):
            rowspan = _rowspan(cell)
            if rowspan <= 1 or _cell_has_semantic_content(cell):
                continue

            colspan = _colspan(cell)
            occupied_cols = set(range(start_vcol, start_vcol + colspan))
            if not occupied_cols:
                continue

            overlap_rows = 0
            while overlap_rows < rowspan:
                covered_cols = initial_occupied.get(row_idx + overlap_rows, set())
                if not occupied_cols.issubset(covered_cols):
                    break
                overlap_rows += 1

            if overlap_rows == 0:
                continue

            remaining_rowspan = rowspan - overlap_rows
            target_row_idx = row_idx + overlap_rows
            if remaining_rowspan > 0 and target_row_idx >= len(rows):
                continue

            cells_to_remove.append(cell)
            if remaining_rowspan > 0:
                moved_cell = deepcopy(cell)
                if remaining_rowspan > 1:
                    moved_cell["rowspan"] = str(remaining_rowspan)
                else:
                    moved_cell.attrs.pop("rowspan", None)
                cells_to_move.append((target_row_idx, start_vcol, moved_cell))

    if not cells_to_remove:
        return False

    for cell in cells_to_remove:
        cell.extract()

    for target_row_idx, start_vcol, moved_cell in sorted(
        cells_to_move,
        key=lambda item: (item[0], item[1]),
        reverse=True,
    ):
        _insert_cell_before_visual_column(rows, target_row_idx, start_vcol, moved_cell)

    return True


def _apply_cell_merge(
    previous_state: TableMergeState,
    current_state: TableMergeState,
    header_count: int,
) -> bool:
    """应用 cell_merge 语义合并。

    当 cell_merge 中的值为 1 时，将下表第一数据行对应单元格的内容
    追加到上表最后一行对应单元格中。全部为 1 时删除该数据行，
    混合时清空已合并单元格的内容但保留行。

    cell_merge 按视觉列索引对齐，通过构建视觉列映射来正确匹配
    两个表格中可能因 rowspan 而具有不同 <td> 元素数量的行。
    元数据仅从当前页 table body 读取，不依赖外层 table block。
    """
    current_body_block = current_state.body_block
    if not isinstance(current_body_block, dict):
        return False

    cell_merge = current_body_block.get("cell_merge")
    if not isinstance(cell_merge, list) or not cell_merge:
        return False

    rows2 = current_state.rows
    if header_count >= len(rows2):
        return False
    if not previous_state.rows:
        return False

    first_data_row = rows2[header_count]
    last_row = previous_state.rows[-1]

    cells1 = last_row.find_all(["td", "th"])
    cells2 = first_data_row.find_all(["td", "th"])

    # 构建视觉列到单元格索引的映射
    last_row_idx = len(previous_state.rows) - 1
    vcol_map1 = build_visual_col_mapping(previous_state.rows, last_row_idx)
    current_merge_rows = rows2[header_count:]
    vcol_map2 = build_visual_col_mapping(
        current_merge_rows,
        0,
        initial_occupied=previous_state.tail_occupied,
    )

    # 构建视觉列 -> 单元格索引的反向映射（展开 colspan）
    vcol_to_cell1: dict[int, int] = {}
    for ci, start_vcol in enumerate(vcol_map1):
        colspan = int(cells1[ci].get("colspan", 1))
        for c in range(start_vcol, start_vcol + colspan):
            vcol_to_cell1[c] = ci
    vcol_to_cell2: dict[int, int] = {}
    for ci, start_vcol in enumerate(vcol_map2):
        colspan = int(cells2[ci].get("colspan", 1))
        for c in range(start_vcol, start_vcol + colspan):
            vcol_to_cell2[c] = ci

    # 按唯一 (src_cell_idx, dst_cell_idx) 对执行一次转移，避免 colspan 重复处理
    transferred_pairs: set[tuple[int, int]] = set()
    for vi, merge_flag in enumerate(cell_merge):
        if merge_flag == 1:
            ci1 = vcol_to_cell1.get(vi)
            ci2 = vcol_to_cell2.get(vi)
            if ci1 is not None and ci2 is not None:
                pair = (ci1, ci2)
                if pair not in transferred_pairs:
                    for child in list(cells2[ci2].children):
                        cells1[ci1].append(child.extract())
                    transferred_pairs.add(pair)

    # 只清空确实成功转移过的源单元格
    cleared_ci2: set[int] = set()
    for vi, merge_flag in enumerate(cell_merge):
        if merge_flag == 1:
            ci1 = vcol_to_cell1.get(vi)
            ci2 = vcol_to_cell2.get(vi)
            if ci1 is not None and ci2 is not None and ci2 not in cleared_ci2:
                cells2[ci2].clear()
                cleared_ci2.add(ci2)

    if not _row_has_semantic_content(first_data_row):
        _carry_rowspan_structure_to_next_row(rows2, header_count)
        first_data_row.extract()
        if first_data_row in rows2:
            rows2.remove(first_data_row)

    return bool(transferred_pairs)


def _perform_table_content_merge(
    previous_state: TableMergeState,
    current_state: TableMergeState,
    previous_table_block: BlockDict,
    current_table_block: BlockDict,
) -> bool:
    """在两个克隆表格上执行 HTML、单元格和 footnote 的内容合并。"""
    header_count, _, _ = detect_table_headers(previous_state, current_state)
    header_count = _expand_header_count_by_rowspan(current_state.rows, header_count)

    rows1 = previous_state.rows
    rows2 = current_state.rows
    if not rows1 or header_count >= len(rows2):
        return False

    previous_adjusted = False

    if header_count < len(rows2):
        current_merge_rows = rows2[header_count:]
        if _clip_overlapped_blank_rowspan_cells(current_merge_rows, previous_state.tail_occupied):
            _refresh_table_state_metrics(current_state)

    if rows1 and rows2 and header_count < len(rows2):
        last_row1 = rows1[-1]
        first_data_row2 = rows2[header_count]
        table_cols1 = previous_state.total_cols
        table_cols2 = current_state.total_cols

        if table_cols1 > table_cols2:
            reference_structure = [int(cell.get("colspan", 1)) for cell in last_row1.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(last_row1)
            adjust_table_rows_colspan(
                rows2,
                header_count,
                len(rows2),
                current_state.row_effective_cols,
                reference_structure,
                reference_visual_cols,
                table_cols1,
                first_data_row2,
            )
        elif table_cols2 > table_cols1:
            reference_structure = [int(cell.get("colspan", 1)) for cell in first_data_row2.find_all(["td", "th"])]
            reference_visual_cols = calculate_visual_columns(first_data_row2)
            adjust_table_rows_colspan(
                rows1,
                0,
                len(rows1),
                previous_state.row_effective_cols,
                reference_structure,
                reference_visual_cols,
                table_cols2,
                last_row1,
            )
            previous_adjusted = True

    if previous_adjusted:
        _refresh_table_state_metrics(previous_state)

    cell_merge_applied = _apply_cell_merge(previous_state, current_state, header_count)

    appended_rows = rows2[header_count:]
    append_start_idx = len(previous_state.rows)
    merged_rows = []

    if previous_state.tbody is None or current_state.tbody is None:
        return False

    for row in appended_rows:
        row.extract()
        previous_state.tbody.append(row)
        merged_rows.append(row)

    if not merged_rows and not cell_merge_applied:
        return False

    previous_state.rows.extend(merged_rows)

    if merged_rows:
        appended_scan = _scan_rows(
            merged_rows,
            initial_occupied=previous_state.tail_occupied,
            start_row_idx=append_start_idx,
        )
        previous_state.row_effective_cols.extend(appended_scan.row_effective_cols)
        previous_state.total_cols = max(previous_state.total_cols, appended_scan.total_cols)
        if appended_scan.last_nonempty_row_metrics is not None:
            previous_state.last_data_row_metrics = appended_scan.last_nonempty_row_metrics
        previous_state.tail_occupied = appended_scan.tail_occupied

    previous_content = previous_table_block.get("content")
    if not isinstance(previous_content, list):
        return False

    previous_table_block["content"] = [
        block for block in previous_content if not isinstance(block, dict) or block.get("type") != BlockType.TABLE_FOOTNOTE
    ]
    current_footnotes = [
        block for block in _table_children(current_table_block) if block.get("type") == BlockType.TABLE_FOOTNOTE
    ]
    footnote_base_index = _build_post_body_child_index(previous_table_block, 0)
    for footnote_offset, table_footnote in enumerate(current_footnotes, start=1):
        temp_table_footnote = deepcopy(table_footnote)
        temp_table_footnote.pop("_cross_page", None)
        if footnote_base_index is None:
            temp_table_footnote["index"] = 0
        else:
            temp_table_footnote["index"] = footnote_base_index + footnote_offset
        previous_table_block["content"].append(temp_table_footnote)

    previous_state.dirty = True
    return _serialize_table_state_html(previous_state)


def merge_table_content(previous_table: BlockDict, current_table: BlockDict) -> BlockDict | None:
    """纯函数式合并两张跨页表格的内容，失败时返回 ``None``。

    两个输入都会先深拷贝；返回块保留前表外层信息，只改克隆表体 HTML
    并用当前表 footnote 替换前表 footnote，不会修改任何输入对象。
    """
    if (
        not isinstance(previous_table, dict)
        or not isinstance(current_table, dict)
        or previous_table.get("type") != BlockType.TABLE
        or current_table.get("type") != BlockType.TABLE
    ):
        return None

    previous_clone = deepcopy(previous_table)
    current_clone = deepcopy(current_table)
    try:
        previous_state = _build_table_state(previous_clone)
        current_state = _build_table_state(current_clone)
        if previous_state is None or current_state is None:
            return None
        if not can_merge_tables(current_state, previous_state):
            return None
        if not _perform_table_content_merge(
            previous_state,
            current_state,
            previous_clone,
            current_clone,
        ):
            return None
    except (AssertionError, TypeError, ValueError):
        return None

    return previous_clone


def _clear_table_continuation_marker(table_block: BlockDict) -> None:
    """递归清除 table 根块及其子块中过期的 ``continues_prev``。"""
    table_block.pop("continues_prev", None)
    content = table_block.get("content")
    if not isinstance(content, list):
        return
    for child in content:
        if isinstance(child, dict):
            child.pop("continues_prev", None)
            _clear_nested_continuation_markers(child)


def _clear_nested_continuation_markers(block: BlockDict) -> None:
    """清除表格子树中的旧延续标记，避免标记落到嵌套子块。"""
    content = block.get("content")
    if not isinstance(content, list):
        return
    for child in content:
        if isinstance(child, dict):
            child.pop("continues_prev", None)
            _clear_nested_continuation_markers(child)


def _find_boundary_table(blocks: list[Any], *, from_end: bool) -> BlockDict | None:
    """从页边界扫描 table；噪声块可跳过，其他语义块立即阻断。"""
    ordered_blocks = reversed(blocks) if from_end else iter(blocks)
    for block in ordered_blocks:
        if not isinstance(block, dict):
            return None
        block_type = block.get("type")
        if block_type in TABLE_BOUNDARY_IGNORED_TYPES:
            continue
        if block_type == BlockType.TABLE:
            return block
        return None
    return None


def _is_consecutive_page_pair(previous_page: PageInfoDict, current_page: PageInfoDict) -> bool:
    """按显式零基 page_idx 判断页面在文档中是否严格连续。"""
    previous_page_idx = previous_page.get("page_idx")
    current_page_idx = current_page.get("page_idx")
    return type(previous_page_idx) is int and type(current_page_idx) is int and current_page_idx == previous_page_idx + 1


def merge_table(page_info_list: list[PageInfoDict]) -> None:
    """倒序识别连续页边界表格，并只在后表写入延续标记。"""
    if not isinstance(page_info_list, list):
        return

    for page_info in page_info_list:
        if not isinstance(page_info, dict):
            continue
        blocks = page_info.get("blocks")
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if isinstance(block, dict) and block.get("type") == BlockType.TABLE:
                _clear_table_continuation_marker(block)

    state_cache: dict[int, TableMergeState] = {}

    for page_position in range(len(page_info_list) - 1, 0, -1):
        current_page = page_info_list[page_position]
        previous_page = page_info_list[page_position - 1]
        if not isinstance(current_page, dict) or not isinstance(previous_page, dict):
            continue
        if not _is_consecutive_page_pair(previous_page, current_page):
            continue

        current_blocks = current_page.get("blocks")
        previous_blocks = previous_page.get("blocks")
        if not isinstance(current_blocks, list) or not isinstance(previous_blocks, list):
            continue

        current_table_block = _find_boundary_table(current_blocks, from_end=False)
        previous_table_block = _find_boundary_table(previous_blocks, from_end=True)
        if current_table_block is None or previous_table_block is None:
            continue

        current_state = _get_or_create_table_state(current_table_block, state_cache)
        previous_state = _get_or_create_table_state(previous_table_block, state_cache)
        if current_state is None or previous_state is None:
            continue

        if not can_merge_tables(current_state, previous_state):
            continue

        current_table_block["continues_prev"] = True
