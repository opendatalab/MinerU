# Copyright (c) Opendatalab. All rights reserved.
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from bs4 import BeautifulSoup

from mineru.backend.vlm.vlm_middle_json_mkcontent import merge_para_with_text
from mineru.utils.char_utils import full_to_half
from mineru.utils.enum_class import BlockType, SplitFlag


CONTINUATION_END_MARKERS = [
    "(续)",
    "(续表)",
    "(续上表)",
    "(continued)",
    "(cont.)",
    "(cont’d)",
    "(…continued)",
    "续表",
]

CONTINUATION_INLINE_MARKERS = [
    "(continued)",
]

MAX_HEADER_ROWS = 5


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
class RowScanResult:
    row_effective_cols: list[int]
    row_metrics: list[RowMetrics]
    total_cols: int
    last_nonempty_row_metrics: RowMetrics | None
    tail_occupied: dict[int, set[int]]


@dataclass
class TableMergeState:
    owner_block: dict[str, Any]
    body_span: dict[str, Any]
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


def _normalize_cell_text(cell) -> str:
    return "".join(full_to_half(cell.get_text()).split())


def _display_cell_text(cell) -> str:
    return full_to_half(cell.get_text().strip())


def _scan_rows(rows, initial_occupied: dict[int, set[int]] | None = None, start_row_idx: int = 0) -> RowScanResult:
    """Scan rows once and cache effective-column metrics.

    initial_occupied stores future-row occupancy relative to the first scanned row
    and preserves rowspans that cross a merge boundary.
    """
    occupied: dict[int, dict[int, bool]] = {}
    max_cols = 0

    for row_offset, cols in (initial_occupied or {}).items():
        if not cols:
            continue
        occupied[row_offset] = {col: True for col in cols}
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

            colspan = int(cell.get("colspan", 1))
            rowspan = int(cell.get("rowspan", 1))
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
        row_idx - len(rows): set(cols.keys())
        for row_idx, cols in occupied.items()
        if row_idx >= len(rows) and cols
    }

    return RowScanResult(
        row_effective_cols=row_effective_cols,
        row_metrics=row_metrics,
        total_cols=max_cols,
        last_nonempty_row_metrics=last_nonempty_row_metrics,
        tail_occupied=tail_occupied,
    )


def _build_row_signature(row, effective_cols: int) -> RowSignature:
    cells = row.find_all(["td", "th"])
    return RowSignature(
        effective_cols=effective_cols,
        colspans=tuple(int(cell.get("colspan", 1)) for cell in cells),
        rowspans=tuple(int(cell.get("rowspan", 1)) for cell in cells),
        normalized_texts=tuple(_normalize_cell_text(cell) for cell in cells),
        display_texts=tuple(_display_cell_text(cell) for cell in cells),
    )


def _build_front_cache(rows, max_header_rows: int = MAX_HEADER_ROWS) -> tuple[list[RowSignature], dict[int, RowMetrics]]:
    front_limit = min(len(rows), max_header_rows + 1)
    front_rows = rows[:front_limit]
    front_scan = _scan_rows(front_rows)

    front_header_info = [
        _build_row_signature(front_rows[idx], front_scan.row_effective_cols[idx])
        for idx in range(min(len(front_rows), max_header_rows))
    ]
    front_first_data_row_metrics = {
        idx: metrics for idx, metrics in enumerate(front_scan.row_metrics)
    }
    return front_header_info, front_first_data_row_metrics


def _find_table_body_span(table_block):
    for block in table_block["blocks"]:
        if block["type"] == BlockType.TABLE_BODY and block["lines"] and block["lines"][0]["spans"]:
            return block["lines"][0]["spans"][0]
    return None


def _refresh_table_state_metrics(state: TableMergeState) -> None:
    scan = _scan_rows(state.rows)
    state.row_effective_cols = scan.row_effective_cols
    state.total_cols = scan.total_cols
    state.last_data_row_metrics = scan.last_nonempty_row_metrics
    state.tail_occupied = scan.tail_occupied
    state.front_header_info, state.front_first_data_row_metrics = _build_front_cache(state.rows)


def _build_table_state(table_block, max_header_rows: int = MAX_HEADER_ROWS) -> TableMergeState | None:
    body_span = _find_table_body_span(table_block)
    if body_span is None:
        return None

    html = body_span.get("html", "")
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")
    tbody = soup.find("tbody") or soup.find("table")
    rows = soup.find_all("tr")
    scan = _scan_rows(rows)
    front_header_info, front_first_data_row_metrics = _build_front_cache(rows, max_header_rows=max_header_rows)

    return TableMergeState(
        owner_block=table_block,
        body_span=body_span,
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
    table_block,
    state_cache: dict[int, TableMergeState],
    max_header_rows: int = MAX_HEADER_ROWS,
) -> TableMergeState | None:
    cache_key = id(table_block)
    state = state_cache.get(cache_key)
    if state is not None:
        return state

    state = _build_table_state(table_block, max_header_rows=max_header_rows)
    if state is not None:
        state_cache[cache_key] = state
    return state


def _serialize_table_state_html(state: TableMergeState) -> None:
    state.body_span["html"] = str(state.soup)
    state.dirty = False


def calculate_table_total_columns(soup):
    """计算表格的总列数，通过分析整个表格结构来处理rowspan和colspan."""
    rows = soup.find_all("tr")
    return _scan_rows(rows).total_cols if rows else 0


def build_table_occupied_matrix(soup):
    """构建表格的占用矩阵，返回每行的有效列数."""
    rows = soup.find_all("tr")
    if not rows:
        return {}

    scan = _scan_rows(rows)
    return {
        row_idx: effective_cols
        for row_idx, effective_cols in enumerate(scan.row_effective_cols)
    }


def calculate_row_effective_columns(soup, row_idx):
    """计算指定行的有效列数（考虑rowspan占用）."""
    row_effective_cols = build_table_occupied_matrix(soup)
    return row_effective_cols.get(row_idx, 0)


def calculate_row_columns(row):
    """计算表格行的实际列数，考虑colspan属性."""
    cells = row.find_all(["td", "th"])
    column_count = 0

    for cell in cells:
        colspan = int(cell.get("colspan", 1))
        column_count += colspan

    return column_count


def calculate_visual_columns(row):
    """计算表格行的视觉列数（实际td/th单元格数量，不考虑colspan）."""
    cells = row.find_all(["td", "th"])
    return len(cells)


def detect_table_headers(state1: TableMergeState, state2: TableMergeState, max_header_rows: int = MAX_HEADER_ROWS):
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
        header_rows, headers_match, header_texts = _detect_table_headers_visual(
            state1, state2, max_header_rows=max_header_rows
        )

    return header_rows, headers_match, header_texts


def _detect_table_headers_visual(
    state1: TableMergeState,
    state2: TableMergeState,
    max_header_rows: int = MAX_HEADER_ROWS,
):
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
        if row1.normalized_texts == row2.normalized_texts and row1.effective_cols == row2.effective_cols:
            header_rows += 1
            header_texts.append(list(row1.display_texts))
        else:
            headers_match = header_rows > 0
            break

    if header_rows == 0:
        headers_match = False

    return header_rows, headers_match, header_texts


def can_merge_tables(current_state: TableMergeState, previous_state: TableMergeState):
    """判断两个表格是否可以合并."""
    current_table_block = current_state.owner_block
    previous_table_block = previous_state.owner_block

    footnote_count = sum(
        1 for block in previous_table_block["blocks"] if block["type"] == BlockType.TABLE_FOOTNOTE
    )
    caption_blocks = [
        block for block in current_table_block["blocks"] if block["type"] == BlockType.TABLE_CAPTION
    ]
    if caption_blocks:
        has_continuation_marker = False
        for block in caption_blocks:
            caption_text = full_to_half(merge_para_with_text(block).strip()).lower()
            if (
                any(caption_text.endswith(marker.lower()) for marker in CONTINUATION_END_MARKERS)
                or any(marker.lower() in caption_text for marker in CONTINUATION_INLINE_MARKERS)
            ):
                has_continuation_marker = True
                break

        if not has_continuation_marker:
            return False

        if footnote_count > 1:
            return False
    elif footnote_count > 0:
        return False

    x0_t1, _, x1_t1, _ = current_table_block["bbox"]
    x0_t2, _, x1_t2, _ = previous_table_block["bbox"]
    table1_width = x1_t1 - x0_t1
    table2_width = x1_t2 - x0_t2

    if abs(table1_width - table2_width) / min(table1_width, table2_width) >= 0.1:
        return False

    if previous_state.total_cols == current_state.total_cols:
        return True

    return check_rows_match(previous_state, current_state)


def check_rows_match(previous_state: TableMergeState, current_state: TableMergeState):
    """检查表格边界行是否匹配."""
    last_row_metrics = previous_state.last_data_row_metrics
    if last_row_metrics is None:
        return False

    header_count, _, _ = detect_table_headers(previous_state, current_state)
    first_data_row_metrics = current_state.front_first_data_row_metrics.get(header_count)
    if first_data_row_metrics is None:
        return False

    return (
        last_row_metrics.effective_cols == first_data_row_metrics.effective_cols
        or last_row_metrics.actual_cols == first_data_row_metrics.actual_cols
        or last_row_metrics.visual_cols == first_data_row_metrics.visual_cols
    )


def check_row_columns_match(row1, row2):
    cells1 = row1.find_all(["td", "th"])
    cells2 = row2.find_all(["td", "th"])
    if len(cells1) != len(cells2):
        return False
    for cell1, cell2 in zip(cells1, cells2):
        colspan1 = int(cell1.get("colspan", 1))
        colspan2 = int(cell2.get("colspan", 1))
        if colspan1 != colspan2:
            return False
    return True


def adjust_table_rows_colspan(
    rows,
    start_idx,
    end_idx,
    row_effective_cols,
    reference_structure,
    reference_visual_cols,
    target_cols,
    match_reference_row,
):
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

        if (
            calculate_visual_columns(row) == reference_visual_cols
            and check_row_columns_match(row, reference_row_copy)
        ):
            if len(cells) <= len(reference_structure):
                for cell_idx, cell in enumerate(cells):
                    if cell_idx < len(reference_structure) and reference_structure[cell_idx] > 1:
                        cell["colspan"] = str(reference_structure[cell_idx])
        else:
            cols_diff = target_cols - current_row_effective_cols
            if cols_diff > 0:
                last_cell = cells[-1]
                current_last_span = int(last_cell.get("colspan", 1))
                last_cell["colspan"] = str(current_last_span + cols_diff)


def perform_table_merge(
    previous_state: TableMergeState,
    current_state: TableMergeState,
    previous_table_block,
    wait_merge_table_footnotes,
):
    """执行表格合并操作."""
    header_count, _, _ = detect_table_headers(previous_state, current_state)

    rows1 = previous_state.rows
    rows2 = current_state.rows

    previous_adjusted = False

    if rows1 and rows2 and header_count < len(rows2):
        last_row1 = rows1[-1]
        first_data_row2 = rows2[header_count]
        table_cols1 = previous_state.total_cols
        table_cols2 = current_state.total_cols

        if table_cols1 > table_cols2:
            reference_structure = [
                int(cell.get("colspan", 1)) for cell in last_row1.find_all(["td", "th"])
            ]
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
            reference_structure = [
                int(cell.get("colspan", 1)) for cell in first_data_row2.find_all(["td", "th"])
            ]
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

    appended_rows = rows2[header_count:]
    append_start_idx = len(previous_state.rows)
    merged_rows = []

    if previous_state.tbody and current_state.tbody:
        for row in appended_rows:
            row.extract()
            previous_state.tbody.append(row)
            merged_rows.append(row)

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

    previous_table_block["blocks"] = [
        block for block in previous_table_block["blocks"] if block["type"] != BlockType.TABLE_FOOTNOTE
    ]
    for table_footnote in wait_merge_table_footnotes:
        temp_table_footnote = table_footnote.copy()
        temp_table_footnote[SplitFlag.CROSS_PAGE] = True
        previous_table_block["blocks"].append(temp_table_footnote)

    previous_state.dirty = True


def merge_table(page_info_list):
    """合并跨页表格."""
    state_cache: dict[int, TableMergeState] = {}
    merged_away_blocks: set[int] = set()

    for page_idx in range(len(page_info_list) - 1, -1, -1):
        if page_idx == 0:
            continue

        page_info = page_info_list[page_idx]
        previous_page_info = page_info_list[page_idx - 1]

        if not (page_info["para_blocks"] and page_info["para_blocks"][0]["type"] == BlockType.TABLE):
            continue

        if not (
            previous_page_info["para_blocks"]
            and previous_page_info["para_blocks"][-1]["type"] == BlockType.TABLE
        ):
            continue

        current_table_block = page_info["para_blocks"][0]
        previous_table_block = previous_page_info["para_blocks"][-1]

        current_state = _get_or_create_table_state(current_table_block, state_cache)
        previous_state = _get_or_create_table_state(previous_table_block, state_cache)
        if current_state is None or previous_state is None:
            continue

        wait_merge_table_footnotes = [
            block for block in current_table_block["blocks"] if block["type"] == BlockType.TABLE_FOOTNOTE
        ]

        if not can_merge_tables(current_state, previous_state):
            continue

        perform_table_merge(
            previous_state,
            current_state,
            previous_table_block,
            wait_merge_table_footnotes,
        )

        merged_away_blocks.add(id(current_table_block))
        for block in current_table_block["blocks"]:
            block["lines"] = []
            block[SplitFlag.LINES_DELETED] = True

    for state in state_cache.values():
        if state.dirty and id(state.owner_block) not in merged_away_blocks:
            _serialize_table_state_html(state)
