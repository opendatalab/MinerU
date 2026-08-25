# Copyright (c) Opendatalab. All rights reserved.
"""验证 Native PDF 表格结构恢复的网格、文本和候选仲裁。"""

from __future__ import annotations

from collections.abc import Iterable
from io import BytesIO

import pytest
from reportlab.pdfgen.canvas import Canvas

from mineru.utils.native_pdf_table import (
    NativeTableCell,
    NativeTableInput,
    NativeTableRectangle,
    NativeTableRule,
    coerce_native_table_rectangles,
    coerce_native_table_rules,
    recover_native_pdf_table,
)
from mineru.utils.native_pdf_table.candidate import GridCellSpec, build_candidate
from mineru.utils.native_pdf_table.contracts import NativeTableCandidate
from mineru.utils.native_pdf_table.engine import (
    _remove_undercounted_vector_candidates,
    _select_candidate,
    diagnose_native_pdf_table,
)
from mineru.utils.native_pdf_table.text import build_native_table_text
from mineru.utils.native_pdf_table.vector import (
    MAX_PRIMITIVES_PER_TABLE,
    build_vector_candidates,
)
from mineru.utils.pdf_document import PDFDocument


def _char_items(
    entries: Iterable[tuple[str, tuple[float, float, float, float]]],
) -> tuple[dict[str, object], ...]:
    """把文本框拆成带稳定 char_idx 的逐字符测试输入。"""

    rebuilt: list[dict[str, object]] = []
    for text, bbox in entries:
        left, top, right, bottom = bbox
        char_width = (right - left) / max(1, len(text))
        for offset, char in enumerate(text):
            rebuilt.append(
                {
                    "char": char,
                    "bbox": (
                        left + char_width * offset,
                        top,
                        left + char_width * (offset + 1),
                        bottom,
                    ),
                    "char_idx": len(rebuilt),
                }
            )
    return tuple(rebuilt)


def _grid_rules(
    *,
    width: float = 100.0,
    height: float = 60.0,
    internal_vertical: tuple[float, float] = (0.0, 60.0),
    internal_horizontal: tuple[float, float] = (0.0, 100.0),
) -> tuple[NativeTableRule, ...]:
    """构造二行二列表格边框，并允许裁剪内部横竖隔断。"""

    return (
        NativeTableRule((0.0, 0.0, width, 1.0), 1.0, "horizontal"),
        NativeTableRule(
            (internal_horizontal[0], 29.5, internal_horizontal[1], 30.5),
            1.0,
            "horizontal",
        ),
        NativeTableRule((0.0, height - 1.0, width, height), 1.0, "horizontal"),
        NativeTableRule((0.0, 0.0, 1.0, height), 1.0, "vertical"),
        NativeTableRule(
            (49.5, internal_vertical[0], 50.5, internal_vertical[1]),
            1.0,
            "vertical",
        ),
        NativeTableRule((width - 1.0, 0.0, width, height), 1.0, "vertical"),
    )


def _local_to_page_bbox(
    bbox: tuple[float, float, float, float],
    page_width: float,
    page_height: float,
    angle: int,
) -> tuple[float, float, float, float]:
    """把正向局部测试框逆变换回页面坐标。"""

    x0, y0, x1, y1 = bbox
    if angle == 90:
        return page_width - y1, x0, page_width - y0, x1
    if angle == 270:
        return y0, page_height - x1, y1, page_height - x0
    if angle == 180:
        return page_width - x1, page_height - y1, page_width - x0, page_height - y0
    return bbox


def _aligned_text_input(
    rows: list[list[str | None]],
    *,
    angle: int = 0,
    rules: tuple[NativeTableRule, ...] = (),
    rectangles: tuple[NativeTableRectangle, ...] = (),
) -> NativeTableInput:
    """构造标准旋转下的等距文本表格输入。"""

    page_width, page_height = 120.0, 90.0
    local_width, local_height = (page_height, page_width) if angle in {90, 270} else (page_width, page_height)
    col_count = max(len(row) for row in rows)
    x_positions = [local_width * (0.10 + 0.75 * col / max(1, col_count - 1)) for col in range(col_count)]
    y_positions = [local_height * (0.12 + 0.70 * row / max(1, len(rows) - 1)) for row in range(len(rows))]
    entries = []
    for row_index, row in enumerate(rows):
        for col_index, text in enumerate(row):
            if text is None:
                continue
            width = max(4.0, min(local_width * 0.20, len(text) * 4.0))
            local_bbox = (
                x_positions[col_index],
                y_positions[row_index],
                min(local_width - 1.0, x_positions[col_index] + width),
                min(local_height - 1.0, y_positions[row_index] + 8.0),
            )
            entries.append(
                (
                    text,
                    _local_to_page_bbox(
                        local_bbox,
                        page_width,
                        page_height,
                        angle,
                    ),
                )
            )
    return NativeTableInput(
        table_bbox=(0.0, 0.0, page_width, page_height),
        page_size=(page_width, page_height),
        angle=angle,
        chars=_char_items(entries),
        drawing_lines=rules,
        rectangles=rectangles,
    )


def _rotated_single_row_input(angle: int) -> NativeTableInput:
    """构造四种标准旋转下的三列单物理行强线框。"""

    page_width, page_height = 120.0, 90.0
    local_width, local_height = (page_height, page_width) if angle in {90, 270} else (page_width, page_height)
    x_tracks = (0.0, local_width / 3.0, 2.0 * local_width / 3.0, local_width)
    local_rule_bboxes = [
        (0.0, 0.0, local_width, 1.0),
        (0.0, local_height - 1.0, local_width, local_height),
        *((x - 0.5, 0.0, x + 0.5, local_height) for x in x_tracks),
    ]
    page_rule_bboxes = [
        _local_to_page_bbox(
            bbox,
            page_width,
            page_height,
            angle,
        )
        for bbox in local_rule_bboxes
    ]
    rules = tuple(
        NativeTableRule(
            bbox,
            1.0,
            "horizontal" if bbox[2] - bbox[0] >= bbox[3] - bbox[1] else "vertical",
        )
        for bbox in page_rule_bboxes
    )
    entries = []
    for col, text in enumerate(("A", "B", "C")):
        local_bbox = (
            x_tracks[col] + 5.0,
            10.0,
            x_tracks[col] + 10.0,
            18.0,
        )
        entries.append(
            (
                text,
                _local_to_page_bbox(
                    local_bbox,
                    page_width,
                    page_height,
                    angle,
                ),
            )
        )
    return NativeTableInput(
        table_bbox=(0.0, 0.0, page_width, page_height),
        page_size=(page_width, page_height),
        angle=angle,
        chars=_char_items(entries),
        drawing_lines=rules,
    )


def _rotated_single_column_input(angle: int) -> NativeTableInput:
    """构造四种标准旋转下的二行单列强线框表单。"""

    page_width, page_height = 120.0, 90.0
    local_width, local_height = (page_height, page_width) if angle in {90, 270} else (page_width, page_height)
    y_tracks = (0.0, local_height / 2.0, local_height)
    local_rule_bboxes = [
        *((0.0, y - 0.5, local_width, y + 0.5) for y in y_tracks),
        (0.0, 0.0, 1.0, local_height),
        (local_width - 1.0, 0.0, local_width, local_height),
    ]
    page_rule_bboxes = [
        _local_to_page_bbox(
            bbox,
            page_width,
            page_height,
            angle,
        )
        for bbox in local_rule_bboxes
    ]
    rules = tuple(
        NativeTableRule(
            bbox,
            1.0,
            "horizontal" if bbox[2] - bbox[0] >= bbox[3] - bbox[1] else "vertical",
        )
        for bbox in page_rule_bboxes
    )
    chars: list[dict[str, object]] = []
    for row, text in enumerate(("A&B", "Value A B")):
        top = row * local_height / 2.0 + 10.0
        for offset, char in enumerate(text):
            chars.append(
                {
                    "char": char,
                    "bbox": _local_to_page_bbox(
                        (
                            8.0 + 4.0 * offset,
                            top,
                            12.0 + 4.0 * offset,
                            top + 8.0,
                        ),
                        page_width,
                        page_height,
                        angle,
                    ),
                    "char_idx": len(chars),
                }
            )
    return NativeTableInput(
        table_bbox=(0.0, 0.0, page_width, page_height),
        page_size=(page_width, page_height),
        angle=angle,
        chars=tuple(chars),
        drawing_lines=rules,
    )


def _rotated_sparse_hybrid_input(angle: int) -> NativeTableInput:
    """构造四种标准旋转下仅有表头横线的三行三列表格。"""

    page_width, page_height = 120.0, 90.0
    local_width, local_height = (page_height, page_width) if angle in {90, 270} else (page_width, page_height)
    x_tracks = (0.0, local_width / 3.0, 2.0 * local_width / 3.0, local_width)
    header_boundary = 0.30 * local_height
    local_rule_bboxes = [
        (0.0, 0.0, local_width, 1.0),
        (0.0, header_boundary - 0.5, local_width, header_boundary + 0.5),
        (0.0, local_height - 1.0, local_width, local_height),
        *((track - 0.5, 0.0, track + 0.5, local_height) for track in x_tracks),
    ]
    rules = tuple(
        NativeTableRule(
            page_bbox,
            1.0,
            "horizontal" if page_bbox[2] - page_bbox[0] >= page_bbox[3] - page_bbox[1] else "vertical",
        )
        for page_bbox in (
            _local_to_page_bbox(
                bbox,
                page_width,
                page_height,
                angle,
            )
            for bbox in local_rule_bboxes
        )
    )
    y_positions = (5.0, 0.42 * local_height, 0.70 * local_height)
    entries = []
    for row, values in enumerate((("H", "I", "J"), ("A", "1", "2"), ("B", "3", "4"))):
        for col, value in enumerate(values):
            local_bbox = (
                x_tracks[col] + 5.0,
                y_positions[row],
                x_tracks[col] + 13.0,
                y_positions[row] + 8.0,
            )
            entries.append(
                (
                    value,
                    _local_to_page_bbox(
                        local_bbox,
                        page_width,
                        page_height,
                        angle,
                    ),
                )
            )
    return NativeTableInput(
        table_bbox=(0.0, 0.0, page_width, page_height),
        page_size=(page_width, page_height),
        angle=angle,
        chars=_char_items(entries),
        drawing_lines=rules,
    )


def _candidate(
    *,
    source: str,
    rows: int,
    cols: int,
    score: float,
    issues: tuple[str, ...] = (),
) -> NativeTableCandidate:
    """构造候选仲裁测试需要的最小内部对象。"""

    cells = tuple(
        NativeTableCell(
            row=row,
            col=col,
            rowspan=1,
            colspan=1,
            bbox=(float(col), float(row), float(col + 1), float(row + 1)),
            content="",
        )
        for row in range(rows)
        for col in range(cols)
    )
    return NativeTableCandidate(
        source=source,  # type: ignore[arg-type]
        rows=rows,
        cols=cols,
        cells=cells,
        score=score,
        text_capture=1.0,
        structure_support=1.0,
        row_stability=1.0,
        column_stability=1.0,
        order_consistency=1.0,
        issues=issues,
    )


def _build_reportlab_table_pdf() -> bytes:
    """生成带真实 PDF drawing 和文本层的二行二列表格 fixture。"""

    output = BytesIO()
    canvas = Canvas(output, pagesize=(200, 200))
    canvas.setLineWidth(1)
    for y in (40, 100, 160):
        canvas.line(20, y, 180, y)
    for x in (20, 100, 180):
        canvas.line(x, 40, x, 160)
    canvas.drawString(35, 130, "A")
    canvas.drawString(115, 130, "B")
    canvas.drawString(35, 70, "C")
    canvas.drawString(115, 70, "D")
    canvas.showPage()
    canvas.save()
    return output.getvalue()


def test_vector_grid_recovers_cells_and_escapes_html() -> None:
    """验证完整矢量网格生成稳定 HTML，并转义单元格文本。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("<&", (10.0, 10.0, 20.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=_grid_rules(),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert result.source == "vector_grid"
    assert result.html == ("<table><tbody><tr><td>&lt;&amp;</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></tbody></table>")
    assert sorted(index for cell in result.cells for index in cell.source_char_indices) == [0, 1, 2, 3, 4]


def test_vector_grid_splits_one_pdf_text_run_by_character_boundary() -> None:
    """验证物理列边界可将同一 PDF 文本对象按字符安全落格。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("AB", (45.0, 10.0, 55.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=_grid_rules(),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert "<tr><td>A</td><td>B</td></tr>" in result.html


def test_text_candidate_rejects_token_split_across_cell_boundary() -> None:
    """验证纯文本轨道不将一个原子 token 分割到两个逻辑单元格。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("AB", (45.0, 10.0, 55.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
    )
    text = build_native_table_text(table_input)
    assert text is not None
    specs = (
        GridCellSpec(0, 0, 1, 1, (0.0, 0.0, 50.0, 30.0)),
        GridCellSpec(0, 1, 1, 1, (50.0, 0.0, 100.0, 30.0)),
        GridCellSpec(1, 0, 1, 1, (0.0, 30.0, 50.0, 60.0)),
        GridCellSpec(1, 1, 1, 1, (50.0, 30.0, 100.0, 60.0)),
    )
    diagnostics: dict[str, object] = {}

    candidate = build_candidate(
        source="text_grid",
        rows=2,
        cols=2,
        specs=specs,
        text=text,
        structure_support=1.0,
        row_stability=1.0,
        column_stability=1.0,
        require_atomic_tokens=True,
        diagnostics=diagnostics,
    )

    assert candidate is None
    assert diagnostics["candidate_rejection_gate"] == "token_split"
    assert diagnostics["token_split_count"] == 1


def test_vector_grid_accepts_open_outer_vertical_edges() -> None:
    """验证横线端点可补齐缺失的左右外框而不伪造内部隔断。"""

    rules = (
        NativeTableRule((0.0, 0.0, 100.0, 1.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 29.5, 100.0, 30.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 59.0, 100.0, 60.0), 1.0, "horizontal"),
        NativeTableRule((49.5, 0.0, 50.5, 60.0), 1.0, "vertical"),
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=rules,
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert result.source == "vector_grid"
    assert (result.rows, result.cols) == (2, 2)


def test_single_long_rule_endpoint_does_not_create_global_column() -> None:
    """验证单条长横线的内缩端点不能创建全局幽灵列。"""

    rules = list(_grid_rules())
    rules[0] = NativeTableRule((10.0, 0.0, 100.0, 1.0), 1.0, "horizontal")
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=tuple(rules),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (2, 2)


def test_sparse_hybrid_splits_dense_keyed_baselines_after_line_undercount() -> None:
    """验证只有表头横线时按独立关键列把稠密正文基线恢复成多行。"""

    rules = (
        NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 19.5, 120.0, 20.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 59.0, 120.0, 60.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
        NativeTableRule((39.5, 0.0, 40.5, 60.0), 1.0, "vertical"),
        NativeTableRule((79.5, 0.0, 80.5, 60.0), 1.0, "vertical"),
        NativeTableRule((119.0, 0.0, 120.0, 60.0), 1.0, "vertical"),
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 60.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("H1", (8.0, 5.0, 16.0, 13.0)),
                ("H2", (48.0, 5.0, 56.0, 13.0)),
                ("H3", (88.0, 5.0, 96.0, 13.0)),
                ("A", (8.0, 27.0, 12.0, 35.0)),
                ("1", (48.0, 27.0, 52.0, 35.0)),
                ("2", (88.0, 27.0, 92.0, 35.0)),
                ("B", (8.0, 42.0, 12.0, 50.0)),
                ("3", (48.0, 42.0, 52.0, 50.0)),
                ("4", (88.0, 42.0, 92.0, 50.0)),
            ]
        ),
        drawing_lines=rules,
    )

    diagnostics = diagnose_native_pdf_table(table_input)

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert result.source == "sparse_hybrid"
    assert (result.rows, result.cols) == (3, 3)
    assert "<tr><td>B</td><td>3</td><td>4</td></tr>" in result.html
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["first_rejection_gate"] == "physical_row_undercount"
    assert line_attempt["physical_row_dense_baseline_pairs"] == [
        {
            "physical_row": 1,
            "visual_rows": [1, 2],
            "occupied_cols": [0, 1, 2],
        }
    ]


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_sparse_hybrid_supports_standard_table_rotations(angle: int) -> None:
    """验证竖线加视觉正文行的少线候选支持四种标准旋转。"""

    result = recover_native_pdf_table(_rotated_sparse_hybrid_input(angle))

    assert result is not None
    assert result.source == "sparse_hybrid"
    assert (result.rows, result.cols) == (3, 3)
    assert "<tr><td>B</td><td>3</td><td>4</td></tr>" in result.html


def test_sparse_hybrid_recovers_two_level_header_spans() -> None:
    """验证局部表头横线恢复左侧 rowspan 和分组标题 colspan。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 100.0),
        page_size=(120.0, 100.0),
        angle=0,
        chars=_char_items(
            [
                ("Site", (8.0, 5.0, 24.0, 13.0)),
                ("Metrics", (60.0, 5.0, 88.0, 13.0)),
                ("Q1", (48.0, 25.0, 56.0, 33.0)),
                ("Q2", (88.0, 25.0, 96.0, 33.0)),
                ("A", (8.0, 55.0, 12.0, 63.0)),
                ("1", (48.0, 55.0, 52.0, 63.0)),
                ("2", (88.0, 55.0, 92.0, 63.0)),
                ("B", (8.0, 78.0, 12.0, 86.0)),
                ("3", (48.0, 78.0, 52.0, 86.0)),
                ("4", (88.0, 78.0, 92.0, 86.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((30.0, 19.5, 120.0, 20.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 99.0, 120.0, 100.0), 1.0, "horizontal"),
        ),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert result.source == "sparse_hybrid"
    assert (result.rows, result.cols) == (4, 3)
    assert '<td rowspan="2">Site</td><td colspan="2">Metrics</td>' in result.html
    assert "<tr><td>Q1</td><td>Q2</td></tr>" in result.html


def test_sparse_hybrid_rejects_ambiguous_partial_header_separator() -> None:
    """验证表头横线只覆盖部分叶子列时不猜测 rowspan/colspan。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 100.0),
        page_size=(120.0, 100.0),
        angle=0,
        chars=_char_items(
            [
                ("Site", (8.0, 5.0, 24.0, 13.0)),
                ("Metrics", (60.0, 5.0, 88.0, 13.0)),
                ("Q1", (48.0, 25.0, 56.0, 33.0)),
                ("Q2", (88.0, 25.0, 96.0, 33.0)),
                ("A", (8.0, 55.0, 12.0, 63.0)),
                ("1", (48.0, 55.0, 52.0, 63.0)),
                ("2", (88.0, 55.0, 92.0, 63.0)),
                ("B", (8.0, 78.0, 12.0, 86.0)),
                ("3", (48.0, 78.0, 52.0, 86.0)),
                ("4", (88.0, 78.0, 92.0, 86.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((40.0, 19.5, 120.0, 20.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 99.0, 120.0, 100.0), 1.0, "horizontal"),
        ),
    )

    diagnostics = diagnose_native_pdf_table(table_input)

    assert recover_native_pdf_table(table_input) is None
    sparse_attempt = diagnostics["sparse_hybrid_attempts"][0]
    assert sparse_attempt["first_rejection_gate"] == "header_topology"


def test_vector_grid_joins_small_collinear_gaps() -> None:
    """验证内部竖线的小间隙会先连接，不会误判为 colspan。"""

    rules = tuple(rule for rule in _grid_rules() if not (rule.orientation == "vertical" and 49.0 <= rule.bbox[0] <= 51.0)) + (
        NativeTableRule((49.5, 0.0, 50.5, 14.0), 1.0, "vertical"),
        NativeTableRule((49.5, 16.0, 50.5, 60.0), 1.0, "vertical"),
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=rules,
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert "colspan" not in result.html


def test_line_grid_ignores_stroked_rectangle_duplicate_tracks() -> None:
    """验证 drawing 与描边矩形重复边界不会混成幽灵行列。"""

    rectangles = tuple(
        NativeTableRectangle(bbox, 5, False, True)
        for bbox in (
            (0.5, 0.5, 49.5, 29.5),
            (50.5, 0.5, 99.5, 29.5),
            (0.5, 30.5, 49.5, 59.5),
            (50.5, 30.5, 99.5, 59.5),
        )
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=_grid_rules(),
        rectangles=rectangles,
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (2, 2)
    assert "evidence=line_grid" in result.diagnostics


def test_rect_grid_requires_repeated_two_dimensional_lattice() -> None:
    """验证四个重复单元格矩形可独立形成 rect_grid。"""

    rectangles = tuple(
        NativeTableRectangle(bbox, 5, True, False)
        for bbox in (
            (0.0, 0.0, 50.0, 30.0),
            (50.0, 0.0, 100.0, 30.0),
            (0.0, 30.0, 50.0, 60.0),
            (50.0, 30.0, 100.0, 60.0),
        )
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        rectangles=rectangles,
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (2, 2)
    assert "evidence=rect_grid" in result.diagnostics


def test_vector_grid_collapses_narrow_empty_duplicate_track() -> None:
    """验证窄空列会保留较强真实边界并删除重复竖轨。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=(
            *_grid_rules(),
            NativeTableRule((50.5, 0.0, 51.5, 60.0), 1.0, "vertical"),
        ),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (2, 2)
    assert "colspan" not in result.html


def test_vector_grid_uses_aliases_for_offset_complementary_separators() -> None:
    """验证表头正文错位的同一竖线折叠后仍按 alias 保持分隔。"""

    rules = tuple(rule for rule in _grid_rules() if not (rule.orientation == "vertical" and 49.0 <= rule.bbox[0] <= 51.0)) + (
        NativeTableRule((48.5, 0.0, 49.5, 30.0), 1.0, "vertical"),
        NativeTableRule((50.5, 30.0, 51.5, 60.0), 1.0, "vertical"),
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=rules,
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (2, 2)
    assert "colspan" not in result.html
    assert "alias_separator_recoveries=2" in result.diagnostics


def test_vector_grid_rejects_non_unique_alias_chain() -> None:
    """验证连续窄轨的 alias 总跨度超限时主动放弃。"""

    rules = (
        NativeTableRule((0.0, 0.0, 100.0, 1.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 29.5, 100.0, 30.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 59.0, 100.0, 60.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
        NativeTableRule((44.5, 0.0, 45.5, 60.0), 1.0, "vertical"),
        NativeTableRule((48.5, 0.0, 49.5, 60.0), 1.0, "vertical"),
        NativeTableRule((52.5, 0.0, 53.5, 60.0), 1.0, "vertical"),
        NativeTableRule((99.0, 0.0, 100.0, 60.0), 1.0, "vertical"),
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (70.0, 10.0, 75.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (70.0, 40.0, 75.0, 48.0)),
            ]
        ),
        drawing_lines=rules,
    )

    text = build_native_table_text(table_input)

    assert text is not None
    assert build_vector_candidates(table_input, text) == []


def test_vector_grid_rejects_single_column_region_with_narrow_ghost_track() -> None:
    """验证单列区域不能借一个窄空轨伪装成二列表格。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("Header", (20.0, 10.0, 44.0, 18.0)),
                ("Value", (20.0, 40.0, 40.0, 48.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 100.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 29.5, 100.0, 30.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 59.0, 100.0, 60.0), 1.0, "horizontal"),
            NativeTableRule((4.5, 0.0, 5.5, 60.0), 1.0, "vertical"),
        ),
    )

    text = build_native_table_text(table_input)

    assert text is not None
    assert build_vector_candidates(table_input, text) == []


def test_vector_grid_accepts_one_text_row_with_blank_physical_row() -> None:
    """验证强物理二行网格即使只有一行原生文本也可保留空白数据行。"""

    rules = (
        NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 29.5, 120.0, 30.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 59.0, 120.0, 60.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
        NativeTableRule((39.5, 0.0, 40.5, 60.0), 1.0, "vertical"),
        NativeTableRule((79.5, 0.0, 80.5, 60.0), 1.0, "vertical"),
        NativeTableRule((119.0, 0.0, 120.0, 60.0), 1.0, "vertical"),
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 60.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 14.0, 18.0)),
                ("B", (50.0, 10.0, 54.0, 18.0)),
                ("C", (90.0, 10.0, 94.0, 18.0)),
            ]
        ),
        drawing_lines=rules,
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (2, 3)
    assert result.html.endswith("<tr><td></td><td></td><td></td></tr></tbody></table>")


def test_line_grid_accepts_multilevel_header_and_blank_template_row() -> None:
    """验证强线框多层表头可保留全宽空白模板行。"""

    rules = (
        NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 19.5, 120.0, 20.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 39.5, 120.0, 40.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 59.0, 120.0, 60.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
        NativeTableRule((39.5, 0.0, 40.5, 40.0), 1.0, "vertical"),
        NativeTableRule((79.5, 0.0, 80.5, 40.0), 1.0, "vertical"),
        NativeTableRule((119.0, 0.0, 120.0, 60.0), 1.0, "vertical"),
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 60.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 5.0, 14.0, 13.0)),
                ("B", (50.0, 5.0, 54.0, 13.0)),
                ("C", (90.0, 5.0, 94.0, 13.0)),
                ("D", (10.0, 25.0, 14.0, 33.0)),
                ("E", (50.0, 25.0, 54.0, 33.0)),
                ("F", (90.0, 25.0, 94.0, 33.0)),
            ]
        ),
        drawing_lines=rules,
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (3, 3)
    assert result.confidence >= 0.95
    assert result.html.endswith('<tr><td colspan="3"></td></tr></tbody></table>')
    assert "physical_blank_rows=2" in result.diagnostics


def test_line_grid_accepts_blank_row_when_alias_recovery_is_elsewhere() -> None:
    """验证其他物理行的 alias 恢复不会误伤独立封闭的空白行。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 60.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 5.0, 14.0, 13.0)),
                ("B", (70.0, 5.0, 74.0, 13.0)),
                ("C", (10.0, 25.0, 14.0, 33.0)),
                ("D", (70.0, 25.0, 74.0, 33.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 19.5, 120.0, 20.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 39.5, 120.0, 40.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 59.0, 120.0, 60.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
            NativeTableRule((48.5, 0.0, 49.5, 20.0), 1.0, "vertical"),
            NativeTableRule((50.5, 20.0, 51.5, 40.0), 1.0, "vertical"),
            NativeTableRule((119.0, 0.0, 120.0, 60.0), 1.0, "vertical"),
        ),
    )

    result = recover_native_pdf_table(table_input)
    diagnostics = diagnose_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (3, 2)
    assert result.html.endswith('<tr><td colspan="2"></td></tr></tbody></table>')
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["alias_affected_rows"] == [0, 1]
    assert line_attempt["empty_rows"] == [2]


def test_line_grid_rejects_blank_row_touched_by_alias_recovery() -> None:
    """验证空白行自身依赖 alias separator 时仍主动回退。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 60.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 5.0, 14.0, 13.0)),
                ("B", (70.0, 5.0, 74.0, 13.0)),
                ("C", (10.0, 25.0, 14.0, 33.0)),
                ("D", (70.0, 25.0, 74.0, 33.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 19.5, 120.0, 20.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 39.5, 120.0, 40.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 59.0, 120.0, 60.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
            NativeTableRule((48.5, 0.0, 49.5, 20.0), 1.0, "vertical"),
            NativeTableRule((50.5, 20.0, 51.5, 40.0), 1.0, "vertical"),
            NativeTableRule((48.5, 40.0, 49.5, 60.0), 1.0, "vertical"),
            NativeTableRule((119.0, 0.0, 120.0, 60.0), 1.0, "vertical"),
        ),
    )

    result = recover_native_pdf_table(table_input)
    diagnostics = diagnose_native_pdf_table(table_input)
    assert result is None
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["first_rejection_gate"] == "empty_row"
    assert 2 in line_attempt["alias_affected_rows"]


def test_line_grid_rejects_blank_row_without_bottom_boundary() -> None:
    """验证缺少独立下边界的空白行不能借端点推断进入 HTML。"""

    base = (
        NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 19.5, 120.0, 20.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 39.5, 120.0, 40.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
        NativeTableRule((39.5, 0.0, 40.5, 40.0), 1.0, "vertical"),
        NativeTableRule((79.5, 0.0, 80.5, 40.0), 1.0, "vertical"),
        NativeTableRule((119.0, 0.0, 120.0, 60.0), 1.0, "vertical"),
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 60.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 5.0, 14.0, 13.0)),
                ("B", (50.0, 5.0, 54.0, 13.0)),
                ("C", (90.0, 5.0, 94.0, 13.0)),
                ("D", (10.0, 25.0, 14.0, 33.0)),
                ("E", (50.0, 25.0, 54.0, 33.0)),
                ("F", (90.0, 25.0, 94.0, 33.0)),
            ]
        ),
        drawing_lines=base,
    )

    text = build_native_table_text(table_input)

    assert text is not None
    assert build_vector_candidates(table_input, text) == []
    diagnostics = diagnose_native_pdf_table(table_input)
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["first_rejection_gate"] == "empty_row"
    assert line_attempt["empty_rows"] == [2]


def test_line_grid_rejects_too_narrow_blank_row() -> None:
    """验证高度不足的空白物理行仍主动回退。"""

    rules = (
        NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 29.5, 120.0, 30.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 55.5, 120.0, 56.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 59.0, 120.0, 60.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
        NativeTableRule((39.5, 0.0, 40.5, 56.0), 1.0, "vertical"),
        NativeTableRule((79.5, 0.0, 80.5, 56.0), 1.0, "vertical"),
        NativeTableRule((119.0, 0.0, 120.0, 60.0), 1.0, "vertical"),
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 60.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 8.0, 14.0, 16.0)),
                ("B", (50.0, 8.0, 54.0, 16.0)),
                ("C", (90.0, 8.0, 94.0, 16.0)),
                ("D", (10.0, 38.0, 14.0, 46.0)),
                ("E", (50.0, 38.0, 54.0, 46.0)),
                ("F", (90.0, 38.0, 94.0, 46.0)),
            ]
        ),
        drawing_lines=rules,
    )

    text = build_native_table_text(table_input)

    assert text is not None
    assert build_vector_candidates(table_input, text) == []


def test_line_grid_prunes_inset_decorative_underlines() -> None:
    """验证单元格内缩短下划线不会创建全表横轨。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 60.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("Left top", (5.0, 5.0, 45.0, 14.0)),
                ("Right top", (65.0, 5.0, 110.0, 14.0)),
                ("Left bottom", (5.0, 35.0, 50.0, 44.0)),
                ("Right bottom", (65.0, 35.0, 115.0, 44.0)),
            ]
        ),
        drawing_lines=(
            *_grid_rules(width=120.0),
            NativeTableRule((5.0, 17.5, 55.0, 18.5), 1.0, "horizontal"),
            NativeTableRule((5.0, 23.5, 20.0, 24.5), 1.0, "horizontal"),
            NativeTableRule((5.0, 47.5, 55.0, 48.5), 1.0, "horizontal"),
        ),
    )

    result = recover_native_pdf_table(table_input)
    diagnostics = diagnose_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (2, 2)
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["track_hypothesis"] == "supported"
    assert len(line_attempt["removed_horizontal_tracks"]) == 3
    assert len(line_attempt["track_hypotheses"]) == 2


def test_line_grid_keeps_partial_separator_spanning_complete_columns() -> None:
    """验证跨完整列带的局部横线保留，同时删除单元格内短下划线。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 60.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 14.0, 18.0)),
                ("B", (50.0, 10.0, 54.0, 18.0)),
                ("C", (90.0, 10.0, 94.0, 18.0)),
                ("D", (50.0, 40.0, 54.0, 48.0)),
                ("E", (90.0, 40.0, 94.0, 48.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((40.0, 29.5, 120.0, 30.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 59.0, 120.0, 60.0), 1.0, "horizontal"),
            NativeTableRule((45.0, 22.5, 58.0, 23.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
            NativeTableRule((39.5, 0.0, 40.5, 60.0), 1.0, "vertical"),
            NativeTableRule((79.5, 0.0, 80.5, 60.0), 1.0, "vertical"),
            NativeTableRule((119.0, 0.0, 120.0, 60.0), 1.0, "vertical"),
        ),
    )

    result = recover_native_pdf_table(table_input)
    diagnostics = diagnose_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (2, 3)
    assert result.cells[0].rowspan == 2
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["track_hypothesis"] == "supported"
    assert len(line_attempt["removed_horizontal_tracks"]) == 1


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_single_column_line_grid_supports_standard_rotations(angle: int) -> None:
    """验证强线框单列表单在四种标准方向下保留文本和转义。"""

    result = recover_native_pdf_table(_rotated_single_column_input(angle))

    assert result is not None
    assert result.source == "vector_grid"
    assert (result.rows, result.cols) == (2, 1)
    assert result.html == ("<table><tbody><tr><td>A&amp;B</td></tr><tr><td>Value A B</td></tr></tbody></table>")
    assert "single_column_line_grid=true" in result.diagnostics


@pytest.mark.parametrize("missing", ["right", "middle"])
def test_single_column_line_grid_requires_complete_physical_frame(
    missing: str,
) -> None:
    """验证缺少侧边或行分隔的单列区域不能进入 HTML。"""

    rules = [
        NativeTableRule((0.0, 0.0, 100.0, 1.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 29.5, 100.0, 30.5), 1.0, "horizontal"),
        NativeTableRule((0.0, 59.0, 100.0, 60.0), 1.0, "horizontal"),
        NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
        NativeTableRule((99.0, 0.0, 100.0, 60.0), 1.0, "vertical"),
    ]
    if missing == "right":
        rules.pop()
    else:
        rules.pop(1)
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("Header", (10.0, 10.0, 34.0, 18.0)),
                ("Value", (10.0, 40.0, 30.0, 48.0)),
            ]
        ),
        drawing_lines=tuple(rules),
    )

    assert recover_native_pdf_table(table_input) is None


def test_line_grid_rejects_single_cell_frame() -> None:
    """验证完整外框中的单个文本块不能作为一行一列表格采用。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 40.0),
        page_size=(100.0, 40.0),
        angle=0,
        chars=_char_items([("Paragraph", (10.0, 10.0, 46.0, 18.0))]),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 100.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 39.0, 100.0, 40.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 0.0, 1.0, 40.0), 1.0, "vertical"),
            NativeTableRule((99.0, 0.0, 100.0, 40.0), 1.0, "vertical"),
        ),
    )

    assert recover_native_pdf_table(table_input) is None


def test_line_grid_collapses_duplicate_outer_y_track_in_multirow_table() -> None:
    """验证纵线端点与底部描边形成的重复外缘不会制造幽灵行。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 14.0, 18.0)),
                ("B", (60.0, 10.0, 64.0, 18.0)),
                ("C", (10.0, 40.0, 14.0, 48.0)),
                ("D", (60.0, 40.0, 64.0, 48.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 100.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 29.5, 100.0, 30.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 59.5, 100.0, 60.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 0.0, 1.0, 59.0), 1.0, "vertical"),
            NativeTableRule((49.5, 0.0, 50.5, 59.0), 1.0, "vertical"),
            NativeTableRule((99.0, 0.0, 100.0, 59.0), 1.0, "vertical"),
        ),
    )

    result = recover_native_pdf_table(table_input)
    diagnostics = diagnose_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (2, 2)
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["outer_track_collapses"]["y"] == 1


def test_line_grid_preserves_independent_nearby_outer_boundaries() -> None:
    """验证两条各自有强物理线的近邻外缘不会被重复轨规则折叠。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 14.0, 18.0)),
                ("B", (60.0, 10.0, 64.0, 18.0)),
                ("C", (10.0, 35.0, 14.0, 43.0)),
                ("D", (60.0, 35.0, 64.0, 43.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 100.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 29.5, 100.0, 30.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 53.5, 100.0, 54.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 59.0, 100.0, 60.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
            NativeTableRule((49.5, 0.0, 50.5, 54.0), 1.0, "vertical"),
            NativeTableRule((99.0, 0.0, 100.0, 60.0), 1.0, "vertical"),
        ),
    )

    result = recover_native_pdf_table(table_input)
    diagnostics = diagnose_native_pdf_table(table_input)
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["outer_track_collapses"]["y"] == 0
    assert result is not None
    assert result.source != "vector_grid"


def test_line_grid_does_not_fold_distinct_short_rule_into_outer_border() -> None:
    """验证外缘附近存在独立短物理线时保留原有 rowspan 拓扑。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 400.0, 60.0),
        page_size=(400.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 14.0, 18.0)),
                ("B", (120.0, 10.0, 124.0, 18.0)),
                ("C", (270.0, 10.0, 274.0, 18.0)),
                ("D", (10.0, 40.0, 14.0, 48.0)),
                ("E", (120.0, 40.0, 124.0, 48.0)),
                ("F", (270.0, 40.0, 274.0, 48.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 400.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 29.5, 400.0, 30.5), 1.0, "horizontal"),
            NativeTableRule((5.0, 53.5, 17.0, 54.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 59.0, 400.0, 60.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
            NativeTableRule((99.5, 0.0, 100.5, 60.0), 1.0, "vertical"),
            NativeTableRule((249.5, 0.0, 250.5, 60.0), 1.0, "vertical"),
            NativeTableRule((399.0, 0.0, 400.0, 60.0), 1.0, "vertical"),
        ),
    )

    result = recover_native_pdf_table(table_input)
    diagnostics = diagnose_native_pdf_table(table_input)

    assert result is not None
    assert result.source == "vector_grid"
    assert (result.rows, result.cols) == (3, 3)
    assert all(cell.rowspan == 2 for cell in result.cells if cell.row == 1)
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["outer_track_collapses"]["y"] == 0


def test_single_row_line_grid_recovers_clipped_outer_borders() -> None:
    """验证 drawing halo 可恢复 bbox 外邻近边框且保留物理空列。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 2.0, 120.0, 58.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("C", (90.0, 38.0, 95.0, 46.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.5, 120.0, 1.0), 0.5, "horizontal"),
            NativeTableRule((0.0, 59.0, 120.0, 59.5), 0.5, "horizontal"),
            NativeTableRule((0.0, 1.0, 0.5, 59.0), 0.5, "vertical"),
            NativeTableRule((39.75, 1.0, 40.25, 59.0), 0.5, "vertical"),
            NativeTableRule((79.75, 1.0, 80.25, 59.0), 0.5, "vertical"),
            NativeTableRule((119.5, 1.0, 120.0, 59.0), 0.5, "vertical"),
        ),
    )

    result = recover_native_pdf_table(table_input)
    diagnostics = diagnose_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (1, 3)
    assert result.html == ("<table><tbody><tr><td>A</td><td></td><td>C</td></tr></tbody></table>")
    assert "single_row_line_grid=true" in result.diagnostics
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["evidence_halo"] > 0
    assert line_attempt["single_row_evidence"]["verified"] is True


def test_single_row_line_grid_rejects_missing_outer_border() -> None:
    """验证单物理行缺少任一横向外框时继续主动回退。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 60.0),
        page_size=(120.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (50.0, 10.0, 55.0, 18.0)),
                ("C", (90.0, 10.0, 95.0, 18.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 0.0, 1.0, 60.0), 1.0, "vertical"),
            NativeTableRule((39.5, 0.0, 40.5, 60.0), 1.0, "vertical"),
            NativeTableRule((79.5, 0.0, 80.5, 60.0), 1.0, "vertical"),
            NativeTableRule((119.0, 0.0, 120.0, 60.0), 1.0, "vertical"),
        ),
    )

    assert recover_native_pdf_table(table_input) is None
    diagnostics = diagnose_native_pdf_table(table_input)
    assert diagnostics["first_rejection_gate"] == ("vector_single_row_physical_evidence")


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_single_row_line_grid_supports_standard_rotations(angle: int) -> None:
    """验证强线框单物理行在四种标准旋转下保持同一拓扑。"""

    result = recover_native_pdf_table(_rotated_single_row_input(angle))

    assert result is not None
    assert (result.rows, result.cols) == (1, 3)
    assert [cell.content for cell in result.cells] == ["A", "B", "C"]


def test_vector_cell_preserves_explicit_small_gap_space() -> None:
    """验证 PDF 字符流显式空格在几何间距很小时仍不会丢失。"""

    chars = (
        {"char": "A", "bbox": (10.0, 10.0, 14.0, 18.0), "char_idx": 0},
        {"char": " ", "bbox": (14.0, 10.0, 14.5, 18.0), "char_idx": 1},
        {"char": "B", "bbox": (14.5, 10.0, 18.5, 18.0), "char_idx": 2},
        {"char": "X", "bbox": (60.0, 10.0, 64.0, 18.0), "char_idx": 3},
        {"char": "C", "bbox": (10.0, 40.0, 14.0, 48.0), "char_idx": 4},
        {"char": "D", "bbox": (60.0, 40.0, 64.0, 48.0), "char_idx": 5},
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=chars,  # type: ignore[arg-type]
        drawing_lines=_grid_rules(),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert "<td>A B</td>" in result.html


def test_reportlab_pdf_primitives_recover_expected_html() -> None:
    """验证真实 ReportLab PDF 的字符与 drawing 可端到端恢复结构。"""

    with PDFDocument(_build_reportlab_table_pdf()) as document:
        page = document[0]
        result = recover_native_pdf_table(
            NativeTableInput(
                table_bbox=(20.0, 40.0, 180.0, 160.0),
                page_size=page.size,
                angle=0,
                chars=tuple(page.get_chars()),
                drawing_lines=coerce_native_table_rules(page.get_drawing_lines()),
                rectangles=coerce_native_table_rectangles(page.get_path_infos()),
            )
        )

    assert result is not None
    assert result.source == "vector_grid"
    assert result.html == ("<table><tbody><tr><td>A</td><td>B</td></tr><tr><td>C</td><td>D</td></tr></tbody></table>")


@pytest.mark.parametrize(
    ("vertical_range", "horizontal_range", "expected_attribute"),
    [
        ((30.0, 60.0), (0.0, 100.0), 'colspan="2"'),
        ((0.0, 60.0), (50.0, 100.0), 'rowspan="2"'),
    ],
)
def test_vector_grid_recovers_rectangular_spans(
    vertical_range: tuple[float, float],
    horizontal_range: tuple[float, float],
    expected_attribute: str,
) -> None:
    """验证内部隔断缺失只生成矩形横向或纵向合并格。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=_grid_rules(
            internal_vertical=vertical_range,
            internal_horizontal=horizontal_range,
        ),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert expected_attribute in result.html


def test_vector_grid_rejects_non_rectangular_merge_component() -> None:
    """验证横纵缺边形成 L 形连通域时拒绝矢量候选。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=_grid_rules(
            internal_vertical=(30.0, 60.0),
            internal_horizontal=(50.0, 100.0),
        ),
    )
    text = build_native_table_text(table_input)

    assert text is not None
    assert build_vector_candidates(table_input, text) == []


def test_vector_cell_concatenates_multiple_visual_lines() -> None:
    """验证同一逻辑单元格中的多行文本直接拼接且不写入 HTML 换行。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 6.0, 15.0, 12.0)),
                ("B", (10.0, 18.0, 15.0, 24.0)),
                ("X", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=_grid_rules(),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert "<td>AB</td>" in result.html
    assert "<br>" not in result.html


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_text_grid_supports_standard_table_rotations(angle: int) -> None:
    """验证三列无线表在四个标准方向下恢复相同结构。"""

    result = recover_native_pdf_table(
        _aligned_text_input(
            [
                ["A", "B", "C"],
                ["1", "2", "3"],
                ["4", "5", "6"],
            ],
            angle=angle,
        )
    )

    assert result is not None
    assert result.source == "text_grid"
    assert (result.rows, result.cols) == (3, 3)
    assert "<tr><td>4</td><td>5</td><td>6</td></tr>" in result.html


def test_text_grid_preserves_empty_cells() -> None:
    """验证缺少文本的叶子列保留空 td，且不会被推断为合并格。"""

    result = recover_native_pdf_table(
        _aligned_text_input(
            [
                ["A", "B", "C"],
                ["1", None, "3"],
                ["4", "5", "6"],
            ]
        )
    )

    assert result is not None
    assert "<tr><td>1</td><td></td><td>3</td></tr>" in result.html
    assert "rowspan" not in result.html
    assert "colspan" not in result.html


def test_text_grid_groups_tight_subset_continuation_without_break() -> None:
    """验证无线表紧邻且不引入新列的续行并入同一逻辑行且直接拼接。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 75.0),
        page_size=(120.0, 75.0),
        angle=0,
        chars=_char_items(
            [
                ("A1", (8.0, 5.0, 16.0, 13.0)),
                ("B1", (48.0, 5.0, 56.0, 13.0)),
                ("C1", (88.0, 5.0, 96.0, 13.0)),
                ("A2", (8.0, 14.0, 16.0, 22.0)),
                ("1", (8.0, 36.0, 12.0, 44.0)),
                ("2", (48.0, 36.0, 52.0, 44.0)),
                ("3", (88.0, 36.0, 92.0, 44.0)),
                ("4", (8.0, 60.0, 12.0, 68.0)),
                ("5", (48.0, 60.0, 52.0, 68.0)),
                ("6", (88.0, 60.0, 92.0, 68.0)),
            ]
        ),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (3, 3)
    assert "<tr><td>A1A2</td><td>B1</td><td>C1</td></tr>" in result.html


def test_text_grid_rejects_tight_equal_dense_baselines() -> None:
    """验证紧邻且占用列相同的稠密基线不被冒充续行合并。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 75.0),
        page_size=(120.0, 75.0),
        angle=0,
        chars=_char_items(
            [
                ("A1", (8.0, 5.0, 16.0, 13.0)),
                ("B1", (48.0, 5.0, 56.0, 13.0)),
                ("C1", (88.0, 5.0, 96.0, 13.0)),
                ("A2", (8.0, 14.0, 16.0, 22.0)),
                ("B2", (48.0, 14.0, 56.0, 22.0)),
                ("C2", (88.0, 14.0, 96.0, 22.0)),
                ("A3", (8.0, 50.0, 16.0, 58.0)),
                ("B3", (48.0, 50.0, 56.0, 58.0)),
                ("C3", (88.0, 50.0, 96.0, 58.0)),
            ]
        ),
    )

    diagnostics = diagnose_native_pdf_table(table_input)

    assert recover_native_pdf_table(table_input) is None
    text_attempt = next(attempt for attempt in diagnostics["text_attempts"] if attempt["source"] == "text_grid")
    assert text_attempt["first_rejection_gate"] == "dense_row_ambiguity"
    assert text_attempt["dense_row_ambiguities"] == [[0, 1]]


def test_text_grid_accepts_right_aligned_numeric_columns() -> None:
    """验证数字宽度变化但右缘稳定时仍能恢复同一列结构。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 110.0, 70.0),
        page_size=(110.0, 70.0),
        angle=0,
        chars=_char_items(
            [
                ("Name", (10.0, 5.0, 26.0, 13.0)),
                ("Count", (40.0, 5.0, 60.0, 13.0)),
                ("Rate", (84.0, 5.0, 100.0, 13.0)),
                ("A", (10.0, 28.0, 14.0, 36.0)),
                ("1", (56.0, 28.0, 60.0, 36.0)),
                ("10", (92.0, 28.0, 100.0, 36.0)),
                ("B", (10.0, 51.0, 14.0, 59.0)),
                ("1000", (44.0, 51.0, 60.0, 59.0)),
                ("2", (96.0, 51.0, 100.0, 59.0)),
            ]
        ),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert (result.rows, result.cols) == (3, 3)
    assert "<tr><td>B</td><td>1000</td><td>2</td></tr>" in result.html


def test_text_grid_recovers_conservative_multicolumn_header() -> None:
    """验证前导单项文本真实横跨叶子列时生成 colspan。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 70.0),
        page_size=(100.0, 70.0),
        angle=0,
        chars=_char_items(
            [
                ("ABCDEFGHIJKLMNOPQR", (5.0, 5.0, 95.0, 13.0)),
                ("A", (10.0, 30.0, 14.0, 38.0)),
                ("B", (45.0, 30.0, 49.0, 38.0)),
                ("C", (80.0, 30.0, 84.0, 38.0)),
                ("1", (10.0, 52.0, 14.0, 60.0)),
                ("2", (45.0, 52.0, 49.0, 60.0)),
                ("3", (80.0, 52.0, 84.0, 60.0)),
            ]
        ),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert '<td colspan="3">ABCDEFGHIJKLMNOPQR</td>' in result.html


def test_text_grid_rejects_header_that_requires_rowspan() -> None:
    """验证表头角落空缺且需要 rowspan 才能还原时不生成伪网格。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 90.0),
        page_size=(120.0, 90.0),
        angle=0,
        chars=_char_items(
            [
                ("Metrics", (52.0, 5.0, 104.0, 13.0)),
                ("Name", (8.0, 30.0, 24.0, 38.0)),
                ("Q1", (48.0, 30.0, 56.0, 38.0)),
                ("Q2", (88.0, 30.0, 96.0, 38.0)),
                ("A", (8.0, 60.0, 12.0, 68.0)),
                ("1", (48.0, 60.0, 52.0, 68.0)),
                ("2", (88.0, 60.0, 92.0, 68.0)),
            ]
        ),
    )

    diagnostics = diagnose_native_pdf_table(table_input)

    assert recover_native_pdf_table(table_input) is None
    text_attempt = next(attempt for attempt in diagnostics["text_attempts"] if attempt["source"] == "text_grid")
    assert text_attempt["first_rejection_gate"] == "header_requires_rowspan"
    assert text_attempt["header_representable"] is False


def test_key_value_candidate_handles_two_stable_columns() -> None:
    """验证两列字段值表走独立 key-value 候选。"""

    result = recover_native_pdf_table(
        _aligned_text_input(
            [
                ["Name", "Alice"],
                ["Age", "30"],
                ["City", "Paris"],
            ]
        )
    )

    assert result is not None
    assert result.source == "key_value"
    assert "<td>Name</td><td>Alice</td>" in result.html


def test_text_grid_abstains_for_prose_without_repeated_columns() -> None:
    """验证已知 bbox 内只有连续正文行时仍主动放弃结构化。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 70.0),
        page_size=(120.0, 70.0),
        angle=0,
        chars=_char_items(
            [
                ("This is a complete sentence.", (5.0, 5.0, 110.0, 13.0)),
                ("It continues as ordinary prose.", (5.0, 28.0, 112.0, 36.0)),
                ("No stable columns are present.", (5.0, 51.0, 108.0, 59.0)),
            ]
        ),
    )

    assert recover_native_pdf_table(table_input) is None


@pytest.mark.parametrize("evidence", ["booktabs", "stripes"])
def test_sparse_candidate_uses_rules_or_row_stripes(evidence: str) -> None:
    """验证三线规则和重复行底纹均可提升同拓扑少线候选。"""

    rules: tuple[NativeTableRule, ...] = ()
    rectangles: tuple[NativeTableRectangle, ...] = ()
    if evidence == "booktabs":
        rules = (
            NativeTableRule((0.0, 5.0, 120.0, 6.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 35.0, 120.0, 36.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 84.0, 120.0, 85.0), 1.0, "horizontal"),
        )
    else:
        rectangles = (
            NativeTableRectangle((0.0, 8.0, 120.0, 18.0), 5, True, False),
            NativeTableRectangle((0.0, 38.0, 120.0, 48.0), 5, True, False),
            NativeTableRectangle((0.0, 68.0, 120.0, 78.0), 5, True, False),
        )
    result = recover_native_pdf_table(
        _aligned_text_input(
            [
                ["A", "B", "C"],
                ["1", "2", "3"],
                ["4", "5", "6"],
            ],
            rules=rules,
            rectangles=rectangles,
        )
    )

    assert result is not None
    assert result.source == ("sparse_hybrid" if evidence == "booktabs" else "sparse_grid")


def test_candidate_conflict_with_close_scores_abstains() -> None:
    """验证近分异构候选无法建立明确优势时主动放弃。"""

    assert (
        _select_candidate(
            [
                _candidate(source="vector_grid", rows=2, cols=2, score=0.94),
                _candidate(source="text_grid", rows=3, cols=3, score=0.92),
            ]
        )
        is None
    )


def test_verified_line_grid_wins_conflicting_text_topology() -> None:
    """验证无歧义 drawing 网格以独立物理证据胜出异构文本候选。"""

    line = _candidate(
        source="vector_grid",
        rows=2,
        cols=4,
        score=1.0,
        issues=(
            "evidence=line_grid",
            "ambiguous_separator_ratio=0.0000",
        ),
    )
    text = _candidate(source="text_grid", rows=3, cols=4, score=1.0)

    assert _select_candidate([line, text]) is line


def test_verified_line_and_rect_topology_conflict_abstains() -> None:
    """验证两类独立物理证据拓扑不一致时不由 line-grid 抢占。"""

    line = _candidate(
        source="vector_grid",
        rows=2,
        cols=2,
        score=1.0,
        issues=(
            "evidence=line_grid",
            "ambiguous_separator_ratio=0.0000",
        ),
    )
    rect = _candidate(
        source="vector_grid",
        rows=3,
        cols=2,
        score=1.0,
        issues=(
            "evidence=rect_grid",
            "ambiguous_separator_ratio=0.0000",
        ),
    )

    assert _select_candidate([line, rect]) is None


def test_text_candidate_removes_significantly_undercounted_vector() -> None:
    """验证稳定文本候选显著多出行列时剔除欠分割矢量候选。"""

    vector = _candidate(source="vector_grid", rows=2, cols=2, score=0.96)
    text = _candidate(source="text_grid", rows=3, cols=3, score=0.93)

    assert _remove_undercounted_vector_candidates([vector, text]) == [text]


def test_verified_line_grid_is_not_removed_by_text_undercount() -> None:
    """验证无歧义强物理网格不会被多 baseline 文本候选误判欠分割。"""

    vector = _candidate(
        source="vector_grid",
        rows=2,
        cols=2,
        score=0.96,
        issues=(
            "evidence=line_grid",
            "ambiguous_separator_ratio=0.0000",
        ),
    )
    text = _candidate(source="text_grid", rows=4, cols=3, score=0.93)

    assert _remove_undercounted_vector_candidates([vector, text]) == [vector, text]


def test_primitive_limit_falls_back_before_candidate_generation() -> None:
    """验证单表原语超过保护上限时直接主动放弃。"""

    base = _aligned_text_input(
        [
            ["A", "B", "C"],
            ["1", "2", "3"],
            ["4", "5", "6"],
        ]
    )
    rules = tuple(NativeTableRule((0.0, 1.0, 120.0, 2.0), 1.0, "horizontal") for _ in range(MAX_PRIMITIVES_PER_TABLE + 1))
    table_input = NativeTableInput(
        table_bbox=base.table_bbox,
        page_size=base.page_size,
        angle=base.angle,
        chars=base.chars,
        drawing_lines=rules,
    )

    assert recover_native_pdf_table(table_input) is None
    diagnostics = diagnose_native_pdf_table(table_input)
    assert diagnostics["first_rejection_gate"] == "primitive_limit"
    assert diagnostics["generated_candidates"] == []


def test_diagnostics_records_tracks_components_and_adoption() -> None:
    """验证调试入口记录轨道、分项可靠度和最终采用状态。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 100.0, 60.0),
        page_size=(100.0, 60.0),
        angle=0,
        chars=_char_items(
            [
                ("A", (10.0, 10.0, 15.0, 18.0)),
                ("B", (60.0, 10.0, 65.0, 18.0)),
                ("C", (10.0, 40.0, 15.0, 48.0)),
                ("D", (60.0, 40.0, 65.0, 48.0)),
            ]
        ),
        drawing_lines=_grid_rules(),
    )

    diagnostics = diagnose_native_pdf_table(table_input)

    assert diagnostics["first_rejection_gate"] is None
    assert diagnostics["adopted"]["tracks"] == {"x": 3, "y": 3}
    assert diagnostics["adopted"]["score_components"]["text_capture"] == 1.0
    assert diagnostics["counterfactual_best"]["source"] == "vector_grid"
    line_attempt = next(attempt for attempt in diagnostics["vector_attempts"] if attempt["evidence"] == "line_grid")
    assert line_attempt["first_rejection_gate"] is None
    assert line_attempt["grid"] == {"rows": 2, "cols": 2}


def test_sparse_multiline_recovers_keyed_long_records() -> None:
    """验证连续关键行和右列续行会合并为稳定长文本记录。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 145.0),
        page_size=(120.0, 145.0),
        angle=0,
        chars=_char_items(
            [
                ("Name", (5.0, 5.0, 21.0, 13.0)),
                ("Detail", (45.0, 5.0, 69.0, 13.0)),
                ("Alpha", (5.0, 20.0, 25.0, 28.0)),
                ("Title A", (45.0, 20.0, 73.0, 28.0)),
                ("(A1)", (5.0, 30.0, 21.0, 38.0)),
                ("body a1", (45.0, 30.0, 73.0, 38.0)),
                ("body a2", (45.0, 40.0, 73.0, 48.0)),
                ("body a3", (45.0, 50.0, 73.0, 58.0)),
                ("Beta", (5.0, 65.0, 21.0, 73.0)),
                ("Title B", (45.0, 65.0, 73.0, 73.0)),
                ("(B1)", (5.0, 75.0, 21.0, 83.0)),
                ("body b1", (45.0, 75.0, 73.0, 83.0)),
                ("body b2", (45.0, 85.0, 73.0, 93.0)),
                ("body b3", (45.0, 95.0, 73.0, 103.0)),
                ("Gamma", (5.0, 110.0, 25.0, 118.0)),
                ("Title C", (45.0, 110.0, 73.0, 118.0)),
                ("(C1)", (5.0, 120.0, 21.0, 128.0)),
                ("body c1", (45.0, 120.0, 73.0, 128.0)),
                ("body c2", (45.0, 130.0, 73.0, 138.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 144.0, 120.0, 145.0), 1.0, "horizontal"),
        ),
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert result.source == "sparse_multiline"
    assert (result.rows, result.cols) == (4, 2)
    assert "<td>Alpha(A1)</td><td>Title Abody a1body a2body a3</td>" in result.html


def test_sparse_multiline_recovers_filled_record_continuations() -> None:
    """验证填充行带和关键列可恢复含续行的三列记录表。"""

    rectangles = tuple(
        NativeTableRectangle((left, top, right, bottom), 5, True, False)
        for top, bottom in ((20.0, 36.0), (62.0, 78.0))
        for left, right in ((0.0, 30.0), (30.0, 60.0), (60.0, 120.0))
    )
    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 100.0),
        page_size=(120.0, 100.0),
        angle=0,
        chars=_char_items(
            [
                ("Date", (5.0, 5.0, 21.0, 13.0)),
                ("From", (35.0, 5.0, 51.0, 13.0)),
                ("Title", (65.0, 5.0, 85.0, 13.0)),
                ("D1", (5.0, 22.0, 13.0, 30.0)),
                ("S1", (35.0, 22.0, 43.0, 30.0)),
                ("A", (65.0, 22.0, 69.0, 30.0)),
                ("A2", (65.0, 32.0, 73.0, 40.0)),
                ("D2", (5.0, 47.0, 13.0, 55.0)),
                ("S2", (35.0, 47.0, 43.0, 55.0)),
                ("B", (65.0, 47.0, 69.0, 55.0)),
                ("D3", (5.0, 67.0, 13.0, 75.0)),
                ("S3", (35.0, 67.0, 43.0, 75.0)),
                ("C", (65.0, 67.0, 69.0, 75.0)),
                ("C2", (65.0, 79.0, 73.0, 87.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 99.0, 120.0, 100.0), 1.0, "horizontal"),
        ),
        rectangles=rectangles,
    )

    result = recover_native_pdf_table(table_input)

    assert result is not None
    assert result.source == "sparse_multiline"
    assert (result.rows, result.cols) == (4, 3)
    assert "<td>D1</td><td>S1</td><td>AA2</td>" in result.html


def test_sparse_multiline_merges_one_level_multiline_header() -> None:
    """验证没有局部分隔线时多条表头基线只形成一层表头。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 120.0),
        page_size=(120.0, 120.0),
        angle=0,
        chars=_char_items(
            [
                ("Item", (5.0, 5.0, 21.0, 13.0)),
                ("Value", (45.0, 5.0, 65.0, 13.0)),
                ("Result", (85.0, 5.0, 109.0, 13.0)),
                ("unit", (45.0, 16.0, 61.0, 24.0)),
                ("flag", (85.0, 16.0, 101.0, 24.0)),
                ("kg", (45.0, 27.0, 53.0, 35.0)),
                ("ok", (85.0, 27.0, 93.0, 35.0)),
                ("A", (5.0, 50.0, 9.0, 58.0)),
                ("1", (45.0, 50.0, 49.0, 58.0)),
                ("Y", (85.0, 50.0, 89.0, 58.0)),
                ("B", (5.0, 75.0, 9.0, 83.0)),
                ("2", (45.0, 75.0, 49.0, 83.0)),
                ("N", (85.0, 75.0, 89.0, 83.0)),
                ("C", (5.0, 100.0, 9.0, 108.0)),
                ("3", (45.0, 100.0, 49.0, 108.0)),
                ("Y", (85.0, 100.0, 89.0, 108.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 39.5, 120.0, 40.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 119.0, 120.0, 120.0), 1.0, "horizontal"),
        ),
    )

    diagnostics = diagnose_native_pdf_table(table_input)

    multiline_attempt = diagnostics["sparse_multiline_attempts"][0]
    assert multiline_attempt["first_rejection_gate"] is None
    assert multiline_attempt["grid"] == {"rows": 4, "cols": 3}
    assert multiline_attempt["header_rows"] == 1


def test_sparse_multiline_rejects_ambiguous_body_rowspan() -> None:
    """验证无线正文首列空缺后再次出现时不猜测正文 rowspan。"""

    table_input = NativeTableInput(
        table_bbox=(0.0, 0.0, 120.0, 100.0),
        page_size=(120.0, 100.0),
        angle=0,
        chars=_char_items(
            [
                ("Model", (5.0, 5.0, 25.0, 13.0)),
                ("Param", (45.0, 5.0, 65.0, 13.0)),
                ("Value", (85.0, 5.0, 105.0, 13.0)),
                ("A", (5.0, 35.0, 9.0, 43.0)),
                ("p1", (45.0, 35.0, 53.0, 43.0)),
                ("1", (85.0, 35.0, 89.0, 43.0)),
                ("p2", (45.0, 55.0, 53.0, 63.0)),
                ("2", (85.0, 55.0, 89.0, 63.0)),
                ("B", (5.0, 75.0, 9.0, 83.0)),
                ("p3", (45.0, 75.0, 53.0, 83.0)),
                ("3", (85.0, 75.0, 89.0, 83.0)),
            ]
        ),
        drawing_lines=(
            NativeTableRule((0.0, 0.0, 120.0, 1.0), 1.0, "horizontal"),
            NativeTableRule((0.0, 19.5, 120.0, 20.5), 1.0, "horizontal"),
            NativeTableRule((0.0, 99.0, 120.0, 100.0), 1.0, "horizontal"),
        ),
    )

    diagnostics = diagnose_native_pdf_table(table_input)

    multiline_attempt = diagnostics["sparse_multiline_attempts"][0]
    assert multiline_attempt["first_rejection_gate"] == ("ambiguous_body_rowspan")
