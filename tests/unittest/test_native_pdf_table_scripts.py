# Copyright (c) Opendatalab. All rights reserved.
"""验证原生 PDF 表格上下标的几何识别与安全 HTML 序列化。"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from bs4 import BeautifulSoup
from pdftext.schema import Bbox, Char
import pytest

from mineru.model.flash.pdf.geometry import _rotate_bbox_from_upright
from mineru.model.flash.pdf.document import PDFDocument
from mineru.model.flash.pdf.table_recovery import (
    NativeTableInput,
    coerce_native_table_rectangles,
    coerce_native_table_rules,
    recover_native_pdf_table,
)
from mineru.model.flash.pdf.table_recovery.candidate import serialize_native_table_html
from mineru.model.flash.pdf.table_recovery.contracts import (
    NativeTableCell,
    NativeTableGlyph,
    NativeTableResult,
    NativeTableRule,
    NativeTableText,
    NativeTableTextRow,
)
from mineru.model.flash.pdf.table_text_styles import (
    _non_grid_fraction_rules,
    render_native_table_html_with_scripts,
)


_PROJECT_ROOT = Path(__file__).parents[2]
_SCRIPT_TRUTH_PATH = _PROJECT_ROOT / "tests" / "fixtures" / "native_pdf_table_script_truth.json"


def _char(
    text: str,
    index: int,
    bbox: tuple[float, float, float, float],
) -> Char:
    """构造带稳定来源索引的合成 PDF 字符。"""

    return {
        "char": text,
        "char_idx": index,
        "bbox": Bbox(list(bbox)),
        "rotation": 0.0,
        "font": {},
    }


def _origin_from_upright(
    origin: tuple[float, float],
    page_size: tuple[float, float],
    angle: int,
) -> tuple[float, float]:
    """把正向 origin 逆变换到合成页面坐标。"""

    x, y = origin
    page_width, page_height = page_size
    if angle == 270:
        return y, page_height - x
    if angle == 90:
        return page_width - y, x
    if angle == 180:
        return page_width - x, page_height - y
    return origin


def _result(
    content: str,
    glyphs: tuple[NativeTableGlyph, ...],
    *,
    cells: tuple[NativeTableCell, ...] | None = None,
) -> NativeTableResult:
    """构造单格高置信恢复结果，并保持默认 HTML 与纯文本一致。"""

    resolved_cells = cells or (
        NativeTableCell(
            row=0,
            col=0,
            rowspan=1,
            colspan=1,
            bbox=(0.0, 0.0, 60.0, 60.0),
            content=content,
            source_char_indices=tuple(glyph.source_index for glyph in glyphs),
        ),
    )
    text = NativeTableText(
        glyphs=glyphs,
        rows=tuple(
            NativeTableTextRow(
                row_index=row_index,
                bbox=(
                    min(glyph.bbox[0] for glyph in row_glyphs),
                    min(glyph.bbox[1] for glyph in row_glyphs),
                    max(glyph.bbox[2] for glyph in row_glyphs),
                    max(glyph.bbox[3] for glyph in row_glyphs),
                ),
                tokens=(),
                glyph_ids=tuple(glyph.glyph_id for glyph in row_glyphs),
            )
            for row_index in sorted({glyph.visual_row for glyph in glyphs})
            if (row_glyphs := [glyph for glyph in glyphs if glyph.visual_row == row_index])
        ),
        median_glyph_width=8.0,
        median_glyph_height=10.0,
    )
    return NativeTableResult(
        html=serialize_native_table_html(1, resolved_cells),
        rows=1,
        cols=max(1, len(resolved_cells)),
        cells=resolved_cells,
        text=text,
        source="vector_grid",
        confidence=1.0,
    )


def test_table_script_serialization_escapes_text_and_marks_superscript() -> None:
    """验证原始尖括号保持转义，只有可信数字被包装为上标。"""

    chars = (
        _char("<", 0, (5.0, 40.0, 9.0, 50.0)),
        _char("A", 1, (10.0, 40.0, 20.0, 50.0)),
        _char("2", 2, (20.0, 34.0, 26.0, 40.0)),
    )
    glyphs = tuple(
        NativeTableGlyph(index, index, char["char"], tuple(char["bbox"]), 0)  # type: ignore[arg-type]
        for index, char in enumerate(chars)
    )
    result = _result("<A2", glyphs)
    table_input = NativeTableInput((0.0, 0.0, 60.0, 60.0), (60.0, 60.0), 0, chars)

    styled = render_native_table_html_with_scripts(
        result,
        table_input,
        {0: (5.0, 40.0, 9.0, 50.0), 1: (10.0, 40.0, 20.0, 50.0), 2: (20.0, 34.0, 26.0, 40.0)},
        {0: (5.0, 49.0), 1: (10.0, 49.0), 2: (20.0, 40.0)},
    )

    assert "&lt;A<sup>2</sup>" in styled
    assert BeautifulSoup(styled, "html.parser").get_text() == "<A2"


@pytest.mark.parametrize("angle", [0, 90, 180, 270])
def test_table_script_geometry_uses_upright_coordinates(angle: int) -> None:
    """验证四种标准方向下 cell loose/tight/origin 同步正向化。"""

    page_size = (100.0, 120.0)
    local_bboxes = ((10.0, 40.0, 20.0, 50.0), (20.0, 34.0, 26.0, 40.0))
    local_origins = ((10.0, 49.0), (20.0, 40.0))
    page_bboxes = tuple(_rotate_bbox_from_upright(bbox, page_size, angle) for bbox in local_bboxes)
    page_origins = tuple(_origin_from_upright(origin, page_size, angle) for origin in local_origins)
    chars = tuple(_char(text, index, page_bboxes[index]) for index, text in enumerate("A2"))
    glyphs = (
        NativeTableGlyph(0, 0, "A", local_bboxes[0], 0),
        NativeTableGlyph(1, 1, "2", local_bboxes[1], 0),
    )
    result = _result("A2", glyphs)
    table_input = NativeTableInput((0.0, 0.0, *page_size), page_size, angle, chars)

    styled = render_native_table_html_with_scripts(
        result,
        table_input,
        dict(enumerate(page_bboxes)),
        dict(enumerate(page_origins)),
    )

    assert BeautifulSoup(styled, "html.parser").find("sup").get_text() == "2"  # type: ignore[union-attr]


def test_table_fraction_region_stays_plain() -> None:
    """验证 cell 内短分数线会拒识其上下叠排字符。"""

    chars = (
        _char("x", 0, (10.0, 40.0, 20.0, 50.0)),
        _char("1", 1, (22.0, 32.0, 30.0, 40.0)),
        _char("M", 2, (22.0, 44.0, 30.0, 52.0)),
    )
    glyphs = (
        NativeTableGlyph(0, 0, "x", (10.0, 40.0, 20.0, 50.0), 0),
        NativeTableGlyph(1, 1, "1", (22.0, 32.0, 30.0, 40.0), 0),
        NativeTableGlyph(2, 2, "M", (22.0, 44.0, 30.0, 52.0), 1),
    )
    result = _result("x1M", glyphs)
    table_input = NativeTableInput(
        (0.0, 0.0, 60.0, 60.0),
        (60.0, 60.0),
        0,
        chars,
        drawing_lines=(NativeTableRule((21.0, 41.0, 31.0, 42.0), 1.0, "horizontal"),),
    )

    styled = render_native_table_html_with_scripts(
        result,
        table_input,
        {0: (10.0, 40.0, 20.0, 50.0), 1: (22.0, 32.0, 30.0, 40.0), 2: (22.0, 44.0, 30.0, 52.0)},
        {0: (10.0, 49.0), 1: (22.0, 40.0), 2: (22.0, 52.0)},
    )

    assert "<sup>" not in styled and "<sub>" not in styled
    assert BeautifulSoup(styled, "html.parser").get_text() == "x1M"


def test_table_grid_boundary_is_not_a_fraction_rule() -> None:
    """验证与任一逻辑 cell 边界重合的横线不会进入分式检测。"""

    glyphs = (NativeTableGlyph(0, 0, "A", (5.0, 5.0, 15.0, 15.0), 0),)
    cells = (
        NativeTableCell(0, 0, 1, 1, (0.0, 0.0, 30.0, 30.0), "A", (0,)),
        NativeTableCell(0, 1, 1, 1, (30.0, 0.0, 60.0, 30.0), "", ()),
    )
    result = _result("A", glyphs, cells=cells)
    grid_rule = NativeTableRule((0.0, 29.5, 60.0, 30.5), 1.0, "horizontal")
    fraction_rule = NativeTableRule((5.0, 15.0, 15.0, 16.0), 1.0, "horizontal")
    table_input = NativeTableInput(
        (0.0, 0.0, 60.0, 60.0),
        (60.0, 60.0),
        0,
        (_char("A", 0, (5.0, 5.0, 15.0, 15.0)),),
        drawing_lines=(grid_rule, fraction_rule),
    )

    assert _non_grid_fraction_rules(table_input, result) == [fraction_rule]


def test_table_missing_extended_geometry_keeps_original_html() -> None:
    """验证缺少 tight/origin 时不根据 loose bbox 猜测上下标。"""

    chars = (_char("A", 0, (10.0, 40.0, 20.0, 50.0)),)
    glyphs = (NativeTableGlyph(0, 0, "A", (10.0, 40.0, 20.0, 50.0), 0),)
    result = _result("A", glyphs)
    table_input = NativeTableInput((0.0, 0.0, 60.0, 60.0), (60.0, 60.0), 0, chars)

    assert render_native_table_html_with_scripts(result, table_input, {}, {}) == result.html


def _manifest_table_input(
    document: PDFDocument,
    entry: dict[str, object],
    *,
    cross_page_manifest: bool,
) -> tuple[NativeTableInput, dict[int, tuple[float, float, float, float]], dict[int, tuple[float, float]]]:
    """按 manifest bbox 构造表格输入，并返回同页扩展字符几何。"""

    page_index = int(entry["page_index"])  # type: ignore[arg-type]
    page_size = document.page_size(page_index)
    geometry = document.get_page_chars_with_geometry(page_index)
    raw_bbox = entry["bbox"]  # type: ignore[assignment]
    if cross_page_manifest:
        production_size = int(page_size[0]), int(page_size[1])
        bbox = tuple(
            float(raw_bbox[index]) * (production_size[0] if index % 2 == 0 else production_size[1])  # type: ignore[index]
            for index in range(4)
        )
    else:
        bbox = tuple(
            float(raw_bbox[index]) * (page_size[0] if index % 2 == 0 else page_size[1])  # type: ignore[index]
            for index in range(4)
        )
    return (
        NativeTableInput(
            table_bbox=bbox,  # type: ignore[arg-type]
            page_size=page_size,
            angle=int(entry.get("angle", 0) or 0),
            chars=tuple(geometry.chars),
            drawing_lines=coerce_native_table_rules(document.get_page_drawing_lines(page_index)),
            rectangles=coerce_native_table_rectangles(document.get_page_path_infos(page_index)),
        ),
        geometry.tight_bboxes,
        geometry.origins,
    )


def _real_manifest_script_runs() -> tuple[set[tuple[object, ...]], set[str]]:
    """运行两套仓库表格语料，并收集最终 cell 上下标及带样式 cell。"""

    fields = ("file", "page_index", "table_index", "row", "col", "text", "style")
    actual: set[tuple[object, ...]] = set()
    styled_cell_texts: set[str] = set()
    manifest_specs = (
        ("tests/fixtures/native_pdf_table_demo_manifest.json", "demo/pdfs", "tables", False),
        (
            "tests/fixtures/native_pdf_table_cross_page_manifest.json",
            "tests/unittest/pdfs/native_pdf_tables",
            "entries",
            True,
        ),
    )
    for manifest_path, source_root, key, cross_page_manifest in manifest_specs:
        manifest = json.loads((_PROJECT_ROOT / manifest_path).read_text(encoding="utf-8"))
        by_file: dict[str, list[dict[str, object]]] = defaultdict(list)
        for entry in manifest[key]:
            if entry.get("expected_output") == "html":
                by_file[str(entry["file"])].append(entry)
        for filename, entries in by_file.items():
            with PDFDocument(str(_PROJECT_ROOT / source_root / filename)) as document:
                for entry in entries:
                    table_input, tight_bboxes, origins = _manifest_table_input(
                        document,
                        entry,
                        cross_page_manifest=cross_page_manifest,
                    )
                    result = recover_native_pdf_table(table_input)
                    assert result is not None
                    styled = render_native_table_html_with_scripts(result, table_input, tight_bboxes, origins)
                    html_cells = BeautifulSoup(styled, "html.parser").find_all("td")
                    assert len(html_cells) == len(result.cells)
                    for source_cell, html_cell in zip(result.cells, html_cells, strict=True):
                        tags = html_cell.find_all(["sup", "sub"])
                        if tags:
                            styled_cell_texts.add(source_cell.content)
                        for tag in tags:
                            record = {
                                "file": filename,
                                "page_index": int(entry["page_index"]),
                                "table_index": int(entry["table_index"]),
                                "row": source_cell.row,
                                "col": source_cell.col,
                                "text": tag.get_text(),
                                "style": "superscript" if tag.name == "sup" else "subscript",
                            }
                            actual.add(tuple(record[field] for field in fields))
    return actual, styled_cell_texts


def test_real_native_table_scripts_are_precision_gated() -> None:
    """验证全部实际输出属于人工真值，必召回项存在且强制负例保持纯文本。"""

    truth = json.loads(_SCRIPT_TRUTH_PATH.read_text(encoding="utf-8"))
    fields = tuple(truth["matching"])
    expected = {tuple(record[field] for field in fields) for record in truth["runs"]}
    required = {tuple(record[field] for field in fields) for record in truth["required_runs"]}
    actual, styled_cell_texts = _real_manifest_script_runs()

    assert len(expected) == 77
    assert actual <= expected
    assert required <= actual
    assert not styled_cell_texts.intersection(truth["forbidden_plain_cells"])
    assert all(
        not any(
            actual_record[0] == forbidden["file"]
            and actual_record[1] == forbidden["page_index"]
            and actual_record[2] == forbidden["table_index"]
            and actual_record[5] == forbidden["text"]
            and actual_record[6] == forbidden["style"]
            for actual_record in actual
        )
        for forbidden in truth["forbidden_runs"]
    )
