"""验证表格方向使用页面视觉坐标系，并覆盖原生恢复和模型回退。"""

from __future__ import annotations

import math
from copy import deepcopy
from io import BytesIO
from typing import Any, Literal
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
from bs4 import BeautifulSoup
from docvortex.document.pdf import PDFDocument, PDFPage
from PIL import Image
from reportlab.pdfgen.canvas import Canvas

from mineru.backend.analysis.pdf import snapshots, tables


def _lines(angles: list[float]) -> list[dict[str, Any]]:
    """构造位于同一表格内的非空文本行，角度输入使用度数。"""
    return [
        {"bbox": (10, 10 + index * 15, 90, 20 + index * 15), "rotation": math.radians(angle), "spans": [{"text": "Cell"}]}
        for index, angle in enumerate(angles)
    ]


@pytest.mark.parametrize(
    ("raw_angle", "page_rotation", "expected"),
    [
        (269.91, 90, 0),
        (270.09, 90, 0),
        (-0.09, 0, 0),
        (360.09, 0, 0),
        (89.91, 0, 90),
        (90.09, 0, 90),
        (269.89, 90, None),
        (270.11, 90, None),
        (45, 0, None),
        (float("nan"), 90, None),
        (float("inf"), 90, None),
        (-float("inf"), 90, None),
    ],
)
def test_orientation_uses_circular_tolerance(raw_angle: float, page_rotation: int, expected: int | None) -> None:
    """跨越零度的误差仍按原有容差判定，非标准方向和非有限数值不参与投票。"""
    assert (
        tables._detect_table_angle_from_pdf_lines(_lines([raw_angle] * 3), (0, 0, 100, 100), page_rotation=page_rotation)
        == expected
    )


@pytest.mark.parametrize(
    ("angles", "expected"),
    [
        ([270, 270], None),
        ([270, 270, 0, 0], None),
        ([270, 270, 0, 90, 180], None),
        ([270, 270, 270, 0, 0], 0),
    ],
)
def test_orientation_preserves_vote_thresholds_and_input(angles: list[float], expected: int | None) -> None:
    """保留最少三行、唯一众数和百分之六十门槛，并禁止改写共享行缓存。"""
    lines = _lines(angles)
    original = deepcopy(lines)
    assert tables._detect_table_angle_from_pdf_lines(lines, (0, 0, 100, 100), page_rotation=90) == expected
    assert lines == original


def test_orientation_ignores_empty_outside_and_malformed_lines() -> None:
    """空白、框外或无效角度行不能补足原生方向证据。"""
    lines = _lines([270] * 5)
    lines[2]["spans"] = [{"text": " "}]
    lines[3]["bbox"] = (110, 10, 190, 20)
    lines[4]["rotation"] = "invalid"
    assert tables._detect_table_angle_from_pdf_lines(lines, (0, 0, 100, 100), page_rotation=90) is None


def _rotated_page_pdf(page_rotation: int, visual_angle: int) -> bytes:
    """生成真实页面旋转及独立表格旋转的四行两列表，避免转置后行列数恰好相等。"""
    output = BytesIO()
    pdf = Canvas(output, pagesize=(400, 400))
    pdf.setPageRotation(page_rotation)
    pdf.translate(200, 200)
    pdf.rotate(page_rotation - visual_angle)
    pdf.translate(-200, -200)
    for x in (40, 150, 360):
        pdf.line(x, 40, x, 320)
    for y in (40, 110, 180, 250, 320):
        pdf.line(40, y, 360, y)
    for y, (left, right) in zip((290, 220, 150, 80), (("H", "V"), ("A", "1"), ("B", "2"), ("C", "3")), strict=True):
        pdf.drawString(50, y, left)
        pdf.drawString(160, y, right)
    pdf.save()
    return output.getvalue()


def _table_layout(page: PDFPage) -> dict[str, Any]:
    """从合成表的真实线框取得视觉区域，不用待测的方向换算推导测试真值。"""
    boxes = [line.bbox for line in page.get_drawing_lines()]
    return {
        "label": "table",
        "bbox": [
            min(box[0] for box in boxes),
            min(box[1] for box in boxes),
            max(box[2] for box in boxes),
            max(box[3] for box in boxes),
        ],
    }


@pytest.mark.parametrize("effort", ["medium", "high"])
@pytest.mark.parametrize("channel", ["snapshot", "geometry", "plain"])
@pytest.mark.parametrize("page_rotation", [0, 90, 180, 270])
@pytest.mark.parametrize("visual_angle", [0, 90, 180, 270])
def test_real_page_and_table_rotations_preserve_structure(
    effort: Literal["medium", "high"], channel: str, page_rotation: int, visual_angle: int
) -> None:
    """三种字符路径和两个档位都按真实视觉方向恢复表格，保留横置及倒置表格能力。"""
    with PDFDocument(_rotated_page_pdf(page_rotation, visual_angle)) as document, Image.new("RGB", (400, 400)) as image:
        page = document[0]
        assert page.rotation == page_rotation
        layout = _table_layout(page)
        geometry = page.get_chars_with_geometry() if channel == "geometry" else None
        original_chars = (
            deepcopy([{**char, "bbox": list(char["bbox"])} for char in geometry.chars]) if geometry is not None else None
        )
        geometries = [geometry] if channel != "plain" else None
        cache = snapshots.create_page_snapshot_cache([page], enabled=channel == "snapshot")
        if channel == "snapshot" and page.get_text_snapshot() is None:
            pytest.skip("owned text snapshot unavailable in reference backend")
        images = [{"img_pil": image, "scale": 1.0}]
        model = MagicMock()
        try:
            tables._apply_table_orientations(
                [{"page_idx": 0, "layout_item": layout}],
                "txt",
                [page],
                images,
                model,
                geometries,
                page_snapshots=cache,
            )
            assert layout["angle"] == visual_angle
            model.table_orientation_cls_model.batch_predict.assert_not_called()
            if cache is not None:
                assert cache[0] is not None and cache[0].full_snapshot is None
                assert geometries == [None]

            table = {"type": "table", "bbox": [value / 400 for value in layout["bbox"]], "angle": layout["angle"]}
            summary = tables._apply_native_txt_table_priority(
                [[table]], [[layout]], [page], images, effort=effort, page_text_geometries=geometries, page_snapshots=cache
            )
            assert summary == tables._NativeTablePrioritySummary(total=1, accepted=1)
            soup = BeautifulSoup(table["content"], "html.parser")
            assert [[cell.get_text() for cell in row.find_all("td")] for row in soup.find_all("tr")] == [
                ["H", "V"],
                ["A", "1"],
                ["B", "2"],
                ["C", "3"],
            ]
            if geometry is not None:
                assert [{**char, "bbox": list(char["bbox"])} for char in geometry.chars] == original_chars
            if effort == "medium":
                assert tables._collect_medium_table_tasks([[table]], [[]], [np.asarray(image)]) == []
            else:
                assert tables._split_native_high_table_blocks([[table]]) == ([[]], [[table]])
        finally:
            snapshots.clear_page_snapshot_cache(cache)


@pytest.mark.parametrize("effort", ["medium", "high"])
@pytest.mark.parametrize("visual_angle", [0, 90])
def test_native_rejection_keeps_visual_angle_for_model(
    effort: Literal["medium", "high"], visual_angle: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """原生恢复拒绝后，Basic 裁图和 Standard 模型输入仍使用校正后的视觉方向。"""
    recover = MagicMock(return_value=None)
    monkeypatch.setattr(tables, "recover_table_region", recover)
    with PDFDocument(_rotated_page_pdf(90, visual_angle)) as document, Image.new("RGB", (400, 400)) as image:
        page = document[0]
        layout = _table_layout(page)
        images = [{"img_pil": image, "scale": 1.0}]
        geometries = [None]
        assert (
            tables._resolve_txt_table_orientations([{"page_idx": 0, "layout_item": layout}], [page], images, geometries) == []
        )
        assert layout["angle"] == visual_angle
        table = {"type": "table", "bbox": [value / 400 for value in layout["bbox"]], "angle": layout["angle"]}
        summary = tables._apply_native_txt_table_priority(
            [[table]], [[layout]], [page], images, effort=effort, page_text_geometries=geometries
        )
        assert summary == tables._NativeTablePrioritySummary(total=1, rejected=1)
        assert recover.call_args.kwargs["angle"] == visual_angle
        assert "content" not in table
        if effort == "medium":
            tasks = tables._collect_medium_table_tasks([[table]], [[]], [np.asarray(image)])
            assert len(tasks) == 1 and tasks[0]["angle"] == visual_angle
            assert tasks[0]["table_img"].shape[1] > tasks[0]["table_img"].shape[0]
        else:
            assert tables._split_native_high_table_blocks([[table]]) == ([[table]], [[]])


def test_page_rotation_failure_uses_visual_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """页面旋转读取失败时走现有视觉模型，不能假定零度或继续使用原始文本角度。"""
    with PDFDocument(_rotated_page_pdf(90, 0)) as document, Image.new("RGB", (400, 400)) as image:
        page = document[0]
        layout = _table_layout(page)
        getter = MagicMock(side_effect=AssertionError("unexpected text extraction"))
        monkeypatch.setattr(page, "get_chars", getter)
        monkeypatch.setattr(page, "get_chars_with_geometry", getter)
        monkeypatch.setattr(PDFPage, "rotation", PropertyMock(side_effect=ValueError("rotation unavailable")))
        model = MagicMock()
        model.table_orientation_cls_model.batch_predict.return_value = ["0"]
        item = {"page_idx": 0, "layout_item": layout}
        tables._apply_table_orientations([item], "txt", [page], [{"img_pil": image, "scale": 1.0}], model, [None])
        assert layout["angle"] == 0
        model.table_orientation_cls_model.batch_predict.assert_called_once()
        assert model.table_orientation_cls_model.batch_predict.call_args.args[0] == [item]
        getter.assert_not_called()
