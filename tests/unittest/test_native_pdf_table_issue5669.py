"""通过公开 DocVortex 接口验证少线层级表的真实区域和模型回退边界。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock

import numpy as np
import pytest
from bs4 import BeautifulSoup
from docvortex.document.pdf import PDFPageTextGeometry, PDFPageVectorGeometry
from PIL import Image

from mineru.backend.analysis.pdf import tables


@pytest.mark.parametrize("effort", ["medium", "high"])
@pytest.mark.parametrize("outcome", ["accepted", "rejected", "error"])
def test_real_hierarchical_table_acceptance_and_model_fallback(
    monkeypatch: pytest.MonkeyPatch,
    effort: Literal["medium", "high"],
    outcome: str,
) -> None:
    """真实字符证据通过时跳过模型；证据不足或恢复异常时保留全部成员交给模型。"""
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/native_pdf_table_issue5669.json").read_text())
    assert fixture["source_sha256"] == "dfc5383cf68f63b98d155cad55fa8007bcf80c86b007a975a6f378b8091366e1"
    width, height = fixture["page_size"]
    chars = fixture["chars"]
    if outcome == "rejected":
        # 去掉一项子表值，让重复子列仍可见但完整结构无法验证；不能重新接受两列粘连结果。
        chars = [char for char in chars if not (char["bbox"][0] > 370 and 600 < char["bbox"][1] < 610)]
    page = MagicMock()
    page.size = (width, height)
    page.get_chars_with_geometry.return_value = PDFPageTextGeometry(chars=chars, tight_bboxes={}, origins={})
    page.get_vector_geometry.return_value = PDFPageVectorGeometry(
        drawing_lines=fixture["drawing_lines"], path_infos=fixture["rectangles"]
    )
    if outcome == "error":
        monkeypatch.setattr(tables, "recover_table_region", MagicMock(side_effect=tables.PDFTableRecoveryError("diagnostic")))
    bbox = [v / (width if i % 2 == 0 else height) for i, v in enumerate(fixture["regions"]["hybrid"])]
    table = {"type": "table", "bbox": bbox, "angle": 0}
    text = {"type": "text", "bbox": [0.1, 0.66, 0.3, 0.67], "content": "测试按钮可用"}
    formula = {"type": "equation", "bbox": [0.35, 0.66, 0.4, 0.67]}
    formula_layout = {"label": "inline_formula", "bbox": [210, 556, 240, 564]}
    blocks, layout = [[table, text, formula]], [[formula_layout]]
    with Image.new("RGB", (600, 842), "white") as image:
        summary = tables._apply_native_txt_table_priority(
            blocks, layout, [page], [{"img_pil": image, "scale": 1.0}], effort=effort
        )
    if outcome == "accepted":
        assert summary.accepted == 1 and summary.rejected == summary.errors == 0
        assert blocks == [[table]] and layout == [[]]
        soup = BeautifulSoup(table["content"], "html.parser")
        rows = soup.find_all("tr")
        assert len(rows) == 12
        assert [(cell.get_text(), cell.get("colspan")) for cell in rows[0].find_all("td")] == [
            ("测试按钮可用", None),
            ("可配置", "2"),
        ]
        assert soup.find("td", rowspan="8").get_text() == "UL94阻燃等级组件"
        assert [rows[i].find_all("td")[-1].get_text() for i in (5, 7, 9, 11)] == ["HB", "HB", "V-0", "V-2"]
    else:
        assert summary.accepted == 0
        assert (summary.rejected, summary.errors) == ((1, 0) if outcome == "rejected" else (0, 1))
        assert blocks == [[table, text, formula]] and layout == [[formula_layout]]
        assert "content" not in table
    if effort == "high":
        vlm, native = tables._split_native_high_table_blocks(blocks)
        if outcome == "accepted":
            assert vlm == [[]] and native == [[table]]
        else:
            assert vlm == blocks and native == [[]]
    else:
        tasks = tables._collect_medium_table_tasks(blocks, layout, [np.zeros((842, 600, 3), dtype=np.uint8)])
        assert len(tasks) == (0 if outcome == "accepted" else 1)
        if tasks:
            assert tasks[0]["table_block"] is table
