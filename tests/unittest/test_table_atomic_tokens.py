"""验证表内图片与公式在跨格、旋转和导出过程中完整且唯一归属。"""

from __future__ import annotations

import base64
import html
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from bs4 import BeautifulSoup
from docvortex.assets import parse_image_data_uri_strict
from PIL import Image

from mineru.backend.analysis.pdf import tables
from mineru.model.runtime.contracts import AtomicModelName
from mineru.model.table.rec.slanet_plus.main import PaddleTable
from mineru.model.table.rec.unet_table import main as unet
from mineru.model.table.rec.unet_table.utils_table_recover import match_ocr_cell
from mineru.parser.base import ParseResult
from mineru.parser.task_outputs import build_task_output
from mineru.parser.writer import FileBasedDataWriter
from mineru.types import BlockType, MiddleJson

Rect = tuple[float, float, float, float]
TWO_CELLS = [(0, 0, 100, 50), (100, 0, 200, 50)]
FOUR_CELLS = [*TWO_CELLS, (0, 50, 100, 100), (100, 50, 200, 100)]
LATEX = r"\frac{1}{2} \times 36 = 18"


def _quad(rect: Rect) -> list[list[float]]:
    """把测试矩形转换为现有匹配器使用的顺时针四点框。"""
    x0, y0, x1, y1 = rect
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


@pytest.fixture(params=["image", "formula"])
def atomic_content(request: pytest.FixtureRequest) -> str:
    """提供真实 JPEG token 与包含反斜杠命令的公式 token。"""
    if request.param == "formula":
        return f"<eq>{LATEX}</eq>"
    buffer = io.BytesIO()
    with Image.new("RGB", (64, 64), (200, 30, 30)) as image:
        image.save(buffer, "JPEG")
    return '<img src="data:image/jpeg;base64,' + base64.b64encode(buffer.getvalue()).decode() + '"/>'


@pytest.mark.parametrize(
    ("bbox", "cells", "expected"),
    [
        pytest.param((20, 10, 40, 30), TWO_CELLS, 0, id="inside"),
        pytest.param((95, 20, 105, 30), TWO_CELLS, 1, id="shared-column-edge"),
        pytest.param((95, 45, 105, 55), FOUR_CELLS, 3, id="four-cell-corner"),
        pytest.param((10, 10, 290, 30), [*TWO_CELLS, (200, 0, 300, 50)], 1, id="three-columns"),
        pytest.param((95, 20, 105, 30), [(0, 0, 100, 50), (100, 0, 300, 50)], 1, id="unequal-cells"),
        pytest.param((0, 0, 100, 20), [(0, 0, 45, 20), (48, 0, 52, 20)], 1, id="center-before-coverage"),
        pytest.param((0, 0, 100, 10), [(0, 0, 100, 1000), (40, 0, 60, 10)], 0, id="coverage-before-distance"),
        pytest.param((60, 10, 120, 30), [(0, 0, 100, 50), (80, 0, 200, 50)], 0, id="overlap-center-tie"),
        pytest.param((95, 20, 105, 30), [(0, 0, 200, 50), (50, 0, 150, 50)], 0, id="geometry-tie"),
        pytest.param((95, 20, 105, 30), [(0, 0, 200, 50)] * 2, 0, id="identical-cells"),
        pytest.param((60, 10, 150, 30), [(0, 0, 80, 50), (120, 0, 200, 50)], 1, id="gap-most-overlap"),
        pytest.param((95, 20, 105, 30), [(0, 0, 90, 50), (110, 0, 200, 50)], 0, id="gap-no-overlap-tie"),
        pytest.param((-30, 90, -10, 110), [(0, 0, 200, 200), (5, 90, 15, 110)], 0, id="nearest-rectangle"),
        pytest.param((-30, 90, -10, 110), [(0, 0, 200, 200), (0, 90, 10, 110)], 1, id="nearest-center-tie"),
        pytest.param(
            (95, 20, 105, 30),
            [(100, 0, 100, 50), (0, 0, float("inf"), 50), (float("nan"), 0, 200, 50), (0, 0, 200, 50)],
            3,
            id="ignore-invalid-cells",
        ),
    ],
)
def test_atomic_token_has_one_complete_owner(atomic_content: str, bbox: Rect, cells: list[Rect], expected: int) -> None:
    """所有几何分支只接收原始 token 一次，且预测框排列不影响非重复格的归属。"""
    token = [_quad(bbox), atomic_content, 0.83]
    for order in [list(range(len(cells))), list(reversed(range(len(cells))))]:
        polygons = np.asarray([_quad(cells[index]) for index in order], dtype=float)
        matched, unmatched = match_ocr_cell([token], polygons, atomic_token_indices=frozenset({0}))
        assert unmatched == []
        assert len(matched) == 1
        cell_idx, values = next(iter(matched.items()))
        assert cells[order[cell_idx]] == cells[expected]
        if len(set(cells)) == 1:
            assert cell_idx == 0
        assert len(values) == 1 and values[0] is token
        assert token[0] == _quad(bbox) and token[1:] == [atomic_content, 0.83]


def test_atomic_token_without_valid_cells_stays_unmatched(atomic_content: str) -> None:
    """无有效结构时不向退化格分配对象，并保留空结构的原有返回语义。"""
    token = [_quad((95, 20, 105, 30)), atomic_content, 1.0]
    invalid = np.asarray([_quad((100, 0, 100, 50)), _quad((200, 0, 100, 50))], dtype=float)
    matched, unmatched = match_ocr_cell([token], invalid, atomic_token_indices=frozenset({0}))
    assert matched == {} and len(unmatched) == 1 and unmatched[0] is token
    assert match_ocr_cell([token], np.empty((0, 4, 2)), atomic_token_indices=frozenset({0})) == ({}, [])


def test_identical_object_payloads_are_not_deduplicated(atomic_content: str) -> None:
    """内容相同的两个对象仍分别保留，不能用文本或对象集合去重。"""
    tokens = [[_quad((95, y, 105, y + 10)), atomic_content, 1.0] for y in (10, 30)]
    matched, unmatched = match_ocr_cell(
        tokens, np.asarray([_quad(cell) for cell in TWO_CELLS]), atomic_token_indices=frozenset({0, 1})
    )
    assert unmatched == [] and len(matched[1]) == 2
    assert all(actual is expected for actual, expected in zip(matched[1], tokens, strict=True))


@pytest.mark.parametrize("text", ["123 456", "1,234.56 7,890.12", "2026-01-01 2026-02-02"])
def test_plain_text_keeps_cross_cell_splitting(text: str) -> None:
    """未标记文字继续按普通规则拆分，标记行为由索引而非 HTML 内容决定。"""
    token = [_quad((60, 15, 140, 35)), text, 1.0]
    polygons = np.asarray([_quad(cell) for cell in TWO_CELLS])
    matched, unmatched = match_ocr_cell([token], polygons)
    assert unmatched == []
    assert [matched[index][0][1] for index in (0, 1)] == text.split()
    atomic_matched, _ = match_ocr_cell([token], polygons, atomic_token_indices=frozenset({0}))
    assert list(atomic_matched) == [1] and atomic_matched[1][0] is token


def _result_for_table(markup: str) -> ParseResult:
    """构造包含正常页和实际表格 HTML 的两页结果，贯通完整文件导出。"""
    pages = []
    for index, content in enumerate(["<table><tr><td>normal page</td></tr></table>", markup]):
        pages.append(
            {
                "page_idx": index,
                "blocks": [
                    {
                        "type": "table",
                        "index": 0,
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "content": [{"type": "table_body", "index": 0, "bbox": [0.0, 0.0, 1.0, 1.0], "content": content}],
                    }
                ],
            }
        )
    return ParseResult(
        MiddleJson.from_dict(
            {
                "schema": "docvortex.middle",
                "schema_version": "2.0",
                "is_full_document": True,
                "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": "test"}},
                "extensions": {"mineru": {"tier": "basic", "parse_mode": "ocr"}},
                "pages": pages,
            }
        )
    )


@pytest.mark.parametrize(("angle", "image_cell", "formula_cell"), [(0, 1, 3), (90, 2, 3), (180, 3, 1), (270, 3, 2)])
def test_atomic_objects_survive_table_pipeline_and_exports(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, angle: int, image_cell: int, formula_cell: int
) -> None:
    """只替换模型预测，真实执行对象构建、旋转、有线匹配、HTML 拼接及两种导出。"""
    page = np.full((100, 200, 3), 255, dtype=np.uint8)
    page[10:40, 80:120] = [200, 30, 30]
    table = {"type": BlockType.TABLE, "bbox": [0.0, 0.0, 1.0, 1.0], "angle": angle}
    blocks = [[table, {"type": BlockType.IMAGE, "bbox": [0.4, 0.1, 0.6, 0.4]}]]
    formulas = [[{"bbox": [0.3, 0.6, 0.7, 0.8], "latex": LATEX, "score": 0.93}]]
    height, width = (200, 100) if angle in (90, 270) else (100, 200)
    polygons = np.asarray(
        [
            _quad((column * width / 2, row * height / 2, (column + 1) * width / 2, (row + 1) * height / 2))
            for row in range(2)
            for column in range(2)
        ]
    )
    # 有线结构模型返回逆时针框；真实 TableRecover 和后续方向转换继续执行。
    structure = Mock(return_value=(polygons[:, [0, 3, 2, 1]].copy(), polygons[:, [0, 3, 2, 1]].copy()))
    monkeypatch.setattr(unet, "TSRUnet", Mock(return_value=structure))

    def recognize(image: np.ndarray | list[np.ndarray], *, det: bool = True) -> list[Any]:
        """构造故意乱序的普通 OCR 结果，空格补识别返回空文本。"""
        if not det:
            return [[("", 0.0) for _ in image]]
        return [
            [
                [_quad((5, height - 10, 20, height - 5)), ("tail", 0.9)],
                [_quad((5, 2, 20, 8)), ("<eq>literal</eq>", 0.9)],
            ]
        ]

    ocr = SimpleNamespace(ocr=Mock(side_effect=recognize))
    captured: list[dict[str, Any]] = []

    def classify(tasks: list[dict[str, Any]]) -> None:
        """固定选择有线路径，同时检查模型输入未被新增结构遮罩改变。"""
        for task in tasks:
            task["table_res"].update(cls_label=AtomicModelName.WiredTable, cls_score=1.0)
            assert np.any(task["wired_table_img"] != 255)
            np.testing.assert_array_equal(task["wired_table_img"], task["table_img"])

    def wireless_predict(tasks: list[dict[str, Any]]) -> None:
        """用真实无线输入适配器验证三元组契约，并保留有线识别的验收输入。"""
        for task in tasks:
            dt_boxes, rec_res = PaddleTable.get_boxes_recs(None, task["ocr_result"], height, width)
            assert len(dt_boxes) == len(rec_res) == 4
            assert all(len(token) == 3 for token in task["ocr_result"])
            captured.append(task)
            task["table_res"]["html"] = ""

    context = SimpleNamespace(
        get_ocr_model=Mock(return_value=ocr),
        table_cls_model=SimpleNamespace(batch_predict=classify),
        wireless_table_model=SimpleNamespace(batch_predict=wireless_predict),
        wired_table_model=unet.UnetTableModel(ocr, model_path="unused"),
    )
    tables._apply_medium_table_recognition(context, blocks, formulas, [page])
    task = captured[0]
    tokens = task["ocr_result"]
    atomic_tokens = [tokens[index] for index in sorted(task["atomic_token_indices"])]
    assert len(atomic_tokens) == 2
    assert {token[1] for token in atomic_tokens} == {item["content"] for item in task["inline_objects"]}
    assert tokens[0][1] == html.escape("<eq>literal</eq>") and 0 not in task["atomic_token_indices"]
    coordinates = [(min(point[1] for point in token[0]), min(point[0] for point in token[0])) for token in tokens]
    assert coordinates == sorted(coordinates)
    for inline_object in task["inline_objects"]:
        x0, y0, x1, y1 = map(int, inline_object["mask_bbox"])
        assert np.all(ocr.ocr.call_args_list[0].args[0][y0:y1, x0:x1] == 255)
    assert "atomic_token_indices" not in structure.call_args.kwargs

    markup = table["content"]
    soup = BeautifulSoup(markup, "html.parser")
    cells = soup.find_all("td")
    assert len(cells) == 4 and len(soup.find_all("img")) == len(soup.find_all("eq")) == 1
    assert cells[image_cell].find("img") is not None
    assert cells[formula_cell].find("eq").get_text() == LATEX
    original_src = BeautifulSoup(
        next(item["content"] for item in task["inline_objects"] if item["kind"] == "image"), "html.parser"
    ).img["src"]
    expected_bytes, _ = parse_image_data_uri_strict(original_src)
    assert soup.img["src"] == original_src
    assert all(markup.count(token[1]) == 1 for token in atomic_tokens)

    result = _result_for_table(markup)
    before = result.middle_json.to_json()
    directory = tmp_path / "sdk"
    result.save(FileBasedDataWriter(str(directory)))
    archive_path = build_task_output(result, ["default"], tmp_path / "task")
    with zipfile.ZipFile(archive_path) as archive:
        entries = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    assert entries == {str(path.relative_to(directory)): path.read_bytes() for path in directory.rglob("*") if path.is_file()}
    image_entries = [value for name, value in entries.items() if name.startswith("images/")]
    assert image_entries == [expected_bytes]
    saved = ParseResult.from_json(entries["middle_json.json"].decode())
    saved_soup = BeautifulSoup(saved.pages[1].blocks[0].content[0].content, "html.parser")
    assert len(saved.pages) == 2 and saved_soup.eq.get_text() == LATEX
    assert len(saved_soup.find_all("img")) == 1 and entries[saved_soup.img["src"]] == expected_bytes
    assert result.middle_json.to_json() == before
