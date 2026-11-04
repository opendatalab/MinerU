# Copyright (c) Opendatalab. All rights reserved.
"""PP-DocLayoutV2 页眉页脚、公式框与后处理顺序回归测试。"""

from __future__ import annotations

import json
from copy import deepcopy
from itertools import permutations
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

from mineru.backend.analysis.pdf.formulas import _build_formula_inputs
from mineru.backend.analysis.pdf.layout import _build_vl_style_layout_blocks
from mineru.model.layout.pp_doclayout_v2_base import (
    PP_DOCLAYOUT_V2_LABEL_TO_ID,
    PPDocLayoutV2PostProcessor,
)
from mineru.types import BlockType


@pytest.fixture(params=["base", "onnx", pytest.param("torch", marks=pytest.mark.torch)])
def processor(request: pytest.FixtureRequest) -> type[PPDocLayoutV2PostProcessor]:
    """分别通过共享基类和两个实际后端调用后处理，不加载推理权重。"""
    if request.param == "torch":
        pytest.importorskip("torch")
        pytest.importorskip("transformers", minversion="5.10.1")
        from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel

        return PPDocLayoutV2LayoutModel
    if request.param == "onnx":
        pytest.importorskip("onnxruntime")
        from mineru.model.layout.pp_doclayout_v2_onnx import PPDocLayoutV2LayoutModelONNX

        return PPDocLayoutV2LayoutModelONNX
    return PPDocLayoutV2PostProcessor


def _layout_box(
    label: str,
    bbox: tuple[float, float, float, float],
    *,
    index: int,
    score: float = 0.9,
    cls_id: int | None = None,
) -> dict[str, Any]:
    """构造标签和类别编号同步的最小 layout 检测框。"""

    return {
        "label": label,
        "cls_id": (PP_DOCLAYOUT_V2_LABEL_TO_ID[label] if cls_id is None else cls_id),
        "bbox": list(bbox),
        "score": score,
        "index": index,
    }


def test_filter_formula_boxes_inside_page_marginals_removes_only_covered_formulas() -> None:
    """验证页眉页脚父框只删除覆盖率达标的行内和行间公式。"""

    header = _layout_box("header", (0, 0, 100, 20), index=1)
    header_formula = _layout_box("inline_formula", (20, 2, 40, 18), index=2)
    footer = _layout_box("footer", (0, 180, 100, 200), index=3)
    footer_formula = _layout_box("display_formula", (50, 182, 80, 198), index=4)
    body_text = _layout_box("text", (0, 40, 100, 80), index=5)
    body_formula = _layout_box("inline_formula", (20, 50, 40, 70), index=6)
    formula_number = _layout_box("formula_number", (70, 2, 80, 18), index=7)

    filtered = PPDocLayoutV2PostProcessor._filter_formula_boxes_inside_page_marginals(
        [
            header,
            header_formula,
            footer,
            footer_formula,
            body_text,
            body_formula,
            formula_number,
        ]
    )

    assert filtered == [
        header,
        footer,
        body_text,
        body_formula,
        formula_number,
    ]


def test_filter_formula_boxes_inside_page_marginals_uses_formula_area_threshold() -> None:
    """验证过滤阈值按公式自身面积计算，未达到八成覆盖时保留公式。"""

    header = _layout_box("header", (0, 0, 100, 20), index=1)
    covered_formula = _layout_box("inline_formula", (79, 0, 105, 20), index=2)
    partial_formula = _layout_box("inline_formula", (80, 0, 110, 20), index=3)

    filtered = PPDocLayoutV2PostProcessor._filter_formula_boxes_inside_page_marginals(
        [header, covered_formula, partial_formula],
        cover_threshold=0.8,
    )

    assert filtered == [header, partial_formula]


@pytest.mark.parametrize(
    "parent_label",
    ["header", "header_image", "footer", "footer_image"],
)
def test_filter_formula_boxes_supports_all_page_marginal_parent_labels(
    parent_label: str,
) -> None:
    """验证四类明确页眉页脚父框都可过滤其内部公式。"""

    parent = _layout_box(parent_label, (0, 0, 100, 20), index=1)
    formula = _layout_box("inline_formula", (20, 2, 40, 18), index=2)

    filtered = PPDocLayoutV2PostProcessor._filter_formula_boxes_inside_page_marginals([parent, formula])

    assert filtered == [parent]


def test_layout_postprocess_drops_nested_header_formula_and_preserves_other_formulas() -> None:
    """验证完整后处理删除页眉内公式，同时保留无父框和正文内公式。"""

    header = _layout_box("header", (0, 0, 100, 20), index=4, score=0.95)
    nested_formula = _layout_box("inline_formula", (20, 2, 40, 18), index=5, score=0.8)
    near_margin_formula = _layout_box("inline_formula", (110, 2, 130, 18), index=6, score=0.7)
    body_text = _layout_box("text", (0, 40, 100, 80), index=7)
    body_formula = _layout_box("display_formula", (20, 50, 40, 70), index=8)

    processed = PPDocLayoutV2PostProcessor._apply_layout_post_process(
        [header, nested_formula, near_margin_formula, body_text, body_formula],
        image_size=(200, 140),
    )

    assert [box["label"] for box in processed] == [
        "header",
        "display_formula",
        "text",
        "inline_formula",
    ]
    assert [box["index"] for box in processed] == [1, 2, 3, 4]
    assert processed[0]["bbox"] == [0, 0, 100, 20]
    assert processed[0]["score"] == 0.95


def test_filter_formula_boxes_inside_page_marginals_accepts_formula_cls_id() -> None:
    """验证公式标签漂移时仍可依据 cls_id 删除页眉内公式。"""

    header = _layout_box("header", (0, 0, 100, 20), index=1)
    formula_with_drifted_label = _layout_box(
        "text",
        (20, 2, 40, 18),
        index=2,
        cls_id=PP_DOCLAYOUT_V2_LABEL_TO_ID["inline_formula"],
    )

    filtered = PPDocLayoutV2PostProcessor._filter_formula_boxes_inside_page_marginals([header, formula_with_drifted_label])

    assert filtered == [header]


def test_header_footer_boundary_still_relabels_plain_text_but_not_formula() -> None:
    """验证边界规则继续改标普通文本，同时不误改未被父框覆盖的公式。"""

    boxes = [
        _layout_box("header", (0, 0, 100, 20), index=1),
        _layout_box("text", (105, 2, 130, 18), index=2),
        _layout_box("inline_formula", (140, 2, 160, 18), index=3),
    ]

    processed = PPDocLayoutV2PostProcessor._relabel_header_footer_boundary_blocks(
        boxes,
        image_size=(200, 200),
    )

    assert [box["label"] for box in processed] == [
        "header",
        "header",
        "inline_formula",
    ]


def test_filtered_marginal_formula_does_not_enter_model_or_formula_inputs() -> None:
    """验证页边公式过滤后只保留父 header，且不再进入 MFR 输入。"""

    processed = PPDocLayoutV2PostProcessor._apply_layout_post_process(
        [
            _layout_box("header", (0, 0, 100, 20), index=1),
            _layout_box("inline_formula", (20, 2, 40, 18), index=2),
        ],
        image_size=(200, 200),
    )
    image = Image.new("RGB", (200, 200), "white")
    try:
        model_list = _build_vl_style_layout_blocks([processed], [image])
    finally:
        image.close()

    assert _build_formula_inputs([processed]) == [[]]
    assert model_list == [
        [
            {
                "type": BlockType.HEADER,
                "bbox": [0.0, 0.0, 0.5, 0.1],
                "angle": 0,
            }
        ]
    ]


@pytest.mark.parametrize("detection_backend", ["torch", "onnx"])
def test_real_footnote_layout_preserves_formula_recognition(
    processor: type[PPDocLayoutV2PostProcessor], detection_backend: str
) -> None:
    """复放第七页真实框：小框仍进入公式识别，正文块中只有完整脚注。"""
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/emnlp614_footnote/detection.json").read_text())
    boxes = fixture["raw_layout"][detection_backend]
    before = deepcopy(boxes)
    processed = processor._apply_layout_post_process(
        processor._apply_paddlex_filter_boxes(boxes, drop_inline_formula=False),
        image_size=tuple(reversed(fixture["render_size"])),
    )
    assert boxes == before
    assert [(box["label"], box["cls_id"]) for box in processed] == [("inline_formula", 15), ("footnote", 10)]
    assert _build_formula_inputs([processed])[0][0]["bbox"] == [395, 2089, 806, 2122]
    with Image.new("RGB", tuple(fixture["render_size"])) as image:
        blocks = _build_vl_style_layout_blocks([processed], [image])[0]
    assert len(blocks) == 1
    assert blocks[0]["type"] == BlockType.PAGE_FOOTNOTE
    assert blocks[0]["bbox"] == [0.114, 0.892, 0.489, 0.92]


def test_backend_marginal_and_footnote_rules_are_consistent(processor: type[PPDocLayoutV2PostProcessor]) -> None:
    """两个后端均过滤明确页眉内公式，保护未被覆盖的公式和脚注内编号。"""
    boxes = [
        _layout_box("header", (0, 0, 100, 20), index=1),
        _layout_box("inline_formula", (10, 2, 30, 18), index=2),
        _layout_box("display_formula", (120, 2, 140, 18), index=3),
        _layout_box("footnote", (0, 170, 100, 195), index=4),
        _layout_box("display_formula", (10, 175, 50, 185), index=5),
        _layout_box("formula_number", (80, 175, 95, 185), index=6),
    ]
    result = processor._apply_layout_post_process(boxes, image_size=(200, 200))
    assert [box["label"] for box in result] == ["header", "display_formula", "footnote", "inline_formula", "formula_number"]


@pytest.mark.parametrize("label", sorted(PPDocLayoutV2PostProcessor.NESTED_TEXT_LABELS))
def test_nested_text_filter_keeps_outer_box_in_all_input_orders(label: str) -> None:
    """三级同类文本嵌套总是保留完整大框，不受内框高分和输入排列影响。"""
    boxes = [
        _layout_box(label, (0, 0, 100, 100), index=10, score=0.6),
        _layout_box(label, (10, 10, 90, 90), index=11, score=0.9),
        _layout_box(label, (20, 20, 30, 30), index=12, score=0.99),
    ]
    before = deepcopy(boxes)
    for ordered in permutations(boxes):
        assert PPDocLayoutV2PostProcessor._filter_nested_text_boxes(list(ordered)) == [boxes[0]]
    assert boxes == before


def test_equal_text_boxes_use_score_then_input_order() -> None:
    """相同几何只保留最高置信框，同分时保留首项且不打乱其他块的顺序。"""
    first = _layout_box("text", (0, 0, 100, 100), index=8, score=0.8)
    other = _layout_box("text", (200, 0, 300, 100), index=2)
    second = _layout_box("text", (0, 0, 100, 100), index=3, score=0.9)
    tied = _layout_box("text", (0, 0, 100, 100), index=4, score=0.9)
    assert PPDocLayoutV2PostProcessor._filter_nested_text_boxes([first, other, second, tied]) == [other, second]
    assert PPDocLayoutV2PostProcessor._filter_nested_text_boxes([tied, other, second, first]) == [tied, other]


@pytest.mark.parametrize(
    "label", ["inline_formula", "display_formula", "formula_number", "image", "table", "chart", "seal", "reference", "content"]
)
def test_nested_nontext_regions_are_not_suppressed(label: str) -> None:
    """新增规则不把公式、视觉主体或列表容器的合法层级当作文本重复。"""
    boxes = [_layout_box(label, (0, 0, 100, 100), index=1), _layout_box(label, (10, 10, 40, 40), index=2)]
    assert PPDocLayoutV2PostProcessor._filter_nested_text_boxes(boxes) == boxes


def test_nested_filter_preserves_partial_overlap_cross_labels_and_formula_class_id() -> None:
    """一像素越界、相邻脚注、异类文本及标签漂移的公式都不能由新增规则删除。"""
    boxes = [
        _layout_box("footnote", (0, 0, 100, 100), index=1),
        _layout_box("footnote", (20, 10, 101, 90), index=2),
        _layout_box("footnote", (0, 101, 100, 120), index=3),
        _layout_box("text", (10, 10, 40, 40), index=4),
        _layout_box("footnote", (10, 10, 40, 40), index=5, cls_id=15),
        _layout_box("footnote", (50, 10, 70, 40), index=6, cls_id=11),
    ]
    assert PPDocLayoutV2PostProcessor._filter_nested_text_boxes(boxes) == boxes


def test_final_filter_removes_text_duplicates_created_by_relabeling(processor: type[PPDocLayoutV2PostProcessor]) -> None:
    """先保留不同类别候选，改为同类脚注后再去重，且不删除脚注外的完整文本。"""
    boxes = [
        _layout_box("text", (20, 172, 80, 180), index=1, score=0.99),
        _layout_box("text", (0, 150, 100, 165), index=2),
        _layout_box("footnote", (0, 170, 100, 195), index=3, score=0.8),
    ]
    result = processor._apply_layout_post_process(boxes, image_size=(220, 200))
    assert [(box["label"], box["bbox"]) for box in result] == [("text", [0, 150, 100, 165]), ("footnote", [0, 170, 100, 195])]
    assert [box["index"] for box in result] == [1, 2]


@pytest.mark.parametrize(
    ("input_order", "expected_order"),
    [
        (["parent", "leading", "child", "trailing"], ["leading", "parent", "trailing"]),
        (["leading", "child", "middle", "parent", "trailing"], ["leading", "parent", "middle", "trailing"]),
        (
            ["parent", "leading", "child", "middle", "second_child", "trailing"],
            ["leading", "parent", "middle", "trailing"],
        ),
    ],
)
def test_image_absorption_uses_earliest_child_position(
    processor: type[PPDocLayoutV2PostProcessor], input_order: list[str], expected_order: list[str]
) -> None:
    """大图无论位于子图之前或之后，都取最早被吞并子图的位置且不改变其他块顺序。"""
    boxes = {
        "parent": _layout_box("image", (0, 0, 100, 100), index=0),
        "child": _layout_box("image", (10, 10, 40, 40), index=0),
        "second_child": _layout_box("image", (50, 50, 80, 80), index=0),
        "leading": _layout_box("text", (200, 0, 240, 20), index=0),
        "middle": _layout_box("text", (200, 30, 240, 50), index=0),
        "trailing": _layout_box("text", (200, 60, 240, 80), index=0),
    }
    source = [dict(boxes[name], index=index) for index, name in enumerate(input_order, start=1)]
    original = deepcopy(source)

    result = processor._apply_paddlex_filter_boxes(source, drop_inline_formula=False)

    assert result == [dict(boxes[name], index=index) for index, name in enumerate(expected_order, start=1)]
    assert source == original


def test_image_absorption_propagates_inherited_child_position(processor: type[PPDocLayoutV2PostProcessor]) -> None:
    """中图先吞小图再被大图吞并时，传递小图位置而非中图原先抢占的位置。"""
    middle = _layout_box("image", (10, 10, 60, 60), index=1)
    leading = _layout_box("text", (200, 0, 240, 20), index=2)
    child = _layout_box("image", (20, 20, 40, 40), index=3)
    parent = _layout_box("image", (0, 0, 100, 100), index=4)
    trailing = _layout_box("text", (200, 30, 240, 50), index=5)

    result = processor._apply_paddlex_filter_boxes([middle, leading, child, parent, trailing], drop_inline_formula=False)

    assert result == [dict(leading, index=1), dict(parent, index=2), dict(trailing, index=3)]


def test_image_absorption_positions_independent_groups(processor: type[PPDocLayoutV2PostProcessor]) -> None:
    """多个图组各自继承其子图位置，图组间正文保持原有相对顺序。"""
    left = _layout_box("image", (0, 0, 100, 100), index=1)
    right = _layout_box("image", (200, 0, 300, 100), index=2)
    leading = _layout_box("text", (400, 0, 440, 20), index=3)
    right_child = _layout_box("image", (210, 10, 240, 40), index=4)
    between = _layout_box("text", (400, 30, 440, 50), index=5)
    left_child = _layout_box("image", (10, 10, 40, 40), index=6)
    trailing = _layout_box("text", (400, 60, 440, 80), index=7)

    result = processor._apply_paddlex_filter_boxes(
        [left, right, leading, right_child, between, left_child, trailing], drop_inline_formula=False
    )

    assert result == [dict(box, index=index) for index, box in enumerate([leading, right, between, left, trailing], start=1)]


@pytest.mark.parametrize(("child_x", "absorbed"), [(72, False), (71, True)])
def test_image_position_inheritance_preserves_existing_overlap_rule(
    processor: type[PPDocLayoutV2PostProcessor], child_x: int, absorbed: bool
) -> None:
    """继续使用大于 0.7 的既有重叠判定，既不降低门槛也不额外要求八成或完全包含。"""
    parent = _layout_box("image", (0, 0, 100, 100), index=1)
    text = _layout_box("text", (200, 0, 240, 20), index=2)
    child = _layout_box("image", (child_x, 10, child_x + 40, 50), index=3)

    result = processor._apply_paddlex_filter_boxes([parent, text, child], drop_inline_formula=False)

    expected = [text, parent] if absorbed else [parent, text, child]
    assert result == [dict(box, index=index) for index, box in enumerate(expected, start=1)]


def test_equal_area_image_deduplication_does_not_inherit_position(processor: type[PPDocLayoutV2PostProcessor]) -> None:
    """等面积图片重复框沿用现有首框选择，不能把去重变成吞并定位。"""
    parent = _layout_box("image", (0, 0, 100, 100), index=1, score=0.8)
    text = _layout_box("text", (200, 0, 240, 20), index=2)
    duplicate = _layout_box("image", (0, 0, 100, 100), index=3, score=0.9)

    result = processor._apply_paddlex_filter_boxes([parent, text, duplicate], drop_inline_formula=False)

    assert result == [parent, text]


@pytest.mark.parametrize("label", ["text", "abstract", "table", "chart", "seal", "display_formula"])
def test_non_image_absorption_keeps_parent_position(processor: type[PPDocLayoutV2PostProcessor], label: str) -> None:
    """其他类别仍按原规则删除小框并保留大框自身位置，不应用图片位置继承。"""
    parent = _layout_box(label, (0, 0, 100, 100), index=1)
    text = _layout_box("text", (200, 0, 240, 20), index=2)
    child = _layout_box(label, (10, 10, 40, 40), index=3)

    result = processor._apply_paddlex_filter_boxes([parent, text, child], drop_inline_formula=False)

    assert result == [parent, text]


@pytest.mark.parametrize("backend", ["onnx", pytest.param("torch", marks=pytest.mark.torch)])
@pytest.mark.parametrize("use_filter", [False, True])
def test_layout_batch_respects_image_absorption_filter_switch(
    monkeypatch: pytest.MonkeyPatch, backend: str, use_filter: bool
) -> None:
    """通过真实批量入口验证关闭过滤时不吞并也不重定位，启用时才继承子图位置。"""
    raw = [
        _layout_box("image", (0, 40, 100, 100), index=1),
        _layout_box("text", (0, 0, 100, 20), index=2),
        _layout_box("image", (10, 50, 30, 70), index=3),
    ]
    if backend == "torch":
        torch = pytest.importorskip("torch")
        pytest.importorskip("transformers", minversion="5.10.1")
        from mineru.model.layout.pp_doclayoutv2 import PPDocLayoutV2LayoutModel

        model = object.__new__(PPDocLayoutV2LayoutModel)
        model.device = "cpu"
        model.model = Mock()
        pixels = torch.zeros((3, 16, 16))
        monkeypatch.setattr(model, "_post_process_object_detection", Mock(return_value=[{}]))
    else:
        pytest.importorskip("onnxruntime")
        from mineru.model.layout.pp_doclayout_v2_onnx import PPDocLayoutV2LayoutModelONNX

        model = object.__new__(PPDocLayoutV2LayoutModelONNX)
        pixels = np.zeros((3, 16, 16), dtype=np.float32)
        monkeypatch.setattr(model, "_run_session", Mock(return_value=[{}]))
    model.use_paddlex_filter_boxes = True
    monkeypatch.setattr(model, "_preprocess_single_image", Mock(return_value=(pixels, (200, 200))))
    monkeypatch.setattr(model, "_parse_prediction", Mock(return_value=deepcopy(raw)))

    with Image.new("RGB", (200, 200)) as image:
        result = model.batch_predict([image], use_paddlex_filter_boxes=use_filter)[0]

    expected = [dict(raw[1], index=1), dict(raw[0], index=2)] if use_filter else raw
    assert result == expected


@pytest.mark.parametrize("detection_backend", ["torch", "onnx"])
def test_demo2_image_absorption_restores_order_and_common_block_inputs(
    processor: type[PPDocLayoutV2PostProcessor], detection_backend: str
) -> None:
    """复放 demo2 第四页真实框，整图移到正文后且其余 28 块及共用布局输入保持稳定。"""
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/pp_doclayoutv2/demo2_page4.json").read_text())
    page = fixture["backends"][detection_backend]
    raw = [
        _layout_box(label, tuple(bbox), index=index, score=score)
        for index, (label, score, bbox) in enumerate(page["raw_rows"], start=1)
    ]
    original = deepcopy(raw)

    processed = processor._apply_layout_post_process(
        processor._apply_paddlex_filter_boxes(raw, drop_inline_formula=False), image_size=tuple(fixture["image_size"])
    )

    expected = [
        dict(raw[source_index - 1], index=index) for index, source_index in enumerate(page["expected_source_indices"], start=1)
    ]
    threshold_only = [
        dict(raw[source_index - 1], index=index)
        for index, source_index in enumerate(page["threshold_only_source_indices"], start=1)
    ]
    assert raw == original
    assert processed == expected
    assert len(processed) == 30
    assert processed[2:] == threshold_only[2:]
    assert [box["label"] for box in processed[:2]] == ["text", "image"]
    assert sum(box["label"] == "image" for box in processed) == 1
    assert sum(box["label"] == "chart" and box["bbox"][1] < 1000 for box in processed) == 3
    with Image.new("RGB", tuple(reversed(fixture["image_size"]))) as image:
        blocks = _build_vl_style_layout_blocks([processed], [image])[0]
    assert [block["type"] for block in blocks[:2]] == [BlockType.TEXT, BlockType.IMAGE]
    assert blocks[1]["bbox"] == [0.082, 0.127, 0.478, 0.612]
