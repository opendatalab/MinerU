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
from mineru.types import RAW_FOOTNOTE, BlockType


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
        boxes,
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
def test_image_absorption_uses_median_child_position(
    processor: type[PPDocLayoutV2PostProcessor], input_order: list[str], expected_order: list[str]
) -> None:
    """大图使用全部子图中位数，而非最早子图序号，其他块相对顺序不变。"""
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

    result = processor._apply_layout_post_process(source)

    assert result == [dict(boxes[name], index=index) for index, name in enumerate(expected_order, start=1)]
    assert source == original


def test_image_absorption_collects_all_descendant_positions(processor: type[PPDocLayoutV2PostProcessor]) -> None:
    """一次收集所有大小子图的原始序号，不传递子节点聚合后的顺序。"""
    middle = _layout_box("image", (10, 10, 60, 60), index=1)
    leading = _layout_box("text", (200, 0, 240, 20), index=2)
    child = _layout_box("image", (20, 20, 40, 40), index=3)
    parent = _layout_box("image", (0, 0, 100, 100), index=4)
    trailing = _layout_box("text", (200, 30, 240, 50), index=5)

    result = processor._apply_layout_post_process([middle, leading, child, parent, trailing])

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

    result = processor._apply_layout_post_process([left, right, leading, right_child, between, left_child, trailing])

    assert result == [dict(box, index=index) for index, box in enumerate([leading, right, between, left, trailing], start=1)]


@pytest.mark.parametrize(("child_x", "absorbed"), [(72, False), (71, True)])
def test_image_position_inheritance_preserves_existing_overlap_rule(
    processor: type[PPDocLayoutV2PostProcessor], child_x: int, absorbed: bool
) -> None:
    """超过七成时有界合并并覆盖越界部分，恰好七成时保留两个图片。"""
    parent = _layout_box("image", (0, 0, 100, 100), index=1)
    text = _layout_box("text", (200, 0, 240, 20), index=2)
    child = _layout_box("image", (child_x, 10, child_x + 40, 50), index=3)

    result = processor._apply_layout_post_process([parent, text, child])

    expected = [text, dict(parent, bbox=[0, 0, child_x + 40, 100])] if absorbed else [parent, text, child]
    assert result == [dict(box, index=index) for index, box in enumerate(expected, start=1)]


def test_equal_area_image_deduplication_does_not_inherit_position(processor: type[PPDocLayoutV2PostProcessor]) -> None:
    """等面积重复图片取高分检测自身顺序，不将低分重复框当成子图。"""
    parent = _layout_box("image", (0, 0, 100, 100), index=1, score=0.8)
    text = _layout_box("text", (200, 0, 240, 20), index=2)
    duplicate = _layout_box("image", (0, 0, 100, 100), index=3, score=0.9)

    result = processor._apply_layout_post_process([parent, text, duplicate])

    assert result == [dict(text, index=1), dict(duplicate, index=2)]


@pytest.mark.parametrize("label", ["text", "abstract", "table", "chart", "seal", "display_formula"])
def test_non_image_absorption_keeps_parent_position(processor: type[PPDocLayoutV2PostProcessor], label: str) -> None:
    """其他类别仍按原规则删除小框并保留大框自身位置，不应用图片位置继承。"""
    parent = _layout_box(label, (0, 0, 100, 100), index=1)
    text = _layout_box("text", (200, 0, 240, 20), index=2)
    child = _layout_box(label, (10, 10, 40, 40), index=3)

    result = processor._apply_layout_post_process([parent, text, child])

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

    processed = processor._apply_layout_post_process(raw, image_size=tuple(fixture["image_size"]))

    expected = [
        dict(raw[source_index - 1], index=index) for index, source_index in enumerate(page["expected_source_indices"], start=1)
    ]
    # 最下方子图越出旧总图一像素；新总图必须包含这条原先被裁掉的边缘。
    expected[1] = dict(expected[1], bbox=[140, 280, 812, 1347])
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


@pytest.mark.parametrize("left_label,right_label", [("text", "paragraph_title"), ("image", "text")])
def test_duplicate_uses_high_score_type_and_union(left_label: str, right_label: str) -> None:
    """高 IoU 重复先按分数选类，并保存低分框独有的边缘，不受输入排列影响。"""
    low = _layout_box(left_label, (0, 100, 100, 200), index=1, score=0.6)
    high = _layout_box(right_label, (1, 101, 101, 201), index=2, score=0.95)
    for ordered in permutations([low, high]):
        before = deepcopy(ordered)
        result = PPDocLayoutV2PostProcessor._apply_layout_post_process(list(ordered))
        assert result == [dict(high, bbox=[0, 100, 101, 201], index=1)]
        assert ordered == before


def test_duplicate_equal_scores_prefer_area_then_original_order() -> None:
    """重复检测同分时先保留较大区域，同面积时使用原始序号而非传入排列。"""
    small = _layout_box("text", (1, 101, 99, 199), index=1)
    large = _layout_box("paragraph_title", (0, 100, 100, 200), index=3)
    tied = _layout_box("abstract", (0, 100, 100, 200), index=2)
    for ordered in permutations([small, large, tied]):
        result = PPDocLayoutV2PostProcessor._apply_layout_post_process(list(ordered))
        assert result == [dict(tied, index=1)]


def test_duplicate_threshold_is_strict_and_union_does_not_chain() -> None:
    """关闭通用合并后，恰好 0.9 不去重，重复框并集也不能继续吸收第三个框。"""
    exact = [
        _layout_box("text", (0, 100, 19, 120), index=1),
        _layout_box("paragraph_title", (1, 100, 20, 120), index=2),
    ]
    assert PPDocLayoutV2PostProcessor._calculate_iou(exact[0]["bbox"], exact[1]["bbox"]) == 0.9
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process(exact, use_paddlex_filter_boxes=False)
    assert result == exact
    chain = [
        _layout_box("text", (0, 100, 100, 200), index=1, score=0.95),
        _layout_box("text", (4, 100, 104, 200), index=2, score=0.9),
        _layout_box("text", (8, 100, 108, 200), index=3, score=0.85),
    ]
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process(chain, use_paddlex_filter_boxes=False)
    assert result == [dict(chain[0], bbox=[0, 100, 104, 200]), dict(chain[2], index=2)]


@pytest.mark.parametrize("label", ["text", "display_formula"])
def test_bounded_merge_uses_frozen_anchors_in_all_orders(label: str) -> None:
    """通用和公式合并都不沿重叠链扩张，固定原始序号后输入排列不改变结果。"""
    boxes = [
        _layout_box(label, (offset, 100, offset + 100, 200), index=index, score=1 - index * 0.1)
        for index, offset in enumerate((0, 20, 40, 60), start=1)
    ]
    expected = [dict(boxes[0], bbox=[0, 100, 120, 200]), dict(boxes[2], bbox=[40, 100, 160, 200], index=2)]
    for ordered in permutations(boxes):
        result = PPDocLayoutV2PostProcessor._apply_layout_post_process(list(ordered))
        assert result == expected


def test_bounded_merge_area_limit_is_fixed_and_rejected_member_survives() -> None:
    """并集恰好 1.5 倍可以保留，再加入导致超限的检测必须独立存活。"""
    anchor = _layout_box("table", (100, 100, 200, 200), index=1, score=0.6)
    left = _layout_box("text", (75, 110, 175, 130), index=2, score=0.99)
    right = _layout_box("paragraph_title", (125, 140, 225, 160), index=3, score=0.98)
    top = _layout_box("text", (110, 75, 130, 175), index=4, score=0.97)
    source = [anchor, left, right, top]
    for ordered in permutations(source):
        result = PPDocLayoutV2PostProcessor._apply_layout_post_process(list(ordered))
        assert result == [dict(anchor, bbox=[75, 100, 225, 200]), dict(top, index=2)]
        assert all(
            any(PPDocLayoutV2PostProcessor._calculate_cover_ratio(box["bbox"], kept["bbox"]) == 1 for kept in result)
            for box in source
        )


@pytest.mark.parametrize("outer_label,inner_label", [("table", "text"), ("text", "table"), ("text", "paragraph_title")])
@pytest.mark.parametrize("bottom", [199, 201])
def test_contained_and_partial_cross_label_merge_keep_large_type(outer_label: str, inner_label: str, bottom: int) -> None:
    """完整包含与微小越界统一保留大框类型和分数，不使用小框高分改写区域语义。"""
    parent = _layout_box(outer_label, (0, 100, 100, 200), index=1, score=0.6)
    child = _layout_box(inner_label, (10, 180, 40, bottom), index=2, score=0.99)
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([parent, child])
    assert result == [dict(parent, bbox=[0, 100, 100, max(200, bottom)])]


@pytest.mark.parametrize("inner_label", ["text", "table", "chart", "seal"])
def test_visual_cross_label_coexistence_remains(inner_label: str) -> None:
    """非近重复的视觉跨类别组合仍共存，不因有界合并而丢失内部内容。"""
    boxes = [
        _layout_box("image", (0, 100, 100, 200), index=1),
        _layout_box(inner_label, (10, 120, 40, 150), index=2),
    ]
    assert PPDocLayoutV2PostProcessor._apply_layout_post_process(boxes) == boxes


@pytest.mark.parametrize("child_indices", [(1, 8, 12, 14), (1, 10, 12)])
def test_image_median_resists_first_outlier_and_does_not_round(child_indices: tuple[int, ...]) -> None:
    """奇偶数量子图均使用原始中位数，异常首位子图不会把底部总图拉到开头。"""
    parent = _layout_box("image", (0, 100, 100, 200), index=2)
    before = _layout_box("text", (200, 100, 240, 120), index=9)
    after = _layout_box("text", (200, 140, 240, 160), index=11)
    children = [
        _layout_box("image", (10 + i * 20, 120, 25 + i * 20, 140), index=index) for i, index in enumerate(child_indices)
    ]
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([parent, before, after, *children])
    assert result == [dict(before, index=1), dict(parent, index=2), dict(after, index=3)]
    # 4.5 不能取整到 4；原始父框序号更早也不能抢到第 4 个位置之前。
    children = [_layout_box("image", (10, 120, 30, 140), index=3), _layout_box("image", (40, 120, 60, 140), index=6)]
    before["index"], after["index"] = 4, 5
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([parent, before, after, *children])
    assert result == [dict(before, index=1), dict(parent, index=2), dict(after, index=3)]


def test_duplicate_child_only_votes_once_in_image_median() -> None:
    """子图近重复先归一为高分检测，再参与一次中位数投票，不累计被删重复的位置。"""
    parent = _layout_box("image", (0, 100, 100, 200), index=1)
    duplicate = _layout_box("image", (10, 120, 25, 140), index=2, score=0.7)
    winner = dict(duplicate, index=100, score=0.99)
    children = [_layout_box("image", (30 + i * 20, 120, 45 + i * 20, 140), index=index) for i, index in enumerate((4, 8, 12))]
    before = _layout_box("text", (200, 100, 240, 120), index=9)
    after = _layout_box("text", (200, 140, 240, 160), index=11)
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([parent, duplicate, winner, *children, before, after])
    assert result == [dict(before, index=1), dict(parent, index=2), dict(after, index=3)]


@pytest.mark.parametrize("label", ["figure_title", "vision_footnote"])
def test_visual_annotation_filter_uses_final_body_and_keeps_external_note(label: str) -> None:
    """删除最终图片中的两类注释，保留图外说明；原图被替换成文本后不再充当过滤父框。"""
    body = _layout_box("image", (0, 100, 100, 200), index=1, score=0.6)
    inside = _layout_box(label, (10, 130, 80, 145), index=2)
    outside = _layout_box(label, (10, 210, 80, 225), index=3)
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([body, inside, outside])
    assert result == [body, dict(outside, index=2)]
    text = _layout_box("text", (0, 100, 100, 200), index=1, score=0.99)
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([body, text, inside], use_paddlex_filter_boxes=False)
    assert result == [text, inside]


def test_relabelled_visual_body_deduplicates_without_image_reordering() -> None:
    """图和注释一起成为脚注后只留大框，且不再按已吞并子图的中位数重排。"""
    footnote = _layout_box("footnote", (0, 100, 500, 110), index=1)
    parent = _layout_box("image", (0, 120, 100, 220), index=2)
    text = _layout_box("text", (600, 120, 700, 140), index=3)
    child = _layout_box("image", (10, 140, 40, 170), index=4)
    caption = _layout_box("figure_title", (10, 180, 40, 195), index=5)
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([footnote, parent, text, child, caption])
    assert result == [footnote, dict(parent, label="footnote", cls_id=10), text]


@pytest.mark.parametrize("parent_label", ["text", "footnote"])
@pytest.mark.parametrize("formula_label", ["display_formula", "inline_formula"])
@pytest.mark.parametrize("use_filter", [False, True])
def test_near_duplicate_text_formula_pair_preserves_both_semantic_layers(
    parent_label: str, formula_label: str, use_filter: bool
) -> None:
    """短文字配长公式时高 IoU 不触发互相抑制，行间误标也能转为行内并进入识别。"""
    parent = _layout_box(parent_label, (0, 100, 100, 200), index=1, score=0.6)
    formula = _layout_box(formula_label, (1, 101, 99, 199), index=2, score=0.99)
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([parent, formula], use_paddlex_filter_boxes=use_filter)
    assert result == [parent, dict(formula, label="inline_formula", cls_id=15)]
    assert len(_build_formula_inputs([result])[0]) == 1


def test_formula_number_is_not_suppressed_by_formula_or_text() -> None:
    """公式内部的编号不会被提前删除，脚注内编号也保留给后续编号识别和合并。"""
    formula = _layout_box("display_formula", (0, 100, 100, 140), index=1)
    number = _layout_box("formula_number", (80, 110, 95, 130), index=2)
    assert PPDocLayoutV2PostProcessor._apply_layout_post_process([formula, number]) == [formula, number]
    parent = _layout_box("footnote", (0, 90, 110, 150), index=3)
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([formula, number, parent])
    assert result == [dict(formula, label="inline_formula", cls_id=15), number, parent]


def test_small_box_filter_preserves_inline_protection_without_widening_other_labels() -> None:
    """保留原行内公式及新转行内公式的尺寸豁免，不放宽独立公式和其他类别的六像素门槛。"""
    parent = _layout_box("text", (0, 100, 100, 140), index=1)
    nested = _layout_box("display_formula", (20, 110, 60, 115), index=2)
    orphan_inline = _layout_box("inline_formula", (200, 110, 240, 115), index=3)
    tiny_display = _layout_box("display_formula", (300, 110, 340, 115), index=4)
    tiny_text = _layout_box("text", (400, 110, 440, 115), index=5)
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([parent, nested, orphan_inline, tiny_display, tiny_text])
    assert result == [
        parent,
        dict(nested, label="inline_formula", cls_id=15),
        dict(orphan_inline, label="display_formula", cls_id=5),
    ]


@pytest.mark.parametrize(
    "anchor_label,anchor_bbox,parent_bbox",
    [
        ("header", (800, 150, 900, 180), (100, 80, 700, 160)),
        ("footer", (0, 700, 1000, 730), (100, 750, 700, 850)),
    ],
)
def test_formula_cleanup_runs_after_boundary_relabelling(
    anchor_label: str, anchor_bbox: tuple[int, ...], parent_bbox: tuple[int, ...]
) -> None:
    """新改标的页眉页脚也会清理内部公式，验证清理顺序而非只覆盖原始父框。"""
    anchor = _layout_box(anchor_label, anchor_bbox, index=1)
    parent = _layout_box("text", parent_bbox, index=2)
    formula = _layout_box("display_formula", (200, parent_bbox[1] + 10, 300, parent_bbox[1] + 40), index=3)
    result = PPDocLayoutV2PostProcessor._apply_layout_post_process([anchor, parent, formula], image_size=(1000, 1000))
    assert result == [anchor, dict(parent, label=anchor_label, cls_id=PP_DOCLAYOUT_V2_LABEL_TO_ID[anchor_label])]
    assert _build_formula_inputs([result]) == [[]]


@pytest.mark.parametrize("detection_backend", ["torch", "onnx"])
def test_smac_layout_removes_internal_note_and_restores_image_order(
    processor: type[PPDocLayoutV2PostProcessor], detection_backend: str
) -> None:
    """复放 SMAC 实测框，修正图内脚注和 ONNX 抢序，并只扩展子图越界的两条边。"""
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/pp_doclayoutv2/smac_1045h.json").read_text())
    raw = fixture["backends"][detection_backend]
    before = deepcopy(raw)
    source_indices = [*range(1, 25), 28, 37, 39] if detection_backend == "torch" else [*range(2, 26), 29, 38, 40]
    expected = [dict(raw[index - 1], index=order) for order, index in enumerate(source_indices, start=1)]
    expected[24] = dict(expected[24], bbox=[173, 1523, 1539, 2059])
    result = processor._apply_layout_post_process(raw, image_size=tuple(fixture["image_size"]))
    assert result == expected
    assert raw == before
    assert all(set(item) == {"cls_id", "label", "score", "bbox", "index"} for item in result)
    assert len(_build_formula_inputs([result])[0]) == 9
    with Image.new("RGB", tuple(reversed(fixture["image_size"]))) as image:
        blocks = _build_vl_style_layout_blocks([result], [image])[0]
    assert [block["type"] for block in blocks[-4:]] == [BlockType.TEXT, BlockType.IMAGE, RAW_FOOTNOTE, BlockType.PAGE_NUMBER]
    assert blocks[-3]["bbox"] == [0.102, 0.664, 0.905, 0.898]
