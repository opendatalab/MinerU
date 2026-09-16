# Copyright (c) Opendatalab. All rights reserved.
"""PP-DocLayoutV2 页眉页脚、公式框与后处理顺序回归测试。"""

from __future__ import annotations

from typing import Any

import pytest
from PIL import Image

from mineru.backend.analysis.pdf.formulas import _build_formula_inputs
from mineru.backend.analysis.pdf.layout import _build_vl_style_layout_blocks
from mineru.model.layout.pp_doclayoutv2 import (
    PP_DOCLAYOUT_V2_LABEL_TO_ID,
    PPDocLayoutV2LayoutModel,
)
from mineru.types import BlockType


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
        "cls_id": (
            PP_DOCLAYOUT_V2_LABEL_TO_ID[label]
            if cls_id is None
            else cls_id
        ),
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

    filtered = PPDocLayoutV2LayoutModel._filter_formula_boxes_inside_page_marginals(
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

    filtered = PPDocLayoutV2LayoutModel._filter_formula_boxes_inside_page_marginals(
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

    filtered = PPDocLayoutV2LayoutModel._filter_formula_boxes_inside_page_marginals(
        [parent, formula]
    )

    assert filtered == [parent]


def test_layout_postprocess_drops_nested_header_formula_and_preserves_other_formulas() -> None:
    """验证完整后处理删除页眉内公式，同时保留无父框和正文内公式。"""

    header = _layout_box("header", (0, 0, 100, 20), index=4, score=0.95)
    nested_formula = _layout_box("inline_formula", (20, 2, 40, 18), index=5, score=0.8)
    near_margin_formula = _layout_box("inline_formula", (110, 2, 130, 18), index=6, score=0.7)
    body_text = _layout_box("text", (0, 40, 100, 80), index=7)
    body_formula = _layout_box("display_formula", (20, 50, 40, 70), index=8)

    processed = PPDocLayoutV2LayoutModel._apply_layout_post_process(
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

    filtered = PPDocLayoutV2LayoutModel._filter_formula_boxes_inside_page_marginals(
        [header, formula_with_drifted_label]
    )

    assert filtered == [header]


def test_header_footer_boundary_still_relabels_plain_text_but_not_formula() -> None:
    """验证边界规则继续改标普通文本，同时不误改未被父框覆盖的公式。"""

    boxes = [
        _layout_box("header", (0, 0, 100, 20), index=1),
        _layout_box("text", (105, 2, 130, 18), index=2),
        _layout_box("inline_formula", (140, 2, 160, 18), index=3),
    ]

    processed = PPDocLayoutV2LayoutModel._relabel_header_footer_boundary_blocks(
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

    processed = PPDocLayoutV2LayoutModel._apply_layout_post_process(
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
