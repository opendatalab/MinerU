from __future__ import annotations

import pytest

from mineru.model.flash.pdf import pipeline


@pytest.mark.parametrize(
    "block_type",
    ["text", "ref_text", "doc_title", "paragraph_title", "caption", "footnote"],
)
def test_output_normalization_exposes_lines_for_pdf_text_types(
    block_type: str,
) -> None:
    """验证 Flash 对包括 ref_text 在内的 PDF 文本块公开归一化行框。"""

    block = pipeline._normalize_output_block(
        {
            "type": block_type,
            "bbox": (10.0, 20.0, 90.0, 80.0),
            "angle": 0,
            "content": "line one\nline two",
            "_local_line_bboxes": [
                (10.0, 20.0, 90.0, 40.0),
                (20.0, 50.0, 80.0, 80.0),
            ],
        },
        (100.0, 200.0),
    )

    assert block is not None
    assert block["lines"] == [
        {"bbox": [0.1, 0.1, 0.9, 0.2]},
        {"bbox": [0.2, 0.25, 0.8, 0.4]},
    ]


@pytest.mark.parametrize(
    ("angle", "expected_bbox"),
    [
        (0, [0.2, 0.05, 0.4, 0.15]),
        (90, [0.7, 0.1, 0.9, 0.2]),
        (270, [0.1, 0.8, 0.3, 0.9]),
    ],
)
def test_output_normalization_restores_rotated_line_bbox_to_page(
    angle: int,
    expected_bbox: list[float],
) -> None:
    """验证局部行框会按 block 方向逆变换到原页面坐标。"""

    block = pipeline._normalize_output_block(
        {
            "type": "text",
            "bbox": (10.0, 10.0, 90.0, 190.0),
            "angle": angle,
            "content": "value",
            "_local_line_bboxes": [(20.0, 10.0, 40.0, 30.0)],
        },
        (100.0, 200.0),
    )

    assert block is not None
    assert block["lines"] == [{"bbox": expected_bbox}]


@pytest.mark.parametrize("block_type", ["text", "ref_text"])
@pytest.mark.parametrize(
    "local_line_bboxes",
    [
        None,
        [],
        [(10.0, 20.0, 40.0, 30.0), (1.0, 2.0, 1.0, 3.0)],
        [(40.0, 20.0, 10.0, 30.0)],
    ],
)
def test_output_normalization_fails_closed_for_invalid_line_bboxes(
    block_type: str,
    local_line_bboxes: object,
) -> None:
    """验证 text/ref_text 内部行框缺失、为空或任一非法时输出空 lines。"""

    block = pipeline._normalize_output_block(
        {
            "type": block_type,
            "bbox": (10.0, 20.0, 40.0, 50.0),
            "angle": 0,
            "content": "value",
            "_local_line_bboxes": local_line_bboxes,
        },
        (100.0, 100.0),
    )

    assert block is not None
    assert block["lines"] == []
