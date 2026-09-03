from __future__ import annotations

import pytest

from mineru.model.flash.pdf import line_layout, pipeline, text_blocks

from _flash_pdf_test_utils import _text_line


def test_line_tight_output_bbox_adds_one_point_padding_and_clips_page() -> None:
    """验证可靠 tight 字形框四边各扩 1pt，并在页面边缘安全裁剪。"""

    line = _text_line(
        "value",
        (0.0, 0.0, 80.0, 40.0),
        0,
        ink_bbox=(0.5, 0.25, 40.0, 20.0),
    )

    assert line_layout._line_tight_output_bbox(
        line,
        (100.0, 100.0),
    ) == (0.0, 0.0, 41.0, 21.0)


def test_text_block_applies_tight_output_bbox_after_aggregation() -> None:
    """验证文本先按 layout 聚合，再同步应用 tight+1pt 的 block 与 line bbox。"""

    line = _text_line(
        "value",
        (10.0, 20.0, 90.0, 80.0),
        0,
        ink_bbox=(20.0, 30.0, 40.0, 50.0),
    )
    blocks = text_blocks._build_text_blocks(
        [line],
        [],
        (100.0, 100.0),
    )

    assert len(blocks) == 1
    assert blocks[0]["bbox"] == (10.0, 20.0, 90.0, 80.0)
    pipeline._apply_post_aggregation_tight_bboxes(
        blocks,
        (100.0, 100.0),
    )

    assert blocks[0]["bbox"] == (19.0, 29.0, 41.0, 51.0)
    assert blocks[0]["_local_line_bboxes"] == [
        (19.0, 29.0, 41.0, 51.0),
    ]


def test_direct_formula_tight_bbox_is_applied_and_internal_key_removed() -> None:
    """验证文本公式的聚合后候选替换公开框，且内部字段不会继续外泄。"""

    blocks = [
        {
            "type": "equation",
            "bbox": (10.0, 20.0, 90.0, 80.0),
            "angle": 0,
            "content": "x=1",
            "_tight_output_bbox": (19.0, 29.0, 41.0, 51.0),
        }
    ]

    pipeline._apply_post_aggregation_tight_bboxes(
        blocks,
        (100.0, 100.0),
    )

    assert blocks[0]["bbox"] == (19.0, 29.0, 41.0, 51.0)
    assert "_tight_output_bbox" not in blocks[0]


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
