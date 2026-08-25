from mineru.model.flash.pdf import pipeline


def test_marginal_header_row_does_not_break_two_column_reading_order() -> None:
    """验证边缘页眉按视觉行排序且正文保持先左栏后右栏。"""
    blocks = [
        {"type": "header", "bbox": (80.0, 2.0, 120.0, 12.0), "angle": 0, "content": "center"},
        {"type": "page_number", "bbox": (2.0, 0.0, 12.0, 14.0), "angle": 0, "content": "page"},
        {"type": "header", "bbox": (175.0, 1.0, 198.0, 13.0), "angle": 0, "content": "volume"},
        {"type": "text", "bbox": (0.0, 20.0, 80.0, 40.0), "angle": 0, "content": "left one"},
        {"type": "text", "bbox": (0.0, 45.0, 80.0, 65.0), "angle": 0, "content": "left two"},
        {"type": "text", "bbox": (120.0, 20.0, 200.0, 40.0), "angle": 0, "content": "right one"},
        {"type": "text", "bbox": (120.0, 45.0, 200.0, 65.0), "angle": 0, "content": "right two"},
    ]

    sorted_blocks = pipeline._sort_blocks_with_visual_row_groups(blocks, (200.0, 100.0))

    assert [block["content"] for block in sorted_blocks] == [
        "page",
        "center",
        "volume",
        "left one",
        "left two",
        "right one",
        "right two",
    ]


def test_overlapping_span_captions_follow_visual_center_order() -> None:
    """验证跨栏带中轻微重叠的图注按视觉中心稳定排序。"""
    common = {
        "type": "text",
        "angle": 0,
        "_lane_interval": (20.0, 180.0),
        "_lane_is_span": True,
        "_line_heights": [10.0],
    }
    later = {**common, "bbox": (30.0, 58.0, 170.0, 70.0), "content": "later"}
    earlier = {**common, "bbox": (70.0, 50.0, 130.0, 62.0), "content": "earlier"}

    stabilized = pipeline._stabilize_overlapping_lane_order(
        [later, earlier],
        (200.0, 100.0),
    )

    assert [block["content"] for block in stabilized] == ["earlier", "later"]
