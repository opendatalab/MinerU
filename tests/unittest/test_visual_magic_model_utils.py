from __future__ import annotations

from mineru.backend.utils.visual_magic_model_utils import (
    VISUAL_MAIN_TYPES,
    _bbox_for_calculation,
    fallback_inline_caption_fragments,
    fallback_leading_table_continuation_captions,
    find_best_visual_parent,
    is_block_outside_visual_gap,
)
from mineru.types import Block, BlockType, Line


def _line_metadata(count: int) -> list[dict[str, list[float]]]:
    """构造仅包含 bbox 的 dict 行级元数据。"""
    return [{"bbox": [0.0, float(index), 1.0, float(index + 1)]} for index in range(count)]


def test_bbox_for_calculation_scales_only_all_unit_range_coordinates() -> None:
    """验证仅四个坐标均不大于一时生成放大后的计算副本。"""
    normalized_bbox = (0.123, 0.221, 0.445, 0.556)

    assert _bbox_for_calculation(normalized_bbox) == (123.0, 221.0, 445.0, 556.0)
    assert normalized_bbox == (0.123, 0.221, 0.445, 0.556)
    assert _bbox_for_calculation((123, 221, 445, 556)) == (123, 221, 445, 556)
    assert _bbox_for_calculation((0.123, 0.221, 0.445, 1.0)) == (123.0, 221.0, 445.0, 1000.0)
    assert _bbox_for_calculation((0.123, 0.221, 0.445, 1.001)) == (0.123, 0.221, 0.445, 1.001)


def test_normalized_dict_bbox_matches_scaled_caption_geometry_without_rewrite() -> None:
    """验证 dict 归一化框与千倍坐标使用相同标题几何且不回写 bbox。"""
    companion = {
        "index": 1,
        "type": BlockType.TEXT,
        "bbox": [0.02, 0.125, 0.1, 0.145],
        "content": "unrelated text",
        "_lines": _line_metadata(1),
    }
    normalized_blocks = [
        {
            "index": 0,
            "type": BlockType.CAPTION,
            "bbox": [0.7, 0.1, 0.95, 0.12],
            "content": "Table 1",
            "_lines": _line_metadata(1),
        },
        companion,
        {
            "index": 2,
            "type": BlockType.TABLE_BODY,
            "bbox": [0.7, 0.15, 0.95, 0.8],
            "content": "",
        },
    ]
    original_bboxes = [list(block["bbox"]) for block in normalized_blocks]
    scaled_blocks = [
        {
            **block,
            "bbox": [value * 1000 for value in block["bbox"]],
        }
        for block in normalized_blocks
    ]

    fallback_inline_caption_fragments(normalized_blocks, VISUAL_MAIN_TYPES)
    fallback_inline_caption_fragments(scaled_blocks, VISUAL_MAIN_TYPES)

    assert companion["type"] == BlockType.TEXT
    assert scaled_blocks[1]["type"] == BlockType.TEXT
    assert [block["bbox"] for block in normalized_blocks] == original_bboxes


def test_normalized_block_bbox_matches_scaled_visual_parent_without_rewrite() -> None:
    """验证 Block 归一化框与千倍坐标选择同一视觉父块且保留原框。"""
    previous_table = Block(index=0, type=BlockType.TABLE_BODY, bbox=(0.1, 0.1, 0.4, 0.4))
    caption = Block(index=1, type=BlockType.CAPTION, bbox=(0.1, 0.42, 0.4, 0.44))
    following_table = Block(index=2, type=BlockType.TABLE_BODY, bbox=(0.1, 0.8, 0.4, 0.95))
    normalized_blocks = [previous_table, caption, following_table]
    original_bboxes = [block.bbox for block in normalized_blocks]
    scaled_blocks = [
        Block(
            index=block.index,
            type=block.type,
            bbox=tuple(value * 1000 for value in block.bbox),
        )
        for block in normalized_blocks
    ]

    normalized_parent = find_best_visual_parent(
        caption,
        [previous_table, following_table],
        normalized_blocks,
        {block.index: position for position, block in enumerate(normalized_blocks)},
    )
    scaled_parent = find_best_visual_parent(
        scaled_blocks[1],
        [scaled_blocks[0], scaled_blocks[2]],
        scaled_blocks,
        {block.index: position for position, block in enumerate(scaled_blocks)},
    )

    assert normalized_parent is previous_table
    assert scaled_parent is scaled_blocks[0]
    assert [block.bbox for block in normalized_blocks] == original_bboxes


def test_normalized_block_bbox_matches_scaled_visual_gap_geometry() -> None:
    """验证归一化框在视觉间隔、相交和重叠判断中与千倍坐标一致。"""
    child = Block(index=0, type=BlockType.CAPTION, bbox=(0.1, 0.1, 0.4, 0.2))
    inside_gap = Block(index=1, type=BlockType.TEXT, bbox=(0.8, 0.3, 0.9, 0.4))
    outside_gap = Block(index=2, type=BlockType.TEXT, bbox=(0.8, 0.8, 0.9, 0.9))
    main = Block(index=3, type=BlockType.TABLE_BODY, bbox=(0.1, 0.6, 0.4, 0.7))
    normalized_blocks = [child, inside_gap, outside_gap, main]
    scaled_blocks = [
        Block(
            index=block.index,
            type=block.type,
            bbox=tuple(value * 1000 for value in block.bbox),
        )
        for block in normalized_blocks
    ]

    assert is_block_outside_visual_gap(inside_gap, child, main) is False
    assert is_block_outside_visual_gap(outside_gap, child, main) is True
    assert is_block_outside_visual_gap(scaled_blocks[1], scaled_blocks[0], scaled_blocks[3]) is False
    assert is_block_outside_visual_gap(scaled_blocks[2], scaled_blocks[0], scaled_blocks[3]) is True


def test_dict_inline_caption_fragment_uses_common_fields() -> None:
    """验证 dict 同行标题片段可原地改成通用 caption。"""
    companion = {
        "index": 1,
        "type": BlockType.TEXT,
        "bbox": [40.0, 0.0, 80.0, 10.0],
        "content": "continued caption",
        "_lines": _line_metadata(1),
    }
    blocks = [
        {
            "index": 0,
            "type": BlockType.CAPTION,
            "bbox": [0.0, 0.0, 40.0, 10.0],
            "content": "Table 1",
            "_lines": _line_metadata(1),
        },
        companion,
        {
            "index": 2,
            "type": BlockType.TABLE_BODY,
            "bbox": [0.0, 15.0, 80.0, 60.0],
            "content": "",
        },
    ]

    fallback_inline_caption_fragments(blocks, VISUAL_MAIN_TYPES)

    assert companion["type"] == BlockType.CAPTION
    assert "merge_prev" not in companion


def test_dict_stacked_caption_fragment_uses_line_metadata() -> None:
    """验证堆叠标题片段根据 dict 的 _lines 区分单行和多行。"""
    single_line = {
        "index": 1,
        "type": BlockType.TEXT,
        "bbox": [0.0, 12.0, 80.0, 22.0],
        "content": "single line",
        "_lines": _line_metadata(1),
    }
    blocks = [
        {
            "index": 0,
            "type": BlockType.CAPTION,
            "bbox": [0.0, 0.0, 80.0, 10.0],
            "content": "Table 1",
            "_lines": _line_metadata(1),
        },
        single_line,
        {
            "index": 2,
            "type": BlockType.TABLE_BODY,
            "bbox": [0.0, 25.0, 80.0, 60.0],
            "content": "",
        },
    ]

    fallback_inline_caption_fragments(blocks, VISUAL_MAIN_TYPES)

    assert single_line["type"] == BlockType.CAPTION
    single_line["type"] = BlockType.TEXT
    single_line["_lines"] = _line_metadata(2)

    fallback_inline_caption_fragments(blocks, VISUAL_MAIN_TYPES)

    assert single_line["type"] == BlockType.TEXT


def test_dict_leading_table_continuation_reads_top_level_content() -> None:
    """验证 dict 页首续表使用顶层 content 和 _lines 完成判断。"""
    continuation = {
        "index": 0,
        "type": BlockType.TEXT,
        "bbox": [0.0, 0.0, 80.0, 10.0],
        "content": "Table 1 (continued)",
        "_lines": _line_metadata(1),
        "merge_prev": True,
    }
    blocks = [
        continuation,
        {
            "index": 1,
            "type": BlockType.TABLE_BODY,
            "bbox": [0.0, 15.0, 80.0, 60.0],
            "content": "",
        },
    ]

    fallback_leading_table_continuation_captions(blocks, VISUAL_MAIN_TYPES)

    assert continuation["type"] == BlockType.CAPTION
    assert continuation["merge_prev"] is True


def test_block_class_fallbacks_keep_attribute_access_compatibility() -> None:
    """验证两个 fallback 仍兼容使用属性和 Line 的 Block class。"""
    inline_companion = Block(
        index=1,
        type=BlockType.TEXT,
        bbox=(40.0, 0.0, 80.0, 10.0),
        lines=[Line(bbox=(40.0, 0.0, 80.0, 10.0))],
        content="continued caption",
        merge_prev=True,
    )
    inline_blocks = [
        Block(index=0, type=BlockType.CAPTION, bbox=(0.0, 0.0, 40.0, 10.0), content="Table 1"),
        inline_companion,
        Block(index=2, type=BlockType.TABLE_BODY, bbox=(0.0, 15.0, 80.0, 60.0)),
    ]
    fallback_inline_caption_fragments(inline_blocks, VISUAL_MAIN_TYPES)

    continuation = Block(
        index=0,
        type=BlockType.TEXT,
        bbox=(0.0, 0.0, 80.0, 10.0),
        lines=[Line(bbox=(0.0, 0.0, 80.0, 10.0))],
        content="Table 1 (continued)",
        merge_prev=True,
    )
    continuation_blocks = [
        continuation,
        Block(index=1, type=BlockType.TABLE_BODY, bbox=(0.0, 15.0, 80.0, 60.0)),
    ]
    fallback_leading_table_continuation_captions(continuation_blocks, VISUAL_MAIN_TYPES)

    assert inline_companion.type == BlockType.CAPTION
    assert inline_companion.merge_prev is True
    assert continuation.type == BlockType.CAPTION
    assert continuation.merge_prev is True
