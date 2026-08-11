from __future__ import annotations

from mineru.backend.utils.visual_magic_model_utils import (
    VISUAL_MAIN_TYPES,
    fallback_inline_caption_fragments,
    fallback_leading_table_continuation_captions,
)
from mineru.types import Block, BlockType, Line


def _line_metadata(count: int) -> list[dict[str, list[float]]]:
    """构造仅包含 bbox 的 dict 行级元数据。"""
    return [{"bbox": [0.0, float(index), 1.0, float(index + 1)]} for index in range(count)]


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
