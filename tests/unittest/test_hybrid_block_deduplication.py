# Copyright (c) Opendatalab. All rights reserved.
from copy import deepcopy

from mineru.backend.hybrid.hybrid_block_dedup import (
    deduplicate_list_sub_blocks,
    deduplicate_vlm_model_blocks,
)
from mineru.backend.hybrid.hybrid_magic_model import fix_list_blocks


def _middle_text_block(
    content: str,
    bbox: list[float | int],
    block_index: int,
) -> dict:
    return {
        "bbox": bbox,
        "type": "text",
        "angle": 0,
        "lines": [
            {
                "bbox": bbox,
                "spans": [
                    {
                        "bbox": bbox,
                        "type": "text",
                        "content": content,
                    }
                ],
            }
        ],
        "index": block_index,
    }


def _middle_block_text(block: dict) -> str:
    return block["lines"][0]["spans"][0]["content"]


def test_deduplicate_vlm_model_blocks_removes_repeated_text_and_list_blocks() -> None:
    option_b = {
        "type": "text",
        "bbox": [0.034, 0.868, 0.246, 0.892],
        "content": "B、色谱峰峰高",
    }
    list_block = {
        "type": "list",
        "bbox": [0.032, 0.840, 0.342, 0.976],
    }
    same_text_at_another_position = {
        **option_b,
        "bbox": [0.034, 0.668, 0.246, 0.692],
    }
    different_text_at_same_position = {
        **option_b,
        "content": "B、色谱峰峰宽",
    }
    different_list_at_same_position = {
        **list_block,
        "content": "different list",
    }

    result = deduplicate_vlm_model_blocks(
        [
            option_b,
            list_block,
            *[deepcopy(list_block) for _ in range(4)],
            deepcopy(option_b),
            same_text_at_another_position,
            different_text_at_same_position,
            different_list_at_same_position,
        ]
    )

    assert result == [
        option_b,
        list_block,
        same_text_at_another_position,
        different_text_at_same_position,
        different_list_at_same_position,
    ]


def test_deduplicate_list_sub_blocks_removes_only_semantic_geometric_duplicates() -> None:
    option_b = _middle_text_block(
        "B、色谱峰峰高",
        [14, 561, 107, 577],
        11,
    )
    duplicate_option_b = _middle_text_block(
        " B、色谱峰峰高 ",
        [14.2, 561, 107.2, 577],
        20,
    )
    same_text_at_another_position = _middle_text_block(
        "B、色谱峰峰高",
        [14, 661, 107, 677],
        21,
    )
    different_text_at_same_position = _middle_text_block(
        "B、色谱峰峰宽",
        [14, 561, 107, 577],
        22,
    )
    empty_block = _middle_text_block("", [14, 561, 107, 577], 23)
    duplicate_empty_block = deepcopy(empty_block)
    duplicate_empty_block["index"] = 24

    result = deduplicate_list_sub_blocks(
        [
            option_b,
            duplicate_option_b,
            same_text_at_another_position,
            different_text_at_same_position,
            empty_block,
            duplicate_empty_block,
        ]
    )

    assert result == [
        option_b,
        same_text_at_another_position,
        different_text_at_same_position,
        empty_block,
        duplicate_empty_block,
    ]


def test_fix_list_blocks_removes_repeated_options_from_grouped_list() -> None:
    option_contents = [
        "A、色谱柱理论板数",
        "B、色谱峰峰高",
        "C、色谱峰保留时间",
        "D、色谱峰分离度",
        "E、色谱峰面积重复性",
    ]
    option_bboxes = [
        [13, 543, 135, 559],
        [14, 561, 107, 577],
        [14, 579, 134, 595],
        [14, 598, 120, 613],
        [14, 615, 148, 631],
    ]
    options = [
        _middle_text_block(content, bbox, index)
        for index, (content, bbox) in enumerate(
            zip(option_contents, option_bboxes),
            start=10,
        )
    ]
    duplicated_options = [
        _middle_text_block(content, bbox, index)
        for index, (content, bbox) in enumerate(
            zip(option_contents[1:], option_bboxes[1:]),
            start=20,
        )
    ]
    list_blocks = [
        {
            "bbox": [13, 543, 148, 631],
            "type": "list",
            "lines": [],
        }
    ]

    fixed_lists, remaining_text, remaining_ref_text = fix_list_blocks(
        list_blocks,
        options + duplicated_options,
        [],
    )

    fixed_contents = [_middle_block_text(block) for block in fixed_lists[0]["blocks"]]
    assert fixed_contents == option_contents
    assert remaining_text == []
    assert remaining_ref_text == []
