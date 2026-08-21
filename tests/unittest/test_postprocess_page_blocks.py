from __future__ import annotations

from mineru.backend.postprocess.lists import fix_pdf_list_blocks
from mineru.backend.postprocess.page_blocks import process_page_blocks
from mineru.types import BlockType


def test_page_blocks_keep_canonical_equation_type_and_clean_content() -> None:
    """验证单页后处理清理行间公式内容时不改写 equation 类型。"""
    equation = {
        "type": BlockType.EQUATION,
        "content": r"\[x+1\]",
    }

    blocks = process_page_blocks([equation], use_bbox=False)

    assert blocks == [
        {
            "type": BlockType.EQUATION,
            "content": "x+1",
            "index": 0,
        }
    ]


def test_fix_pdf_list_blocks_supports_unit_bbox_without_rewrite() -> None:
    """验证归一化列表框可与像素文本框计算包含关系且不回写 bbox。"""
    list_bbox = [0.0, 0.0, 1.0, 1.0]
    text_bbox = [100, 100, 200, 200]
    list_block = {
        "type": BlockType.LIST,
        "bbox": list_bbox,
        "content": "",
    }
    text_block = {
        "type": BlockType.TEXT,
        "bbox": text_bbox,
        "content": "list item",
    }

    list_blocks, text_blocks, ref_text_blocks = fix_pdf_list_blocks(
        [list_block],
        [text_block],
        [],
    )

    assert list_blocks == [list_block]
    assert text_blocks == []
    assert ref_text_blocks == []
    assert list_block["content"] == [text_block]
    assert list_block["sub_type"] == BlockType.TEXT
    assert list_bbox == [0.0, 0.0, 1.0, 1.0]
    assert text_bbox == [100, 100, 200, 200]


def test_page_blocks_group_bbox_dict_visual_blocks() -> None:
    """验证单页后处理可将带 bbox 的 dict 视觉块完成分组。"""
    blocks = process_page_blocks(
        [
            {
                "type": BlockType.IMAGE_CAPTION,
                "bbox": [0.0, 0.0, 80.0, 10.0],
                "content": "Figure 1",
            },
            {
                "type": BlockType.IMAGE,
                "bbox": [0.0, 15.0, 80.0, 60.0],
                "image_base64": "data:image/jpeg;base64,image",
            },
        ]
    )

    image_blocks = [block for block in blocks if block["type"] == BlockType.IMAGE]
    assert len(image_blocks) == 1
    image_block = image_blocks[0]
    assert isinstance(image_block, dict)
    assert [block["type"] for block in image_block["content"]] == [
        BlockType.IMAGE_CAPTION,
        BlockType.IMAGE_BODY,
    ]


def test_page_blocks_group_no_bbox_office_caption_by_prefix() -> None:
    """验证单页后处理使用 Office 前缀规则分组无 bbox 视觉块。"""
    blocks = process_page_blocks(
        [
            {
                "type": BlockType.IMAGE,
                "image_base64": "data:image/jpeg;base64,image",
            },
            {
                "type": BlockType.TEXT,
                "content": "Figure 1",
            },
        ]
    )

    assert not [block for block in blocks if block["type"] == BlockType.TEXT]
    image_blocks = [block for block in blocks if block["type"] == BlockType.IMAGE]
    assert len(image_blocks) == 1
    image_block = image_blocks[0]
    assert [block["type"] for block in image_block["content"]] == [
        BlockType.IMAGE_BODY,
        BlockType.IMAGE_CAPTION,
    ]


def test_page_blocks_group_no_bbox_chart_and_code_captions() -> None:
    """验证无 bbox 的 chart/code caption 映射及 code subtype 保留。"""
    blocks = process_page_blocks(
        [
            {"type": BlockType.CHART_CAPTION, "content": "Chart 1"},
            {"type": BlockType.CHART, "content": "<div>chart</div>"},
            {"type": BlockType.CODE_CAPTION, "content": "Algorithm 1"},
            {"type": BlockType.CODE, "content": "print('ok')"},
        ]
    )

    chart_block = next(block for block in blocks if block["type"] == BlockType.CHART)
    assert [block["type"] for block in chart_block["content"]] == [
        BlockType.CHART_CAPTION,
        BlockType.CHART_BODY,
    ]
    code_block = next(block for block in blocks if block["type"] == BlockType.CODE)
    assert code_block["sub_type"] == "code"
    assert code_block["guess_lang"]
    assert [block["type"] for block in code_block["content"]] == [
        BlockType.CODE_CAPTION,
        BlockType.CODE_BODY,
    ]
