from copy import deepcopy

from mineru.backend.model_list_to_midlle_json import model_list_to_pages
from mineru.types import BlockType


def test_model_list_to_pages_preserves_pdf_input() -> None:
    """验证 PDF 块完成规范化和分组后不会污染原始 model_list。"""
    model_list = [
        [
            {
                "type": BlockType.IMAGE_CAPTION,
                "bbox": [0, 0, 100, 10],
                "content": "Figure 1",
            },
            {
                "type": BlockType.IMAGE,
                "bbox": [0, 15, 100, 80],
                "image_base64": "data:image/jpeg;base64,image",
            },
            {
                "type": BlockType.EQUATION,
                "bbox": [0, 85, 100, 100],
                "content": r"\[x+1\]",
            },
            {
                "type": BlockType.LIST,
                "bbox": [0, 105, 100, 150],
                "content": "",
            },
            {
                "type": BlockType.TEXT,
                "bbox": [10, 110, 90, 125],
                "content": "list item",
            },
            {
                "type": BlockType.TABLE_CAPTION,
                "bbox": [0, 155, 100, 165],
                "content": "Table 1",
            },
            {
                "type": BlockType.TABLE,
                "bbox": [0, 170, 100, 240],
                "content": "<table></table>",
            },
        ]
    ]
    original_model_list = deepcopy(model_list)

    pages = model_list_to_pages(model_list, page_index_map=[7])

    assert model_list == original_model_list
    assert pages[0]["page_idx"] == 7
    blocks_by_type = {block["type"]: block for block in pages[0]["blocks"]}

    image_block = blocks_by_type[BlockType.IMAGE]
    assert [block["type"] for block in image_block["content"]] == [
        BlockType.IMAGE_CAPTION,
        BlockType.IMAGE_BODY,
    ]

    equation_block = blocks_by_type[BlockType.INTERLINE_EQUATION]
    assert equation_block["content"] == "x+1"
    assert equation_block is not model_list[0][2]

    list_block = blocks_by_type[BlockType.LIST]
    assert list_block["content"] == [
        {
            "type": BlockType.TEXT,
            "bbox": [10, 110, 90, 125],
            "content": "list item",
            "index": 4,
        }
    ]
    assert list_block["sub_type"] == BlockType.TEXT

    table_block = blocks_by_type[BlockType.TABLE]
    assert [block["type"] for block in table_block["content"]] == [
        BlockType.TABLE_CAPTION,
        BlockType.TABLE_BODY,
    ]

    equation_block["content"] = "changed"
    assert model_list == original_model_list


def test_model_list_to_pages_preserves_office_nested_input() -> None:
    """验证 Office 嵌套列表和目录规范化不会污染原始 model_list。"""
    model_list = [
        [
            {
                "type": BlockType.IMAGE,
                "_image_base64": "data:image/jpeg;base64,image",
            },
            {
                "type": BlockType.TEXT,
                "content": "Figure 1",
            },
            {
                "type": BlockType.LIST,
                "attribute": "ordered",
                "ilevel": 0,
                "start": 1,
                "content": [
                    {"type": BlockType.TEXT, "content": "first"},
                    {
                        "type": BlockType.LIST,
                        "attribute": "unordered",
                        "ilevel": 1,
                        "content": [
                            {"type": BlockType.TEXT, "content": "child"},
                        ],
                    },
                ],
            },
            {
                "type": BlockType.INDEX,
                "ilevel": 0,
                "content": [
                    {
                        "type": BlockType.INDEX,
                        "ilevel": 1,
                        "content": [
                            {"type": BlockType.TEXT, "content": "section"},
                        ],
                    },
                ],
            },
        ]
    ]
    original_model_list = deepcopy(model_list)

    pages = model_list_to_pages(model_list)

    assert model_list == original_model_list
    assert pages[0]["page_idx"] == 0
    blocks_by_type = {block["type"]: block for block in pages[0]["blocks"]}

    image_block = blocks_by_type[BlockType.IMAGE]
    assert [block["type"] for block in image_block["content"]] == [
        BlockType.IMAGE_BODY,
        BlockType.IMAGE_CAPTION,
    ]

    list_block = blocks_by_type[BlockType.LIST]
    assert list_block["content"][0]["content"] == "1. first"
    nested_list = list_block["content"][1]
    assert nested_list["content"][0]["content"] == "- child"
    for metadata_key in ("attribute", "ilevel", "start"):
        assert metadata_key not in list_block
        assert metadata_key not in nested_list
    assert list_block is not model_list[0][2]

    index_block = blocks_by_type[BlockType.INDEX]
    assert "ilevel" not in index_block
    assert "ilevel" not in index_block["content"][0]

    list_block["content"][0]["content"] = "changed"
    assert model_list == original_model_list
