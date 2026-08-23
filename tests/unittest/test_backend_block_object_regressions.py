from typing import Any

from mineru.backend.postprocess.pages import model_json_to_pages
from mineru.types import BlockType, ImageBlock, ModelJson


def _model_json(pages: list[list[dict[str, Any]]]) -> ModelJson:
    """为 raw block 对象化回归构造最小严格 ModelJson。"""
    return ModelJson(
        pages=pages,
        page_index_map=[],
        file_suffix="pdf",
        effort="flash",
        parse_mode="txt",
        mineru_version="test",
    )


def test_postprocess_groups_pdf_visual_blocks_without_dict_access() -> None:
    """验证 PDF raw 视觉块会在严格对象化边界完成分组。"""
    page = model_json_to_pages(
        _model_json(
            [
                [
                    {
                        "type": BlockType.IMAGE_CAPTION,
                        "bbox": [0.05, 0.05, 0.6, 0.12],
                        "content": "Figure 1",
                    },
                    {
                        "type": BlockType.IMAGE,
                        "bbox": [0.05, 0.15, 0.6, 0.45],
                    },
                ]
            ]
        )
    )[0]

    assert len(page.blocks) == 1
    assert isinstance(page.blocks[0], ImageBlock)
    assert {block.type for block in page.blocks[0].content} == {
        BlockType.IMAGE_BODY,
        BlockType.IMAGE_CAPTION,
    }


def test_postprocess_groups_raw_vision_footnote_as_image_footnote() -> None:
    """验证通用 raw footnote 会归入相邻图片而不是页脚注。"""
    page = model_json_to_pages(
        _model_json(
            [
                [
                    {"type": BlockType.IMAGE, "bbox": [0.1, 0.1, 0.6, 0.45]},
                    {"type": "footnote", "bbox": [0.12, 0.47, 0.6, 0.6], "content": "Figure note"},
                ]
            ]
        )
    )[0]

    assert len(page.blocks) == 1
    assert isinstance(page.blocks[0], ImageBlock)
    assert {block.type for block in page.blocks[0].content} == {
        BlockType.IMAGE_BODY,
        BlockType.IMAGE_FOOTNOTE,
    }
