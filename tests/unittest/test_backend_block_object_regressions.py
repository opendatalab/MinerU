from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from mineru.backend.hybrid import hybrid_analyze
from mineru.backend.hybrid.model_output_to_middle_json import blocks_to_page_info as hybrid_blocks_to_page_info
from mineru.types import BlockType


def _fake_pdf_page(width: int = 200, height: int = 200) -> SimpleNamespace:
    return SimpleNamespace(size=(width, height))


def _image_dict(width: int = 200, height: int = 200) -> dict[str, object]:
    return {"scale": 1.0, "img_pil": Image.new("RGB", (width, height), "white")}


def test_hybrid_block_objects_sort_and_group_without_dict_access() -> None:
    page_info = hybrid_blocks_to_page_info(
        [
            {
                "type": "image_caption",
                "bbox": [0.05, 0.05, 0.6, 0.12],
                "content": "Figure 1",
            },
            {
                "type": "image",
                "bbox": [0.05, 0.15, 0.6, 0.45],
            },
        ],
        _image_dict(),
        _fake_pdf_page(),
        0,
        _ocr_enable=True,
        _vlm_ocr_enable=True,
    )

    assert len(page_info.preproc_blocks) == 1
    visual_block = page_info.preproc_blocks[0]
    assert visual_block.type == BlockType.IMAGE
    assert {block.type for block in visual_block.blocks} == {BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION}


def test_hybrid_medium_vision_footnote_groups_as_image_footnote() -> None:
    """medium 的 vision_footnote 应作为视觉脚注挂到相邻 image 下，而不是进入 page footnote。"""
    page_image = Image.new("RGB", (200, 200), "white")
    model_list = hybrid_analyze._build_medium_hybrid_model_list(
        [
            [
                {"label": "image", "bbox": [20, 20, 120, 90], "score": 0.9},
                {"label": "vision_footnote", "bbox": [24, 94, 120, 120], "text": "Figure note", "score": 0.9},
            ]
        ],
        [page_image],
    )

    page_info = hybrid_blocks_to_page_info(
        model_list[0],
        _image_dict(),
        _fake_pdf_page(),
        0,
        _ocr_enable=True,
        _vlm_ocr_enable=False,
    )

    assert len(page_info.preproc_blocks) == 1
    visual_block = page_info.preproc_blocks[0]
    assert visual_block.type == BlockType.IMAGE
    assert {block.type for block in visual_block.blocks} == {BlockType.IMAGE_BODY, BlockType.IMAGE_FOOTNOTE}
