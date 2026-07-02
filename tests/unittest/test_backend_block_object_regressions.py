from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

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
