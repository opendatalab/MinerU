from __future__ import annotations

from types import SimpleNamespace

from PIL import Image

from mineru.backend.hybrid.model_output_to_middle_json import blocks_to_page_info as hybrid_blocks_to_page_info
from mineru.backend.pipeline.model_output_to_middle_json import (
    blocks_to_page_info as pipeline_blocks_to_page_info,
    finalize_middle_json_from_preproc as finalize_pipeline_middle_json_from_preproc,
)
from mineru.types import BlockType


def _fake_pdf_page(width: int = 200, height: int = 200) -> SimpleNamespace:
    return SimpleNamespace(size=(width, height))


def _image_dict(width: int = 200, height: int = 200) -> dict[str, object]:
    return {"scale": 1.0, "img_pil": Image.new("RGB", (width, height), "white")}


def test_pipeline_block_objects_survive_visual_grouping_and_finalize() -> None:
    page_info = pipeline_blocks_to_page_info(
        {
            "layout_dets": [
                {
                    "label": "figure_title",
                    "bbox": [10, 10, 120, 25],
                    "index": 1,
                    "score": 0.99,
                    "text": "Figure 1",
                },
                {
                    "label": "image",
                    "bbox": [10, 30, 120, 90],
                    "index": 2,
                    "score": 0.99,
                },
            ]
        },
        _image_dict(),
        _fake_pdf_page(),
        0,
        ocr_enable=True,
    )

    finalize_pipeline_middle_json_from_preproc([page_info])

    assert len(page_info.preproc_blocks) == 1
    visual_block = page_info.preproc_blocks[0]
    assert visual_block.type == BlockType.IMAGE
    assert {block.type for block in visual_block.blocks} == {BlockType.IMAGE_BODY, BlockType.IMAGE_CAPTION}
    assert page_info.para_blocks[0].type == BlockType.IMAGE


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
