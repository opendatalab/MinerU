from __future__ import annotations

import pytest

from mineru.config import LatexDelimitersConfig, config
from mineru.render._internal.markdown.blocks import render_single_block
from mineru.render.image import image_path_renderer
from mineru.types import (
    BlockBase,
    BlockType,
    ChartBlock,
    ChartBodyBlock,
    EquationBlock,
    ImageAnnotationBlock,
    ImageBlock,
    ImageBodyBlock,
    TableBlock,
    TableBodyBlock,
)


def _delimiters() -> LatexDelimitersConfig:
    return config.render.latex_delimiters


def _image_block() -> ImageBlock:
    body = ImageBodyBlock(
        type=BlockType.IMAGE_BODY,
        index=0,
        bbox=(0.0, 0.0, 0.1, 0.1),
        content="",
        image_path="internal/hash.jpg",
    )
    caption = ImageAnnotationBlock(
        type=BlockType.IMAGE_CAPTION,
        index=0,
        bbox=(0.0, 0.0, 0.1, 0.01),
        content="Figure caption",
    )
    return ImageBlock(
        type=BlockType.IMAGE,
        index=0,
        bbox=(0.0, 0.0, 0.1, 0.1),
        content=[body, caption],
    )


def test_image_path_renderer_preserves_existing_relative_path_behavior() -> None:
    block = _image_block()

    assert image_path_renderer(block, img_bucket_path="images") == "![](images/internal/hash.jpg)"


def test_pipeline_markdown_uses_custom_block_image_renderer_and_keeps_caption() -> None:
    block = _image_block()

    rendered = render_single_block(
        block,
        delimiters=_delimiters(),
        asset_base_url="",
        image_renderer=lambda _block: "![Image block](doc:aaaaaaa/tier:standard/page:1/block:1)",
    )

    assert "![Image block](doc:aaaaaaa/tier:standard/page:1/block:1)" in rendered
    assert "Figure caption" in rendered
    assert "internal/hash.jpg" not in rendered


def test_office_markdown_uses_custom_block_image_renderer_and_keeps_caption() -> None:
    block = _image_block()

    rendered = render_single_block(
        block,
        delimiters=_delimiters(),
        asset_base_url="",
        image_renderer=lambda _block: "![Image block]()",
    )

    assert "![Image block]()" in rendered
    assert "Figure caption" in rendered
    assert "internal/hash.jpg" not in rendered


@pytest.mark.parametrize("block_type", [BlockType.TABLE, BlockType.CHART])
def test_pipeline_markdown_uses_custom_renderer_for_image_only_visual_blocks(
    block_type: BlockType,
) -> None:
    if block_type == BlockType.TABLE:
        block: BlockBase = TableBlock(
            type=BlockType.TABLE,
            index=0,
            bbox=(0.0, 0.0, 0.1, 0.1),
            content=[
                TableBodyBlock(
                    type=BlockType.TABLE_BODY,
                    index=0,
                    bbox=(0.0, 0.0, 0.1, 0.1),
                    content="",
                    image_path="internal/hash.jpg",
                )
            ],
        )
    else:
        block = ChartBlock(
            type=BlockType.CHART,
            index=0,
            bbox=(0.0, 0.0, 0.1, 0.1),
            content=[
                ChartBodyBlock(
                    type=BlockType.CHART_BODY,
                    index=0,
                    bbox=(0.0, 0.0, 0.1, 0.1),
                    content="",
                    image_path="internal/hash.jpg",
                )
            ],
        )

    rendered = render_single_block(block, delimiters=_delimiters(), asset_base_url="", image_renderer=lambda _block: "![Visual block](doc:locator)")

    assert rendered == "![Visual block](doc:locator)"


def test_pipeline_markdown_uses_custom_renderer_for_image_only_formula() -> None:
    block = EquationBlock(
        type=BlockType.EQUATION,
        index=0,
        bbox=(0.0, 0.0, 0.1, 0.1),
        content="",
        image_path="internal/hash.jpg",
    )

    rendered = render_single_block(block, delimiters=_delimiters(), asset_base_url="", image_renderer=lambda _block: "![Formula block](doc:locator)")

    assert rendered == "![Formula block](doc:locator)"


def test_custom_renderer_removes_internal_images_from_structured_table_html() -> None:
    internal_path = "internal/cell-image.png"
    html = f'<table><tr><td>Text<img src="{internal_path}"></td></tr></table>'
    block = TableBlock(
        type=BlockType.TABLE,
        index=0,
        bbox=(0.0, 0.0, 0.1, 0.1),
        content=[
            TableBodyBlock(
                type=BlockType.TABLE_BODY,
                index=0,
                bbox=(0.0, 0.0, 0.1, 0.1),
                content=html,
            )
        ],
    )

    rendered = render_single_block(
        block,
        delimiters=_delimiters(),
        asset_base_url="",
        image_renderer=lambda _block: "![Table block](doc:locator)",
    )

    assert "Text" in rendered
    assert internal_path not in rendered
    assert "<img" not in rendered
