from __future__ import annotations

import pytest

from mineru.render.image import image_path_renderer
from mineru.render.markdown import blocks_to_markdown
from mineru.render.office.output import blocks_to_markdown as office_blocks_to_markdown
from mineru.types import Block, BlockType, ContentType, Line, Span


def _image_block(*, image_path: str = "internal/hash.jpg") -> Block:
    body = Block(
        index=0,
        type=BlockType.IMAGE_BODY,
        bbox=(1, 1, 20, 20),
        lines=[
            Line(
                bbox=(1, 1, 20, 20),
                spans=[Span(type=ContentType.IMAGE, bbox=(1, 1, 20, 20), image_path=image_path)],
            )
        ],
    )
    caption = Block(
        index=1,
        type=BlockType.IMAGE_CAPTION,
        bbox=(1, 21, 20, 25),
        lines=[
            Line(
                bbox=(1, 21, 20, 25),
                spans=[Span(type=ContentType.TEXT, bbox=(1, 21, 20, 25), content="Figure caption")],
            )
        ],
    )
    return Block(index=0, type=BlockType.IMAGE, bbox=(1, 1, 20, 25), blocks=[body, caption])


def test_image_path_renderer_preserves_existing_relative_path_behavior() -> None:
    block = _image_block()

    assert image_path_renderer(block, img_bucket_path="images") == "![](images/internal/hash.jpg)"


def test_pipeline_markdown_uses_custom_block_image_renderer_and_keeps_caption() -> None:
    block = _image_block()

    rendered = blocks_to_markdown(
        [block],
        image_renderer=lambda _block: "![Image block](doc:aaaaaaa/tier:high/page:1/block:1)",
    )

    assert len(rendered) == 1
    assert "![Image block](doc:aaaaaaa/tier:high/page:1/block:1)" in rendered[0]
    assert "Figure caption" in rendered[0]
    assert "internal/hash.jpg" not in rendered[0]


def test_office_markdown_uses_custom_block_image_renderer_and_keeps_caption() -> None:
    block = _image_block()

    rendered = office_blocks_to_markdown(
        [block],
        image_renderer=lambda _block: "![Image block]()",
    )

    assert len(rendered) == 1
    assert "![Image block]()" in rendered[0]
    assert "Figure caption" in rendered[0]
    assert "internal/hash.jpg" not in rendered[0]


@pytest.mark.parametrize(
    ("block_type", "body_type", "span_type"),
    [
        (BlockType.TABLE, BlockType.TABLE_BODY, ContentType.TABLE),
        (BlockType.CHART, BlockType.CHART_BODY, ContentType.CHART),
    ],
)
def test_pipeline_markdown_uses_custom_renderer_for_image_only_visual_blocks(
    block_type: str,
    body_type: str,
    span_type: str,
) -> None:
    body = Block(
        index=0,
        type=body_type,
        bbox=(1, 1, 20, 20),
        lines=[
            Line(
                bbox=(1, 1, 20, 20),
                spans=[Span(type=span_type, bbox=(1, 1, 20, 20), image_path="internal/hash.jpg")],
            )
        ],
    )
    block = Block(index=0, type=block_type, bbox=(1, 1, 20, 20), blocks=[body])

    rendered = blocks_to_markdown([block], image_renderer=lambda _block: "![Visual block](doc:locator)")

    assert rendered == ["![Visual block](doc:locator)"]


def test_pipeline_markdown_uses_custom_renderer_for_image_only_formula() -> None:
    block = Block(
        index=0,
        type=BlockType.INTERLINE_EQUATION,
        bbox=(1, 1, 20, 20),
        lines=[
            Line(
                bbox=(1, 1, 20, 20),
                spans=[
                    Span(
                        type=ContentType.INTERLINE_EQUATION,
                        bbox=(1, 1, 20, 20),
                        image_path="internal/hash.jpg",
                    )
                ],
            )
        ],
    )

    rendered = blocks_to_markdown([block], image_renderer=lambda _block: "![Formula block](doc:locator)")

    assert rendered == ["![Formula block](doc:locator)"]


@pytest.mark.parametrize("backend", ["pipeline", "office"])
def test_custom_renderer_removes_internal_images_from_structured_table_html(backend: str) -> None:
    internal_path = "internal/cell-image.png"
    html = f'<table><tr><td>Text<img src="{internal_path}"></td></tr></table>'
    body = Block(
        index=0,
        type=BlockType.TABLE_BODY,
        bbox=(1, 1, 20, 20),
        lines=[
            Line(
                bbox=(1, 1, 20, 20),
                spans=[Span(type=ContentType.TABLE, bbox=(1, 1, 20, 20), content=html)],
            )
        ],
    )
    block = Block(index=0, type=BlockType.TABLE, bbox=(1, 1, 20, 20), blocks=[body])
    renderer = office_blocks_to_markdown if backend == "office" else blocks_to_markdown

    rendered = renderer(
        [block],
        prefer_markdown_table=True,
        image_renderer=lambda _block: "![Table block](doc:locator)",
    )

    assert len(rendered) == 1
    assert "Text" in rendered[0]
    assert internal_path not in rendered[0]
    assert "<img" not in rendered[0]
