from pathlib import Path

from mineru.backend.office.office_magic_model import fix_two_layer_blocks
from mineru.backend.utils import office_image
from mineru.parser.html import HtmlParser
from mineru.types import EMPTY_BBOX, Block, BlockType, ContentType, Line, Span


def test_html_parser_populates_required_bbox_and_index_fields(tmp_path: Path) -> None:
    html_file = tmp_path / "sample.html"
    html_file.write_text(
        "<html><body><h1>Hello</h1><p>World</p></body></html>", encoding="utf-8"
    )

    result = HtmlParser().parse(html_file)

    blocks = result.pages[0].para_blocks
    assert [block.index for block in blocks] == [0, 1]
    assert [block.bbox for block in blocks] == [EMPTY_BBOX, EMPTY_BBOX]
    assert all(line.bbox == EMPTY_BBOX for block in blocks for line in block.lines)
    assert all(
        span.bbox == EMPTY_BBOX
        for block in blocks
        for line in block.lines
        for span in line.spans
    )


def test_office_two_layer_blocks_accept_block_dataclasses() -> None:
    caption = Block(
        index=0,
        type=BlockType.IMAGE_CAPTION,
        bbox=EMPTY_BBOX,
        lines=[
            Line(
                bbox=EMPTY_BBOX,
                spans=[
                    Span(type=ContentType.TEXT, bbox=EMPTY_BBOX, content="Figure 1")
                ],
            )
        ],
    )
    body = Block(
        index=1,
        type=BlockType.IMAGE_BODY,
        bbox=EMPTY_BBOX,
        lines=[
            Line(
                bbox=EMPTY_BBOX,
                spans=[
                    Span(type=ContentType.IMAGE, bbox=EMPTY_BBOX, image_path="figure.png")
                ],
            )
        ],
    )

    fixed_blocks, not_included = fix_two_layer_blocks([caption, body], BlockType.IMAGE)

    assert not_included == []
    assert len(fixed_blocks) == 1
    assert fixed_blocks[0].type == BlockType.IMAGE
    assert [block.type for block in fixed_blocks[0].blocks] == [
        BlockType.IMAGE_CAPTION,
        BlockType.IMAGE_BODY,
    ]


def test_vector_image_part_skip_log_is_debug(monkeypatch) -> None:
    class _Logger:
        def __init__(self) -> None:
            self.debug_messages: list[str] = []
            self.warning_messages: list[str] = []

        def debug(self, message: str) -> None:
            self.debug_messages.append(message)

        def warning(self, message: str) -> None:
            self.warning_messages.append(message)

    fake_logger = _Logger()
    monkeypatch.setattr(office_image, "logger", fake_logger)
    monkeypatch.setattr(
        office_image,
        "get_standard_vector_placeholder_data_uri",
        lambda: "data:image/jpeg;base64,placeholder",
    )

    assert (
        office_image.serialize_vector_part_with_placeholder(
            "/word/media/image3.wmf", "image/x-wmf"
        )
        == "data:image/jpeg;base64,placeholder"
    )
    assert len(fake_logger.debug_messages) == 1
    assert "Skipping WMF image part before Pillow load" in fake_logger.debug_messages[0]
    assert fake_logger.warning_messages == []
