import base64
from dataclasses import fields
import json

import pytest

from mineru.parser.base import ParseResult
from mineru.parser import MIDDLE_JSON_SCHEMA_VERSION
from mineru.types import BlockType, MiddleJson, PageInfo, TableBlock, TableBodyBlock, TextBlock
from mineru.utils.image_payload import ImagePayloadCache
from mineru.version import __version__


def _data_uri(payload: bytes, image_type: str = "png") -> str:
    return f"data:image/{image_type};base64,{base64.b64encode(payload).decode('ascii')}"


def _table_page_with_cached_inline_image(img_bytes: bytes) -> tuple[PageInfo, ImagePayloadCache, str]:
    """构造已完成图片外置化的表格页，验证 ParseResult 只承载顶层图片缓存。"""
    image_cache = ImagePayloadCache()
    inline_image = _data_uri(img_bytes)
    html = image_cache.replace_html_data_uri_sources(
        f'<table><tr><td><img src="{inline_image}"/></td></tr></table>'
    )
    page = PageInfo(
        page_idx=0,
        blocks=[
            TableBlock(
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
        ],
    )
    return page, image_cache, inline_image


def test_parse_result_does_not_expose_backend_version_or_file_name() -> None:
    field_names = {field.name for field in fields(ParseResult)}

    assert "_backend" not in field_names
    assert "_version_name" not in field_names
    assert "_file_name" not in field_names

    page_field_names = set(PageInfo.model_fields)
    assert "page_size" not in page_field_names
    assert "para_blocks" not in page_field_names
    assert "_backend" not in page_field_names


def test_parse_result_from_dict_restores_pages() -> None:
    result = ParseResult(
        middle_json=MiddleJson(
            pages=[
            PageInfo(
                page_idx=3,
                blocks=[
                    TextBlock(
                        type=BlockType.TEXT,
                        index=0,
                        bbox=(0.0, 0.0, 0.1, 0.1),
                        content="hello",
                    )
                ],
            )
        ],
            file_suffix="pdf", effort="medium", parse_mode="txt", mineru_version=__version__,
        )
        )

    restored = ParseResult.from_dict(result.to_dict())

    assert restored.to_dict() == result.to_dict()
    assert restored.pages[0].page_idx == 3
    assert restored.pages[0].blocks[0].bbox == (0.0, 0.0, 0.1, 0.1)
    assert restored.pages[0].blocks[0].content == "hello"


def test_parse_result_to_dict_includes_schema_version_without_meta() -> None:
    result = ParseResult(middle_json=MiddleJson(pages=[PageInfo(page_idx=0)], file_suffix="pdf", effort="medium", parse_mode="txt", mineru_version=__version__))

    payload = result.to_dict()

    assert payload["schema_version"] == MIDDLE_JSON_SCHEMA_VERSION
    assert "pages" in payload
    assert "_meta" not in payload


def test_parse_result_from_dict_rejects_missing_pages() -> None:
    with pytest.raises(ValueError, match="pages"):
        ParseResult.from_dict({"pdf_info": []})


def test_parse_result_from_json_restores_pages() -> None:
    data = {
        "pages": [
            {
                "page_idx": 0,
                "para_blocks": [
                    {
                        "index": 0,
                        "type": "text",
                        "bbox": [0.0, 0.0, 0.0, 0.0],
                        "lines": [
                            {
                                "bbox": [0.0, 0.0, 0.0, 0.0],
                                "spans": [{"type": "text", "bbox": [0.0, 0.0, 0.0, 0.0], "content": "round trip"}],
                            }
                        ],
                    }
                ],
            }
        ]
    }

    restored = ParseResult.from_json(json.dumps(data))

    assert restored.to_dict() == ParseResult.from_dict(data).to_dict()


def test_parse_result_export_pages_returns_defensive_copy_from_cache() -> None:
    img_bytes = b"defensive-table-image"
    page, image_cache, inline_image = _table_page_with_cached_inline_image(img_bytes)
    result = ParseResult(middle_json=MiddleJson(pages=[page], file_suffix="pdf", effort="medium", parse_mode="txt", mineru_version=__version__), _image_cache=image_cache)
    first_export = result.export_pages()
    first_export[0].blocks[0].content[0].content = "mutated by caller"

    second_export = result.export_pages()
    second_content = second_export[0].blocks[0].content[0].content
    exported_json = json.dumps(result.to_dict(), ensure_ascii=False)

    assert "mutated by caller" not in second_content
    assert "mutated by caller" not in exported_json
    assert inline_image not in second_content
    assert inline_image not in exported_json
    assert next(iter(result.images())) in exported_json


def test_parse_result_export_rewrites_inline_table_base64_images() -> None:
    img_bytes = b"table-image"
    page, image_cache, inline_image = _table_page_with_cached_inline_image(img_bytes)

    result = ParseResult(middle_json=MiddleJson(pages=[page], file_suffix="pdf", effort="medium", parse_mode="txt", mineru_version=__version__), _image_cache=image_cache)
    images = result.images()
    exported_page = result.export_pages()[0]
    exported_body = exported_page.blocks[0].content[0]
    exported_json = json.dumps(result.to_dict(), ensure_ascii=False)

    assert list(images.values()) == [img_bytes]
    assert inline_image not in exported_json
    assert 'src="' in exported_body.content
    assert next(iter(images)) in exported_json
