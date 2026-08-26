from dataclasses import fields
import json

import pytest
from pydantic import ValidationError

from mineru.parser.base import ParseResult
from mineru.parser import MIDDLE_JSON_SCHEMA_VERSION
from mineru.types import BlockType, MiddleJson, PageAuxTextBlock, PageInfo, TableBlock, TableBodyBlock, TextBlock
from mineru.utils.image_payload import ImagePayloadCache
from mineru.version import __version__


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
            is_full_document=True,
            file_suffix="pdf", effort="medium", parse_mode="txt", mineru_version=__version__,
        )
        )

    restored = ParseResult.from_dict(result.to_dict())

    assert restored.to_dict() == result.to_dict()
    assert restored.pages[0].page_idx == 3
    assert restored.pages[0].blocks[0].bbox == (0.0, 0.0, 0.1, 0.1)
    assert restored.pages[0].blocks[0].content == "hello"


def test_parse_result_to_dict_includes_schema_version_without_meta() -> None:
    result = ParseResult(
        middle_json=MiddleJson(
            pages=[PageInfo(page_idx=0)],
            is_full_document=True,
            file_suffix="pdf",
            effort="medium",
            parse_mode="txt",
            mineru_version=__version__,
        )
    )

    payload = result.to_dict()

    assert payload["schema_version"] == MIDDLE_JSON_SCHEMA_VERSION
    assert "pages" in payload
    assert "_meta" not in payload


def test_parse_result_roundtrip_preserves_page_footnote_anchor() -> None:
    """验证 Schema 2.0 ParseResult 往返保留页面脚注 anchor。"""
    result = ParseResult(
        middle_json=MiddleJson(
            pages=[
                PageInfo(
                    page_idx=0,
                    blocks=[
                        PageAuxTextBlock(
                            type=BlockType.PAGE_FOOTNOTE,
                            index=0,
                            content="Footnote",
                            anchor="note-one",
                        )
                    ],
                )
            ],
            is_full_document=True,
            file_suffix="epub",
            effort="flash",
            parse_mode="txt",
            mineru_version=__version__,
        )
    )

    restored = ParseResult.from_dict(result.to_dict())

    footnote = restored.pages[0].blocks[0]
    assert footnote.type == BlockType.PAGE_FOOTNOTE
    assert footnote.anchor == "note-one"  # type: ignore[union-attr]


def test_parse_result_rejects_schema_v2_low_effort() -> None:
    """验证 schema 版本保持 2.0 时，旧 Low effort 载荷仍按新严格枚举失败。"""
    with pytest.raises(ValidationError, match="literal_error"):
        ParseResult.from_dict(
            {
                "schema_version": MIDDLE_JSON_SCHEMA_VERSION,
                "pages": [],
                "is_full_document": True,
                "file_suffix": "pdf",
                "effort": "low",
                "parse_mode": "ocr",
                "mineru_version": __version__,
            }
        )


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
    assert restored.middle_json.is_full_document is True
    assert restored.middle_json.file_suffix == "pdf"


def test_parse_result_export_pages_returns_defensive_copy() -> None:
    """验证调用方修改导出页面副本时不会污染 ParseResult 内部状态。"""
    image_cache = ImagePayloadCache()
    image_path = image_cache.register_bytes(
        b"defensive-table-image",
        "png",
        image_path="images/table.png",
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
                        content=f'<table><tr><td><img src="{image_path}"/></td></tr></table>',
                    )
                ],
            )
        ],
    )
    result = ParseResult(
        middle_json=MiddleJson(
            pages=[page],
            is_full_document=True,
            file_suffix="pdf",
            effort="medium",
            parse_mode="txt",
            mineru_version=__version__,
        ),
        _image_cache=image_cache,
    )
    first_export = result.export_pages()
    first_export[0].blocks[0].content[0].content = "mutated by caller"

    second_export = result.export_pages()
    second_content = second_export[0].blocks[0].content[0].content
    exported_json = json.dumps(result.to_dict(), ensure_ascii=False)

    assert image_path in second_content
    assert result.images() == {image_path: b"defensive-table-image"}
    assert "mutated by caller" not in exported_json
