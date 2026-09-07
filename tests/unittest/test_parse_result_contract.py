from docvortex.schema import Producer
from mineru.integrations.docvortex import build_metadata
from _span_test_utils import inline as _inline
from dataclasses import fields
import json

import pytest
from pydantic import ValidationError

from mineru.parser.base import ParseResult
from mineru.parser import MIDDLE_JSON_SCHEMA_VERSION
from mineru.types import BlockType, MiddleJson, PageFootnoteBlock, PageInfo, TableBlock, TableBodyBlock, TextBlock
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
                            content=_inline("hello"),
                        )
                    ],
                )
            ],
            is_full_document=True,
            metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version=__version__)},
            extensions=build_metadata(
                effort="medium",
                parse_mode="txt",
            ),
        )
    )

    restored = ParseResult.from_dict(result.to_dict())

    assert restored.to_dict() == result.to_dict()
    assert restored.pages[0].page_idx == 3
    assert restored.pages[0].blocks[0].bbox == (0.0, 0.0, 0.1, 0.1)
    assert restored.pages[0].blocks[0].content[0].content == "hello"


def test_parse_result_to_dict_includes_schema_version_without_meta() -> None:
    result = ParseResult(
        middle_json=MiddleJson(
            pages=[PageInfo(page_idx=0)],
            is_full_document=True,
            metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version=__version__)},
            extensions=build_metadata(
                effort="medium",
                parse_mode="txt",
            ),
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
                        PageFootnoteBlock(
                            type=BlockType.PAGE_FOOTNOTE,
                            index=0,
                            content=_inline("Footnote"),
                            anchor="note-one",
                        )
                    ],
                )
            ],
            is_full_document=True,
            metadata={"file_suffix": "epub", "producer": Producer(name="mineru", version=__version__)},
            extensions=build_metadata(
                effort="flash",
                parse_mode="txt",
            ),
        )
    )

    restored = ParseResult.from_dict(result.to_dict())

    footnote = restored.pages[0].blocks[0]
    assert footnote.type == BlockType.PAGE_FOOTNOTE
    assert footnote.anchor == "note-one"  # type: ignore[union-attr]


def test_parse_result_rejects_low_effort() -> None:
    """验证当前 schema 的旧 Low effort 值仍按严格枚举失败。"""
    with pytest.raises(ValidationError, match="literal_error"):
        ParseResult.from_dict(
            {
                "schema_version": MIDDLE_JSON_SCHEMA_VERSION,
                "pages": [],
                "is_full_document": True,
                "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": __version__}},
                "schema": "docvortex.middle",
                "extensions": {"mineru": {"tier": "low", "parse_mode": "ocr"}},
            }
        )


def test_parse_result_from_dict_rejects_missing_pages() -> None:
    with pytest.raises(ValueError, match="pages"):
        ParseResult.from_dict({"schema": "docvortex.middle", "schema_version": MIDDLE_JSON_SCHEMA_VERSION})


def test_parse_result_from_json_rejects_mineru_3_4_5_middle_json() -> None:
    """验证 ParseResult 的真实 JSON 入口拒绝历史 pdf_info 文档。"""
    data = {
        "_backend": "hybrid",
        "_effort": "high",
        "_ocr_enable": True,
        "_version_name": "3.4.4",
        "pdf_info": [
            {
                "page_idx": 2,
                "page_size": [100, 100],
                "preproc_blocks": [
                    {
                        "index": 0,
                        "type": "text",
                        "bbox": [10, 10, 90, 20],
                        "lines": [
                            {
                                "bbox": [10, 10, 90, 20],
                                "spans": [{"type": "text", "bbox": [0.0, 0.0, 0.0, 0.0], "content": "round trip"}],
                            }
                        ],
                    }
                ],
                "discarded_blocks": [],
            }
        ],
    }

    with pytest.raises(ValueError, match="reparse"):
        ParseResult.from_json(json.dumps(data))


def test_parse_result_rejects_schema_v1_page_wrapper() -> None:
    """验证旧 1.0 pages 封装明确拒绝，不生成虚假的空文档。"""
    with pytest.raises(ValueError, match="reparse"):
        ParseResult.from_dict({"schema_version": "1.0", "pages": []})


@pytest.mark.parametrize("schema_version", [None, "3.0"])
def test_parse_result_rejects_legacy_schema_versions(schema_version: str | None) -> None:
    """验证无版本 pages 与未知版本 payload 仍要求重新解析源文件。"""
    payload: dict[str, object] = {"pages": []}
    if schema_version is not None:
        payload["schema_version"] = schema_version

    with pytest.raises(ValueError, match="reparse the source document"):
        ParseResult.from_dict(payload)


def test_parse_result_export_pages_returns_defensive_copy() -> None:
    """验证调用方修改导出页面副本时不会污染 ParseResult 内部状态。"""
    image_path = "images/table.png"
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
            metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version=__version__)},
            extensions=build_metadata(
                effort="medium",
                parse_mode="txt",
            ),
        ),
    )
    first_export = result.export_pages()
    first_export[0].blocks[0].content[0].content = "mutated by caller"

    second_export = result.export_pages()
    second_content = second_export[0].blocks[0].content[0].content
    exported_json = json.dumps(result.to_dict(), ensure_ascii=False)

    assert image_path in second_content
    assert "mutated by caller" not in exported_json
