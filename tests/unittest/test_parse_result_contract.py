from dataclasses import fields
import json

import pytest

from mineru.parser.base import ParseResult
from mineru.schema.middle_json import MIDDLE_JSON_SCHEMA_VERSION
from mineru.types import Block, Line, PageInfo, Span


def test_parse_result_does_not_expose_backend_version_or_file_name() -> None:
    field_names = {field.name for field in fields(ParseResult)}

    assert "_backend" not in field_names
    assert "_version_name" not in field_names
    assert "_file_name" not in field_names


def test_parse_result_from_dict_restores_pages() -> None:
    result = ParseResult(
        pages=[
            PageInfo(
                page_idx=3,
                page_size=(640, 480),
                para_blocks=[
                    Block(
                        index=0,
                        type="text",
                        bbox=(1.0, 2.0, 3.0, 4.0),
                        lines=[
                            Line(
                                bbox=(1.0, 2.0, 3.0, 4.0),
                                spans=[Span(type="text", bbox=(1.0, 2.0, 3.0, 4.0), content="hello")],
                            )
                        ],
                    )
                ],
            )
        ]
    )

    restored = ParseResult.from_dict(result.to_dict())

    assert restored.to_dict() == result.to_dict()
    assert restored.pages[0].page_size == (640, 480)
    assert restored.pages[0].para_blocks[0].bbox == (1.0, 2.0, 3.0, 4.0)
    assert restored.pages[0].para_blocks[0].lines[0].spans[0].content == "hello"


def test_parse_result_to_dict_includes_schema_version_without_meta() -> None:
    result = ParseResult(pages=[PageInfo(page_idx=0)])

    payload = result.to_dict()

    assert payload["schema_version"] == MIDDLE_JSON_SCHEMA_VERSION
    assert "pages" in payload
    assert "_meta" not in payload


def test_parse_result_from_dict_rejects_missing_pages() -> None:
    with pytest.raises(ValueError, match="pages"):
        ParseResult.from_dict({"pdf_info": []})


def test_parse_result_from_dict_accepts_pages_and_preserves_page_backend() -> None:
    restored = ParseResult.from_dict(
        {
            "pages": [
                {
                    "page_idx": 0,
                    "page_size": [100, 200],
                    "_backend": "pipeline",
                    "preproc_blocks": [
                        {
                            "index": 0,
                            "type": "title",
                            "bbox": [1.0, 2.0, 3.0, 4.0],
                            "level": 1,
                            "lines": [
                                {
                                    "bbox": [1.0, 2.0, 3.0, 4.0],
                                    "spans": [
                                        {
                                            "type": "text",
                                            "bbox": [1.0, 2.0, 3.0, 4.0],
                                            "content": "Heading",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ]
        }
    )

    assert len(restored.pages) == 1
    assert restored.pages[0]._backend == "pipeline"
    assert restored.pages[0].page_size == (100, 200)
    assert restored.pages[0].preproc_blocks[0].level == 1


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
