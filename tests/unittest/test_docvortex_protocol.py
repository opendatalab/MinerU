"""验证 MinerU 独立维护产品封装，DocVortex 只提供共享语义对象。"""

from collections.abc import Callable
from copy import deepcopy
from typing import Literal

import pytest
from pydantic import ValidationError

from docvortex.schema import EquationBlock, ImageBlock, ImageBodyBlock, MiddleJson, PageInfo, TextBlock, TextSpan
from mineru.integrations.docvortex import (
    MinerUMetadata,
    build_metadata,
    from_mineru_middle,
    to_mineru_middle,
    with_mineru_metadata,
)
from mineru.render.structured_content import render_structured_content
from mineru.parser import ParseResult


def _document() -> MiddleJson:
    """建立不携带宿主信息的最小语义文档，覆盖文字和公式。"""
    return MiddleJson(
        pages=[
            PageInfo(
                page_idx=0,
                blocks=[
                    TextBlock(type="text", index=0, content=[TextSpan(type="text", content="Hello 文档")]),
                    EquationBlock(type="equation", index=1, content="x^2"),
                ],
            )
        ],
        is_full_document=True,
        file_suffix="html",
    )


def test_mineru_envelope_roundtrip() -> None:
    """从引擎迁入的往返契约保证产品字段和页面语义不变。"""
    payload = _document().to_dict(skip_defaults=False)
    for key in ("schema", "schema_version", "producer", "extensions"):
        payload.pop(key, None)
    payload.update(schema_version="2.0", effort="flash", parse_mode="txt", mineru_version="3.4.5")
    before = deepcopy(payload)

    assert to_mineru_middle(from_mineru_middle(payload), skip_defaults=False) == payload
    assert payload == before


def test_host_metadata_preserves_other_extensions_and_rendering_does_not_mutate_document() -> None:
    """宿主附加产品信息后仍保留其它扩展，公共渲染只加工输出封装。"""
    document = _document()
    document.extensions = {"application": {"source": "fixture"}}
    metadata = MinerUMetadata(effort="high", parse_mode="ocr", mineru_version="3.4.5")
    with_mineru_metadata(document, metadata)
    before = document.to_dict(skip_defaults=False)

    payload = to_mineru_middle(document)
    structured = render_structured_content(document)

    assert document.extensions["application"] == {"source": "fixture"}
    assert document.extensions["mineru"] == metadata.model_dump(mode="json")
    assert build_metadata(**metadata.model_dump()) == {"mineru": document.extensions["mineru"]}
    assert payload["schema_version"] == "2.0"
    assert structured["pages"][0]["blocks"][0]["content"] == "Hello 文档"
    for output in (payload, structured):
        assert {key: output[key] for key in MinerUMetadata.model_fields} == metadata.model_dump(mode="json")
        assert not {"schema", "producer", "extensions"} & output.keys()
    assert "schema_version" not in structured
    assert document.to_dict(skip_defaults=False) == before


@pytest.mark.parametrize("operation", [to_mineru_middle, render_structured_content])
def test_host_output_requires_explicit_mineru_metadata(operation: Callable[[MiddleJson], object]) -> None:
    """没有产品元数据的原生返回值不能被静默伪装为 MinerU 输出。"""
    with pytest.raises(ValidationError):
        operation(_document())


@pytest.mark.parametrize("suffix", ["pdf", "docx"])
def test_parse_result_preserves_format_specific_image_omission(suffix: Literal["pdf", "docx"]) -> None:
    """宿主结果写出时仅省略 PDF 内嵌图片，保留图片路径且不修改语义对象。"""
    body = ImageBodyBlock(
        type="image_body",
        index=0,
        content="figure",
        bbox=(0.1, 0.1, 0.9, 0.9),
        image_path="images/figure.png",
        image_base64="data:image/png;base64,aW1hZ2U=",
    )
    document = MiddleJson(
        pages=[PageInfo(page_idx=0, blocks=[ImageBlock(type="image", index=0, bbox=body.bbox, content=[body])])],
        is_full_document=True,
        file_suffix=suffix,
        extensions=build_metadata(effort="flash", parse_mode="txt", mineru_version="3.4.5"),
    )
    before = document.to_dict(skip_defaults=False)

    payload = ParseResult(document).to_dict()

    exported_body = payload["pages"][0]["blocks"][0]["content"][0]
    assert ("image_base64" in exported_body) is (suffix != "pdf")
    assert exported_body["image_path"] == "images/figure.png"
    assert document.to_dict(skip_defaults=False) == before


@pytest.mark.parametrize(
    ("field", "value"),
    [("effort", "low"), ("parse_mode", "auto"), ("mineru_version", ""), ("mineru_version", 345)],
)
def test_host_metadata_validation_stays_strict(field: str, value: object) -> None:
    """产品枚举、版本非空约束及类型检查由 MinerU 继续严格执行。"""
    metadata = {"effort": "flash", "parse_mode": "txt", "mineru_version": "3.4.5", field: value}
    with pytest.raises(ValidationError):
        build_metadata(**metadata)
