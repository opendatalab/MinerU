"""验证宿主与原生文档共享协议，同时保留产品扩展及素材约定。"""

from copy import deepcopy
import pytest
from pydantic import ValidationError
from docvortex.schema import DocumentMetadata, Producer, MiddleJson, PageInfo, TextBlock, TextSpan, ImageBodyBlock, ImageBlock
from mineru.integrations.docvortex import MinerUMetadata, build_metadata, validate_mineru_metadata, with_mineru_metadata
from mineru.parser import ParseResult
from mineru.render.structured_content import render_structured_content


def _document() -> MiddleJson:
    """建立有真实正文且无 MinerU 运行记录的原生文档。"""
    return MiddleJson(
        pages=[
            PageInfo(page_idx=0, blocks=[TextBlock(type="text", index=0, content=[TextSpan(type="text", content="保留正文")])])
        ],
        is_full_document=True,
        metadata=DocumentMetadata(file_suffix="html", producer=Producer(name="docvortex", version="0.2.0")),
    )


def test_native_middle_does_not_lose_text_or_rewrite_producer() -> None:
    """回归原生 1.0 被误认作宿主历史格式后正文丢失的问题。"""
    document = _document()
    before = document.to_dict()
    result = ParseResult.from_dict(before)
    assert result.to_dict() == before
    assert result.pages[0].blocks[0].content[0].content == "保留正文"
    assert result.middle_json.metadata.producer.name == "docvortex"
    structured = render_structured_content(document)
    assert structured["metadata"] == before["metadata"]
    assert structured["extensions"] == {}
    assert not {"schema", "schema_version", "schema_id"} & structured.keys()
    assert structured["pages"][0]["blocks"][0]["content"] == "保留正文"
    assert document.to_dict() == before


@pytest.mark.parametrize("effort,tier", [("flash", "flash"), ("medium", "basic"), ("high", "standard"), ("xhigh", "advanced")])
@pytest.mark.parametrize("mode", ["txt", "ocr"])
def test_actual_effort_maps_to_product_tier(effort: str, tier: str, mode: str) -> None:
    """所有分析强度均映射为实际公开档位，扩展不重复存放版本。"""
    assert build_metadata(effort=effort, parse_mode=mode) == {"mineru": {"tier": tier, "parse_mode": mode}}


def test_extensions_survive_host_rendering_and_roundtrip() -> None:
    """保留未知应用扩展，产品登记和渲染均不改写真实来源。"""
    document = _document()
    document.extensions = {"application": {"items": [None, True, 3, {"text": "中文"}]}}
    with_mineru_metadata(document, MinerUMetadata(tier="advanced", parse_mode="ocr"))
    before = deepcopy(document.to_dict())
    assert ParseResult.from_dict(before).to_dict() == before
    assert render_structured_content(document)["extensions"] == before["extensions"]
    assert document.to_dict() == before


@pytest.mark.parametrize(
    "metadata",
    [
        {"tier": "low", "parse_mode": "txt"},
        {"tier": "basic", "parse_mode": "auto"},
        {"tier": "flash", "parse_mode": "txt", "effort": "flash"},
        None,
    ],
)
def test_mineru_validates_only_present_product_extension(metadata: object) -> None:
    """原生扩展可缺省，已存在的产品记录则必须满足严格枚举。"""
    document = _document()
    validate_mineru_metadata(document)
    document.extensions = {"mineru": metadata}
    with pytest.raises(ValidationError):
        ParseResult.from_dict(document.to_dict())


@pytest.mark.parametrize("suffix", ["pdf", "docx"])
def test_format_specific_image_omission_is_preserved(suffix: str) -> None:
    """PDF 包装继续省略内嵌图片，其他格式保留，且源对象不变。"""
    body = ImageBodyBlock(
        type="image_body",
        index=0,
        content="figure",
        bbox=(0.1, 0.1, 0.9, 0.9),
        image_path="images/a.png",
        image_base64="data:image/png;base64,aW1hZ2U=",
    )
    document = MiddleJson(
        pages=[PageInfo(page_idx=0, blocks=[ImageBlock(type="image", index=0, bbox=body.bbox, content=[body])])],
        is_full_document=True,
        metadata=DocumentMetadata(file_suffix=suffix, producer=Producer(name="external", version="original")),
    )
    before = document.to_dict(skip_defaults=False)
    payload = ParseResult(document).to_dict()
    assert ("image_base64" in payload["pages"][0]["blocks"][0]["content"][0]) is (suffix != "pdf")
    assert payload["pages"][0]["blocks"][0]["content"][0]["image_path"] == "images/a.png"
    assert document.to_dict(skip_defaults=False) == before
