"""Doclib 专属历史文档读取的结构、来源及通用边界回归。"""

from copy import deepcopy
from collections.abc import Callable
import json

import pytest
from docvortex.schema import MiddleJson

from mineru.doclib.core.middle_json import read_cached_middle_json
from mineru.parser import ParseResult


def _legacy_v2() -> dict:
    """构造带续接、标题、表格素材和应用扩展的旧 Schema 2.0 文档。"""
    return {
        "schema_version": "2.0",
        "file_suffix": "pdf",
        "mineru_version": "3.4.4",
        "effort": "high",
        "parse_mode": "ocr",
        "is_full_document": False,
        "extensions": {"application": {"items": [None, True, 1, {"type": "image", "image_base64": "keep"}]}},
        "pages": [
            {
                "page_idx": 3,
                "blocks": [
                    {
                        "type": "paragraph_title",
                        "index": 0,
                        "bbox": [0.1, 0.1, 0.9, 0.2],
                        "level": 4,
                        "anchor": "section",
                        "content": [{"type": "text", "content": "标题"}],
                    },
                    {
                        "type": "text",
                        "index": 1,
                        "bbox": [0.1, 0.3, 0.9, 0.4],
                        "continues_prev": True,
                        "content": [
                            {
                                "type": "hyperlink",
                                "url": "https://example.com",
                                "content": [{"type": "text", "content": "保留正文"}],
                            }
                        ],
                    },
                    {
                        "type": "table",
                        "index": 2,
                        "bbox": [0.1, 0.5, 0.9, 0.8],
                        "cell_merge": [0, 1],
                        "content": [
                            {
                                "type": "table_body",
                                "index": 2,
                                "bbox": [0.1, 0.5, 0.9, 0.8],
                                "content": "<table><tr><td>保留表格</td></tr></table>",
                                "image_path": "images/table.png",
                            }
                        ],
                    },
                ],
            },
            {"page_idx": 5},
        ],
    }


def _legacy_345(*, wrapped: bool = False) -> dict:
    """构造真实旧字段布局及非连续页号，版本保留发行标签中的 3.4.4。"""
    pages = [
        {
            "page_idx": index,
            "page_size": [100, 200],
            "preproc_blocks": [
                {
                    "type": "text",
                    "bbox": [10, 20, 90, 40],
                    "lines": [{"bbox": [10, 20, 90, 40], "spans": [{"type": "text", "content": "正文 <eq>x^2</eq>"}]}],
                },
            ],
            "discarded_blocks": [],
        }
        for index in [3, 5]
    ]
    payload = {"_version_name": "3.4.4", "_effort": "high", "_ocr_enable": True}
    if wrapped:
        payload.update(schema_version="1.0", pages=pages)
    else:
        payload["pdf_info"] = pages
    return payload


@pytest.mark.parametrize("wrapped", [False, True])
def test_legacy_345_roundtrip_preserves_origin_and_pages(wrapped: bool) -> None:
    """两个旧页面封装均可在缓存边界转换，输入和来源不受污染。"""
    payload = _legacy_345(wrapped=wrapped)
    before = deepcopy(payload)
    result = read_cached_middle_json(payload)
    assert payload == before
    assert [page.page_idx for page in result.pages] == [3, 5]
    assert result.is_full_document is False
    assert result.metadata.producer.version == "3.4.4"
    assert result.extensions["mineru"] == {"tier": "standard", "parse_mode": "ocr"}
    assert result.pages[0].blocks[0].content[0].content == "正文 "
    assert result.pages[0].blocks[0].content[1].content == "x^2"
    assert read_cached_middle_json(result.to_dict()) == result
    assert MiddleJson.from_json(result.to_json()) == result


def test_v2_changes_only_envelope_without_postprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """旧 2.0 的页面保持不变，转换不触发段落表格后处理或 LLM。"""
    from mineru.backend.postprocess import document as host_document

    def unexpected(*_args: object, **_kwargs: object) -> None:
        """捕获任何越过外层迁移范围的后处理调用。"""
        raise AssertionError("Envelope migration must not postprocess pages")

    monkeypatch.setattr("docvortex.postprocess.document.model_json_to_middle_json", unexpected)
    monkeypatch.setattr("docvortex.postprocess.pages.model_json_to_pages", unexpected)
    monkeypatch.setattr(host_document, "apply_llm_aided_postprocess", unexpected)
    monkeypatch.setattr(host_document, "build_middle_json", unexpected)
    payload = _legacy_v2()
    before = deepcopy(payload)
    result = read_cached_middle_json(payload)
    assert payload == before
    assert result.to_dict()["pages"] == payload["pages"]
    assert result.metadata.file_suffix == "pdf"
    assert result.metadata.producer.version == "3.4.4"
    assert result.extensions["application"] == payload["extensions"]["application"]
    assert result.extensions["mineru"] == {"tier": "standard", "parse_mode": "ocr"}
    assert read_cached_middle_json(result.to_dict()).to_dict() == result.to_dict()
    result.extensions["application"]["items"].append("changed")
    assert payload == before


@pytest.mark.parametrize(
    "updates,expected",
    [
        ({}, None),
        ({"_effort": "high"}, None),
        ({"_ocr_enable": True}, None),
        ({"_effort": "high", "_ocr_enable": False}, {"tier": "standard", "parse_mode": "txt"}),
        ({"_effort": "xhigh", "parse_mode": "ocr"}, {"tier": "advanced", "parse_mode": "ocr"}),
        ({"_effort": "unrecognized", "parse_mode": "txt"}, None),
    ],
)
def test_missing_metadata_stays_unknown(updates: dict, expected: dict | None) -> None:
    """缺失或不可识别的记录保持未知，不制造当前版本或默认解析档位。"""
    payload = {"pdf_info": [], **updates}
    result = read_cached_middle_json(payload)
    assert result.metadata.producer.model_dump() == {"name": "mineru", "version": "unknown"}
    assert result.metadata.file_suffix == "pdf"
    assert result.extensions.get("mineru") == expected


def test_v2_missing_product_records_preserve_application_extensions() -> None:
    """旧 2.0 缺少版本及运行记录时也保持未知，应用扩展不受影响。"""
    payload = _legacy_v2()
    for key in ("mineru_version", "effort", "parse_mode"):
        del payload[key]
    document = read_cached_middle_json(payload)
    assert document.metadata.producer.version == "unknown"
    assert document.extensions == payload["extensions"]


def test_consistent_existing_product_extension_is_preserved() -> None:
    """已有可靠产品扩展与旧字段一致时直接保留，不产生重复记录。"""
    payload = _legacy_v2()
    payload["extensions"]["mineru"] = {"tier": "standard", "parse_mode": "ocr"}
    assert read_cached_middle_json(payload).extensions == payload["extensions"]


def test_345_conflicting_full_document_flag_fails() -> None:
    """旧根字段与实际抽页映射矛盾时不得静默覆盖整本语义。"""
    payload = _legacy_345()
    payload["is_full_document"] = True
    with pytest.raises(ValueError, match="Conflicting legacy is_full_document"):
        read_cached_middle_json(payload)


@pytest.mark.parametrize(
    "updates",
    [
        {"mineru_version": "other-version"},
        {"effort": "medium"},
        {"parse_mode": "txt"},
    ],
)
def test_conflicting_legacy_metadata_aliases_fail(updates: dict) -> None:
    """同一来源或运行参数的旧别名互相矛盾时明确报错，不静默选取一方。"""
    payload = _legacy_345()
    payload.update(updates)
    with pytest.raises(ValueError, match="Conflicting legacy"):
        read_cached_middle_json(payload)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(schema="docvortex.middle", schema_version="1.0"),
        lambda p: p.update(schema=None),
        lambda p: p.update(schema="other.middle"),
        lambda p: p.update(metadata={}),
        lambda p: p.update(schema_version="99.0"),
        lambda p: p.update(file_suffix="unsupported"),
        lambda p: p.pop("is_full_document"),
        lambda p: p["extensions"].update(mineru={"tier": "basic", "parse_mode": "ocr"}),
    ],
)
def test_ambiguous_or_conflicting_v2_is_rejected(mutate: Callable[[dict], object]) -> None:
    """新协议损坏、未知版本及保留字段冲突不能被历史兼容掩盖。"""
    payload = _legacy_v2()
    mutate(payload)
    with pytest.raises(ValueError):
        read_cached_middle_json(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": "1.0", "pages": [{"page_idx": 0, "blocks": []}]},
        {"pdf_info": [{"page_idx": 0, "blocks": []}]},
        {"pdf_info": [], "pages": []},
        {"pdf_info": [], "schema_version": "99.0"},
        {"pages": []},
    ],
)
def test_mixed_pages_are_not_silently_emptied(payload: dict) -> None:
    """拒绝把当前 blocks 误当作旧页面后悄悄丢弃正文。"""
    with pytest.raises(ValueError):
        read_cached_middle_json(payload)


@pytest.mark.parametrize("indices", [[3, 3], [5, 3], [-1, 3], [True, 3]])
def test_invalid_legacy_page_indices_fail(indices: list) -> None:
    """旧页号仍受唯一、顺序、非负和整数约束。"""
    payload = _legacy_345()
    for page, index in zip(payload["pdf_info"], indices):
        page["page_idx"] = index
    with pytest.raises(ValueError):
        read_cached_middle_json(payload)


@pytest.mark.parametrize("family", ["345", "v1", "v2"])
def test_compatibility_is_not_exposed_by_generic_readers(family: str) -> None:
    """Doclib 兼容成功不改变通用 ParseResult 或 DocVortex 的严格边界。"""
    payload = _legacy_v2() if family == "v2" else _legacy_345(wrapped=family == "v1")
    assert read_cached_middle_json(payload).pages
    for reader in [ParseResult.from_dict, MiddleJson.from_dict]:
        with pytest.raises(ValueError):
            reader(payload)
    with pytest.raises(ValueError):
        ParseResult.from_json(json.dumps(payload))
