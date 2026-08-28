from __future__ import annotations

import pytest
from pydantic import ValidationError

from mineru import ModelJson as PublicModelJson
from mineru.backend.postprocess import document
from mineru.backend.postprocess.document import model_json_to_middle_json
from mineru.config import LLMAidedConfig
from mineru.types import MiddleJson, ModelJson

from _span_test_utils import inline


def _model_json(
    *,
    pages: list[list[dict[str, object]]] | None = None,
    page_index_map: list[int] | None = None,
) -> ModelJson:
    """构造包含全部必填元数据的严格 ModelJson 测试对象。"""
    return ModelJson(
        pages=pages if pages is not None else [[{"type": "text", "content": inline("正文")}]],
        page_index_map=page_index_map if page_index_map is not None else [],
        file_suffix="docx",
        effort="flash",
        parse_mode="txt",
        mineru_version="3.4.0",
    )


def test_model_json_is_public_and_serializes_exact_envelope() -> None:
    """验证公开 ModelJson 固定输出六个顶层字段且空映射不会被省略。"""
    model_json = _model_json()

    payload = model_json.to_dict()

    assert PublicModelJson is ModelJson
    assert list(payload) == [
        "pages",
        "page_index_map",
        "file_suffix",
        "effort",
        "parse_mode",
        "mineru_version",
    ]
    assert payload == {
        "pages": [[{"type": "text", "content": inline("正文")}]],
        "page_index_map": [],
        "file_suffix": "docx",
        "effort": "flash",
        "parse_mode": "txt",
        "mineru_version": "3.4.0",
    }
    assert model_json.is_full_document is True
    assert model_json.resolved_page_indices == [0]
    assert "is_full_document" not in payload
    assert "resolved_page_indices" not in payload
    assert ModelJson.model_validate_json(model_json.to_json()) == model_json


def test_non_empty_page_index_map_represents_partial_input() -> None:
    """验证非空页号映射表示显式抽页并与 raw pages 一一对应。"""
    model_json = _model_json(pages=[[], []], page_index_map=[3, 5])

    assert model_json.page_index_map == [3, 5]
    assert model_json.is_full_document is False
    assert model_json.resolved_page_indices == [3, 5]


def test_empty_model_json_resolves_to_no_page_indices() -> None:
    """验证空文档仍属于整本解析且不会产生虚构页号。"""
    model_json = _model_json(pages=[])

    assert model_json.is_full_document is True
    assert model_json.resolved_page_indices == []


@pytest.mark.parametrize(
    ("page_index_map", "message"),
    [
        ([0], "length mismatch"),
        ([0, 0], "unique"),
        ([1, 0], "increasing"),
        ([0, -1], "non-negative"),
    ],
)
def test_model_json_rejects_invalid_page_index_map(page_index_map: list[int], message: str) -> None:
    """验证显式抽页映射不允许截断、重复、逆序或负数。"""
    with pytest.raises(ValidationError, match=message):
        _model_json(pages=[[], []], page_index_map=page_index_map)


@pytest.mark.parametrize(
    "pages",
    ["[]", [{}], [[[]]]],
)
def test_model_json_rejects_invalid_page_structure(pages: object) -> None:
    """验证 pages 必须保持页列表、块列表和块字典三层结构。"""
    with pytest.raises(ValidationError):
        ModelJson(
            pages=pages,  # type: ignore[arg-type]
            page_index_map=[],
            file_suffix="pdf",
            effort="flash",
            parse_mode="txt",
            mineru_version="3.4.0",
        )


def test_model_json_requires_page_index_map_and_forbids_extra_fields() -> None:
    """验证页映射不可省略且 ModelJson 顶层不接受未声明字段。"""
    payload = {
        "pages": [],
        "file_suffix": "pdf",
        "effort": "flash",
        "parse_mode": "txt",
        "mineru_version": "3.4.0",
    }
    with pytest.raises(ValidationError, match="page_index_map"):
        ModelJson.model_validate(payload)

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelJson.model_validate({**payload, "page_index_map": [], "unexpected": True})


def test_strict_document_models_reject_removed_low_effort() -> None:
    """验证 ModelJson 与 MiddleJson 的 schema 2.0 均不再接受 Low effort。"""
    with pytest.raises(ValidationError, match="literal_error"):
        ModelJson.model_validate(
            {
                "pages": [],
                "page_index_map": [],
                "file_suffix": "pdf",
                "effort": "low",
                "parse_mode": "ocr",
                "mineru_version": "3.4.0",
            }
        )

    with pytest.raises(ValidationError, match="literal_error"):
        MiddleJson.model_validate(
            {
                "pages": [],
                "is_full_document": True,
                "file_suffix": "pdf",
                "effort": "low",
                "parse_mode": "ocr",
                "mineru_version": "3.4.0",
            }
        )


def test_middle_json_requires_and_serializes_full_document_flag() -> None:
    """验证 MiddleJson 将整本语义作为必填字段稳定序列化并往返。"""
    middle_json = MiddleJson(
        pages=[],
        is_full_document=False,
        file_suffix="pdf",
        effort="flash",
        parse_mode="txt",
        mineru_version="3.4.0",
    )

    payload = middle_json.to_dict()

    assert list(payload) == [
        "pages",
        "is_full_document",
        "file_suffix",
        "effort",
        "parse_mode",
        "mineru_version",
    ]
    assert payload["is_full_document"] is False
    assert MiddleJson.model_validate_json(middle_json.to_json()) == middle_json


def test_model_json_to_middle_json_builds_strict_document_before_pdf_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证文档编排完整继承元数据，并让 PDF LLM 直接消费同一 MiddleJson。"""
    model_json = ModelJson(
        pages=[[]],
        page_index_map=[3],
        file_suffix="pdf",
        effort="xhigh",
        parse_mode="ocr",
        mineru_version="3.4.0",
    )
    observed: list[MiddleJson] = []

    def fake_llm_postprocess(middle_json: MiddleJson, _config: LLMAidedConfig) -> None:
        """记录 LLM 收到的严格 MiddleJson，并校验抽页语义已写入。"""
        assert middle_json.is_full_document is False
        assert [page.page_idx for page in middle_json.pages] == [3]
        observed.append(middle_json)

    monkeypatch.setattr(document, "apply_llm_aided_postprocess", fake_llm_postprocess)

    middle_json = model_json_to_middle_json(model_json, llm_aided_config=LLMAidedConfig())

    assert observed == [middle_json]
    assert middle_json.file_suffix == "pdf"
    assert middle_json.effort == "xhigh"
    assert middle_json.parse_mode == "ocr"
    assert middle_json.mineru_version == "3.4.0"


@pytest.mark.parametrize("invalid_value", [None, 0, 1, "true"])
def test_middle_json_rejects_missing_or_non_boolean_full_document_flag(invalid_value: object) -> None:
    """验证 MiddleJson 不为整本语义提供缺省值或宽松布尔转换。"""
    payload = {
        "pages": [],
        "file_suffix": "pdf",
        "effort": "flash",
        "parse_mode": "txt",
        "mineru_version": "3.4.0",
    }
    if invalid_value is not None:
        payload["is_full_document"] = invalid_value

    with pytest.raises(ValidationError, match="is_full_document"):
        MiddleJson.model_validate(payload)
