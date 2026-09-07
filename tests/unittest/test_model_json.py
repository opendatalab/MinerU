from __future__ import annotations

import json

import pytest
from _span_test_utils import inline
from docvortex.codecs.json import load_middle, load_model
from docvortex.schema import Producer
from pydantic import ValidationError

from mineru import ModelJson as PublicModelJson
from mineru.backend.postprocess import document
from mineru.backend.postprocess.document import model_json_to_middle_json
from mineru.config import LLMAidedConfig
from mineru.integrations.docvortex import build_metadata

from mineru.types import MiddleJson, ModelJson


def _model_json(
    *,
    pages: list[list[dict[str, object]]] | None = None,
    page_index_map: list[int] | None = None,
) -> ModelJson:
    """构造包含全部必填元数据的严格 ModelJson 测试对象。"""
    return ModelJson(
        pages=pages if pages is not None else [[{"type": "text", "content": inline("正文")}]],
        page_index_map=page_index_map if page_index_map is not None else [],
        metadata={"file_suffix": "docx", "producer": Producer(name="mineru", version="3.4.0")},
        extensions=build_metadata(
            effort="flash",
            parse_mode="txt",
        ),
    )


def test_model_json_is_public_and_serializes_exact_envelope() -> None:
    """验证公开 ModelJson 固定输出六个顶层字段且空映射不会被省略。"""
    model_json = _model_json()

    payload = model_json.to_dict()

    assert PublicModelJson is ModelJson
    assert set(payload) == {"schema", "schema_version", "metadata", "extensions", "pages", "page_index_map"}
    assert payload == {
        "pages": [[{"type": "text", "content": inline("正文")}]],
        "page_index_map": [],
        "metadata": {"file_suffix": "docx", "producer": {"name": "mineru", "version": "3.4.0"}},
        "schema": "docvortex.model",
        "schema_version": "2.0",
        "extensions": {"mineru": {"tier": "flash", "parse_mode": "txt"}},
    }
    assert model_json.is_full_document is True
    assert model_json.resolved_page_indices == [0]
    assert "is_full_document" not in payload
    assert "resolved_page_indices" not in payload
    assert load_model(json.loads(model_json.to_json())) == model_json


def test_model_json_requires_page_index_map_and_forbids_extra_fields() -> None:
    """验证页映射不可省略且 ModelJson 顶层不接受未声明字段。"""
    payload = {
        "pages": [],
        "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": "3.4.0"}},
        "schema": "docvortex.model",
        "schema_version": "2.0",
        "extensions": {"mineru": {"tier": "flash", "parse_mode": "txt"}},
    }
    with pytest.raises(ValidationError, match="page_index_map"):
        ModelJson.from_dict(payload)

    with pytest.raises(ValidationError, match="extra_forbidden"):
        ModelJson.from_dict({**payload, "page_index_map": [], "unexpected": True})


def test_middle_json_requires_and_serializes_full_document_flag() -> None:
    """验证 MiddleJson 将整本语义作为必填字段稳定序列化并往返。"""
    middle_json = MiddleJson(
        pages=[],
        is_full_document=False,
        metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version="3.4.0")},
        extensions=build_metadata(
            effort="flash",
            parse_mode="txt",
        ),
    )

    payload = middle_json.to_dict()

    assert set(payload) == {"schema", "schema_version", "metadata", "extensions", "pages", "is_full_document"}
    assert payload["is_full_document"] is False
    assert load_middle(json.loads(middle_json.to_json())) == middle_json


def test_model_json_to_middle_json_builds_strict_document_before_pdf_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证文档编排完整继承元数据，并让 PDF LLM 直接消费同一 MiddleJson。"""
    model_json = ModelJson(
        pages=[[]],
        page_index_map=[3],
        metadata={"file_suffix": "pdf", "producer": Producer(name="mineru", version="3.4.0")},
        extensions=build_metadata(
            effort="xhigh",
            parse_mode="ocr",
        ),
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
    assert middle_json.metadata.file_suffix == "pdf"
    assert middle_json.extensions["mineru"]["tier"] == "advanced"
    assert middle_json.extensions["mineru"]["parse_mode"] == "ocr"
    assert middle_json.metadata.producer.version == "3.4.0"


@pytest.mark.parametrize("invalid_value", [None, 0, 1, "true"])
def test_middle_json_rejects_missing_or_non_boolean_full_document_flag(invalid_value: object) -> None:
    """验证 MiddleJson 不为整本语义提供缺省值或宽松布尔转换。"""
    payload = {
        "pages": [],
        "metadata": {"file_suffix": "pdf", "producer": {"name": "mineru", "version": "3.4.0"}},
        "schema": "docvortex.middle",
        "schema_version": "2.0",
        "extensions": {"mineru": {"tier": "flash", "parse_mode": "txt"}},
    }
    if invalid_value is not None:
        payload["is_full_document"] = invalid_value

    with pytest.raises(ValidationError, match="is_full_document"):
        MiddleJson.from_dict(payload)
