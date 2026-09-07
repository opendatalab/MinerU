"""在 DocVortex 共享文档对象与 MinerU 产品协议之间进行显式适配。"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from docvortex.schema import MiddleJson, ModelJson, Producer


class MinerUMetadata(BaseModel):
    """校验旧协议的产品元数据，禁止将未知档位静默映射。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    effort: Literal["flash", "medium", "high", "xhigh"]
    parse_mode: Literal["txt", "ocr"]
    mineru_version: str = Field(min_length=1)


def build_metadata(*, effort: str, parse_mode: str, mineru_version: str) -> dict[str, JsonValue]:
    """校验产品参数并生成命名空间扩展，不给公共语义类型添加宿主字段。"""
    metadata = MinerUMetadata.model_validate({"effort": effort, "parse_mode": parse_mode, "mineru_version": mineru_version})
    return {"mineru": metadata.model_dump(mode="json")}


def with_mineru_metadata(document: ModelJson | MiddleJson, metadata: MinerUMetadata) -> None:
    """由宿主显式附加产品信息，文档本身不猜测 MinerU 版本。"""
    document.extensions = {**document.extensions, "mineru": metadata.model_dump(mode="json")}


def _from_payload(payload: dict[str, Any], *, middle: bool) -> ModelJson | MiddleJson:
    """把旧版顶层字段移入扩展，并复用唯一一套语义类型验证。"""
    data = dict(payload)
    data.pop("schema_version", None)
    if "pages" not in data:
        raise ValueError("Missing required pages in MinerU document")
    metadata = MinerUMetadata.model_validate({key: data.pop(key, None) for key in MinerUMetadata.model_fields})
    data["producer"] = Producer(name="mineru", version=metadata.mineru_version)
    data["extensions"] = {"mineru": metadata.model_dump(mode="json")}
    return MiddleJson.model_validate(data) if middle else ModelJson.model_validate(data)


def from_mineru_model(payload: dict[str, Any]) -> ModelJson:
    """将当前 MinerU Model JSON 转换成中性分析对象。"""
    result = _from_payload(payload, middle=False)
    assert isinstance(result, ModelJson)
    return result


def from_mineru_middle(payload: dict[str, Any]) -> MiddleJson:
    """将当前 MinerU Middle JSON 转换成中性语义对象。"""
    result = _from_payload(payload, middle=True)
    assert isinstance(result, MiddleJson)
    return result


def _to_payload(
    document: ModelJson | MiddleJson, *, skip_defaults: bool, exclude_block_fields: set[str] | None = None
) -> dict[str, Any]:
    """恢复旧版顶层字段，仅接受调用方真实提供的产品元数据。"""
    metadata = MinerUMetadata.model_validate(document.extensions.get("mineru"))
    payload = document.to_dict(skip_defaults=skip_defaults, exclude_block_fields=exclude_block_fields)
    for key in ("schema", "schema_version", "producer", "extensions"):
        payload.pop(key, None)
    return {**payload, **metadata.model_dump(mode="json")}


def to_mineru_model(document: ModelJson, *, skip_defaults: bool = False) -> dict[str, Any]:
    """写出与现有 MinerU 相同的 Model JSON 封装。"""
    return _to_payload(document, skip_defaults=skip_defaults)


def to_mineru_middle(
    document: MiddleJson,
    *,
    skip_defaults: bool = True,
    include_schema_version: bool = True,
    exclude_block_fields: set[str] | None = None,
) -> dict[str, Any]:
    """写出 MinerU schema 2.0，允许调用方明确选择素材省略策略。"""
    payload = _to_payload(document, skip_defaults=skip_defaults, exclude_block_fields=exclude_block_fields)
    return {"schema_version": "2.0", **payload} if include_schema_version else payload


def to_mineru_structured_content(value: dict[str, Any]) -> dict[str, Any]:
    """将共享 renderer 的中性文档头还原为现有 Structured Content 头。"""
    metadata = MinerUMetadata.model_validate(value.get("extensions", {}).get("mineru"))
    payload = {key: item for key, item in value.items() if key not in {"schema", "schema_version", "producer", "extensions"}}
    return {**payload, **metadata.model_dump(mode="json")}


__all__ = [
    "build_metadata",
    "MinerUMetadata",
    "from_mineru_model",
    "from_mineru_middle",
    "to_mineru_model",
    "to_mineru_middle",
    "to_mineru_structured_content",
    "with_mineru_metadata",
]
