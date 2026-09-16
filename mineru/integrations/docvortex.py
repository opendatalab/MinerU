"""校验 MinerU 产品扩展，共享文档协议由 DocVortex 统一维护。"""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, ConfigDict, JsonValue
from docvortex.schema import DocumentProperties, FileSuffix, MiddleJson, ModelJson
from docvortex.document.contracts import HtmlSourceContext


class MinerUMetadata(BaseModel):
    """仅记录实际执行的产品档位和最终文字提取模式。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    tier: Literal["flash", "basic", "standard", "advanced"]
    parse_mode: Literal["txt", "ocr"]


def build_metadata(*, effort: str, parse_mode: str) -> dict[str, JsonValue]:
    """将分析结果的实际 effort 映射为产品档位，不记录请求值。"""
    tiers = {"flash": "flash", "medium": "basic", "high": "standard", "xhigh": "advanced"}
    if effort not in tiers:
        raise ValueError(f"Unsupported actual analyze effort: {effort}")
    metadata = MinerUMetadata.model_validate({"tier": tiers[effort], "parse_mode": parse_mode})
    return {"mineru": metadata.model_dump(mode="json")}


def read_source_properties(
    data: bytes,
    suffix: FileSuffix,
    source_context: HtmlSourceContext | None = None,
) -> DocumentProperties:
    """提取原始输入属性；正文引擎决定输入有效性，属性失败仅记录诊断。"""
    from docvortex import extract_metadata
    from docvortex.errors import DocumentError
    from loguru import logger

    try:
        result = extract_metadata(data, file_suffix=suffix, source_context=source_context)
    except DocumentError as exc:
        logger.warning("Source metadata unavailable: {}", exc)
        return DocumentProperties()
    for diagnostic in result.diagnostics:
        logger.warning("Source metadata: {}", diagnostic.message)
    return result.metadata.document or DocumentProperties()


def with_mineru_metadata(document: ModelJson | MiddleJson, metadata: MinerUMetadata) -> None:
    """附加产品扩展并保留其他应用信息，不修改文档生产者。"""
    document.extensions = {**document.extensions, "mineru": metadata.model_dump(mode="json")}


def validate_mineru_metadata(document: ModelJson | MiddleJson) -> None:
    """只校验已存在的 MinerU 扩展，原生及其他生产者文档可直接消费。"""
    if "mineru" in document.extensions:
        MinerUMetadata.model_validate(document.extensions["mineru"])


__all__ = ["read_source_properties", "MinerUMetadata", "build_metadata", "with_mineru_metadata", "validate_mineru_metadata"]
