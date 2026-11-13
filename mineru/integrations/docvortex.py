"""校验 MinerU 产品扩展，共享文档协议由 DocVortex 统一维护。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal
from zipfile import BadZipFile
from xml.etree.ElementTree import ParseError

from docvortex.errors import DocumentError
from pydantic import BaseModel, ConfigDict, JsonValue
from docvortex.schema import DocumentProperties, FileSuffix, MiddleJson, ModelJson
from docvortex.document.contracts import HtmlSourceContext

from ..errors import InvalidRequestError


@contextmanager
def translate_document_errors(*, native_input: bool = False) -> Iterator[None]:
    """只映射共享引擎已确认的输入错误，不将程序错误或模型失败泛化成坏文件。"""
    from lxml.etree import XMLSyntaxError

    try:
        yield
    except DocumentError as exc:
        code = "file_corrupted" if exc.code == "open_failed" else exc.code
        if code in {"resource_limit", "file_corrupted", "file_encrypted", "file_type_unsupported", "file_too_large"}:
            raise InvalidRequestError(code, exc.message, exc.param) from exc
        raise
    except (BadZipFile, ParseError, XMLSyntaxError) as exc:
        if not native_input:
            raise
        raise InvalidRequestError("file_corrupted", str(exc), "input") from exc


@contextmanager
def translate_resource_limit() -> Iterator[None]:
    """将共享文档资源限制映射为产品输入错误，其他文档异常保留原有处理。"""
    try:
        yield
    except DocumentError as exc:
        if exc.code == "resource_limit":
            raise InvalidRequestError(exc.code, exc.message, exc.param) from exc
        raise


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
    """提取原始属性，资源限制立即终止处理，其他属性失败仅记录诊断。"""
    from docvortex import extract_metadata
    from loguru import logger

    try:
        with translate_resource_limit():
            result = extract_metadata(data, file_suffix=suffix, source_context=source_context)
    except DocumentError as exc:
        logger.warning("metadata_unavailable: source metadata skipped: {}", exc)
        return DocumentProperties()
    for diagnostic in result.diagnostics:
        logger.warning("metadata_diagnostic: {}", diagnostic.message)
    return result.metadata.document or DocumentProperties()


def with_mineru_metadata(document: ModelJson | MiddleJson, metadata: MinerUMetadata) -> None:
    """附加产品扩展并保留其他应用信息，不修改文档生产者。"""
    document.extensions = {**document.extensions, "mineru": metadata.model_dump(mode="json")}


def validate_mineru_metadata(document: ModelJson | MiddleJson) -> None:
    """只校验已存在的 MinerU 扩展，原生及其他生产者文档可直接消费。"""
    if "mineru" in document.extensions:
        MinerUMetadata.model_validate(document.extensions["mineru"])


__all__ = [
    "read_source_properties",
    "translate_resource_limit",
    "translate_document_errors",
    "MinerUMetadata",
    "build_metadata",
    "with_mineru_metadata",
    "validate_mineru_metadata",
]
