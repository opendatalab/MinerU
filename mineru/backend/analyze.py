# Copyright (c) Opendatalab. All rights reserved.
"""统一 PDF、EPUB、HTML、OFD、CSV 与 Office/RTF 文档分析的稳定公共门面。"""

from __future__ import annotations

from typing import cast

from docvortex.document.contracts import HtmlSourceContext
from docvortex.schema import DocumentMetadata, DocumentProperties, Producer
from loguru import logger

from ..config import VlmConfig, config
from ..integrations.docvortex import build_metadata, read_source_properties
from ..types import FILE_SUFFIXES, FileSuffix, MiddleJson, ModelJson
from ..utils.async_utils import run_sync
from ..version import __version__ as mineru_version
from .analysis.contracts import AnalysisResult, AnalyzeEffort, OfficeSuffix, ParseMode

_SUPPORTED_ANALYZE_EFFORTS = {"flash", "medium", "high", "xhigh"}


def _log_infer_performance(file_suffix: str, page_count: int, elapsed: float) -> None:
    """使用未舍入耗时统一记录 model-list 生产速度。"""
    speed = page_count / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"model_list infer finished, file_suffix={file_suffix}, pages={page_count}, "
        f"cost={elapsed:.6f}s, speed={speed:.3f} page/s"
    )


def doc_analyze(
    file_bytes: bytes,
    effort: AnalyzeEffort = "high",
    parse_mode: ParseMode = "auto",
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    file_suffix: FileSuffix = "pdf",
    source_context: HtmlSourceContext | None = None,
    vlm_config: VlmConfig | None = None,
    source_properties: DocumentProperties | None = None,
) -> tuple[MiddleJson, ModelJson]:
    """生产严格 ModelJson，并在统一边界构造严格 MiddleJson。"""
    _validate_analyze(effort, file_suffix, page_index_map)

    if source_properties is None:
        source_properties = read_source_properties(file_bytes, file_suffix, source_context)

    if file_suffix == "pdf":
        from .analysis.pdf.pipeline import analyze_pdf

        result = analyze_pdf(
            file_bytes,
            effort=effort,
            parse_mode=parse_mode,
            image_analysis=image_analysis,
            vlm_config=vlm_config,
        )
    elif file_suffix == "csv":
        from .analysis.csv import analyze_csv

        result = analyze_csv(file_bytes)
    elif file_suffix == "epub":
        from .analysis.epub import analyze_epub

        result = analyze_epub(file_bytes)
    elif file_suffix == "html":
        from .analysis.html import analyze_html

        result = analyze_html(file_bytes, source_context=source_context)
    elif file_suffix == "ofd":
        from .analysis.ofd import analyze_ofd

        result = analyze_ofd(file_bytes)
    else:
        from .analysis.office import analyze_office

        result = analyze_office(file_bytes, cast(OfficeSuffix, file_suffix))

    model_json = _build_model_json(result, file_suffix, page_index_map, source_properties)
    from .postprocess.document import model_json_to_middle_json

    middle_json = model_json_to_middle_json(
        model_json,
        llm_aided_config=config.llm_aided,
    )
    return middle_json, model_json


async def aio_doc_analyze(
    file_bytes: bytes,
    effort: AnalyzeEffort = "high",
    parse_mode: ParseMode = "auto",
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    file_suffix: FileSuffix = "pdf",
    source_context: HtmlSourceContext | None = None,
    vlm_config: VlmConfig | None = None,
    source_properties: DocumentProperties | None = None,
) -> tuple[MiddleJson, ModelJson]:
    """vLLM/HTTP 的 PDF 分析使用原生异步编排，其余路径保持受控线程回退。"""
    _validate_analyze(effort, file_suffix, page_index_map)
    native_async = False
    if file_suffix == "pdf" and effort in {"high", "xhigh"}:
        from ..model.vlm.client import uses_native_async_vlm

        native_async = await run_sync(uses_native_async_vlm, vlm_config)
    if not native_async:
        return await run_sync(
            doc_analyze,
            file_bytes=file_bytes,
            effort=effort,
            parse_mode=parse_mode,
            image_analysis=image_analysis,
            page_index_map=page_index_map,
            file_suffix=file_suffix,
            source_context=source_context,
            vlm_config=vlm_config,
            source_properties=source_properties,
        )
    if source_properties is None:
        source_properties = await run_sync(read_source_properties, file_bytes, file_suffix, source_context)
    from .analysis.pdf.pipeline import aio_analyze_pdf
    from .postprocess.document import aio_model_json_to_middle_json

    result = await aio_analyze_pdf(
        file_bytes,
        effort=effort,
        parse_mode=parse_mode,
        image_analysis=image_analysis,
        vlm_config=vlm_config,
    )
    model_json = await run_sync(_build_model_json, result, file_suffix, page_index_map, source_properties)
    middle_json = await aio_model_json_to_middle_json(model_json, llm_aided_config=config.llm_aided)
    return middle_json, model_json


def _validate_analyze(effort: AnalyzeEffort, file_suffix: FileSuffix, page_index_map: list[int] | None) -> None:
    """在加载重依赖之前统一验证同步、异步入口参数。"""
    if file_suffix not in FILE_SUFFIXES:
        raise ValueError(f"Unsupported file suffix: {file_suffix!r}")
    if file_suffix != "pdf" and page_index_map:
        raise ValueError(f"page_index_map is only supported for PDF files, got {file_suffix!r}")
    if effort not in _SUPPORTED_ANALYZE_EFFORTS:
        raise ValueError(f"Unsupported analyze effort: {effort}")


def _build_model_json(
    result: AnalysisResult,
    file_suffix: FileSuffix,
    page_index_map: list[int] | None,
    source_properties: DocumentProperties,
) -> ModelJson:
    """共享模型协议构造，保持生产者、页映射和 MinerU 扩展完全一致。"""
    _log_infer_performance(file_suffix, len(result.model_list), result.elapsed)
    return ModelJson(
        pages=result.model_list,
        page_index_map=page_index_map or [],
        metadata=DocumentMetadata(
            file_suffix=file_suffix,
            producer=Producer(name="mineru", version=mineru_version),
            document=source_properties.model_copy(deep=True),
        ),
        extensions=build_metadata(effort=result.effort, parse_mode=result.parse_mode),
    )


__all__ = ["doc_analyze", "aio_doc_analyze"]
