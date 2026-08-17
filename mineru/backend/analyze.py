# Copyright (c) Opendatalab. All rights reserved.
"""统一 PDF 与 Office 文档分析的稳定公共门面。"""

from __future__ import annotations

import asyncio
from typing import Any, Literal, cast

from loguru import logger

from mineru.backend.analysis.contracts import AnalyzeEffort, OfficeSuffix, ParseMode
from mineru.backend.analysis.office import analyze_office
from mineru.backend.analysis.pdf.pipeline import analyze_pdf
from mineru.backend.postprocess.pages import model_list_to_pages
from mineru.types import MiddleJson
from mineru.version import __version__ as mineru_version

_SUPPORTED_FILE_SUFFIXES = {"pdf", "docx", "pptx", "xlsx"}


def _log_infer_performance(file_suffix: str, page_count: int, elapsed: float) -> None:
    """使用未舍入耗时统一记录 model-list 生产速度。"""
    speed = page_count / elapsed if elapsed > 0 else 0.0
    logger.debug(
        f"model_list infer finished, file_suffix={file_suffix}, pages={page_count}, "
        f"cost={elapsed:.6f}s, speed={speed:.3f} page/s"
    )


def doc_analyze(
    file_bytes: bytes,
    effort: Literal["flash", "low", "medium", "high", "xhigh"] = "high",
    parse_mode: Literal["auto", "txt", "ocr"] = "auto",
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    file_suffix: Literal["pdf", "docx", "pptx", "xlsx"] = "pdf",
) -> tuple[MiddleJson, list[list[dict[str, Any]]]]:
    """生产 model-list，并在统一边界构造严格 MiddleJson。"""
    if file_suffix not in _SUPPORTED_FILE_SUFFIXES:
        raise ValueError(f"Unsupported file suffix: {file_suffix!r}")

    if file_suffix == "pdf":
        result = analyze_pdf(
            file_bytes,
            effort=cast(AnalyzeEffort, effort),
            parse_mode=cast(ParseMode, parse_mode),
            image_analysis=image_analysis,
        )
    else:
        result = analyze_office(file_bytes, cast(OfficeSuffix, file_suffix))

    _log_infer_performance(file_suffix, len(result.model_list), result.elapsed)
    middle_json = MiddleJson(
        pages=model_list_to_pages(result.model_list, page_index_map),
        file_suffix=file_suffix,
        effort=result.effort,
        parse_mode=result.parse_mode,
        mineru_version=mineru_version,
    )
    return middle_json, result.model_list


async def aio_doc_analyze(
    file_bytes: bytes,
    effort: Literal["flash", "low", "medium", "high", "xhigh"] = "high",
    parse_mode: Literal["auto", "txt", "ocr"] = "auto",
    image_analysis: bool = True,
    page_index_map: list[int] | None = None,
    file_suffix: Literal["pdf", "docx", "pptx", "xlsx"] = "pdf",
) -> tuple[MiddleJson, list[list[dict[str, Any]]]]:
    """在线程中执行统一文档分析，避免阻塞调用方事件循环。"""
    return await asyncio.to_thread(
        doc_analyze,
        file_bytes=file_bytes,
        effort=effort,
        parse_mode=parse_mode,
        image_analysis=image_analysis,
        page_index_map=page_index_map,
        file_suffix=file_suffix,
    )
