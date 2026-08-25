# Copyright (c) Opendatalab. All rights reserved.
"""统一 PDF 与 Office 文档分析的稳定公共门面。"""

from __future__ import annotations

import asyncio
from typing import cast

from loguru import logger

from .analysis.contracts import AnalyzeEffort, OfficeSuffix, ParseMode
from ..config import config
from ..types import FILE_SUFFIXES, FileSuffix, MiddleJson, ModelJson
from ..version import __version__ as mineru_version

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
) -> tuple[MiddleJson, ModelJson]:
    """生产严格 ModelJson，并在统一边界构造严格 MiddleJson。"""
    if file_suffix not in FILE_SUFFIXES:
        raise ValueError(f"Unsupported file suffix: {file_suffix!r}")
    if effort not in _SUPPORTED_ANALYZE_EFFORTS:
        raise ValueError(f"Unsupported analyze effort: {effort}")

    if file_suffix == "pdf":
        from .analysis.pdf.pipeline import analyze_pdf

        result = analyze_pdf(
            file_bytes,
            effort=effort,
            parse_mode=parse_mode,
            image_analysis=image_analysis,
        )
    else:
        from .analysis.office import analyze_office

        result = analyze_office(file_bytes, cast(OfficeSuffix, file_suffix))

    _log_infer_performance(file_suffix, len(result.model_list), result.elapsed)
    model_json = ModelJson(
        pages=result.model_list,
        page_index_map=page_index_map or [],
        file_suffix=file_suffix,
        effort=result.effort,
        parse_mode=result.parse_mode,
        mineru_version=mineru_version,
    )
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
) -> tuple[MiddleJson, ModelJson]:
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
