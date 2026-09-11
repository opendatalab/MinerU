# Copyright (c) Opendatalab. All rights reserved.
"""PDF 分析的文档级生命周期与领域阶段编排。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import cast

from docvortex.document.pdf import PDFDocument

from ....config import VlmConfig
from ....model.runtime.execution import acquire_document, release_document
from ....utils.async_utils import run_sync
from ....model.vlm.contracts import VlmPredictor
from ....model.runtime.hybrid import HybridLocalModelContext, HybridLocalModelContextSingleton
from ....model.runtime.memory import clean_memory, trim_process_heap
from ....model.vlm.client import get_vlm_predictor
from ..contracts import AnalysisResult, AnalyzeEffort, ParseMode, ResolvedParseMode
from .normalization import _normalize_pdf_model_list
from .window import aio_process_pdf_windows, process_pdf_windows

_SUPPORTED_PDF_EFFORTS = {"flash", "medium", "high", "xhigh"}


@dataclass
class _PDFAnalysis:
    """显式持有准备阶段部分完成时的资源，保证取消和初始化错误也能清理。"""

    document: PDFDocument | None = None
    hybrid_model: HybridLocalModelContext | None = None
    predictor: VlmPredictor | None = None
    parse_mode: ResolvedParseMode = "txt"
    flash_txt_mode: bool = False


def _prepare_analysis(
    state: _PDFAnalysis,
    file_bytes: bytes,
    effort: AnalyzeEffort,
    parse_mode: ParseMode,
    vlm_config: VlmConfig | None,
) -> None:
    """创建 PDF、解析模式和模型，所有阶段共享同一文档级资源状态。"""
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if effort not in _SUPPORTED_PDF_EFFORTS:
        raise ValueError(f"Unsupported analyze effort: {effort}")
    state.document = PDFDocument(file_bytes)
    if parse_mode == "auto":
        parse_mode = state.document.classify()
    if parse_mode not in ["txt", "ocr"]:
        raise ValueError(f"parse_mode {parse_mode} is not supported")
    state.parse_mode = cast(ResolvedParseMode, parse_mode)
    state.flash_txt_mode = effort == "flash" and state.parse_mode == "txt"
    if not state.flash_txt_mode:
        state.hybrid_model = HybridLocalModelContextSingleton().get_model()
        acquire_document(state.hybrid_model.device)
        if effort in {"high", "xhigh"}:
            state.predictor, _backend = get_vlm_predictor(vlm_config)


def _close_analysis(state: _PDFAnalysis) -> None:
    """先释放文档，最后一份相关文档退出后再回收共享设备缓存。"""
    try:
        if state.document is not None:
            state.document.close()
    finally:
        try:
            if state.hybrid_model is not None:
                release_document(state.hybrid_model.device, clean_memory)
        finally:
            trim_process_heap()


def analyze_pdf(
    file_bytes: bytes,
    effort: AnalyzeEffort = "high",
    parse_mode: ParseMode = "auto",
    image_analysis: bool = True,
    vlm_config: VlmConfig | None = None,
) -> AnalysisResult:
    """使用共享资源生命周期与同步窗口编排生产 PDF 模型结果。"""
    state = _PDFAnalysis()
    try:
        _prepare_analysis(state, file_bytes, effort, parse_mode, vlm_config)
        infer_started_at = time.perf_counter()
        model_list = process_pdf_windows(
            file_bytes,
            state.document,
            effort=effort,
            parse_mode=state.parse_mode,
            image_analysis=image_analysis,
            flash_txt_mode=state.flash_txt_mode,
            hybrid_model=state.hybrid_model,
            vlm_predictor=state.predictor,
        )
        _normalize_pdf_model_list(model_list)
        infer_elapsed = time.perf_counter() - infer_started_at
    finally:
        _close_analysis(state)
    return AnalysisResult(model_list=model_list, effort=effort, parse_mode=state.parse_mode, elapsed=infer_elapsed)


async def aio_analyze_pdf(
    file_bytes: bytes,
    effort: AnalyzeEffort = "high",
    parse_mode: ParseMode = "auto",
    image_analysis: bool = True,
    vlm_config: VlmConfig | None = None,
) -> AnalysisResult:
    """原生异步调度 VLM；同步准备、回填与清理使用取消安全的线程边界。"""
    state = _PDFAnalysis()
    try:
        await run_sync(_prepare_analysis, state, file_bytes, effort, parse_mode, vlm_config)
        if state.document is None or state.hybrid_model is None or state.predictor is None:
            raise ValueError("Native async PDF analysis requires a high/xhigh VLM pipeline")
        infer_started_at = time.perf_counter()
        model_list = await aio_process_pdf_windows(
            file_bytes,
            state.document,
            effort=effort,
            parse_mode=state.parse_mode,
            image_analysis=image_analysis,
            hybrid_model=state.hybrid_model,
            vlm_predictor=state.predictor,
        )
        await run_sync(_normalize_pdf_model_list, model_list)
        infer_elapsed = time.perf_counter() - infer_started_at
    finally:
        await run_sync(_close_analysis, state)
    return AnalysisResult(model_list=model_list, effort=effort, parse_mode=state.parse_mode, elapsed=infer_elapsed)


__all__ = ["analyze_pdf", "aio_analyze_pdf"]
