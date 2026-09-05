# Copyright (c) Opendatalab. All rights reserved.
"""PDF 分析的文档级生命周期与领域阶段编排。"""

from __future__ import annotations

import time
from typing import Any, cast

from ..contracts import AnalysisResult, AnalyzeEffort, ParseMode, ResolvedParseMode
from ....config import VlmConfig
from ....model.runtime.hybrid import HybridLocalModelContextSingleton
from ....model.runtime.memory import clean_memory
from ....model.vlm.client import get_vlm_predictor
from ....model.flash.pdf.document import PDFDocument

from .normalization import _normalize_pdf_model_list
from .window import process_pdf_windows

_SUPPORTED_PDF_EFFORTS = {"flash", "medium", "high", "xhigh"}


def analyze_pdf(
    file_bytes: bytes,
    effort: AnalyzeEffort = "high",
    parse_mode: ParseMode = "auto",
    image_analysis: bool = True,
    vlm_config: VlmConfig | None = None,
) -> AnalysisResult:
    """生产 PDF model-list，并返回最终路由元数据和精确推理耗时。"""
    # 只在真实 PDF 分析开始时配置 MPS 回退，避免 import backend 修改进程环境。
    import os

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if effort not in _SUPPORTED_PDF_EFFORTS:
        raise ValueError(f"Unsupported analyze effort: {effort}")

    document = PDFDocument(file_bytes)
    hybrid_model = None
    try:
        if parse_mode == "auto":
            parse_mode = document.classify()
        if parse_mode not in ["txt", "ocr"]:
            raise ValueError(f"parse_mode {parse_mode} is not supported")
        resolved_parse_mode = cast(ResolvedParseMode, parse_mode)
        flash_txt_mode = effort == "flash" and resolved_parse_mode == "txt"
        vlm_predictor = None

        if not flash_txt_mode:
            hybrid_model_singleton = HybridLocalModelContextSingleton()
            hybrid_model = hybrid_model_singleton.get_model()

            if effort in ["high", "xhigh"]:
                vlm_predictor, _vlm_backend = get_vlm_predictor(vlm_config)
            else:
                vlm_predictor = None

        infer_started_at = time.perf_counter()
        model_list: list[list[dict[str, Any]]] = process_pdf_windows(
            file_bytes,
            document,
            effort=effort,
            parse_mode=resolved_parse_mode,
            image_analysis=image_analysis,
            flash_txt_mode=flash_txt_mode,
            hybrid_model=hybrid_model,
            vlm_predictor=vlm_predictor,
        )

        # 仅 PDF 模型结果需要统一清理块元数据并规范化行内公式。
        _normalize_pdf_model_list(model_list)
        infer_elapsed = time.perf_counter() - infer_started_at

    finally:
        try:
            document.close()
        finally:
            # 无论窗口处理是否异常，都释放已初始化的 Hybrid 模型资源。
            if hybrid_model is not None:
                clean_memory(hybrid_model.device)

    return AnalysisResult(
        model_list=model_list,
        effort=effort,
        parse_mode=resolved_parse_mode,
        elapsed=infer_elapsed,
    )
