# Copyright (c) Opendatalab. All rights reserved.
"""PDF 分析的文档级生命周期与领域阶段编排。"""

from __future__ import annotations

import os
import time
from typing import Any, Literal

from mineru.backend.analysis.contracts import AnalysisResult
from mineru.backend.local_model_runtime import HybridLocalModelContextSingleton
from mineru.utils.engine_utils import get_vlm_engine
from mineru.utils.model_utils import clean_memory
from mineru.utils.pdf_document import PDFDocument

from .layout import _load_vlm_runtime
from .normalization import _normalize_pdf_model_list
from .window import process_pdf_windows

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

_SUPPORTED_PDF_EFFORTS = {"flash", "low", "medium", "high", "xhigh"}


def analyze_pdf(
    file_bytes: bytes,
    effort: Literal["flash", "low", "medium", "high", "xhigh"] = "high",
    parse_mode: Literal["auto", "txt", "ocr"] = "auto",
    image_analysis: bool = True,
) -> AnalysisResult:
    """生产 PDF model-list，并返回最终路由元数据和精确推理耗时。"""
    document = PDFDocument(file_bytes)
    document_closed = False
    hybrid_model = None
    model_list: list[list[dict[str, Any]]] = []
    try:
        if parse_mode == "auto":
            parse_mode = document.classify()
        if parse_mode not in ["txt", "ocr"]:
            raise ValueError(f"parse_mode {parse_mode} is not supported")
        if effort not in _SUPPORTED_PDF_EFFORTS:
            raise ValueError(f"Unsupported analyze effort: {effort}")

        # Flash 只处理原生文本，OCR 文档继续复用 Hybrid low 流程。
        if effort == "flash" and parse_mode == "ocr":
            effort = "low"

        flash_txt_mode = effort == "flash" and parse_mode == "txt"
        vlm_predictor = None

        if not flash_txt_mode:
            hybrid_model_singleton = HybridLocalModelContextSingleton()
            hybrid_model = hybrid_model_singleton.get_model()

            if effort in ["high", "xhigh"]:
                vlm_runtime = _load_vlm_runtime()
                vlm_backend = get_vlm_engine(inference_engine="auto", is_async=False)
                vlm_predictor = vlm_runtime["ModelSingleton"]().get_model(
                    backend=vlm_backend,
                    model_path=None,
                    server_url=None,
                )
                vlm_predictor = vlm_runtime["_maybe_enable_serial_execution"](vlm_predictor, vlm_backend)
            else:
                vlm_predictor = None

        infer_started_at = time.perf_counter()
        model_list = process_pdf_windows(
            file_bytes,
            document,
            effort=effort,
            parse_mode=parse_mode,  # type: ignore[arg-type]
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
            if not document_closed:
                document.close()
                document_closed = True
        finally:
            # 无论窗口处理是否异常，都释放已初始化的 Hybrid 模型资源。
            if hybrid_model is not None:
                clean_memory(hybrid_model.device)

    return AnalysisResult(
        model_list=model_list,
        effort=effort,
        parse_mode=parse_mode,
        elapsed=infer_elapsed,
    )
