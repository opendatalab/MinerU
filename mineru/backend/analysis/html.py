# Copyright (c) Opendatalab. All rights reserved.
"""HTML 文档 model-list 生产入口。"""

from __future__ import annotations

import time
from io import BytesIO

from ...model.flash import HtmlModel
from ...model.flash.html import HtmlSourceContext
from .contracts import AnalysisResult


def analyze_html(
    file_bytes: bytes,
    *,
    source_context: HtmlSourceContext | None = None,
) -> AnalysisResult:
    """调用轻量 HTML 模型，并精确统计静态 DOM 转换耗时。"""
    html_model = HtmlModel()
    infer_started_at = time.perf_counter()
    model_list = html_model.predict(BytesIO(file_bytes), source_context=source_context)
    infer_elapsed = time.perf_counter() - infer_started_at
    return AnalysisResult(
        model_list=model_list,
        effort="flash",
        parse_mode="txt",
        elapsed=infer_elapsed,
    )


__all__ = ["analyze_html"]
