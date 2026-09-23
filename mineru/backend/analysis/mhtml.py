# Copyright (c) Opendatalab. All rights reserved.
"""MHTML 文档的 DocVortex 原生分析适配。"""

from __future__ import annotations

from docvortex import analyze
from docvortex.document.contracts import HtmlSourceContext

from .contracts import AnalysisResult


def analyze_mhtml(file_bytes: bytes, *, source_context: HtmlSourceContext | None = None) -> AnalysisResult:
    """调用公开原生入口，仅把 model-list 交给 MinerU 统一后处理。"""
    native = analyze(file_bytes, file_suffix="mhtml", source_context=source_context)
    return AnalysisResult(
        model_list=native.model_json.pages,
        effort="flash",
        parse_mode="txt",
        elapsed=native.elapsed_seconds,
    )


__all__ = ["analyze_mhtml"]
