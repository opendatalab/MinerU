# Copyright (c) Opendatalab. All rights reserved.
"""EPUB 文档 model-list 生产入口。"""

from __future__ import annotations

import time
from io import BytesIO

from ...model.flash import EpubModel
from .contracts import AnalysisResult


def analyze_epub(file_bytes: bytes) -> AnalysisResult:
    """调用轻量 EPUB 模型，并精确统计整本目录与 spine 内容的转换耗时。"""
    epub_model = EpubModel()
    infer_started_at = time.perf_counter()
    model_list = epub_model.predict(BytesIO(file_bytes))
    infer_elapsed = time.perf_counter() - infer_started_at
    return AnalysisResult(
        model_list=model_list,
        effort="flash",
        parse_mode="txt",
        elapsed=infer_elapsed,
    )


__all__ = ["analyze_epub"]
