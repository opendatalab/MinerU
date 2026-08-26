# Copyright (c) Opendatalab. All rights reserved.
"""CSV 文档 model-list 生产入口。"""

from __future__ import annotations

import time
from io import BytesIO

from ...model.flash import CsvModel
from .contracts import AnalysisResult


def analyze_csv(file_bytes: bytes) -> AnalysisResult:
    """调用轻量 CSV 模型并精确统计 predict 阶段耗时。"""
    csv_model = CsvModel()
    infer_started_at = time.perf_counter()
    model_list = csv_model.predict(BytesIO(file_bytes))
    infer_elapsed = time.perf_counter() - infer_started_at
    return AnalysisResult(
        model_list=model_list,
        effort="flash",
        parse_mode="txt",
        elapsed=infer_elapsed,
    )


__all__ = ["analyze_csv"]
