# Copyright (c) Opendatalab. All rights reserved.
"""OFD 文档 model-list 生产入口。"""

from __future__ import annotations

import time
from io import BytesIO

from ...model.flash import OfdModel
from .contracts import AnalysisResult


def analyze_ofd(file_bytes: bytes) -> AnalysisResult:
    """调用原生 OFD 模型并统计整份固定版式文档转换耗时。"""
    model = OfdModel()
    infer_started_at = time.perf_counter()
    model_list = model.predict(BytesIO(file_bytes))
    infer_elapsed = time.perf_counter() - infer_started_at
    return AnalysisResult(
        model_list=model_list,
        effort="flash",
        parse_mode="txt",
        elapsed=infer_elapsed,
    )


__all__ = ["analyze_ofd"]
