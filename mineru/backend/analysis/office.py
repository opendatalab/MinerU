# Copyright (c) Opendatalab. All rights reserved.
"""Office 文档 model-list 生产入口。"""

from __future__ import annotations

import time
from io import BytesIO

from mineru.model.flash import DocxModel, PptxModel, XlsxModel

from .contracts import AnalysisResult, OfficeSuffix

_OFFICE_MODEL_MAP = {
    "docx": DocxModel,
    "pptx": PptxModel,
    "xlsx": XlsxModel,
}


def analyze_office(file_bytes: bytes, file_suffix: OfficeSuffix) -> AnalysisResult:
    """调用对应 Flash Office 模型并精确统计 predict 阶段耗时。"""
    office_model = _OFFICE_MODEL_MAP[file_suffix]()
    infer_started_at = time.perf_counter()
    model_list = office_model.predict(BytesIO(file_bytes))
    infer_elapsed = time.perf_counter() - infer_started_at
    return AnalysisResult(
        model_list=model_list,
        effort="flash",
        parse_mode="txt",
        elapsed=infer_elapsed,
    )
