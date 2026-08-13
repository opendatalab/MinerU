# Copyright (c) Opendatalab. All rights reserved.
"""文档分析分支汇合前使用的内部结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

AnalyzeEffort: TypeAlias = Literal["flash", "low", "medium", "high", "xhigh"]
ParseMode: TypeAlias = Literal["auto", "txt", "ocr"]
OfficeSuffix: TypeAlias = Literal["docx", "pptx", "xlsx"]


@dataclass(slots=True)
class AnalysisResult:
    """封装 model-list 及分析分支最终采用的元数据和计时结果。"""

    model_list: list[list[dict[str, Any]]]
    effort: AnalyzeEffort
    parse_mode: ParseMode
    elapsed: float
