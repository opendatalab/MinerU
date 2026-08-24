# Copyright (c) Opendatalab. All rights reserved.
"""文档分析分支汇合前使用的内部结果契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

AnalyzeEffort: TypeAlias = Literal["flash", "low", "medium", "high", "xhigh"]
# 请求阶段允许自动分类，分析结果中的模式必须已经收敛为 txt 或 ocr。
ParseMode: TypeAlias = Literal["auto", "txt", "ocr"]
ResolvedParseMode: TypeAlias = Literal["txt", "ocr"]
OfficeSuffix: TypeAlias = Literal["docx", "ppt", "pptx", "xls", "xlsx"]


@dataclass(slots=True)
class AnalysisResult:
    """封装 model-list 及分析分支最终采用的元数据和计时结果。"""

    model_list: list[list[dict[str, Any]]]
    effort: AnalyzeEffort
    parse_mode: ResolvedParseMode
    elapsed: float
