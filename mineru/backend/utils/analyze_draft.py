# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...types import BBox


@dataclass(slots=True)
class _AnalyzeSpan:
    """Analyze 文本回填阶段使用的私有 span，不属于公开 Middle JSON 模型。"""

    type: str
    bbox: BBox
    content: str = ""
    score: float = 0.0
    image: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class _AnalyzeLine:
    """Analyze 组行阶段使用的私有 line，只承载 bbox 与私有 span。"""

    bbox: BBox
    spans: list[_AnalyzeSpan] = field(default_factory=list)
