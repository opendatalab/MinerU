# Copyright (c) Opendatalab. All rights reserved.
"""HTML Flash 解析使用的来源上下文契约。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class HtmlSourceContext:
    """保存相对链接解析及 HTML 解码所需的来源上下文。"""

    source_uri: str | None = None
    local_resource_root: Path | None = None
    transport_encoding: str | None = None


__all__ = ["HtmlSourceContext"]
