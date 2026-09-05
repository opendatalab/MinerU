# Copyright (c) Opendatalab. All rights reserved.
"""提供原生文本与布局共享的字体族归一化。"""

from __future__ import annotations

import re


def _normalized_font_family(
    signature: tuple[str, int] | None,
) -> str | None:
    """移除 PDF 字体子集前缀并归一化字体族名称，供几何续行作软兼容判断。"""

    if signature is None:
        return None
    name = re.sub(r"^[A-Z]{6}\+", "", signature[0])
    return re.sub(r"[\s_-]+", "", name).casefold() or None


__all__ = ["_normalized_font_family"]
