# Copyright (c) Opendatalab. All rights reserved.
"""兼容导出位于 utils 层的共享超链接安全策略。"""

from __future__ import annotations

from ....utils.hyperlink import (
    DEFAULT_EXTERNAL_HYPERLINK_SCHEMES,
    OFFICE_EXTERNAL_HYPERLINK_SCHEMES,
    sanitize_hyperlink_target,
)


__all__ = [
    "DEFAULT_EXTERNAL_HYPERLINK_SCHEMES",
    "OFFICE_EXTERNAL_HYPERLINK_SCHEMES",
    "sanitize_hyperlink_target",
]
