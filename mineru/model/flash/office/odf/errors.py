# Copyright (c) Opendatalab. All rights reserved.
"""OpenDocument 内部稳定错误类型。"""

from __future__ import annotations


class OdfParseError(ValueError):
    """表示 OpenDocument 包或语义结构不可解析。"""


class OdfResourceLimitError(OdfParseError):
    """表示 OpenDocument 输入超过固定安全边界。"""


class OdfEncryptedError(OdfParseError):
    """表示 OpenDocument 包包含不支持的加密成员。"""


__all__ = ["OdfEncryptedError", "OdfParseError", "OdfResourceLimitError"]
