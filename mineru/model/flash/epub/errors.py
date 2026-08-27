# Copyright (c) Opendatalab. All rights reserved.
"""EPUB 解析器内部使用的稳定错误类型。"""


class EpubError(ValueError):
    """所有 EPUB 解析错误的共同基类。"""


class EpubParseError(EpubError):
    """表示 EPUB 容器或正文结构不可用。"""


class EpubEncryptedError(EpubError):
    """表示解析所需的 EPUB 资源已加密。"""


class EpubResourceLimitError(EpubError):
    """表示 EPUB 输入超过固定资源预算。"""


__all__ = ["EpubEncryptedError", "EpubError", "EpubParseError", "EpubResourceLimitError"]
