# Copyright (c) Opendatalab. All rights reserved.
"""OFD 原生解析错误类型。"""


class OfdParseError(ValueError):
    """表示 OFD 包结构或必需内容不合法。"""


class OfdEncryptedError(OfdParseError):
    """表示 OFD 包或成员使用了不支持的加密。"""


class OfdResourceLimitError(OfdParseError):
    """表示 OFD 输入超过固定资源预算。"""


__all__ = ["OfdEncryptedError", "OfdParseError", "OfdResourceLimitError"]
