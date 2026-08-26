# Copyright (c) Opendatalab. All rights reserved.

"""Flash Office 二进制、嵌入对象与 RTF 解析共享的稳定错误类型。"""

from __future__ import annotations


class LegacyOfficeError(ValueError):
    """旧版 Office 解析错误基类，并携带稳定错误码。"""

    code = "legacy_office_error"


class LegacyOfficeMalformedError(LegacyOfficeError):
    """输入容器或核心二进制记录无法形成有效文档。"""

    code = "malformed"


class LegacyOfficeMissingPartError(LegacyOfficeError):
    """缺少完成解析所必需的 OLE stream。"""

    code = "missing_part"


class LegacyOfficeEncryptedError(LegacyOfficeError):
    """输入使用了当前纯 Python 解析链不支持的加密。"""

    code = "encrypted"


class LegacyOfficeResourceLimitError(LegacyOfficeError):
    """输入超过固定安全限制。"""

    code = "resource_limit"
