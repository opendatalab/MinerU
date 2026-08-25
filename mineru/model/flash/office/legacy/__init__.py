# Copyright (c) Opendatalab. All rights reserved.

"""旧版 Office 二进制格式共享基础设施。"""

from .errors import (
    LegacyOfficeEncryptedError,
    LegacyOfficeMalformedError,
    LegacyOfficeMissingPartError,
    LegacyOfficeResourceLimitError,
)
from .ole import BoundedOleReader

__all__ = [
    "BoundedOleReader",
    "LegacyOfficeEncryptedError",
    "LegacyOfficeMalformedError",
    "LegacyOfficeMissingPartError",
    "LegacyOfficeResourceLimitError",
]
