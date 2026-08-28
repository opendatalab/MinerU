# Copyright (c) Opendatalab. All rights reserved.
"""OFD 固定版式 Flash 解析入口。"""

from .errors import OfdEncryptedError, OfdParseError, OfdResourceLimitError
from .metadata import extract_ofd_metadata
from .package import detect_ofd, detect_ofd_path

__all__ = [
    "OfdEncryptedError",
    "OfdParseError",
    "OfdResourceLimitError",
    "detect_ofd",
    "detect_ofd_path",
    "extract_ofd_metadata",
]
