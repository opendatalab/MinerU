# Copyright (c) Opendatalab. All rights reserved.
"""ocr-mineru skill 公共 API"""

from mineru.skill.core import ParseOptions, parse_file, parse_file_sync
from mineru.skill.result import ParseResult

__all__ = [
    "ParseOptions",
    "ParseResult",
    "parse_file",
    "parse_file_sync",
]
