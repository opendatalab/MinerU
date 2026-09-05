# Copyright (c) Opendatalab. All rights reserved.
"""共享原生文档格式识别的稳定公共入口。"""

from __future__ import annotations
from pathlib import Path
from docgale.document import detection


def guess_suffix_by_bytes(file_bytes: bytes, file_path: str | None = None) -> str:
    """读取文件签名和容器结构，保持已有可选文件名提示。"""
    return detection.guess_suffix_by_bytes(file_bytes, file_path)


def guess_suffix_by_path(file_path: str | Path) -> str:
    """将本地文件识别委托给共享的原生文档访问层。"""
    return detection.guess_suffix_by_path(file_path)


__all__ = ["guess_suffix_by_bytes", "guess_suffix_by_path"]
