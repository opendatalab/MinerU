# Copyright (c) Opendatalab. All rights reserved.
"""从 EPUB OPF 提取 doclib 使用的基础元数据。"""

from __future__ import annotations

from typing import BinaryIO

from .package import EpubPackage


def _epub_output_page_count(package: EpubPackage) -> int:
    """按 OPF spine 项数量返回稳定的逻辑页数。"""
    return len(package.spine)


def extract_epub_metadata(file_binary: BinaryIO) -> dict[str, object | None]:
    """读取 EPUB 标题、作者、主题、关键词和 spine 逻辑页数。"""
    package = EpubPackage(file_binary.read())
    try:
        metadata = package.metadata
        return {
            "page_count": _epub_output_page_count(package),
            "title": metadata.title,
            "author": metadata.author,
            "subject": metadata.subject,
            "keywords": metadata.keywords,
        }
    finally:
        package.close()


__all__ = ["extract_epub_metadata"]
