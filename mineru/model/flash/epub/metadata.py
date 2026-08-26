# Copyright (c) Opendatalab. All rights reserved.
"""从 EPUB OPF 提取 doclib 使用的基础元数据。"""

from __future__ import annotations

from typing import BinaryIO

from .constants import XHTML_MEDIA_TYPES
from .package import EpubPackage
from .toc import build_epub_toc_index
from .xhtml import build_anchor_registry


def _epub_output_page_count(package: EpubPackage) -> int:
    """按去重后的正文 spine 和可生成目录页计算最终输出页数。"""
    chapters = []
    for item in package.content_spine_items():
        if item.path is None or item.media_type not in XHTML_MEDIA_TYPES:
            continue
        root = package.xml_part(item.path, allow_external_doctype=True)
        if root is not None:
            chapters.append((item.path, root))
    anchors = build_anchor_registry(chapters, package)
    toc_page_count = 1 if build_epub_toc_index(package, anchors) is not None else 0
    return len(package.content_spine_items()) + toc_page_count


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
