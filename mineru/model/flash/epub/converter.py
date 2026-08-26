# Copyright (c) Opendatalab. All rights reserved.
"""EPUB OCF/OPF/spine 到 MinerU raw model-list 的原生 converter。"""

from __future__ import annotations

from typing import Any, BinaryIO

from loguru import logger
from lxml import etree  # type: ignore[reportMissingImports]

from .constants import SVG_MEDIA_TYPE, XHTML_MEDIA_TYPES
from .errors import EpubEncryptedError, EpubParseError, EpubResourceLimitError
from .package import EpubPackage
from .toc import build_epub_toc_index
from .xhtml import EpubChapterConverter, build_anchor_registry, convert_svg_spine


class EpubConverter:
    """把 EPUB spine 转换为稳定的逐逻辑页 raw blocks。"""

    def __init__(self) -> None:
        """初始化空页面结果。"""
        self.pages: list[list[dict[str, Any]]] = []

    def convert(self, file_binary: BinaryIO) -> None:
        """读取调用方 EPUB 流，转换整本内容并保持输入流所有权。"""
        package = EpubPackage(file_binary.read())
        try:
            parsed: list[tuple[int, str, str, etree._Element | None]] = []
            readable_count = 0
            for index, spine_item in enumerate(package.content_spine_items()):
                if spine_item.path is None or spine_item.media_type is None:
                    parsed.append((index, "", "", None))
                    logger.warning("Skipping unsupported EPUB spine item index={} idref={!r}", index, spine_item.idref)
                    continue
                try:
                    root = package.xml_part(spine_item.path, allow_external_doctype=True)
                except (EpubEncryptedError, EpubResourceLimitError):
                    raise
                except EpubParseError as exc:
                    logger.warning("Skipping corrupt EPUB spine item index={} path={!r}: {}", index, spine_item.path, exc)
                    root = None
                if root is not None:
                    readable_count += 1
                else:
                    logger.warning("Skipping unreadable EPUB spine item index={} path={!r}", index, spine_item.path)
                parsed.append((index, spine_item.path, spine_item.media_type, root))

            if readable_count == 0:
                raise EpubParseError("Malformed EPUB package: no selected spine content could be read")

            xhtml_chapters = [
                (path, root)
                for _, path, media_type, root in parsed
                if root is not None and media_type in XHTML_MEDIA_TYPES
            ]
            anchors = build_anchor_registry(xhtml_chapters, package)
            toc_index = build_epub_toc_index(package, anchors)
            pages: list[list[dict[str, Any]]] = [[toc_index]] if toc_index is not None else []
            for index, path, media_type, root in parsed:
                if root is None:
                    pages.append([])
                    continue
                try:
                    if media_type in XHTML_MEDIA_TYPES:
                        blocks = EpubChapterConverter(package, path, root, anchors).convert()
                    elif media_type == SVG_MEDIA_TYPE:
                        blocks = convert_svg_spine(package, path, root)
                    else:
                        blocks = []
                except (EpubEncryptedError, EpubResourceLimitError):
                    raise
                except Exception as exc:
                    logger.warning("Skipping unusable EPUB spine item index={} path={!r}: {}", index, path, exc)
                    blocks = []
                pages.append(blocks)
            self.pages = pages
        finally:
            package.close()


__all__ = ["EpubConverter"]
