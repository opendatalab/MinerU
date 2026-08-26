# Copyright (c) Opendatalab. All rights reserved.
"""EPUB 媒体类型、命名空间与固定资源上限。"""

from __future__ import annotations

from typing import Final


EPUB_MIME: Final = "application/epub+zip"
EPUB_PACKAGE_MIME: Final = "application/oebps-package+xml"
XHTML_MEDIA_TYPES: Final = frozenset({"application/xhtml+xml", "text/html"})
SVG_MEDIA_TYPE: Final = "image/svg+xml"

MAX_ENTRY_BYTES: Final = 128 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 512 * 1024 * 1024
MAX_ENTRY_COUNT: Final = 100_000
MAX_XML_DEPTH: Final = 256
MAX_XML_NODES: Final = 2_000_000
MAX_ASSET_TOTAL_BYTES: Final = 128 * 1024 * 1024

IMAGE_MEDIA_BY_EXTENSION: Final[dict[str, str]] = {
    "bmp": "image/bmp",
    "gif": "image/gif",
    "jpeg": "image/jpeg",
    "jpg": "image/jpeg",
    "png": "image/png",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "webp": "image/webp",
    "svg": SVG_MEDIA_TYPE,
}


__all__ = [
    "EPUB_MIME",
    "EPUB_PACKAGE_MIME",
    "IMAGE_MEDIA_BY_EXTENSION",
    "MAX_ASSET_TOTAL_BYTES",
    "MAX_ENTRY_BYTES",
    "MAX_ENTRY_COUNT",
    "MAX_TOTAL_BYTES",
    "MAX_XML_DEPTH",
    "MAX_XML_NODES",
    "SVG_MEDIA_TYPE",
    "XHTML_MEDIA_TYPES",
]
