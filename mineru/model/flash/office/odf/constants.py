# Copyright (c) Opendatalab. All rights reserved.
"""OpenDocument 命名空间、MIME 与固定资源上限。"""

from __future__ import annotations

from typing import Final, Literal, TypeAlias


OdfSuffix: TypeAlias = Literal["odt", "ods", "odp"]

NS: Final[dict[str, str]] = {
    "chart": "urn:oasis:names:tc:opendocument:xmlns:chart:1.0",
    "dc": "http://purl.org/dc/elements/1.1/",
    "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0",
    "fo": "urn:oasis:names:tc:opendocument:xmlns:xsl-fo-compatible:1.0",
    "form": "urn:oasis:names:tc:opendocument:xmlns:form:1.0",
    "manifest": "urn:oasis:names:tc:opendocument:xmlns:manifest:1.0",
    "math": "http://www.w3.org/1998/Math/MathML",
    "meta": "urn:oasis:names:tc:opendocument:xmlns:meta:1.0",
    "number": "urn:oasis:names:tc:opendocument:xmlns:datastyle:1.0",
    "office": "urn:oasis:names:tc:opendocument:xmlns:office:1.0",
    "presentation": "urn:oasis:names:tc:opendocument:xmlns:presentation:1.0",
    "style": "urn:oasis:names:tc:opendocument:xmlns:style:1.0",
    "svg": "urn:oasis:names:tc:opendocument:xmlns:svg-compatible:1.0",
    "table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
    "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
    "xlink": "http://www.w3.org/1999/xlink",
    "xml": "http://www.w3.org/XML/1998/namespace",
}

ODF_MIME_BY_SUFFIX: Final[dict[OdfSuffix, str]] = {
    "odt": "application/vnd.oasis.opendocument.text",
    "ods": "application/vnd.oasis.opendocument.spreadsheet",
    "odp": "application/vnd.oasis.opendocument.presentation",
}
ODF_SUFFIX_BY_MIME: Final[dict[str, OdfSuffix]] = {
    mime: suffix for suffix, mime in ODF_MIME_BY_SUFFIX.items()
}
ODF_BODY_BY_SUFFIX: Final[dict[OdfSuffix, str]] = {
    "odt": "text",
    "ods": "spreadsheet",
    "odp": "presentation",
}

MAX_ENTRY_BYTES: Final = 128 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 512 * 1024 * 1024
MAX_ENTRY_COUNT: Final = 100_000
MAX_XML_DEPTH: Final = 256
MAX_XML_NODES: Final = 2_000_000
MAX_GRID_SLOTS: Final = 4_000_000
MAX_EXPANSION_TEXT_BYTES: Final = 64 * 1024 * 1024
MAX_ASSET_TOTAL_BYTES: Final = 128 * 1024 * 1024


def qname(prefix: str, local_name: str) -> str:
    """返回指定 ODF 命名空间下的 Clark notation 标签名。"""
    return f"{{{NS[prefix]}}}{local_name}"


__all__ = [
    "MAX_ASSET_TOTAL_BYTES",
    "MAX_ENTRY_BYTES",
    "MAX_ENTRY_COUNT",
    "MAX_EXPANSION_TEXT_BYTES",
    "MAX_GRID_SLOTS",
    "MAX_TOTAL_BYTES",
    "MAX_XML_DEPTH",
    "MAX_XML_NODES",
    "NS",
    "ODF_BODY_BY_SUFFIX",
    "ODF_MIME_BY_SUFFIX",
    "ODF_SUFFIX_BY_MIME",
    "OdfSuffix",
    "qname",
]
