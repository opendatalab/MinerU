# Copyright (c) Opendatalab. All rights reserved.
"""OFD 命名空间、版本与安全资源上限。"""

from __future__ import annotations

from typing import Final


OFD_NAMESPACE: Final = "http://www.ofdspec.org/2016"
OFD_LEGACY_NAMESPACE: Final = "http://www.ofdspec.org"
OFD_NAMESPACES: Final = frozenset({OFD_NAMESPACE, OFD_LEGACY_NAMESPACE})
OFD_KNOWN_VERSIONS: Final = frozenset({"1.0", "1.1", "1.2"})

MAX_ENTRY_BYTES: Final = 128 * 1024 * 1024
MAX_TOTAL_BYTES: Final = 512 * 1024 * 1024
MAX_ENTRY_COUNT: Final = 100_000
MAX_XML_DEPTH: Final = 256
MAX_XML_NODES: Final = 2_000_000
MAX_ASSET_TOTAL_BYTES: Final = 128 * 1024 * 1024
MAX_EXPANDED_TEXT_BYTES: Final = 64 * 1024 * 1024
MAX_EXPANDED_GLYPHS: Final = 2_000_000
MAX_PATH_COMMANDS: Final = 2_000_000
MAX_OBJECT_RECURSION: Final = 64
MAX_FONT_BYTES: Final = 32 * 1024 * 1024

MM_TO_POINTS: Final = 72.0 / 25.4


__all__ = [
    "MAX_ASSET_TOTAL_BYTES",
    "MAX_ENTRY_BYTES",
    "MAX_ENTRY_COUNT",
    "MAX_EXPANDED_GLYPHS",
    "MAX_EXPANDED_TEXT_BYTES",
    "MAX_FONT_BYTES",
    "MAX_OBJECT_RECURSION",
    "MAX_PATH_COMMANDS",
    "MAX_TOTAL_BYTES",
    "MAX_XML_DEPTH",
    "MAX_XML_NODES",
    "MM_TO_POINTS",
    "OFD_KNOWN_VERSIONS",
    "OFD_LEGACY_NAMESPACE",
    "OFD_NAMESPACE",
    "OFD_NAMESPACES",
]
