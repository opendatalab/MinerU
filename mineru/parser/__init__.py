# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from pathlib import Path
from typing import Literal

from ..model.flash.html import HtmlSourceContext
from ..types import Tier
from .api_client import ApiJobStatus, MinerUApiParser
from .base import MIDDLE_JSON_SCHEMA_VERSION, DocumentParser, ParseResult
from .mineru_parser import MinerUParser
from .tier import PARSER_BACKENDS, backend_for_tier

__all__ = [
    "ApiJobStatus",
    "backend_for_tier",
    "PARSER_BACKENDS",
    "DocumentParser",
    "MinerUApiParser",
    "MinerUParser",
    "MIDDLE_JSON_SCHEMA_VERSION",
    "ParseResult",
    "parse",
    "parse_async",
]


def parse(
    path: str | Path,
    *,
    tier: Tier = "standard",
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
    image_analysis: bool = True,
    page_range: str = "",
    source_context: HtmlSourceContext | None = None,
) -> ParseResult:
    """同步解析文档；source_context 仅供保留 HTML 原始来源的内部调用方使用。"""
    parser = MinerUParser(tier=tier, parse_mode=ocr_mode, image_analysis=image_analysis)
    return parser.parse(path, page_range=page_range, source_context=source_context)


async def parse_async(
    path: str | Path,
    *,
    tier: Tier = "standard",
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
    image_analysis: bool = True,
    page_range: str = "",
    source_context: HtmlSourceContext | None = None,
) -> ParseResult:
    """异步解析文档；source_context 仅供保留 HTML 原始来源的内部调用方使用。"""
    parser = MinerUParser(tier=tier, parse_mode=ocr_mode, image_analysis=image_analysis)
    return await parser.parse_async(path, page_range=page_range, source_context=source_context)
