# Copyright (c) Opendatalab. All rights reserved.
"""MinerU HTML v1 canonical wire 的轻量内部入口。"""

from __future__ import annotations

from lxml import etree  # type: ignore[reportMissingImports]

from ..resources import HtmlResourceContext
from .contracts import MINERU_HTML_VERSION, WireDecodeResult


def decode_mineru_html_wire(body: etree._Element, resources: HtmlResourceContext) -> WireDecodeResult:
    """只对 canonical v1 wire 精确解码，其余输入返回通用投影信号。"""
    from .materializer import materialize_mineru_html_wire
    from .parser import parse_mineru_html_wire

    plan, fallback_reason = parse_mineru_html_wire(body)
    if plan is None:
        return WireDecodeResult(None, fallback_reason)
    return WireDecodeResult(materialize_mineru_html_wire(plan, resources))


__all__ = ["MINERU_HTML_VERSION", "WireDecodeResult", "decode_mineru_html_wire"]
