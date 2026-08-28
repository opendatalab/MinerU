# Copyright (c) Opendatalab. All rights reserved.
"""Standalone HTML 到单页 MinerU raw model-list 的原生 converter。"""

from __future__ import annotations

import json
import re
from typing import Any, BinaryIO

from loguru import logger

from ....types import BlockType
from .._shared.markup import MarkupProjector, MarkupStylesheet
from .._shared.spans import text_spans
from .anchors import HtmlAnchorRegistry, append_referenced_notes
from .constants import MAX_HTML_BYTES, MAX_HTML_RENDERED_BYTES
from .contracts import HtmlSourceContext
from .document import HtmlDocument, parse_html_document
from .errors import HtmlResourceLimitError
from .resources import HtmlResourceContext
from .selector import select_auto_content
from .wire import decode_mineru_html_wire


class HtmlConverter:
    """把静态 HTML 转换为一个无 bbox 的逻辑页。"""

    def __init__(self) -> None:
        """初始化空页面结果。"""
        self.pages: list[list[dict[str, Any]]] = []

    def convert(
        self,
        file_binary: BinaryIO,
        *,
        source_context: HtmlSourceContext | None = None,
    ) -> None:
        """读取调用方 HTML 流，自动选择正文并生成单页 raw blocks。"""
        file_bytes = file_binary.read(MAX_HTML_BYTES + 1)
        if len(file_bytes) > MAX_HTML_BYTES:
            raise HtmlResourceLimitError(f"HTML resource limit exceeded: max_html_bytes={MAX_HTML_BYTES}")
        document = parse_html_document(file_bytes, source_context)
        resources = HtmlResourceContext(document.source_context, base_href=document.base_href)
        wire_result = decode_mineru_html_wire(document.body, resources)
        if wire_result.blocks is not None:
            blocks = wire_result.blocks
            log_values = ("mineru_exact", 1.0, 1.0, "version_1")
        else:
            if wire_result.fallback_reason is not None:
                logger.warning("MinerU HTML marker fallback reason={}", wire_result.fallback_reason)
            stylesheet = _load_stylesheet(document, resources)
            selection = select_auto_content(document.body, stylesheet)
            selected_root = append_referenced_notes(
                selection.root,
                document.body,
                stylesheet=stylesheet,
                resolve_same_document_fragment=resources.same_document_fragment,
            )
            source_key = document.source_context.source_uri or "html"
            anchors = HtmlAnchorRegistry(selected_root, stylesheet, source_key=source_key)
            resources.bind_anchors(anchors)
            blocks = MarkupProjector(
                selected_root,
                resources,
                stylesheet,
                single_document_title=True,
            ).convert()
            if not any(block.get("type") == BlockType.DOC_TITLE for block in blocks):
                if title := _document_title(document):
                    blocks.insert(0, {"type": BlockType.DOC_TITLE, "level": 1, "content": text_spans(title)})
            log_values = (
                selection.mode_used,
                selection.confidence,
                selection.retained_text_ratio,
                selection.reason,
            )
        rendered_bytes = len(json.dumps(blocks, ensure_ascii=False, separators=(",", ":")).encode())
        if rendered_bytes > MAX_HTML_RENDERED_BYTES:
            raise HtmlResourceLimitError(f"HTML projection exceeds max_html_rendered_bytes={MAX_HTML_RENDERED_BYTES}")
        logger.debug(
            "HTML content selection finished mode={} confidence={:.3f} retained_text_ratio={:.3f} reason={}",
            *log_values,
        )
        self.pages = [blocks]


def _load_stylesheet(document: HtmlDocument, resources: HtmlResourceContext) -> MarkupStylesheet:
    """按 head 文档顺序加载本地 stylesheet 与内联 style 的受支持子集。"""
    stylesheet = MarkupStylesheet()
    for source in document.stylesheets:
        if source.kind == "inline":
            resources.charge_inline_stylesheet(source.value)
            stylesheet.add(source.value)
        elif css := resources.load_stylesheet(source.value):
            stylesheet.add(css)
    return stylesheet


def _document_title(document: HtmlDocument) -> str | None:
    """按 OpenGraph/title 优先级返回去重且保守去站点后缀的标题。"""
    title = (document.open_graph_title or document.title or "").strip()
    if not title:
        return None
    site_name = (document.site_name or "").strip()
    if site_name:
        for separator in (" - ", " | ", " · ", " — ", " _ "):
            suffix = f"{separator}{site_name}"
            prefix = f"{site_name}{separator}"
            if title.casefold().endswith(suffix.casefold()):
                title = title[: -len(suffix)].strip()
                break
            if title.casefold().startswith(prefix.casefold()):
                title = title[len(prefix) :].strip()
                break
    return re.sub(r"\s+", " ", title) or None


__all__ = ["HtmlConverter"]
