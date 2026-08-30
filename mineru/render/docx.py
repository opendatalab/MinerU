# Copyright (c) Opendatalab. All rights reserved.
"""严格 MiddleJson 到 DOCX 的轻量公共门面与稳定异常。"""

from __future__ import annotations

from ..types import MiddleJson
from .contracts import AssetResolver


class DocxRenderError(RuntimeError):
    """表示 renderer 无法在不丢失必需内容的前提下生成 DOCX。"""

    def __init__(
        self,
        message: str,
        *,
        page_idx: int,
        block_index: int | None,
        block_type: str,
    ) -> None:
        """保存错误消息及稳定的 page/block 定位字段。"""
        self.page_idx = page_idx
        self.block_index = block_index
        self.block_type = block_type
        super().__init__(f"{message} (page_idx={page_idx}, block_index={block_index}, block_type={block_type})")


def render_docx(
    middle_json: MiddleJson,
    *,
    asset_resolver: AssetResolver | None = None,
) -> bytes:
    """惰性加载 DOCX 实现并渲染严格 MiddleJson。"""
    from ._internal.docx.renderer import render_docx as _render_docx

    return _render_docx(middle_json, asset_resolver=asset_resolver)


__all__ = ["DocxRenderError", "render_docx"]
