# Copyright (c) Opendatalab. All rights reserved.
"""WMF/EMF 内部渲染入口。"""

from __future__ import annotations

from .models import MetafileOutputFormat, MetafileRenderResult
from .parser import parse_metafile
from .render import encode_document


def render_metafile(
    data: bytes,
    *,
    output_format: MetafileOutputFormat = "png",
    dpi: int = 144,
    size_hint: tuple[int, int] | None = None,
) -> MetafileRenderResult:
    """把 WMF/EMF 字节渲染为 PNG、JPEG 或安全 SVG。"""
    if output_format not in {"png", "jpeg", "svg"}:
        raise ValueError(f"unsupported metafile output format: {output_format}")
    document = parse_metafile(data, dpi=dpi, size_hint=size_hint)
    output, media_type = encode_document(document, output_format)
    return MetafileRenderResult(
        data=output,
        output_format=output_format,
        media_type=media_type,
        width=document.width,
        height=document.height,
        source_format=document.source_format,
        emfplus_mode=document.emfplus_mode,
        partial=document.partial,
        diagnostics=document.diagnostics,
    )


__all__ = ["render_metafile"]
