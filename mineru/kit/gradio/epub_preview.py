"""Gradio 专用 EPUB 阅读器资源适配，不参与 EPUB 语义解析。"""

from __future__ import annotations

import mimetypes
from pathlib import Path

__all__ = ["register_epub_preview_resources"]


def register_epub_preview_resources() -> Path:
    """注册本地 EPUB viewer 静态资源，并返回 viewer 页面路径。"""
    import gradio as gr

    resource_root = Path(__file__).resolve().parents[2] / "resources" / "epub_preview"
    mimetypes.add_type("text/javascript", ".js")
    gr.set_static_paths(paths=[resource_root])
    return resource_root / "viewer.html"
