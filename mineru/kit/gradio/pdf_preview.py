"""原生 Gradio 文件组件与独立 PDF.js 预览页面之间的资源适配。"""

from __future__ import annotations

import json
import mimetypes
from pathlib import Path
from typing import Literal

PreviewAction = Literal["reset", "clear", "begin", "source", "result"]


def register_pdf_preview_resources() -> Path:
    """在构建应用时公开可信预览资源，并统一跨平台脚本与 WASM 的 MIME 类型。"""
    import gradio as gr

    resource_root = Path(__file__).resolve().parents[2] / "resources" / "pdf_preview"
    mimetypes.add_type("text/javascript", ".mjs")
    mimetypes.add_type("application/wasm", ".wasm")
    gr.set_static_paths(paths=[resource_root])
    return resource_root / "viewer.html"


def pdf_preview_js(action: PreviewAction) -> str:
    """为原生组件成功事件生成前端适配函数，不增加浏览操作的 Python 请求。"""
    script = (Path(__file__).resolve().parents[2] / "resources" / "gradio_pdf_preview.js").read_text(encoding="utf-8")
    return f"(...args) => ({script})({json.dumps(action)}, ...args)"


__all__ = ["pdf_preview_js", "register_pdf_preview_resources"]
