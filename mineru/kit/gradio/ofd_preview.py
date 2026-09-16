"""OFD 源文档预览适配，不参与语义解析或结果导出。"""

from __future__ import annotations

import asyncio
import html
import json
from pathlib import Path

from loguru import logger

from .i18n import preview_placeholder

__all__ = ["build_ofd_preview", "prepare_ofd_preview"]


def build_ofd_preview(payload: bytes) -> str:
    """惰性转换 OFD，并在无脚本权限的隔离框架中显示自包含页面。"""
    from ._vendor.ofd2html import ofd_to_html

    document = ofd_to_html(payload)
    style = (
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
        "style-src 'unsafe-inline'; img-src data:; font-src data:\">"
        "<style>body>div>div:has(>svg){max-width:100%;}"
        "svg{display:block;max-width:100%;height:auto}</style>"
    )
    document = document.replace("<head>", "<head>" + style, 1)
    return (
        '<iframe class="mineru-ofd-frame" title="OFD preview" sandbox="" '
        f'srcdoc="{html.escape(document, quote=True)}"></iframe>'
    )


async def prepare_ofd_preview(file_path: str | None, ticket: str) -> str:
    """后台生成带请求标识的预览回执，交由浏览器丢弃过期结果。"""
    request = json.loads(ticket)
    result = {"id": request["id"], "html": ""}
    if not file_path or request.get("path") != file_path or Path(file_path).suffix.lower() != ".ofd":
        return json.dumps(result)
    try:
        payload = await asyncio.to_thread(Path(file_path).read_bytes)
        result["html"] = await asyncio.to_thread(build_ofd_preview, payload)
    except Exception as exc:
        logger.warning("OFD source preview failed: {}", exc)
        result["html"] = preview_placeholder("ofd_preview_failed")
    return json.dumps(result)
