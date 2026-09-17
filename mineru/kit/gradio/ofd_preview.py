"""OFD 源文档预览转换，不参与语义解析或结果导出。"""

from __future__ import annotations

import html

__all__ = ["build_ofd_preview"]


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
        '<iframe class="mineru-source-frame" title="OFD preview" sandbox="" '
        f'srcdoc="{html.escape(document, quote=True)}"></iframe>'
    )
