"""High-level entry point: convert OFD bytes into a single HTML string."""

from __future__ import annotations

from ..reader.ofd_reader import OFDReader
from ..render.page_renderer import render_page_to_svg
from .template import FOOT, HEAD, PAGE_GAP


# 将 OFD 字节转换为完整 HTML 页面。
def ofd_to_html(ofd_bytes: bytes) -> str:
    """Convert one OFD file (as bytes) into a complete HTML document.

    Pure function with no I/O side effects -- the FastAPI layer just wraps it.
    """
    if not ofd_bytes:
        raise ValueError("empty OFD payload")
    parts: list[str] = [HEAD]
    with OFDReader(ofd_bytes) as reader:
        first_page = True
        for doc in reader.documents:
            for page in doc.pages:
                if not first_page:
                    parts.append(PAGE_GAP)
                first_page = False
                parts.append(
                    '<div style="background:#fff;box-shadow:0 0 5px rgba(0,0,0,0.15);">'
                )
                parts.append(render_page_to_svg(reader, doc, page))
                parts.append("</div>")
    parts.append(FOOT)
    return "".join(parts)
