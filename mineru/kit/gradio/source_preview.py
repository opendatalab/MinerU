"""源文档 HTML 预览的异步装配，不参与语义解析或结果导出。"""

from __future__ import annotations

import asyncio
import codecs
import html
import json
import re
from collections.abc import Callable
from pathlib import Path

from loguru import logger

from .i18n import preview_placeholder
from .ofd_preview import build_ofd_preview

__all__ = ["build_html_preview", "prepare_source_preview"]

# 与结果预览一致允许脚本执行：MathJax/KaTeX 等公式排版依赖内联配置、eval 加载
# 与 XHR 拉取字体；沙箱保持 opaque origin（无 allow-same-origin），隔离父页面、
# 存储与顶级导航，媒体、内嵌框架和表单仍由 default-src 'none' 封禁。
_CSP_META = (
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "script-src 'unsafe-inline' 'unsafe-eval' https: http:; "
    "style-src 'unsafe-inline' https: http:; img-src data: blob: https: http:; "
    'font-src data: blob: https:; connect-src https: http:">'
)

# 只嗅探文档头部即可覆盖 meta 声明，避免扫描超大文件。
_CHARSET_SNIFF_SIZE = 4096
_CHARSET_PATTERN = re.compile(rb"""charset\s*=\s*["']?\s*([A-Za-z0-9_.:+-]+)""", re.IGNORECASE)


def _decode_html(payload: bytes) -> str:
    """按 BOM 与头部 charset 声明解码，未知或损坏编码一律回退 UTF-8 替换字符。"""
    # BOM_UTF32_LE 以 FF FE 00 00 开头，包含 BOM_UTF16_LE 前缀，必须先判定 UTF-32。
    if payload.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
        return payload.decode("utf-32", errors="replace")
    if payload.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return payload.decode("utf-16", errors="replace")
    if payload.startswith(codecs.BOM_UTF8):
        return payload.decode("utf-8-sig", errors="replace")
    encoding = "utf-8"
    if match := _CHARSET_PATTERN.search(payload[:_CHARSET_SNIFF_SIZE]):
        try:
            codecs.lookup(match.group(1).decode("ascii", errors="ignore"))
            encoding = match.group(1).decode("ascii", errors="ignore")
        except (LookupError, UnicodeDecodeError):
            pass
    return payload.decode(encoding, errors="replace")


def _inject_head(document: str, snippet: str) -> str:
    """把头部片段注入既有 head；缺失时补建，绝不置于 doctype 之前以免触发 quirks mode。"""
    # (?:\s[^>]*)? 限定标签名边界，避免 <head 规则误吞 <header>、<html 规则误吞其他前缀。
    if match := re.search(r"(?is)<head(?:\s[^>]*)?>", document):
        return document[: match.end()] + snippet + document[match.end() :]
    if match := re.search(r"(?is)<html(?:\s[^>]*)?>", document):
        return document[: match.end()] + f"<head>{snippet}</head>" + document[match.end() :]
    if match := re.match(r"(?is)\s*<!doctype[^>]*>", document):
        return document[: match.end()] + f"<head>{snippet}</head>" + document[match.end() :]
    return f"<head>{snippet}</head>" + document


def build_html_preview(payload: bytes) -> str:
    """按声明编码解码源 HTML，并在仅开放脚本的隔离框架中显示自包含页面。"""
    document = _inject_head(_decode_html(payload), _CSP_META)
    return (
        '<iframe class="mineru-source-frame" title="HTML preview" sandbox="allow-scripts" '
        f'srcdoc="{html.escape(document, quote=True)}"></iframe>'
    )


# 后缀到预览构造器和失败占位键的显式映射，避免运行时注册。
_PREVIEW_KINDS: dict[str, tuple[Callable[[bytes], str], str]] = {
    ".ofd": (build_ofd_preview, "ofd_preview_failed"),
    ".html": (build_html_preview, "html_preview_failed"),
    ".htm": (build_html_preview, "html_preview_failed"),
    ".shtml": (build_html_preview, "html_preview_failed"),
}


async def prepare_source_preview(file_path: str | None, ticket: str) -> str:
    """后台生成带请求标识的预览回执，交由浏览器丢弃过期结果。"""
    request = json.loads(ticket)
    result = {"id": request["id"], "html": ""}
    suffix = Path(file_path).suffix.lower() if file_path else ""
    kind = _PREVIEW_KINDS.get(suffix) if request.get("path") == file_path else None
    if kind is None:
        return json.dumps(result)
    builder, failure_key = kind
    try:
        payload = await asyncio.to_thread(Path(file_path).read_bytes)
        result["html"] = await asyncio.to_thread(builder, payload)
    except Exception as exc:
        logger.warning("Source preview failed for '{}': {}", suffix, exc)
        result["html"] = preview_placeholder(failure_key)
    return json.dumps(result)
