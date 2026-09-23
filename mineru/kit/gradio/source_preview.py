"""源文档 HTML 预览的异步装配，不参与语义解析或结果导出。"""

from __future__ import annotations

import asyncio
import codecs
import html
import json
import re
from collections.abc import Callable
from pathlib import Path
from urllib.parse import urlsplit

from bs4 import BeautifulSoup
from loguru import logger

from .i18n import preview_placeholder
from .ofd_preview import build_ofd_preview

__all__ = ["build_html_preview", "build_mhtml_preview", "prepare_source_preview"]

# 源 HTML 预览以视觉兼容性优先：允许原页面脚本和联网资源，但继续依赖 sandbox 的
# opaque origin 隔离 MinerU 父页面，同时显式禁用表单、子框架、对象和 Worker。
_REFERRER_META = '<meta name="referrer" content="no-referrer">'

_CSP_META = (
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "script-src 'unsafe-inline' 'unsafe-eval' https: http:; "
    "style-src 'unsafe-inline' https: http:; img-src data: blob: https: http:; "
    "font-src data: blob: https: http:; connect-src https: http:; media-src data: blob: https: http:; "
    "object-src 'none'; frame-src 'none'; worker-src 'none'; form-action 'none'\">"
)

_NAVIGATION_HANDLER_PATTERN = re.compile(
    r"""(?ix)
    (?:
        (?:window|document|top|parent|self|globalThis)\s*\.\s*location\b
        |(?<![\w$.])location\s*(?:\.|\[|=)
        |window\s*\.\s*open\s*\(
    )
    """
)

# sandbox 的 opaque origin 让父页面无法读取 iframe 的实际排版宽度，因此由源文档主动
# 上报首轮稳定布局宽度。不能直接使用整页 scrollWidth：Safari/WebKit 会把横向轮播的
# 离屏 slide 计入文档宽度，导致虚拟视口被错误放大到数千像素。
_SOURCE_PREVIEW_BRIDGE = """<script id="mineru-source-preview-bridge">
(function () {
    function explicitViewportWidth() {
        var injected = document.querySelector('meta[name="mineru-source-preview-width"]');
        var injectedWidth = injected ? Number(injected.getAttribute("content")) : 0;
        if (injectedWidth >= 320 && injectedWidth <= 2400) return injectedWidth;

        var viewport = document.querySelector('meta[name="viewport"]');
        var content = viewport ? String(viewport.getAttribute("content") || "") : "";
        var match = content.match(/(?:^|[,;\\s])width\\s*=\\s*(\\d{3,4})(?:$|[,;\\s])/i);
        if (match) {
            var numeric = Number(match[1]);
            if (numeric >= 320 && numeric <= 2400) return numeric;
        }
        var device = document.querySelector('meta[name="applicable-device"]');
        var deviceContent = device ? String(device.getAttribute("content") || "").toLowerCase() : "";
        if (/(^|[,;\\s])pc($|[,;\\s])/.test(deviceContent)) return 1200;
        return 0;
    }

    function structuralWidth() {
        var root = document.documentElement;
        var body = document.body;
        var viewportWidth = root ? root.clientWidth : innerWidth;
        var width = viewportWidth || innerWidth || 0;
        if (!body) return width;
        for (var i = 0; i < body.children.length; i++) {
            var element = body.children[i];
            var style = getComputedStyle(element);
            if (style.display === "none" || style.visibility === "hidden") continue;
            var rect = element.getBoundingClientRect();
            if (!Number.isFinite(rect.width) || rect.width <= 0) continue;
            // 只看 body 顶层布局盒，忽略 swiper/横向滚动容器内部的离屏子项。
            width = Math.max(width, rect.right - Math.min(0, rect.left));
        }
        return Math.min(Math.max(width, viewportWidth), 2400);
    }

    var preferredWidth = 0;
    function measurePreferredWidth() {
        var explicit = explicitViewportWidth();
        if (explicit) {
            preferredWidth = explicit;
            return preferredWidth;
        }
        var measured = structuralWidth();
        if (!preferredWidth || measured > preferredWidth) preferredWidth = measured;
        return preferredWidth;
    }

    function publishSize() {
        var root = document.documentElement;
        var body = document.body;
        var height = Math.max(
            root ? root.scrollHeight : 0,
            body ? body.scrollHeight : 0,
            root ? root.clientHeight : 0
        );
        parent.postMessage({
            type: "mineru-source-preview-size",
            width: measurePreferredWidth(),
            height: height
        }, "*");
    }

    var initialWidth = explicitViewportWidth();
    if (initialWidth) {
        preferredWidth = initialWidth;
        parent.postMessage({
            type: "mineru-source-preview-size",
            width: initialWidth,
            height: 0
        }, "*");
    }

    addEventListener("load", function () {
        // 无显式宽度时等外部 CSS 与页面初始化脚本完成首轮布局，避免 Safari 过早锁定单列宽度。
        requestAnimationFrame(function () {
            requestAnimationFrame(publishSize);
        });
        setTimeout(publishSize, 250);
        setTimeout(publishSize, 1000);
    }, {once: true});
})();
</script>"""

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


def _remove_auto_navigation_handlers(soup: BeautifulSoup) -> None:
    """移除内联事件中明确会发起页面导航的处理器，普通交互事件保持原样。"""
    for tag in soup.find_all(True):
        for attribute, value in list(tag.attrs.items()):
            if not attribute.lower().startswith("on"):
                continue
            source = " ".join(value) if isinstance(value, list) else str(value)
            if _NAVIGATION_HANDLER_PATTERN.search(source):
                del tag.attrs[attribute]


def _source_viewport_width_hint(soup: BeautifulSoup) -> int:
    """从源文档声明推断稳定的桌面预览宽度，供 Safari 在资源加载前立即完成缩放。"""
    viewport = soup.find("meta", attrs={"name": re.compile(r"^viewport$", re.IGNORECASE)})
    if viewport is not None:
        content = str(viewport.get("content", ""))
        match = re.search(r"(?:^|[,;\s])width\s*=\s*(\d{3,4})(?:$|[,;\s])", content, re.IGNORECASE)
        if match:
            width = int(match.group(1))
            if 320 <= width <= 2400:
                return width

    device = soup.find("meta", attrs={"name": re.compile(r"^applicable-device$", re.IGNORECASE)})
    if device is not None:
        content = str(device.get("content", "")).lower()
        if re.search(r"(^|[,;\s])pc($|[,;\s])", content):
            return 1200
    return 0


def _source_base_url(soup: BeautifulSoup) -> str | None:
    """从已有 base、canonical 或 og:url 中恢复网页资源的原始解析基址。"""
    existing = soup.find("base", href=True)
    if existing is not None:
        try:
            if urlsplit(str(existing["href"])).scheme.lower() in {"http", "https"}:
                return None
        except ValueError:
            pass
        existing.decompose()

    canonical = None
    for link in soup.find_all("link", href=True):
        rel = link.get("rel", [])
        values = rel if isinstance(rel, list) else [rel]
        if "canonical" in {str(item).lower() for item in values}:
            canonical = link
            break
    candidates = [canonical.get("href") if canonical is not None else None]
    og_url = soup.find("meta", attrs={"property": "og:url"})
    candidates.append(og_url.get("content") if og_url is not None else None)
    for candidate in candidates:
        if not candidate:
            continue
        try:
            if urlsplit(str(candidate)).scheme.lower() in {"http", "https"}:
                return str(candidate)
        except ValueError:
            continue
    return None


def _prepare_source_html_preview(
    document: str, *, source_base_url: str | None = None, csp_meta: str = _CSP_META
) -> tuple[str, int]:
    """清理自动导航、恢复原网页资源基址，并返回预览文档与稳定视口宽度提示。"""
    soup = BeautifulSoup(document, "html.parser")
    for meta in soup.find_all("meta"):
        directive = str(meta.get("http-equiv", "")).strip().lower()
        name = str(meta.get("name", "")).strip().lower()
        if directive in {"refresh", "content-security-policy"} or name == "referrer":
            meta.decompose()

    _remove_auto_navigation_handlers(soup)
    # 文档策略无法覆盖标签显式声明的策略，统一已有覆盖值以免重新发送本地 Referer。
    for tag in soup.find_all(attrs={"referrerpolicy": True}):
        tag["referrerpolicy"] = "no-referrer"

    for link in soup.find_all("a", href=True):
        try:
            scheme = urlsplit(str(link["href"])).scheme.lower()
        except ValueError:
            continue
        if scheme not in {"http", "https"}:
            continue
        link["target"] = "_blank"
        link["rel"] = list(dict.fromkeys([*link.get("rel", []), "noopener", "noreferrer"]))

    width_hint = _source_viewport_width_hint(soup)
    hint = f'<meta name="mineru-source-preview-width" content="{width_hint}">' if width_hint else ""
    base_url = _source_base_url(soup)
    if base_url is None and source_base_url:
        try:
            if urlsplit(source_base_url).scheme.lower() in {"http", "https"}:
                base_url = source_base_url
        except ValueError:
            pass
    base = f'<base href="{html.escape(base_url, quote=True)}">' if base_url is not None else ""
    prepared = _inject_head(str(soup), f"{_REFERRER_META}{hint}{base}{csp_meta}{_SOURCE_PREVIEW_BRIDGE}")
    return prepared, width_hint


def build_html_preview(payload: bytes) -> str:
    """按声明编码解码源 HTML，并在允许脚本但保持 opaque origin 的隔离框架中显示页面。"""
    document, width_hint = _prepare_source_html_preview(_decode_html(payload))
    width_attribute = f' data-mineru-source-content-width="{width_hint}"' if width_hint else ""
    frame = (
        f'<iframe class="mineru-source-frame"{width_attribute} title="HTML preview" '
        'sandbox="allow-scripts" referrerpolicy="no-referrer" '
        f'srcdoc="{html.escape(document, quote=True)}"></iframe>'
    )
    # 外层舞台负责缩放，iframe 保留桌面布局视口，避免窄预览面板改变源页面排版。
    return f'<div class="mineru-source-viewport"><div class="mineru-source-stage">{frame}</div></div>'


def build_mhtml_preview(payload: bytes) -> str:
    """惰性调用网页归档预览，避免普通 HTML 预览引入 CSS 解析依赖。"""
    from .mhtml_preview import build_mhtml_preview as build

    return build(payload)


# 后缀到预览构造器和失败占位键的显式映射，避免运行时注册。
_PREVIEW_KINDS: dict[str, tuple[Callable[[bytes], str], str]] = {
    ".ofd": (build_ofd_preview, "ofd_preview_failed"),
    ".html": (build_html_preview, "html_preview_failed"),
    ".htm": (build_html_preview, "html_preview_failed"),
    ".shtml": (build_html_preview, "html_preview_failed"),
    ".mhtml": (build_mhtml_preview, "mhtml_preview_failed"),
    ".mht": (build_mhtml_preview, "mhtml_preview_failed"),
}

_EPUB_SUFFIX = ".epub"


async def prepare_source_preview(file_path: str | None, ticket: str) -> str:
    """后台生成带请求标识的预览回执，交由浏览器丢弃过期结果。"""
    request = json.loads(ticket)
    result = {"id": request["id"], "html": ""}
    suffix = Path(file_path).suffix.lower() if file_path else ""
    if suffix == _EPUB_SUFFIX and request.get("path") == file_path:
        # EPUB 由浏览器端 viewer 直接读取 Gradio 文件 URL；此处不重复实现 EPUB 解析。
        result["kind"] = "epub"
        return json.dumps(result)
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
