"""将网页归档的主页面与已保存资源装配为隔离的原文预览。"""

from __future__ import annotations

import base64
import codecs
import html
import json
from urllib.parse import urljoin, urlsplit

import tinycss2
from bs4 import BeautifulSoup
from docvortex.document.mhtml import ArchivePart, MhtmlArchive, MhtmlParseError

from .source_preview import _CSP_META, _decode_html, _prepare_source_html_preview

MAX_PREVIEW_BYTES = 64 * 1024 * 1024
MAX_CSS_IMPORT_DEPTH = 32
_MHTML_CSP_META = _CSP_META.replace("style-src 'unsafe-inline' https:", "style-src 'unsafe-inline' data: https:")


def _decode_root(archive: MhtmlArchive) -> str:
    """优先使用主 MIME 部件的字符集，再沿用 HTML 声明及 BOM 解码。"""
    encoding = archive.source_context.transport_encoding
    if encoding:
        try:
            codecs.lookup(encoding)
            return archive.html.decode(encoding, errors="replace")
        except LookupError:
            pass
    return _decode_html(archive.html)


class _PreviewResources:
    """共享 MIME 索引和预览预算，按资源原地址处理 CSS 嵌套引用。"""

    def __init__(self, archive: MhtmlArchive) -> None:
        """绑定单份归档，并为重复资源建立内嵌地址缓存。"""
        self.archive = archive
        self.added_bytes = 0
        self.embedded: dict[ArchivePart, str] = {}
        self.stylesheets: dict[ArchivePart, str] = {}

    def _charge(self, value: str) -> str:
        """限制重复资源展开后的预览尺寸，避免 iframe 文本异常膨胀。"""
        self.added_bytes += len(value.encode("utf-8"))
        if self.added_bytes > MAX_PREVIEW_BYTES:
            raise ValueError("MHTML preview exceeds 64 MiB")
        return value

    def _fallback(self, reference: str, base_uri: str) -> str:
        """仅保留网络地址或原有 data URL，其他未归档地址不可进入预览。"""
        try:
            if urlsplit(reference).scheme == "data":
                return reference
            resolved = urljoin(base_uri, reference)
            return resolved if urlsplit(resolved).scheme.lower() in {"http", "https"} else ""
        except ValueError:
            return ""

    def _part(self, reference: str, base_uri: str) -> ArchivePart | None:
        """使用 DocVortex 公开索引解析 CID、绝对或相对资源地址。"""
        return self.archive.find(reference, base_href=base_uri)

    def _data_uri(self, part: ArchivePart) -> str:
        """按 MIME 类型缓存资源字节；每次插入仍独立计入输出预算。"""
        if part not in self.embedded:
            payload = self.archive.decode(part)
            self.embedded[part] = f"data:{part.media_type};base64,{base64.b64encode(payload).decode('ascii')}"
        return self._charge(self.embedded[part])

    def resource_url(self, reference: str, base_uri: str) -> str:
        """有归档资源时嵌入，缺失时交由浏览器请求正确的远端地址。"""
        part = self._part(reference, base_uri)
        if part is not None:
            try:
                if part.media_type.startswith(("image/", "font/")) or part.media_type in {
                    "application/font-woff",
                    "application/x-font-woff",
                    "application/vnd.ms-fontobject",
                    "application/octet-stream",
                }:
                    return self._data_uri(part)
            except MhtmlParseError:
                pass
        return self._fallback(reference, base_uri)

    def rewrite_srcset(self, value: str, base_uri: str) -> str:
        """按候选地址和描述符分词，保留 data URL 内部的逗号。"""
        candidates: list[str] = []
        index = 0
        while index < len(value):
            while index < len(value) and (value[index].isspace() or value[index] == ","):
                index += 1
            start = index
            while index < len(value) and not value[index].isspace():
                index += 1
            reference = value[start:index]
            if not reference:
                continue
            descriptors: list[str] = []
            if reference.endswith(","):
                reference = reference.rstrip(",")
            else:
                while index < len(value) and value[index] != ",":
                    if value[index].isspace():
                        index += 1
                        continue
                    start = index
                    while index < len(value) and not value[index].isspace() and value[index] != ",":
                        index += 1
                    descriptors.append(value[start:index])
                if index < len(value):
                    index += 1
            target = self.resource_url(reference, base_uri) if reference else ""
            if target:
                candidates.append(" ".join([target, *descriptors]))
        return ", ".join(candidates)

    def stylesheet(self, part: ArchivePart, *, active: frozenset[ArchivePart]) -> str:
        """按源样式表地址递归还原资源，循环导入直接终止。"""
        if part in active or len(active) >= MAX_CSS_IMPORT_DEPTH:
            return ""
        if part in self.stylesheets:
            return self._charge(self.stylesheets[part])
        if part.media_type != "text/css":
            return ""
        try:
            payload = self.archive.decode(part)
        except MhtmlParseError:
            return ""
        encoding = part.charset or "utf-8-sig"
        try:
            css = payload.decode(encoding, errors="replace")
        except LookupError:
            css = payload.decode("utf-8-sig", errors="replace")
        rewritten = self.rewrite_css(css, part.source_uri, active | {part})
        self.stylesheets[part] = rewritten
        return self._charge(rewritten)

    def _css_reference(self, token: object) -> str | None:
        """从 CSS 词法 token 中读取 URL 或引号字符串。"""
        kind = getattr(token, "type", "")
        if kind in {"url", "string"}:
            return str(token.value)
        if kind == "function" and getattr(token, "lower_name", "") == "url":
            values = [item for item in token.arguments if item.type not in {"whitespace", "comment"}]
            if len(values) == 1 and values[0].type == "string":
                return str(values[0].value)
        return None

    def _css_url(self, value: str) -> str:
        """以 CSS 可解析的引号 URL 序列化内嵌或远端资源地址。"""
        return f"url({json.dumps(value, ensure_ascii=False)})"

    def _tokens(self, tokens: list[object], base_uri: str, active: frozenset[ArchivePart]) -> str:
        """递归替换声明、函数和嵌套块中的 url()，保留其余 CSS 原序。"""
        pieces = []
        for token in tokens:
            reference = self._css_reference(token)
            if reference is not None and getattr(token, "type", "") != "string":
                pieces.append(self._css_url(self.resource_url(reference, base_uri)))
            elif getattr(token, "type", "") == "function":
                pieces.append(f"{token.name}({self._tokens(token.arguments, base_uri, active)})")
            elif getattr(token, "type", "") in {"() block", "[] block", "{} block"}:
                opening, closing = {"() block": ("(", ")"), "[] block": ("[", "]"), "{} block": ("{", "}")}[token.type]
                pieces.append(f"{opening}{self._tokens(token.content, base_uri, active)}{closing}")
            else:
                pieces.append(token.serialize())
        return "".join(pieces)

    def rewrite_css(self, css: str, base_uri: str, active: frozenset[ArchivePart] = frozenset()) -> str:
        """保留层叠和媒体条件，把归档中的 @import 改为内嵌样式表。"""
        pieces = []
        for rule in tinycss2.parse_stylesheet(css):
            kind = rule.type
            if kind not in {"qualified-rule", "at-rule"}:
                pieces.append(rule.serialize())
                continue
            prelude = rule.prelude
            if kind == "at-rule" and rule.lower_at_keyword == "import":
                index = next((i for i, token in enumerate(prelude) if token.type not in {"whitespace", "comment"}), None)
                reference = self._css_reference(prelude[index]) if index is not None else None
                if reference is not None:
                    part = self._part(reference, base_uri)
                    if part is not None and part.media_type == "text/css":
                        nested = self.stylesheet(part, active=active)
                        if not nested:
                            continue
                        encoded = base64.b64encode(nested.encode()).decode("ascii")
                        target = self._charge(f"data:text/css;charset=utf-8;base64,{encoded}")
                    else:
                        target = self._fallback(reference, base_uri)
                    if not target:
                        continue
                    prelude_text = self._tokens(prelude[:index], base_uri, active) + self._css_url(target)
                    prelude_text += self._tokens(prelude[index + 1 :], base_uri, active)
                    pieces.append(f"@{rule.at_keyword}{prelude_text};")
                    continue
            prelude_text = self._tokens(prelude, base_uri, active)
            content = rule.content
            if kind == "at-rule":
                if content is None:
                    pieces.append(f"@{rule.at_keyword}{prelude_text};")
                else:
                    body = self._tokens(content, base_uri, active)
                    pieces.append(f"@{rule.at_keyword}{prelude_text}{{{body}}}")
            else:
                pieces.append(f"{prelude_text}{{{self._tokens(content, base_uri, active)}}}")
        return "".join(pieces)


def build_mhtml_preview(payload: bytes) -> str:
    """还原主 HTML 及归档资源，并复用现有沙箱、缩放与导航保护。"""
    archive = MhtmlArchive(payload)
    if len(archive.html) > MAX_PREVIEW_BYTES:
        raise ValueError("MHTML preview exceeds 64 MiB")
    document = BeautifulSoup(_decode_root(archive), "html.parser")
    resources = _PreviewResources(archive)
    source_uri = archive.source_context.source_uri or ""
    base = document.find("base", href=True)
    base_uri = urljoin(source_uri, str(base["href"])) if base is not None else source_uri
    # 归档中的子页面常指向 CID 广告资源；预览只展示主文档，避免浏览器尝试装载子框架。
    for tag in document.find_all(["iframe", "frame", "frameset"]):
        if tag.parent is not None:
            tag.decompose()

    for link in document.find_all("link", href=True):
        rel = {str(value).lower() for value in link.get("rel", [])}
        if "stylesheet" not in rel:
            continue
        part = archive.find(str(link["href"]), base_href=base_uri)
        if part is None or part.media_type != "text/css":
            link["href"] = resources._fallback(str(link["href"]), base_uri)
            continue
        stylesheet = resources.stylesheet(part, active=frozenset())
        # 保留 link 的 title、alternate、disabled 与 media 语义，只替换归档样式的地址。
        encoded = base64.b64encode(stylesheet.encode("utf-8")).decode("ascii")
        link["href"] = resources._charge(f"data:text/css;charset=utf-8;base64,{encoded}")

    for style in document.find_all("style"):
        if style.string:
            style.string = resources.rewrite_css(str(style.string), base_uri).replace("</style", "<\\/style")
    for tag in document.find_all(style=True):
        tag["style"] = resources._tokens(tinycss2.parse_component_value_list(str(tag["style"])), base_uri, frozenset())
    for tag in document.find_all(["img", "source"]):
        if tag.get("src"):
            tag["src"] = resources.resource_url(str(tag["src"]), base_uri)
        if tag.get("srcset"):
            tag["srcset"] = resources.rewrite_srcset(str(tag["srcset"]), base_uri)

    prepared, width_hint = _prepare_source_html_preview(str(document), source_base_url=source_uri, csp_meta=_MHTML_CSP_META)
    if len(prepared.encode("utf-8")) > MAX_PREVIEW_BYTES:
        raise ValueError("MHTML preview exceeds 64 MiB")
    width_attribute = f' data-mineru-source-content-width="{width_hint}"' if width_hint else ""
    frame = (
        f'<iframe class="mineru-source-frame"{width_attribute} title="MHTML preview" '
        'sandbox="allow-scripts allow-popups" referrerpolicy="no-referrer" '
        f'srcdoc="{html.escape(prepared, quote=True)}"></iframe>'
    )
    if len(frame.encode("utf-8")) > MAX_PREVIEW_BYTES:
        raise ValueError("MHTML preview exceeds 64 MiB")
    return f'<div class="mineru-source-viewport"><div class="mineru-source-stage">{frame}</div></div>'


__all__ = ["build_mhtml_preview"]
