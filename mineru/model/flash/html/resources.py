# Copyright (c) Opendatalab. All rights reserved.
"""Standalone HTML 链接、图片与本地 stylesheet 的安全解析。"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Protocol
from urllib.parse import SplitResult, unquote, urljoin, urlsplit, urlunsplit

from lxml import etree  # type: ignore[reportMissingImports]

from ....utils.image_payload import (
    parse_image_data_uri_strict,
    validate_remote_image_url,
)
from .._shared.hyperlink import sanitize_hyperlink_target
from .._shared.markup import ResolvedMarkupImage
from .constants import (
    MAX_HTML_IMAGE_BYTES,
    MAX_HTML_IMAGE_TOTAL_BYTES,
    MAX_HTML_STYLESHEET_BYTES,
    MAX_HTML_STYLESHEET_TOTAL_BYTES,
)
from .contracts import HtmlSourceContext
from .errors import HtmlResourceLimitError


_IMAGE_MIME_SIGNATURES: tuple[tuple[str, tuple[bytes, ...]], ...] = (
    ("image/jpeg", (b"\xff\xd8\xff",)),
    ("image/png", (b"\x89PNG\r\n\x1a\n",)),
    ("image/gif", (b"GIF87a", b"GIF89a")),
    ("image/webp", (b"RIFF",)),
    ("image/bmp", (b"BM",)),
    ("image/tiff", (b"II*\x00", b"MM\x00*")),
)
_SAME_DOCUMENT_SCHEMES = frozenset({"file", "http", "https"})


class HtmlAnchorResolver(Protocol):
    """定义资源解析与共享 projector 所需的文档内 anchor 查询。"""

    def resolve_fragment(self, fragment: str) -> str | None:
        """把源 fragment 转为统一内部链接。"""

    def heading_anchor(self, heading: etree._Element) -> str | None:
        """返回标题节点的统一 anchor。"""

    def heading_label(self, anchor: str) -> str | None:
        """返回统一 anchor 对应的标题文本。"""

    def note_anchor(self, note: etree._Element) -> str | None:
        """返回脚注节点的统一 anchor。"""


class HtmlResourceContext:
    """实现共享 projector 所需的 HTML 来源、资源与 anchor 适配。"""

    def __init__(
        self,
        source_context: HtmlSourceContext,
        *,
        base_href: str | None = None,
    ) -> None:
        """绑定来源上下文并计算不会逃逸本地根目录的 base。"""
        self.source_context = source_context
        self.base_href = base_href
        self.anchors: HtmlAnchorResolver | None = None
        self._image_bytes = 0
        self._stylesheet_bytes = 0
        self._image_cache: dict[Path, str] = {}
        self._data_image_cache: dict[str, ResolvedMarkupImage] = {}
        self._stylesheet_cache: dict[Path, str] = {}
        self._local_root = source_context.local_resource_root.resolve() if source_context.local_resource_root else None
        self._local_base = self._resolve_local_base()
        self._remote_base = self._resolve_remote_base()

    def bind_anchors(self, anchors: HtmlAnchorResolver) -> None:
        """在正文选择完成后绑定仅包含实际输出目标的 anchor 表。"""
        self.anchors = anchors

    def same_document_fragment(self, href: str) -> str | None:
        """解析纯 fragment 或与来源文档身份相同的相对、绝对 URL fragment。"""
        normalized = sanitize_hyperlink_target(
            href,
            allowed_schemes=_SAME_DOCUMENT_SCHEMES,
            allow_relative=True,
            allow_fragment=True,
            allow_root_relative=True,
        )
        if normalized is None:
            return None
        base_href = (self.base_href or "").strip()
        if normalized.startswith("#") and (not base_href or base_href.startswith("#")):
            fragment = unquote(normalized[1:]).strip()
            return fragment or None
        source_uri = (self.source_context.source_uri or "").strip()
        if not source_uri:
            return None
        try:
            source_parts = urlsplit(source_uri)
            base_uri = urljoin(source_uri, base_href)
            target_parts = urlsplit(urljoin(base_uri, normalized))
        except ValueError:
            return None
        fragment = unquote(target_parts.fragment).strip()
        if not fragment or _document_url_identity(target_parts) != _document_url_identity(source_parts):
            return None
        return fragment

    def resolve_link(self, href: str) -> str | None:
        """解析安全外链、相对链接或实际存在的文档内 fragment。"""
        candidate = (href or "").strip()
        if self._remote_base and candidate.startswith("//"):
            try:
                candidate = urljoin(self._remote_base, candidate)
            except ValueError:
                return None
        if self.anchors is not None and (fragment := self.same_document_fragment(candidate)) is not None:
            if internal := self.anchors.resolve_fragment(fragment):
                return internal
        normalized = sanitize_hyperlink_target(
            candidate,
            allow_relative=True,
            allow_fragment=True,
            allow_root_relative=True,
        )
        if normalized is None:
            return None
        if normalized.startswith("#"):
            base_href = (self.base_href or "").strip()
            if base_href:
                source_uri = (self.source_context.source_uri or "").strip()
                try:
                    target = urljoin(urljoin(source_uri, base_href), normalized)
                except ValueError:
                    return None
                return sanitize_hyperlink_target(
                    target,
                    allow_relative=True,
                    allow_fragment=True,
                    allow_root_relative=True,
                )
            return self.anchors.resolve_fragment(normalized) if self.anchors else None
        parsed = urlsplit(normalized)
        if parsed.scheme:
            return normalized
        if self._remote_base:
            resolved = urljoin(self._remote_base, normalized)
            return sanitize_hyperlink_target(resolved)
        if self._local_root is not None and self._local_base is not None and parsed.path and not parsed.path.startswith("/"):
            local_path = self._resolve_local_path(normalized)
            if local_path is None:
                return None
            relative_path = local_path.relative_to(self._local_root).as_posix()
            return sanitize_hyperlink_target(
                urlunsplit(("", "", relative_path, parsed.query, parsed.fragment)),
                allow_relative=True,
                allow_fragment=True,
            )
        return sanitize_hyperlink_target(normalized, allow_relative=True, allow_fragment=True)

    def resolve_image(self, source: str, *, alt: str = "") -> ResolvedMarkupImage | None:
        """按 data URI、本地安全文件、远程 URL 的顺序解析图片。"""
        normalized = (source or "").strip()
        if not normalized:
            return ResolvedMarkupImage(alt=alt) if alt else None
        if normalized.casefold().startswith("data:"):
            return self._resolve_data_image(normalized, alt=alt)
        if remote_url := self._resolve_remote_image_url(normalized):
            return ResolvedMarkupImage(image_url=remote_url, alt=alt)
        local_path = self._resolve_local_path(normalized)
        if local_path is None or not local_path.is_file():
            return ResolvedMarkupImage(alt=alt) if alt else None
        cached = self._image_cache.get(local_path)
        if cached is not None:
            return ResolvedMarkupImage(image_base64=cached, alt=alt)
        with local_path.open("rb") as source_file:
            payload = source_file.read(MAX_HTML_IMAGE_BYTES + 1)
        if len(payload) > MAX_HTML_IMAGE_BYTES:
            raise HtmlResourceLimitError(f"HTML image exceeds max_html_image_bytes={MAX_HTML_IMAGE_BYTES}")
        data_uri = _image_data_uri(payload)
        if data_uri is None:
            return ResolvedMarkupImage(alt=alt) if alt else None
        self._charge_image_bytes(len(payload))
        self._image_cache[local_path] = data_uri
        return ResolvedMarkupImage(image_base64=data_uri, alt=alt)

    def load_stylesheet(self, href: str) -> str | None:
        """只读取安全本地根目录内的 stylesheet，远程 CSS 始终忽略。"""
        path = self._resolve_local_path(href)
        if path is None or not path.is_file():
            return None
        cached = self._stylesheet_cache.get(path)
        if cached is not None:
            return cached
        with path.open("rb") as source_file:
            payload = source_file.read(MAX_HTML_STYLESHEET_BYTES + 1)
        self._charge_stylesheet_bytes(len(payload))
        stylesheet = payload.decode("utf-8-sig", errors="replace")
        self._stylesheet_cache[path] = stylesheet
        return stylesheet

    def charge_inline_stylesheet(self, stylesheet: str) -> None:
        """把一段内联 CSS 的 UTF-8 字节数计入统一 stylesheet 预算。"""
        self._charge_stylesheet_bytes(len(stylesheet.encode("utf-8")))

    def _charge_stylesheet_bytes(self, byte_count: int) -> None:
        """执行单份和整文档 stylesheet 字节限制，并只在校验通过后累计。"""
        if byte_count > MAX_HTML_STYLESHEET_BYTES:
            raise HtmlResourceLimitError(f"HTML stylesheet exceeds max_html_stylesheet_bytes={MAX_HTML_STYLESHEET_BYTES}")
        total_bytes = self._stylesheet_bytes + byte_count
        if total_bytes > MAX_HTML_STYLESHEET_TOTAL_BYTES:
            raise HtmlResourceLimitError(
                f"HTML stylesheets exceed max_html_stylesheet_total_bytes={MAX_HTML_STYLESHEET_TOTAL_BYTES}"
            )
        self._stylesheet_bytes = total_bytes

    def heading_anchor(self, heading: etree._Element) -> str | None:
        """把标题 anchor 查询委托给已绑定的注册表。"""
        return self.anchors.heading_anchor(heading) if self.anchors else None

    def heading_label(self, anchor: str) -> str | None:
        """把标题标签查询委托给已绑定的注册表。"""
        return self.anchors.heading_label(anchor) if self.anchors else None

    def note_anchor(self, note: etree._Element) -> str | None:
        """把脚注 anchor 查询委托给已绑定的注册表。"""
        return self.anchors.note_anchor(note) if self.anchors else None

    def _resolve_data_image(self, data_uri: str, *, alt: str) -> ResolvedMarkupImage | None:
        """严格解析 data URI，并执行单图与累计图片预算。"""
        if cached := self._data_image_cache.get(data_uri):
            return ResolvedMarkupImage(image_base64=cached.image_base64, alt=alt)
        if len(data_uri) > MAX_HTML_IMAGE_BYTES * 2:
            raise HtmlResourceLimitError(f"HTML image exceeds max_html_image_bytes={MAX_HTML_IMAGE_BYTES}")
        try:
            payload, extension = parse_image_data_uri_strict(data_uri)
        except ValueError:
            return ResolvedMarkupImage(alt=alt) if alt else None
        if extension == "svg":
            return ResolvedMarkupImage(alt=alt) if alt else None
        if len(payload) > MAX_HTML_IMAGE_BYTES:
            raise HtmlResourceLimitError(f"HTML image exceeds max_html_image_bytes={MAX_HTML_IMAGE_BYTES}")
        self._charge_image_bytes(len(payload))
        resolved = ResolvedMarkupImage(image_base64=data_uri, alt=alt)
        self._data_image_cache[data_uri] = resolved
        return resolved

    def _charge_image_bytes(self, byte_count: int) -> None:
        """累计实际保留的图片字节，并在超限时终止整份文档。"""
        self._image_bytes += byte_count
        if self._image_bytes > MAX_HTML_IMAGE_TOTAL_BYTES:
            raise HtmlResourceLimitError(f"HTML images exceed max_html_image_total_bytes={MAX_HTML_IMAGE_TOTAL_BYTES}")

    def _resolve_remote_image_url(self, source: str) -> str | None:
        """把远程或远程来源相对图片解析为受限 HTTP(S) 绝对 URL。"""
        try:
            parsed = urlsplit(source)
        except ValueError:
            return None
        if parsed.scheme:
            try:
                return validate_remote_image_url(source)
            except ValueError:
                return None
        if not self._remote_base:
            return None
        try:
            return validate_remote_image_url(urljoin(self._remote_base, source))
        except ValueError:
            return None

    def _resolve_remote_base(self) -> str | None:
        """返回来源 URI 与 base href 合成后的 HTTP(S) 资源基址。"""
        source_uri = (self.source_context.source_uri or "").strip()
        base_href = (self.base_href or "").strip()
        try:
            base_scheme = urlsplit(base_href).scheme.casefold() if base_href else ""
            source_scheme = urlsplit(source_uri).scheme.casefold() if source_uri else ""
        except ValueError:
            return None
        if base_href and base_scheme in {"http", "https"}:
            base = base_href
        elif source_uri and source_scheme in {"http", "https"}:
            base = urljoin(source_uri, base_href) if base_href else source_uri
        else:
            return None
        try:
            parsed = urlsplit(base)
        except ValueError:
            return None
        return base if parsed.scheme.casefold() in {"http", "https"} and parsed.hostname else None

    def _resolve_local_base(self) -> Path | None:
        """按本地安全根与相对 base href 计算资源起始目录。"""
        root = self._local_root
        if root is None:
            return None
        raw_base = (self.base_href or "").strip()
        if not raw_base:
            return root
        try:
            parsed = urlsplit(raw_base)
        except ValueError:
            return root
        if parsed.scheme or parsed.netloc or parsed.path.startswith(("/", "\\")) or "\\" in parsed.path:
            return root
        decoded = unquote(parsed.path)
        if not decoded:
            return root
        candidate = (root / decoded).resolve()
        base = candidate if raw_base.endswith("/") else candidate.parent
        return base if _is_within(base, root) else root

    def _resolve_local_path(self, source: str) -> Path | None:
        """解析本地相对资源，并在 resolve 后再次校验根目录边界。"""
        root = self._local_root
        base = self._local_base
        if root is None or base is None:
            return None
        try:
            parsed = urlsplit(source)
        except ValueError:
            return None
        if parsed.scheme or parsed.netloc or parsed.path.startswith(("/", "\\")) or "\\" in parsed.path:
            return None
        decoded = unquote(parsed.path)
        if not decoded or "\x00" in decoded or ".." in Path(decoded).parts:
            return None
        candidate = (base / decoded).resolve()
        return candidate if _is_within(candidate, root) else None


def _image_data_uri(payload: bytes) -> str | None:
    """按文件签名构造并复核受支持栅格图片的 data URI。"""
    mime = next(
        (
            media_type
            for media_type, signatures in _IMAGE_MIME_SIGNATURES
            if any(payload.startswith(signature) for signature in signatures)
        ),
        None,
    )
    if mime == "image/webp" and (len(payload) < 12 or payload[8:12] != b"WEBP"):
        mime = None
    if mime is None:
        return None
    data_uri = f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"
    try:
        parse_image_data_uri_strict(data_uri)
    except ValueError:
        return None
    return data_uri


def _document_url_identity(parts: SplitResult) -> tuple[str, str, str, str]:
    """返回忽略 fragment 的确定文档身份，并规范层级 URL 的空路径。"""
    scheme = parts.scheme.casefold()
    path = parts.path or ("/" if scheme in {"http", "https"} else "")
    return scheme, parts.netloc.casefold(), unquote(path), parts.query


def _is_within(path: Path, root: Path) -> bool:
    """判断已解析路径是否等于安全根或位于其内部。"""
    return path == root or root in path.parents


__all__ = ["HtmlAnchorResolver", "HtmlResourceContext"]
