# Copyright (c) Opendatalab. All rights reserved.
"""EPUB renderer 的图片解析、规范化与包内去重。"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import PurePosixPath
import re
from urllib.parse import unquote, urlsplit

from PIL import Image, UnidentifiedImageError

from ...contracts import AssetResolver
from ....types import ImagePayloadBlock
from ....utils.image_payload import (
    extract_mineru_generated_svg_fallback,
    normalize_image_extension,
    parse_image_data_uri_strict,
    validate_image_sidecar_path,
)

_SVG_START_RE = re.compile(rb"^(?:\xef\xbb\xbf)?\s*(?:<\?xml\b.*?\?>\s*)?<svg\b", re.IGNORECASE | re.DOTALL)
_PASSTHROUGH_FORMATS = {"gif", "jpg", "png"}
_MEDIA_TYPES = {
    "gif": "image/gif",
    "jpg": "image/jpeg",
    "png": "image/png",
}


@dataclass(frozen=True, slots=True)
class EpubAsset:
    """保存一个已规范化且可写入 EPUB 的图片资源。"""

    file_name: str
    media_type: str
    data: bytes


class EpubAssetRegistry:
    """按内容哈希登记 EPUB 图片，并隔离所有外部素材读取。"""

    def __init__(self, asset_resolver: AssetResolver | None) -> None:
        """保存显式 resolver，并初始化来源缓存与内容去重表。"""
        self._asset_resolver = asset_resolver
        self._source_cache: dict[str, str | None] = {}
        self._assets: dict[str, EpubAsset] = {}

    @property
    def assets(self) -> tuple[EpubAsset, ...]:
        """按稳定文件名顺序返回所有已登记包内图片。"""
        return tuple(self._assets[name] for name in sorted(self._assets))

    def resolve_block(self, block: ImagePayloadBlock) -> str | None:
        """按 sidecar、data URI、远程 URL 优先级解析一个图片 block。"""
        if block.image_path:
            resolved = self._resolve_relative_path(block.image_path)
            if resolved is not None:
                return resolved
        if block.image_base64:
            resolved = self._resolve_data_uri(block.image_base64)
            if resolved is not None:
                return resolved
        # EPUB 图片必须位于容器中；远程 image_url 只作为不可下载的最后候选跳过。
        return None

    def resolve_embedded_source(self, source: str) -> str | None:
        """解析富 HTML img 的 data URI 或安全相对 sidecar，拒绝网络来源。"""
        normalized = source.strip()
        if not normalized:
            return None
        if normalized.lower().startswith("data:"):
            return self._resolve_data_uri(normalized)
        try:
            parsed = urlsplit(normalized)
        except ValueError:
            return None
        if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment or normalized.startswith(("/", "\\")):
            return None
        decoded_path = unquote(parsed.path)
        try:
            safe_path = validate_image_sidecar_path(decoded_path)
        except ValueError:
            return None
        return self._resolve_relative_path(safe_path)

    def _resolve_relative_path(self, image_path: str) -> str | None:
        """通过显式 resolver 读取相对图片，失败时返回可降级的空结果。"""
        cache_key = f"path:{image_path}"
        if cache_key in self._source_cache:
            return self._source_cache[cache_key]
        if self._asset_resolver is None:
            self._source_cache[cache_key] = None
            return None
        try:
            safe_path = validate_image_sidecar_path(image_path)
            data = self._asset_resolver(safe_path)
            if not isinstance(data, bytes):
                raise TypeError("asset_resolver must return bytes")
            declared_extension = PurePosixPath(safe_path).suffix.removeprefix(".") or None
            href = self._register_prepared(data, declared_extension=declared_extension)
        except Exception:
            href = None
        self._source_cache[cache_key] = href
        return href

    def _resolve_data_uri(self, data_uri: str) -> str | None:
        """严格解码图片 data URI，失败时返回可降级的空结果。"""
        cache_key = f"data:{hashlib.sha256(data_uri.encode('utf-8', errors='replace')).hexdigest()}"
        if cache_key in self._source_cache:
            return self._source_cache[cache_key]
        try:
            data, extension = parse_image_data_uri_strict(data_uri)
            href = self._register_prepared(data, declared_extension=extension)
        except (OSError, TypeError, ValueError):
            href = None
        self._source_cache[cache_key] = href
        return href

    def _register_prepared(self, data: bytes, *, declared_extension: str | None) -> str:
        """规范化图片格式，按 SHA-256 去重并返回正文相对 href。"""
        normalized, extension = _prepare_epub_image(data, declared_extension=declared_extension)
        digest = hashlib.sha256(normalized).hexdigest()
        file_name = f"{digest}.{extension}"
        if file_name not in self._assets:
            self._assets[file_name] = EpubAsset(
                file_name=file_name,
                media_type=_MEDIA_TYPES[extension],
                data=normalized,
            )
        return f"../assets/{file_name}"


def _prepare_epub_image(data: bytes, *, declared_extension: str | None) -> tuple[bytes, str]:
    """校验图片字节，并把非 PNG/JPEG/GIF 栅格统一转为 PNG。"""
    if not isinstance(data, bytes) or not data:
        raise ValueError("Image payload must contain bytes")
    expected = normalize_image_extension(declared_extension or "") or None
    if expected == "svg" or _SVG_START_RE.match(data[:4096]) is not None:
        if expected not in {None, "svg"}:
            raise ValueError("Image bytes do not match the declared format")
        fallback, _, _ = extract_mineru_generated_svg_fallback(data)
        return _prepare_epub_image(fallback, declared_extension="png")

    try:
        with Image.open(BytesIO(data)) as image:
            detected = normalize_image_extension(image.format or "")
            if not detected:
                raise ValueError("Unsupported image format")
            image.load()
            if expected is not None and expected != detected:
                raise ValueError("Image bytes do not match the declared format")
            if detected in _PASSTHROUGH_FORMATS:
                return data, detected
            has_alpha = "A" in image.getbands() or (image.mode == "P" and "transparency" in image.info)
            converted = image.convert("RGBA" if has_alpha else "RGB")
            output = BytesIO()
            converted.save(output, format="PNG")
            return output.getvalue(), "png"
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Invalid or unsupported image payload") from exc


__all__ = ["EpubAsset", "EpubAssetRegistry"]
