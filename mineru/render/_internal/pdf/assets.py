# Copyright (c) Opendatalab. All rights reserved.
"""PDF renderer 的离线图片解析、签名校验与格式准备。"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
import re
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError

from ...contracts import AssetResolver
from ....types import ImagePayloadBlock
from ....utils.image_payload import (
    extract_mineru_generated_svg_fallback,
    parse_image_data_uri_strict,
    validate_image_sidecar_path,
)

_PIL_FORMAT_EXTENSIONS = {
    "BMP": "bmp",
    "GIF": "gif",
    "JPEG": "jpg",
    "PNG": "png",
    "TIFF": "tiff",
    "WEBP": "webp",
}
_EXTENSION_ALIASES = {"jpeg": "jpg", "jpe": "jpg", "tif": "tiff"}
_SVG_START_RE = re.compile(rb"^(?:\xef\xbb\xbf)?\s*(?:<\?xml\b.*?\?>\s*)?<svg\b", re.IGNORECASE | re.DOTALL)


class PdfAssetError(ValueError):
    """表示 PDF renderer 无法离线安全加载或识别图片素材。"""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """保存可直接交给 ReportLab 的图片字节与像素尺寸。"""

    data: bytes
    extension: str
    width_px: int
    height_px: int


def prepare_block_image(
    block: ImagePayloadBlock,
    asset_resolver: AssetResolver | None = None,
) -> PreparedImage:
    """按 sidecar、data URI 的固定优先级加载一个图片载荷 block。"""
    if not isinstance(block, ImagePayloadBlock):
        raise TypeError("block must be an ImagePayloadBlock")
    if block.image_path is not None:
        return _prepare_relative_asset(block.image_path, asset_resolver)
    if block.image_base64 is not None:
        return _prepare_data_uri(block.image_base64)
    if block.image_url is not None:
        raise PdfAssetError(f"Remote image is not downloaded: {block.image_url}")
    raise PdfAssetError("Image block does not contain an image source")


def prepare_html_image(
    source: str,
    asset_resolver: AssetResolver | None = None,
) -> PreparedImage:
    """从 HTML img 的 data URI 或安全相对 sidecar 加载图片。"""
    if not isinstance(source, str):
        raise TypeError("HTML image source must be a string")
    normalized = source.strip()
    if not normalized:
        raise PdfAssetError("HTML image source must not be empty")
    if normalized.lower().startswith("data:"):
        if not normalized.lower().startswith("data:image/"):
            raise PdfAssetError("Only image data URIs are supported")
        return _prepare_data_uri(normalized)
    parsed = urlsplit(normalized)
    if parsed.scheme or parsed.netloc:
        raise PdfAssetError(f"Remote HTML image is not downloaded: {normalized}")
    return _prepare_relative_asset(normalized, asset_resolver)


def prepare_image_bytes(data: bytes, *, declared_extension: str | None = None) -> PreparedImage:
    """严格解码图片字节，并把 WebP 与安全 SVG fallback 转成 PNG。"""
    if not isinstance(data, bytes):
        raise PdfAssetError("Image resolver must return bytes")
    if not data:
        raise PdfAssetError("Image payload must not be empty")
    expected_extension = _normalize_extension(declared_extension)
    if expected_extension == "svg" or _looks_like_svg(data):
        try:
            fallback, logical_width, logical_height = extract_mineru_generated_svg_fallback(data)
        except ValueError as exc:
            raise PdfAssetError("SVG images require a validated MinerU PNG fallback") from exc
        prepared = prepare_image_bytes(fallback, declared_extension="png")
        return PreparedImage(
            data=prepared.data,
            extension="png",
            width_px=logical_width,
            height_px=logical_height,
        )
    try:
        with Image.open(BytesIO(data)) as image:
            detected_extension = _PIL_FORMAT_EXTENSIONS.get((image.format or "").upper())
            if detected_extension is None:
                raise PdfAssetError(f"Unsupported image format: {image.format or 'unknown'}")
            width_px, height_px = image.size
            image.load()
            if expected_extension is not None and expected_extension != detected_extension:
                raise PdfAssetError(
                    f"Image bytes do not match declared format: expected {expected_extension}, detected {detected_extension}"
                )
            if detected_extension == "webp":
                return _convert_webp_to_png(image, width_px, height_px)
    except PdfAssetError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise PdfAssetError("Invalid or corrupted image payload") from exc
    return PreparedImage(
        data=data,
        extension=detected_extension,
        width_px=width_px,
        height_px=height_px,
    )


def _prepare_data_uri(data_uri: str) -> PreparedImage:
    """严格解析图片 data URI 并校验 MIME、签名与完整载荷。"""
    try:
        data, extension = parse_image_data_uri_strict(data_uri)
    except ValueError as exc:
        raise PdfAssetError(str(exc)) from exc
    return prepare_image_bytes(data, declared_extension=extension)


def _prepare_relative_asset(image_path: str, asset_resolver: AssetResolver | None) -> PreparedImage:
    """校验相对 sidecar 路径，并仅通过显式 resolver 获取字节。"""
    parsed = urlsplit(image_path)
    if parsed.scheme or parsed.netloc:
        raise PdfAssetError(f"Remote or scheme-based image source is not supported: {image_path}")
    try:
        safe_path = validate_image_sidecar_path(image_path)
    except ValueError as exc:
        raise PdfAssetError(str(exc)) from exc
    if asset_resolver is None:
        raise PdfAssetError(f"asset_resolver is required for image_path: {safe_path}")
    try:
        data = asset_resolver(safe_path)
    except Exception as exc:
        raise PdfAssetError(f"Failed to resolve image asset: {safe_path}") from exc
    extension = PurePosixPath(safe_path).suffix.removeprefix(".") or None
    return prepare_image_bytes(data, declared_extension=extension)


def _normalize_extension(extension: str | None) -> str | None:
    """规范化可识别扩展名，未知扩展不作为签名声明。"""
    if extension is None:
        return None
    normalized = extension.lower().lstrip(".").split("+", 1)[0]
    normalized = _EXTENSION_ALIASES.get(normalized, normalized)
    if normalized == "svg":
        return normalized
    return normalized if normalized in set(_PIL_FORMAT_EXTENSIONS.values()) else None


def _looks_like_svg(data: bytes) -> bool:
    """识别带可选 BOM 与 XML 声明的 SVG 字节。"""
    return _SVG_START_RE.match(data[:4096]) is not None


def _convert_webp_to_png(image: Image.Image, width_px: int, height_px: int) -> PreparedImage:
    """在内存中把已解码 WebP 转成 ReportLab 可稳定读取的 PNG。"""
    output = BytesIO()
    image.save(output, format="PNG")
    return PreparedImage(
        data=output.getvalue(),
        extension="png",
        width_px=width_px,
        height_px=height_px,
    )


__all__ = [
    "PdfAssetError",
    "PreparedImage",
    "prepare_block_image",
    "prepare_html_image",
    "prepare_image_bytes",
]
