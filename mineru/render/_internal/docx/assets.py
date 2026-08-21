# Copyright (c) Opendatalab. All rights reserved.
"""DOCX renderer 的安全图片素材解析与格式准备。"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
import re
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError

from mineru.render.contracts import AssetResolver
from mineru.types import ImagePayloadBlock
from mineru.utils.image_payload import parse_image_data_uri_strict, validate_image_sidecar_path

_PIL_FORMAT_EXTENSIONS = {
    "BMP": "bmp",
    "GIF": "gif",
    "JPEG": "jpg",
    "PNG": "png",
    "TIFF": "tiff",
    "WEBP": "webp",
}
_EXTENSION_ALIASES = {
    "jpeg": "jpg",
    "jpe": "jpg",
    "tif": "tiff",
}
_SVG_START_RE = re.compile(rb"^(?:\xef\xbb\xbf)?\s*(?:<\?xml\b.*?\?>\s*)?<svg\b", re.IGNORECASE | re.DOTALL)


class DocxAssetError(ValueError):
    """表示 DOCX renderer 无法安全加载或识别图片素材。"""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """保存可直接交给 DOCX 的图片字节、格式与像素尺寸。"""

    data: bytes
    extension: str
    width_px: int
    height_px: int


def prepare_block_image(
    block: ImagePayloadBlock,
    asset_resolver: AssetResolver | None = None,
) -> PreparedImage:
    """从图片载荷 block 加载图片，始终优先使用内嵌 data URI。"""
    if not isinstance(block, ImagePayloadBlock):
        raise TypeError("block must be an ImagePayloadBlock")
    if block.image_base64 is not None:
        return _prepare_data_uri(block.image_base64)
    if block.image_path is not None:
        return _prepare_relative_asset(block.image_path, asset_resolver)
    raise DocxAssetError("Image block does not contain image_base64 or image_path")


def prepare_html_image(
    source: str,
    asset_resolver: AssetResolver | None = None,
) -> PreparedImage:
    """从 HTML ``img`` 的 src 加载 data URI 或安全的相对 sidecar 图片。"""
    if not isinstance(source, str):
        raise TypeError("HTML image source must be a string")
    normalized_source = source.strip()
    if not normalized_source:
        raise DocxAssetError("HTML image source must not be empty")
    if normalized_source.lower().startswith("data:"):
        if not normalized_source.lower().startswith("data:image/"):
            raise DocxAssetError("Only image data URIs are supported")
        return _prepare_data_uri(normalized_source)
    return _prepare_relative_asset(normalized_source, asset_resolver)


def prepare_image_bytes(
    data: bytes,
    *,
    declared_extension: str | None = None,
) -> PreparedImage:
    """严格解码图片字节，并将 Word 不直接支持的 WebP 在内存中转为 PNG。"""
    if not isinstance(data, bytes):
        raise DocxAssetError("Image resolver must return bytes")
    if not data:
        raise DocxAssetError("Image payload must not be empty")

    expected_extension = _normalize_extension(declared_extension)
    if expected_extension == "svg" or _looks_like_svg(data):
        raise DocxAssetError("SVG images are not supported by the DOCX renderer")

    try:
        with Image.open(BytesIO(data)) as image:
            detected_extension = _PIL_FORMAT_EXTENSIONS.get((image.format or "").upper())
            if detected_extension is None:
                raise DocxAssetError(f"Unsupported image format: {image.format or 'unknown'}")
            width_px, height_px = image.size
            image.load()
            if expected_extension is not None and expected_extension != detected_extension:
                raise DocxAssetError(
                    f"Image bytes do not match declared format: expected {expected_extension}, detected {detected_extension}"
                )
            if detected_extension == "webp":
                return _convert_webp_to_png(image, width_px, height_px)
    except DocxAssetError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise DocxAssetError("Invalid or corrupted image payload") from exc

    return PreparedImage(
        data=data,
        extension=detected_extension,
        width_px=width_px,
        height_px=height_px,
    )


def _prepare_data_uri(data_uri: str) -> PreparedImage:
    """严格解析 data URI，并校验 MIME、签名和完整图片数据。"""
    try:
        data, extension = parse_image_data_uri_strict(data_uri)
    except ValueError as exc:
        raise DocxAssetError(str(exc)) from exc
    if extension == "svg":
        raise DocxAssetError("SVG images are not supported by the DOCX renderer")
    return prepare_image_bytes(data, declared_extension=extension)


def _prepare_relative_asset(
    image_path: str,
    asset_resolver: AssetResolver | None,
) -> PreparedImage:
    """校验相对 sidecar 路径，通过调用方 resolver 读取字节且不访问文件系统或网络。"""
    parsed_source = urlsplit(image_path)
    if parsed_source.scheme or parsed_source.netloc:
        raise DocxAssetError(f"Remote or scheme-based image source is not supported: {image_path}")
    try:
        safe_path = validate_image_sidecar_path(image_path)
    except ValueError as exc:
        raise DocxAssetError(str(exc)) from exc
    if asset_resolver is None:
        raise DocxAssetError(f"asset_resolver is required for image_path: {safe_path}")
    try:
        data = asset_resolver(safe_path)
    except Exception as exc:
        raise DocxAssetError(f"Failed to resolve image asset: {safe_path}") from exc

    extension = PurePosixPath(safe_path).suffix.removeprefix(".") or None
    return prepare_image_bytes(data, declared_extension=extension)


def _normalize_extension(extension: str | None) -> str | None:
    """规范化可识别的扩展名；未知扩展名不作为图片格式声明。"""
    if extension is None:
        return None
    normalized = extension.lower().lstrip(".").split("+", 1)[0]
    normalized = _EXTENSION_ALIASES.get(normalized, normalized)
    if normalized == "svg":
        return normalized
    return normalized if normalized in set(_PIL_FORMAT_EXTENSIONS.values()) else None


def _looks_like_svg(data: bytes) -> bool:
    """识别带可选 BOM/XML 声明的 SVG 字节，以便给出明确的不支持错误。"""
    return _SVG_START_RE.match(data[:4096]) is not None


def _convert_webp_to_png(image: Image.Image, width_px: int, height_px: int) -> PreparedImage:
    """在内存中把已解码 WebP 转换为 PNG，保留透明通道与像素尺寸。"""
    output = BytesIO()
    image.save(output, format="PNG")
    return PreparedImage(
        data=output.getvalue(),
        extension="png",
        width_px=width_px,
        height_px=height_px,
    )


__all__ = [
    "AssetResolver",
    "DocxAssetError",
    "PreparedImage",
    "prepare_block_image",
    "prepare_html_image",
    "prepare_image_bytes",
]
