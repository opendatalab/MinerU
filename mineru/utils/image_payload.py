# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import base64
import binascii
import hashlib
from pathlib import Path, PureWindowsPath
import re
import xml.etree.ElementTree as ElementTree
from typing import Final
from urllib.parse import urlsplit

INLINE_IMAGE_DATA_URI_RE = re.compile(r"data:image/([^;\"']+);base64,([^\"']+)", re.DOTALL)
_SVG_NAMESPACE = "http://www.w3.org/2000/svg"
_MINERU_SVG_MARKER = "wmf-emf"
_MINERU_SVG_FALLBACK_ID = "mineru-raster-fallback"
MAX_IMAGE_PAYLOAD_BYTES: Final = 64 * 1024 * 1024
MAX_GENERATED_SVG_BYTES: Final = MAX_IMAGE_PAYLOAD_BYTES
MAX_RASTER_IMAGE_BYTES: Final = MAX_IMAGE_PAYLOAD_BYTES
MAX_IMAGE_DATA_URI_BYTES: Final = len("data:image/svg+xml;base64,") + ((MAX_IMAGE_PAYLOAD_BYTES + 2) // 3) * 4
MAX_DECODED_RASTER_DIMENSION: Final = 8_192
MAX_DECODED_RASTER_PIXELS: Final = 16_000_000
_MAX_GENERATED_SVG_NODES = 100_000
_SAFE_SVG_ATTRIBUTES: dict[str, set[str]] = {
    "svg": {"width", "height", "viewBox", "data-mineru-generated"},
    "metadata": {"id", "data-mime"},
    "defs": set(),
    "clipPath": {"id"},
    "path": {
        "d",
        "fill",
        "fill-rule",
        "fill-opacity",
        "opacity",
        "stroke",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-miterlimit",
        "stroke-opacity",
        "stroke-dasharray",
    },
    "g": {"clip-path"},
    "image": {"width", "height", "transform", "href"},
    "text": {
        "x",
        "y",
        "fill",
        "opacity",
        "font-family",
        "font-size",
        "font-weight",
        "font-style",
        "text-anchor",
        "dominant-baseline",
        "text-decoration",
        "transform",
    },
    "rect": {"x", "y", "width", "height", "fill", "opacity"},
}


class _RejectingSvgTreeBuilder(ElementTree.TreeBuilder):
    """构造拒绝任何 DTD 的 SVG XML 树。"""

    def doctype(self, _name: str, _pubid: str | None, _system: str | None) -> None:
        """在实体声明被处理前拒绝任意偏移和编码的 DOCTYPE。"""
        raise ValueError("Generated SVG must not contain a DTD or entity declaration")


def normalize_image_extension(fmt: str) -> str:
    """规范化图片扩展名，保证同一图片格式生成稳定文件名。"""
    normalized = fmt.lower().split("+", 1)[0]
    return "jpg" if normalized in {"jpeg", "jpg"} else normalized


def validate_decoded_raster_size(width: int, height: int) -> None:
    """在 Pillow 解码像素前校验 raster 单边尺寸与总像素预算。"""
    if (
        width <= 0
        or height <= 0
        or width > MAX_DECODED_RASTER_DIMENSION
        or height > MAX_DECODED_RASTER_DIMENSION
        or width * height > MAX_DECODED_RASTER_PIXELS
    ):
        raise ValueError(
            f"Decoded raster image exceeds limits: {width}x{height}; "
            f"max_dimension={MAX_DECODED_RASTER_DIMENSION}, max_pixels={MAX_DECODED_RASTER_PIXELS}"
        )


def _parse_svg_root_strict(payload: bytes) -> ElementTree.Element:
    """在固定字节预算内解析 SVG 根节点，并在实体展开前拒绝 DTD。"""
    if len(payload) > MAX_GENERATED_SVG_BYTES:
        raise ValueError("SVG image payload exceeds its byte limit")
    try:
        parser = ElementTree.XMLParser(target=_RejectingSvgTreeBuilder())
        root = ElementTree.fromstring(payload, parser=parser)
    except ElementTree.ParseError as exc:
        raise ValueError("Invalid SVG image payload") from exc
    if not isinstance(root.tag, str) or root.tag.rsplit("}", 1)[-1].lower() != "svg":
        raise ValueError("Image signature does not match MIME subtype: svg+xml")
    return root


def parse_image_data_uri_strict(data_uri: str) -> tuple[bytes, str]:
    """严格解析图片 data URI，并同时校验 MIME 与文件签名是否一致。"""
    if len(data_uri) > MAX_IMAGE_DATA_URI_BYTES:
        raise ValueError("Image data URI exceeds its byte limit")
    match = re.fullmatch(r"data:image/([^;]+);base64,([A-Za-z0-9+/]*={0,2})", data_uri)
    if match is None:
        raise ValueError("Invalid image data URI")

    mime_subtype = match.group(1).lower()
    extension = normalize_image_extension(mime_subtype)
    encoded_payload = match.group(2)
    payload_limit = MAX_GENERATED_SVG_BYTES if extension == "svg" else MAX_RASTER_IMAGE_BYTES
    max_encoded_bytes = ((payload_limit + 2) // 3) * 4
    if len(encoded_payload) > max_encoded_bytes:
        payload_kind = "SVG" if extension == "svg" else "Raster"
        raise ValueError(f"{payload_kind} image payload exceeds its byte limit")
    if extension == "svg":
        if mime_subtype != "svg+xml":
            raise ValueError(f"Unsupported image MIME subtype: {mime_subtype}")
    try:
        payload = base64.b64decode(encoded_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload") from exc
    if not payload:
        raise ValueError("Image payload must not be empty")
    if len(payload) > payload_limit:
        payload_kind = "SVG" if extension == "svg" else "Raster"
        raise ValueError(f"{payload_kind} image payload exceeds its byte limit")

    if extension == "svg":
        _parse_svg_root_strict(payload)
        return payload, extension

    signatures: dict[str, tuple[bytes, ...]] = {
        "jpg": (b"\xff\xd8\xff",),
        "png": (b"\x89PNG\r\n\x1a\n",),
        "gif": (b"GIF87a", b"GIF89a"),
        "webp": (b"RIFF",),
        "bmp": (b"BM",),
        "tiff": (b"II*\x00", b"MM\x00*"),
    }
    expected = signatures.get(extension)
    if expected is None:
        raise ValueError(f"Unsupported image MIME subtype: {mime_subtype}")
    if not any(payload.startswith(signature) for signature in expected):
        raise ValueError(f"Image signature does not match MIME subtype: {mime_subtype}")
    if extension == "webp" and (len(payload) < 12 or payload[8:12] != b"WEBP"):
        raise ValueError("Image signature does not match MIME subtype: webp")
    return payload, extension


def _decode_png_data_uri(value: str) -> bytes:
    """严格解码生成 SVG 内嵌的 PNG data URI。"""
    match = re.fullmatch(r"data:image/png;base64,([A-Za-z0-9+/]*={0,2})", value)
    if match is None:
        raise ValueError("Generated SVG image href must contain a PNG data URI")
    try:
        payload = base64.b64decode(match.group(1), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Generated SVG contains invalid PNG base64") from exc
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("Generated SVG fallback does not contain a PNG")
    return payload


def _validate_generated_svg_attribute(tag: str, name: str, value: str) -> None:
    """校验生成 SVG 属性只使用静态数值、颜色、本地 clip 或 PNG。"""
    if name not in _SAFE_SVG_ATTRIBUTES[tag] or "\x00" in value or name.lower().startswith("on"):
        raise ValueError(f"Generated SVG contains an unsafe attribute: {tag}.{name}")
    normalized = value.strip().casefold()
    if "javascript:" in normalized or "data:text" in normalized or "url(" in normalized and name != "clip-path":
        raise ValueError(f"Generated SVG contains an unsafe attribute value: {tag}.{name}")
    if name == "href":
        _decode_png_data_uri(value)
    elif name == "clip-path" and re.fullmatch(r"url\(#mineru-clip-\d+\)", value) is None:
        raise ValueError("Generated SVG clip-path must reference a local MinerU clip")
    elif name == "transform" and re.fullmatch(r"(?:matrix|rotate)\([0-9eE+.,\- ]+\)", value) is None:
        raise ValueError("Generated SVG transform is outside the supported subset")
    elif name == "d" and re.fullmatch(r"[MmLlCcZz0-9eE+.,\- ]*", value) is None:
        raise ValueError("Generated SVG path data is outside the supported subset")


def extract_mineru_generated_svg_fallback(payload: bytes) -> tuple[bytes, int, int]:
    """验证 MinerU 生成 SVG，并返回 PNG fallback 与逻辑像素尺寸。"""
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("Generated SVG payload is empty or exceeds its byte limit")
    root = _parse_svg_root_strict(payload)
    if root.tag != f"{{{_SVG_NAMESPACE}}}svg" or root.get("data-mineru-generated") != _MINERU_SVG_MARKER:
        raise ValueError("SVG is not marked as a MinerU generated metafile")
    try:
        width = int(root.get("width", ""))
        height = int(root.get("height", ""))
    except ValueError as exc:
        raise ValueError("Generated SVG dimensions must be integers") from exc
    if width <= 0 or height <= 0 or width > 8192 or height > 8192 or root.get("viewBox") != f"0 0 {width} {height}":
        raise ValueError("Generated SVG dimensions or viewBox are outside the supported bounds")

    fallback: bytes | None = None
    node_count = 0
    for element in root.iter():
        node_count += 1
        if node_count > _MAX_GENERATED_SVG_NODES:
            raise ValueError("Generated SVG exceeds its node limit")
        if not isinstance(element.tag, str) or not element.tag.startswith(f"{{{_SVG_NAMESPACE}}}"):
            raise ValueError("Generated SVG contains a foreign namespace")
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in _SAFE_SVG_ATTRIBUTES:
            raise ValueError(f"Generated SVG contains an unsupported element: {tag}")
        for attribute_name, attribute_value in element.attrib.items():
            if "}" in attribute_name:
                raise ValueError("Generated SVG contains a namespaced attribute")
            _validate_generated_svg_attribute(tag, attribute_name, attribute_value)
        if element.tail and element.tail.strip():
            raise ValueError("Generated SVG contains unexpected tail text")
        if tag == "metadata":
            if element.get("id") != _MINERU_SVG_FALLBACK_ID or element.get("data-mime") != "image/png" or fallback is not None:
                raise ValueError("Generated SVG fallback metadata is invalid or duplicated")
            encoded = (element.text or "").strip()
            fallback = _decode_png_data_uri(f"data:image/png;base64,{encoded}")
        elif tag != "text" and element.text and element.text.strip():
            raise ValueError(f"Generated SVG element must not contain text: {tag}")
    if fallback is None:
        raise ValueError("Generated SVG does not contain a PNG fallback")
    return fallback, width, height


def image_path_from_key(path_key: str, image_format: str = "JPEG") -> str:
    """复用旧裁图路径哈希规则，根据逻辑路径生成稳定图片文件名。"""
    ext = normalize_image_extension(image_format)
    return f"{path_key}.{ext}"


class ImagePayloadCache:
    """保存运行时图片载荷，public middle_json 只通过 image_path 引用图片。"""

    def __init__(self, images: dict[str, bytes] | None = None) -> None:
        self._images: dict[str, bytes] = dict(images or {})

    def register_bytes(
        self,
        img_bytes: bytes,
        image_format: str = "JPEG",
        *,
        path_key: str | None = None,
        image_path: str | None = None,
    ) -> str:
        """登记图片字节，优先使用显式路径，其次使用逻辑 path_key 生成稳定路径。"""
        if image_path:
            img_path = image_path
        elif path_key:
            img_path = image_path_from_key(path_key, image_format)
        else:
            ext = normalize_image_extension(image_format)
            payload_key = base64.b64encode(img_bytes).decode("ascii")
            img_path = f"{hashlib.sha256(payload_key.encode('utf-8')).hexdigest()}.{ext}"
        self._images[img_path] = img_bytes
        return img_path

    def update(self, images: dict[str, bytes]) -> None:
        """合并外部 sidecar 图片字节，供 API client 绑定远端输出。"""
        for image_path, image_bytes in images.items():
            self.register_bytes(image_bytes, image_path=image_path)

    def images(self) -> dict[str, bytes]:
        """返回 image_path 到图片字节的副本，避免调用方改写内部缓存。"""
        return dict(self._images)


def validate_image_sidecar_path(image_path: str) -> str:
    """校验图片 sidecar 路径只能是安全的相对子路径，并返回规范化 POSIX 路径。"""
    posix_path = Path(image_path)
    windows_path = PureWindowsPath(image_path)
    if (
        not image_path
        or image_path == "."
        or "\x00" in image_path
        or "\\" in image_path
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"Unsafe image sidecar path: {image_path}")
    return posix_path.as_posix()


def validate_remote_image_url(image_url: str) -> str:
    """校验远程图片只能使用无凭据的 HTTP(S) 绝对地址。"""
    normalized = image_url.strip()
    if (
        not normalized
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in normalized)
        or any(char in normalized for char in ("<", ">", "\\"))
        or normalized.startswith("//")
    ):
        raise ValueError(f"Unsafe remote image URL: {image_url}")
    try:
        parsed = urlsplit(normalized)
        _ = parsed.port
        hostname = parsed.hostname
    except ValueError as exc:
        raise ValueError(f"Unsafe remote image URL: {image_url}") from exc
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.netloc or hostname is None:
        raise ValueError(f"Unsafe remote image URL: {image_url}")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"Remote image URL must not contain credentials: {image_url}")
    return normalized
