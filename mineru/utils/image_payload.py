# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import base64
import binascii
from pathlib import Path, PureWindowsPath
import re
import xml.etree.ElementTree as ElementTree

from .hash_utils import str_sha256

INLINE_IMAGE_DATA_URI_RE = re.compile(r"data:image/([^;\"']+);base64,([^\"']+)", re.DOTALL)


def normalize_image_extension(fmt: str) -> str:
    """规范化图片扩展名，保证同一图片格式生成稳定文件名。"""
    normalized = fmt.lower().split("+", 1)[0]
    return "jpg" if normalized in {"jpeg", "jpg"} else normalized


def parse_image_data_uri_strict(data_uri: str) -> tuple[bytes, str]:
    """严格解析图片 data URI，并同时校验 MIME 与文件签名是否一致。"""
    match = re.fullmatch(r"data:image/([^;]+);base64,([A-Za-z0-9+/]*={0,2})", data_uri)
    if match is None:
        raise ValueError("Invalid image data URI")

    mime_subtype = match.group(1).lower()
    extension = normalize_image_extension(mime_subtype)
    try:
        payload = base64.b64decode(match.group(2), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image payload") from exc
    if not payload:
        raise ValueError("Image payload must not be empty")

    if extension == "svg":
        if mime_subtype != "svg+xml":
            raise ValueError(f"Unsupported image MIME subtype: {mime_subtype}")
        try:
            svg_root = ElementTree.fromstring(payload)
        except ElementTree.ParseError as exc:
            raise ValueError("Invalid SVG image payload") from exc
        if not isinstance(svg_root.tag, str) or svg_root.tag.rsplit("}", 1)[-1].lower() != "svg":
            raise ValueError("Image signature does not match MIME subtype: svg+xml")
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
            img_path = f"{str_sha256(payload_key)}.{ext}"
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
