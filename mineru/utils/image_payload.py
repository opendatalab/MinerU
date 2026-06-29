# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import base64
from pathlib import Path, PureWindowsPath
import re

from .hash_utils import str_sha256

INLINE_IMAGE_DATA_URI_RE = re.compile(r"data:image/([^;\"']+);base64,([^\"']+)", re.DOTALL)
INLINE_IMAGE_SRC_RE = re.compile(r"src=(?P<quote>[\"'])(?P<data_uri>data:image/.*?)(?P=quote)", re.DOTALL)


def normalize_image_extension(fmt: str) -> str:
    """规范化图片扩展名，保证同一图片格式生成稳定文件名。"""
    normalized = fmt.lower().split("+", 1)[0]
    return "jpg" if normalized in {"jpeg", "jpg"} else normalized


def image_bytes_to_data_uri(img_bytes: bytes, image_format: str = "JPEG") -> str:
    """把图片字节编码成 data URI，作为 middle_json 中的临时图片载荷。"""
    fmt = image_format.lower()
    mime_fmt = "jpeg" if fmt in {"jpg", "jpeg"} else fmt
    payload = base64.b64encode(img_bytes).decode("ascii")
    return f"data:image/{mime_fmt};base64,{payload}"


def parse_image_data_uri(data_uri: str) -> tuple[bytes, str] | None:
    """解析 data URI，返回图片字节和规范化扩展名；非法输入返回 None。"""
    match = re.fullmatch(r"data:image/([^;]+);base64,(.+)", data_uri, re.DOTALL)
    if match is None:
        return None
    try:
        return base64.b64decode(match.group(2)), normalize_image_extension(match.group(1))
    except Exception:
        return None


def image_path_from_data_uri(data_uri: str) -> str | None:
    """根据 data URI 内容生成稳定图片路径，供没有显式 image_path 的 span 使用。"""
    parsed = parse_image_data_uri(data_uri)
    if parsed is None:
        return None
    _, ext = parsed
    return f"{str_sha256(data_uri)}.{ext}"


def image_path_from_key(path_key: str, image_format: str = "JPEG") -> str:
    """复用旧裁图路径哈希规则，根据逻辑路径生成稳定图片文件名。"""
    ext = normalize_image_extension(image_format)
    return f"{str_sha256(path_key)}.{ext}"


class ImagePayloadCache:
    """保存运行时图片载荷，public middle_json 只通过 image_path 引用图片。"""

    def __init__(
        self,
        images: dict[str, bytes] | None = None,
        data_uris: dict[str, str] | None = None,
    ) -> None:
        self._images: dict[str, bytes] = dict(images or {})
        self._data_uris: dict[str, str] = dict(data_uris or {})

    def register_data_uri(self, data_uri: str, image_path: str | None = None) -> str:
        """登记 data URI 图片，返回稳定的相对 image_path；非法载荷返回空串。"""
        parsed = parse_image_data_uri(data_uri)
        img_path = image_path or image_path_from_data_uri(data_uri)
        if parsed is None or not img_path:
            return ""
        self._images[img_path] = parsed[0]
        self._data_uris[img_path] = data_uri
        return img_path

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
        self._data_uris[img_path] = image_bytes_to_data_uri(img_bytes, image_format)
        return img_path

    def replace_html_data_uri_sources(self, markup: str) -> str:
        """把 HTML 内联 data URI 图片替换为 image_path，并同步写入顶层图片缓存。"""
        if not markup or "data:image/" not in markup:
            return markup

        def _replace_src(match: re.Match[str]) -> str:
            img_path = self.register_data_uri(match.group("data_uri"))
            if not img_path:
                return match.group(0)
            return f'src={match.group("quote")}{img_path}{match.group("quote")}'

        replaced = INLINE_IMAGE_SRC_RE.sub(_replace_src, markup)

        def _replace_bare_data_uri(match: re.Match[str]) -> str:
            img_path = self.register_data_uri(match.group(0))
            return img_path or match.group(0)

        return INLINE_IMAGE_DATA_URI_RE.sub(_replace_bare_data_uri, replaced)

    def update(self, images: dict[str, bytes]) -> None:
        """合并外部 sidecar 图片字节，供 API client 绑定远端输出。"""
        for image_path, image_bytes in images.items():
            self.register_bytes(image_bytes, image_path=image_path)

    def images(self) -> dict[str, bytes]:
        """返回 image_path 到图片字节的副本，避免调用方改写内部缓存。"""
        return dict(self._images)

    def data_uris(self) -> dict[str, str]:
        """返回 image_path 到 data URI 的副本，便于调试运行时图片来源。"""
        return dict(self._data_uris)


def validate_image_sidecar_path(image_path: str) -> str:
    """校验图片 sidecar 路径只能是安全的相对子路径，并返回规范化 POSIX 路径。"""
    posix_path = Path(image_path)
    windows_path = PureWindowsPath(image_path)
    if (
        not image_path
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or windows_path.root
        or ".." in posix_path.parts
        or ".." in windows_path.parts
    ):
        raise ValueError(f"Unsafe image sidecar path: {image_path}")
    return posix_path.as_posix()


def collect_image_data_uri_bytes(markup: str, images: dict[str, bytes]) -> None:
    """只收集文本或 HTML 中的 data URI 图片字节，不修改原始内容。"""
    if not markup or "data:image/" not in markup:
        return
    for match in INLINE_IMAGE_DATA_URI_RE.finditer(markup):
        data_uri = match.group(0)
        parsed = parse_image_data_uri(data_uri)
        img_path = image_path_from_data_uri(data_uri)
        if parsed is None or img_path is None:
            continue
        images[img_path] = parsed[0]


def replace_inline_data_uri_sources(markup: str, images: dict[str, bytes]) -> str:
    """把 HTML src 中的 data URI 图片替换为本地路径，并同步收集图片字节。"""
    cache = ImagePayloadCache(images)
    replaced = cache.replace_html_data_uri_sources(markup)
    images.clear()
    images.update(cache.images())
    return replaced
