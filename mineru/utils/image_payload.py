# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

import base64
import re

from .hash_utils import str_sha256

INLINE_IMAGE_DATA_URI_RE = re.compile(r"data:image/([^;\"']+);base64,([^\"']+)", re.DOTALL)
INLINE_IMAGE_SRC_RE = re.compile(r'src="(data:image/[^"]+)"', re.DOTALL)


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
    if not markup or "data:image/" not in markup:
        return markup

    def _store_data_uri(data_uri: str) -> str | None:
        parsed = parse_image_data_uri(data_uri)
        img_path = image_path_from_data_uri(data_uri)
        if parsed is None or img_path is None:
            return None
        images[img_path] = parsed[0]
        return img_path

    def _replace_src(match: re.Match[str]) -> str:
        img_path = _store_data_uri(match.group(1))
        if img_path is None:
            return match.group(0)
        return f'src="{img_path}"'

    replaced = INLINE_IMAGE_SRC_RE.sub(_replace_src, markup)

    def _replace_bare_data_uri(match: re.Match[str]) -> str:
        img_path = _store_data_uri(match.group(0))
        return img_path or match.group(0)

    return INLINE_IMAGE_DATA_URI_RE.sub(_replace_bare_data_uri, replaced)
