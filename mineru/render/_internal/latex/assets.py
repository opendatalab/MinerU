# Copyright (c) Opendatalab. All rights reserved.
"""LaTeX renderer 的纯路径 sidecar 图片解析。"""

from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import urlsplit

from ....types import ImagePayloadBlock
from ....utils.image_payload import validate_image_sidecar_path

_SUPPORTED_IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".pdf", ".png"}
_FORBIDDEN_TEX_PATH_CHARACTERS = {"%", "{", "}"}


def normalize_asset_base_path(asset_base_path: str) -> str:
    """把调用方路径前缀规范化为 TeX 可跨平台读取的正斜杠形式。"""
    if not isinstance(asset_base_path, str):
        raise TypeError("asset_base_path must be a string")
    if any(character == "\x00" or character in "\r\n" for character in asset_base_path):
        raise ValueError("asset_base_path must not contain control characters")
    normalized = asset_base_path.replace("\\", "/")
    if normalized == "/" or (len(normalized) == 3 and normalized[1:] == ":/"):
        return normalized
    return normalized.rstrip("/")


def resolve_block_image_path(block: ImagePayloadBlock, asset_base_path: str) -> str | None:
    """只从 block 的安全 sidecar 路径解析 XeLaTeX 可直接读取的图片。"""
    if block.image_path is None:
        return None
    return resolve_relative_image_path(block.image_path, asset_base_path)


def resolve_html_image_path(source: str, asset_base_path: str) -> str | None:
    """解析 HTML img 的安全相对路径，拒绝 data URI 与远程地址。"""
    if not isinstance(source, str):
        return None
    normalized = source.strip()
    if not normalized:
        return None
    parsed = urlsplit(normalized)
    if parsed.scheme or parsed.netloc or normalized.startswith(("/", "#")):
        return None
    try:
        return resolve_relative_image_path(normalized, asset_base_path)
    except ValueError:
        return None


def resolve_relative_image_path(relative_path: str, asset_base_path: str) -> str | None:
    """拼接安全相对路径，并限制为 XeLaTeX 原生支持的图片扩展名。"""
    safe_path = validate_image_sidecar_path(relative_path)
    if PurePosixPath(safe_path).suffix.casefold() not in _SUPPORTED_IMAGE_EXTENSIONS:
        return None
    if not asset_base_path:
        resolved = safe_path
    elif asset_base_path.endswith("/"):
        resolved = f"{asset_base_path}{safe_path}"
    else:
        resolved = f"{asset_base_path}/{safe_path}"
    if any(character in _FORBIDDEN_TEX_PATH_CHARACTERS for character in resolved):
        return None
    if any(ord(character) < 0x20 or ord(character) == 0x7F for character in resolved):
        return None
    return resolved


def remote_block_image_url(block: ImagePayloadBlock) -> str | None:
    """返回 block 已由严格模型校验过的远程图片 URL。"""
    return block.image_url


def tex_image_path(path: str) -> str:
    """使用 detokenize 包装已校验路径，保留空格及 TeX 保留字符。"""
    return rf"\detokenize{{{path}}}"


__all__ = [
    "normalize_asset_base_path",
    "remote_block_image_url",
    "resolve_block_image_path",
    "resolve_html_image_path",
    "resolve_relative_image_path",
    "tex_image_path",
]
