# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from pathlib import Path


def normalize_upload_filename(upload_name: str) -> str:
    """清理上传文件名，只保留 basename 并限制 stem 的 UTF-8 字节长度。"""
    max_stem_bytes = 200
    sanitized_name = Path(upload_name).name
    sanitized_path = Path(sanitized_name)
    encoded_stem = sanitized_path.stem.encode("utf-8")
    if len(encoded_stem) <= max_stem_bytes:
        normalized_stem = sanitized_path.stem
    else:
        truncated = encoded_stem[:max_stem_bytes]
        # 截断时回退到合法 UTF-8 边界，避免中文等多字节字符被切坏。
        while truncated:
            try:
                normalized_stem = truncated.decode("utf-8")
                break
            except UnicodeDecodeError as exc:
                truncated = truncated[: exc.start]
        else:
            normalized_stem = ""
    return f"{normalized_stem}{sanitized_path.suffix}"
