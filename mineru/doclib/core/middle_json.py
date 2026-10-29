"""Doclib 持久化缓存的 Middle JSON 协议读取边界。"""

from __future__ import annotations

import json
from pathlib import Path

from ...backend.postprocess.legacy_middle_json import read_legacy_middle_json
from ...types import MiddleJson


def read_cached_middle_json(path: str | Path) -> MiddleJson:
    """仅在 Doclib 持久化缓存读取边界接受受支持的历史协议。"""
    with open(path, encoding="utf-8") as stream:
        return read_legacy_middle_json(json.load(stream))


__all__ = ["read_cached_middle_json"]
