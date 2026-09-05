"""在共享文档对象与 MinerU 产品协议之间进行显式适配。"""

from __future__ import annotations

from docgale.compat.mineru import MinerUMetadata
from pydantic import JsonValue


def build_metadata(*, effort: str, parse_mode: str, mineru_version: str) -> dict[str, JsonValue]:
    """校验产品参数并生成命名空间扩展，不给公共语义类型添加宿主字段。"""
    metadata = MinerUMetadata.model_validate({"effort": effort, "parse_mode": parse_mode, "mineru_version": mineru_version})
    return {"mineru": metadata.model_dump(mode="json")}


__all__ = ["build_metadata"]
