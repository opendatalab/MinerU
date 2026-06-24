# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ...types import BBox


@dataclass(slots=True)
class VlmContentBlockDraft:
    """VLM raw content block 的内部适配层，不作为 middle_json block 输出。"""

    raw_type: str
    bbox: BBox
    angle: int | None = None
    content: str | None = None
    merge_prev: bool = False
    sub_type: str | None = None
    cell_merge: list[int] = field(default_factory=list)

    @classmethod
    def from_content_block(cls, content_block: Mapping[str, Any], width: int, height: int) -> VlmContentBlockDraft:
        """将 mineru_vl_utils 的相对坐标 content block 转为页面绝对坐标 draft。"""
        x1, y1, x2, y2 = content_block["bbox"]
        x_1, y_1, x_2, y_2 = (
            int(x1 * width),
            int(y1 * height),
            int(x2 * width),
            int(y2 * height),
        )
        if x_2 < x_1:
            x_1, x_2 = x_2, x_1
        if y_2 < y_1:
            y_1, y_2 = y_2, y_1

        return cls(
            raw_type=str(content_block["type"]),
            bbox=(x_1, y_1, x_2, y_2),
            angle=content_block.get("angle", 0),
            content=content_block.get("content", ""),
            merge_prev=bool(content_block.get("merge_prev", False)),
            sub_type=content_block.get("sub_type"),
            cell_merge=list(content_block.get("cell_merge") or []),
        )
