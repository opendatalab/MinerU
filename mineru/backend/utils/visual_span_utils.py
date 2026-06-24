# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from typing import Any

from ...types import Block, ContentType
from ...utils.cut_image import cut_image_and_table
from .para_block_utils import iter_block_spans

VISUAL_SPAN_TYPES = {
    ContentType.IMAGE,
    ContentType.TABLE,
    ContentType.CHART,
    ContentType.INTERLINE_EQUATION,
}


def cut_visual_spans_in_blocks(
    blocks: list[Block],
    page_pil_img: Any,
    page_img_md5: str,
    page_index: int,
    image_writer: Any,
    scale: float = 2,
) -> None:
    """在最终 block tree 中裁剪视觉 span，避免 regroup/deepcopy 后 image_path 写回旧对象。"""
    for block in blocks:
        for span in iter_block_spans(block):
            if span.type in VISUAL_SPAN_TYPES:
                cut_image_and_table(
                    span,
                    page_pil_img,
                    page_img_md5,
                    page_index,
                    image_writer,
                    scale=scale,
                )
