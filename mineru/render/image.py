from __future__ import annotations

import re
from collections.abc import Callable

from ..types import Block, BlockType, ContentType

ImageRenderer = Callable[[Block], str]

_VISUAL_SPAN_TYPES: dict[str, str] = {
    BlockType.IMAGE: ContentType.IMAGE,
    BlockType.TABLE: ContentType.TABLE,
    BlockType.CHART: ContentType.CHART,
    BlockType.INTERLINE_EQUATION: ContentType.INTERLINE_EQUATION,
}


def image_path_renderer(block: Block, *, img_bucket_path: str = "") -> str:
    """Render the image paths belonging to one top-level visual block."""
    span_type = _VISUAL_SPAN_TYPES.get(block.type)
    if span_type is None:
        return ""
    references: list[str] = []
    for span in block.all_spans():
        if span.type != span_type or not span.image_path:
            continue
        media_path = f"{img_bucket_path}/{span.image_path}" if img_bucket_path else span.image_path
        references.append(f"![]({media_path})")
    return "  \n".join(references)


def strip_embedded_image_tags(content: str) -> str:
    """Remove embedded HTML images when a custom renderer owns image output."""
    return re.sub(r"<img\b[^>]*>", "", content, flags=re.IGNORECASE)


__all__ = ["ImageRenderer", "image_path_renderer"]
