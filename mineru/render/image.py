from __future__ import annotations

import re
from collections.abc import Callable

from ..types import BlockBase, BlockType, ImagePayloadBlock, _iter_child_blocks

ImageRenderer = Callable[[BlockBase], str]

_VISUAL_BLOCK_TYPES: frozenset[str] = frozenset({
    BlockType.IMAGE,
    BlockType.TABLE,
    BlockType.CHART,
    BlockType.EQUATION,
})


def image_path_renderer(block: BlockBase, *, img_bucket_path: str = "") -> str:
    """Render the image paths belonging to one top-level visual block."""
    if block.type not in _VISUAL_BLOCK_TYPES:
        return ""
    references: list[str] = []
    if isinstance(block, ImagePayloadBlock) and block.image_path:
        references.append(_build_image_reference(block.image_path, img_bucket_path))
    for child in _iter_child_blocks(block):
        if isinstance(child, ImagePayloadBlock) and child.image_path:
            references.append(_build_image_reference(child.image_path, img_bucket_path))
    return "  \n".join(references)


def _build_image_reference(image_path: str, img_bucket_path: str) -> str:
    media_path = f"{img_bucket_path}/{image_path}" if img_bucket_path else image_path
    return f"![]({media_path})"


def strip_embedded_image_tags(content: str) -> str:
    """Remove embedded HTML images when a custom renderer owns image output."""
    return re.sub(r"<img\b[^>]*>", "", content, flags=re.IGNORECASE)


__all__ = ["ImageRenderer", "image_path_renderer", "strip_embedded_image_tags"]
