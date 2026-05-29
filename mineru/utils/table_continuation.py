# Copyright (c) Opendatalab. All rights reserved.

from mineru.utils.char_utils import full_to_half


CONTINUATION_END_MARKERS = [
    "(续)",
    "(续表)",
    "(续上表)",
    "(continued)",
    "(cont.)",
    "(cont’d)",
    "(…continued)",
    "continued",
    "续表",
]

CONTINUATION_INLINE_MARKERS = [
    "(continued)",
]


def is_table_continuation_text(text: str) -> bool:
    """判断文本是否表达续表语义，供表格归组和跨页合并共同复用。"""
    continuation_text = full_to_half((text or "").strip()).lower()
    if not continuation_text:
        return False

    return (
        any(
            _matches_continuation_end_marker(continuation_text, marker.lower())
            for marker in CONTINUATION_END_MARKERS
        )
        or any(marker.lower() in continuation_text for marker in CONTINUATION_INLINE_MARKERS)
    )


def _matches_continuation_end_marker(text: str, marker: str) -> bool:
    """判断续表后缀是否按词边界命中，避免 discontinued 误命中 continued。"""
    if not text.endswith(marker):
        return False

    if marker == "continued":
        marker_start = len(text) - len(marker)
        return marker_start == 0 or not text[marker_start - 1].isalpha()

    return True
