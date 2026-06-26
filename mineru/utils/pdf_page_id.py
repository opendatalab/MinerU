# Copyright (c) Opendatalab. All rights reserved.
from __future__ import annotations

from loguru import logger


def get_end_page_id(end_page_id: int | None, pdf_page_num: int) -> int:
    """归一化旧 CLI 的 0-based 结束页，越界时钳制到最后一页。"""
    normalized_end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
    if normalized_end_page_id > pdf_page_num - 1:
        logger.debug("end_page_id is out of range, use images length")
        normalized_end_page_id = pdf_page_num - 1
    return normalized_end_page_id


def parse_page_range(raw: str, page_count: int) -> list[int]:
    """Parse a 1-based page-range string into a 0-based flattened list of page indices.

    Format (aligned with NEXT-API ``page_range``):
        ``"1~5"``       → pages 1 through 5
        ``"1,3,5~7"``   → pages 1, 3, 5, 6, 7
        ``"-5~-1"``     → last 5 pages  (page_count=100 → [95, 96, 97, 98, 99])
        ``"5"``         → single page 5
        ``""``          → all pages

    Separator: ``~`` recommended, ``-`` also accepted.
    Negative values are resolved against *page_count* (e.g. ``-1`` = last page).
    Returns an empty list when every resolved segment is outside ``[0, page_count)``.
    """
    if not raw:
        return list(range(page_count))

    result: list[int] = []
    for segment in raw.split(","):
        segment = segment.strip()
        if not segment:
            continue
        sep = "~" if "~" in segment else "-"
        parts = segment.split(sep, 1)

        def _resolve(token: str) -> int:
            n = int(token.strip())
            return n + page_count if n < 0 else n

        start_1b = _resolve(parts[0])
        end_1b = _resolve(parts[1]) if len(parts) > 1 and parts[1].strip() else start_1b

        start_0b = start_1b - 1  # 1-based → 0-based
        end_0b = end_1b - 1

        lo = max(0, min(start_0b, end_0b))
        hi = min(page_count - 1, max(start_0b, end_0b))
        if lo <= hi:
            result.extend(range(lo, hi + 1))

    if not result:
        return list(range(page_count))
    return sorted(set(result))
