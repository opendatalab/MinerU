"""Doclib content locator helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..types import Tier

_CURSOR_RE = re.compile(
    r"^doc:(?P<short_id>[0-9a-fA-F]+)"
    r"/tier:(?P<tier>flash|medium|high|extra_high)"
    r"/page:(?P<page_no>[1-9][0-9]*)"
    r"(?:/block:(?P<block_no>[1-9][0-9]*)(?:/char:(?P<char_offset>0|[1-9][0-9]*))?)?$"
)


@dataclass(frozen=True)
class ContentCursor:
    short_id: str
    tier: Tier
    page_no: int
    block_no: int | None = None
    char_offset: int | None = None


def locator_for_block(page_no: int, block_no: int) -> str:
    _validate_positive("page_no", page_no)
    _validate_positive("block_no", block_no)
    return f"page:{page_no}/block:{block_no}"


def page_ref(short_id: str, tier: Tier, page_no: int) -> str:
    _validate_short_id(short_id)
    _validate_positive("page_no", page_no)
    return f"doc:{short_id}/tier:{tier}/page:{page_no}"


def block_ref(short_id: str, tier: Tier, page_no: int, block_no: int) -> str:
    _validate_positive("block_no", block_no)
    return f"{page_ref(short_id, tier, page_no)}/block:{block_no}"


def block_char_ref(short_id: str, tier: Tier, page_no: int, block_no: int, char_offset: int) -> str:
    _validate_non_negative("char_offset", char_offset)
    return f"{block_ref(short_id, tier, page_no, block_no)}/char:{char_offset}"


def parse_content_cursor(ref: str) -> ContentCursor:
    match = _CURSOR_RE.match(ref)
    if match is None:
        raise ValueError(f"Invalid doclib content cursor: {ref}")

    block_no = match.group("block_no")
    char_offset = match.group("char_offset")
    return ContentCursor(
        short_id=match.group("short_id"),
        tier=match.group("tier"),  # type: ignore[arg-type]
        page_no=int(match.group("page_no")),
        block_no=int(block_no) if block_no is not None else None,
        char_offset=int(char_offset) if char_offset is not None else None,
    )


def _validate_short_id(short_id: str) -> None:
    if not short_id or not re.fullmatch(r"[0-9a-fA-F]+", short_id):
        raise ValueError("short_id must be a non-empty hex string")


def _validate_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1")


def _validate_non_negative(name: str, value: int) -> None:
    if value < 0:
        raise ValueError(f"{name} must be >= 0")


__all__ = [
    "ContentCursor",
    "block_char_ref",
    "block_ref",
    "locator_for_block",
    "page_ref",
    "parse_content_cursor",
]
