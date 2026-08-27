# Copyright (c) Opendatalab. All rights reserved.

"""解析 DOC 标准书签名称及其主文档 CP 范围。"""

from __future__ import annotations

from ..legacy.binary import bounded_slice, get_u16
from .records import DocBudget, parse_plc


def _parse_string_table(data: bytes, budget: DocBudget) -> list[str]:
    """解析扩展或单字节 STTB 字符串表。"""

    if len(data) < 2:
        return []
    extended = get_u16(data, 0) == 0xFFFF
    if extended:
        count = get_u16(data, 2)
        extra = get_u16(data, 4)
        cursor = 6
    else:
        count = get_u16(data, 0)
        extra = get_u16(data, 2)
        cursor = 4
    if count is None or extra is None:
        return []
    strings: list[str] = []
    for _ in range(count):
        if extended:
            length = get_u16(data, cursor)
            cursor += 2
            width = 2
        else:
            length = data[cursor] if cursor < len(data) else None
            cursor += 1
            width = 1
        if length is None or length == 0xFFFF:
            strings.append("")
            continue
        payload = bounded_slice(data, cursor, length * width)
        if payload is None:
            break
        cursor += length * width
        strings.append(payload.decode("utf-16le" if extended else "cp1252", errors="replace"))
        cursor += extra
        budget.charge()
    return strings


def parse_bookmarks(
    table_stream: bytes,
    *,
    names_offset: int,
    names_size: int,
    starts_offset: int,
    starts_size: int,
    ends_offset: int,
    ends_size: int,
    budget: DocBudget,
) -> dict[int, list[str]]:
    """返回主文档中书签起始 CP 到名称列表的映射。"""

    names_payload = bounded_slice(table_stream, names_offset, names_size)
    starts_payload = bounded_slice(table_stream, starts_offset, starts_size)
    ends_payload = bounded_slice(table_stream, ends_offset, ends_size)
    if names_payload is None or starts_payload is None or ends_payload is None:
        return {}
    names = _parse_string_table(names_payload, budget)
    start_cps, start_items = parse_plc(starts_payload, item_size=4, budget=budget)
    end_cps, _ = parse_plc(ends_payload, item_size=0, budget=budget)
    result: dict[int, list[str]] = {}
    for index, (name, item) in enumerate(zip(names, start_items, strict=False)):
        if not name or name.startswith("_GoBack") or index >= len(start_cps):
            continue
        end_index = get_u16(item, 0)
        if end_index is None or end_index >= max(len(end_cps) - 1, 0):
            continue
        start = start_cps[index]
        end = end_cps[end_index]
        if end < start:
            continue
        result.setdefault(start, []).append(name)
    return result
