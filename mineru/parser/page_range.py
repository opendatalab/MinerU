# Copyright (c) Opendatalab. All rights reserved.
"""所有入口共享的页码范围语法、求值和格式化；默认选页策略由调用方决定。"""

from __future__ import annotations

import re
from collections.abc import Iterable

from loguru import logger

from ..errors import InvalidRequestError

PAGE_RANGE_DESCRIPTION = (
    "PDF page selection: 1-based inclusive ranges, e.g. '1-5,8,r3-r1'; "
    "'r1' is the last page and 'all' selects every page. Blank means unspecified. "
    "Selections are sorted and deduplicated; out-of-bounds pages are omitted. "
    "Reversed ranges, '~' and negative page numbers are invalid."
)
_SEGMENT_PATTERN = re.compile(r"(r?[1-9][0-9]*)(?:\s*-\s*(r?[1-9][0-9]*))?")


def get_end_page_id(end_page_id: int | None, pdf_page_num: int) -> int:
    """归一化旧 CLI 的 0-based 结束页，越界时钳制到最后一页。"""
    normalized_end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else pdf_page_num - 1
    if normalized_end_page_id > pdf_page_num - 1:
        logger.debug("end_page_id is out of range, use images length")
        normalized_end_page_id = pdf_page_num - 1
    return normalized_end_page_id


def _invalid_range(raw: str | None, reason: str) -> InvalidRequestError:
    """统一生成可跨 Python、CLI 和 HTTP 传递的页码错误。"""
    return InvalidRequestError("page_range_invalid", f"Invalid page range {raw!r}: {reason}", "page_range")


def _parse_endpoint(token: str) -> int:
    """以负整数在内部标记倒数端点，外部输入只接受 rN。"""
    return -int(token[1:]) if token.startswith("r") else int(token)


def _parse_segments(raw: str | None) -> list[tuple[int, int]] | None:
    """解析完整表达式；None 表示全部，倒数端点留待获得总页数后求值。"""
    value = (raw or "").strip()
    if not value or value == "all":
        return None
    segments: list[tuple[int, int]] = []
    for part in value.split(","):
        match = _SEGMENT_PATTERN.fullmatch(part.strip())
        if match is None:
            raise _invalid_range(raw, "expected a page, an inclusive range, or 'all'")
        try:
            start = _parse_endpoint(match[1])
            end = _parse_endpoint(match[2]) if match[2] else start
        except ValueError as exc:
            raise _invalid_range(raw, "page number is too large") from exc
        if (start > 0) == (end > 0) and start > end:
            raise _invalid_range(raw, "reversed ranges are not supported")
        segments.append((start, end))
    return segments


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """合并已求值区间，无需逐页展开即可排序、去重和连接相邻区间。"""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _format_endpoint(value: int) -> str:
    """把内部端点转换为正整数或 rN 文本。"""
    return str(value) if value > 0 else f"r{-value}"


def _format_intervals(intervals: Iterable[tuple[int, int]]) -> str:
    """生成无空白的范围文本，单页只输出一个端点。"""
    return ",".join(
        _format_endpoint(start) if start == end else f"{_format_endpoint(start)}-{_format_endpoint(end)}"
        for start, end in intervals
    )


def normalize_page_range_input(raw: str | None) -> str:
    """校验未绑定文档的输入，清理空白并保留 all/rN；空值统一为未指定。"""
    segments = _parse_segments(raw)
    if segments is None:
        return "all" if (raw or "").strip() else ""
    if all(start > 0 and end > 0 for start, end in segments):
        segments = _merge_intervals(segments)
    return _format_intervals(segments)


def _resolved_intervals(raw: str | None, page_count: int) -> list[tuple[int, int]]:
    """按文档总页数求值并裁剪，先验证所有区间再合并有效交集。"""
    segments = _parse_segments(raw)
    if page_count <= 0:
        raise _invalid_range(raw, "document has no available pages")
    if segments is None:
        return [(1, page_count)]
    intervals: list[tuple[int, int]] = []
    for start, end in segments:
        start = start if start > 0 else page_count + start + 1
        end = end if end > 0 else page_count + end + 1
        if start > end:
            raise _invalid_range(raw, "reversed ranges are not supported")
        lo, hi = max(1, start), min(page_count, end)
        if lo <= hi:
            intervals.append((lo, hi))
    if not intervals:
        raise _invalid_range(raw, "selection does not contain any available pages")
    return _merge_intervals(intervals)


def parse_page_range(raw: str, page_count: int) -> list[int]:
    """把 1-based 表达式转换为去重升序的 0-based 页索引，空值选择全部。"""
    return [page_idx for start, end in _resolved_intervals(raw, page_count) for page_idx in range(start - 1, end)]


def expand_page_range(raw: str | None, page_count: int) -> str:
    """将含 all/rN 的表达式展开为实际文档的正整数规范范围。"""
    return _format_intervals(_resolved_intervals(raw, page_count))


def _absolute_intervals(raw: str) -> list[tuple[int, int]]:
    """读取已求值范围，兼容历史半角 ~；空文本为空集合，禁止 all/rN。"""
    if not raw.strip():
        return []
    # 只在结果读取边界兼容旧分隔符，新请求及全角波浪号仍由严格语法拒绝。
    segments = _parse_segments(raw.replace("~", "-"))
    if segments is None or any(start < 0 or end < 0 for start, end in segments):
        raise _invalid_range(raw, "page count is required to resolve all/rN")
    return _merge_intervals(segments)


def normalize_result_page_range(raw: str) -> str:
    """将已求值的新旧结果范围规范化为连字符格式，不修改持久化记录或文件名。"""
    return _format_intervals(_absolute_intervals(raw))


def parse_page_range_set(raw: str) -> set[int]:
    """把已求值的范围转换为 1-based 页码集合，供缓存覆盖与内容过滤使用。"""
    return {page_no for start, end in _absolute_intervals(raw) for page_no in range(start, end + 1)}


def count_pages_in_range(raw: str) -> int:
    """统计已求值范围的唯一页数，不为统计分配逐页集合。"""
    return sum(end - start + 1 for start, end in _absolute_intervals(raw))


def format_page_range(page_numbers: Iterable[int]) -> str:
    """将 1-based 页码集合格式化为去重升序的连续范围；空集合输出空字符串。"""
    return _format_intervals(_merge_intervals((page_no, page_no) for page_no in page_numbers))


__all__ = [
    "PAGE_RANGE_DESCRIPTION",
    "count_pages_in_range",
    "expand_page_range",
    "format_page_range",
    "get_end_page_id",
    "normalize_page_range_input",
    "normalize_result_page_range",
    "parse_page_range",
    "parse_page_range_set",
]
