"""Doclib 持久化解析产物：路径、协议读取、页选择和覆盖计算。"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from ...errors import MineruError
from ...parser.page_range import format_page_range as _page_numbers_to_range_str, parse_page_range_set
from ...types import PageInfo, Tier
from ..rows import FileRow, ParseBatchRow, ParseRow
from ..types import FILE_STATUS_ACTIVE, PARSE_STATUS_DONE, PARSE_STATUS_PENDING, PARSE_STATUS_PARSING, ParseCoverage
from .db import DatabaseManager
from .middle_json import read_cached_middle_json


def parse_artifact_directory(data_dir: str, sha256: str, tier: Tier) -> str:
    """统一解析批次目录，供任务写出、内容读取和后台压缩使用。"""
    return os.path.join(os.path.expanduser(data_dir), "parsed", sha256[:2], sha256, tier)


def parse_batch_json_path(data_dir: str, sha256: str, tier: Tier, page_range: str, done_at: int | None = 0) -> str:
    """生成持久化批次路径，保持现有页范围与完成时间文件名约定。"""
    filename = f"{page_range}_{done_at}.json" if done_at else page_range
    return os.path.join(parse_artifact_directory(data_dir, sha256, tier), filename)


def write_parse_payload(path: str | Path, payload: dict[str, Any]) -> None:
    """按现有 JSON 编码写出批次；目录、原子替换和事务时机由调用者协调。"""
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)


async def accessible_file_for_sha256(db: DatabaseManager, sha256: str) -> FileRow | None:
    """按更新时间寻找仍可访问的同内容源文件，跳过尚未刷新的失效路径。"""
    rows = cast(
        list[FileRow],
        await db.fetchall(
            "SELECT * FROM files WHERE sha256=? AND status=? ORDER BY updated_at DESC",
            (sha256, FILE_STATUS_ACTIVE),
        ),
    )
    for row in rows:
        if os.path.exists(row["path"]):
            return row
    return None


def filter_pages_by_user_range(pages: list[PageInfo], page_range: str) -> list[PageInfo]:
    """按用户的一基页码筛选缓存中的零基页面。"""
    requested_page_numbers = parse_page_range_set(page_range)
    return [page for page in pages if page.page_idx + 1 in requested_page_numbers]


def parse_batch_is_readable(data_dir: str, sha256: str, tier: Tier, batch: ParseRow) -> bool:
    """将可转换为当前协议且覆盖记录页范围的持久化批次视为有效缓存。"""
    json_path = parse_batch_json_path(data_dir, sha256, tier, batch["page_range"], batch["done_at"])
    try:
        result = read_cached_middle_json(json_path)
        available = {page.page_idx + 1 for page in result.pages}
        return bool(available) and parse_page_range_set(batch["page_range"]) <= available
    except (OSError, ValueError, TypeError):
        return False


def load_pages_from_done_batches(
    data_dir: str,
    sha256: str,
    tier: Tier,
    done_rows: Sequence[ParseBatchRow],
    *,
    requested_page_numbers: set[int] | None = None,
) -> list[PageInfo]:
    """读取当前或受支持的历史批次，保留最新重复页；损坏或未知缓存须重新解析。"""
    pages_by_page_idx: dict[int, PageInfo] = {}
    stale_page_numbers: set[int] = set()
    for row in reversed(done_rows):
        fpath = parse_batch_json_path(data_dir, sha256, tier, row["page_range"], row["done_at"])
        if not os.path.isfile(fpath):
            continue
        try:
            parse_result = read_cached_middle_json(fpath)
            for page in parse_result.pages:
                pages_by_page_idx[page.page_idx] = page
        except (OSError, ValueError, TypeError):
            stale_page_numbers.update(parse_page_range_set(row["page_range"]))
    uncovered_stale = stale_page_numbers - {index + 1 for index in pages_by_page_idx}
    if requested_page_numbers is not None:
        uncovered_stale &= requested_page_numbers
    if uncovered_stale:
        raise MineruError(
            "stale_cache", "Cached document protocol is outdated or invalid; reparse the source document.", "page_range"
        )
    return [pages_by_page_idx[page_idx] for page_idx in sorted(pages_by_page_idx)]


def parse_coverage(request_page_range: str, rows: list[ParseRow]) -> ParseCoverage:
    """统一计算请求页范围内的已完成、处理中和缺失覆盖。"""
    requested_page_numbers = parse_page_range_set(request_page_range)
    done_page_numbers: set[int] = set()
    active_page_numbers: set[int] = set()
    for row in rows:
        row_page_numbers = parse_page_range_set(row["page_range"]) & requested_page_numbers
        if row["status"] == PARSE_STATUS_DONE:
            done_page_numbers |= row_page_numbers
        elif row["status"] in {PARSE_STATUS_PENDING, PARSE_STATUS_PARSING}:
            active_page_numbers |= row_page_numbers
    active_page_numbers -= done_page_numbers
    missing_page_numbers = requested_page_numbers - done_page_numbers - active_page_numbers
    return ParseCoverage(
        done_page_range=_page_numbers_to_range_str(done_page_numbers),
        active_page_range=_page_numbers_to_range_str(active_page_numbers),
        missing_page_range=_page_numbers_to_range_str(missing_page_numbers),
    )


__all__ = [
    "accessible_file_for_sha256",
    "filter_pages_by_user_range",
    "load_pages_from_done_batches",
    "parse_artifact_directory",
    "parse_batch_is_readable",
    "parse_batch_json_path",
    "parse_coverage",
    "write_parse_payload",
]
