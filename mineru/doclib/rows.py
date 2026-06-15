"""Typed SQLite row shapes used inside doclib services."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from ..types import Tier
from .types import ParseStatus, RuleType, ScanKind, ScanSource, FileStatus, ScanStatus, WatchStatus


class FileRow(TypedDict):
    id: int
    path: str
    filename: str
    ext: str
    size_bytes: int
    mtime_ms: int
    sha256: str | None
    watch_id: int | None
    status: FileStatus
    locked_at: int | None
    error_code: str | None
    error_msg: str | None
    deleted_at: int | None
    first_seen_at: int
    updated_at: int


class DocRow(TypedDict):
    sha256: str
    size_bytes: int
    file_type: str | None
    page_count: int | None
    language: str | None
    title: str | None
    author: str | None
    subject: str | None
    keywords: str | None
    is_image_based: int
    meta_tier: Tier | None
    error_code: str | None
    error_msg: str | None
    first_seen_at: int
    updated_at: int


class ParseRow(TypedDict):
    id: int
    sha256: str
    tier: Tier
    pages: str
    status: ParseStatus
    priority: int
    locked_at: int | None
    error_code: str | None
    error_msg: str | None
    privacy: str
    via: str | None
    done_at: int | None
    created_at: int
    updated_at: int


class ScanRow(TypedDict):
    id: int
    path: str
    kind: ScanKind
    source: ScanSource
    watch_id: int | None
    status: ScanStatus
    locked_at: int | None
    files_seen: int
    files_refreshed: int
    files_new: int
    files_changed: int
    files_deleted: int
    files_unreachable: int
    files_error: int
    files_unsupported: int
    files_excluded: int
    error_code: str | None
    error_msg: str | None
    started_at: int | None
    finished_at: int | None
    created_at: int
    updated_at: int


class WatchTargetRow(TypedDict):
    id: int
    path: str
    label: str | None
    removable: int
    enabled: int
    recursive: int
    status: WatchStatus
    unreachable_at: int | None
    last_scan_at: int | None
    last_scan_files: int
    created_at: int
    updated_at: int


class ExcludeRuleRow(TypedDict):
    id: int
    name: str | None
    pattern: str
    enabled: int
    priority: int
    hit_count: int
    created_at: int
    updated_at: int
    rule_type: NotRequired[RuleType]
    tier: NotRequired[None]
    pages: NotRequired[None]
    remote: NotRequired[int]


class ParsingRuleRow(TypedDict):
    id: int
    name: str | None
    pattern: str
    tier: Tier | None
    pages: str | None
    remote: int
    enabled: int
    priority: int
    hit_count: int
    created_at: int
    updated_at: int
    rule_type: NotRequired[RuleType]


RuleRow = ExcludeRuleRow | ParsingRuleRow


class ConfigRow(TypedDict):
    key: str
    value: str


class IdRow(TypedDict):
    id: int


class PathRow(TypedDict):
    path: str


class CountRow(TypedDict):
    cnt: int


class PageCountRow(TypedDict):
    page_count: int | None


class Sha256Row(TypedDict):
    sha256: str | None


class ParseGroupRow(TypedDict):
    sha256: str
    tier: Tier


class ParseBatchRow(TypedDict):
    pages: str
    done_at: int | None


class IngestTaskRow(TypedDict):
    id: int
    path: str
    watch_id: int | None


class FtsContentSearchRow(TypedDict):
    sha256: str
    title: str
    author: str
    filename: str
    tier: Tier
    snippet: str
    rank: float


class FtsFilenameSearchRow(TypedDict):
    file_id: int
    ext: str
    snippet: str


class SearchFileRow(FileRow):
    title: str | None
    author: str | None
    page_count: int | None
    file_type: str | None


class FilenameSearchFileRow(FileRow):
    title: str | None


class ContentSearchResultRow(TypedDict):
    sha256: str
    title: str | None
    author: str | None
    filename: str | None
    ext: str | None
    size_bytes: int | None
    tier: Tier
    snippet: str
    paths: list[str]


class FilenameSearchResultRow(TypedDict):
    sha256: str | None
    title: str | None
    filename: str
    ext: str | None
    size_bytes: int | None
    tier: str
    snippet: str
    paths: list[str]


class WatchStatsFileRow(TypedDict):
    watch_id: int
    total_files: int
    active_files: int
    deleted_files: int
    unreachable_files: int
    pending_ingest_files: int
    file_error_count: int
    doc_count: int


class WatchParseCountRow(TypedDict):
    watch_id: int
    status: ParseStatus
    cnt: int


class ErrorBucketRow(TypedDict):
    code: str
    cnt: int


class RecentScanRow(TypedDict):
    id: int
    path: str
    kind: ScanKind
    source: ScanSource
    watch_id: int | None
    status: ScanStatus
    files_seen: int
    files_refreshed: int
    files_new: int
    files_changed: int
    files_deleted: int
    files_unreachable: int
    files_error: int
    files_unsupported: int
    files_excluded: int
    error_code: str | None
    started_at: int | None
    finished_at: int | None
    created_at: int
    updated_at: int
