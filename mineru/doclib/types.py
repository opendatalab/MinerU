"""Typed request and response models for the doclib public interface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..types import Tier

ParseStatus = Literal["pending", "parsing", "done", "failed", "superseded"]
ScanStatus = Literal["active", "deleted", "unreachable"]
RuleType = Literal["exclude", "parsing_rule"]
WatchStatus = Literal["active", "unreachable"]
InvalidateTarget = Literal["parses"]

PARSE_STATUS_PENDING: ParseStatus = "pending"
PARSE_STATUS_PARSING: ParseStatus = "parsing"
PARSE_STATUS_DONE: ParseStatus = "done"
PARSE_STATUS_FAILED: ParseStatus = "failed"
PARSE_STATUS_SUPERSEDED: ParseStatus = "superseded"

SCAN_STATUS_ACTIVE: ScanStatus = "active"
SCAN_STATUS_DELETED: ScanStatus = "deleted"
SCAN_STATUS_UNREACHABLE: ScanStatus = "unreachable"

RULE_TYPE_EXCLUDE: RuleType = "exclude"
RULE_TYPE_PARSING_RULE: RuleType = "parsing_rule"

WATCH_STATUS_ACTIVE: WatchStatus = "active"
WATCH_STATUS_UNREACHABLE: WatchStatus = "unreachable"


class DoclibModel(BaseModel):
    """Base model for doclib interface schemas."""


class ErrorInfo(DoclibModel):
    type: str
    code: str
    message: str
    param: str | None = None


class ErrorResponse(DoclibModel):
    error: ErrorInfo


class ShutdownResponse(DoclibModel):
    accepted: bool
    message: str


class RemoveWatchResponse(DoclibModel):
    path: str
    removed: bool


class RemoveExcludeRuleResponse(DoclibModel):
    rule_id: int
    removed: bool


class RemoveParsingRuleResponse(DoclibModel):
    rule_id: int
    removed: bool


class ParseRequest(DoclibModel):
    path: str
    tier: Tier | None = None
    pages: str | None = None
    force: bool = False
    remote: bool = False


class ParseResponse(DoclibModel):
    sha256: str
    tier: Tier
    pages: str
    status: ParseStatus
    cache_hit: bool = False
    wait_parse_ids: list[int] = Field(default_factory=list)
    created_parse_ids: list[int] = Field(default_factory=list)
    reused_parse_ids: list[int] = Field(default_factory=list)
    tip: str | None = None


class ParseCoverage(DoclibModel):
    done_pages: str
    active_pages: str
    missing_pages: str


class ParseInfo(DoclibModel):
    id: int
    sha256: str
    tier: Tier
    pages: str
    status: ParseStatus
    priority: int = 0
    privacy: str
    via: str | None = None
    coverage: ParseCoverage | None = None
    created_at: int
    updated_at: int
    done_at: int | None = None
    error_code: str | None = None
    error_msg: str | None = None


class ListParsesResponse(DoclibModel):
    parses: list[ParseInfo] = Field(default_factory=list)
    coverage: ParseCoverage | None = None


class InvalidateRequest(DoclibModel):
    target: InvalidateTarget = "parses"
    path: str | None = None
    sha256: str | None = None
    tier: Tier | None = None


class InvalidateResponse(DoclibModel):
    target: str
    sha256: str
    tier: Tier | None = None
    invalidated_count: int


class FileInfo(DoclibModel):
    filename: str
    path: str
    ext: str
    size_bytes: int
    mtime_ms: int
    birthtime_ms: int | None = None
    sha256: str | None = None
    watch_id: int | None = None
    scan_status: ScanStatus
    error_code: str | None = None
    error_msg: str | None = None
    deleted_at: int | None = None
    first_seen_at: int
    updated_at: int


class DocInfo(DoclibModel):
    sha256: str
    size_bytes: int
    mime_type: str | None = None
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    page_count: int | None = None
    lang: str | None = None
    is_scanned: int = 0
    meta_tier: Tier | None = None
    error_code: str | None = None
    error_msg: str | None = None
    first_seen_at: int
    updated_at: int
    files: list[FileInfo] | None = None


class ListDocsResponse(DoclibModel):
    docs: list[DocInfo] = Field(default_factory=list)


class DocContentResponse(DoclibModel):
    sha256: str
    tier: Tier
    content: str | None = None
    output: str | None = None


class SearchResult(DoclibModel):
    sha256: str
    title: str | None = None
    author: str | None = None
    filename: str | None = None
    ext: str | None = None
    size_bytes: int | None = None
    page_count: int | None = None
    tier: Tier
    snippet: str
    paths: list[str] = Field(default_factory=list)


class FindResult(DoclibModel):
    filename: str
    ext: str | None = None
    size_bytes: int | None = None
    page_count: int | None = None
    paths: list[str] = Field(default_factory=list)


class SearchResponse(DoclibModel):
    results: list[SearchResult] = Field(default_factory=list)
    total: int
    query: str


class FindResponse(DoclibModel):
    results: list[FindResult] = Field(default_factory=list)
    total: int
    query: str


class FileInfoResponse(DoclibModel):
    file: FileInfo
    doc: DocInfo | None = None
    parsed_tiers: list["TierParseInfo"] = Field(default_factory=list)
    active_parses: list[ParseInfo] = Field(default_factory=list)


class TierParseInfo(DoclibModel):
    tier: Tier
    status: ParseStatus
    pages: str
    done_at: int | None = None


class ConfigSetRequest(DoclibModel):
    key: str
    value: str


class ConfigResponse(DoclibModel):
    config: dict[str, str] = Field(default_factory=dict)


class ConfigSetResponse(DoclibModel):
    key: str
    value: str


class WatchRequest(DoclibModel):
    path: str
    removable: bool = False
    label: str | None = None


class WatchInfo(DoclibModel):
    id: int
    path: str
    label: str | None = None
    removable: bool = False
    enabled: bool = True
    recursive: bool = False
    watch_status: WatchStatus
    unreachable_at: int | None = None
    last_scan_at: int | None = None
    last_scan_files: int = 0
    created_at: int = 0
    updated_at: int = 0


class WatchListResponse(DoclibModel):
    watches: list[WatchInfo] = Field(default_factory=list)


class ExcludeRuleRequest(DoclibModel):
    pattern: str
    name: str | None = None
    priority: int = 0


class ParsingRuleRequest(DoclibModel):
    pattern: str
    name: str | None = None
    tier: Tier | None = None
    pages: str | None = None
    remote: bool = False
    priority: int = 0


class ExcludeRuleInfo(DoclibModel):
    id: int
    pattern: str
    name: str | None = None
    enabled: bool = True
    priority: int = 0
    hit_count: int = 0
    created_at: int = 0
    updated_at: int = 0


class ParsingRuleInfo(DoclibModel):
    id: int
    pattern: str
    name: str | None = None
    tier: Tier | None = None
    pages: str | None = None
    remote: bool = False
    enabled: bool = True
    priority: int = 0
    hit_count: int = 0
    created_at: int = 0
    updated_at: int = 0


class ExcludeRuleListResponse(DoclibModel):
    rules: list[ExcludeRuleInfo] = Field(default_factory=list)


class ParsingRuleListResponse(DoclibModel):
    rules: list[ParsingRuleInfo] = Field(default_factory=list)


class LocalParseServerStatus(DoclibModel):
    mode: str | None = None
    healthy: bool = False
    starting: bool = False
    started_at: float | int | None = None
    supported_tiers: list[Tier] = Field(default_factory=list)


class RemoteParseServerStatus(DoclibModel):
    healthy: bool = False
    supported_tiers: list[Tier] = Field(default_factory=list)


class ParseServerStatus(DoclibModel):
    local: LocalParseServerStatus = Field(default_factory=LocalParseServerStatus)
    remote: RemoteParseServerStatus = Field(default_factory=RemoteParseServerStatus)


class ServerStatusResponse(DoclibModel):
    running: bool
    pid: int | None = None
    uptime_seconds: float | None = None
    socket_path: str
    data_dir: str
    files_total: int = 0
    docs_total: int = 0
    parse_queue_length: int = 0
    ingest_queue_length: int = 0
    parse_server: ParseServerStatus | None = None
    watch_count: int = 0
    watches: list[WatchInfo] = Field(default_factory=list)
    recent_logs: list[str] = Field(default_factory=list)


class CleanupDeletedRequest(DoclibModel):
    older_than_days: int = 30
    dry_run: bool = True


class CleanupDeletedResponse(DoclibModel):
    deleted_files: int
    dry_run: bool


class CleanupOrphansRequest(DoclibModel):
    dry_run: bool = True


class CleanupOrphansResponse(DoclibModel):
    orphan_docs: int
    dry_run: bool


class CleanupTempRequest(DoclibModel):
    older_than_days: int = 7


class CleanupTempResponse(DoclibModel):
    temp_files_removed: int
