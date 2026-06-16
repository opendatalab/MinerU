"""Typed request and response models for the doclib public interface."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from ..types import Tier

ParseStatus = Literal["pending", "parsing", "done", "failed", "superseded"]
FileStatus = Literal["active", "deleted", "unreachable"]
ScanStatus = Literal["pending", "running", "done", "failed"]
ScanKind = Literal["manual", "watch"]
ScanSource = Literal["unknown", "cli", "sdk", "api", "watch", "system"]
RuleType = Literal["exclude", "parsing_rule"]
WatchStatus = Literal["active", "unreachable"]
InvalidateTarget = Literal["parses"]
ForgetMatchedAs = Literal["file", "directory", "none"]
ConfigSource = Literal["default", "override"]
ContentFormat = Literal["markdown"]

PARSE_STATUS_PENDING: ParseStatus = "pending"
PARSE_STATUS_PARSING: ParseStatus = "parsing"
PARSE_STATUS_DONE: ParseStatus = "done"
PARSE_STATUS_FAILED: ParseStatus = "failed"
PARSE_STATUS_SUPERSEDED: ParseStatus = "superseded"

FILE_STATUS_ACTIVE: FileStatus = "active"
FILE_STATUS_DELETED: FileStatus = "deleted"
FILE_STATUS_UNREACHABLE: FileStatus = "unreachable"

SCAN_STATUS_PENDING: ScanStatus = "pending"
SCAN_STATUS_RUNNING: ScanStatus = "running"
SCAN_STATUS_DONE: ScanStatus = "done"
SCAN_STATUS_FAILED: ScanStatus = "failed"

SCAN_KIND_MANUAL: ScanKind = "manual"
SCAN_KIND_WATCH: ScanKind = "watch"

SCAN_SOURCE_UNKNOWN: ScanSource = "unknown"
SCAN_SOURCE_CLI: ScanSource = "cli"
SCAN_SOURCE_SDK: ScanSource = "sdk"
SCAN_SOURCE_API: ScanSource = "api"
SCAN_SOURCE_WATCH: ScanSource = "watch"
SCAN_SOURCE_SYSTEM: ScanSource = "system"

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
    watch_id: int
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
    page_range: str | None = None
    force: bool = False
    remote: bool = False


class ParseResponse(DoclibModel):
    sha256: str
    tier: Tier
    page_range: str
    status: ParseStatus
    cache_hit: bool = False
    wait_parse_ids: list[int] = Field(default_factory=list)
    created_parse_ids: list[int] = Field(default_factory=list)
    reused_parse_ids: list[int] = Field(default_factory=list)
    tip: str | None = None


class ParseCoverage(DoclibModel):
    done_page_range: str
    active_page_range: str
    missing_page_range: str


class ParseInfo(DoclibModel):
    id: int
    sha256: str
    tier: Tier
    page_range: str
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
    total: int
    limit: int
    offset: int = 0


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


class ForgetPathRequest(DoclibModel):
    path: str
    dry_run: bool = True


class ForgetPathResponse(DoclibModel):
    path: str
    matched_as: ForgetMatchedAs
    forgotten_files: int
    dry_run: bool
    warnings: list[str] = Field(default_factory=list)


class ScanRequest(DoclibModel):
    path: str
    kind: ScanKind = "manual"
    source: ScanSource = "unknown"
    watch_id: int | None = None


class ScanInfo(DoclibModel):
    id: int
    path: str
    kind: ScanKind
    source: ScanSource = "unknown"
    watch_id: int | None = None
    status: ScanStatus
    files_seen: int = 0
    files_refreshed: int = 0
    files_new: int = 0
    files_changed: int = 0
    files_deleted: int = 0
    files_unreachable: int = 0
    files_error: int = 0
    files_unsupported: int = 0
    files_excluded: int = 0
    error_code: str | None = None
    error_msg: str | None = None
    started_at: int | None = None
    finished_at: int | None = None
    created_at: int
    updated_at: int


class ScanListResponse(DoclibModel):
    scans: list[ScanInfo] = Field(default_factory=list)
    total: int
    limit: int
    offset: int = 0


class FileInfo(DoclibModel):
    filename: str
    path: str
    ext: str
    size_bytes: int
    mtime_ms: int
    sha256: str | None = None
    watch_id: int | None = None
    status: FileStatus
    error_code: str | None = None
    error_msg: str | None = None
    deleted_at: int | None = None
    first_seen_at: int
    updated_at: int


class ListFilesResponse(DoclibModel):
    files: list[FileInfo] = Field(default_factory=list)
    total: int
    limit: int
    offset: int = 0


class DocInfo(DoclibModel):
    sha256: str
    short_id: str
    size_bytes: int
    file_type: str | None = None
    title: str | None = None
    author: str | None = None
    subject: str | None = None
    keywords: str | None = None
    page_count: int | None = None
    language: str | None = None
    is_image_based: bool = False
    meta_tier: Tier | None = None
    error_code: str | None = None
    error_msg: str | None = None
    first_seen_at: int
    updated_at: int
    files: list[FileInfo] | None = None


class ListDocsResponse(DoclibModel):
    docs: list[DocInfo] = Field(default_factory=list)
    total: int
    limit: int
    offset: int = 0


class DocContentResponse(DoclibModel):
    sha256: str
    short_id: str
    tier: Tier
    format: ContentFormat = "markdown"
    content: str
    request_scope: "ContentRequestScope"
    content_ranges: list["ContentRange"] = Field(default_factory=list)
    truncated: bool = False
    next_request: "ContentNextRequest | None" = None


class ContentRequestScope(DoclibModel):
    page_range: str | None = None
    after: str | None = None
    limit: int = 30000


class ContentRange(DoclibModel):
    page_range: str | None = None
    start: str
    end: str


class ContentNextRequest(DoclibModel):
    page_range: str | None = None
    after: str | None = None


class DocContentExportRequest(DoclibModel):
    tier: Tier
    page_range: str | None = None
    format: str = "markdown"
    output: str
    no_marker: bool = False


class DocContentExportResponse(DoclibModel):
    sha256: str
    tier: Tier
    output: str


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
    page_range: str
    done_at: int | None = None


class ConfigValueRequest(DoclibModel):
    value: str


class ConfigResponse(DoclibModel):
    config: dict[str, str] = Field(default_factory=dict)
    sources: dict[str, ConfigSource] = Field(default_factory=dict)


class ConfigValueResponse(DoclibModel):
    key: str
    value: str
    source: ConfigSource


class ConfigSetRequest(ConfigValueRequest):
    pass


class ConfigSetResponse(ConfigValueResponse):
    pass


class ConfigUnsetResponse(ConfigValueResponse):
    removed: bool = True


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
    status: WatchStatus
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
    page_range: str | None = None
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
    page_range: str | None = None
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


class WatchStats(DoclibModel):
    watch_id: int
    path: str
    label: str | None = None
    removable: bool = False
    status: WatchStatus
    total_files: int = 0
    active_files: int = 0
    deleted_files: int = 0
    unreachable_files: int = 0
    pending_ingest_files: int = 0
    file_error_count: int = 0
    doc_count: int = 0
    parse_pending_count: int = 0
    parse_parsing_count: int = 0
    parse_failed_count: int = 0
    parse_done_count: int = 0
    last_scan_at: int | None = None
    last_scan_files: int = 0


class ErrorBucket(DoclibModel):
    code: str
    count: int


class ErrorSummary(DoclibModel):
    file_errors: list[ErrorBucket] = Field(default_factory=list)
    doc_errors: list[ErrorBucket] = Field(default_factory=list)
    parse_errors: list[ErrorBucket] = Field(default_factory=list)


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
    watch_stats: list[WatchStats] = Field(default_factory=list)
    recent_scans: list[ScanInfo] = Field(default_factory=list)
    error_summary: ErrorSummary | None = None
    recent_logs: list[str] = Field(default_factory=list)


class CleanupDeletedRequest(DoclibModel):
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
