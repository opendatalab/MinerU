"""Shared Pydantic models for request/response contracts between client and server."""

from __future__ import annotations

from pydantic import BaseModel


# ── parse ──────────────────────────────────────────────────────────


class ParseRequest(BaseModel):
    path: str
    tier: str | None = None
    pages: str | None = None
    force: bool = False


class ParseResponse(BaseModel):
    sha256: str
    tier: str
    pages: str
    status: str  # pending / parsing / done / failed
    markdown: str | None = None
    tip: str | None = None  # hint for user when status is not 'done'


class ParseStatusResponse(BaseModel):
    sha256: str
    tier: str
    status: str
    pages: str
    markdown: str | None = None
    error: dict | None = None


# ── search ─────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    sha256: str
    title: str | None = None
    author: str | None = None
    filename: str | None = None
    ext: str | None = None
    size_bytes: int | None = None
    tier: str | None = None
    snippet: str | None = None
    paths: list[str] = []


class SearchResponse(BaseModel):
    results: list[SearchResult]
    total: int
    query: str


# ── info ───────────────────────────────────────────────────────────


class FileInfo(BaseModel):
    path: str
    filename: str
    ext: str
    size_bytes: int
    mtime_ms: int
    sha256: str | None = None
    scan_status: str
    # from docs
    title: str | None = None
    author: str | None = None
    page_count: int | None = None
    lang: str | None = None
    is_encrypted: int = 0
    # from parses (aggregated)
    tiers: list[dict] = []  # [{tier, status, pages, done_at}, ...]
    watch_label: str | None = None


# ── config ─────────────────────────────────────────────────────────


class WatchTargetInfo(BaseModel):
    id: int
    path: str
    label: str | None = None
    removable: bool = False
    enabled: bool = True
    watch_status: str


class RuleInfo(BaseModel):
    id: int
    name: str | None = None
    rule_type: str
    pattern: str
    tier: str | None = None
    pages: str | None = None
    remote: bool = False
    enabled: bool = True
    priority: int = 0


# ── server ─────────────────────────────────────────────────────────


class ServerStatus(BaseModel):
    running: bool
    pid: int | None = None
    uptime_seconds: float | None = None
    socket_path: str
    data_dir: str
    files_total: int = 0
    docs_total: int = 0
    parse_queue_length: int = 0
    ingest_queue_length: int = 0
    watch_count: int = 0
    watches: list[dict] = []
