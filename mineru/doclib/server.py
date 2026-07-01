"""New doclib HTTP server implementation backed by the public interface."""

from __future__ import annotations

import asyncio
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Iterator, cast
from urllib.parse import urlparse

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

from ..config import config
from ..errors import InvalidRequestError, MineruError, NotFoundError, error_response, http_status_for
from ..render import render_markdown
from ..render.markdown import blocks_to_markdown
from ..render.office.output import blocks_to_markdown as office_blocks_to_markdown
from ..types import EMPTY_BBOX, TIER_ORDER, TIERS, Block, PageInfo, Span, Tier
from ..utils.pdf_document import PDFDocument
from ..version import __version__
from .background.parse_server_health import get_health
from .base import AsyncDoclibInterface
from .core.db import DatabaseManager
from .locators import ContentCursor, block_char_ref, block_ref, page_ref, parse_content_cursor
from .rows import (
    ContentSearchResultRow,
    DocRow,
    ErrorBucketRow,
    ExcludeRuleRow,
    FilenameSearchResultRow,
    FileRow,
    ParseBatchRow,
    ParseRow,
    ParsingRuleRow,
    RecentScanRow,
    Sha256Row,
    ShortIdRow,
    WatchParseCountRow,
    WatchStatsFileRow,
    WatchTargetRow,
)
from .services.parse_svc import (
    filter_pages_by_user_range,
    load_pages_from_done_batches,
    parse_image_sidecar_dir,
    parse_page_range_set,
    resolve_image_sidecar_path,
)
from .telemetry.buckets import pages_bucket, results_bucket
from .types import (
    FILE_STATUS_ACTIVE,
    FILE_STATUS_DELETED,
    FILE_STATUS_UNREACHABLE,
    PARSE_STATUS_DONE,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PARSING,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_SUPERSEDED,
    RULE_TYPE_EXCLUDE,
    RULE_TYPE_PARSING_RULE,
    CleanupDeletedRequest,
    CleanupDeletedResponse,
    CleanupOrphansRequest,
    CleanupOrphansResponse,
    CleanupTempRequest,
    CleanupTempResponse,
    ConfigResponse,
    ConfigSetRequest,
    ConfigSetResponse,
    ConfigUnsetResponse,
    ConfigValueResponse,
    ContentAsset,
    ContentFormat,
    ContentNextRequest,
    ContentRange,
    ContentRequestScope,
    DocContentExportRequest,
    DocContentExportResponse,
    DocContentResponse,
    DocInfo,
    ErrorBucket,
    ErrorSummary,
    ExcludeRuleInfo,
    ExcludeRuleListResponse,
    ExcludeRuleRequest,
    FileInfo,
    FileInfoResponse,
    FileStatus,
    FindResponse,
    FindResult,
    ForgetPathRequest,
    ForgetPathResponse,
    ImageFormat,
    InvalidateRequest,
    InvalidateResponse,
    ListDocsResponse,
    ListFilesResponse,
    ListParsesResponse,
    LocalParseServerStatus,
    ParseCoverage,
    ParseInfo,
    ParseRequest,
    ParseResponse,
    ParseServerStatus,
    ParseStatus,
    ParsingRuleInfo,
    ParsingRuleListResponse,
    ParsingRuleRequest,
    RemoteParseServerStatus,
    RemoveExcludeRuleResponse,
    RemoveParsingRuleResponse,
    RemoveWatchResponse,
    ScanInfo,
    ScanKind,
    ScanListResponse,
    ScanRequest,
    ScanStatus,
    SearchResponse,
    SearchResult,
    ServerStatusResponse,
    ShutdownResponse,
    TCPServerStatus,
    TelemetryAction,
    TelemetryActionResponse,
    TelemetryObservation,
    TelemetryObservationsRequest,
    TelemetryObservationsResponse,
    TelemetryPayload,
    TelemetryPreviewResponse,
    TelemetryStatusResponse,
    TierParseInfo,
    WatchInfo,
    WatchListResponse,
    WatchRequest,
    WatchStats,
    WorkerStatus,
)
from .utils.route_utils import get_route_info, has_route_info, route


@dataclass(frozen=True)
class _RenderedContent:
    content: str
    content_ranges: list[ContentRange]
    truncated: bool
    last_page_no: int
    next_page_no: int | None = None
    cut_inside_page: bool = False


@dataclass(frozen=True)
class _ReadPlan:
    sha256: str
    short_id: str
    tier: Tier
    page_range: str | None
    after: str | None
    locator: str | None
    context: int
    limit: int
    format: ContentFormat
    no_marker: bool
    image_format: ImageFormat = "jpeg"
    target: ContentCursor | None = None
    next_mode: str = "parse"


_PUBLIC_IMAGE_BUCKET_PATH = "images"
_NO_MATCHING_DOC_SHA256 = "__mineru_no_matching_doc__"


@dataclass(frozen=True)
class _LocatorParts:
    short_id: str
    tier: Tier | None = None
    page_no: int | None = None
    block_no: int | None = None
    char_offset: int | None = None


class DoclibServer(AsyncDoclibInterface):
    """Async doclib server whose routes are declared on interface methods."""

    def __init__(self, state: Any, *, title: str = "MinerU Doclib API") -> None:
        self.state = state
        self.app = FastAPI(
            title=title,
            docs_url="/api/docs",
            redoc_url="/api/redoc",
            openapi_url="/api/openapi.json",
        )
        self.app.add_exception_handler(MineruError, _mineru_error_handler)
        self.app.add_exception_handler(Exception, _unexpected_error_handler)
        self.app.include_router(self._build_router())

    def _build_router(self) -> APIRouter:
        endpoints = []
        for attr_name in dir(self):
            endpoint = getattr(self, attr_name)
            if callable(endpoint) and has_route_info(endpoint):
                route_info = get_route_info(endpoint)
                endpoints.append((route_info.index, route_info, endpoint))

        router = APIRouter(prefix="/api/v1")
        for _, route_info, endpoint in sorted(endpoints):
            router.add_api_route(
                path=route_info.path,
                endpoint=endpoint,
                methods=[route_info.method],
                tags=list(route_info.tags),
            )
        return router

    @route("GET", "/server/status", tags=("server",))
    async def get_server_status(self) -> ServerStatusResponse:
        files_total = await self.state.db.fetchone("SELECT COUNT(*) as cnt FROM files WHERE status=?", (FILE_STATUS_ACTIVE,))
        docs_total = await self.state.db.fetchone("SELECT COUNT(*) as cnt FROM docs")
        active_scans = await self.state.db.fetchone(
            "SELECT COUNT(*) as cnt FROM scans WHERE status IN (?, ?)",
            ("pending", "running"),
        )
        last_scan = await self.state.db.fetchone(
            "SELECT MAX(COALESCE(finished_at, updated_at)) as ts FROM scans WHERE status IN (?, ?)",
            ("done", "failed"),
        )
        ingest_q = await self.state.db.fetchone(
            "SELECT COUNT(*) as cnt FROM files WHERE sha256 IS NULL AND status=? AND error_code IS NULL",
            (FILE_STATUS_ACTIVE,),
        )
        local_mode = (await self.state.config_svc.get("parse_server.local.mode")) or "disabled"
        managed_tier = (await self.state.config_svc.get("parse_server.local.managed_tier")) or "standard"
        self_hosted_url = await self.state.config_svc.get("parse_server.local.self_hosted_url")
        remote_url = await self.state.config_svc.get("parse_server.remote.url")
        watches = await self.state.config_svc.list_watches()
        uptime = time.time() - self.state.start_time if hasattr(self.state, "start_time") else 0
        sqlite_path = getattr(self.state, "sqlite_path", "")
        log_path = getattr(self.state, "log_path", "")
        access_log_path = getattr(self.state, "access_log_path", "")
        stdout_log_path = getattr(self.state, "stdout_log_path", "")
        stderr_log_path = getattr(self.state, "stderr_log_path", "")
        sqlite_journal_mode = await _sqlite_journal_mode(self.state.db)
        health = get_health()

        return ServerStatusResponse(
            running=True,
            pid=getattr(self.state, "pid", os.getpid()),
            uptime_seconds=uptime,
            mineru_home=getattr(self.state, "mineru_home", ""),
            version=__version__,
            python_version=sys.version.split()[0],
            socket_path=getattr(self.state, "socket_path", ""),
            data_dir=getattr(self.state, "data_dir", ""),
            sqlite_path=sqlite_path,
            log_path=log_path,
            access_log_path=access_log_path,
            stdout_log_path=stdout_log_path,
            stderr_log_path=stderr_log_path,
            tcp=TCPServerStatus(
                enabled=bool(getattr(self.state, "tcp_enabled", False)),
                host=getattr(self.state, "tcp_host", "") or None,
                port=getattr(self.state, "tcp_port", None),
            ),
            active_scan_count=active_scans["cnt"] if active_scans else 0,
            last_scan_at=last_scan["ts"] if last_scan else None,
            sqlite_journal_mode=sqlite_journal_mode,
            sqlite_size_bytes=_file_size(sqlite_path),
            sqlite_wal_size_bytes=_file_size(f"{sqlite_path}-wal") if sqlite_path else None,
            workers=WorkerStatus(
                watch_running=bool(getattr(self.state.watch, "running", False)),
                scan_running=bool(getattr(self.state.scan_workers, "running", False)),
                scan_workers=int(getattr(self.state.scan_workers, "num_workers", 0) or 0),
                ingest_running=bool(getattr(self.state.ingest_workers, "running", False)),
                ingest_workers=int(getattr(self.state.ingest_workers, "num_workers", 0) or 0),
                parse_running=bool(getattr(self.state.parse_workers, "running", False)),
                parse_workers=int(getattr(self.state.parse_workers, "num_workers", 0) or 0),
                device_monitor_running=bool(getattr(self.state.device_monitor, "running", False)),
                compaction_running=bool(getattr(self.state.compaction, "running", False)),
                health_check_running=bool(getattr(self.state.health_check, "running", False)),
            ),
            files_total=files_total["cnt"] if files_total else 0,
            docs_total=docs_total["cnt"] if docs_total else 0,
            parse_queue_length=await self.state.parse_svc.get_queue_length(),
            ingest_queue_length=ingest_q["cnt"] if ingest_q else 0,
            parse_server=_parse_server_status(
                local_mode=local_mode,
                managed_tier=cast(Tier, managed_tier),
                self_hosted_url=self_hosted_url if self_hosted_url else None,
                remote_url=remote_url if remote_url else None,
                health=health,
            ),
            watch_count=len(watches),
            watches=[_watch_info(row) for row in watches],
            watch_stats=await _watch_stats(self.state.db, watches),
            recent_scans=await _recent_scans(self.state.db),
            error_summary=await _error_summary(self.state.db),
            recent_logs=_tail_log(log_path, lines=25),
            app_logs=_tail_log(log_path, lines=25),
            access_logs=_tail_log(access_log_path, lines=10),
            stdout_logs=_tail_log(stdout_log_path, lines=10),
            stderr_logs=_tail_log(stderr_log_path, lines=10),
        )

    @route("POST", "/server/shutdown", tags=("server",))
    async def shutdown_server(self) -> ShutdownResponse:
        if hasattr(self.state, "shutdown"):
            result = self.state.shutdown()
            if hasattr(result, "__await__"):
                await result
        else:
            asyncio.create_task(_signal_shutdown())
        return ShutdownResponse(accepted=True, message="Server shutting down...")

    @route("POST", "/parses", tags=("parse",))
    async def ensure_parse(self, request: ParseRequest) -> ParseResponse:
        start_ms = _now_ms()
        await _record_telemetry_count(self.state, "parse.request.count")
        try:
            result = await self.state.parse_svc.request_parse(
                request.path,
                tier=request.tier,
                page_range=request.page_range,
                force=request.force,
                remote=request.remote,
            )
            response = ParseResponse.model_validate(result)
            status = _parse_route_status(response)
            dims = {"status": status, "tier": response.tier}
            await _record_telemetry_count(self.state, "parse.finished.count", dimensions=dims)
            await _record_telemetry_duration(self.state, "parse.duration_bucket.count", start_ms, dimensions=dims)
            return response
        except Exception as exc:
            dims = {"status": "failed", "tier": request.tier or "default", "error_code": _telemetry_error_code(exc)}
            await _record_telemetry_count(self.state, "parse.finished.count", dimensions=dims)
            await _record_telemetry_duration(self.state, "parse.duration_bucket.count", start_ms, dimensions=dims)
            raise

    @route("GET", "/parses", tags=("parse",))
    async def list_parses(
        self,
        *,
        ids: list[int] | None = None,
        doc_ref: str | None = None,
        tier: Tier | None = None,
        status: ParseStatus | None = None,
        page_range: str | None = None,
        include_superseded: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> ListParsesResponse:
        resolved_sha256 = await self._resolve_doc_filter_sha256(doc_ref)
        rows, total, limit, offset = await self._select_parse_rows(
            ids=ids,
            sha256=resolved_sha256,
            tier=tier,
            status=status,
            include_superseded=include_superseded,
            limit=limit,
            offset=offset,
        )
        coverage = _parse_coverage(page_range, rows) if resolved_sha256 and tier and page_range else None
        return ListParsesResponse(
            parses=[_parse_info(row, coverage=coverage) for row in rows],
            coverage=coverage,
            total=total,
            limit=limit,
            offset=offset,
        )

    @route("GET", "/parses/{parse_id}", tags=("parse",))
    async def get_parse(self, parse_id: int) -> ParseInfo:
        row = await self.state.db.fetchone(
            "SELECT p.*, d.short_id AS short_id FROM parses p JOIN docs d ON d.sha256=p.sha256 WHERE p.id=?",
            (parse_id,),
        )
        if row is None:
            raise NotFoundError("job_not_found", f"Parse {parse_id} not found.", "parse_id")
        return _parse_info(row)

    @route("POST", "/invalidate", tags=("parse",))
    async def invalidate(self, request: InvalidateRequest) -> InvalidateResponse:
        if request.target != "parses":
            raise InvalidRequestError("invalid_request", f"Unsupported invalidate target: {request.target}", "target")
        doc_row = await self._doc_for_ref(request.doc_ref, param="doc_ref") if request.doc_ref else None
        sha256 = doc_row["sha256"] if doc_row else None
        if sha256 is None and request.path:
            file_row = await self.state.parse_svc.ensure_ingested(request.path)
            sha256 = file_row["sha256"] if file_row and file_row.get("sha256") else None
        if sha256 is None:
            raise NotFoundError("file_not_found", "Document not found.", "path")

        short_id = doc_row["short_id"] if doc_row else await self._short_id_for_sha256(sha256)
        count = await self.state.parse_svc.invalidate(sha256, request.tier)
        return InvalidateResponse(
            target=request.target, sha256=sha256, short_id=short_id, tier=request.tier, invalidated_count=count
        )

    @route("POST", "/forget", tags=("files",))
    async def forget_path(self, request: ForgetPathRequest) -> ForgetPathResponse:
        result = await self.state.cleanup_svc.forget_path(request.path, dry_run=request.dry_run)
        return ForgetPathResponse.model_validate(result)

    @route("POST", "/scans", tags=("scan",))
    async def create_scan(self, request: ScanRequest) -> ScanInfo:
        await _record_telemetry_count(self.state, "scan.request.count")
        return await self.state.scan_svc.create_scan(
            request.path,
            kind=request.kind,
            source=request.source,
            watch_id=request.watch_id,
        )

    @route("GET", "/scans", tags=("scan",))
    async def list_scans(
        self,
        *,
        limit: int = 50,
        status: ScanStatus | None = None,
        kind: ScanKind | None = None,
        watch_id: int | None = None,
        offset: int = 0,
    ) -> ScanListResponse:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        total = await self.state.scan_svc.count_scans(status=status, kind=kind, watch_id=watch_id)
        scans = await self.state.scan_svc.list_scans(
            limit=limit,
            status=status,
            kind=kind,
            watch_id=watch_id,
            offset=offset,
        )
        return ScanListResponse(scans=scans, total=total, limit=limit, offset=offset)

    @route("GET", "/scans/{scan_id}", tags=("scan",))
    async def get_scan(self, scan_id: int) -> ScanInfo:
        return await self.state.scan_svc.get_scan(scan_id)

    @route("GET", "/files", tags=("files",))
    async def list_files(
        self,
        *,
        status: FileStatus | None = None,
        ext: str | None = None,
        watch_id: int | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> ListFilesResponse:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        file_status = status or FILE_STATUS_ACTIVE
        clauses = ["f.status=?"]
        params: list[Any] = [file_status]
        if ext:
            clauses.append("f.ext=?")
            params.append(ext.lstrip(".").lower())
        if watch_id is not None:
            clauses.append("f.watch_id=?")
            params.append(watch_id)
        where = " AND ".join(clauses)
        total_row = await self.state.db.fetchone(f"SELECT COUNT(*) AS cnt FROM files f WHERE {where}", tuple(params))
        rows = cast(
            list[FileRow],
            await self.state.db.fetchall(
                f"""
                SELECT f.*, d.short_id AS short_id
                FROM files f
                LEFT JOIN docs d ON d.sha256=f.sha256
                WHERE {where}
                ORDER BY f.updated_at DESC LIMIT ? OFFSET ?
                """,
                (*params, limit, offset),
            ),
        )
        total = total_row["cnt"] if total_row else 0
        return ListFilesResponse(files=[_file_info(row) for row in rows], total=total, limit=limit, offset=offset)

    @route("GET", "/docs", tags=("docs",))
    async def list_docs(
        self,
        *,
        file_type: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> ListDocsResponse:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        clauses: list[str] = []
        params: list[Any] = []
        if file_type:
            clauses.append("d.file_type=?")
            params.append(file_type.lstrip(".").lower())
        clauses.append("EXISTS (SELECT 1 FROM files f WHERE f.sha256=d.sha256 AND f.status=?)")
        params.append(FILE_STATUS_ACTIVE)
        where = " AND ".join(clauses)
        total_row = await self.state.db.fetchone(f"SELECT COUNT(*) AS cnt FROM docs d WHERE {where}", tuple(params))
        rows = await self.state.db.fetchall(
            f"SELECT d.* FROM docs d WHERE {where} ORDER BY d.updated_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        total = total_row["cnt"] if total_row else 0
        return ListDocsResponse(docs=[_doc_info(row) for row in rows], total=total, limit=limit, offset=offset)

    @route("GET", "/docs/by-path", tags=("docs",))
    async def get_doc_by_path(self, path: str) -> DocInfo:
        await self.state.parse_svc.ensure_ingested(path)
        row = await self.state.db.fetchone(
            "SELECT d.* FROM docs d JOIN files f ON f.sha256=d.sha256 WHERE f.path=? AND f.status=?",
            (path, FILE_STATUS_ACTIVE),
        )
        if row is None:
            raise NotFoundError("doc_not_found", f"Document for file {path} not found.", "path")
        return _doc_info(row)

    @route("GET", "/docs/{doc_ref}", tags=("docs",))
    async def get_doc(self, doc_ref: str, *, expand_files: bool = False) -> DocInfo:
        row = await self._doc_for_ref(doc_ref, param="doc_ref")
        files = await self._files_for_sha256(row["sha256"]) if expand_files else None
        return _doc_info(row, files=files)

    @route("GET", "/docs/{doc_ref}/content", tags=("docs",))
    async def get_doc_content(
        self,
        doc_ref: str,
        *,
        tier: Tier,
        page_range: str | None = None,
        after: str | None = None,
        limit: int = 30000,
        format: str = "markdown",
        no_marker: bool = False,
    ) -> DocContentResponse:
        start_ms = _now_ms()
        dims = {"content_mode": "read", "output_format": _telemetry_output_format(format), "tier": tier}
        await _record_telemetry_count(self.state, "content.request.count", dimensions=dims)
        try:
            response = await self._execute_read_plan(
                await self._build_read_plan_from_parse(
                    doc_ref,
                    tier=tier,
                    page_range=page_range,
                    after=after,
                    limit=limit,
                    format=format,
                    no_marker=no_marker,
                )
            )
            finished_dims = dims | {"status": "succeeded"}
            await _record_telemetry_count(self.state, "content.finished.count", dimensions=finished_dims)
            await _record_telemetry_duration(self.state, "content.duration_bucket.count", start_ms, dimensions=finished_dims)
            return response
        except Exception:
            finished_dims = dims | {"status": "failed"}
            await _record_telemetry_count(self.state, "content.finished.count", dimensions=finished_dims)
            await _record_telemetry_duration(self.state, "content.duration_bucket.count", start_ms, dimensions=finished_dims)
            raise

    @route("GET", "/content", tags=("docs",))
    async def read_content(
        self,
        locator: str,
        *,
        context: int = 0,
        limit: int = 30000,
        format: ContentFormat = "markdown",
        image_format: ImageFormat = "jpeg",
        no_marker: bool = False,
    ) -> DocContentResponse:
        start_ms = _now_ms()
        dims = {"content_mode": "read", "output_format": _telemetry_output_format(format), "tier": "unknown"}
        await _record_telemetry_count(self.state, "content.request.count", dimensions=dims)
        try:
            response = await self._execute_read_plan(
                await self._build_read_plan_from_locator(
                    locator,
                    context=context,
                    limit=limit,
                    format=format,
                    image_format=image_format,
                    no_marker=no_marker,
                )
            )
            finished_dims = dims | {"status": "succeeded", "tier": response.tier}
            await _record_telemetry_count(self.state, "content.finished.count", dimensions=finished_dims)
            await _record_telemetry_duration(self.state, "content.duration_bucket.count", start_ms, dimensions=finished_dims)
            return response
        except Exception:
            finished_dims = dims | {"status": "failed"}
            await _record_telemetry_count(self.state, "content.finished.count", dimensions=finished_dims)
            await _record_telemetry_duration(self.state, "content.duration_bucket.count", start_ms, dimensions=finished_dims)
            raise

    async def _build_read_plan_from_parse(
        self,
        doc_ref: str,
        *,
        tier: Tier,
        page_range: str | None,
        after: str | None,
        limit: int,
        format: str,
        no_marker: bool,
    ) -> _ReadPlan:
        if format != "markdown":
            raise InvalidRequestError("invalid_request", "Only markdown is currently implemented for parse content.", "format")
        limit = max(1, limit)
        doc = await self._doc_for_ref(doc_ref, param="doc_ref")

        normalized_page_range = _normalize_content_page_range(page_range, after, doc)
        after_cursor = _parse_after_cursor(after)
        if after_cursor:
            _validate_cursor_for_doc(after_cursor, doc, tier, normalized_page_range)
        return _ReadPlan(
            sha256=doc["sha256"],
            tier=tier,
            short_id=doc["short_id"],
            page_range=normalized_page_range,
            after=after,
            limit=limit,
            format="markdown",
            no_marker=no_marker,
            locator=None,
            context=0,
            target=None,
            next_mode="parse",
        )

    async def _build_read_plan_from_locator(
        self,
        locator: str,
        *,
        context: int,
        limit: int,
        format: ContentFormat,
        image_format: ImageFormat,
        no_marker: bool,
    ) -> _ReadPlan:
        limit = max(1, limit)
        context = max(0, context)
        cursor = _parse_doc_locator(locator)
        doc = cast(DocRow | None, await self.state.db.fetchone("SELECT * FROM docs WHERE short_id=?", (cursor.short_id,)))
        if doc is None:
            raise NotFoundError("doc_not_found", f"Document {cursor.short_id} not found.", "locator")

        tier = cursor.tier or await self._default_read_tier(doc["sha256"])
        if tier is None:
            raise NotFoundError("tier_not_cached", f"No parsed tier is cached for document {cursor.short_id}.", "locator")
        if format == "image" and context:
            raise InvalidRequestError("context_not_applicable", "context is not supported for image reads.", "context")
        if cursor.page_no is None and context:
            raise InvalidRequestError("context_not_applicable", "context requires a page, block, or char locator.", "context")

        page_range = _locator_page_range(cursor, doc, context)
        return _ReadPlan(
            sha256=doc["sha256"],
            short_id=doc["short_id"],
            tier=tier,
            page_range=page_range,
            after=_locator_after(cursor),
            locator=_canonical_locator(doc["short_id"], tier, cursor),
            context=context,
            limit=limit,
            format=format,
            image_format=image_format,
            no_marker=no_marker,
            target=ContentCursor(
                short_id=doc["short_id"],
                tier=tier,
                page_no=cursor.page_no or 1,
                block_no=cursor.block_no,
                char_offset=cursor.char_offset,
            )
            if cursor.page_no is not None
            else None,
            next_mode="read",
        )

    @route("POST", "/docs/{doc_ref}/exports", tags=("docs",))
    async def export_doc_content(self, doc_ref: str, request: DocContentExportRequest) -> DocContentExportResponse:
        start_ms = _now_ms()
        dims = {"content_mode": "export", "output_format": _telemetry_output_format(request.format), "tier": request.tier}
        await _record_telemetry_count(self.state, "content.request.count", dimensions=dims)
        try:
            doc = await self._doc_for_ref(doc_ref, param="doc_ref")
            content = await self._render_doc_content(
                doc["sha256"],
                tier=request.tier,
                page_range=request.page_range,
                format=request.format,
                no_marker=request.no_marker,
            )
            output_path = os.path.abspath(request.output)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            finished_dims = dims | {"status": "succeeded"}
            await _record_telemetry_count(self.state, "content.finished.count", dimensions=finished_dims)
            await _record_telemetry_duration(self.state, "content.duration_bucket.count", start_ms, dimensions=finished_dims)
            return DocContentExportResponse(
                sha256=doc["sha256"], short_id=doc["short_id"], tier=request.tier, output=output_path
            )
        except Exception:
            finished_dims = dims | {"status": "failed"}
            await _record_telemetry_count(self.state, "content.finished.count", dimensions=finished_dims)
            await _record_telemetry_duration(self.state, "content.duration_bucket.count", start_ms, dimensions=finished_dims)
            raise

    @route("GET", "/search", tags=("search",))
    async def search(
        self,
        query: str,
        *,
        file_type: str | None = None,
        tier: Tier | None = None,
        min_tier: Tier | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> SearchResponse:
        start_ms = _now_ms()
        await _record_telemetry_count(self.state, "search.request.count")
        try:
            results, total = await self.state.search_svc.search(
                query=query,
                file_type=file_type,
                tier=tier,
                min_tier=min_tier,
                limit=limit,
                offset=offset,
            )
            response = SearchResponse(
                results=[_search_result(row) for row in results if row.get("tier") in TIERS], total=total, query=query
            )
            dims = {"status": "succeeded"}
            await _record_telemetry_count(self.state, "search.finished.count", dimensions=dims)
            await _record_telemetry_duration(self.state, "search.duration_bucket.count", start_ms, dimensions=dims)
            await _record_telemetry_count(
                self.state, "search.results_bucket.count", dimensions={"bucket": results_bucket(total)}
            )
            return response
        except Exception:
            dims = {"status": "failed"}
            await _record_telemetry_count(self.state, "search.finished.count", dimensions=dims)
            await _record_telemetry_duration(self.state, "search.duration_bucket.count", start_ms, dimensions=dims)
            raise

    @route("GET", "/find", tags=("search",))
    async def find(self, query: str, *, ext: str | None = None, limit: int = 50) -> FindResponse:
        start_ms = _now_ms()
        await _record_telemetry_count(self.state, "find.request.count")
        try:
            results, total = await self.state.search_svc.search_filenames(
                query=query,
                ext=ext,
                limit=limit,
                refresh_file=self.state.parse_svc.refresh_file,
            )
            response = FindResponse(results=[_find_result(row) for row in results], total=total, query=query)
            dims = {"status": "succeeded"}
            await _record_telemetry_count(self.state, "find.finished.count", dimensions=dims)
            await _record_telemetry_duration(self.state, "find.duration_bucket.count", start_ms, dimensions=dims)
            await _record_telemetry_count(self.state, "find.results_bucket.count", dimensions={"bucket": results_bucket(total)})
            return response
        except Exception:
            dims = {"status": "failed"}
            await _record_telemetry_count(self.state, "find.finished.count", dimensions=dims)
            await _record_telemetry_duration(self.state, "find.duration_bucket.count", start_ms, dimensions=dims)
            raise

    @route("GET", "/files/by-path", tags=("files",))
    async def get_file_by_path(self, path: str) -> FileInfoResponse:
        await self.state.parse_svc.ensure_ingested(path)
        file_row = await self.state.db.fetchone(
            """
            SELECT f.*, d.short_id AS short_id
            FROM files f
            LEFT JOIN docs d ON d.sha256=f.sha256
            WHERE f.path=? AND f.status=?
            """,
            (path, FILE_STATUS_ACTIVE),
        )
        if file_row is None:
            raise NotFoundError("file_not_found", f"File record {path} not found in doclib.", "path")
        sha256 = file_row["sha256"]
        doc_row = await self.state.db.fetchone("SELECT * FROM docs WHERE sha256=?", (sha256,)) if sha256 else None
        parse_rows = (
            await self.state.db.fetchall(
                """
                SELECT p.*, d.short_id AS short_id
                FROM parses p
                JOIN docs d ON d.sha256=p.sha256
                WHERE p.sha256=?
                ORDER BY p.tier, p.created_at DESC
                """,
                (sha256,),
            )
            if sha256
            else []
        )
        parsed_tiers = [_tier_parse_info(row) for row in parse_rows if row["status"] == PARSE_STATUS_DONE]
        active_parses = [
            _parse_info(row) for row in parse_rows if row["status"] in {PARSE_STATUS_PENDING, PARSE_STATUS_PARSING}
        ]
        return FileInfoResponse(
            file=_file_info(file_row),
            doc=_doc_info(doc_row) if doc_row else None,
            parsed_tiers=parsed_tiers,
            active_parses=active_parses,
        )

    @route("GET", "/configs", tags=("config",))
    async def get_config(self) -> ConfigResponse:
        config, sources = await self.state.config_svc.get_all_with_sources()
        return ConfigResponse(config=_mask_config(config), sources=sources)

    @route("GET", "/configs/{key}", tags=("config",))
    async def get_config_key(self, key: str) -> ConfigValueResponse:
        value = await self.state.config_svc.get(key)
        source = await self.state.config_svc.get_source(key)
        return ConfigValueResponse(key=key, value=_mask_config_value(key, value or ""), source=source)

    @route("PUT", "/configs/{key}", tags=("config",))
    async def set_config(self, key: str, request: ConfigSetRequest) -> ConfigSetResponse:
        await self.state.config_svc.set(key, request.value)
        value = await self.state.config_svc.get(key)
        source = await self.state.config_svc.get_source(key)
        return ConfigSetResponse(key=key, value=_mask_config_value(key, value or ""), source=source)

    @route("DELETE", "/configs/{key}", tags=("config",))
    async def unset_config(self, key: str) -> ConfigUnsetResponse:
        removed = await self.state.config_svc.unset(key)
        value = await self.state.config_svc.get(key)
        source = await self.state.config_svc.get_source(key)
        return ConfigUnsetResponse(key=key, value=_mask_config_value(key, value or ""), source=source, removed=removed)

    @route("POST", "/watches", tags=("watches",))
    async def add_watch(self, request: WatchRequest) -> WatchInfo:
        await _record_telemetry_count(self.state, "watch.add.count")
        try:
            row = await self.state.config_svc.add_watch(request.path, removable=request.removable, label=request.label)
            watch_loop = getattr(self.state, "watch", None)
            if watch_loop is not None:
                watch_loop.wakeup()
            await _record_telemetry_count(self.state, "watch.add.finished.count", dimensions={"status": "succeeded"})
            return _watch_info(row)
        except Exception:
            await _record_telemetry_count(self.state, "watch.add.finished.count", dimensions={"status": "failed"})
            raise

    @route("GET", "/watches", tags=("watches",))
    async def list_watches(self) -> WatchListResponse:
        return WatchListResponse(watches=[_watch_info(row) for row in await self.state.config_svc.list_watches()])

    @route("DELETE", "/watches/{watch_id}", tags=("watches",))
    async def remove_watch(self, watch_id: int) -> RemoveWatchResponse:
        await _record_telemetry_count(self.state, "watch.remove.count")
        try:
            existing = await self.state.db.fetchone("SELECT id FROM watches WHERE id=?", (watch_id,))
            if existing is None:
                raise NotFoundError("watch_not_found", f"Watch target {watch_id} not found.", "watch_id")
            await self.state.config_svc.remove_watch_by_id(watch_id)
            await _record_telemetry_count(self.state, "watch.remove.finished.count", dimensions={"status": "succeeded"})
            return RemoveWatchResponse(watch_id=watch_id, removed=True)
        except Exception:
            await _record_telemetry_count(self.state, "watch.remove.finished.count", dimensions={"status": "failed"})
            raise

    @route("POST", "/exclude-rules", tags=("rules",))
    async def add_exclude_rule(self, request: ExcludeRuleRequest) -> ExcludeRuleInfo:
        rule_id = await self.state.config_svc.add_rule(
            request.name or "",
            RULE_TYPE_EXCLUDE,
            request.pattern,
            priority=request.priority,
        )
        row = await self.state.db.fetchone("SELECT * FROM exclude_rules WHERE id=?", (rule_id,))
        return _exclude_rule_info(row)

    @route("GET", "/exclude-rules", tags=("rules",))
    async def list_exclude_rules(self) -> ExcludeRuleListResponse:
        rows = await self.state.config_svc.list_rules(RULE_TYPE_EXCLUDE)
        return ExcludeRuleListResponse(rules=[_exclude_rule_info(row) for row in rows])

    @route("DELETE", "/exclude-rules/{rule_id}", tags=("rules",))
    async def remove_exclude_rule(self, rule_id: int) -> RemoveExcludeRuleResponse:
        existing = await self.state.db.fetchone("SELECT id FROM exclude_rules WHERE id=?", (rule_id,))
        if existing is None:
            raise NotFoundError("rule_not_found", f"Exclude rule {rule_id} not found.", "rule_id")
        await self.state.config_svc.remove_rule(rule_id, RULE_TYPE_EXCLUDE)
        return RemoveExcludeRuleResponse(rule_id=rule_id, removed=True)

    @route("POST", "/parsing-rules", tags=("rules",))
    async def add_parsing_rule(self, request: ParsingRuleRequest) -> ParsingRuleInfo:
        rule_id = await self.state.config_svc.add_rule(
            request.name or "",
            RULE_TYPE_PARSING_RULE,
            request.pattern,
            tier=request.tier,
            page_range=request.page_range,
            remote=request.remote,
            priority=request.priority,
        )
        row = await self.state.db.fetchone("SELECT * FROM parsing_rules WHERE id=?", (rule_id,))
        return _parsing_rule_info(row)

    @route("GET", "/parsing-rules", tags=("rules",))
    async def list_parsing_rules(self) -> ParsingRuleListResponse:
        rows = await self.state.config_svc.list_rules(RULE_TYPE_PARSING_RULE)
        return ParsingRuleListResponse(rules=[_parsing_rule_info(row) for row in rows])

    @route("DELETE", "/parsing-rules/{rule_id}", tags=("rules",))
    async def remove_parsing_rule(self, rule_id: int) -> RemoveParsingRuleResponse:
        existing = await self.state.db.fetchone("SELECT id FROM parsing_rules WHERE id=?", (rule_id,))
        if existing is None:
            raise NotFoundError("rule_not_found", f"Parsing rule {rule_id} not found.", "rule_id")
        await self.state.config_svc.remove_rule(rule_id, RULE_TYPE_PARSING_RULE)
        return RemoveParsingRuleResponse(rule_id=rule_id, removed=True)

    @route("POST", "/cleanup/deleted-files", tags=("cleanup",))
    async def cleanup_deleted_files(self, request: CleanupDeletedRequest) -> CleanupDeletedResponse:
        count = await self.state.cleanup_svc.cleanup_deleted(dry_run=request.dry_run)
        return CleanupDeletedResponse(deleted_files=count, dry_run=request.dry_run)

    @route("POST", "/cleanup/orphan-docs", tags=("cleanup",))
    async def cleanup_orphan_docs(self, request: CleanupOrphansRequest) -> CleanupOrphansResponse:
        count = await self.state.cleanup_svc.cleanup_orphans(dry_run=request.dry_run)
        return CleanupOrphansResponse(orphan_docs=count, dry_run=request.dry_run)

    @route("POST", "/cleanup/temp", tags=("cleanup",))
    async def cleanup_temp_files(self, request: CleanupTempRequest) -> CleanupTempResponse:
        count = await self.state.cleanup_svc.cleanup_temp_files(older_than_days=request.older_than_days)
        return CleanupTempResponse(temp_files_removed=count)

    @route("GET", "/telemetry/status", tags=("telemetry",))
    async def get_telemetry_status(self) -> TelemetryStatusResponse:
        status = await self.state.telemetry_svc.status()
        return TelemetryStatusResponse.model_validate(status)

    @route("GET", "/telemetry/preview", tags=("telemetry",))
    async def get_telemetry_preview(self) -> TelemetryPreviewResponse:
        body = await self.state.telemetry_svc.preview_body()
        return TelemetryPreviewResponse(body=TelemetryPayload.model_validate(body))

    @route("POST", "/telemetry/actions/{action}", tags=("telemetry",))
    async def telemetry_action(self, action: TelemetryAction) -> TelemetryActionResponse:
        if action == "enable":
            result = await self.state.telemetry_svc.set_consent("enabled")
            return TelemetryActionResponse(action=action, **result)
        if action == "disable":
            result = await self.state.telemetry_svc.set_consent("disabled")
            return TelemetryActionResponse(action=action, **result)
        status = await self.state.telemetry_svc.status()
        if status["state"] != "enabled":
            return TelemetryActionResponse(
                action=action,
                state=status["state"],
                installation_id=status["installation_id"],
                executed=False,
                reason="telemetry_not_enabled",
            )
        flush_result = await self.state.telemetry_svc.flush_once()
        return TelemetryActionResponse(
            action=action,
            state=status["state"],
            installation_id=status["installation_id"],
            executed=flush_result.status not in {"disabled", "locked", "no_metrics", "failed"},
            reason=None if flush_result.status == "success" else flush_result.status,
            flush_result={
                "status": flush_result.status,
                "attempted": flush_result.attempted,
                "succeeded": flush_result.succeeded,
                "discarded": flush_result.discarded,
            },
        )

    @route("POST", "/observations", tags=("telemetry",))
    async def record_observations(self, request: TelemetryObservationsRequest) -> TelemetryObservationsResponse:
        accepted = 0
        for observation in request.observations:
            if observation.metric_name == "parse.wait":
                await self._record_parse_wait_observation(observation)
                accepted += 1
                continue
            if observation.duration_ms is None:
                await self.state.telemetry_svc.record_count(
                    observation.metric_name,
                    value=observation.value,
                    dimensions=observation.dimensions,
                )
            else:
                await self.state.telemetry_svc.record_duration_bucket(
                    observation.metric_name,
                    duration_ms=observation.duration_ms,
                    dimensions=observation.dimensions,
                )
            accepted += 1
        return TelemetryObservationsResponse(accepted=accepted)

    async def _record_parse_wait_observation(self, observation: TelemetryObservation) -> None:
        if observation.duration_ms is None or not observation.parse_ids:
            return
        placeholders = ",".join("?" * len(observation.parse_ids))
        rows = cast(
            list[ParseRow],
            await self.state.db.fetchall(
                f"SELECT * FROM parses WHERE id IN ({placeholders}) ORDER BY id ASC",
                tuple(observation.parse_ids),
            ),
        )
        if not rows:
            return
        tier = rows[0]["tier"] if rows[0].get("tier") else "unknown"
        page_numbers: set[int] = set()
        for row in rows:
            try:
                page_numbers |= parse_page_range_set(row["page_range"])
            except Exception:
                continue
        status = observation.dimensions.get("status") or "unknown"
        dims = {
            "status": status,
            "tier": tier,
            "pages_bucket": pages_bucket(len(page_numbers) or 1),
        }
        await self.state.telemetry_svc.record_count("parse.wait.count", dimensions=dims)
        await self.state.telemetry_svc.record_duration_bucket(
            "parse.wait_duration_bucket.count",
            duration_ms=observation.duration_ms,
            dimensions=dims,
        )

    async def _select_parse_rows(
        self,
        *,
        ids: list[int] | None,
        sha256: str | None,
        tier: Tier | None,
        status: ParseStatus | None,
        include_superseded: bool,
        limit: int,
        offset: int,
    ) -> tuple[list[ParseRow], int, int, int]:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        clauses: list[str] = []
        params: list[object] = []
        if ids:
            placeholders = ",".join("?" * len(ids))
            rows = cast(
                list[ParseRow],
                await self.state.db.fetchall(
                    f"""
                    SELECT p.*, d.short_id AS short_id
                    FROM parses p
                    JOIN docs d ON d.sha256=p.sha256
                    WHERE p.id IN ({placeholders})
                    """,
                    tuple(ids),
                ),
            )
            by_id = {row["id"]: row for row in rows}
            ordered = [by_id[parse_id] for parse_id in ids if parse_id in by_id]
            return ordered[offset : offset + limit], len(ordered), limit, offset
        if sha256:
            clauses.append("p.sha256=?")
            params.append(sha256)
        if tier:
            clauses.append("p.tier=?")
            params.append(tier)
        if status:
            clauses.append("p.status=?")
            params.append(status)
        elif not include_superseded:
            clauses.append("p.status!=?")
            params.append(PARSE_STATUS_SUPERSEDED)

        sql = "SELECT p.*, d.short_id AS short_id FROM parses p JOIN docs d ON d.sha256=p.sha256"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        count_sql = "SELECT COUNT(*) AS cnt FROM parses p JOIN docs d ON d.sha256=p.sha256"
        if clauses:
            count_sql += " WHERE " + " AND ".join(clauses)
        total_row = await self.state.db.fetchone(count_sql, tuple(params))
        sql += " ORDER BY p.created_at DESC LIMIT ? OFFSET ?"
        rows = cast(list[ParseRow], await self.state.db.fetchall(sql, (*params, limit, offset)))
        total = total_row["cnt"] if total_row else 0
        return rows, total, limit, offset

    async def _render_doc_content(
        self,
        sha256: str,
        *,
        tier: Tier,
        page_range: str | None,
        format: str,
        no_marker: bool,
    ) -> str:
        if format != "markdown":
            raise InvalidRequestError("invalid_request", "Only markdown is currently implemented.", "format")
        data_dir = _effective_data_dir(self.state)
        rows = await self.state.db.fetchall(
            "SELECT page_range, done_at FROM parses WHERE sha256=? AND tier=? AND status=? ORDER BY done_at DESC",
            (sha256, tier, PARSE_STATUS_DONE),
        )
        loaded_pages = load_pages_from_done_batches(data_dir, sha256, tier, cast(list[ParseBatchRow], rows))
        if page_range:
            loaded_pages = filter_pages_by_user_range(loaded_pages, page_range)
        if not loaded_pages:
            raise NotFoundError("not_cached", "Requested parsed content is not cached.", "page_range")
        return render_markdown(
            loaded_pages,
            img_bucket_path=_PUBLIC_IMAGE_BUCKET_PATH,
            add_markers=not no_marker,
            prefer_markdown_table=True,
        )

    async def _execute_read_plan(self, plan: _ReadPlan) -> DocContentResponse:
        data_dir = _effective_data_dir(self.state)
        rows = await self.state.db.fetchall(
            "SELECT page_range, done_at FROM parses WHERE sha256=? AND tier=? AND status=? ORDER BY done_at DESC",
            (plan.sha256, plan.tier, PARSE_STATUS_DONE),
        )
        loaded_pages = load_pages_from_done_batches(data_dir, plan.sha256, plan.tier, cast(list[ParseBatchRow], rows))
        if plan.page_range:
            loaded_pages = filter_pages_by_user_range(loaded_pages, plan.page_range)
        if not loaded_pages:
            raise NotFoundError("not_cached", "Requested parsed content is not cached.", "page_range")

        if plan.format == "image":
            return await self._render_image_response(plan, loaded_pages)

        pages_for_render = _select_context_pages(loaded_pages, plan.target, plan.context)
        rendered = _render_progressive_markdown(
            pages_for_render,
            short_id=plan.short_id,
            tier=plan.tier,
            after=_parse_after_cursor(plan.after),
            limit=plan.limit,
            add_markers=not plan.no_marker,
            target=plan.target,
            context=plan.context,
            img_bucket_path=_PUBLIC_IMAGE_BUCKET_PATH,
        )
        if not rendered.content_ranges:
            raise NotFoundError("not_cached", "Requested parsed content is not cached after cursor.", "after")

        doc = cast(DocRow | None, await self.state.db.fetchone("SELECT * FROM docs WHERE sha256=?", (plan.sha256,)))
        page_count = doc["page_count"] if doc else None
        paginated = _is_paginated_doc(doc) if doc else True
        next_request = (
            _next_read_request(rendered, plan.short_id, plan.tier, page_count)
            if plan.next_mode == "read"
            else _next_content_request(
                rendered=rendered,
                request_page_range=plan.page_range,
                after=plan.after,
                page_count=page_count,
                paginated=paginated,
            )
        )
        return DocContentResponse(
            sha256=plan.sha256,
            short_id=plan.short_id,
            tier=plan.tier,
            format="markdown",
            content=rendered.content,
            request_scope=ContentRequestScope(
                page_range=plan.page_range,
                after=plan.after,
                limit=plan.limit,
                locator=plan.locator,
                context=plan.context,
            ),
            content_ranges=rendered.content_ranges,
            truncated=rendered.truncated,
            next_request=next_request,
        )

    async def _render_image_response(self, plan: _ReadPlan, loaded_pages: list[PageInfo]) -> DocContentResponse:
        if plan.target is None or plan.target.page_no is None:
            raise InvalidRequestError("format_not_supported", "image format requires a page or block locator.", "format")
        if plan.page_range and len(parse_page_range_set(plan.page_range)) != 1:
            raise InvalidRequestError("multi_page_image_not_supported", "image format supports only one page.", "locator")
        page = _find_page(loaded_pages, plan.target.page_no)
        if page is None:
            raise NotFoundError("page_not_cached", f"Page {plan.target.page_no} is not cached.", "locator")

        if _is_office_page(page):
            asset = await self._render_office_image_asset(plan, page)
        else:
            asset = await self._render_pdf_image_asset(plan, page)

        target_ref = plan.locator or page_ref(plan.short_id, plan.tier, plan.target.page_no)
        return DocContentResponse(
            sha256=plan.sha256,
            short_id=plan.short_id,
            tier=plan.tier,
            format="image",
            content="",
            request_scope=ContentRequestScope(
                page_range=plan.page_range,
                after=plan.after,
                limit=plan.limit,
                locator=plan.locator,
                context=plan.context,
                image_format=plan.image_format,
            ),
            content_ranges=[ContentRange(page_range=str(plan.target.page_no), start=target_ref, end=target_ref)],
            truncated=False,
            next_request=None,
            asset=asset,
        )

    async def _sha256_for_path(self, path: str) -> str | None:
        row = cast(
            Sha256Row | None,
            await self.state.db.fetchone("SELECT sha256 FROM files WHERE path=? AND status=?", (path, FILE_STATUS_ACTIVE)),
        )
        return row["sha256"] if row and row["sha256"] else None

    async def _doc_for_ref(self, doc_ref: str, *, param: str) -> DocRow:
        row = await self._maybe_doc_for_ref(doc_ref)
        if row is None:
            raise NotFoundError("doc_not_found", f"Document {doc_ref} not found.", param)
        return row

    async def _maybe_doc_for_ref(self, doc_ref: str) -> DocRow | None:
        row = cast(DocRow | None, await self.state.db.fetchone("SELECT * FROM docs WHERE sha256=?", (doc_ref,)))
        if row is not None:
            return row
        return cast(DocRow | None, await self.state.db.fetchone("SELECT * FROM docs WHERE short_id=?", (doc_ref,)))

    async def _resolve_doc_filter_sha256(self, doc_ref: str | None) -> str | None:
        if doc_ref is None:
            return None
        doc = await self._maybe_doc_for_ref(doc_ref)
        if doc is None:
            return _NO_MATCHING_DOC_SHA256
        return doc["sha256"]

    async def _short_id_for_sha256(self, sha256: str) -> str:
        row = cast(ShortIdRow | None, await self.state.db.fetchone("SELECT short_id FROM docs WHERE sha256=?", (sha256,)))
        if row is None or not row["short_id"]:
            raise NotFoundError("doc_not_found", f"Document {sha256} not found.", "sha256")
        return row["short_id"]

    async def _files_for_sha256(self, sha256: str) -> list[FileInfo]:
        rows = cast(
            list[FileRow],
            await self.state.db.fetchall(
                """
                SELECT f.*, d.short_id AS short_id
                FROM files f
                JOIN docs d ON d.sha256=f.sha256
                WHERE f.sha256=? AND f.status=?
                """,
                (sha256, FILE_STATUS_ACTIVE),
            ),
        )
        return [_file_info(row) for row in rows]

    async def _default_read_tier(self, sha256: str) -> Tier | None:
        rows = cast(
            list[ParseRow],
            await self.state.db.fetchall(
                "SELECT tier FROM parses WHERE sha256=? AND status=? GROUP BY tier",
                (sha256, PARSE_STATUS_DONE),
            ),
        )
        tiers: set[Tier] = {row["tier"] for row in rows if row["tier"] in TIERS and row["tier"] != "flash"}
        if not tiers:
            return None
        return max(tiers, key=lambda t: TIER_ORDER[t])

    async def _render_pdf_image_asset(self, plan: _ReadPlan, page: PageInfo) -> ContentAsset:
        if plan.target is None:
            raise InvalidRequestError("format_not_supported", "image format requires a page or block locator.", "format")
        file_row = cast(
            FileRow | None,
            await self.state.db.fetchone(
                "SELECT * FROM files WHERE sha256=? AND status=? LIMIT 1",
                (plan.sha256, FILE_STATUS_ACTIVE),
            ),
        )
        if file_row is None:
            raise NotFoundError("no_accessible_file", "No active source file found for this document.", "locator")
        source_path = file_row["path"]
        if not source_path.lower().endswith(".pdf"):
            raise InvalidRequestError(
                "format_not_supported", "image format for page/block is only supported for PDF sources.", "format"
            )
        pdf_bytes = Path(source_path).read_bytes()
        with PDFDocument(pdf_bytes) as doc:
            if plan.target.block_no is None:
                image = doc.render_page(plan.target.page_no - 1, scale=2).pil_image
                image_bytes = _pil_image_to_bytes(image, plan.image_format)
                width, height = image.size
            else:
                block = _find_block_by_no(page, plan.target.block_no)
                if block is None:
                    raise NotFoundError("block_not_found", f"Block {plan.target.block_no} not found.", "locator")
                if _is_empty_bbox(block.bbox):
                    raise InvalidRequestError("bbox_not_available", "Block bbox is not available for image output.", "locator")
                image_bytes = _transcode_image_bytes(
                    doc.crop_image(block.bbox, plan.target.page_no - 1, scale=2), plan.image_format
                )
                width, height = _image_size_from_bytes(image_bytes)
        return _write_temp_asset(
            self.state.data_dir,
            plan.short_id,
            _image_format_ext(plan.image_format),
            image_bytes,
            mime_type=_mime_type_for_image_format(plan.image_format),
            width=width,
            height=height,
        )

    async def _render_office_image_asset(self, plan: _ReadPlan, page: PageInfo) -> ContentAsset:
        if plan.target is None or plan.target.block_no is None:
            raise InvalidRequestError("format_not_supported", "Office image output requires an image block locator.", "format")
        block = _find_block_by_no(page, plan.target.block_no)
        if block is None:
            raise NotFoundError("block_not_found", f"Block {plan.target.block_no} not found.", "locator")
        image_span = _first_image_span(block)
        if image_span is None:
            raise InvalidRequestError(
                "format_not_supported", "Office image output is only supported for image blocks.", "format"
            )
        image_bytes, _, _ = _span_image_bytes(image_span)
        if image_bytes is None and image_span.image_path:
            image_dir = parse_image_sidecar_dir(self.state.data_dir, plan.sha256, plan.tier)
            candidate = resolve_image_sidecar_path(image_dir, image_span.image_path)
            if candidate is not None and candidate.is_file():
                image_bytes = candidate.read_bytes()
        if image_bytes is None:
            raise NotFoundError("asset_not_available", "Office image asset is not available.", "locator")
        image_bytes = _transcode_image_bytes(image_bytes, plan.image_format)
        width, height = _image_size_from_bytes(image_bytes)
        return _write_temp_asset(
            self.state.data_dir,
            plan.short_id,
            _image_format_ext(plan.image_format),
            image_bytes,
            mime_type=_mime_type_for_image_format(plan.image_format),
            width=width,
            height=height,
        )


_READ_LOCATOR_RE = re.compile(
    r"^doc:(?P<short_id>[0-9a-fA-F]+)"
    r"(?:/tier:(?P<tier>flash|standard|pro)"
    r"(?:/page:(?P<page_no>[1-9][0-9]*)"
    r"(?:/block:(?P<block_no>[1-9][0-9]*)(?:/char:(?P<char_offset>0|[1-9][0-9]*))?)?)?)?$"
)


def _parse_doc_locator(locator: str) -> _LocatorParts:
    match = _READ_LOCATOR_RE.match(locator)
    if match is None:
        raise InvalidRequestError("invalid_locator", f"Invalid doclib locator: {locator}", "locator")
    page_no = match.group("page_no")
    block_no = match.group("block_no")
    char_offset = match.group("char_offset")
    return _LocatorParts(
        short_id=match.group("short_id"),
        tier=cast(Tier | None, match.group("tier")),
        page_no=int(page_no) if page_no is not None else None,
        block_no=int(block_no) if block_no is not None else None,
        char_offset=int(char_offset) if char_offset is not None else None,
    )


def _canonical_locator(short_id: str, tier: Tier, locator: _LocatorParts) -> str:
    if locator.page_no is None:
        return f"doc:{short_id}/tier:{tier}"
    if locator.block_no is None:
        return page_ref(short_id, tier, locator.page_no)
    if locator.char_offset is None:
        return block_ref(short_id, tier, locator.page_no, locator.block_no)
    return block_char_ref(short_id, tier, locator.page_no, locator.block_no, locator.char_offset)


def _locator_after(locator: _LocatorParts) -> str | None:
    if locator.page_no is None:
        return None
    if locator.block_no is None:
        return None
    return (
        _canonical_locator(locator.short_id, locator.tier or "standard", locator) if locator.char_offset is not None else None
    )


def _locator_page_range(locator: _LocatorParts, doc: DocRow, context: int) -> str | None:
    page_count = doc.get("page_count") or 1
    if locator.page_no is None:
        return _normalize_content_page_range(None, None, doc)
    start = max(1, locator.page_no - context)
    end = min(page_count, locator.page_no + context)
    return _range_str(start, end)


def _parse_after_cursor(after: str | None) -> ContentCursor | None:
    if not after:
        return None
    try:
        return parse_content_cursor(after)
    except ValueError as exc:
        raise InvalidRequestError("invalid_locator", str(exc), "after") from None


def _validate_cursor_for_doc(cursor: ContentCursor, doc: DocRow, tier: Tier, page_range: str | None) -> None:
    if cursor.short_id != doc["short_id"]:
        raise InvalidRequestError("invalid_request", "after cursor does not belong to this document.", "after")
    if cursor.tier != tier:
        raise InvalidRequestError("invalid_request", "after cursor tier does not match request tier.", "after")
    if page_range and cursor.page_no not in parse_page_range_set(page_range):
        raise InvalidRequestError("invalid_request", "after cursor is outside requested page_range.", "after")


def _mask_config(config: dict[str, str]) -> dict[str, str]:
    return {key: _mask_config_value(key, value) for key, value in config.items()}


def _mask_config_value(key: str, value: str) -> str:
    if not _is_sensitive_config_key(key) or not value:
        return value
    if len(value) <= 12:
        return "******"
    return f"{value[:6]}******{value[-6:]}"


def _is_sensitive_config_key(key: str) -> bool:
    lowered = key.lower()
    return "api_key" in lowered or "token" in lowered or "secret" in lowered or "password" in lowered


async def _mineru_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    mineru_error = exc if isinstance(exc, MineruError) else MineruError("internal_error", str(exc))
    return JSONResponse(status_code=http_status_for(mineru_error.code), content=error_response(mineru_error))


async def _unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content=error_response(MineruError("internal_error", str(exc))))


async def _signal_shutdown() -> None:
    await asyncio.sleep(0.1)
    os.kill(os.getpid(), signal.SIGTERM)


def _parse_server_status(
    *,
    local_mode: str,
    managed_tier: Tier,
    self_hosted_url: str | None,
    remote_url: str | None,
    health: Any,
) -> ParseServerStatus:
    local_url = health.managed_url if local_mode == "managed" else self_hosted_url
    managed_proc = getattr(health, "managed_proc", None)
    return ParseServerStatus(
        local=LocalParseServerStatus(
            mode=local_mode,
            healthy=health.local_healthy,
            starting=health.local_starting,
            started_at=health.local_started_at or None,
            url=local_url,
            port=_port_from_url(local_url),
            managed_pid=managed_proc.pid if managed_proc else None,
            managed_running=bool(managed_proc and managed_proc.poll() is None),
            managed_tier=managed_tier,
            self_hosted_url=self_hosted_url,
            restart_count=getattr(health, "restart_count", 0),
            max_restart_attempts=3,
            last_probe_at=health.local_last_probe_at,
            last_success_at=health.local_last_success_at,
            last_failure_at=health.local_last_failure_at,
            supported_tiers=health.local_supported_tiers,
        ),
        remote=RemoteParseServerStatus(
            healthy=health.remote_healthy,
            url=remote_url,
            port=_port_from_url(remote_url),
            last_probe_at=health.remote_last_probe_at,
            last_success_at=health.remote_last_success_at,
            last_failure_at=health.remote_last_failure_at,
            supported_tiers=health.remote_supported_tiers,
        ),
    )


def _port_from_url(url: str | None) -> int | None:
    if not url:
        return None
    try:
        parsed = urlparse(url)
        if parsed.port is not None:
            return parsed.port
        if parsed.scheme == "http":
            return 80
        if parsed.scheme == "https":
            return 443
    except ValueError:
        return None
    return None


def _file_size(path: str) -> int | None:
    if not path:
        return None
    try:
        return os.path.getsize(path)
    except OSError:
        return None


async def _sqlite_journal_mode(db: DatabaseManager) -> str | None:
    row = await db.fetchone("PRAGMA journal_mode")
    if not row:
        return None
    return cast(str | None, row.get("journal_mode"))


def _is_paginated_doc(doc: DocRow) -> bool:
    page_count = doc.get("page_count")
    return page_count is not None and page_count > 1


def _normalize_content_page_range(page_range: str | None, after: str | None, doc: DocRow) -> str | None:
    page_count = doc.get("page_count") or 1
    if _is_paginated_doc(doc):
        if page_range and page_range.strip() == "all":
            return f"1~{page_count}"
        if page_range:
            return _normalize_page_range(page_range, page_count)
        if after:
            cursor = _parse_after_cursor(after)
            assert cursor is not None
            end = min(page_count, cursor.page_no + 9)
            return f"{cursor.page_no}~{end}" if cursor.page_no != end else str(cursor.page_no)
        end = min(page_count, 10)
        return f"1~{end}" if end > 1 else "1"
    if page_range and page_range.strip() == "all":
        return None
    return _normalize_page_range(page_range, page_count) if page_range else None


def _normalize_page_range(page_range: str, page_count: int) -> str:
    return _expand_page_range(page_range, page_count)


def _expand_page_range(page_range: str, page_count: int) -> str:
    available_page_numbers = set(range(1, page_count + 1))
    if not page_range or page_range.strip() == "all":
        page_numbers = available_page_numbers
    else:
        page_numbers: set[int] = set()
        for part in page_range.split(","):
            part = part.strip()
            if not part:
                continue
            if "~" in part:
                raw_start, raw_end = part.split("~", 1)
                start = int(raw_start.strip())
                end = int(raw_end.strip())
                if start < 0:
                    start = page_count + start + 1
                if end < 0:
                    end = page_count + end + 1
                if start <= end:
                    page_numbers.update(range(start, end + 1))
            else:
                page_no = int(part)
                if page_no < 0:
                    page_no = page_count + page_no + 1
                page_numbers.add(page_no)
        page_numbers &= available_page_numbers
    if not page_numbers:
        raise InvalidRequestError("page_range_invalid", f"Page range does not select any pages: {page_range}", "page_range")
    return _page_numbers_to_range_str(page_numbers)


def _page_numbers_to_range_str(page_numbers: set[int]) -> str:
    if not page_numbers:
        return ""
    ranges: list[str] = []
    ordered = sorted(page_numbers)
    start = ordered[0]
    end = ordered[0]
    for page_no in ordered[1:]:
        if page_no == end + 1:
            end = page_no
            continue
        ranges.append(f"{start}~{end}" if start != end else str(start))
        start = page_no
        end = page_no
    ranges.append(f"{start}~{end}" if start != end else str(start))
    return ",".join(ranges)


def _select_context_pages(pages: list[PageInfo], target: ContentCursor | None, context: int) -> list[PageInfo]:
    if target is None or target.page_no is None or context <= 0:
        return pages
    start = target.page_no - context
    end = target.page_no + context
    return [page for page in pages if start <= page.page_idx + 1 <= end]


def _find_page(pages: list[PageInfo], page_no: int) -> PageInfo | None:
    return next((page for page in pages if page.page_idx + 1 == page_no), None)


def _find_block_by_no(page: PageInfo, block_no: int) -> Block | None:
    target_index = block_no - 1
    for block in page.para_blocks:
        found = _find_block_by_index(block, target_index)
        if found is not None:
            return found
    for block in page.discarded_blocks:
        found = _find_block_by_index(block, target_index)
        if found is not None:
            return found
    return None


def _find_block_by_index(block: Block, index: int) -> Block | None:
    if block.index == index:
        return block
    for child in block.blocks:
        found = _find_block_by_index(child, index)
        if found is not None:
            return found
    return None


def _is_office_page(page: PageInfo) -> bool:
    return page._backend == "office"


def _is_empty_bbox(bbox: object) -> bool:
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return True
    try:
        values = [float(v) for v in bbox]
    except (TypeError, ValueError):
        return True
    return tuple(values) == tuple(float(v) for v in EMPTY_BBOX) or values[0] >= values[2] or values[1] >= values[3]


def _first_image_span(block: Block) -> Span | None:
    for span in _iter_block_spans(block):
        if span.type == "image":
            return span
    return None


def _iter_block_spans(block: Block) -> Iterator[Span]:
    for line in block.lines:
        yield from line.spans
    for child in block.blocks:
        yield from _iter_block_spans(child)


def _span_image_bytes(span: Span) -> tuple[bytes | None, str, str]:
    # TODO: how to get image from office span?
    if not span.image_base64:
        return None, "png", "image/png"
    match = re.match(r"^data:image/(?P<ext>[^;]+);base64,(?P<data>.+)$", span.image_base64, re.DOTALL)
    if match is None:
        return None, "png", "image/png"
    ext = match.group("ext").lower()
    try:
        return base64.b64decode(match.group("data")), ext, _mime_type_for_ext(ext)
    except Exception:
        return None, ext, _mime_type_for_ext(ext)


_PIL_IMAGE_FORMATS: dict[ImageFormat, str] = {
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
}

_IMAGE_FORMAT_EXTENSIONS: dict[ImageFormat, str] = {
    "jpeg": "jpg",
    "png": "png",
    "webp": "webp",
}

_IMAGE_FORMAT_MIME_TYPES: dict[ImageFormat, str] = {
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
}


def _pil_image_to_bytes(image: Image.Image, image_format: ImageFormat) -> bytes:
    output_image = image
    if image_format == "jpeg" and image.mode != "RGB":
        output_image = image.convert("RGB")
    with BytesIO() as buffer:
        output_image.save(buffer, format=_PIL_IMAGE_FORMATS[image_format])
        return buffer.getvalue()


def _transcode_image_bytes(image_bytes: bytes, image_format: ImageFormat) -> bytes:
    with Image.open(BytesIO(image_bytes)) as image:
        return _pil_image_to_bytes(image, image_format)


def _image_format_ext(image_format: ImageFormat) -> str:
    return _IMAGE_FORMAT_EXTENSIONS[image_format]


def _mime_type_for_image_format(image_format: ImageFormat) -> str:
    return _IMAGE_FORMAT_MIME_TYPES[image_format]


def _image_size_from_bytes(image_bytes: bytes) -> tuple[int | None, int | None]:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return image.size
    except Exception:
        return None, None


def _write_temp_asset(
    data_dir: str,
    short_id: str,
    ext: str,
    image_bytes: bytes,
    *,
    mime_type: str,
    width: int | None,
    height: int | None,
) -> ContentAsset:
    safe_ext = ext.lower().lstrip(".") or "png"
    output_dir = Path(os.path.expanduser(data_dir)) / "temp" / "read-assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{short_id}-{int(time.time() * 1000)}.{safe_ext}"
    path.write_bytes(image_bytes)
    return ContentAsset(
        path=str(path),
        mime_type=mime_type,
        size_bytes=len(image_bytes),
        width=width,
        height=height,
    )


def _mime_type_for_ext(ext: str) -> str:
    normalized = ext.lower().lstrip(".")
    if normalized in {"jpg", "jpeg"}:
        return "image/jpeg"
    if normalized == "webp":
        return "image/webp"
    if normalized == "gif":
        return "image/gif"
    return "image/png"


def _render_progressive_markdown(
    pages: list[PageInfo],
    *,
    short_id: str,
    tier: Tier,
    after: Any,
    limit: int,
    add_markers: bool,
    target: ContentCursor | None = None,
    context: int = 0,
    img_bucket_path: str = "",
) -> _RenderedContent:
    output: list[str] = []
    ranges: list[ContentRange] = []
    started = after is None
    last_page_no = 0
    next_page_no: int | None = None
    cut_inside_page = False

    for page in pages:
        page_no = page.page_idx + 1
        if after and page_no < after.page_no:
            continue
        page_blocks = _page_markdown_blocks(page, img_bucket_path=img_bucket_path)
        if target and target.block_no is not None and page_no == target.page_no:
            start_block_no = max(1, target.block_no - context)
            end_block_no = target.block_no + context
            page_blocks = [(index, text) for index, text in page_blocks if start_block_no <= index + 1 <= end_block_no]
        page_output: list[str] = []
        page_start_cursor = page_ref(short_id, tier, page_no)
        page_end_cursor = page_ref(short_id, tier, page_no)
        page_started = False
        empty_page_selected = False

        for block_index, block_text in page_blocks:
            block_no = block_index + 1
            if after and page_no == after.page_no:
                if after.block_no is None:
                    continue
                if block_no < after.block_no:
                    continue
                if block_no == after.block_no and after.char_offset is None:
                    continue
            started = True
            text = block_text
            block_start_offset = 0
            if after and page_no == after.page_no and after.block_no == block_no and after.char_offset is not None:
                block_start_offset = after.char_offset
                text = text[block_start_offset:]
            if not text:
                continue
            block_start_cursor = (
                block_char_ref(short_id, tier, page_no, block_no, block_start_offset)
                if block_start_offset
                else block_ref(short_id, tier, page_no, block_no)
            )
            if not page_started:
                page_start_cursor = block_start_cursor if after and after.block_no is not None else page_start_cursor
                page_started = True
            candidate_page = [*page_output, text]
            candidate_content = _join_markdown([*output, *candidate_page])
            if len(candidate_content) <= limit:
                page_output.append(text)
                page_end_cursor = block_ref(short_id, tier, page_no, block_no)
                continue

            if page_output and not output:
                cut_inside_page = True
                break
            if output:
                next_page_no = page_no
                return _RenderedContent(
                    content=_join_markdown(output),
                    content_ranges=_merge_content_ranges(ranges),
                    truncated=True,
                    last_page_no=last_page_no,
                    next_page_no=next_page_no,
                    cut_inside_page=False,
                )

            cut = _soft_cut(text, limit)
            if cut <= 0:
                cut = min(len(text), limit)
            page_output.append(text[:cut])
            end_offset = block_start_offset + cut
            page_end_cursor = block_char_ref(short_id, tier, page_no, block_no, end_offset)
            cut_inside_page = True
            break

        if not page_output and after and page_no == after.page_no:
            continue
        if not page_output and (target is None or target.block_no is None):
            started = True
            empty_page_selected = True
            if add_markers:
                page_output.append(f"<!-- page {page_no} -->")

        if not started or (not page_output and not empty_page_selected):
            continue
        if add_markers:
            marker = f"<!-- page {page_no} -->"
            if not page_output or page_output[0] != marker:
                page_output.insert(0, marker)
        if page_output:
            output.extend(page_output)
        last_page_no = page_no
        ranges.append(ContentRange(page_range=str(page_no), start=page_start_cursor, end=page_end_cursor))
        if cut_inside_page:
            next_page_no = page_no
            return _RenderedContent(
                content=_join_markdown(output),
                content_ranges=_merge_content_ranges(ranges),
                truncated=True,
                last_page_no=last_page_no,
                next_page_no=next_page_no,
                cut_inside_page=True,
            )
        # Keep whole pages as the normal truncation boundary for paginated docs.
        if len(_join_markdown(output)) >= limit:
            next_page_no = page_no + 1
            return _RenderedContent(
                content=_join_markdown(output),
                content_ranges=_merge_content_ranges(ranges),
                truncated=next_page_no <= (pages[-1].page_idx + 1),
                last_page_no=last_page_no,
                next_page_no=next_page_no,
                cut_inside_page=False,
            )

    return _RenderedContent(
        content=_join_markdown(output),
        content_ranges=_merge_content_ranges(ranges),
        truncated=False,
        last_page_no=last_page_no,
        next_page_no=None,
        cut_inside_page=False,
    )


def _page_markdown_blocks(page: PageInfo, *, img_bucket_path: str = "") -> list[tuple[int, str]]:
    backend = page._backend
    result: list[tuple[int, str]] = []
    for block in page.para_blocks:
        if backend == "office":
            rendered = office_blocks_to_markdown([block], img_bucket_path=img_bucket_path, prefer_markdown_table=True)
        else:
            rendered = blocks_to_markdown([block], img_bucket_path=img_bucket_path, prefer_markdown_table=True)
        text = _join_markdown([item for item in rendered if item.strip()])
        if text.strip():
            result.append((block.index, text))
    return result


def _join_markdown(parts: list[str]) -> str:
    return "\n\n".join(part for part in parts if part)


def _soft_cut(text: str, limit: int) -> int:
    if len(text) <= limit:
        return len(text)
    window = text[:limit]
    for pattern in ("\n\n", "\n", "。", ". ", " "):
        pos = window.rfind(pattern)
        if pos > max(0, limit // 2):
            return pos + len(pattern)
    return limit


def _merge_content_ranges(ranges: list[ContentRange]) -> list[ContentRange]:
    if not ranges:
        return []
    merged: list[ContentRange] = []
    current = ranges[0]
    for item in ranges[1:]:
        if current.page_range and item.page_range:
            current_page_numbers = parse_page_range_set(current.page_range)
            item_page_numbers = parse_page_range_set(item.page_range)
            if max(current_page_numbers) + 1 == min(item_page_numbers):
                current = ContentRange(
                    page_range=_page_numbers_to_range_str(current_page_numbers | item_page_numbers),
                    start=current.start,
                    end=item.end,
                )
                continue
        merged.append(current)
        current = item
    merged.append(current)
    return merged


def _next_content_request(
    *,
    rendered: _RenderedContent,
    request_page_range: str | None,
    after: str | None,
    page_count: int | None,
    paginated: bool,
) -> ContentNextRequest | None:
    if not rendered.content_ranges:
        return None
    if not paginated:
        if rendered.truncated:
            return ContentNextRequest(after=rendered.content_ranges[-1].end)
        return None

    total_page_count = page_count or rendered.last_page_no
    if rendered.truncated:
        if rendered.cut_inside_page:
            start = rendered.next_page_no or rendered.last_page_no
            end = _last_requested_page(request_page_range) or start
            return ContentNextRequest(page_range=_range_str(start, end), after=rendered.content_ranges[-1].end)
        start = rendered.next_page_no or (rendered.last_page_no + 1)
        end = _last_requested_page(request_page_range) or min(total_page_count, start + 9)
        if start <= end:
            return ContentNextRequest(page_range=_range_str(start, end))
        if after:
            return ContentNextRequest(page_range=request_page_range, after=rendered.content_ranges[-1].end)
        return None

    start = rendered.last_page_no + 1
    if start > total_page_count:
        return None
    end = min(total_page_count, start + 9)
    return ContentNextRequest(page_range=_range_str(start, end))


def _next_read_request(
    rendered: _RenderedContent,
    short_id: str,
    tier: Tier,
    page_count: int | None,
) -> ContentNextRequest | None:
    if not rendered.content_ranges:
        return None
    if rendered.truncated:
        return ContentNextRequest(locator=rendered.content_ranges[-1].end)
    total_page_count = page_count or rendered.last_page_no
    next_page_no = rendered.last_page_no + 1
    if next_page_no > total_page_count:
        return None
    return ContentNextRequest(locator=page_ref(short_id, tier, next_page_no))


def _last_requested_page(page_range: str | None) -> int | None:
    if not page_range:
        return None
    page_numbers = parse_page_range_set(page_range)
    return max(page_numbers) if page_numbers else None


def _range_str(start: int, end: int) -> str:
    return f"{start}~{end}" if start != end else str(start)


def _effective_data_dir(state: Any) -> str:
    data_dir = getattr(state, "data_dir", "")
    if data_dir:
        return os.path.expanduser(data_dir)
    return os.path.expanduser(config.doclib.data_dir)


def _tail_log(log_path: str, lines: int = 100) -> list[str]:
    log_path = os.path.expanduser(log_path) if log_path else os.path.expanduser(config.doclib.log.resolved_app_path)
    if not os.path.isfile(log_path):
        return []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        return fh.readlines()[-lines:]


async def _watch_stats(db: DatabaseManager, watches: list[WatchTargetRow]) -> list[WatchStats]:
    file_rows = cast(
        list[WatchStatsFileRow],
        await db.fetchall(
            "SELECT watch_id, "
            "COUNT(*) AS total_files, "
            "SUM(CASE WHEN status=? THEN 1 ELSE 0 END) AS active_files, "
            "SUM(CASE WHEN status=? THEN 1 ELSE 0 END) AS deleted_files, "
            "SUM(CASE WHEN status=? THEN 1 ELSE 0 END) AS unreachable_files, "
            "SUM(CASE WHEN status=? AND sha256 IS NULL AND error_code IS NULL THEN 1 ELSE 0 END) AS pending_ingest_files, "
            "SUM(CASE WHEN error_code IS NOT NULL THEN 1 ELSE 0 END) AS file_error_count, "
            "COUNT(DISTINCT sha256) AS doc_count "
            "FROM files WHERE watch_id IS NOT NULL GROUP BY watch_id",
            (FILE_STATUS_ACTIVE, FILE_STATUS_DELETED, FILE_STATUS_UNREACHABLE, FILE_STATUS_ACTIVE),
        ),
    )
    parse_rows = cast(
        list[WatchParseCountRow],
        await db.fetchall(
            "SELECT f.watch_id AS watch_id, p.status AS status, COUNT(*) AS cnt "
            "FROM parses p "
            "JOIN (SELECT DISTINCT watch_id, sha256 FROM files WHERE watch_id IS NOT NULL AND sha256 IS NOT NULL) f "
            "ON f.sha256 = p.sha256 "
            "GROUP BY f.watch_id, p.status"
        ),
    )

    files_by_watch = {row["watch_id"]: row for row in file_rows}
    parses_by_watch: dict[int, dict[str, int]] = {}
    for row in parse_rows:
        watch_id = row["watch_id"]
        parses_by_watch.setdefault(watch_id, {})[row["status"]] = row["cnt"]

    stats: list[WatchStats] = []
    for watch in watches:
        watch_id = watch["id"]
        file_counts = files_by_watch.get(watch_id, {})
        parse_counts = parses_by_watch.get(watch_id, {})
        stats.append(
            WatchStats(
                watch_id=watch_id,
                path=watch["path"],
                label=watch.get("label"),
                removable=bool(watch.get("removable", False)),
                status=watch["status"],
                total_files=file_counts.get("total_files", 0) or 0,
                active_files=file_counts.get("active_files", 0) or 0,
                deleted_files=file_counts.get("deleted_files", 0) or 0,
                unreachable_files=file_counts.get("unreachable_files", 0) or 0,
                pending_ingest_files=file_counts.get("pending_ingest_files", 0) or 0,
                file_error_count=file_counts.get("file_error_count", 0) or 0,
                doc_count=file_counts.get("doc_count", 0) or 0,
                parse_pending_count=parse_counts.get(PARSE_STATUS_PENDING, 0),
                parse_parsing_count=parse_counts.get(PARSE_STATUS_PARSING, 0),
                parse_failed_count=parse_counts.get(PARSE_STATUS_FAILED, 0),
                parse_done_count=parse_counts.get(PARSE_STATUS_DONE, 0),
                last_scan_at=watch.get("last_scan_at"),
                last_scan_files=watch.get("last_scan_files", 0) or 0,
            )
        )
    return stats


async def _error_summary(db: DatabaseManager) -> ErrorSummary:
    file_errors = await _error_buckets(db, "files")
    doc_errors = await _error_buckets(db, "docs")
    parse_errors = await _error_buckets(db, "parses")
    return ErrorSummary(file_errors=file_errors, doc_errors=doc_errors, parse_errors=parse_errors)


async def _recent_scans(db: DatabaseManager, limit: int = 5) -> list[ScanInfo]:
    rows = cast(
        list[RecentScanRow],
        await db.fetchall(
            "SELECT id, path, kind, source, watch_id, status, files_seen, files_refreshed, files_new, files_changed, "
            "files_deleted, files_unreachable, files_error, files_unsupported, files_excluded, error_code, error_msg, "
            "started_at, finished_at, created_at, updated_at "
            "FROM scans ORDER BY created_at DESC, id DESC LIMIT ?",
            (limit,),
        ),
    )
    return [ScanInfo.model_validate(row) for row in rows]


async def _record_telemetry_count(state: Any, metric_name: str, *, dimensions: dict[str, str] | None = None) -> None:
    telemetry_svc = getattr(state, "telemetry_svc", None)
    if telemetry_svc is None:
        return
    await telemetry_svc.record_count(metric_name, dimensions=dimensions)


async def _record_telemetry_duration(
    state: Any,
    metric_name: str,
    start_ms: int,
    *,
    dimensions: dict[str, str] | None = None,
) -> None:
    telemetry_svc = getattr(state, "telemetry_svc", None)
    if telemetry_svc is None:
        return
    await telemetry_svc.record_duration_bucket(metric_name, duration_ms=max(0, _now_ms() - start_ms), dimensions=dimensions)


def _parse_route_status(response: ParseResponse) -> str:
    if response.cache_hit:
        return "cached"
    if response.created_parse_ids:
        return "queued"
    if response.reused_parse_ids:
        return "reused"
    return "direct"


def _telemetry_error_code(exc: Exception) -> str:
    if isinstance(exc, MineruError):
        return exc.code
    return "internal_error"


def _telemetry_output_format(value: str) -> str:
    if value == "markdown":
        return "markdown"
    if value == "image":
        return "image"
    return "other"


def _now_ms() -> int:
    return int(time.time() * 1000)


async def _error_buckets(db: DatabaseManager, table: str) -> list[ErrorBucket]:
    sql_by_table = {
        "files": (
            "SELECT error_code AS code, COUNT(*) AS cnt FROM files "
            "WHERE error_code IS NOT NULL GROUP BY error_code ORDER BY cnt DESC, error_code"
        ),
        "docs": (
            "SELECT error_code AS code, COUNT(*) AS cnt FROM docs "
            "WHERE error_code IS NOT NULL GROUP BY error_code ORDER BY cnt DESC, error_code"
        ),
        "parses": (
            "SELECT error_code AS code, COUNT(*) AS cnt FROM parses "
            "WHERE error_code IS NOT NULL GROUP BY error_code ORDER BY cnt DESC, error_code"
        ),
    }
    rows = cast(list[ErrorBucketRow], await db.fetchall(sql_by_table[table]))
    return [ErrorBucket(code=row["code"], count=row["cnt"]) for row in rows]


def _file_info(row: FileRow) -> FileInfo:
    return FileInfo.model_validate(row)


def _doc_info(row: DocRow, *, files: list[FileInfo] | None = None) -> DocInfo:
    data = dict(row)
    data["files"] = files
    return DocInfo.model_validate(data)


def _parse_info(row: ParseRow, *, coverage: ParseCoverage | None = None) -> ParseInfo:
    data = dict(row)
    data["coverage"] = coverage
    return ParseInfo.model_validate(data)


def _tier_parse_info(row: ParseRow) -> TierParseInfo:
    return TierParseInfo.model_validate(row)


def _watch_info(row: WatchTargetRow) -> WatchInfo:
    data = dict(row)
    data["removable"] = bool(data.get("removable", False))
    data["enabled"] = bool(data.get("enabled", True))
    data["recursive"] = bool(data.get("recursive", False))
    return WatchInfo.model_validate(data)


def _exclude_rule_info(row: ExcludeRuleRow | None) -> ExcludeRuleInfo:
    if row is None:
        raise NotFoundError("rule_not_found", "Exclude rule not found.", "rule_id")
    return ExcludeRuleInfo.model_validate(row)


def _parsing_rule_info(row: ParsingRuleRow | None) -> ParsingRuleInfo:
    if row is None:
        raise NotFoundError("rule_not_found", "Parsing rule not found.", "rule_id")
    data = dict(row)
    data["remote"] = bool(data.get("remote", False))
    data["enabled"] = bool(data.get("enabled", True))
    return ParsingRuleInfo.model_validate(data)


def _search_result(row: ContentSearchResultRow) -> SearchResult:
    data = dict(row)
    data["tier"] = cast(Tier, data["tier"])
    return SearchResult.model_validate(data)


def _find_result(row: FilenameSearchResultRow) -> FindResult:
    return FindResult.model_validate(row)


def _parse_coverage(request_page_range: str, rows: list[ParseRow]) -> ParseCoverage:
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
