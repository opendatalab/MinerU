"""New doclib HTTP server implementation backed by the public interface."""

from __future__ import annotations

import asyncio
import os
import signal
import time
from typing import Any, cast

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

from ..errors import InvalidRequestError, MineruError, NotFoundError, error_response
from ..render import render_markdown
from ..types import TIERS, Tier
from .background.parse_server_health import get_health
from .base import AsyncDoclibInterface
from .services.parse_svc import filter_pages_by_user_range, load_pages_from_done_batches, parse_range_set
from .types import (
    CleanupDeletedRequest,
    CleanupDeletedResponse,
    CleanupOrphansRequest,
    CleanupOrphansResponse,
    CleanupTempRequest,
    CleanupTempResponse,
    ConfigResponse,
    ConfigSetRequest,
    ConfigSetResponse,
    DocContentResponse,
    DocInfo,
    ExcludeRuleInfo,
    ExcludeRuleListResponse,
    ExcludeRuleRequest,
    FileInfo,
    FileInfoResponse,
    FindResponse,
    FindResult,
    InvalidateRequest,
    InvalidateResponse,
    ListDocsResponse,
    ListParsesResponse,
    LocalParseServerStatus,
    PARSE_STATUS_DONE,
    PARSE_STATUS_PARSING,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_SUPERSEDED,
    RULE_TYPE_EXCLUDE,
    RULE_TYPE_PARSING_RULE,
    SCAN_STATUS_ACTIVE,
    ParseCoverage,
    ParseInfo,
    ParseRequest,
    ParseResponse,
    ParseServerStatus,
    ParsingRuleInfo,
    ParsingRuleListResponse,
    ParsingRuleRequest,
    RemoteParseServerStatus,
    RemoveExcludeRuleResponse,
    RemoveParsingRuleResponse,
    RemoveWatchResponse,
    SearchResponse,
    SearchResult,
    ServerStatusResponse,
    ShutdownResponse,
    TierParseInfo,
    WatchInfo,
    WatchListResponse,
    WatchRequest,
)
from .utils.route_utils import get_route_info, has_route_info, route


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
        files_total = await self.state.db.fetchone(
            "SELECT COUNT(*) as cnt FROM files WHERE scan_status=?", (SCAN_STATUS_ACTIVE,)
        )
        docs_total = await self.state.db.fetchone("SELECT COUNT(*) as cnt FROM docs")
        ingest_q = await self.state.db.fetchone(
            "SELECT COUNT(*) as cnt FROM files WHERE sha256 IS NULL AND scan_status=?",
            (SCAN_STATUS_ACTIVE,),
        )
        watches = await self.state.config_svc.list_watches()
        uptime = time.time() - self.state.start_time if hasattr(self.state, "start_time") else 0

        return ServerStatusResponse(
            running=True,
            pid=getattr(self.state, "pid", os.getpid()),
            uptime_seconds=uptime,
            socket_path=getattr(self.state, "socket_path", ""),
            data_dir=getattr(self.state, "data_dir", ""),
            files_total=files_total["cnt"] if files_total else 0,
            docs_total=docs_total["cnt"] if docs_total else 0,
            parse_queue_length=await self.state.parse_svc.get_queue_length(),
            ingest_queue_length=ingest_q["cnt"] if ingest_q else 0,
            parse_server=_parse_server_status(),
            watch_count=len(watches),
            watches=[_watch_info(row) for row in watches],
            recent_logs=_tail_log(getattr(self.state, "data_dir", "")),
        )

    @route("POST", "/shutdown", tags=("server",))
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
        result = await self.state.parse_svc.request_parse(
            request.path,
            tier=request.tier,
            pages=request.pages,
            force=request.force,
            remote=request.remote,
        )
        return ParseResponse.model_validate(result)

    @route("GET", "/parses", tags=("parse",))
    async def list_parses(
        self,
        *,
        ids: list[int] | None = None,
        sha256: str | None = None,
        tier: Tier | None = None,
        status: str | None = None,
        pages: str | None = None,
        include_superseded: bool = False,
    ) -> ListParsesResponse:
        rows = await self._select_parse_rows(
            ids=ids,
            sha256=sha256,
            tier=tier,
            status=status,
            include_superseded=include_superseded,
        )
        coverage = _parse_coverage(pages, rows) if sha256 and tier and pages else None
        return ListParsesResponse(parses=[_parse_info(row, coverage=coverage) for row in rows], coverage=coverage)

    @route("GET", "/parses/{parse_id}", tags=("parse",))
    async def get_parse(self, parse_id: int) -> ParseInfo:
        row = await self.state.db.fetchone("SELECT * FROM parses WHERE id=?", (parse_id,))
        if row is None:
            raise NotFoundError("job_not_found", f"Parse {parse_id} not found.", "parse_id")
        return _parse_info(row)

    @route("POST", "/invalidate", tags=("parse",))
    async def invalidate(self, request: InvalidateRequest) -> InvalidateResponse:
        if request.target != "parses":
            raise InvalidRequestError("invalid_request", f"Unsupported invalidate target: {request.target}", "target")
        sha256 = request.sha256
        if sha256 is None and request.path:
            file_row = await self.state.parse_svc.ensure_ingested(request.path)
            sha256 = file_row["sha256"] if file_row and file_row.get("sha256") else None
        if sha256 is None:
            raise NotFoundError("file_not_found", "Document not found.", "path")

        count = await self.state.parse_svc.invalidate(sha256, request.tier)
        return InvalidateResponse(target=request.target, sha256=sha256, tier=request.tier, invalidated_count=count)

    @route("GET", "/docs", tags=("docs",))
    async def list_docs(self, *, path: str | None = None) -> ListDocsResponse:
        if path:
            await self.state.parse_svc.ensure_ingested(path)
            rows = await self.state.db.fetchall(
                "SELECT d.* FROM docs d JOIN files f ON f.sha256=d.sha256 "
                "WHERE f.path=? AND f.scan_status=?",
                (path, SCAN_STATUS_ACTIVE),
            )
        else:
            rows = await self.state.db.fetchall(
                "SELECT d.* FROM docs d WHERE EXISTS ("
                "  SELECT 1 FROM files f WHERE f.sha256=d.sha256 AND f.scan_status=?"
                ") ORDER BY d.updated_at DESC LIMIT 200",
                (SCAN_STATUS_ACTIVE,),
            )
        return ListDocsResponse(docs=[_doc_info(row) for row in rows])

    @route("GET", "/docs/{sha256}", tags=("docs",))
    async def get_doc(self, sha256: str, *, expand_files: bool = False) -> DocInfo:
        row = await self.state.db.fetchone("SELECT * FROM docs WHERE sha256=?", (sha256,))
        if row is None:
            raise NotFoundError("file_not_found", f"Document {sha256} not found.", "sha256")
        files = await self._files_for_sha256(sha256) if expand_files else None
        return _doc_info(row, files=files)

    @route("GET", "/docs/{sha256}/content", tags=("docs",))
    async def get_doc_content(
        self,
        sha256: str,
        *,
        tier: Tier,
        pages: str | None = None,
        format: str = "markdown",
        output: str | None = None,
        no_marker: bool = False,
    ) -> DocContentResponse:
        if format != "markdown":
            raise InvalidRequestError("invalid_request", "Only markdown is currently implemented.", "format")
        data_dir = getattr(self.state, "data_dir", os.path.expanduser("~/MinerU"))
        rows = await self.state.db.fetchall(
            "SELECT pages, done_at FROM parses WHERE sha256=? AND tier=? AND status=? ORDER BY done_at DESC",
            (sha256, tier, PARSE_STATUS_DONE),
        )
        loaded_pages = load_pages_from_done_batches(data_dir, sha256, tier, rows)
        if pages:
            loaded_pages = filter_pages_by_user_range(loaded_pages, pages)
        if not loaded_pages:
            raise NotFoundError("not_cached", "Requested parsed content is not cached.", "pages")

        content = render_markdown(loaded_pages, add_markers=not no_marker)
        output_path = None
        if output and output != "-":
            output_path = os.path.abspath(output)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            content = None
        return DocContentResponse(sha256=sha256, tier=tier, content=content, output=output_path)

    @route("GET", "/search", tags=("search",))
    async def search(self, query: str, *, file_type: str | None = None, limit: int = 20, offset: int = 0) -> SearchResponse:
        results, total = await self.state.search_svc.search(query=query, file_type=file_type, limit=limit, offset=offset)
        return SearchResponse(results=[_search_result(row) for row in results if row.get("tier") in TIERS], total=total, query=query)

    @route("GET", "/find", tags=("search",))
    async def find(self, query: str, *, limit: int = 50) -> FindResponse:
        results, total = await self.state.search_svc.search_filenames(query=query, limit=limit)
        return FindResponse(results=[_find_result(row) for row in results], total=total, query=query)

    @route("GET", "/info", tags=("info",))
    async def get_file_info(self, path: str) -> FileInfoResponse:
        await self.state.parse_svc.ensure_ingested(path)
        file_row = await self.state.db.fetchone("SELECT * FROM files WHERE path=? AND scan_status=?", (path, SCAN_STATUS_ACTIVE))
        if file_row is None:
            raise NotFoundError("file_not_found", f"File {path} not found.", "path")
        sha256 = file_row["sha256"]
        doc_row = await self.state.db.fetchone("SELECT * FROM docs WHERE sha256=?", (sha256,)) if sha256 else None
        parse_rows = (
            await self.state.db.fetchall("SELECT * FROM parses WHERE sha256=? ORDER BY tier, created_at DESC", (sha256,))
            if sha256
            else []
        )
        parsed_tiers = [_tier_parse_info(row) for row in parse_rows if row["status"] == PARSE_STATUS_DONE]
        active_parses = [_parse_info(row) for row in parse_rows if row["status"] in {PARSE_STATUS_PENDING, PARSE_STATUS_PARSING}]
        return FileInfoResponse(
            file=_file_info(file_row),
            doc=_doc_info(doc_row) if doc_row else None,
            parsed_tiers=parsed_tiers,
            active_parses=active_parses,
        )

    @route("GET", "/config", tags=("config",))
    async def get_config(self) -> ConfigResponse:
        return ConfigResponse(config=await self.state.config_svc.get_all())

    @route("POST", "/config", tags=("config",))
    async def set_config(self, request: ConfigSetRequest) -> ConfigSetResponse:
        await self.state.config_svc.set(request.key, request.value)
        return ConfigSetResponse(key=request.key, value=request.value)

    @route("POST", "/config/watch", tags=("config",))
    async def add_watch(self, request: WatchRequest) -> WatchInfo:
        row = await self.state.config_svc.add_watch(request.path, removable=request.removable, label=request.label)
        return _watch_info(row)

    @route("GET", "/config/watch", tags=("config",))
    async def list_watches(self) -> WatchListResponse:
        return WatchListResponse(watches=[_watch_info(row) for row in await self.state.config_svc.list_watches()])

    @route("DELETE", "/config/watch", tags=("config",))
    async def remove_watch(self, path: str) -> RemoveWatchResponse:
        existing = await self.state.db.fetchone("SELECT path FROM watch_targets WHERE path=?", (path,))
        if existing is None:
            raise NotFoundError("file_not_found", f"Watch target {path} not found.", "path")
        await self.state.config_svc.remove_watch(path)
        return RemoveWatchResponse(path=path, removed=True)

    @route("POST", "/config/exclude", tags=("config",))
    async def add_exclude_rule(self, request: ExcludeRuleRequest) -> ExcludeRuleInfo:
        rule_id = await self.state.config_svc.add_rule(
            request.name or "",
            RULE_TYPE_EXCLUDE,
            request.pattern,
            priority=request.priority,
        )
        row = await self.state.db.fetchone("SELECT * FROM exclude_rules WHERE id=?", (rule_id,))
        return _exclude_rule_info(row)

    @route("GET", "/config/exclude", tags=("config",))
    async def list_exclude_rules(self) -> ExcludeRuleListResponse:
        rows = await self.state.config_svc.list_rules(RULE_TYPE_EXCLUDE)
        return ExcludeRuleListResponse(rules=[_exclude_rule_info(row) for row in rows])

    @route("DELETE", "/config/exclude/{rule_id}", tags=("config",))
    async def remove_exclude_rule(self, rule_id: int) -> RemoveExcludeRuleResponse:
        existing = await self.state.db.fetchone("SELECT id FROM exclude_rules WHERE id=?", (rule_id,))
        if existing is None:
            raise NotFoundError("invalid_request", f"Exclude rule {rule_id} not found.", "rule_id")
        await self.state.config_svc.remove_rule(rule_id, RULE_TYPE_EXCLUDE)
        return RemoveExcludeRuleResponse(rule_id=rule_id, removed=True)

    @route("POST", "/config/parsing-rules", tags=("config",))
    async def add_parsing_rule(self, request: ParsingRuleRequest) -> ParsingRuleInfo:
        rule_id = await self.state.config_svc.add_rule(
            request.name or "",
            RULE_TYPE_PARSING_RULE,
            request.pattern,
            tier=request.tier,
            pages=request.pages,
            remote=request.remote,
            priority=request.priority,
        )
        row = await self.state.db.fetchone("SELECT * FROM parsing_rules WHERE id=?", (rule_id,))
        return _parsing_rule_info(row)

    @route("GET", "/config/parsing-rules", tags=("config",))
    async def list_parsing_rules(self) -> ParsingRuleListResponse:
        rows = await self.state.config_svc.list_rules(RULE_TYPE_PARSING_RULE)
        return ParsingRuleListResponse(rules=[_parsing_rule_info(row) for row in rows])

    @route("DELETE", "/config/parsing-rules/{rule_id}", tags=("config",))
    async def remove_parsing_rule(self, rule_id: int) -> RemoveParsingRuleResponse:
        existing = await self.state.db.fetchone("SELECT id FROM parsing_rules WHERE id=?", (rule_id,))
        if existing is None:
            raise NotFoundError("invalid_request", f"Parsing rule {rule_id} not found.", "rule_id")
        await self.state.config_svc.remove_rule(rule_id, RULE_TYPE_PARSING_RULE)
        return RemoveParsingRuleResponse(rule_id=rule_id, removed=True)

    @route("POST", "/cleanup/deleted", tags=("cleanup",))
    async def cleanup_deleted_files(self, request: CleanupDeletedRequest) -> CleanupDeletedResponse:
        count = await self.state.cleanup_svc.cleanup_deleted(
            older_than_days=request.older_than_days,
            dry_run=request.dry_run,
        )
        return CleanupDeletedResponse(deleted_files=count, dry_run=request.dry_run)

    @route("POST", "/cleanup/orphans", tags=("cleanup",))
    async def cleanup_orphan_docs(self, request: CleanupOrphansRequest) -> CleanupOrphansResponse:
        count = await self.state.cleanup_svc.cleanup_orphans(dry_run=request.dry_run)
        return CleanupOrphansResponse(orphan_docs=count, dry_run=request.dry_run)

    @route("POST", "/cleanup/temp", tags=("cleanup",))
    async def cleanup_temp_files(self, request: CleanupTempRequest) -> CleanupTempResponse:
        count = await self.state.cleanup_svc.cleanup_temp_files(older_than_days=request.older_than_days)
        return CleanupTempResponse(temp_files_removed=count)

    async def _select_parse_rows(
        self,
        *,
        ids: list[int] | None,
        sha256: str | None,
        tier: Tier | None,
        status: str | None,
        include_superseded: bool,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[object] = []
        if ids:
            placeholders = ",".join("?" * len(ids))
            rows = await self.state.db.fetchall(f"SELECT * FROM parses WHERE id IN ({placeholders})", tuple(ids))
            by_id = {row["id"]: row for row in rows}
            return [by_id[parse_id] for parse_id in ids if parse_id in by_id]
        if sha256:
            clauses.append("sha256=?")
            params.append(sha256)
        if tier:
            clauses.append("tier=?")
            params.append(tier)
        if status:
            statuses = [part.strip() for part in status.split(",") if part.strip()]
            placeholders = ",".join("?" * len(statuses))
            clauses.append(f"status IN ({placeholders})")
            params.extend(statuses)
        elif not include_superseded:
            clauses.append("status!=?")
            params.append(PARSE_STATUS_SUPERSEDED)

        sql = "SELECT * FROM parses"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        return await self.state.db.fetchall(sql, tuple(params))

    async def _sha256_for_path(self, path: str) -> str | None:
        row = await self.state.db.fetchone("SELECT sha256 FROM files WHERE path=? AND scan_status=?", (path, SCAN_STATUS_ACTIVE))
        return row["sha256"] if row and row["sha256"] else None

    async def _files_for_sha256(self, sha256: str) -> list[FileInfo]:
        rows = await self.state.db.fetchall("SELECT * FROM files WHERE sha256=? AND scan_status=?", (sha256, SCAN_STATUS_ACTIVE))
        return [_file_info(row) for row in rows]


async def _mineru_error_handler(_request: Request, exc: MineruError) -> JSONResponse:
    status_map = {
        "invalid_request_error": 400,
        "authentication_error": 401,
        "permission_error": 403,
        "rate_limit_error": 429,
        "engine_error": 503,
        "api_error": 500,
    }
    return JSONResponse(status_code=status_map.get(exc.type, 500), content=error_response(exc))


async def _unexpected_error_handler(_request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content=error_response(MineruError("internal_error", str(exc))))


async def _signal_shutdown() -> None:
    await asyncio.sleep(0.1)
    os.kill(os.getpid(), signal.SIGTERM)


def _parse_server_status() -> ParseServerStatus:
    health = get_health()
    return ParseServerStatus(
        local=LocalParseServerStatus(
            mode=health.local_mode,
            healthy=health.local_healthy,
            starting=health.local_starting,
            started_at=health.local_started_at or None,
            supported_tiers=health.local_supported_tiers,
        ),
        remote=RemoteParseServerStatus(
            healthy=health.remote_healthy,
            supported_tiers=health.remote_supported_tiers,
        ),
    )


def _tail_log(data_dir: str, lines: int = 100) -> list[str]:
    log_path = os.path.join(os.path.expanduser(data_dir or "~/MinerU"), "mineru.log")
    if not os.path.isfile(log_path):
        return []
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        return fh.readlines()[-lines:]


def _file_info(row: dict[str, Any]) -> FileInfo:
    return FileInfo.model_validate(row)


def _doc_info(row: dict[str, Any], *, files: list[FileInfo] | None = None) -> DocInfo:
    data = dict(row)
    data["files"] = files
    return DocInfo.model_validate(data)


def _parse_info(row: dict[str, Any], *, coverage: ParseCoverage | None = None) -> ParseInfo:
    data = dict(row)
    data["coverage"] = coverage
    return ParseInfo.model_validate(data)


def _tier_parse_info(row: dict[str, Any]) -> TierParseInfo:
    return TierParseInfo.model_validate(row)


def _watch_info(row: dict[str, Any]) -> WatchInfo:
    data = dict(row)
    data["removable"] = bool(data.get("removable", False))
    data["enabled"] = bool(data.get("enabled", True))
    data["recursive"] = bool(data.get("recursive", False))
    return WatchInfo.model_validate(data)


def _exclude_rule_info(row: dict[str, Any] | None) -> ExcludeRuleInfo:
    if row is None:
        raise NotFoundError("invalid_request", "Exclude rule not found.", "rule_id")
    return ExcludeRuleInfo.model_validate(row)


def _parsing_rule_info(row: dict[str, Any] | None) -> ParsingRuleInfo:
    if row is None:
        raise NotFoundError("invalid_request", "Parsing rule not found.", "rule_id")
    data = dict(row)
    data["remote"] = bool(data.get("remote", False))
    data["enabled"] = bool(data.get("enabled", True))
    return ParsingRuleInfo.model_validate(data)


def _search_result(row: dict[str, Any]) -> SearchResult:
    data = dict(row)
    data["tier"] = cast(Tier, data["tier"])
    return SearchResult.model_validate(data)


def _find_result(row: dict[str, Any]) -> FindResult:
    return FindResult.model_validate(row)


def _parse_coverage(request_pages: str, rows: list[dict[str, Any]]) -> ParseCoverage:
    requested = parse_range_set(request_pages)
    done_pages: set[int] = set()
    active_pages: set[int] = set()
    for row in rows:
        row_pages = parse_range_set(row["pages"]) & requested
        if row["status"] == PARSE_STATUS_DONE:
            done_pages |= row_pages
        elif row["status"] in {PARSE_STATUS_PENDING, PARSE_STATUS_PARSING}:
            active_pages |= row_pages
    active_pages -= done_pages
    missing_pages = requested - done_pages - active_pages
    return ParseCoverage(
        done_pages=_pages_set_to_str(done_pages),
        active_pages=_pages_set_to_str(active_pages),
        missing_pages=_pages_set_to_str(missing_pages),
    )


def _pages_set_to_str(pages: set[int]) -> str:
    if not pages:
        return ""
    sorted_pages = sorted(pages)
    ranges: list[str] = []
    start = sorted_pages[0]
    end = start
    for page in sorted_pages[1:]:
        if page == end + 1:
            end = page
        else:
            ranges.append(f"{start}~{end}" if start != end else str(start))
            start = end = page
    ranges.append(f"{start}~{end}" if start != end else str(start))
    return ",".join(ranges)
