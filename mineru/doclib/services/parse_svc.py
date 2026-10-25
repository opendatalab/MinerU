"""Doclib 解析请求、队列执行、产物写出和索引维护。"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import cast
from urllib.parse import urlparse

from ...errors import InvalidRequestError, MineruError
from ...filetypes import (
    OFD_EXTENSIONS,
    OFFICE_EXTENSIONS,
    PAGE_RANGE_PARSE_EXTENSIONS,
    TEXT_EXTENSIONS,
    TIERED_PARSE_EXTENSIONS,
    is_office_temp_lock_file,
)
from ...parser.api_client import _APITransportError, _V1APIError
from ...parser.base import ParseResult
from ...parser.page_range import (
    expand_page_range,
    normalize_page_range_input,
    normalize_result_page_range,
    parse_page_range_set,
)
from ...parser.page_range import (
    format_page_range as _page_numbers_to_range_str,
)
from ...types import QUALITY_TIERS, TIER_ORDER, Tier, select_default_quality_tier
from ..core import fts as fts_utils
from ..core.db import DatabaseManager
from ..core.file_io import (
    metadata_to_doclib,
)
from ..core.fts import FTSManager
from ..core.parse_artifacts import (
    accessible_file_for_sha256,
    load_pages_from_done_batches,
    parse_artifact_directory,
    parse_batch_is_readable,
    parse_batch_json_path,
    parse_coverage,
    write_parse_payload,
)
from ..remote_api import resolve_remote_api_key
from ..rows import DocRow, FileRow, ParseRow
from ..types import (
    FILE_STATUS_ACTIVE,
    PARSE_STATUS_DONE,
    PARSE_STATUS_FAILED,
    PARSE_STATUS_PARSING,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_SUPERSEDED,
    ParseResponse,
)
from . import parse_policy
from .config_svc import ConfigService
from .ingest_svc import IngestService, raise_refresh_error

logger = logging.getLogger("mineru.parse")


def _now_ms() -> int:
    return int(time.time() * 1000)


class ParseFailure(MineruError):
    """Raised when a parse cannot be completed.  Carries an error code for the parses row."""

    def __init__(self, code: str, message: str, param: str | None = None) -> None:
        super().__init__(code, message, param)
        self.message = message


# ── range helpers ──────────────────────────────────────────────────


def page_range_covered(request_page_range: str, done_batches: list[ParseRow]) -> bool:
    """Check whether request_page_range is fully covered by done batches.
    Batches are sorted by done_at DESC so newer batches take precedence."""
    needed_page_numbers = parse_page_range_set(request_page_range)
    covered_page_numbers: set[int] = set()
    for b in sorted(done_batches, key=lambda r: r.get("done_at") or 0, reverse=True):
        covered_page_numbers |= parse_page_range_set(b["page_range"])
        if needed_page_numbers <= covered_page_numbers:
            return True
    return False


def page_range_uncovered(request_page_range: str, done_batches: list[ParseRow]) -> set[int]:
    """Return the subset of request_page_range NOT covered by any done batch."""
    needed_page_numbers = parse_page_range_set(request_page_range)
    covered_page_numbers: set[int] = set()
    for b in done_batches:
        covered_page_numbers |= parse_page_range_set(b["page_range"])
    return needed_page_numbers - covered_page_numbers


# ── ParseService ───────────────────────────────────────────────────


class ParseService:
    def __init__(
        self,
        db: DatabaseManager,
        fts: FTSManager,
        config_svc: ConfigService,
        data_dir: str,
        *,
        ingest_svc: IngestService,
        parse_lock_timeout_sec: int,
        telemetry_svc: object | None = None,
    ) -> None:
        """保存解析依赖，文件刷新和入库由显式注入的服务处理。"""
        self.ingest_svc = ingest_svc
        self.db = db
        self.fts = fts
        self.config_svc = config_svc
        self.data_dir = os.path.expanduser(data_dir)
        self.parse_lock_timeout_ms = parse_lock_timeout_sec * 1000
        self.telemetry_svc = telemetry_svc

    # ── parse request ───────────────────────────────────────────

    async def request_parse(
        self,
        path: str,
        *,
        tier: Tier | None = None,
        page_range: str | None = None,
        force: bool = False,
        remote: bool = False,
    ) -> ParseResponse:
        """Handle a parse request from CLI.  Returns info for status polling."""
        page_range = normalize_page_range_input(page_range) or None
        # ensure the path is current before trusting files.sha256
        refreshed = await self.ingest_svc.refresh_file(path, ensure_ingested=True, allow_images=True)
        if refreshed.status == "unsupported":
            ext = Path(path).suffix.lower() or Path(path).name
            raise InvalidRequestError("file_type_unsupported", _unsupported_file_type_message(ext), "path")
        if refreshed.status in {"missing", "deleted", "unreachable"}:
            raise InvalidRequestError("file_not_found", f"File {path} not found.", "path")
        if refreshed.status == "error":
            raise_refresh_error(refreshed)
        if refreshed.file is None:
            raise MineruError("ingest_failed", "File could not be ingested.", "path")

        file_row = cast(FileRow | None, await self.db.fetchone("SELECT * FROM files WHERE path=?", (path,)))
        if file_row is None:
            raise MineruError("ingest_failed", "File could not be ingested.", "path")
        sha256 = file_row["sha256"]
        if sha256 is None:
            raise MineruError(
                file_row.get("error_code") or "ingest_failed",
                file_row.get("error_msg") or "File could not be ingested.",
                "path",
            )
        doc = cast(DocRow | None, await self.db.fetchone("SELECT * FROM docs WHERE sha256=?", (sha256,)))
        page_count = doc["page_count"] if doc else 1
        short_id = doc["short_id"] if doc else None
        privacy = "remote" if remote else "local"
        ext = file_row["ext"]

        # ── resolve tier (before cache check) ──
        if ext in TEXT_EXTENSIONS:
            raise InvalidRequestError(
                "parse_not_required",
                "Text files do not require MinerU parsing. Read the file directly.",
                "path",
            )
        if page_range and ext not in PAGE_RANGE_PARSE_EXTENSIONS:
            raise InvalidRequestError(
                "page_range_invalid",
                f"Page range is only supported for PDF files; '{ext}' uses full-document parsing.",
                "page_range",
            )
        if remote and ext not in TIERED_PARSE_EXTENSIONS:
            raise InvalidRequestError(
                "remote_unsupported_for_file_type",
                f"Remote parsing is only supported for PDF and image files; '{ext}' files use local flash parsing.",
                "remote",
            )
        if remote and tier == "flash":
            raise InvalidRequestError(
                "tier_unsupported_for_remote",
                "--remote does not support tier 'flash'; use --tier basic, --tier standard, or --tier advanced.",
                "tier",
            )
        if tier in QUALITY_TIERS and ext not in TIERED_PARSE_EXTENSIONS:
            raise InvalidRequestError(
                "tier_unsupported_for_file_type",
                f"Tier '{tier}' is only supported for PDF and image files; '{ext}' files use --tier flash.",
                "tier",
            )
        if ext in TIERED_PARSE_EXTENSIONS:
            requested_tier = tier or _resolve_default_tier(remote)
        else:
            requested_tier = "flash"

        # ── expand page range ──
        supports_page_range = ext in PAGE_RANGE_PARSE_EXTENSIONS
        requested_page_range_input = page_range or (
            parse_policy.default_parse_range(page_count) if supports_page_range else None
        )
        request_page_range = expand_page_range(requested_page_range_input, page_count or 1)
        needed_page_numbers = parse_page_range_set(request_page_range)

        # ── step 1: remove page numbers covered by valid done batches ──
        if not force:
            done_batches = cast(
                list[ParseRow],
                await self.db.fetchall(
                    "SELECT * FROM parses WHERE sha256=? AND tier=? AND status=? ORDER BY done_at DESC",
                    (sha256, requested_tier, PARSE_STATUS_DONE),
                ),
            )
            valid_done_batches = [
                batch for batch in done_batches if parse_batch_is_readable(self.data_dir, sha256, requested_tier, batch)
            ]
            if supports_page_range:
                for batch in valid_done_batches:
                    needed_page_numbers -= parse_page_range_set(batch["page_range"])
                if not needed_page_numbers:
                    return _done_response(sha256, short_id, requested_tier, request_page_range)
            elif any(needed_page_numbers <= parse_page_range_set(batch["page_range"]) for batch in valid_done_batches):
                return _done_response(sha256, short_id, requested_tier, request_page_range)

        # ── step 2: remove page numbers covered by pending/parsing batches ──
        reused_parse_ids: list[int] = []
        active_batches = cast(
            list[ParseRow],
            await self.db.fetchall(
                "SELECT * FROM parses WHERE sha256=? AND tier=? AND status IN (?, ?)",
                (sha256, requested_tier, PARSE_STATUS_PENDING, PARSE_STATUS_PARSING),
            ),
        )
        if active_batches:
            if supports_page_range:
                active_covered_page_numbers: set[int] = set()
                for batch in active_batches:
                    covered_page_numbers = parse_page_range_set(batch["page_range"])
                    if needed_page_numbers & covered_page_numbers:
                        reused_parse_ids.append(batch["id"])
                    active_covered_page_numbers |= covered_page_numbers
                needed_page_numbers -= active_covered_page_numbers
            else:
                reused_parse_ids = [batch["id"] for batch in active_batches]
                needed_page_numbers.clear()

            # bump priority for reused in-progress batches
            now = _now_ms()
            for batch in active_batches:
                if batch["id"] in reused_parse_ids and batch["priority"] < 1:
                    await self.db.execute(
                        "UPDATE parses SET priority=1, updated_at=? WHERE id=?",
                        (now, batch["id"]),
                    )

            if not needed_page_numbers:
                return ParseResponse(
                    sha256=sha256,
                    short_id=short_id,
                    tier=requested_tier,
                    page_range=request_page_range,
                    status=PARSE_STATUS_PENDING,
                    cache_hit=False,
                    wait_parse_ids=reused_parse_ids,
                    created_parse_ids=[],
                    reused_parse_ids=reused_parse_ids,
                    tip="Pages already queued. Priority bumped.",
                )

        # ── step 3: enqueue remaining uncovered page numbers ──
        uncovered_page_range = _page_numbers_to_range_str(needed_page_numbers) if supports_page_range else request_page_range
        now = _now_ms()
        parse_id = await self.db.execute_insert(
            "INSERT INTO parses (sha256, tier, page_range, status, privacy, priority, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
            (sha256, requested_tier, uncovered_page_range, PARSE_STATUS_PENDING, privacy, now, now),
        )
        await self._record_count("parse_task.created.count", dimensions={"tier": requested_tier})
        created_parse_ids = [parse_id]
        return ParseResponse(
            sha256=sha256,
            short_id=short_id,
            tier=requested_tier,
            page_range=request_page_range,
            status=PARSE_STATUS_PENDING,
            cache_hit=False,
            wait_parse_ids=reused_parse_ids + created_parse_ids,
            created_parse_ids=created_parse_ids,
            reused_parse_ids=reused_parse_ids,
        )

    # ── worker ──────────────────────────────────────────────────

    async def get_queue_length(self) -> int:
        timeout_ms = _now_ms() - self.parse_lock_timeout_ms
        row = await self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM parses WHERE status=? AND (locked_at IS NULL OR locked_at < ?)",
            (PARSE_STATUS_PENDING, timeout_ms),
        )
        return row["cnt"] if row else 0

    async def acquire_task(self) -> ParseRow | None:
        now = _now_ms()
        timeout = now - self.parse_lock_timeout_ms
        task = cast(
            ParseRow | None,
            await self.db.fetchone_write(
                "UPDATE parses SET locked_at=?, status=? "
                "WHERE id = ("
                "  SELECT id FROM parses WHERE status=? "
                "  AND (locked_at IS NULL OR locked_at < ?) "
                "  ORDER BY priority DESC, created_at ASC LIMIT 1"
                ") RETURNING *",
                (now, PARSE_STATUS_PARSING, PARSE_STATUS_PENDING, timeout),
            ),
        )
        if task is not None:
            await self._record_count("parse_task.started.count", dimensions={"tier": task["tier"]})
        return task

    async def process_doc(self, task: ParseRow) -> bool:
        """Execute parse for a batch.  Returns True on success."""
        task_start_ms = _now_ms()
        sha256 = task["sha256"]
        tier = task["tier"]
        page_range = task["page_range"]
        privacy = task.get("privacy", "local")

        # guard
        current = await self.db.fetchone("SELECT status FROM parses WHERE id=?", (task["id"],))
        if current is None or current["status"] != PARSE_STATUS_PARSING:
            return False

        # find the file
        file_row = await accessible_file_for_sha256(self.db, sha256)
        if file_row is None:
            await self._fail_task(task["id"], "no_accessible_file", "No active file found for this document")
            await self._record_parse_task_finished(
                task_start_ms,
                tier=tier,
                status="failed",
                error_code="no_accessible_file",
            )
            return False
        if is_office_temp_lock_file(file_row["path"]):
            await self._fail_task(task["id"], "file_type_unsupported", "Office temporary lock files are not supported")
            await self._record_parse_task_finished(
                task_start_ms,
                tier=tier,
                status="failed",
                error_code="file_type_unsupported",
            )
            return False
        file_ext = file_row.get("ext") or Path(file_row["path"]).suffix.lower().lstrip(".")

        output_dir = parse_artifact_directory(self.data_dir, sha256, tier)

        # route parse based on tier
        via = "local"
        execute_start_ms = _now_ms()
        server_dim = "local(flash)" if tier == "flash" else "unknown"
        try:
            if tier == "flash":
                logger.debug(
                    "Local flash parse started task_id=%s path=%s page_range=%s",
                    task["id"],
                    file_row["path"],
                    page_range,
                )
                result = await self._parse_via_local(file_row, tier, page_range)
                logger.debug(
                    "Local flash parse completed task_id=%s elapsed_ms=%d",
                    task["id"],
                    _now_ms() - execute_start_ms,
                )
            else:
                result, via = await self._parse_via_api(file_row, tier, page_range, privacy)
                server_dim = await self._telemetry_server_dim(via=via, tier=tier)
        except ParseFailure as exc:
            await self._record_execute(
                execute_start_ms,
                tier=tier,
                server=server_dim,
                status="failed",
                error_code=exc.code,
            )
            await self._fail_task(task["id"], exc.code, exc.message)
            await self._record_parse_task_finished(task_start_ms, tier=tier, status="failed", error_code=exc.code)
            return False
        except _V1APIError as exc:
            await self._record_execute(
                execute_start_ms,
                tier=tier,
                server=server_dim,
                status="failed",
                error_code=exc.code,
            )
            await self._fail_task(task["id"], exc.code, exc.message[:500])
            await self._record_parse_task_finished(task_start_ms, tier=tier, status="failed", error_code=exc.code)
            return False
        except Exception as exc:
            logger.error(
                "Parse execution failed for task_id=%s path=%s tier=%s page_range=%s: %s",
                task["id"],
                file_row["path"],
                tier,
                page_range,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            await self._record_execute(
                execute_start_ms,
                tier=tier,
                server=server_dim,
                status="failed",
                error_code="parse_failed",
            )
            await self._fail_task(task["id"], "parse_failed", str(exc)[:500])
            await self._record_parse_task_finished(task_start_ms, tier=tier, status="failed", error_code="parse_failed")
            return False
        await self._record_execute(execute_start_ms, tier=tier, server=server_dim, status="succeeded")

        export_payload = result.to_dict(skip_defaults=True)
        new_pages = export_payload["pages"]
        if not new_pages:
            await self._fail_task(task["id"], "parse_empty", "Parse completed but returned no pages")
            await self._record_parse_task_finished(task_start_ms, tier=tier, status="failed", error_code="parse_failed")
            return False

        persisted_page_range = page_range
        full_document_page_count: int | None = None
        if file_ext not in PAGE_RANGE_PARSE_EXTENSIONS:
            actual_page_numbers = {page.page_idx + 1 for page in result.pages}
            persisted_page_range = _page_numbers_to_range_str(actual_page_numbers)
            full_document_page_count = max(actual_page_numbers)

        # save per-batch JSON (markdown is generated on read from /docs/{doc_ref}/content)
        done_at_ms = _now_ms()
        json_path = parse_batch_json_path(self.data_dir, sha256, tier, persisted_page_range, done_at_ms)
        write_start_ms = _now_ms()
        try:
            os.makedirs(output_dir, exist_ok=True)
            write_parse_payload(json_path, export_payload)
        except Exception as exc:
            logger.error(
                "Parse post-write finalization failed for task_id=%s path=%s tier=%s page_range=%s: %s",
                task["id"],
                file_row["path"],
                tier,
                persisted_page_range,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
            await self._fail_task(task["id"], "parse_json_write_failed", str(exc)[:500])
            await self._record_write(
                write_start_ms,
                tier=tier,
                status="failed",
                error_code="parse_json_write_failed",
            )
            await self._record_parse_task_finished(
                task_start_ms,
                tier=tier,
                status="failed",
                error_code="parse_json_write_failed",
            )
            return False

        try:
            # generate markdown for FTS (not persisted, only for search indexing)
            md_text = ""
            md = result.markdown(add_markers=False) if hasattr(result, "markdown") else ""
            md_text += md + "\n"

            # 原文件属性不受推理档位限制，先更新数据库和已有索引，再处理正文索引。
            await self._update_source_metadata(sha256, result)
            refreshed_doc = await self.db.fetchone("SELECT title, author FROM docs WHERE sha256=?", (sha256,))
            if refreshed_doc:
                file_row = cast(FileRow, {**file_row, **refreshed_doc})

            # update fts_contents (tier-gated)
            await self._maybe_update_fts(sha256, tier, md_text, file_row)

            # update docs metadata (tier-gated)
            await self._maybe_update_docs_meta(sha256, tier)
            if full_document_page_count is not None:
                await self.db.execute(
                    "UPDATE docs SET page_count=?, updated_at=? WHERE sha256=?",
                    (full_document_page_count, done_at_ms, sha256),
                )

            # mark done
            await self.db.execute(
                "UPDATE parses SET status=?, page_range=?, done_at=?, locked_at=NULL, via=?, updated_at=? WHERE id=?",
                (PARSE_STATUS_DONE, persisted_page_range, done_at_ms, via, done_at_ms, task["id"]),
            )
        except Exception as exc:
            await self._fail_task(task["id"], "parse_json_write_failed", str(exc)[:500])
            await self._record_write(
                write_start_ms,
                tier=tier,
                status="failed",
                error_code="parse_json_write_failed",
            )
            await self._record_parse_task_finished(
                task_start_ms,
                tier=tier,
                status="failed",
                error_code="parse_json_write_failed",
            )
            return False

        await self._record_write(write_start_ms, tier=tier, status="succeeded")
        await self._record_count("parse_task.files.count", dimensions={"tier": tier})
        await self._record_count("parse_task.pages.count", value=len(new_pages), dimensions={"tier": tier})
        await self._record_parse_task_finished(task_start_ms, tier=tier, status="succeeded")
        return True

    # ── parse routing helpers ─────────────────────────────────────

    async def _parse_via_local(
        self,
        file_row: FileRow,
        tier: Tier,
        page_range: str,
    ) -> ParseResult:
        """Parse via local library call."""
        from ...parser import parse

        file_ext = file_row.get("ext") or Path(file_row["path"]).suffix.lower().lstrip(".")
        parser_page_range = page_range if file_ext in PAGE_RANGE_PARSE_EXTENSIONS else ""
        result = await asyncio.to_thread(
            parse,
            file_row["path"],
            tier=tier,
            page_range=parser_page_range,
        )
        return result

    async def _parse_via_api(
        self,
        file_row: FileRow,
        tier: Tier,
        page_range: str,
        privacy: str,
    ) -> tuple[ParseResult, str]:
        """Parse via HTTP API (local or remote parse-server). Returns (results, via)."""
        from ...parser.api_client import MinerUApiParser

        base_url, api_key, via = await self._resolve_api_target(privacy, tier)

        # resolve and validate tier against server capabilities
        resolved_tier = self._resolve_tier(tier, via)

        parser = MinerUApiParser(
            api_url=base_url,
            api_key=api_key,
            tier=resolved_tier,
            include_images=Path(file_row["path"]).suffix.lower().lstrip(".") in (OFFICE_EXTENSIONS | OFD_EXTENSIONS),
        )
        try:
            file_ext = file_row.get("ext") or Path(file_row["path"]).suffix.lower().lstrip(".")
            parser_page_range = page_range if file_ext in PAGE_RANGE_PARSE_EXTENSIONS else ""
            result = await parser.parse_async(file_row["path"], page_range=parser_page_range)
        except _APITransportError as exc:
            if via == "remote":
                code = "remote_timeout" if exc.timed_out else "remote_unreachable"
                target = "Remote API"
            else:
                code = "parse_server_unavailable"
                target = "Local parse-server"
            raise ParseFailure(
                code,
                f"{target} transport failed during {exc.stage} after {exc.attempts} attempt(s) ({type(exc.cause).__name__}).",
            ) from exc
        if file_ext in PAGE_RANGE_PARSE_EXTENSIONS:
            _remap_api_result_pages_to_page_range(result, page_range)
        return result, via

    @staticmethod
    def _resolve_tier(tier: Tier, via: str) -> Tier:
        """Resolve and validate tier against parse-server capabilities.

        - validate specified tier against supported_tiers
        - raise tier_mismatch if unsupported
        """
        from ..background.parse_server_health import get_health

        health = get_health()
        if via == "remote":
            supported = health.remote.probe.tiers
        else:
            supported = health.local.probe.tiers

        if not supported:
            raise ParseFailure("engine_unavailable", "Parse-server health status unknown, cannot validate tier")

        if tier not in supported:
            raise ParseFailure("tier_mismatch", f"Tier '{tier}' not supported by parse-server. Available: {supported}")
        return tier

    async def _resolve_api_target(self, privacy: str, tier: Tier) -> tuple[str, str | None, str]:
        """Resolve the api target URL based on privacy and config.

        Returns (base_url, api_key, via_label).
        """
        from ..background.parse_server_health import get_health

        health = get_health()

        if privacy == "remote":
            url = cast(str, await self.config_svc.get("parse_server.remote.url"))
            api_key = (await resolve_remote_api_key(self.config_svc)).value
            if not health.remote.probe.healthy:
                if health.remote.probe.error_code == "invalid_api_key":
                    raise ParseFailure(
                        "invalid_api_key",
                        _probe_error_message("Remote parse-server authentication failed", health.remote.probe.error_msg),
                        "parse_server.remote.api_key",
                    )
                # try fallback to local
                local_mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
                if local_mode != "disabled" and health.local.probe.healthy:
                    local_url = _local_parse_server_url(local_mode, health)
                    if local_url:
                        return local_url, await _local_parse_server_api_key(self.config_svc), "local"
                raise ParseFailure("parse_server_unavailable", "Remote parse-server unavailable and no local fallback")
            return url, api_key, "remote"

        # privacy == local
        local_mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
        if local_mode == "disabled":
            raise ParseFailure("no_engine", "Local parse-server is disabled. Use --tier flash or --remote.")

        local_url = _local_parse_server_url(local_mode, health)
        if local_url is None:
            raise ParseFailure("engine_unavailable", "Local parse-server URL not configured.")

        if not health.local.probe.healthy:
            if local_mode == "self_hosted" and health.local.probe.error_code == "invalid_api_key":
                raise ParseFailure(
                    "invalid_api_key",
                    _probe_error_message("Local self-hosted parse-server authentication failed", health.local.probe.error_msg),
                    "parse_server.local.self_hosted_api_key",
                )
            raise ParseFailure("engine_unavailable", "Local parse-server is not ready. Please wait or check server status.")

        api_key = await _local_parse_server_api_key(self.config_svc)
        return local_url, api_key, "local"

    async def _fail_task(self, task_id: int, code: str, message: str) -> None:
        now = _now_ms()
        await self.db.execute(
            "UPDATE parses SET status=?, error_code=?, error_msg=?, locked_at=NULL, updated_at=? WHERE id=?",
            (PARSE_STATUS_FAILED, code, message, now, task_id),
        )

    async def _telemetry_server_dim(self, *, via: str, tier: Tier) -> str:
        if tier == "flash":
            return "local(flash)"
        if via == "local":
            local_mode = (await self.config_svc.get("parse_server.local.mode")) or "disabled"
            if local_mode == "managed":
                return "local(managed)"
            if local_mode == "self_hosted":
                return "local(self-hosted)"
            return "unknown"
        if via == "remote":
            url = cast(str | None, await self.config_svc.get("parse_server.remote.url"))
            return "remote(official)" if _is_official_remote_url(url or "") else "remote(custom)"
        return "unknown"

    async def _record_parse_task_finished(
        self,
        start_ms: int,
        *,
        tier: str,
        status: str,
        error_code: str | None = None,
    ) -> None:
        dimensions = {"status": status, "tier": tier}
        if error_code:
            dimensions["error_code"] = error_code
        await self._record_count("parse_task.finished.count", dimensions=dimensions)
        await self._record_duration("parse_task.duration_bucket.count", start_ms, dimensions=dimensions)

    async def _record_execute(
        self,
        start_ms: int,
        *,
        tier: str,
        server: str,
        status: str,
        error_code: str | None = None,
    ) -> None:
        dimensions = {"status": status, "tier": tier, "server": server}
        if error_code:
            dimensions["error_code"] = error_code
        await self._record_count("parse_task.execute.count", dimensions=dimensions)
        await self._record_duration("parse_task.execute_duration_bucket.count", start_ms, dimensions=dimensions)

    async def _record_write(
        self,
        start_ms: int,
        *,
        tier: str,
        status: str,
        error_code: str | None = None,
    ) -> None:
        dimensions = {"status": status, "tier": tier}
        if error_code:
            dimensions["error_code"] = error_code
        await self._record_count("parse_task.write.count", dimensions=dimensions)
        await self._record_duration("parse_task.write_duration_bucket.count", start_ms, dimensions=dimensions)

    async def _record_count(
        self,
        metric_name: str,
        *,
        value: int = 1,
        dimensions: dict[str, str] | None = None,
    ) -> None:
        if self.telemetry_svc is None:
            return
        record = getattr(self.telemetry_svc, "record_count", None)
        if record is not None:
            await record(metric_name, value=value, dimensions=dimensions)

    async def _record_duration(self, metric_name: str, start_ms: int, *, dimensions: dict[str, str] | None = None) -> None:
        if self.telemetry_svc is None:
            return
        record = getattr(self.telemetry_svc, "record_duration_bucket", None)
        if record is not None:
            await record(metric_name, duration_ms=max(0, _now_ms() - start_ms), dimensions=dimensions)

    async def get_parse_record(self, parse_id: int) -> dict | None:
        row = cast(ParseRow | None, await self.db.fetchone("SELECT * FROM parses WHERE id=?", (parse_id,)))
        return _parse_record_response(row) if row else None

    async def list_parse_records(
        self,
        *,
        ids: list[int] | None = None,
        sha256: str | None = None,
        tier: Tier | None = None,
        status: list[str] | None = None,
        page_range: str | None = None,
        include_superseded: bool = False,
    ) -> dict:
        if ids:
            placeholders = ",".join("?" * len(ids))
            rows = cast(
                list[ParseRow], await self.db.fetchall(f"SELECT * FROM parses WHERE id IN ({placeholders})", tuple(ids))
            )
            rows_by_id = {row["id"]: row for row in rows}
            ordered_rows = [rows_by_id[parse_id] for parse_id in ids if parse_id in rows_by_id]
            return {"parses": [_parse_record_response(row) for row in ordered_rows]}

        clauses: list[str] = []
        params: list[object] = []
        if sha256:
            clauses.append("sha256=?")
            params.append(sha256)
        if tier:
            clauses.append("tier=?")
            params.append(tier)
        if status:
            placeholders = ",".join("?" * len(status))
            clauses.append(f"status IN ({placeholders})")
            params.extend(status)
        elif not include_superseded:
            clauses.append("status!=?")
            params.append(PARSE_STATUS_SUPERSEDED)

        sql = "SELECT * FROM parses"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC"
        rows = cast(list[ParseRow], await self.db.fetchall(sql, tuple(params)))
        result: dict[str, object] = {"parses": [_parse_record_response(row) for row in rows]}
        page_range = normalize_page_range_input(page_range)
        if sha256 and tier and page_range:
            doc = await self.db.fetchone("SELECT * FROM docs WHERE sha256=?", (sha256,))
            if doc is None:
                raise InvalidRequestError("doc_not_found", f"Document {sha256} not found.", "doc_ref")
            result["coverage"] = parse_coverage(expand_page_range(page_range, doc["page_count"] or 1), rows).model_dump()
        return result

    async def invalidate(self, sha256: str, tier: Tier | None = None) -> int:
        """Mark done parses as superseded.  Returns count of affected rows."""
        now = _now_ms()
        sql = "UPDATE parses SET status=?, updated_at=? WHERE sha256=? AND status=?"
        params = [PARSE_STATUS_SUPERSEDED, now, sha256, PARSE_STATUS_DONE]
        if tier:
            sql += " AND tier=?"
            params.append(tier)
        cursor = await self.db.execute(sql, tuple(params))
        if cursor.rowcount:
            await self._rebuild_fts_after_invalidate(sha256)
        return cursor.rowcount

    # ── internal helpers ────────────────────────────────────────

    async def _maybe_update_fts(self, sha256: str, tier: Tier, text: str, file_row: FileRow) -> None:
        existing_tier = await self.fts.get_tier(sha256)
        if existing_tier and TIER_ORDER.get(existing_tier, -1) >= TIER_ORDER.get(tier, -1):
            return  # current FTS data is from a higher or equal tier

        text = fts_utils.truncate_head_tail(text)
        await self.fts.replace(
            sha256=sha256,
            tier=tier,
            text=text,
            title=file_row.get("title") or "",
            author=file_row.get("author") or "",
        )

    async def _update_source_metadata(self, sha256: str, result: ParseResult) -> None:
        """原文档属性填补现有列并同步索引，空值不清空已有信息。"""
        values = metadata_to_doclib(result.middle_json.metadata)
        await self.db.execute(
            "UPDATE docs SET title=COALESCE(?,title), author=COALESCE(?,author), "
            "subject=COALESCE(?,subject), keywords=COALESCE(?,keywords), language=COALESCE(?,language), "
            "updated_at=? WHERE sha256=?",
            (values["title"], values["author"], values["subject"], values["keywords"], values["language"], _now_ms(), sha256),
        )
        await self.db.execute(
            "UPDATE fts_contents SET title=COALESCE((SELECT title FROM docs WHERE sha256=?),''), "
            "author=COALESCE((SELECT author FROM docs WHERE sha256=?),'') WHERE sha256=?",
            (sha256, sha256, sha256),
        )
        if result.middle_json.metadata.file_suffix == "pdf" and values["page_count"] is not None:
            await self.db.execute("UPDATE docs SET page_count=? WHERE sha256=?", (values["page_count"], sha256))

    async def _maybe_update_docs_meta(self, sha256: str, tier: Tier) -> None:
        doc = await self.db.fetchone("SELECT meta_tier FROM docs WHERE sha256=?", (sha256,))
        if doc is None:
            return
        existing_tier = doc.get("meta_tier")
        now = _now_ms()
        if existing_tier and TIER_ORDER.get(existing_tier, -1) >= TIER_ORDER.get(tier, -1):
            await self.db.execute(
                "UPDATE docs SET error_code=NULL, error_msg=NULL, updated_at=? WHERE sha256=?",
                (now, sha256),
            )
            return

        # for now just update meta_tier; full metadata update comes when
        # engine provides richer output
        await self.db.execute(
            "UPDATE docs SET meta_tier=?, error_code=NULL, error_msg=NULL, updated_at=? WHERE sha256=?",
            (tier, now, sha256),
        )

    async def _rebuild_fts_after_invalidate(self, sha256: str) -> None:
        # FTS rebuild reads only cached parse batches, so any active row works;
        # keep it deterministic and tolerate rows whose file moved away.
        file_row = cast(
            FileRow | None,
            await self.db.fetchone(
                "SELECT * FROM files WHERE sha256=? AND status=? ORDER BY updated_at DESC",
                (sha256, FILE_STATUS_ACTIVE),
            ),
        )
        if file_row is None:
            await self.fts.delete(sha256)
            return
        doc_row = cast(DocRow | None, await self.db.fetchone("SELECT title, author FROM docs WHERE sha256=?", (sha256,)))

        rows = cast(
            list[ParseRow],
            await self.db.fetchall(
                "SELECT * FROM parses WHERE sha256=? AND status=? ORDER BY done_at DESC",
                (sha256, PARSE_STATUS_DONE),
            ),
        )
        tier_set: set[Tier] = {row["tier"] for row in rows}
        tiers = sorted(tier_set, key=lambda item: TIER_ORDER.get(item, -1), reverse=True)
        for tier in tiers:
            tier_rows = [
                row for row in rows if row["tier"] == tier and parse_batch_is_readable(self.data_dir, sha256, tier, row)
            ]
            pages = load_pages_from_done_batches(self.data_dir, sha256, tier, tier_rows)
            if not pages:
                continue

            from docvortex.render.markdown import render_single_block

            from ...config import get_config

            delimiters = get_config().render.latex_delimiters
            parts: list[str] = []
            for page in pages:
                for block in page.blocks:
                    rendered = render_single_block(
                        block,
                        delimiters=delimiters,
                        asset_base_url="",
                        image_renderer=lambda _block: "",
                    )
                    if rendered and rendered.strip():
                        parts.append(rendered.strip())
            text = fts_utils.truncate_head_tail("\n\n".join(parts))
            await self.fts.replace(
                sha256=sha256,
                tier=tier,
                text=text,
                title=(doc_row or {}).get("title") or "",
                author=(doc_row or {}).get("author") or "",
            )
            return

        await self.fts.delete(sha256)


# ── tier resolution helpers ────────────────────────────────────────


def _resolve_default_tier(remote: bool = False) -> Tier:
    """从健康检查中选择默认质量 tier，顺序为 standard > basic。"""
    from ..background.parse_server_health import get_health

    health = get_health()
    probe = health.remote.probe if remote else health.local.probe
    supported = probe.tiers
    selected = select_default_quality_tier(supported)
    if selected is not None:
        return selected
    if remote and probe.error_code == "invalid_api_key":
        raise ParseFailure(
            "invalid_api_key",
            _probe_error_message("Remote parse-server authentication failed", probe.error_msg),
            "parse_server.remote.api_key",
        )
    if not remote and health.local_mode == "self_hosted" and probe.error_code == "invalid_api_key":
        raise ParseFailure(
            "invalid_api_key",
            _probe_error_message("Local self-hosted parse-server authentication failed", probe.error_msg),
            "parse_server.local.self_hosted_api_key",
        )
    actions = ["start a local parse-server"]
    if not remote and health.remote.probe.healthy and select_default_quality_tier(health.remote.probe.tiers) is not None:
        actions.append("use --remote")
    actions.append("explicitly pass --tier flash for text-only preview")
    raise ParseFailure(
        "quality_tier_unavailable",
        f"No basic, standard, or advanced engine available. You can {_format_action_list(actions)}.",
    )


def _format_action_list(actions: list[str]) -> str:
    if len(actions) == 1:
        return actions[0]
    if len(actions) == 2:
        return f"{actions[0]} or {actions[1]}"
    return f"{', '.join(actions[:-1])}, or {actions[-1]}"


def _probe_error_message(prefix: str, error_msg: str | None) -> str:
    if error_msg:
        return f"{prefix}: {error_msg}"
    return prefix


def _done_response(sha256: str, short_id: str | None, tier: Tier, page_range: str) -> ParseResponse:
    return ParseResponse(
        sha256=sha256,
        short_id=short_id,
        tier=tier,
        page_range=page_range,
        status=PARSE_STATUS_DONE,
        cache_hit=True,
        wait_parse_ids=[],
        created_parse_ids=[],
        reused_parse_ids=[],
        tip="Cached. Use --force to re-parse.",
    )


def _parse_record_response(row: ParseRow) -> dict:
    error = None
    if row.get("error_code") or row.get("error_msg"):
        error = {"code": row.get("error_code"), "message": row.get("error_msg")}
    return {
        "id": row["id"],
        "sha256": row["sha256"],
        "tier": row["tier"],
        "page_range": normalize_result_page_range(row["page_range"]),
        "status": row["status"],
        "done_at": row.get("done_at"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
        "error": error,
    }


def _remap_api_result_pages_to_page_range(result: ParseResult, page_range: str) -> None:
    """将 API 分页结果的正文与布局几何一起映射回源文档页号。"""
    if not result.pages:
        return
    requested_page_numbers = sorted(parse_page_range_set(page_range))
    actual_page_numbers = [page.page_idx + 1 for page in result.pages]
    if actual_page_numbers == requested_page_numbers:
        return
    if len(requested_page_numbers) != len(result.pages):
        raise ParseFailure(
            "parse_page_remap_failed",
            (
                "Parse result page count does not match requested page_range: "
                f"requested={page_range}, returned={actual_page_numbers}"
            ),
        )
    from docvortex.document.pdf.layout import LAYOUT_EXTENSION, remap_layout_geometry

    page_index_map = [page_no - 1 for page_no in requested_page_numbers]
    extensions = result.middle_json.extensions
    geometry = None
    if LAYOUT_EXTENSION in extensions:
        try:
            geometry = remap_layout_geometry(extensions[LAYOUT_EXTENSION], page_index_map)
        except (ValueError, TypeError, KeyError) as exc:
            raise ParseFailure(
                "parse_page_remap_failed",
                f"Cannot remap PDF layout geometry for page_range={page_range}: {exc}",
            ) from exc

    # 所有可能失败的映射先完成，避免异常时正文与几何只更新一部分。
    for page, page_idx in zip(result.pages, page_index_map, strict=True):
        page.page_idx = page_idx
    if geometry is not None:
        extensions[LAYOUT_EXTENSION] = geometry


def _local_parse_server_url(mode: str, health: object) -> str | None:
    """Resolve local parse-server URL from mode and health state."""
    if mode == "managed":
        return getattr(health, "managed_url", None)
    if mode == "self_hosted":
        return getattr(health, "self_hosted_url", None)
    return None


async def _local_parse_server_api_key(config_svc: ConfigService) -> str | None:
    """Resolve local parse-server API key (self-hosted only; managed servers take no key)."""
    return (await config_svc.get("parse_server.local.self_hosted_api_key")) or None


def _is_official_remote_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    host = host.lower().rstrip(".")
    return host in {"mineru.net", "mineru.org.cn"} or host.endswith(".mineru.net") or host.endswith(".mineru.org.cn")


def _unsupported_file_type_message(ext_or_name: str) -> str:
    display = ext_or_name if ext_or_name.startswith(".") else Path(ext_or_name).name
    return f"File type is not supported for parsing: {display}"
