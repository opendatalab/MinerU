"""Doclib 解析请求与队列、查询及失效状态管理。"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from docvortex.options import LatexDelimitersConfig

from ...errors import InvalidRequestError, MineruError
from ...filetypes import (
    PAGE_RANGE_PARSE_EXTENSIONS,
    TEXT_EXTENSIONS,
    TIERED_PARSE_EXTENSIONS,
)
from ...parser.page_range import (
    expand_page_range,
    normalize_page_range_input,
    normalize_result_page_range,
    parse_page_range_set,
)
from ...parser.page_range import (
    format_page_range as _page_numbers_to_range_str,
)
from ...types import QUALITY_TIERS, TIER_ORDER, Tier
from ..core import fts as fts_utils
from ..core.db import DatabaseManager
from ..parse_server_state import ParseServerHealth
from ..core.fts import FTSManager
from ..core.parse_artifacts import (
    load_pages_from_done_batches,
    parse_batch_is_readable,
    parse_coverage,
)
from ..rows import DocRow, FileRow, ParseRow
from ..types import (
    FILE_STATUS_ACTIVE,
    PARSE_STATUS_DONE,
    PARSE_STATUS_PARSING,
    PARSE_STATUS_PENDING,
    PARSE_STATUS_SUPERSEDED,
    ParseResponse,
)
from . import parse_policy
from .ingest_svc import IngestService, raise_refresh_error

if TYPE_CHECKING:
    from ..telemetry import TelemetryService


def _now_ms() -> int:
    """返回当前毫秒时间，保持数据库状态与锁超时的原有单位。"""
    return int(time.time() * 1000)


class ParseQueueService:
    """拥有解析请求、队列、查询和缓存失效状态，不构造执行器。"""

    def __init__(
        self,
        db: DatabaseManager,
        fts: FTSManager,
        data_dir: str,
        *,
        ingest_svc: IngestService,
        health: ParseServerHealth,
        latex_delimiters: LatexDelimitersConfig,
        parse_lock_timeout_sec: int,
        telemetry_svc: TelemetryService | None = None,
    ) -> None:
        """保存此服务所需的显式依赖，不隐式组装另一侧服务。"""
        self.health = health
        self.db = db
        self.fts = fts
        self.data_dir = os.path.expanduser(data_dir)
        self.ingest_svc = ingest_svc
        self.latex_delimiters = latex_delimiters.model_copy(deep=True)
        self.parse_lock_timeout_ms = parse_lock_timeout_sec * 1000
        self.telemetry_svc = telemetry_svc

    async def request_parse(
        self,
        path: str,
        *,
        tier: Tier | None = None,
        page_range: str | None = None,
        force: bool = False,
        remote: bool = False,
    ) -> ParseResponse:
        """校验解析请求，复用已完成及活动页范围，只为缺页创建高优先级任务。"""
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
            requested_tier = tier or parse_policy.resolve_default_tier(health=self.health, remote=remote)
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

    async def get_queue_length(self) -> int:
        """统计未被有效锁占用的待解析批次。"""
        timeout_ms = _now_ms() - self.parse_lock_timeout_ms
        row = await self.db.fetchone(
            "SELECT COUNT(*) as cnt FROM parses WHERE status=? AND (locked_at IS NULL OR locked_at < ?)",
            (PARSE_STATUS_PENDING, timeout_ms),
        )
        return row["cnt"] if row else 0

    async def acquire_task(self) -> ParseRow | None:
        """按优先级与创建时间原子抢占批次，沿用过期锁和开始指标语义。"""
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

    async def get_parse_record(self, parse_id: int) -> dict[str, Any] | None:
        """查询单个解析记录，并保持历史页范围的读时规范化。"""
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
    ) -> dict[str, Any]:
        """按请求条件查询解析记录，保留状态过滤、调用方顺序和覆盖信息。"""
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
        """将已完成批次标记为失效，再从剩余缓存重建索引。"""
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

    async def _rebuild_fts_after_invalidate(self, sha256: str) -> None:
        # FTS rebuild reads only cached parse batches, so any active row works;
        # keep it deterministic and tolerate rows whose file moved away.
        """按档位和缓存可读性重建正文索引，不修改历史缓存或源文件。"""
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

            delimiters = self.latex_delimiters
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

    async def _record_count(self, metric_name: str, *, value: int = 1, dimensions: dict[str, str] | None = None) -> None:
        """通过显式指标接口记录计数，未配置时跳过。"""
        if self.telemetry_svc is not None:
            await self.telemetry_svc.record_count(metric_name, value=value, dimensions=dimensions)


def _done_response(sha256: str, short_id: str | None, tier: Tier, page_range: str) -> ParseResponse:
    """生成既有缓存命中回执，不创建额外解析任务。"""
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


def _parse_record_response(row: ParseRow) -> dict[str, Any]:
    """将数据库记录转换为稳定查询字段，规范化历史页范围。"""
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


def _unsupported_file_type_message(ext_or_name: str) -> str:
    """沿用不支持文件类型的路径显示与错误文案。"""
    display = ext_or_name if ext_or_name.startswith(".") else Path(ext_or_name).name
    return f"File type is not supported for parsing: {display}"


__all__ = ["ParseQueueService"]
