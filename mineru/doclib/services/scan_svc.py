"""Scan service: create and execute file refresh tasks."""

from __future__ import annotations

import os
import time
import logging
from pathlib import Path
from typing import Any, cast

from ...errors import InvalidRequestError, NotFoundError
from ...filetypes import DISCOVERABLE_EXTENSIONS, is_office_temp_lock_file
from ..core.file_io import get_file_stat
from ..rows import PathRow, ScanRow, WatchTargetRow
from ..types import (
    SCAN_KIND_MANUAL,
    SCAN_KIND_WATCH,
    FILE_STATUS_ACTIVE,
    SCAN_STATUS_DONE,
    SCAN_STATUS_FAILED,
    SCAN_STATUS_PENDING,
    SCAN_STATUS_RUNNING,
    SCAN_SOURCE_UNKNOWN,
    WATCH_STATUS_ACTIVE,
    WATCH_STATUS_UNREACHABLE,
    ScanInfo,
    ScanKind,
    ScanSource,
    ScanStatus,
)
from ..core.db import DatabaseManager
from ..utils.path_utils import normalize_doclib_path
from .config_svc import ConfigService
from .parse_svc import FileRefreshResult, ParseService

logger = logging.getLogger("mineru.scan")

SCAN_LOG_RETENTION_LIMIT = 1000


def _now_ms() -> int:
    return int(time.time() * 1000)


def _scan_filters(
    *,
    status: ScanStatus | None,
    kind: ScanKind | None,
    watch_id: int | None,
) -> tuple[list[str], list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    if status is not None:
        clauses.append("status=?")
        params.append(status)
    if kind is not None:
        clauses.append("kind=?")
        params.append(kind)
    if watch_id is not None:
        clauses.append("watch_id=?")
        params.append(watch_id)
    return clauses, params


class ScanService:
    """Create, query, and execute background scan tasks."""

    def __init__(
        self,
        db: DatabaseManager,
        config_svc: ConfigService,
        parse_svc: ParseService,
        *,
        scan_lock_timeout_sec: int,
        telemetry_svc: object | None = None,
    ) -> None:
        self.db = db
        self.config_svc = config_svc
        self.parse_svc = parse_svc
        self.scan_lock_timeout_ms = scan_lock_timeout_sec * 1000
        self.telemetry_svc = telemetry_svc

    async def create_scan(
        self,
        path: str,
        *,
        kind: ScanKind = SCAN_KIND_MANUAL,
        source: ScanSource = SCAN_SOURCE_UNKNOWN,
        watch_id: int | None = None,
    ) -> ScanInfo:
        normalized_path = normalize_doclib_path(path)
        if kind == SCAN_KIND_WATCH:
            watch = await self._resolve_watch(normalized_path, watch_id)
            normalized_path = watch["path"]
            watch_id = watch["id"]
        elif watch_id is not None:
            raise InvalidRequestError("invalid_request", "watch_id is only valid for watch scans.", "watch_id")

        await self._fail_stale_running_scans(kind=kind, path=normalized_path)

        active = cast(
            ScanRow | None,
            await self.db.fetchone(
                "SELECT * FROM scans WHERE kind=? AND path=? AND status IN (?, ?) ORDER BY created_at ASC LIMIT 1",
                (kind, normalized_path, SCAN_STATUS_PENDING, SCAN_STATUS_RUNNING),
            ),
        )
        if active is not None:
            await self._record_count("scan.reuse.count", dimensions={"status": "reused"})
            return _scan_info(active)

        if source == "watch":
            await self._record_count("scan.request.count", dimensions={"source": "watch", "caller": "system"})

        now = _now_ms()
        scan_id = await self.db.execute_insert(
            "INSERT INTO scans (path, kind, source, watch_id, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (normalized_path, kind, source, watch_id, SCAN_STATUS_PENDING, now, now),
        )
        row = cast(ScanRow | None, await self.db.fetchone("SELECT * FROM scans WHERE id=?", (scan_id,)))
        if row is None:
            raise NotFoundError("scan_not_found", f"Scan {scan_id} not found after creation.", "scan_id")
        return _scan_info(row)

    async def list_scans(
        self,
        *,
        limit: int = 50,
        status: ScanStatus | None = None,
        kind: ScanKind | None = None,
        watch_id: int | None = None,
        offset: int = 0,
    ) -> list[ScanInfo]:
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        clauses, params = _scan_filters(status=status, kind=kind, watch_id=watch_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = cast(
            list[ScanRow],
            await self.db.fetchall(
                f"SELECT * FROM scans {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (*params, limit, offset),
            ),
        )
        return [_scan_info(row) for row in rows]

    async def count_scans(
        self,
        *,
        status: ScanStatus | None = None,
        kind: ScanKind | None = None,
        watch_id: int | None = None,
    ) -> int:
        clauses, params = _scan_filters(status=status, kind=kind, watch_id=watch_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        row = await self.db.fetchone(f"SELECT COUNT(*) AS cnt FROM scans {where}", tuple(params))
        return row["cnt"] if row else 0

    async def get_scan(self, scan_id: int) -> ScanInfo:
        row = cast(ScanRow | None, await self.db.fetchone("SELECT * FROM scans WHERE id=?", (scan_id,)))
        if row is None:
            raise NotFoundError("scan_not_found", f"Scan {scan_id} not found.", "scan_id")
        return _scan_info(row)

    async def acquire_task(self) -> ScanRow | None:
        now = _now_ms()
        timeout = now - self.scan_lock_timeout_ms
        return cast(
            ScanRow | None,
            await self.db.fetchone(
                "UPDATE scans SET locked_at=?, status=?, started_at=COALESCE(started_at, ?), updated_at=? "
                "WHERE id = ("
                "  SELECT id FROM scans WHERE status=? AND (locked_at IS NULL OR locked_at < ?) "
                "  ORDER BY created_at ASC LIMIT 1"
                ") RETURNING *",
                (now, SCAN_STATUS_RUNNING, now, now, SCAN_STATUS_PENDING, timeout),
            ),
        )

    async def process_scan(self, task: ScanRow) -> bool:
        start_ms = _now_ms()
        try:
            counters = await self._run_scan(task)
        except Exception as exc:
            now = _now_ms()
            await self.db.execute(
                "UPDATE scans SET status=?, locked_at=NULL, error_code=?, error_msg=?, finished_at=?, updated_at=? WHERE id=?",
                (SCAN_STATUS_FAILED, "scan_failed", str(exc)[:500], now, now, task["id"]),
            )
            await self.cleanup_terminal_scan_logs()
            await self._record_count("scan.finished.count", dimensions={"status": "failed"})
            await self._record_duration("scan.duration_bucket.count", start_ms, dimensions={"status": "failed"})
            return False

        now = _now_ms()
        await self.db.execute(
            "UPDATE scans SET status=?, locked_at=NULL, files_seen=?, files_refreshed=?, files_new=?, files_changed=?, "
            "files_deleted=?, files_unreachable=?, files_error=?, files_unsupported=?, files_excluded=?, "
            "finished_at=?, updated_at=? WHERE id=?",
            (
                SCAN_STATUS_DONE,
                counters.files_seen,
                counters.files_refreshed,
                counters.files_new,
                counters.files_changed,
                counters.files_deleted,
                counters.files_unreachable,
                counters.files_error,
                counters.files_unsupported,
                counters.files_excluded,
                now,
                now,
                task["id"],
            ),
        )
        task_watch_id = task["watch_id"]
        if task["kind"] == SCAN_KIND_WATCH and task_watch_id is not None:
            await self.config_svc.update_watch_scan_stats(task_watch_id, counters.files_refreshed)
        await self.cleanup_terminal_scan_logs()
        await self._record_count("scan.finished.count", dimensions={"status": "succeeded"})
        await self._record_duration("scan.duration_bucket.count", start_ms, dimensions={"status": "succeeded"})
        await self._record_scan_file_counts(counters)
        return True

    async def _fail_stale_running_scans(self, *, kind: ScanKind, path: str) -> None:
        now = _now_ms()
        timeout = now - self.scan_lock_timeout_ms
        await self.db.execute(
            "UPDATE scans SET status=?, locked_at=NULL, error_code=COALESCE(error_code, ?), "
            "error_msg=COALESCE(error_msg, ?), finished_at=COALESCE(finished_at, ?), updated_at=? "
            "WHERE kind=? AND path=? AND status=? AND locked_at IS NOT NULL AND locked_at < ?",
            (
                SCAN_STATUS_FAILED,
                "scan_interrupted",
                "Scan lock expired before completion.",
                now,
                now,
                kind,
                path,
                SCAN_STATUS_RUNNING,
                timeout,
            ),
        )

    async def cleanup_terminal_scan_logs(self, *, keep: int = SCAN_LOG_RETENTION_LIMIT) -> None:
        try:
            await self.db.execute(
                "DELETE FROM scans WHERE status IN (?, ?) AND id NOT IN ("
                "  SELECT id FROM scans WHERE status IN (?, ?) ORDER BY created_at DESC, id DESC LIMIT ?"
                ")",
                (SCAN_STATUS_DONE, SCAN_STATUS_FAILED, SCAN_STATUS_DONE, SCAN_STATUS_FAILED, keep),
            )
        except Exception as exc:
            logger.warning("Failed to cleanup terminal scan logs: %s", exc)

    async def _run_scan(self, task: ScanRow) -> "_ScanCounters":
        path = normalize_doclib_path(task["path"])
        kind = task["kind"]
        watch_id = task.get("watch_id")
        counters = _ScanCounters()

        if kind == SCAN_KIND_WATCH:
            watch = await self._resolve_watch(path, watch_id)
            watch_id = watch["id"]
            if not watch["enabled"]:
                raise InvalidRequestError("invalid_request", "Watch target is disabled.", "watch_id")
            try:
                await get_file_stat(watch["path"])
            except OSError:
                await self.config_svc.update_watch_status(watch["id"], WATCH_STATUS_UNREACHABLE)
                return counters
            if watch["status"] == WATCH_STATUS_UNREACHABLE:
                await self.config_svc.update_watch_status(watch["id"], WATCH_STATUS_ACTIVE)

        if os.path.isdir(path):
            await self._scan_directory(path, watch_id=watch_id, apply_excludes=True, counters=counters)
            return counters

        if os.path.exists(path):
            await self._scan_file(path, watch_id=watch_id, apply_excludes=False, counters=counters)
            return counters

        await self._scan_missing_path(path, watch_id=watch_id, counters=counters)
        return counters

    async def _scan_directory(
        self,
        path: str,
        *,
        watch_id: int | None,
        apply_excludes: bool,
        counters: "_ScanCounters",
    ) -> None:
        await self._refresh_known_active_files_under(path, watch_id=watch_id, counters=counters)
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                filepath = os.path.join(root, fname)
                counters.files_seen += 1
                ext = Path(filepath).suffix.lstrip(".").lower()
                if ext not in DISCOVERABLE_EXTENSIONS or is_office_temp_lock_file(filepath):
                    counters.files_unsupported += 1
                    continue
                if apply_excludes and await self.config_svc.is_path_excluded(filepath):
                    counters.files_excluded += 1
                    continue
                await self._refresh_file(filepath, watch_id=watch_id, counters=counters)

    async def _scan_file(
        self,
        path: str,
        *,
        watch_id: int | None,
        apply_excludes: bool,
        counters: "_ScanCounters",
    ) -> None:
        counters.files_seen += 1
        if apply_excludes and await self.config_svc.is_path_excluded(path):
            counters.files_excluded += 1
            return
        await self._refresh_file(path, watch_id=watch_id, counters=counters)

    async def _scan_missing_path(self, path: str, *, watch_id: int | None, counters: "_ScanCounters") -> None:
        existing = cast(PathRow | None, await self.db.fetchone("SELECT path FROM files WHERE path=?", (path,)))
        if existing is not None:
            await self._refresh_file(existing["path"], watch_id=watch_id, counters=counters)

        prefix = _dir_prefix(path)
        rows = cast(
            list[PathRow],
            await self.db.fetchall(
                "SELECT path FROM files WHERE path>=? AND path<? AND status=? ORDER BY path",
                (prefix, prefix + chr(0x10FFFF), FILE_STATUS_ACTIVE),
            ),
        )
        for row in rows:
            await self._refresh_file(row["path"], watch_id=watch_id, counters=counters)

    async def _refresh_known_active_files_under(
        self,
        path: str,
        *,
        watch_id: int | None,
        counters: "_ScanCounters",
    ) -> None:
        prefix = _dir_prefix(path)
        params: tuple[Any, ...]
        if watch_id is None:
            sql = "SELECT path FROM files WHERE path>=? AND path<? AND status=? ORDER BY path"
            params = (prefix, prefix + chr(0x10FFFF), FILE_STATUS_ACTIVE)
        else:
            sql = "SELECT path FROM files WHERE watch_id=? AND status=? ORDER BY path"
            params = (watch_id, FILE_STATUS_ACTIVE)
        rows = cast(list[PathRow], await self.db.fetchall(sql, params))
        for row in rows:
            await self._refresh_file(row["path"], watch_id=watch_id, counters=counters)

    async def _refresh_file(self, path: str, *, watch_id: int | None, counters: "_ScanCounters") -> FileRefreshResult:
        result = await self.parse_svc.refresh_file(path, watch_id=watch_id)
        counters.add_refresh(result.status)
        return result

    async def _resolve_watch(self, path: str, watch_id: int | None) -> WatchTargetRow:
        if watch_id is not None:
            watch = cast(WatchTargetRow | None, await self.db.fetchone("SELECT * FROM watches WHERE id=?", (watch_id,)))
        else:
            watch = cast(WatchTargetRow | None, await self.db.fetchone("SELECT * FROM watches WHERE path=?", (path,)))
        if watch is None:
            raise NotFoundError("watch_not_found", "Watch target not found.", "watch_id")
        return watch

    async def _record_scan_file_counts(self, counters: "_ScanCounters") -> None:
        values = {
            "seen": counters.files_seen,
            "refreshed": counters.files_refreshed,
            "new": counters.files_new,
            "changed": counters.files_changed,
            "deleted": counters.files_deleted,
            "unreachable": counters.files_unreachable,
            "error": counters.files_error,
            "unsupported": counters.files_unsupported,
            "excluded": counters.files_excluded,
        }
        for result, value in values.items():
            if value:
                await self._record_count("scan.files.count", value=value, dimensions={"result": result})

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


class _ScanCounters:
    def __init__(self) -> None:
        self.files_seen = 0
        self.files_refreshed = 0
        self.files_new = 0
        self.files_changed = 0
        self.files_deleted = 0
        self.files_unreachable = 0
        self.files_error = 0
        self.files_unsupported = 0
        self.files_excluded = 0

    def add_refresh(self, status: str) -> None:
        self.files_refreshed += 1
        if status == "new":
            self.files_new += 1
        elif status == "changed":
            self.files_changed += 1
        elif status == "deleted":
            self.files_deleted += 1
        elif status == "unreachable":
            self.files_unreachable += 1
        elif status == "unsupported":
            self.files_unsupported += 1
        elif status == "error":
            self.files_error += 1


def _dir_prefix(path: str) -> str:
    return path.rstrip(os.sep) + os.sep


def _scan_info(row: ScanRow) -> ScanInfo:
    return ScanInfo.model_validate(row)
