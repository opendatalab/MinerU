"""Cleanup service: orphan docs, deleted files, temp files."""

from __future__ import annotations

import os
import shutil
import time
from typing import cast

from ...errors import InvalidRequestError
from ..core.db import DatabaseManager
from ..rows import CountRow, DocRow, IdRow, WatchTargetRow
from ..types import FILE_STATUS_DELETED
from ..utils.path_utils import normalize_doclib_path

FORGET_WATCH_ROOT_WARNING = "Path is a configured watch root and may be rediscovered on the next scan."
FORGET_UNDER_ACTIVE_WATCH_WARNING = "Path is under an active watch and may be rediscovered on the next scan."


class CleanupService:
    def __init__(self, db: DatabaseManager, data_dir: str) -> None:
        self.db = db
        self.data_dir = os.path.expanduser(data_dir)

    # ── orphan docs ─────────────────────────────────────────────

    async def find_orphan_docs(self) -> list[DocRow]:
        return cast(
            list[DocRow],
            await self.db.fetchall(
                "SELECT d.* FROM docs d WHERE NOT EXISTS (  SELECT 1 FROM files f WHERE f.sha256 = d.sha256)"
            ),
        )

    async def cleanup_orphans(self, dry_run: bool = True) -> int:
        orphans = await self.find_orphan_docs()
        if dry_run:
            return len(orphans)

        count = 0
        for doc in orphans:
            sha256 = doc["sha256"]
            await self.db.execute("DELETE FROM fts_contents WHERE sha256=?", (sha256,))
            await self.db.execute("DELETE FROM parses WHERE sha256=?", (sha256,))
            await self.db.execute("DELETE FROM docs WHERE sha256=?", (sha256,))
            # remove parsed output dir
            parsed_dir = os.path.join(self.data_dir, "parsed", sha256[:2], sha256)
            if os.path.isdir(parsed_dir):
                shutil.rmtree(parsed_dir, ignore_errors=True)
            count += 1
        await self.db.commit()
        return count

    # ── deleted files ───────────────────────────────────────────

    async def cleanup_deleted(self, dry_run: bool = True) -> int:
        row = cast(
            CountRow | None,
            await self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM files WHERE status=?",
                (FILE_STATUS_DELETED,),
            ),
        )
        count = row["cnt"] if row else 0

        if not dry_run and count > 0:
            await self.db.execute(
                "DELETE FROM fts_filenames WHERE file_id IN (SELECT id FROM files WHERE status=?)",
                (FILE_STATUS_DELETED,),
            )
            await self.db.execute(
                "DELETE FROM files WHERE status=?",
                (FILE_STATUS_DELETED,),
            )
            await self.db.commit()

        return count

    async def cleanup_deleted_older_than(self, older_than_days: int = 7) -> int:
        threshold = _days_ago_ms(older_than_days)
        row = cast(
            CountRow | None,
            await self.db.fetchone(
                "SELECT COUNT(*) as cnt FROM files WHERE status=? AND deleted_at < ?",
                (FILE_STATUS_DELETED, threshold),
            ),
        )
        count = row["cnt"] if row else 0

        if count > 0:
            await self.db.execute(
                "DELETE FROM fts_filenames WHERE file_id IN (SELECT id FROM files WHERE status=? AND deleted_at < ?)",
                (FILE_STATUS_DELETED, threshold),
            )
            await self.db.execute(
                "DELETE FROM files WHERE status=? AND deleted_at < ?",
                (FILE_STATUS_DELETED, threshold),
            )
            await self.db.commit()

        return count

    # ── forget path ─────────────────────────────────────────────

    async def forget_path(self, path: str, *, dry_run: bool = True) -> dict:
        normalized = _normalize_path(path)
        if not normalized:
            raise InvalidRequestError("invalid_request", "Path is required.", "path")

        watches = cast(list[WatchTargetRow], await self.db.fetchall("SELECT * FROM watches WHERE enabled=1"))

        file_ids = await self._forget_file_ids(normalized)
        if file_ids and _is_watch_root(normalized, watches):
            raise InvalidRequestError(
                "invalid_request",
                "Cannot forget a configured watch root while it contains tracked files."
                " Remove the watch first or forget a child path.",
                "path",
            )
        matched_as = await self._forget_matched_as(normalized, file_ids)
        warnings = _forget_warnings(normalized, watches)

        if not dry_run and file_ids:
            placeholders = ",".join("?" * len(file_ids))
            await self.db.execute(
                f"DELETE FROM fts_filenames WHERE file_id IN ({placeholders})",
                tuple(file_ids),
            )
            await self.db.execute(
                f"DELETE FROM files WHERE id IN ({placeholders})",
                tuple(file_ids),
            )
            await self.db.commit()

        return {
            "path": normalized,
            "matched_as": matched_as,
            "forgotten_files": len(file_ids),
            "dry_run": dry_run,
            "warnings": warnings,
        }

    async def _forget_file_ids(self, path: str) -> list[int]:
        prefix = _path_prefix(path)
        upper = prefix + "\U0010ffff"
        rows = cast(
            list[IdRow],
            await self.db.fetchall(
                "SELECT id FROM files WHERE path=? OR (path>=? AND path<?) ORDER BY path",
                (path, prefix, upper),
            ),
        )
        return [row["id"] for row in rows]

    async def _forget_matched_as(self, path: str, file_ids: list[int]) -> str:
        if not file_ids:
            return "none"
        if os.path.isdir(path):
            return "directory"
        prefix = _path_prefix(path)
        row = await self.db.fetchone("SELECT 1 FROM files WHERE path>=? AND path<? LIMIT 1", (prefix, prefix + "\U0010ffff"))
        if row is not None:
            return "directory"
        return "file"

    # ── temp files ──────────────────────────────────────────────

    async def cleanup_temp_files(self, older_than_days: int = 7) -> int:
        """Remove process temp files (e.g. incomplete parse artifacts)."""
        if older_than_days < 0:
            raise InvalidRequestError("invalid_request", "older_than_days must be non-negative.", "older_than_days")

        temp_dir = os.path.join(self.data_dir, "temp")
        if not os.path.isdir(temp_dir):
            return 0

        threshold = _days_ago_ms(older_than_days) / 1000.0
        count = 0
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                fp = os.path.join(root, name)
                try:
                    if os.path.getmtime(fp) < threshold:
                        os.remove(fp)
                        count += 1
                except OSError:
                    pass
            for name in dirs:
                dp = os.path.join(root, name)
                try:
                    if not os.listdir(dp):
                        os.rmdir(dp)
                except OSError:
                    pass
        return count


def _days_ago_ms(days: int) -> int:
    return int((time.time() - days * 86400) * 1000)


def _normalize_path(path: str) -> str:
    return normalize_doclib_path(path)


def _path_prefix(path: str) -> str:
    return path if path.endswith(os.sep) else path + os.sep


def _is_watch_root(path: str, watches: list[WatchTargetRow]) -> bool:
    return any(watch["path"] == path for watch in watches)


def _active_watch_warning(path: str, watches: list[WatchTargetRow]) -> str | None:
    for watch in watches:
        watch_path = watch["path"]
        if watch.get("status") != "active":
            continue
        if path != watch_path and path.startswith(_path_prefix(watch_path)):
            return FORGET_UNDER_ACTIVE_WATCH_WARNING
    return None


def _is_watch_root(path: str, watches: list[WatchTargetRow]) -> bool:
    """判断路径是否为 watch root，用于有实际匹配文件时保护扫描入口。"""
    for watch in watches:
        if _normalize_path(watch["path"]) == path:
            return True
    return False


def _forget_warnings(path: str, watches: list[WatchTargetRow]) -> list[str]:
    for watch in watches:
        if watch["path"] == path:
            return [FORGET_WATCH_ROOT_WARNING]
    active_watch_warning = _active_watch_warning(path, watches)
    return [active_watch_warning] if active_watch_warning else []
