"""Filesystem watch loop — monitors configured directories for new/changed files."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

from watchfiles import awatch

from ..core.db import DatabaseManager
from ..constants import DISCOVERABLE_EXTENSIONS, is_office_temp_lock_file
from ..services.config_svc import ConfigService
from ..services.parse_svc import ParseService
from ..services.scan_svc import ScanService
from ..types import FILE_STATUS_ACTIVE, WATCH_STATUS_ACTIVE, WATCH_STATUS_UNREACHABLE
from ..utils.path_utils import normalize_doclib_path, rebase_watch_event_path

logger = logging.getLogger("mineru.watch")


class WatchLoop:
    def __init__(
        self,
        db: DatabaseManager,
        config_svc: ConfigService,
        parse_svc: ParseService,
        *,
        scan_interval_sec: int,
        scan_svc: ScanService | None = None,
    ) -> None:
        self.db = db
        self.config_svc = config_svc
        self.parse_svc = parse_svc
        self.scan_svc = scan_svc
        self.scan_interval_sec = scan_interval_sec
        self.running = False
        self._active_watchers: dict[int, asyncio.Task] = {}
        self._wakeup_event = asyncio.Event()

    async def run(self) -> None:
        self.running = True
        while self.running:
            watches = await self.config_svc.list_watches()
            active_ids: set[int] = set()

            for w in watches:
                if w["status"] != WATCH_STATUS_ACTIVE:
                    continue
                active_ids.add(w["id"])

                if w["id"] not in self._active_watchers:
                    task = asyncio.create_task(self._watch_one(w["path"], w["id"]))
                    task.add_done_callback(
                        lambda completed, wid=w["id"], path=w["path"]: self._log_watcher_result(wid, path, completed)
                    )
                    self._active_watchers[w["id"]] = task
                    await asyncio.sleep(0)
                    await self._initial_scan(w["path"], w["id"])

            # cancel watchers for removed watches
            for wid in list(self._active_watchers):
                if wid not in active_ids:
                    self._active_watchers[wid].cancel()
                    task = self._active_watchers.pop(wid)
                    await asyncio.gather(task, return_exceptions=True)

            try:
                await asyncio.wait_for(self._wakeup_event.wait(), timeout=self.scan_interval_sec)
            except asyncio.TimeoutError:
                pass
            self._wakeup_event.clear()

    async def _initial_scan(self, path: str, watch_id: int) -> None:
        """Verify known files, then walk directory tree to discover current files."""
        if self.scan_svc is not None:
            await self.scan_svc.create_scan(path, kind="watch", source="watch", watch_id=watch_id)
            return

        try:
            os.stat(path)
        except OSError:
            await self.config_svc.update_watch_status(watch_id, WATCH_STATUS_UNREACHABLE)
            return

        try:
            rows = await self.db.fetchall(
                "SELECT path FROM files WHERE watch_id=? AND status=?",
                (watch_id, FILE_STATUS_ACTIVE),
            )
            for row in rows:
                await self._refresh_file(row["path"], watch_id)

            file_count = 0
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fname in files:
                    filepath = os.path.join(root, fname)
                    ext = Path(filepath).suffix.lstrip(".").lower()
                    if ext not in DISCOVERABLE_EXTENSIONS or is_office_temp_lock_file(filepath):
                        continue
                    if await self.config_svc.is_path_excluded(filepath):
                        continue
                    await self._refresh_file(filepath, watch_id)
                    file_count += 1
            await self.config_svc.update_watch_scan_stats(watch_id, file_count)
        except Exception as exc:
            logger.error(
                "Watch initial scan failed for watch_id=%s path=%s: %s",
                watch_id,
                path,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _watch_one(self, path: str, watch_id: int) -> None:
        try:
            async for changes in awatch(path, recursive=True, debounce=500):
                if not self.running:
                    break
                for _change_type, filepath in changes:
                    await self._handle_event(filepath, watch_id, watch_root=path)
        except Exception as exc:
            logger.error(
                "Watch loop failed for watch_id=%s path=%s: %s",
                watch_id,
                path,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )

    async def _handle_event(self, filepath: str, watch_id: int, *, watch_root: str | None = None) -> None:
        filepath = (
            rebase_watch_event_path(filepath, watch_root)
            if watch_root is not None
            else normalize_doclib_path(filepath)
        )
        ext = Path(filepath).suffix.lstrip(".").lower()
        if ext not in DISCOVERABLE_EXTENSIONS or is_office_temp_lock_file(filepath):
            return
        if await self.config_svc.is_path_excluded(filepath):
            return
        await self._refresh_file(filepath, watch_id)

    async def _refresh_file(self, filepath: str, watch_id: int) -> None:
        await self.parse_svc.refresh_file(filepath, watch_id=watch_id)

    async def stop(self) -> None:
        self.running = False
        self.wakeup()
        tasks = list(self._active_watchers.values())
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active_watchers.clear()

    def wakeup(self) -> None:
        self._wakeup_event.set()

    @staticmethod
    def _log_watcher_result(watch_id: int, path: str, task: asyncio.Task[None]) -> None:
        if task.cancelled():
            return
        try:
            exc = task.exception()
        except asyncio.CancelledError:
            return
        if exc is not None:
            logger.error(
                "Watch task crashed for watch_id=%s path=%s: %s",
                watch_id,
                path,
                exc,
                exc_info=(type(exc), exc, exc.__traceback__),
            )
