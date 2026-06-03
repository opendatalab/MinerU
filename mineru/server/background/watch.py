"""Filesystem watch loop — monitors configured directories for new/changed files."""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

from watchfiles import awatch

from mineru.constants import ALLOWED_EXTENSIONS


class WatchLoop:
    def __init__(self, db, config_svc, parse_svc) -> None:
        self.db = db
        self.config_svc = config_svc
        self.parse_svc = parse_svc
        self.running = False
        self._active_watchers: dict[int, asyncio.Task] = {}

    async def run(self) -> None:
        self.running = True
        while self.running:
            watches = await self.config_svc.list_watches()
            active_ids: set[int] = set()

            for w in watches:
                if w["watch_status"] != "active":
                    continue
                active_ids.add(w["id"])

                if w["id"] not in self._active_watchers:
                    await self._initial_scan(w["path"], w["id"])
                    task = asyncio.create_task(self._watch_one(w["path"], w["id"]))
                    self._active_watchers[w["id"]] = task

            # cancel watchers for removed watches
            for wid in list(self._active_watchers):
                if wid not in active_ids:
                    self._active_watchers[wid].cancel()
                    del self._active_watchers[wid]

            await asyncio.sleep(30)

    async def _initial_scan(self, path: str, watch_id: int) -> None:
        """Walk directory tree and discover all existing files."""
        try:
            file_count = 0
            for root, dirs, files in os.walk(path):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for fname in files:
                    filepath = os.path.join(root, fname)
                    ext = Path(filepath).suffix.lstrip(".").lower()
                    if ext not in ALLOWED_EXTENSIONS:
                        continue
                    if await self.config_svc.is_path_excluded(filepath):
                        continue
                    await self._discover_file(filepath, watch_id)
                    file_count += 1
            await self.config_svc.update_watch_scan_stats(watch_id, file_count)
        except Exception:
            pass

    async def _watch_one(self, path: str, watch_id: int) -> None:
        try:
            async for changes in awatch(path, recursive=True, debounce=500):
                if not self.running:
                    break
                for _change_type, filepath in changes:
                    await self._handle_event(filepath, watch_id)
        except Exception:
            pass

    async def _handle_event(self, filepath: str, watch_id: int) -> None:
        ext = Path(filepath).suffix.lstrip(".").lower()
        if ext not in ALLOWED_EXTENSIONS:
            return
        if await self.config_svc.is_path_excluded(filepath):
            return
        await self._discover_file(filepath, watch_id)

    async def _discover_file(self, filepath: str, watch_id: int) -> None:
        now = int(time.time() * 1000)

        await self.db.execute(
            "INSERT OR IGNORE INTO files (path, filename, ext, size_bytes, mtime_ms, "
            "watch_id, first_seen_at, updated_at) VALUES (?, ?, ?, 0, 0, ?, ?, ?)",
            (
                filepath,
                Path(filepath).name,
                Path(filepath).suffix.lstrip(".").lower(),
                watch_id,
                now,
                now,
            ),
        )

    async def stop(self) -> None:
        self.running = False
        for t in self._active_watchers.values():
            t.cancel()
