"""Device monitor — polls removable watch paths for plug/unplug events."""

from __future__ import annotations

import asyncio
import os
import time

from ..core.db import DatabaseManager
from ..services.config_svc import ConfigService
from ..services.scan_svc import ScanService
from ..types import (
    FILE_STATUS_ACTIVE,
    FILE_STATUS_UNREACHABLE,
    SCAN_KIND_WATCH,
    SCAN_SOURCE_SYSTEM,
    WATCH_STATUS_ACTIVE,
    WATCH_STATUS_UNREACHABLE,
)


class DeviceMonitor:
    def __init__(self, db: DatabaseManager, config_svc: ConfigService, *, interval_sec: int, scan_svc: ScanService | None = None) -> None:
        self.db = db
        self.config_svc = config_svc
        self.scan_svc = scan_svc
        self.interval_sec = interval_sec
        self.running = False

    async def run(self) -> None:
        self.running = True
        while self.running:
            await self._poll_once()
            await asyncio.sleep(self.interval_sec)

    async def _poll_once(self) -> None:
        watches = await self.config_svc.get_watches_by_status(WATCH_STATUS_ACTIVE)
        for w in watches:
            if not w["removable"]:
                continue
            try:
                os.stat(w["path"])
            except (FileNotFoundError, OSError):
                await self.config_svc.update_watch_status(w["id"], WATCH_STATUS_UNREACHABLE)
                now = int(time.time() * 1000)
                await self.db.execute(
                    "UPDATE files SET "
                    "status=?, locked_at=NULL, error_code=NULL, error_msg=NULL, deleted_at=NULL, updated_at=? "
                    "WHERE watch_id=? AND status=?",
                    (FILE_STATUS_UNREACHABLE, now, w["id"], FILE_STATUS_ACTIVE),
                )

        unreachable = await self.config_svc.get_watches_by_status(WATCH_STATUS_UNREACHABLE)
        for w in unreachable:
            try:
                os.stat(w["path"])
            except (FileNotFoundError, OSError):
                continue

            await self.config_svc.update_watch_status(w["id"], WATCH_STATUS_ACTIVE)
            now = int(time.time() * 1000)
            await self.db.execute(
                "UPDATE files SET status=?, updated_at=? WHERE watch_id=? AND status=?",
                (FILE_STATUS_ACTIVE, now, w["id"], FILE_STATUS_UNREACHABLE),
            )
            if self.scan_svc is not None:
                await self.scan_svc.create_scan(
                    w["path"],
                    kind=SCAN_KIND_WATCH,
                    source=SCAN_SOURCE_SYSTEM,
                    watch_id=w["id"],
                )

    async def stop(self) -> None:
        self.running = False
