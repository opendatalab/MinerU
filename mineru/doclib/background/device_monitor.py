"""Device monitor — polls removable watch paths for plug/unplug events."""

from __future__ import annotations

import asyncio
import os

from ..core.db import DatabaseManager
from ..services.config_svc import ConfigService


class DeviceMonitor:
    def __init__(self, db: DatabaseManager, config_svc: ConfigService) -> None:
        self.db = db
        self.config_svc = config_svc
        self.running = False

    async def run(self) -> None:
        self.running = True
        while self.running:
            watches = await self.config_svc.get_watches_by_status("active")
            for w in watches:
                if not w["removable"]:
                    continue
                try:
                    os.stat(w["path"])
                except (FileNotFoundError, OSError):
                    await self.config_svc.update_watch_status(w["id"], "unreachable")
                    await self.db.execute(
                        "UPDATE files SET scan_status='unreachable' "
                        "WHERE watch_id=? AND scan_status='active'",
                        (w["id"],),
                    )

            unreachable = await self.config_svc.get_watches_by_status("unreachable")
            for w in unreachable:
                try:
                    os.stat(w["path"])
                    await self.config_svc.update_watch_status(w["id"], "active")
                    await self.db.execute(
                        "UPDATE files SET scan_status='active' "
                        "WHERE watch_id=? AND scan_status='unreachable'",
                        (w["id"],),
                    )
                except (FileNotFoundError, OSError):
                    pass

            await asyncio.sleep(5)

    async def stop(self) -> None:
        self.running = False
