"""Background telemetry flush loop."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ..telemetry.constants import TELEMETRY_FLUSH_INTERVAL_SEC

logger = logging.getLogger("mineru.doclib.telemetry")


class TelemetryFlushLoop:
    def __init__(self, telemetry_svc: Any, *, interval_sec: int = TELEMETRY_FLUSH_INTERVAL_SEC) -> None:
        self.telemetry_svc = telemetry_svc
        self.interval_sec = interval_sec
        self.running = False
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        self.running = True
        try:
            while not self._stop_event.is_set():
                try:
                    await self.telemetry_svc.flush_once()
                except Exception as exc:
                    logger.debug("Telemetry flush loop failed: %s", exc)
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=self.interval_sec)
                except TimeoutError:
                    continue
        finally:
            self.running = False

    async def stop(self) -> None:
        self._stop_event.set()


__all__ = ["TelemetryFlushLoop"]
