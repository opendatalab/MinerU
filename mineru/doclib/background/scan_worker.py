"""Scan worker pool — executes queued filesystem scan tasks."""

from __future__ import annotations

import asyncio
import logging

from ..services.scan_svc import ScanService

logger = logging.getLogger("mineru.scan_worker")


class ScanWorkerPool:
    def __init__(self, scan_svc: ScanService, num_workers: int = 1) -> None:
        self.scan_svc = scan_svc
        self.num_workers = num_workers
        self.running = False
        self._tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        self.running = True
        self._tasks = [asyncio.create_task(self._worker(i)) for i in range(self.num_workers)]

    async def _worker(self, worker_id: int) -> None:
        logger.info("Scan worker %s started", worker_id)
        processed = 0
        while self.running:
            task = await self.scan_svc.acquire_task()
            if task is None:
                await asyncio.sleep(0.5)
                continue
            try:
                success = await self.scan_svc.process_scan(task)
                if success:
                    processed += 1
            except Exception as exc:
                logger.error("Scan worker %s error on scan %s: %s", worker_id, task.get("id"), exc)
        logger.info("Scan worker %s stopped, processed %s total", worker_id, processed)

    async def stop(self) -> None:
        self.running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
