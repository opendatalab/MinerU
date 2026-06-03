"""Parse worker pool — picks up pending parse batches and executes engine."""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("mineru.parse_worker")


class ParseWorkerPool:
    def __init__(self, parse_svc, num_workers: int = 2) -> None:
        self.parse_svc = parse_svc
        self.num_workers = num_workers
        self.running = False
        self._tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        self.running = True
        self._tasks = [asyncio.create_task(self._worker(i)) for i in range(self.num_workers)]

    async def _worker(self, worker_id: int) -> None:
        logger.info(f"Parse worker {worker_id} started")
        processed = 0
        no_task_count = 0

        while self.running:
            task = await self.parse_svc.acquire_task()
            if task is None:
                no_task_count += 1
                if no_task_count == 1 or no_task_count % 20 == 0:
                    q = await self.parse_svc.get_queue_length()
                    logger.info(f"Parse worker {worker_id}: queue={q}")
                await asyncio.sleep(0.5)
                continue

            no_task_count = 0
            try:
                success = await self.parse_svc.process_doc(task)
                if success:
                    processed += 1
                    if processed % 50 == 0:
                        logger.info(f"Parse worker {worker_id} processed {processed} batches")
            except Exception as exc:
                logger.error(f"Parse worker {worker_id} error on {task.get('sha256')}: {exc}")
                try:
                    now = int(time.time() * 1000)
                    await self.parse_svc.db.execute(
                        "UPDATE parses SET status='failed', error_code=?, error_msg=?, "
                        "locked_at=NULL, updated_at=? WHERE id=?",
                        ("parse_failed", str(exc)[:500], now, task["id"]),
                    )
                except Exception:
                    pass

        logger.info(f"Parse worker {worker_id} stopped, processed {processed} total")

    async def stop(self) -> None:
        self.running = False
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
