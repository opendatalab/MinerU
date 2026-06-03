"""Registration worker pool — picks up unregistered files and runs register_file()."""

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("mineru.reg_worker")


class RegistrationWorkerPool:
    def __init__(self, parse_svc, num_workers: int = 2) -> None:
        self.parse_svc = parse_svc
        self.num_workers = num_workers
        self.running = False
        self._tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        self.running = True
        self._tasks = [asyncio.create_task(self._worker(i)) for i in range(self.num_workers)]

    async def _worker(self, worker_id: int) -> None:
        logger.info(f"Reg worker {worker_id} started")
        processed = 0
        no_task_count = 0

        while self.running:
            task = await self._acquire_task()
            if task is None:
                no_task_count += 1
                if no_task_count == 1 or no_task_count % 20 == 0:
                    q = await self.parse_svc.db.fetchone(
                        "SELECT COUNT(*) as cnt FROM files WHERE sha256 IS NULL AND scan_status='active'"
                    )
                    logger.info(f"Reg worker {worker_id}: queue={q['cnt'] if q else 0}")
                await asyncio.sleep(0.5)
                continue

            no_task_count = 0
            try:
                await self.parse_svc.register_file(task["path"], watch_id=task.get("watch_id"))
                processed += 1
                if processed % 100 == 0:
                    logger.info(f"Reg worker {worker_id} registered {processed} files")
            except Exception as exc:
                logger.error(f"Reg worker {worker_id} error on {task.get('path')}: {exc}")
                try:
                    now = int(time.time() * 1000)
                    await self.parse_svc.db.execute(
                        "UPDATE files SET sha256=NULL, locked_at=NULL, error_code=?, error_msg=?, "
                        "updated_at=? WHERE id=?",
                        ("reg_failed", str(exc)[:500], now, task["id"]),
                    )
                except Exception:
                    pass

        logger.info(f"Reg worker {worker_id} stopped, processed {processed} total")

    async def _acquire_task(self) -> dict | None:
        now = int(time.time() * 1000)
        timeout = now - 60 * 1000  # 60s reg lock timeout
        return await self.parse_svc.db.fetchone(
            "UPDATE files SET locked_at=? "
            "WHERE id = ("
            "  SELECT id FROM files "
            "  WHERE sha256 IS NULL AND scan_status='active' "
            "  AND (locked_at IS NULL OR locked_at < ?) "
            "  ORDER BY first_seen_at ASC LIMIT 1"
            ") RETURNING *",
            (now, timeout),
        )

    async def stop(self) -> None:
        self.running = False
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
