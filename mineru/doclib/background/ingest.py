"""Ingest worker pool — picks up files not yet ingested and runs ingest_file()."""

from __future__ import annotations

import asyncio
import logging
import time
from typing import cast

from ..rows import CountRow, FileRow, IngestTaskRow
from ..services.parse_svc import ParseService
from ..types import FILE_STATUS_ACTIVE

logger = logging.getLogger("mineru.ingest")


class IngestWorkerPool:
    def __init__(self, parse_svc: ParseService, *, num_workers: int, lock_timeout_sec: int) -> None:
        self.parse_svc = parse_svc
        self.num_workers = num_workers
        self.lock_timeout_ms = lock_timeout_sec * 1000
        self.running = False
        self._tasks: list[asyncio.Task] = []

    async def run(self) -> None:
        self.running = True
        self._tasks = [asyncio.create_task(self._worker(i)) for i in range(self.num_workers)]

    async def _worker(self, worker_id: int) -> None:
        logger.info(f"Ingest worker {worker_id} started")
        processed = 0
        no_task_count = 0

        while self.running:
            task = await self._acquire_task()
            if task is None:
                no_task_count += 1
                if no_task_count == 1 or no_task_count % 20 == 0:
                    q = cast(
                        CountRow | None,
                        await self.parse_svc.db.fetchone(
                            "SELECT COUNT(*) as cnt FROM files WHERE sha256 IS NULL AND status=? AND error_code IS NULL",
                            (FILE_STATUS_ACTIVE,),
                        ),
                    )
                    logger.debug(f"Ingest worker {worker_id}: queue={q['cnt'] if q else 0}")
                await asyncio.sleep(0.5)
                continue

            no_task_count = 0
            try:
                await self.parse_svc.ingest_file(task["path"], watch_id=task.get("watch_id"))
                processed += 1
                if processed % 100 == 0:
                    logger.info(f"Ingest worker {worker_id} ingested {processed} files")
            except Exception as exc:
                logger.error(f"Ingest worker {worker_id} error on {task.get('path')}: {exc}")
                await self._handle_ingest_error(task, exc)

        logger.info(f"Ingest worker {worker_id} stopped, processed {processed} total")

    async def _acquire_task(self) -> FileRow | None:
        now = int(time.time() * 1000)
        timeout = now - self.lock_timeout_ms
        return cast(
            FileRow | None,
            await self.parse_svc.db.fetchone(
                "UPDATE files SET locked_at=? "
                "WHERE id = ("
                "  SELECT id FROM files "
                "  WHERE sha256 IS NULL AND status=? "
                "  AND error_code IS NULL "
                "  AND (locked_at IS NULL OR locked_at < ?) "
                "  ORDER BY first_seen_at ASC LIMIT 1"
                ") RETURNING *",
                (now, FILE_STATUS_ACTIVE, timeout),
            ),
        )

    async def _handle_ingest_error(self, task: IngestTaskRow, exc: Exception) -> None:
        try:
            if isinstance(exc, FileNotFoundError):
                watch_id = task.get("watch_id")
                await self.parse_svc.refresh_file(
                    str(task["path"]),
                    watch_id=watch_id if isinstance(watch_id, int) else None,
                )
                return

            error_code = "ingest_failed"
            if isinstance(exc, PermissionError):
                error_code = "file_permission_denied"
            elif isinstance(exc, OSError):
                error_code = "stat_failed"

            now = int(time.time() * 1000)
            await self.parse_svc.db.execute(
                "UPDATE files SET sha256=NULL, locked_at=NULL, error_code=?, error_msg=?, updated_at=? WHERE id=?",
                (error_code, str(exc)[:500], now, task["id"]),
            )
        except Exception:
            pass

    async def stop(self) -> None:
        self.running = False
        for t in self._tasks:
            t.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
