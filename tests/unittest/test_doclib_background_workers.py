import asyncio
import sqlite3

from mineru.doclib.background.ingest import IngestWorkerPool
from mineru.doclib.background.parse_worker import ParseWorkerPool
from mineru.doclib.background.scan_worker import ScanWorkerPool


def test_parse_worker_survives_acquire_error() -> None:
    async def _run() -> None:
        class _ParseService:
            pool: ParseWorkerPool

            async def acquire_task(self) -> None:
                self.pool.running = False
                raise sqlite3.OperationalError("database is locked")

            async def get_queue_length(self) -> int:
                return 0

        service = _ParseService()
        pool = ParseWorkerPool(service, num_workers=1)  # type: ignore[arg-type]
        service.pool = pool
        pool.running = True

        await pool._worker(0)

    asyncio.run(_run())


def test_ingest_worker_survives_acquire_error() -> None:
    async def _run() -> None:
        class _ParseService:
            pass

        service = _ParseService()
        pool = IngestWorkerPool(service, num_workers=1, lock_timeout_sec=60)  # type: ignore[arg-type]

        async def _raise_once() -> None:
            pool.running = False
            raise sqlite3.OperationalError("database is locked")

        pool._acquire_task = _raise_once  # type: ignore[method-assign]
        pool.running = True

        await pool._worker(0)

    asyncio.run(_run())


def test_scan_worker_survives_acquire_error() -> None:
    async def _run() -> None:
        class _ScanService:
            pool: ScanWorkerPool

            async def acquire_task(self) -> None:
                self.pool.running = False
                raise sqlite3.OperationalError("database is locked")

        service = _ScanService()
        pool = ScanWorkerPool(service, num_workers=1)  # type: ignore[arg-type]
        service.pool = pool
        pool.running = True

        await pool._worker(0)

    asyncio.run(_run())
