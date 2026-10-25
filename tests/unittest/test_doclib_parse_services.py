"""验证平级解析队列和执行器的依赖、抢占及异常收尾边界。"""

from __future__ import annotations

from docvortex.options import LatexDelimitersConfig
from mineru.doclib.parse_server_state import ParseServerHealth

import asyncio
import time
from pathlib import Path

import pytest

from mineru.doclib.background.parse_worker import ParseWorkerPool
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.rows import ParseRow
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.ingest_svc import IngestService
from mineru.doclib.services.parse_execution_svc import ParseExecutionService
from mineru.doclib.services.parse_queue_svc import ParseQueueService


def test_queue_request_and_claim_do_not_construct_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """队列可独立完成入库校验、任务复用、提权、查询和抢占。"""

    def forbidden(*args: object, **kwargs: object) -> None:
        """任何隐式执行器组装都会使队列独立性测试失败。"""
        raise AssertionError("Queue constructed an executor")

    monkeypatch.setattr(ParseExecutionService, "__init__", forbidden)

    async def scenario() -> None:
        """只构造真实入库和队列服务，不使用测试组合工厂。"""
        db = DatabaseManager(str(tmp_path / "library.sqlite"))
        await db.initialize()
        try:
            fts = FTSManager(db)
            ingest = IngestService(db, fts, ConfigService(db), health=ParseServerHealth())
            queue = ParseQueueService(
                db,
                fts,
                str(tmp_path),
                ingest_svc=ingest,
                parse_lock_timeout_sec=60,
                health=ingest.health,
                latex_delimiters=LatexDelimitersConfig(),
            )
            source = tmp_path / "document.html"
            source.write_text("<html><body>Independent queue</body></html>")
            response = await queue.request_parse(str(source), tier="flash")
            assert response.reused_parse_ids and not response.created_parse_ids
            assert await queue.get_queue_length() == 1
            task = await queue.acquire_task()
            assert task is not None and task["priority"] == 1 and task["status"] == "parsing"
            assert task["id"] == response.reused_parse_ids[0]
            assert (await queue.get_parse_record(task["id"]))["status"] == "parsing"
            assert await queue.acquire_task() is None
        finally:
            await db.close()

    asyncio.run(scenario())


def test_two_queues_claim_unique_batches_in_priority_order(tmp_path: Path) -> None:
    """原子抢占保持优先级、有效锁排除和过期锁重取，不会重复分配同一批次。"""

    async def scenario() -> None:
        """两个队列对象共享真实数据库，对三个不同文档执行并发抢占。"""
        db = DatabaseManager(str(tmp_path / "library.sqlite"))
        await db.initialize()
        try:
            fts = FTSManager(db)
            ingest = IngestService(db, fts, ConfigService(db), health=ParseServerHealth())
            for index in range(3):
                path = tmp_path / f"source-{index}.html"
                path.write_text(f"<html><body>Source {index}</body></html>")
                await ingest.ensure_ingested(str(path))
            rows = await db.fetchall("SELECT id FROM parses ORDER BY id")
            first, second, third = [row["id"] for row in rows]
            now = int(time.time() * 1000)
            await db.execute("UPDATE parses SET priority=2, locked_at=? WHERE id=?", (now, first))
            await db.execute("UPDATE parses SET priority=1, locked_at=? WHERE id=?", (now - 61_000, third))
            queues = [
                ParseQueueService(
                    db,
                    fts,
                    str(tmp_path),
                    ingest_svc=ingest,
                    parse_lock_timeout_sec=60,
                    health=ingest.health,
                    latex_delimiters=LatexDelimitersConfig(),
                )
                for _ in range(2)
            ]
            assert await queues[0].get_queue_length() == 2
            highest = await queues[0].acquire_task()
            assert highest is not None and highest["id"] == third
            claimed = await asyncio.gather(*(queue.acquire_task() for queue in queues))
            assert [task["id"] for task in claimed if task is not None] == [second]
            assert await queues[0].get_queue_length() == 0
            assert await queues[1].acquire_task() is None
            active = await db.fetchone("SELECT status, locked_at FROM parses WHERE id=?", (first,))
            assert active == {"status": "pending", "locked_at": now}
        finally:
            await db.close()

    asyncio.run(scenario())


def test_worker_unexpected_execution_error_records_failure_and_unlocks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """分离队列与执行依赖后，工作池异常兜底仍写入原有失败码并解除锁。"""

    async def scenario() -> None:
        """执行一个真实抢占的任务，仅将模型执行替换为受控异常。"""
        db = DatabaseManager(str(tmp_path / "library.sqlite"))
        await db.initialize()
        try:
            fts = FTSManager(db)
            config = ConfigService(db)
            ingest = IngestService(db, fts, config, health=ParseServerHealth())
            source = tmp_path / "source.html"
            source.write_text("<html><body>Worker failure</body></html>")
            await ingest.ensure_ingested(str(source))
            queue = ParseQueueService(
                db,
                fts,
                str(tmp_path),
                ingest_svc=ingest,
                parse_lock_timeout_sec=60,
                health=ingest.health,
                latex_delimiters=LatexDelimitersConfig(),
            )
            execution = ParseExecutionService(db, fts, config, str(tmp_path), health=ingest.health)
            pool = ParseWorkerPool(queue, execution, num_workers=1)
            acquire = queue.acquire_task

            async def once() -> ParseRow | None:
                """抢占之后停止下一轮调度，使本用例只处理一个批次。"""
                task = await acquire()
                pool.running = False
                return task

            async def fail(task: ParseRow) -> bool:
                """模拟执行器在自身错误处理之外抛出的异常。"""
                raise RuntimeError("x" * 600)

            monkeypatch.setattr(queue, "acquire_task", once)
            monkeypatch.setattr(execution, "process_doc", fail)
            pool.running = True
            await pool._worker(0)
            row = await db.fetchone("SELECT status, locked_at, error_code, error_msg FROM parses")
            assert row == {"status": "failed", "locked_at": None, "error_code": "parse_failed", "error_msg": "x" * 500}
        finally:
            await db.close()

    asyncio.run(scenario())
