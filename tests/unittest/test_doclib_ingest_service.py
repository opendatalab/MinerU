"""验证独立入库服务和宿主实例中的共享依赖，不执行后台解析模型。"""

from __future__ import annotations

from mineru.doclib.parse_server_state import ParseServerHealth

from mineru.doclib.services.parse_queue_svc import ParseQueueService
from mineru.doclib.services.parse_execution_svc import ParseExecutionService

import asyncio
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mineru.config import PatchedConfig
from mineru.doclib import app as doclib_app
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.services.config_svc import ConfigService
from mineru.doclib.services.ingest_svc import IngestService


def test_ingestion_queues_native_document_without_constructing_parse_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """独立入库可保存 HTML 属性和文本索引，不需要构造解析执行服务。"""

    def forbidden_parse_service(*args: object, **kwargs: object) -> None:
        """任何隐式解析服务组装都使本用例失败。"""
        raise AssertionError("IngestService constructed a ParseService")

    monkeypatch.setattr(ParseQueueService, "__init__", forbidden_parse_service)
    monkeypatch.setattr(ParseExecutionService, "__init__", forbidden_parse_service)

    async def scenario() -> None:
        """真实 SQLite 和 FTS 覆盖文件刷新、纯文本索引以及初始原生任务登记。"""
        db = DatabaseManager(str(tmp_path / "library.sqlite"))
        await db.initialize()
        try:
            fts = FTSManager(db)
            ingest = IngestService(db, fts, ConfigService(db), health=ParseServerHealth())
            html = tmp_path / "source.html"
            html.write_text("<html><head><title>Independent ingest</title></head><body>Native source</body></html>")
            plain = tmp_path / "note.txt"
            plain.write_text("Independent indexed text", encoding="utf-8")
            document = await ingest.ensure_ingested(str(html))
            text = await ingest.ensure_ingested(str(plain))
            assert document is not None and text is not None
            row = await db.fetchone("SELECT title FROM docs WHERE sha256=?", (document["sha256"],))
            assert row["title"] == "Independent ingest"
            queue = await db.fetchall("SELECT sha256, tier, page_range, status FROM parses")
            assert queue == [{"sha256": document["sha256"], "tier": "flash", "page_range": "1", "status": "pending"}]
            indexed = await fts.search("Independent")
            assert {row["sha256"] for row in indexed} == {text["sha256"]}
        finally:
            await db.close()

    asyncio.run(scenario())


def test_doclib_instances_share_ingest_only_with_their_own_components(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """同一应用各入口共用入库服务，两个应用的数据库和入库状态保持隔离。"""
    monkeypatch.setenv("MINERU_HOME", str(tmp_path / "home"))

    def skip_background(*args: object, **kwargs: object) -> None:
        """保留真实生命周期组装，仅关闭周期后台调度以固定检查时序。"""

    monkeypatch.setattr(doclib_app, "_create_background_task", skip_background)
    configurations = [
        PatchedConfig(doclib={"data_dir": str(tmp_path / name), "sqlite": {"path": str(tmp_path / name / "db.sqlite")}})
        for name in ("first", "second")
    ]
    with TestClient(doclib_app.create_app(configurations[0])) as first:
        with TestClient(doclib_app.create_app(configurations[1])) as second:
            left, right = first.app.state.doclib_state, second.app.state.doclib_state
            for state in (left, right):
                assert state.ingest_svc is state.parse_queue_svc.ingest_svc
                assert state.parse_workers.queue_svc is state.parse_queue_svc
                assert state.parse_workers.execution_svc is state.parse_execution_svc
                assert state.parse_execution_svc.db is state.db
                assert state.parse_execution_svc.fts is state.fts
                assert state.ingest_svc is state.scan_svc.ingest_svc
                assert state.ingest_svc is state.watch.ingest_svc
                assert state.ingest_svc is state.ingest_workers.ingest_svc
                assert state.ingest_svc.db is state.db and state.ingest_svc.fts is state.fts
                assert state.ingest_svc.config_svc is state.config_svc
                assert state.ingest_svc.telemetry_svc is state.telemetry_svc
            assert left.ingest_svc is not right.ingest_svc and left.db is not right.db
            source = tmp_path / "one.txt"
            source.write_text("Only the first application stores this text", encoding="utf-8")
            assert first.portal.call(left.ingest_svc.ensure_ingested, str(source)) is not None
            assert second.portal.call(right.db.fetchall, "SELECT * FROM files") == []
