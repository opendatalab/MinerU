"""默认开启、历史迁移与并发关闭的回归，所有发送均使用本地替身。"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from doclib_test_utils import create_doclib_server
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.telemetry import TelemetryService, TelemetryStore
from mineru.doclib.telemetry import service as service_module


@pytest.fixture(autouse=True)
def sent_payloads(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """隔离主机环境并拦截所有遥测 HTTP，测试不向外部服务发送数据。"""
    monkeypatch.delenv("MINERU_TELEMETRY_DISABLED", raising=False)
    sent: list[dict[str, Any]] = []

    async def send(payload: dict[str, Any]) -> str:
        """保存内存载荷供断言，替代网络传输。"""
        sent.append(payload)
        return "success"

    monkeypatch.setattr(service_module, "send_payload", send)
    return sent


async def _new_service(directory: Path) -> tuple[DatabaseManager, TelemetryStore, TelemetryService]:
    """构造独立数据库和服务，避免触碰用户的文档库状态。"""
    db = DatabaseManager(str(directory / "doclib.db"))
    await db.initialize()
    store = TelemetryStore(db)
    service = TelemetryService(store)
    await service.initialize()
    return db, store, service


def test_new_installation_is_enabled_and_notice_does_not_change_preference(tmp_path: Path) -> None:
    """默认开启与说明确认独立，重启后保留说明和用户关闭状态。"""

    async def run() -> None:
        """核对默认来源、说明标记、显式关闭和重启。"""
        _, store, service = await _new_service(tmp_path)
        status = await service.status()
        assert (status["state"], status["source"], status["notice_shown"]) == ("enabled", "default", False)
        identity = status["installation_id"]
        await service.acknowledge_notice()
        assert await store.preference() == ("enabled", "default")
        await service.set_state("disabled")
        await service.initialize()
        status = await service.status()
        assert (status["state"], status["source"], status["notice_shown"]) == ("disabled", "user", True)
        assert status["installation_id"] == identity

    asyncio.run(run())


@pytest.mark.parametrize(
    ("legacy", "expected", "source"),
    [
        ("unset", "enabled", "default"),
        ("enabled", "enabled", "user"),
        ("disabled", "disabled", "user"),
        ("corrupt", "disabled", "error"),
    ],
)
def test_legacy_state_migrates_once(tmp_path: Path, legacy: str, expected: str, source: str) -> None:
    """旧 unset 仅在持久化读取边界转换；关闭和异常值不会变成开启。"""

    async def run() -> None:
        """保留已有 installation_id，并验证迁移可重复执行。"""
        db = DatabaseManager(str(tmp_path / "doclib.db"))
        await db.initialize()
        store = TelemetryStore(db)
        await store.set_state("installation_id", "inst_existing")
        await store.set_state("consent_state", legacy)
        await store.increment(metric_name="search.request.count", value=1, dimensions={})
        service = TelemetryService(store)
        await service.initialize()
        await service.initialize()
        status = await service.status()
        assert (status["state"], status["source"]) == (expected, source)
        assert status["installation_id"] == "inst_existing"
        assert await store.get_state("consent_state") is None
        assert bool(await store.list_periods_for_flush()) == (expected == "enabled")

    asyncio.run(run())


@pytest.mark.parametrize("damage", ["bad_state", "missing_state", "bad_source"])
def test_corrupt_state_blocks_recording_and_sending(tmp_path: Path, damage: str, sent_payloads: list[dict[str, Any]]) -> None:
    """损坏配置始终关闭，不能由默认值回退意外开启。"""

    async def run() -> None:
        """同时检查首次发现异常和重启初始化后的行为。"""
        db, store, service = await _new_service(tmp_path)
        if damage == "missing_state":
            await db.execute("DELETE FROM telemetry_state WHERE key='telemetry_state'")
        else:
            await store.set_state("telemetry_state" if damage == "bad_state" else "state_source", "broken")
        for _ in range(2):
            status = await service.status()
            assert (status["state"], status["source"]) == ("disabled", "error")
            await service.record_count("search.request.count")
            assert (await service.flush_once()).status == "disabled"
            assert await store.pending_counts() == (0, 0)
            await store.initialize()
        assert sent_payloads == []

    asyncio.run(run())


def test_state_read_failure_blocks_recording_and_sending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, sent_payloads: list[dict[str, Any]]
) -> None:
    """数据库读取异常不进入发送或采集路径。"""

    async def run() -> None:
        """只破坏状态查询，保留其他元数据读取以验证诊断来源。"""
        _, store, service = await _new_service(tmp_path)

        async def fail() -> tuple[str, str]:
            """模拟持久化状态读取失败。"""
            raise OSError("state unavailable")

        monkeypatch.setattr(store, "preference", fail)
        assert (await service.status())["source"] == "error"
        await service.record_count("search.request.count")
        assert (await service.flush_once()).status == "disabled"
        assert await store.pending_counts() == (0, 0)
        assert sent_payloads == []

    asyncio.run(run())


@pytest.mark.parametrize("value", ["1", "TRUE", "yes", " on ", "invalid-value"])
def test_environment_override_blocks_automatic_and_manual_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str, sent_payloads: list[dict[str, Any]]
) -> None:
    """环境关闭在首次启动即生效，enable 不越过覆盖也不替它写入关闭偏好。"""

    async def run() -> None:
        """通过真实服务端 action 验证状态来源和手动 flush。"""
        monkeypatch.setenv("MINERU_TELEMETRY_DISABLED", value)
        _, store, service = await _new_service(tmp_path)
        assert await store.preference() == ("enabled", "default")
        server = create_doclib_server(SimpleNamespace(telemetry_svc=service))
        response = await server.telemetry_action("enable")
        assert (response.state, response.source, response.reason) == ("disabled", "environment", "environment_override")
        assert await store.preference() == ("enabled", "user")
        await service.record_count("search.request.count")
        assert (await service.flush_once()).status == "disabled"
        assert not (await server.telemetry_action("flush")).executed
        assert sent_payloads == []
        monkeypatch.delenv("MINERU_TELEMETRY_DISABLED")
        # 环境开关只在服务构造时读取；重建实例后恢复持久化用户偏好。
        assert (await service.status())["source"] == "environment"
        replacement = TelemetryService(store)
        assert (await replacement.status())["state"] == "enabled"

    asyncio.run(run())


def test_disable_cancels_inflight_flush_and_prevents_later_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """关闭等待本轮网络请求取消完成，后续周期和重复 flush 都不得发送。"""

    async def run() -> None:
        """用事件控制真实协程时序，而非依靠 sleep 碰运气。"""
        _, store, service = await _new_service(tmp_path)
        for hour in range(2):
            await service.record_count("search.request.count", timestamp_ms=1_700_000_000_000 + hour * 3_600_000)
        entered, canceled = asyncio.Event(), asyncio.Event()
        sent: list[dict[str, Any]] = []

        async def send(payload: dict[str, Any]) -> str:
            """模拟已开始但尚未完成的 HTTP 请求。"""
            sent.append(payload)
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()
            return "success"

        monkeypatch.setattr(service_module, "send_payload", send)
        server = create_doclib_server(SimpleNamespace(telemetry_svc=service))
        flushing = asyncio.create_task(server.telemetry_action("flush"))
        await asyncio.wait_for(entered.wait(), 5)
        assert (await service.flush_once()).status == "locked"
        response = await asyncio.wait_for(server.telemetry_action("disable"), 5)
        assert response.state == "disabled" and canceled.is_set()
        flushed = await asyncio.wait_for(flushing, 5)
        assert flushed.state == "disabled" and not flushed.executed
        assert await store.pending_counts() == (0, 0)
        assert await store.get_state("flush_locked_at") == "0"
        await service.record_count("search.request.count")
        assert (await service.flush_once()).status == "disabled"
        assert len(sent) == 1
        await service.set_state("enabled")
        assert (await service.flush_once()).status == "no_metrics"

    asyncio.run(run())


def test_disable_serializes_with_record_and_clears_its_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """正在落库的采集先结束，再原子关闭清空，禁止关闭确认后出现旧聚合。"""

    async def run() -> None:
        """暂停一次增量写入，确保关闭在同一服务锁上等待。"""
        _, store, service = await _new_service(tmp_path)
        entered, release = asyncio.Event(), asyncio.Event()
        increment = store.increment

        async def paused_increment(**kwargs: Any) -> None:
            """在真正写入前等待测试放行。"""
            entered.set()
            await release.wait()
            await increment(**kwargs)

        monkeypatch.setattr(store, "increment", paused_increment)
        recording = asyncio.create_task(service.record_count("search.request.count"))
        await asyncio.wait_for(entered.wait(), 5)
        closing = asyncio.create_task(service.set_state("disabled"))
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        await asyncio.wait_for(asyncio.gather(recording, closing), 5)
        await service.record_count("search.request.count")
        assert await store.pending_counts() == (0, 0)

    asyncio.run(run())


def test_disable_and_clear_are_one_transaction(tmp_path: Path) -> None:
    """清空失败时不能留下半成功的关闭状态。"""

    async def run() -> None:
        """通过 SQLite 触发器制造事务内部失败并核对完整回滚。"""
        db, store, service = await _new_service(tmp_path)
        await service.record_count("search.request.count")
        await db.execute(
            "CREATE TRIGGER reject_clear BEFORE DELETE ON telemetry_aggregates "
            "BEGIN SELECT RAISE(ABORT, 'test clear failure'); END"
        )
        with pytest.raises(sqlite3.IntegrityError):
            await service.set_state("disabled")
        assert await store.preference() == ("enabled", "default")
        assert await store.pending_counts() == (1, 1)

    asyncio.run(run())


def test_canceled_flush_caller_releases_sender_and_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """手动请求或后台任务自身取消时，不留下脱离生命周期的发送任务。"""

    async def run() -> None:
        """验证父任务取消沿发送链传播并完整释放持久化锁。"""
        _, store, service = await _new_service(tmp_path)
        await service.record_count("search.request.count")
        entered, canceled = asyncio.Event(), asyncio.Event()

        async def send(payload: dict[str, Any]) -> str:
            """把发送保持在可取消的网络等待阶段。"""
            entered.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()
            return "success"

        monkeypatch.setattr(service_module, "send_payload", send)
        flushing = asyncio.create_task(service.flush_once())
        await asyncio.wait_for(entered.wait(), 5)
        flushing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await flushing
        assert canceled.is_set()
        assert await store.get_state("flush_locked_at") == "0"
        assert await store.pending_counts() == (1, 1)
        await service.set_state("disabled")
        assert await store.pending_counts() == (0, 0)

    asyncio.run(run())


def test_false_environment_override_preserves_explicit_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """环境值为假只表示不覆盖，不能把用户已关闭的遥测重新开启。"""

    async def run() -> None:
        """在新实例和重复初始化后核对持久化选择。"""
        _, store, service = await _new_service(tmp_path)
        await service.set_state("disabled")
        monkeypatch.setenv("MINERU_TELEMETRY_DISABLED", "0")
        replacement = TelemetryService(store)
        await replacement.initialize()
        status = await replacement.status()
        assert (status["state"], status["source"]) == ("disabled", "user")

    asyncio.run(run())
