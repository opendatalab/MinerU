"""验证启动失败、退出重启竞态、组件错误和取消时的应用资源清理。"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mineru.config import Config, LogConfig
from mineru.doclib import app as doclib_app
from mineru.doclib.background import parse_server_health as probes
from mineru.doclib.background import watch as watch_module
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.parse_server_state import ProbeResult


def _configuration(root: Path) -> Config:
    """将本用例的数据库、日志和产物全部放入临时目录。"""
    cfg = Config()
    cfg.doclib.data_dir = str(root / "data")
    cfg.doclib.sqlite.path = str(root / "library.sqlite")
    cfg.doclib.log = LogConfig(dir=str(root / "logs"))
    return cfg


async def _enable_managed(cfg: Config) -> None:
    """使用真实配置表启用托管模式，不启动模型或真实子进程。"""
    db = DatabaseManager(cfg.doclib.sqlite.path)
    await db.initialize()
    await db.execute("INSERT OR REPLACE INTO config(key,value) VALUES ('parse_server.local.mode','managed')")


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_failed_startup_releases_managed_process_and_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_fails: bool
) -> None:
    """托管进程创建后发生初始化错误，仍清理已拥有资源并保留最初错误。"""
    events: list[str] = []
    process = SimpleNamespace(pid=101)

    def start(**kwargs: object) -> tuple[Any, str, None]:
        """记录托管进程已经属于当前应用。"""
        events.append("start")
        return process, "http://127.0.0.1:16580", None

    def stop(proc: object, **kwargs: object) -> None:
        """仅记录真实拥有的非空进程。"""
        if proc is not None:
            assert proc is process
            events.append("stop")

    def fail_watch(*args: object, **kwargs: object) -> None:
        """在进程已创建但后台组件尚未完整组装时抛出主错误。"""
        raise RuntimeError("watch setup failed")

    async def close_db(self: DatabaseManager) -> None:
        """验证失败启动也进入数据库清理，附加错误不能覆盖主错误。"""
        events.append("db.close")
        if cleanup_fails:
            raise ValueError("database close failed")

    monkeypatch.setattr(probes, "start_managed_parse_server", start)
    monkeypatch.setattr(probes, "stop_managed_parse_server", stop)
    monkeypatch.setattr(watch_module, "WatchLoop", fail_watch)
    monkeypatch.setattr(DatabaseManager, "close", close_db)

    async def scenario() -> None:
        """在同一事件循环内检查清理，不依赖测试框架退出时兜底取消。"""
        cfg = _configuration(tmp_path)
        await _enable_managed(cfg)
        app = doclib_app.create_app(cfg)
        state = app.state.doclib_state
        state.discovery_cleanup = lambda: events.append("discovery")
        with pytest.raises(RuntimeError, match="watch setup failed"):
            async with app.router.lifespan_context(app):
                raise AssertionError("startup unexpectedly succeeded")
        assert events == ["start", "stop", "db.close", "discovery"]
        assert state.health.managed_proc is None and not state.background_tasks

    asyncio.run(scenario())


def test_shutdown_does_not_restart_managed_process_after_inflight_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """停止应用时先结束在途探测，避免组件收尾期间再次拉起托管进程。"""

    async def scenario() -> None:
        """用事件控制真实探测循环，确定性触发旧关闭顺序的重启窗口。"""
        cfg = _configuration(tmp_path)
        await _enable_managed(cfg)
        probe_entered, release_probe, restarted = asyncio.Event(), asyncio.Event(), asyncio.Event()
        starts: list[object] = []
        stopped: list[object] = []
        tasks: list[asyncio.Task[Any]] = []
        create = doclib_app._create_background_task

        def start(**kwargs: object) -> tuple[Any, str, None]:
            """第二次创建代表退出过程中发生了不应有的重启。"""
            proc = SimpleNamespace(pid=len(starts) + 1)
            starts.append(proc)
            if len(starts) > 1:
                restarted.set()
            return proc, "http://127.0.0.1:16580", None

        def stop(proc: object, **kwargs: object) -> None:
            """收集最终被应用清理的进程。"""
            if proc is not None:
                stopped.append(proc)

        def schedule(state: Any, name: str, runner: Any) -> Any:
            """只运行本用例需要的真实健康循环，其余组件保留真实停止接口。"""
            if name == "health_check":
                task = create(state, name, runner)
                tasks.append(task)
                return task
            return None

        async def probe(self: Any, url: str, *, api_key: str | None = None) -> ProbeResult:
            """本地探测挂起到应用收尾，远端探测立即返回。"""
            if url == "http://127.0.0.1:16580":
                probe_entered.set()
                await release_probe.wait()
                return ProbeResult(error_code="parse_server_unavailable")
            return ProbeResult(healthy=True, tiers=["standard"])

        monkeypatch.setattr(probes, "start_managed_parse_server", start)
        monkeypatch.setattr(probes, "stop_managed_parse_server", stop)
        monkeypatch.setattr(probes.ParseServerHealthCheck, "_probe", probe)
        monkeypatch.setattr(doclib_app, "_create_background_task", schedule)
        app = doclib_app.create_app(cfg)
        manager = app.router.lifespan_context(app)
        await manager.__aenter__()
        state = app.state.doclib_state
        await asyncio.wait_for(probe_entered.wait(), timeout=2)

        async def stop_watch() -> None:
            """模拟其它组件让出执行权，暴露在途探测返回后的重启机会。"""
            release_probe.set()
            if not tasks[0].done():
                await asyncio.wait_for(restarted.wait(), timeout=2)

        monkeypatch.setattr(state.watch, "stop", stop_watch)
        try:
            await manager.__aexit__(None, None, None)
            assert len(starts) == 1 and stopped == starts
            assert all(task.done() for task in tasks)
        finally:
            release_probe.set()
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    asyncio.run(scenario())


def test_component_stop_failure_does_not_skip_remaining_resources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """一个组件停止失败仍应清理其它工作池、数据库及发现信息，然后报告最初错误。"""
    events: list[str] = []

    def skip_background(*args: object, **kwargs: object) -> None:
        """固定本用例的组件状态，保留应用真实生命周期组装。"""

    monkeypatch.setattr(doclib_app, "_create_background_task", skip_background)

    async def scenario() -> None:
        """对实际组件实例注入停止错误，观察后续资源仍被处理。"""
        app = doclib_app.create_app(_configuration(tmp_path))
        manager = app.router.lifespan_context(app)
        await manager.__aenter__()
        state = app.state.doclib_state

        async def bad_stop() -> None:
            """模拟一个组件的意外停止错误。"""
            events.append("watch")
            raise ValueError("watch stop failed")

        async def pool_stop() -> None:
            """确认解析工作池不因前一组件错误而被跳过。"""
            events.append("parse_workers")

        async def close() -> None:
            """确认数据库仍进入最终清理。"""
            events.append("db.close")

        monkeypatch.setattr(state.watch, "stop", bad_stop)
        monkeypatch.setattr(state.parse_workers, "stop", pool_stop)
        monkeypatch.setattr(state.db, "close", close)
        state.discovery_cleanup = lambda: events.append("discovery")
        with pytest.raises(ValueError, match="watch stop failed"):
            await manager.__aexit__(None, None, None)
        assert events == ["watch", "parse_workers", "db.close", "discovery"]

    asyncio.run(scenario())


def test_teardown_cancellation_waits_for_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """退出请求被取消时先排空已有清理任务，再向调用方传播取消。"""
    events: list[str] = []

    def skip_background(*args: object, **kwargs: object) -> None:
        """仅隔离周期后台任务，不替代实际关闭编排。"""

    monkeypatch.setattr(doclib_app, "_create_background_task", skip_background)

    async def scenario() -> None:
        """挂起一个关闭阶段并取消外部等待者，检查数据库最终仍关闭。"""
        app = doclib_app.create_app(_configuration(tmp_path))
        manager = app.router.lifespan_context(app)
        await manager.__aenter__()
        state = app.state.doclib_state
        entered, release = asyncio.Event(), asyncio.Event()

        async def stop() -> None:
            """受控关闭阶段必须完成，不能被外部取消截断。"""
            entered.set()
            await release.wait()
            events.append("watch.done")

        async def close() -> None:
            """记录清理任务最终关闭数据库。"""
            events.append("db.close")

        monkeypatch.setattr(state.watch, "stop", stop)
        monkeypatch.setattr(state.db, "close", close)
        shutdown = asyncio.create_task(manager.__aexit__(None, None, None))
        await asyncio.wait_for(entered.wait(), timeout=2)
        shutdown.cancel()
        await asyncio.sleep(0)
        shutdown.cancel()
        await asyncio.sleep(0)
        try:
            assert not shutdown.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        assert events == ["watch.done", "db.close"]

    asyncio.run(scenario())
