from __future__ import annotations

from doclib_test_utils import create_doclib_server
from doclib_test_utils import create_health_check
from mineru.doclib.parse_server_state import ParseServerHealth, ProbeResult, ProbeState

from doclib_test_utils import ParseServices
from mineru.doclib.services.parse_errors import ParseFailure

from doclib_test_utils import create_parse_services

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from mineru.config import LogConfig, ManagedParseServerConfig
from mineru.doclib.background.parse_server_health import ParseServerHealthCheck
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.core.fts import FTSManager
from mineru.doclib.server import DoclibServer, _parse_server_status
from mineru.doclib.services.config_svc import ConfigService

from mineru.doclib.types import ConfigSetRequest
from mineru.types import DeploymentTier


@dataclass
class _ManagedProcess:
    pid: int = 12345

    def poll(self) -> None:
        """模拟仍在运行的托管子进程。"""
        return None


async def _services(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, mode: str = "managed"
) -> tuple[ConfigService, DoclibServer, ParseServices, ParseServerHealth, ParseServerHealthCheck]:
    """用真实 SQLite 配置和 API 路由搭建隔离环境，不触碰用户服务或加载模型。"""
    db = DatabaseManager(str(tmp_path / "doclib.db"))
    await db.initialize()
    cfg = ConfigService(db)
    await cfg.set("parse_server.local.mode", mode)
    await cfg.set("parse_server.local.managed_tier", "basic")
    health = ParseServerHealth(
        local=ProbeState(url="http://127.0.0.1:16580", probe=ProbeResult(healthy=True, tiers=["basic"])),
        remote=ProbeState(probe=ProbeResult(healthy=True, tiers=["standard"])),
        local_mode=mode,
        managed_tier="basic",
        running_managed_tier="basic",
        managed_proc=_ManagedProcess(),
    )

    monkeypatch.setattr("mineru.doclib.server._ensure_managed_parse_server_tier_available", lambda tier, param: None)
    checker = create_health_check(
        cfg, interval_sec=30, probe_timeout_sec=1, startup_grace_sec=30, stop_timeout_sec=10, health=health
    )
    server = create_doclib_server(SimpleNamespace(config_svc=cfg, health_check=checker), health=health)
    parse_svc = create_parse_services(db, FTSManager(db), cfg, str(tmp_path), parse_lock_timeout_sec=60, health=health)
    return cfg, server, parse_svc, health, checker


def _mock_child_lifecycle(
    monkeypatch: pytest.MonkeyPatch, health: ParseServerHealth, *, fail_start: bool = False
) -> list[DeploymentTier]:
    """记录托管进程启动档位，同时验证切换不缩短已就绪旧进程的停止预算。"""
    started: list[DeploymentTier] = []

    def _stop(proc: object, *, control: object, timeout_sec: int, reason: str, startup_in_progress: bool) -> None:
        """停止前必须已经撤销旧探测结果，但旧进程仍按真实启动状态退出。"""
        assert health.local.probe.healthy is False
        assert health.local.probe.tiers == []
        assert startup_in_progress is False

    def _start(
        *, tier: DeploymentTier, managed_cfg: ManagedParseServerConfig, log_cfg: LogConfig | None, marker: str
    ) -> tuple[_ManagedProcess, str, None]:
        """模拟子进程成功启动或启动失败，不把启动成功当作服务就绪。"""
        started.append(tier)
        if fail_start:
            raise RuntimeError("simulated startup failure")
        return _ManagedProcess(pid=23456), "http://127.0.0.1:16581", None

    monkeypatch.setattr("mineru.doclib.background.parse_server_health.stop_managed_parse_server", _stop)
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.start_managed_parse_server", _start)
    return started


@pytest.mark.parametrize("operation", ["set", "unset"])
def test_config_api_invalidates_old_tier_before_return(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str) -> None:
    """配置 API 返回时旧 basic 服务已不可路由，unset 回到默认 standard 也遵守此约束。"""

    async def _run() -> None:
        """经 HTTP API 保存配置，然后检查真实状态投影和路由行为。"""
        cfg, server, parse_svc, health, checker = await _services(tmp_path, monkeypatch)
        old_remote = health.remote
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://doclib") as client:
            url = "/api/v1/configs/parse_server.local.managed_tier"
            response = await client.put(url, json={"value": "standard"}) if operation == "set" else await client.delete(url)
        assert response.status_code == 200
        assert response.json()["value"] == "standard"
        assert await cfg.get("parse_server.local.managed_tier") == "standard"
        assert health.managed_tier == "standard"
        assert health.running_managed_tier == "basic"
        assert health.local_starting is False
        assert health.local.probe.healthy is False
        assert health.local.probe.tiers == []
        assert health.remote is old_remote
        assert checker._wakeup_event.is_set()
        status = _parse_server_status(
            local_mode="managed", managed_tier="standard", self_hosted_url=None, remote_url=None, health=health
        )
        assert status.local.starting is True
        assert status.local.healthy is False
        assert status.local.supported_tiers == []
        with pytest.raises(ParseFailure) as exc:
            await parse_svc.execution._resolve_api_target("local", "standard")
        assert exc.value.code == "engine_unavailable"

    asyncio.run(_run())


@pytest.mark.parametrize("mode,new_tier", [("managed", "basic"), ("self_hosted", "standard"), ("disabled", "standard")])
def test_config_change_preserves_unaffected_local_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, new_tier: DeploymentTier
) -> None:
    """同档位设置、禁用模式和自托管模式不撤销现有探测结果或触发托管重启。"""

    async def _run() -> None:
        """保存配置并验证共享探测对象及版本号保持不变。"""
        _, server, _, health, checker = await _services(tmp_path, monkeypatch, mode=mode)
        old_local = health.local
        await server.set_config("parse_server.local.managed_tier", ConfigSetRequest(value=new_tier))
        assert health.local is old_local
        assert health.local_generation == 0
        assert health.managed_tier_transition is False
        assert not checker._wakeup_event.is_set()

    asyncio.run(_run())


@pytest.mark.parametrize("old_mode", ["self_hosted", "disabled"])
def test_enabling_managed_mode_invalidates_previous_local_probe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, old_mode: str
) -> None:
    """切回 managed 时不能把其他模式的探测结果用于托管服务，即使档位名称相同。"""

    async def _run() -> None:
        """调用模式配置 API，验证健康检查尚未更新模式时旧状态也已撤销。"""
        _, server, parse_svc, health, checker = await _services(tmp_path, monkeypatch, mode=old_mode)
        await server.set_config("parse_server.local.mode", ConfigSetRequest(value="managed"))
        assert health.local.probe.healthy is False
        assert health.local.probe.tiers == []
        assert checker._wakeup_event.is_set()
        with pytest.raises(ParseFailure) as exc:
            await parse_svc.execution._resolve_api_target("local", "basic")
        assert exc.value.code == "engine_unavailable"

    asyncio.run(_run())


def test_superseded_restart_preserves_process_and_restart_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """重启读取配置时又发生切换，丢弃旧重启计划且不消耗故障恢复预算。"""

    async def _run() -> None:
        """在重启读取目标档位的 await 边界撤销切换，旧进程保留且等待重新探测。"""
        cfg, server, _, health, checker = await _services(tmp_path, monkeypatch)
        started = _mock_child_lifecycle(monkeypatch, health)
        await server.set_config("parse_server.local.managed_tier", ConfigSetRequest(value="standard"))
        old_proc = health.managed_proc
        original_get = cfg.get
        changed = False

        async def _get(key: str, default: str | None = None) -> str | None:
            """返回旧目标档位前保存新的 basic 配置，模拟旧读取结果迟到。"""
            nonlocal changed
            value = await original_get(key, default)
            if key == "parse_server.local.managed_tier" and not changed:
                changed = True
                await server.set_config("parse_server.local.managed_tier", ConfigSetRequest(value="basic"))
            return value

        monkeypatch.setattr(cfg, "get", _get)
        await checker._try_restart_managed(health)
        assert started == []
        assert health.managed_proc is old_proc
        assert health.managed_tier == "basic"
        assert health.restart_count == 0
        assert health.local.probe.healthy is False

    asyncio.run(_run())


@pytest.mark.parametrize("change_during", ["config_read", "local_probe", "remote_probe"])
def test_health_check_discards_stale_config_and_probe_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change_during: str
) -> None:
    """配置读取或任一探测期间切换档位，旧结果不能恢复健康或触发错误的重启。"""

    async def _run() -> None:
        """在确定的 await 边界切换配置，验证下一轮只启动最新档位并等待新探测就绪。"""
        cfg, server, parse_svc, health, checker = await _services(tmp_path, monkeypatch)
        started = _mock_child_lifecycle(monkeypatch, health)
        changed = False
        original_get = cfg.get

        async def _change() -> None:
            """只触发一次真实配置切换，并立即检查旧状态已失效。"""
            nonlocal changed
            changed = True
            await server.set_config("parse_server.local.managed_tier", ConfigSetRequest(value="standard"))
            assert health.local.probe.healthy is False
            assert health.local.probe.tiers == []

        async def _get(key: str, default: str | None = None) -> str | None:
            """模拟读取旧目标档位后，配置 API 在返回旧快照前完成切换。"""
            value = await original_get(key, default)
            if key == "parse_server.local.managed_tier" and change_during == "config_read" and not changed:
                await _change()
            return value

        async def _probe(url: str, *, api_key: str | None = None) -> ProbeResult:
            """模拟旧探测迟到；只有新子进程的目标档位探测可以恢复健康。"""
            local = url.startswith("http://127.0.0.1")
            if not changed and ((local and change_during == "local_probe") or (not local and change_during == "remote_probe")):
                await _change()
                return ProbeResult(healthy=True, tiers=["basic"])
            if local and changed:
                assert started == ["standard"]
                assert health.running_managed_tier == "standard"
                assert health.local.probe.healthy is False
                assert health.managed_tier_transition is True
                checker.running = False
                return ProbeResult(healthy=True, tiers=["standard"])
            return ProbeResult(healthy=True, tiers=["basic"])

        monkeypatch.setattr(cfg, "get", _get)
        monkeypatch.setattr(checker, "_probe", _probe)
        await asyncio.wait_for(checker.run(), timeout=2)
        assert started == ["standard"]
        assert health.managed_tier_transition is False
        assert health.local_starting is False
        assert health.local.probe.healthy is True
        assert health.local.probe.tiers == ["standard"]
        target = await parse_svc.execution._resolve_api_target("local", "standard")
        assert target[0] == "http://127.0.0.1:16581"
        assert parse_svc.execution._resolve_tier("standard", target[2]) == "standard"

    asyncio.run(_run())


def test_health_check_continues_after_interval_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """正常间隔超时后继续探测；Python 3.10 的 asyncio 超时异常不能终止后台任务。"""

    async def _run() -> None:
        """用零等待间隔触发真实超时，确认无配置唤醒时仍能完成第二轮探测。"""
        _, _, _, health, checker = await _services(tmp_path, monkeypatch)
        checker.interval_sec = 0
        local_probes = 0
        remote_probes = 0

        async def _probe(url: str, *, api_key: str | None = None) -> ProbeResult:
            """记录本地与远端探测次数，仅在第二轮本地探测完成时停止循环。"""
            nonlocal local_probes, remote_probes
            if url == health.managed_url:
                local_probes += 1
                if local_probes == 2:
                    checker.running = False
            else:
                remote_probes += 1
            return ProbeResult(healthy=True, tiers=["basic"])

        monkeypatch.setattr(checker, "_probe", _probe)
        await asyncio.wait_for(checker.run(), timeout=2)
        assert local_probes == 2
        assert remote_probes == 2
        assert health.local.probe.healthy is True
        assert not checker._wakeup_event.is_set()

    asyncio.run(_run())


def test_config_change_wakes_sleeping_health_check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """健康检查正在等待 30 秒间隔时，配置变更仍能立即触发目标档位重启。"""

    async def _run() -> None:
        """通过事件确定等待边界，不依赖计时猜测或缩短生产探测间隔。"""
        _, server, _, health, checker = await _services(tmp_path, monkeypatch)
        started = _mock_child_lifecycle(monkeypatch, health)
        waiting = asyncio.Event()

        def _interval(failures: int) -> int:
            """通知测试健康检查已经到达正常间隔等待点。"""
            waiting.set()
            return 30

        async def _probe(url: str, *, api_key: str | None = None) -> ProbeResult:
            """新子进程启动后的首次本地探测报告目标档位就绪。"""
            if url.startswith("http://127.0.0.1") and started:
                checker.running = False
                return ProbeResult(healthy=True, tiers=["standard"])
            return ProbeResult(healthy=True, tiers=["basic"])

        monkeypatch.setattr(checker, "_next_interval_sec", _interval)
        monkeypatch.setattr(checker, "_probe", _probe)
        task = asyncio.create_task(checker.run())
        try:
            await asyncio.wait_for(waiting.wait(), timeout=2)
            await server.set_config("parse_server.local.managed_tier", ConfigSetRequest(value="standard"))
            await asyncio.wait_for(task, timeout=2)
        finally:
            if not task.done():
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
        assert started == ["standard"]
        assert health.local.probe.healthy is True

    asyncio.run(_run())


def test_rapid_tier_changes_use_latest_effective_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """连续切换及取消覆盖只按最终有效档位启动，旧探测结果始终不可复用。"""

    async def _run() -> None:
        """在重启任务处理前执行多个配置写入，最终回到默认 standard。"""
        _, server, _, health, checker = await _services(tmp_path, monkeypatch)
        started = _mock_child_lifecycle(monkeypatch, health)
        for tier in ("standard", "basic", "standard", "basic"):
            await server.set_config("parse_server.local.managed_tier", ConfigSetRequest(value=tier))
            assert health.local.probe.healthy is False
        await server.unset_config("parse_server.local.managed_tier")
        await checker._try_restart_managed_for_tier_change(health, "standard")
        assert started == ["standard"]
        assert health.local.probe.healthy is False
        assert health.local_starting is True
        assert health.managed_tier_transition is True

    asyncio.run(_run())


@pytest.mark.parametrize("failure", ["startup", "wrong_tier"])
def test_transition_stays_unavailable_until_target_tier_is_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """启动失败或服务仍广告旧档位时，不能把目标档位标记为可用。"""

    async def _run() -> None:
        """区分创建子进程与目标档位实际就绪，验证失败时解析仍被拦截。"""
        _, server, parse_svc, health, checker = await _services(tmp_path, monkeypatch)
        started = _mock_child_lifecycle(monkeypatch, health, fail_start=failure == "startup")
        await server.set_config("parse_server.local.managed_tier", ConfigSetRequest(value="standard"))
        if failure == "startup":
            await checker._try_restart_managed_for_tier_change(health, "standard")
            assert health.managed_proc is None
            assert health.local_starting is False
        else:

            async def _probe(url: str, *, api_key: str | None = None) -> ProbeResult:
                """即使 HTTP 探测成功，缺少目标档位也必须保持未就绪。"""
                checker.running = False
                return ProbeResult(healthy=True, tiers=["basic"])

            monkeypatch.setattr(checker, "_probe", _probe)
            await checker.run()
        assert started == ["standard"]
        assert health.local.probe.healthy is False
        assert health.local.probe.tiers == []
        with pytest.raises(ParseFailure) as exc:
            await parse_svc.execution._resolve_api_target("local", "standard")
        assert exc.value.code == "engine_unavailable"

    asyncio.run(_run())
