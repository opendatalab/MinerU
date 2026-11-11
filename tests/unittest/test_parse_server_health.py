# Copyright (c) Opendatalab. All rights reserved.
"""Parse-server 健康探测的失败退避节奏。"""

from doclib_test_utils import create_health_check

import asyncio
import errno
import logging
import socket
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mineru.doclib.background import parse_server_health as health_module
from mineru.doclib.background.parse_server_health import ParseServerHealthCheck
from mineru.doclib.parse_server_state import ProbeResult


def _checker(interval_sec: int = 30) -> ParseServerHealthCheck:
    return create_health_check(
        config_svc=None,
        interval_sec=interval_sec,
        probe_timeout_sec=10,
        startup_grace_sec=30,
        stop_timeout_sec=10,
    )


def test_healthy_or_unprobed_cycles_keep_normal_interval() -> None:
    checker = _checker()
    assert checker._next_interval_sec(0) == 30
    # 退避序列走完后封顶回正常间隔（长期宕机不放大探测频率）。
    assert checker._next_interval_sec(9) == 30


def test_failure_backoff_speeds_up_probe() -> None:
    checker = _checker()
    # 连续失败按 5/10/15/20/25/30 秒加速重试：start 阶段 server ready 后 5 秒内被发现。
    assert checker._next_interval_sec(1) == 5
    assert checker._next_interval_sec(2) == 10
    assert checker._next_interval_sec(3) == 15
    assert checker._next_interval_sec(4) == 20
    assert checker._next_interval_sec(5) == 25
    assert checker._next_interval_sec(6) == 30
    assert checker._next_interval_sec(7) == 30


def test_backoff_never_exceeds_normal_interval() -> None:
    """interval 配置小于退避值时，取较小值。"""
    checker = _checker(interval_sec=8)
    assert checker._next_interval_sec(1) == 5
    assert checker._next_interval_sec(2) == 8
    assert checker._next_interval_sec(0) == 8


@pytest.mark.parametrize("failure", ["spawn", "readiness"])
def test_restart_exhaustion_survives_config_reload_and_keeps_remote_probes(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, failure: str
) -> None:
    """真实循环连续失败后只熔断一次，后续配置刷新保留故障且远端探测继续。"""
    checker = _checker(interval_sec=0)
    checker.startup_timeout_sec = 0
    health = checker.health
    health.local_mode = "managed"
    health.managed_tier = "standard"
    config = {
        "parse_server.local.mode": "managed",
        "parse_server.local.managed_tier": "standard",
        "parse_server.remote.url": "https://remote.example",
    }
    checker.config_svc = SimpleNamespace(get=AsyncMock(side_effect=config.get))
    processes = [Mock(pid=index, poll=Mock(return_value=None)) for index in range(3)]
    start = Mock(
        side_effect=OSError(errno.EADDRINUSE, "occupied")
        if failure == "spawn"
        else [(proc, health.managed_url, None) for proc in processes]
    )
    stop = Mock()
    monkeypatch.setattr(health_module, "start_managed_parse_server", start)
    monkeypatch.setattr(health_module, "stop_managed_parse_server", stop)
    remote_probes = 0
    local_probes = 0

    async def probe(url: str, *, api_key: str | None = None) -> ProbeResult:
        """运行八轮后结束，远端始终可用且本地始终无法就绪。"""
        nonlocal remote_probes, local_probes
        if url == config["parse_server.remote.url"]:
            remote_probes += 1
            if remote_probes == 8:
                checker.running = False
            return ProbeResult(healthy=True, tiers=["standard"])
        assert not health.managed_restart_exhausted
        local_probes += 1
        return ProbeResult(error_code="parse_server_unavailable", error_msg="simulated failure")

    monkeypatch.setattr(checker, "_probe", probe)
    with caplog.at_level(logging.ERROR, logger=health_module.logger.name):
        asyncio.run(checker.run())

    assert start.call_count == health_module.MAX_RESTART_ATTEMPTS
    assert health.restart_count == health_module.MAX_RESTART_ATTEMPTS
    assert health.managed_restart_exhausted is True
    assert health.local_mode == config["parse_server.local.mode"] == "managed"
    assert health.managed_proc is None and health.managed_control is None
    assert health.running_managed_tier is None
    assert health.local_starting is False and health.managed_tier_transition is False
    assert health.local.probe.healthy is False
    assert health.local.probe.error_code == "parse_server_unavailable"
    assert "failed 3 restarts" in health.local.probe.error_msg
    assert caplog.text.count("disabling automatic restarts") == 1
    assert remote_probes == 8 and health.remote.probe.healthy is True
    assert local_probes == (3 if failure == "spawn" else 4)
    stopped = [call.args[0] for call in stop.call_args_list if call.args[0] is not None]
    assert stopped == ([] if failure == "spawn" else processes)


def test_exhausted_restart_retries_cleanup_without_losing_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """熔断回收失败时保留进程和控制通道，再次检查只回收残留资源。"""
    checker = _checker()
    health = checker.health
    health.restart_count = health_module.MAX_RESTART_ATTEMPTS
    process, control = Mock(), Mock()
    health.managed_proc = process
    health.managed_control = control
    health.local_starting = True
    health.managed_tier_transition = True
    stop = Mock(side_effect=[RuntimeError("still alive"), None])
    start = Mock()
    monkeypatch.setattr(health_module, "stop_managed_parse_server", stop)
    monkeypatch.setattr(health_module, "start_managed_parse_server", start)

    asyncio.run(checker._try_restart_managed(health))
    assert health.managed_restart_exhausted is True
    assert health.managed_proc is process and health.managed_control is control
    assert health.local_starting is False and health.managed_tier_transition is False
    asyncio.run(checker._try_restart_managed(health))
    assert health.managed_proc is None and health.managed_control is None
    assert [call.args[0] for call in stop.call_args_list] == [process, process]
    start.assert_not_called()


@pytest.mark.parametrize("kill_fails", [True, False])
def test_stop_managed_process_reports_unconfirmed_exit(kill_fails: bool) -> None:
    """强杀失败或强杀后等待超时均向调用方报告失败，并关闭控制通道。"""
    process = Mock(pid=123, poll=Mock(return_value=None))
    process.wait.side_effect = subprocess.TimeoutExpired("parse-server", 0)
    if kill_fails:
        process.kill.side_effect = PermissionError("kill denied")
    control = Mock(request_shutdown=Mock(return_value=False))

    with pytest.raises(RuntimeError, match="did not stop"):
        health_module.stop_managed_parse_server(process, control=control, timeout_sec=0, reason="test")

    process.kill.assert_called_once()
    control.close.assert_called_once()


def test_failed_stop_retains_owned_process_and_prevents_replacement(monkeypatch: pytest.MonkeyPatch) -> None:
    """旧进程回收失败时阻止创建新进程，应用退出仍能重试同一资源。"""
    checker = _checker()
    checker.config_svc = SimpleNamespace(get=AsyncMock(return_value="standard"))
    health = checker.health
    process, control = Mock(), Mock()
    health.managed_proc = process
    health.managed_control = control
    start = Mock()
    monkeypatch.setattr(health_module, "start_managed_parse_server", start)
    monkeypatch.setattr(health_module, "stop_managed_parse_server", Mock(side_effect=RuntimeError("still alive")))

    asyncio.run(checker._try_restart_managed(health))
    start.assert_not_called()
    assert health.managed_proc is process and health.managed_control is control
    with pytest.raises(RuntimeError, match="still alive"):
        asyncio.run(checker.stop())
    assert health.managed_proc is process and health.managed_control is control


def test_failed_tier_change_cleanup_is_bounded_by_restart_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """切档回收失败也计入连续失败预算，避免旧进程仍在时无限绕过熔断。"""
    checker = _checker()
    checker.config_svc = SimpleNamespace(get=AsyncMock(return_value="advanced"))
    health = checker.health
    process = Mock()
    health.managed_proc = process
    health.running_managed_tier = "standard"
    start = Mock()
    monkeypatch.setattr(health_module, "start_managed_parse_server", start)
    monkeypatch.setattr(health_module, "stop_managed_parse_server", Mock(side_effect=RuntimeError("still alive")))
    for _attempt in range(health_module.MAX_RESTART_ATTEMPTS):
        assert asyncio.run(checker._try_restart_managed_for_tier_change(health, "advanced")) is True
    assert health.managed_restart_exhausted is True
    assert health.restart_count == health_module.MAX_RESTART_ATTEMPTS
    assert health.managed_proc is process
    assert asyncio.run(checker._try_restart_managed_for_tier_change(health, "advanced")) is False
    start.assert_not_called()


def test_windows_managed_port_probe_rejects_reusable_occupied_address(monkeypatch: pytest.MonkeyPatch) -> None:
    """模拟 Windows 已有监听允许复用的情形，探测必须跳过占用端口而非共享绑定成功。"""
    sockets: list[Mock] = []

    def create_socket(family: int, kind: int) -> Mock:
        """为每个候选端口创建独立 socket，按 Winsock 地址复用语义处理绑定。"""
        sock = Mock()
        sockets.append(sock)

        def bind(address: tuple[str, int]) -> None:
            """已占用端口仅允许开启 SO_REUSEADDR 的探测共享绑定。"""
            reuse_enabled = any(
                call.args == (socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) for call in sock.setsockopt.call_args_list
            )
            if address[1] == 16580 and not reuse_enabled:
                raise OSError(errno.EADDRINUSE, "occupied reusable address")
            sock.getsockname.return_value = address

        sock.bind.side_effect = bind
        return sock

    monkeypatch.setattr(health_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(health_module.socket, "socket", create_socket)

    selected = health_module.select_available_managed_port("127.0.0.1", 16580, strict_port=False, port_probe_count=2)

    assert selected == 16581
    assert len(sockets) == 2
    for sock in sockets:
        sock.close.assert_called_once()


@pytest.mark.parametrize("strict_port", [True, False])
def test_managed_port_probe_rejects_real_reusable_listener(strict_port: bool) -> None:
    """用真实允许复用的监听 socket 验证冲突处理，在 Windows CI 覆盖实际 Winsock 行为。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        occupied_port = occupied.getsockname()[1]
        if strict_port:
            with pytest.raises(OSError):
                health_module.select_available_managed_port("127.0.0.1", occupied_port, strict_port=True, port_probe_count=1)
        else:
            probe_count = min(100, 65536 - occupied_port)
            if probe_count < 2:
                pytest.skip("系统分配了最后一个端口，没有后续候选端口")
            selected = health_module.select_available_managed_port(
                "127.0.0.1", occupied_port, strict_port=False, port_probe_count=probe_count
            )
            assert occupied_port < selected < occupied_port + probe_count


@pytest.mark.parametrize("winerror", [10013, 10048, None])
@pytest.mark.parametrize("strict_port", [True, False])
def test_managed_port_selection_handles_windows_conflicts(
    monkeypatch: pytest.MonkeyPatch, winerror: int | None, strict_port: bool
) -> None:
    """Windows 端口冲突在非严格模式下继续探测，普通权限错误仍直接报告。"""
    error = PermissionError(errno.EACCES, "forbidden")
    if winerror is not None:
        error.winerror = winerror
    sock = Mock()
    sock.bind.side_effect = [error, None]
    sock.getsockname.return_value = ("127.0.0.1", 16581)
    monkeypatch.setattr(health_module.socket, "socket", Mock(return_value=sock))

    if strict_port or winerror is None:
        with pytest.raises(PermissionError) as caught:
            health_module.select_available_managed_port("127.0.0.1", 16580, strict_port=strict_port, port_probe_count=3)
        assert caught.value is error
        assert sock.bind.call_count == 1
    else:
        selected = health_module.select_available_managed_port("127.0.0.1", 16580, strict_port=False, port_probe_count=3)
        assert selected == 16581
        assert [call.args[0][1] for call in sock.bind.call_args_list] == [16580, 16581]


def test_managed_port_exhaustion_preserves_windows_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """所有 Windows 候选端口被拒绝后报告范围耗尽，并保留最后的系统异常。"""
    error = PermissionError(errno.EACCES, "forbidden")
    error.winerror = 10013
    sock = Mock(bind=Mock(side_effect=error))
    monkeypatch.setattr(health_module.socket, "socket", Mock(return_value=sock))
    with pytest.raises(RuntimeError, match="16580-16582") as caught:
        health_module.select_available_managed_port("127.0.0.1", 16580, strict_port=False, port_probe_count=3)
    assert caught.value.__cause__ is error
    assert sock.bind.call_count == sock.close.call_count == 3


def test_managed_start_closes_control_if_listener_thread_cannot_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """创建监听线程失败时也关闭控制通道，避免启动异常留下监听资源。"""
    checker = _checker()
    control = Mock(start_accepting=Mock(side_effect=RuntimeError("cannot start new thread")))
    control.child_env.return_value = {}
    monkeypatch.setattr(health_module, "select_available_managed_port", Mock(return_value=16580))
    monkeypatch.setattr(health_module.ManagedProcessControl, "create", Mock(return_value=control))
    popen = Mock()
    monkeypatch.setattr(health_module.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="cannot start new thread"):
        health_module.start_managed_parse_server(
            tier="standard", managed_cfg=checker.managed_parse_server, log_cfg=checker.log_cfg, marker="test"
        )
    control.close.assert_called_once()
    popen.assert_not_called()
