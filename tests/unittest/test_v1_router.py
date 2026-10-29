from __future__ import annotations

import asyncio
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from task_test_utils import router_execution

import mineru.parser.api_server as api_server
from mineru.kit.router import RouterSettings, create_app
from mineru.kit.router.workers import (
    ManagedLocalWorker,
    WorkerState,
    parse_local_gpus,
    worker_base_url,
    worker_connect_host,
)


@dataclass
class _FakeV1Upstream:
    """仅用于 worker 能力和故障隔离测试；完整 Tasks 链路使用真实双服务 fixture。"""

    name: str
    tiers: tuple[str, ...]
    tier_models: dict[str, str] = field(default_factory=dict)

    async def handle(self, request: httpx.Request) -> httpx.Response:
        """只声明 Tasks 能力，不提供已移除的旧上传和 Jobs 路由。"""
        if request.url.path == "/v1/health":
            payload = {
                "status": "ok",
                "version": "test",
                "features": {
                    "webhook": False,
                    "sources": ["upload", "url"],
                    "task_output_formats": ["default", "docx", "html", "latex"],
                },
            }
        elif request.url.path == "/v1/tiers":
            payload = {
                "object": "list",
                "data": [
                    {
                        "id": tier,
                        "description": f"{self.name}-{tier}",
                        "current_model": self.tier_models.get(tier, f"{self.name}-model"),
                    }
                    for tier in self.tiers
                ],
            }
        elif request.url.path == "/v1/models":
            payload = {"object": "list", "data": [{"id": f"{self.name}-model", "object": "model", "created": 1}]}
        elif request.url.path == "/v1/usage":
            payload = {"object": "usage", "current": {}}
        else:
            return httpx.Response(404, json={"error": {"code": "task_not_found", "message": "missing"}}, request=request)
        return httpx.Response(200, json=payload, request=request)


def _make_router(
    *upstreams: _FakeV1Upstream,
) -> tuple[Any, dict[str, _FakeV1Upstream]]:
    """创建使用 host 分发 MockTransport 的 Router 应用。"""
    by_host = {upstream.name: upstream for upstream in upstreams}

    async def _handler(request: httpx.Request) -> httpx.Response:
        """把请求交给 URL host 对应的 fake upstream。"""
        return await by_host[request.url.host].handle(request)

    settings = RouterSettings(
        upstream_urls=tuple(f"http://{upstream.name}" for upstream in upstreams),
        local_gpus="none",
        worker_refresh_interval_seconds=0,
    )
    return create_app(settings, transport=httpx.MockTransport(_handler)), by_host


def test_native_routing_requires_one_worker_with_complete_capabilities() -> None:
    """原生文档绕过 tier 筛选，但来源和输出能力必须由同一个健康 worker 满足。"""
    app, _ = _make_router(_FakeV1Upstream("worker-a", ("standard",)), _FakeV1Upstream("worker-b", ("basic",)))
    with TestClient(app) as client:
        first, second = app.state.worker_pool.workers
        first.features = {"sources": ["upload"], "task_output_formats": ["default"]}
        second.features = {"sources": ["url"], "task_output_formats": ["html"]}
        body = {"files": [{"file_name": "native.docx", "bytes": 1}], "tier": "advanced", "output_formats": ["html"]}
        rejected = client.post("/v1/tasks", json=body)
        assert rejected.status_code == 503
        assert not app.state.task_store._records
        assert first.active_jobs == second.active_jobs == 0
        first.features["task_output_formats"].append("html")
        accepted = client.post("/v1/tasks", json=body)
        assert accepted.status_code == 202, accepted.text
        assert app.state.task_store._records[accepted.json()["task_id"]].options.tier == "flash"


def test_v1_router_preserves_upstream_tier_metadata() -> None:
    """验证 Router 不再从全局 models 列表推断各 tier 的 current_model。"""
    upstream = _FakeV1Upstream(
        "worker-a",
        ("flash", "basic", "standard"),
        tier_models={
            "flash": "flash",
            "basic": "hybrid-basic",
            "standard": "MinerU2.5-Pro-2605-1.2B",
        },
    )
    router_app, _ = _make_router(upstream)

    with TestClient(router_app) as client:
        tier_data = client.get("/v1/tiers").json()["data"]

    assert [(item["id"], item["description"], item["current_model"]) for item in tier_data] == [
        ("flash", "worker-a-flash", "flash"),
        ("basic", "worker-a-basic", "hybrid-basic"),
        ("standard", "worker-a-standard", "MinerU2.5-Pro-2605-1.2B"),
    ]


def test_managed_router_worker_builds_new_api_server_command(tmp_path: Path) -> None:
    """验证本地 Router worker 仅启动正式 mineru-kit api-server 参数面。"""
    settings = RouterSettings(
        local_gpus="none",
        worker_tier="basic",
        worker_concurrency=3,
        preload_models=True,
    )
    worker = ManagedLocalWorker("local-0", "127.0.0.1", "0", settings)

    command = worker.command(18000, tmp_path)
    assert command[command.index("--log-level") + 1] == "warning"

    assert command[:5] == [command[0], "-m", "mineru.kit.main", "api-server", "--host"]
    assert command[0].endswith("python")
    assert ["--tier", "basic"] == command[command.index("--tier") : command.index("--tier") + 2]
    assert ["--concurrency", "3"] == command[command.index("--concurrency") : command.index("--concurrency") + 2]
    assert "--preload-models" in command


@pytest.mark.parametrize(
    ("bind_host", "expected_connect_host", "expected_url"),
    [
        ("", "127.0.0.1", "http://127.0.0.1:18000"),
        ("0.0.0.0", "127.0.0.1", "http://127.0.0.1:18000"),
        ("127.0.0.1", "127.0.0.1", "http://127.0.0.1:18000"),
        ("::", "::1", "http://[::1]:18000"),
        ("::1", "::1", "http://[::1]:18000"),
    ],
)
def test_managed_worker_formats_connectable_ipv4_and_ipv6_urls(
    bind_host: str,
    expected_connect_host: str,
    expected_url: str,
) -> None:
    """验证 wildcard、IPv4 与 IPv6 worker 地址生成合法可连接 URL。"""
    assert worker_connect_host(bind_host) == expected_connect_host
    assert worker_base_url(bind_host, 18000) == expected_url


def test_auto_local_gpus_expands_visible_accelerator_count(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 auto 在无显式设备 CSV 或值为 all 时枚举全部可见 GPU。"""
    import mineru.kit.router.workers as workers

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(workers, "get_device", lambda: "cuda")
    monkeypatch.setattr(workers, "accelerator_device_count", lambda _device: 4)
    assert parse_local_gpus("auto") == ["0", "1", "2", "3"]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
    assert parse_local_gpus("auto") == ["2", "4"]

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "all")
    monkeypatch.setattr(workers, "accelerator_device_count", lambda _device: 3)
    assert parse_local_gpus("auto") == ["0", "1", "2"]


def test_xpu_local_gpus_use_level_zero_affinity_mask(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 XPU worker 通过 ZE_AFFINITY_MASK 继承与隔离可见设备。"""
    import mineru.kit.router.workers as workers

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    monkeypatch.delenv("ZE_AFFINITY_MASK", raising=False)
    monkeypatch.setattr(workers, "get_device", lambda: "xpu")
    assert workers.visible_device_env_name() == "ZE_AFFINITY_MASK"

    monkeypatch.setattr(workers, "accelerator_device_count", lambda _device: 2)
    assert parse_local_gpus("auto") == ["0", "1"]

    monkeypatch.setenv("ZE_AFFINITY_MASK", "1")
    assert parse_local_gpus("auto") == ["1"]


@pytest.mark.parametrize(
    ("device", "expected", "env_name"),
    [
        ("cuda", ["3"], "CUDA_VISIBLE_DEVICES"),
        ("npu", ["2"], "ASCEND_RT_VISIBLE_DEVICES"),
        ("xpu", ["0", "1"], "ZE_AFFINITY_MASK"),
        ("xpu:1", ["0", "1"], "ZE_AFFINITY_MASK"),
        ("cpu", [None], "CUDA_VISIBLE_DEVICES"),
        ("mps", [None], "CUDA_VISIBLE_DEVICES"),
    ],
)
def test_auto_local_gpus_ignores_other_device_family_masks(
    monkeypatch: pytest.MonkeyPatch, device: str, expected: list[str | None], env_name: str
) -> None:
    """多个设备族掩码共存时只继承当前设备族，CPU/MPS 不据此创建 GPU worker。"""
    import mineru.kit.router.workers as workers

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("ZE_AFFINITY_MASK", "0,1")
    monkeypatch.setattr(workers, "get_device", lambda: device)
    assert parse_local_gpus("auto") == expected
    assert workers.visible_device_env_name() == env_name


@pytest.mark.parametrize("mask", [None, "all", "-1", "none", "void", "0.1, 1.0"])
def test_xpu_local_gpus_mask_fallbacks(monkeypatch: pytest.MonkeyPatch, mask: str | None) -> None:
    """XPU 无掩码时枚举自身设备，禁用标记与子设备编号不受其他设备族影响。"""
    import mineru.kit.router.workers as workers

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2")
    monkeypatch.delenv("ZE_AFFINITY_MASK", raising=False)
    if mask is not None:
        monkeypatch.setenv("ZE_AFFINITY_MASK", mask)
    monkeypatch.setattr(workers, "get_device", lambda: "xpu:1")
    monkeypatch.setattr(workers, "accelerator_device_count", lambda _device: 2)
    expected = ["0", "1"] if mask in {None, "all"} else (["0.1", "1.0"] if mask == "0.1, 1.0" else [None])
    assert parse_local_gpus("auto") == expected
    assert parse_local_gpus("4,5") == ["4", "5"]
    assert parse_local_gpus("none") == []


def test_compose_router_healthcheck_uses_v1_path() -> None:
    """验证 Router Compose profile 只探测正式 `/v1/health`。"""
    repo_root = Path(__file__).resolve().parents[2]
    compose_text = (repo_root / "docker/compose.yaml").read_text(encoding="utf-8")
    router_section = compose_text.split("  mineru-router:", 1)[1].split("  mineru-webui:", 1)[0]

    assert "http://localhost:8002/v1/health" in router_section
    assert "http://localhost:8002/health" not in router_section


def test_managed_router_worker_uses_control_shutdown() -> None:
    """验证托管 worker 优先通过 ManagedProcessControl 正常退出。"""

    class _FakeProcess:
        """模拟收到控制关闭后可正常 wait 的子进程。"""

        def __init__(self) -> None:
            """初始化进程存活和信号调用状态。"""
            self.exited = False
            self.terminated = False
            self.killed = False

        def poll(self) -> int | None:
            """返回当前模拟退出码。"""
            return 0 if self.exited else None

        def wait(self, timeout: float | None = None) -> int:
            """模拟控制通道关闭后进程正常退出。"""
            del timeout
            self.exited = True
            return 0

        def terminate(self) -> None:
            """记录不应触发的 terminate。"""
            self.terminated = True

        def kill(self) -> None:
            """记录不应触发的 kill。"""
            self.killed = True

    class _FakeControl:
        """模拟可用的父子进程控制通道。"""

        def __init__(self) -> None:
            """初始化关闭请求和资源关闭状态。"""
            self.requested = False
            self.closed = False

        def request_shutdown(self, timeout_sec: float) -> bool:
            """记录优雅关闭请求并返回成功。"""
            assert timeout_sec > 0
            self.requested = True
            return True

        def close(self) -> None:
            """记录控制通道已释放。"""
            self.closed = True

    worker = ManagedLocalWorker("local-0", "127.0.0.1", None, RouterSettings(local_gpus="none"))
    process = _FakeProcess()
    control = _FakeControl()
    worker.process = process  # type: ignore[assignment]
    worker.control = control  # type: ignore[assignment]

    asyncio.run(worker.stop())

    assert control.requested is True
    assert control.closed is True
    assert process.exited is True
    assert process.terminated is False
    assert process.killed is False


def test_local_worker_replacement_invalidates_previous_generation_routes() -> None:
    """验证本地 worker 崩溃后，新 generation 发布前清除旧资源和负载。"""

    class _ExitedProcess:
        """模拟已经退出的本地 api-server 进程。"""

        @staticmethod
        def poll() -> int:
            """返回非零退出码。"""
            return 1

    class _RunningProcess:
        """模拟 replacement 启动后的存活进程。"""

        @staticmethod
        def poll() -> None:
            """返回 None 表示进程仍在运行。"""
            return None

    class _ReplacementLocalWorker:
        """模拟复用同一 worker ID 的本地 api-server replacement。"""

        def __init__(self) -> None:
            """初始化退出进程和旧 base URL。"""
            self.process: Any = _ExitedProcess()
            self.base_url = "http://old-worker"

        async def stop(self) -> None:
            """清理旧 generation 的进程状态。"""
            self.process = None
            self.base_url = ""

        async def start(self, _client: Any) -> None:
            """发布 replacement 测试 upstream。"""
            self.process = _RunningProcess()
            self.base_url = "http://worker-a"

    upstream = _FakeV1Upstream("worker-a", ("standard",))
    router_app, _ = _make_router(upstream)

    with TestClient(router_app) as client:
        pool = router_app.state.worker_pool
        registry = router_app.state.registry
        local_worker = _ReplacementLocalWorker()
        state = WorkerState(
            worker_id="local-replaced",
            base_url=local_worker.base_url,
            source="local",
            local_worker=local_worker,  # type: ignore[arg-type]
            healthy=True,
            active_jobs=2,
            generation=1,
            tier_metadata={"standard": {"id": "standard", "current_model": "old-model"}},
        )
        pool._workers[state.worker_id] = state
        file_route = registry.register_file(
            owner_scope="anonymous",
            worker_id=state.worker_id,
            upstream_id="file-old",
        )
        job_route = router_execution(
            registry,
            owner_scope="anonymous",
            worker_id=state.worker_id,
            upstream_id="job-old",
        )

        client.portal.call(pool.refresh, state)

        assert state.generation == 2
        assert state.base_url == "http://worker-a"
        assert state.healthy is True
        assert state.active_jobs == 0
        assert state.tier_metadata["standard"]["current_model"] == "worker-a-model"
        assert registry.find("file", file_route.public_id) is None
        assert registry.find("job", job_route.public_id) is None


def test_router_console_script_points_to_new_cli() -> None:
    """校验过渡 mineru-router console script 指向正式 Router CLI。"""
    repo_root = Path(__file__).resolve().parents[2]
    pyproject = tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))

    assert pyproject["project"]["scripts"]["mineru-router"] == "mineru.kit.router.cli:main"


def test_force_exit_timer_fires_within_deadline() -> None:
    """控制通道关闭的兜底强杀应在超时后触发 exit(1)，daemon 线程不阻塞进程退出。"""
    exited: list[int] = []

    timer = api_server._start_force_exit_timer(0.05, exit_fn=exited.append)

    assert timer.daemon is True
    time.sleep(0.2)
    assert exited == [1]


def test_force_exit_timer_does_not_fire_when_cancelled() -> None:
    """进程在窗口内自然退出（cancel）时不应强杀。"""
    exited: list[int] = []

    timer = api_server._start_force_exit_timer(0.3, exit_fn=exited.append)
    timer.cancel()
    time.sleep(0.45)

    assert exited == []
