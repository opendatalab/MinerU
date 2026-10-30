# Copyright (c) Opendatalab. All rights reserved.
"""V1 Router 的 upstream 能力发现与本地 api-server 生命周期。"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import urlsplit

import httpx
from loguru import logger

from ...model.runtime.device import get_device
from ...parser.process_control import ManagedProcessControl
from ...types import SERVER_TIERS, TIERS, ServerTier, Tier
from ...utils.stdio import utf8_subprocess_env
from ...utils.retention import resolve_retention_seconds

DEFAULT_WORKER_REFRESH_INTERVAL_SECONDS = 2.0
DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS = 300.0
DEFAULT_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def normalize_base_url(url: str) -> str:
    """统一校验并规范化 HTTP base URL，保留路径前缀且禁止凭证、查询和片段。"""
    normalized = url.strip()
    try:
        parts = urlsplit(normalized)
        parsed = httpx.URL(normalized)
        valid = (
            parsed.scheme in {"http", "https"}
            and bool(parsed.host)
            and parts.username is None
            and parts.password is None
            and "?" not in normalized
            and "#" not in normalized
            and not any(char.isspace() or ord(char) < 32 for char in normalized)
            and "\\" not in normalized
        )
    except (ValueError, httpx.InvalidURL):
        valid = False
    if not valid:
        raise ValueError("Upstream must be an HTTP(S) base URL without credentials, query, or fragment")
    return str(parsed).rstrip("/")


def _read_tier_metadata(payload: Any) -> dict[Tier, dict[str, Any]]:
    """在发布探测结果前校验全部 tier，拒绝未知 ID 和畸形条目，保留合法附加元数据。"""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Invalid upstream tiers response: expected a data array")
    metadata: dict[Tier, dict[str, Any]] = {}
    for item in payload["data"]:
        if not isinstance(item, dict) or item.get("id") not in TIERS:
            raise ValueError("Invalid upstream tiers response: unsupported tier ID")
        metadata[cast(Tier, item["id"])] = dict(item)
    return metadata


def worker_connect_host(bind_host: str) -> str:
    """把 worker 监听地址转换成 Router 可主动连接的 loopback 地址。"""
    normalized = str(bind_host).strip()
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1]
    if normalized in {"", "0.0.0.0"}:
        return "127.0.0.1"
    if normalized == "::":
        return "::1"
    return normalized


def worker_base_url(bind_host: str, port: int) -> str:
    """构造托管 worker 的可连接 HTTP URL，并为 IPv6 literal 添加方括号。"""
    connect_host = worker_connect_host(bind_host)
    authority = f"[{connect_host}]" if ":" in connect_host else connect_host
    return f"http://{authority}:{port}"


def accelerator_device_count(device: str) -> int:
    """惰性读取指定加速器 runtime 当前可见的设备数量。"""
    try:
        if device == "npu":
            import torch_npu

            return int(torch_npu.npu.device_count())
        import torch

        accelerator = torch.cuda if device == "cuda" else getattr(torch, device, None)
        if accelerator is None or not hasattr(accelerator, "device_count"):
            return 0
        return int(accelerator.device_count())
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        return 0


def reserve_local_port(host: str) -> int:
    """在指定 host 上申请一个当前可用的本地 TCP 端口。"""
    bind_host = str(host).strip() or "127.0.0.1"
    if bind_host.startswith("[") and bind_host.endswith("]"):
        bind_host = bind_host[1:-1]
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return cast(int, sock.getsockname()[1])


def parse_local_gpus(value: str) -> list[str | None]:
    """解析 auto、none 或 GPU CSV，并为 CPU/MPS 返回一个无 GPU worker。"""
    normalized = str(value).strip().lower()
    if normalized == "none":
        return []
    if normalized != "auto":
        devices = [item.strip() for item in value.split(",") if item.strip()]
        if not devices:
            raise ValueError("--local-gpus must be auto, none, or a non-empty CSV")
        return devices

    device = get_device().split(":", 1)[0]
    if device in {"cpu", "mps"}:
        return [None]
    # 只继承当前设备族的掩码，避免把 CUDA/NPU 的编号写成 XPU 的可见设备限制。
    configured = os.getenv(visible_device_env_name(device))
    if configured and configured.strip().lower() != "all":
        devices = [
            item.strip()
            for item in configured.split(",")
            if item.strip() and item.strip().lower() not in {"-1", "none", "void"}
        ]
        return devices or [None]
    device_count = accelerator_device_count(device)
    return [str(index) for index in range(device_count)] if device_count > 0 else ["0"]


# 非默认 CUDA 语义的设备族对应的可见设备环境变量；XPU 用 Level Zero 的 ZE_AFFINITY_MASK（索引 CSV 同构）。
_VISIBLE_DEVICE_ENV_NAMES: dict[str, str] = {
    "npu": "ASCEND_RT_VISIBLE_DEVICES",
    "xpu": "ZE_AFFINITY_MASK",
}


def visible_device_env_name(device: str | None = None) -> str:
    """根据当前设备族返回本地 worker 的可见设备环境变量名。"""
    device_type = (get_device() if device is None else device).split(":", 1)[0]
    return _VISIBLE_DEVICE_ENV_NAMES.get(device_type, "CUDA_VISIBLE_DEVICES")


@dataclass(frozen=True)
class RouterSettings:
    """保存 Router 服务和其托管 api-server worker 的显式配置。"""

    upstream_urls: tuple[str, ...] = ()
    local_gpus: str = "auto"
    worker_host: str = "127.0.0.1"
    worker_tier: ServerTier = "standard"
    worker_concurrency: int = 1
    preload_models: bool = False
    retention_seconds: int | None = None
    worker_refresh_interval_seconds: float = DEFAULT_WORKER_REFRESH_INTERVAL_SECONDS
    admin_api_key: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        """校验 worker tier、并发数和刷新间隔。"""
        object.__setattr__(self, "retention_seconds", resolve_retention_seconds(self.retention_seconds))
        object.__setattr__(self, "upstream_urls", tuple(dict.fromkeys(normalize_base_url(url) for url in self.upstream_urls)))
        object.__setattr__(self, "admin_api_key", (self.admin_api_key or "").strip() or None)
        if self.worker_tier not in SERVER_TIERS:
            raise ValueError(f"Unsupported worker tier: {self.worker_tier}")
        if self.worker_concurrency <= 0:
            raise ValueError("worker_concurrency must be a positive integer")
        if self.worker_refresh_interval_seconds < 0:
            raise ValueError("worker_refresh_interval_seconds must be non-negative")

    @classmethod
    def from_env(cls) -> "RouterSettings":
        """从 reload 子进程使用的环境变量恢复 Router 配置。"""
        raw_urls = os.getenv("MINERU_ROUTER_UPSTREAM_URLS_JSON", "[]")
        try:
            urls = json.loads(raw_urls)
        except json.JSONDecodeError as exc:
            raise ValueError("MINERU_ROUTER_UPSTREAM_URLS_JSON must be valid JSON") from exc
        if not isinstance(urls, list) or not all(isinstance(item, str) for item in urls):
            raise ValueError("MINERU_ROUTER_UPSTREAM_URLS_JSON must be a JSON string array")
        tier = os.getenv("MINERU_ROUTER_WORKER_TIER", "standard")
        return cls(
            upstream_urls=tuple(urls),
            local_gpus=os.getenv("MINERU_ROUTER_LOCAL_GPUS", "auto"),
            worker_host=os.getenv("MINERU_ROUTER_WORKER_HOST", "127.0.0.1"),
            worker_tier=cast(ServerTier, tier),
            worker_concurrency=int(os.getenv("MINERU_ROUTER_WORKER_CONCURRENCY", "1")),
            preload_models=os.getenv("MINERU_ROUTER_PRELOAD_MODELS", "0").lower() in {"1", "true", "yes"},
            admin_api_key=os.getenv("MINERU_ROUTER_ADMIN_API_KEY"),
        )

    def apply_to_env(self) -> None:
        """把 Router 配置写入 reload 子进程可读取的环境变量。"""
        os.environ["MINERU_API_RETENTION_SECONDS"] = str(self.retention_seconds)
        os.environ["MINERU_ROUTER_UPSTREAM_URLS_JSON"] = json.dumps(list(self.upstream_urls))
        os.environ["MINERU_ROUTER_LOCAL_GPUS"] = self.local_gpus
        os.environ["MINERU_ROUTER_WORKER_HOST"] = self.worker_host
        os.environ["MINERU_ROUTER_WORKER_TIER"] = self.worker_tier
        os.environ["MINERU_ROUTER_WORKER_CONCURRENCY"] = str(self.worker_concurrency)
        os.environ["MINERU_ROUTER_PRELOAD_MODELS"] = "1" if self.preload_models else "0"
        if self.admin_api_key:
            os.environ["MINERU_ROUTER_ADMIN_API_KEY"] = self.admin_api_key
        else:
            os.environ.pop("MINERU_ROUTER_ADMIN_API_KEY", None)


@dataclass
class ManagedLocalWorker:
    """管理一个由 Router 启动的 `mineru-kit api-server` 子进程。"""

    worker_id: str
    host: str
    gpu: str | None
    settings: RouterSettings
    base_url: str = ""
    process: subprocess.Popen[bytes] | None = None
    control: ManagedProcessControl | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def command(self, port: int, upload_dir: Path) -> list[str]:
        """构造正式 `mineru-kit api-server` worker 命令。"""
        return [
            sys.executable,
            "-m",
            "mineru.kit.main",
            "api-server",
            "--host",
            self.host,
            "--port",
            str(port),
            "--upload-dir",
            str(upload_dir),
            "--tier",
            self.settings.worker_tier,
            "--concurrency",
            str(self.settings.worker_concurrency),
            "--retention-seconds",
            str(self.settings.retention_seconds),
            "--log-level",
            "warning",
            *(["--preload-models"] if self.settings.preload_models else []),
        ]

    async def start(self, client: httpx.AsyncClient) -> None:
        """启动 worker 子进程并等待 V1 health 就绪。"""
        if self.process is not None and self.process.poll() is None:
            return
        self.temp_dir = tempfile.TemporaryDirectory(prefix=f"mineru-router-{self.worker_id}-")
        upload_dir = Path(self.temp_dir.name) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        port = reserve_local_port(self.host)
        self.base_url = worker_base_url(self.host, port)
        env = utf8_subprocess_env()
        if self.gpu is not None:
            env[visible_device_env_name()] = self.gpu
        control = ManagedProcessControl.create()
        env.update(control.child_env())
        control.start_accepting()
        try:
            self.process = subprocess.Popen(
                self.command(port, upload_dir),
                cwd=os.getcwd(),
                env=env,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            control.close()
            self.temp_dir.cleanup()
            self.temp_dir = None
            raise
        self.control = control
        try:
            await self.wait_until_ready(client)
        except Exception:
            await self.stop()
            raise

    async def wait_until_ready(
        self,
        client: httpx.AsyncClient,
        timeout_seconds: float = DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        """轮询 worker V1 health，直到成功、进程退出或超时。"""
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        last_error = ""
        while asyncio.get_running_loop().time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"Local worker {self.worker_id} exited before becoming healthy")
            try:
                response = await client.get(f"{self.base_url}/v1/health")
                if response.status_code == 200:
                    return
                last_error = response.text
            except httpx.HTTPError as exc:
                last_error = str(exc)
            await asyncio.sleep(0.1)
        suffix = f": {last_error}" if last_error else ""
        raise TimeoutError(f"Timed out waiting for local worker {self.worker_id}{suffix}")

    async def stop(self, timeout_seconds: float = DEFAULT_WORKER_SHUTDOWN_TIMEOUT_SECONDS) -> None:
        """通过控制通道优雅关闭 worker，并在超时后 terminate/kill。"""
        process = self.process
        control = self.control
        self.process = None
        self.control = None
        try:
            if process is not None and process.poll() is None:
                shutdown_sent = control.request_shutdown(timeout_seconds * 0.4) if control is not None else False
                if shutdown_sent:
                    try:
                        await asyncio.to_thread(process.wait, timeout=timeout_seconds * 0.4)
                    except subprocess.TimeoutExpired:
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        await asyncio.to_thread(process.wait, timeout=timeout_seconds * 0.4)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        await asyncio.to_thread(process.wait)
        finally:
            if control is not None:
                control.close()
            if self.temp_dir is not None:
                self.temp_dir.cleanup()
                self.temp_dir = None
            self.base_url = ""


@dataclass
class WorkerState:
    """保存一个 remote 或 local worker 的健康状态、能力和当前负载。"""

    worker_id: str
    base_url: str
    source: str
    local_worker: ManagedLocalWorker | None = None
    healthy: bool = False
    tier_metadata: dict[Tier, dict[str, Any]] = field(default_factory=dict)
    models: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    active_jobs: int = 0
    max_concurrent_jobs: int = 1
    last_error: str | None = None
    generation: int = 0
    restart_requested: bool = False
    restart_failures: int = 0
    retry_at: float = 0.0
    refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    state: Literal["active", "draining"] = "active"
    active_requests: int = 0

    @property
    def tiers(self) -> set[Tier]:
        """返回当前 generation 保留的 tier 标识集合。"""
        return set(self.tier_metadata)

    def acquire_request(self) -> WorkerRequest:
        """在首次网络等待前保护节点，普通请求及流式响应结束后归还引用。"""
        self.active_requests += 1
        return WorkerRequest(self)


@dataclass
class WorkerRequest:
    """持有一个请求期间的节点引用，与执行预占和结果租约分别计数。"""

    worker: WorkerState
    released: bool = False

    def release(self) -> None:
        """幂等释放请求引用，取消或响应失败也不能遗留 draining 阻塞。"""
        if not self.released:
            self.released = True
            self.worker.active_requests -= 1


@dataclass
class JobReservation:
    """持有特定 worker generation 的一次提交预占，释放操作保持幂等。"""

    worker: WorkerState
    generation: int
    released: bool = False

    def release(self) -> None:
        """仅归还所属 generation 的名额，避免迟到清理影响 replacement。"""
        if not self.released:
            self.released = True
            if self.worker.generation == self.generation:
                self.worker.active_jobs = max(0, self.worker.active_jobs - 1)


class WorkerPool:
    """维护 Router worker 集合，按能力、负载与独立轮询分配请求。"""

    def __init__(
        self,
        settings: RouterSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        on_local_worker_replaced: Callable[[WorkerState], Awaitable[None]] | None = None,
    ) -> None:
        """按显式 upstream 与本地 GPU 配置创建 worker 状态。"""
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), transport=transport)
        self._on_local_worker_replaced = on_local_worker_replaced
        self._workers: dict[str, WorkerState] = {}
        self._remote_urls: dict[str, WorkerState] = {}
        self._next_remote_id = 1
        self._monitor_task: asyncio.Task[None] | None = None
        self._built = False
        self._selection_cursors: dict[str, str] = {}

    def _build_workers(self) -> None:
        """在 lifespan 启动阶段解析设备并创建 remote/local worker 状态。"""
        if self._built:
            return
        for url in self.settings.upstream_urls:
            self.register_remote(url)
        for index, gpu in enumerate(parse_local_gpus(self.settings.local_gpus), start=1):
            worker_id = f"local-{gpu if gpu is not None else 'cpu'}-{index}"
            local_worker = ManagedLocalWorker(worker_id, self.settings.worker_host, gpu, self.settings)
            self._workers[worker_id] = WorkerState(
                worker_id=worker_id,
                base_url="",
                source="local",
                local_worker=local_worker,
                max_concurrent_jobs=self.settings.worker_concurrency,
            )
        self._built = True

    @property
    def workers(self) -> list[WorkerState]:
        """返回当前全部 worker 状态。"""
        return list(self._workers.values())

    def get(self, worker_id: str) -> WorkerState:
        """按稳定 worker 标识读取状态。"""
        return self._workers[worker_id]

    def register_remote(self, url: str) -> tuple[WorkerState, bool]:
        """同步登记或恢复远程节点；首次 await 前完成去重，删除后不复用 ID。"""
        normalized = normalize_base_url(url)
        existing = self._remote_urls.get(normalized)
        if existing is not None:
            existing.state = "active"
            return existing, False
        worker_id = f"remote-{self._next_remote_id}"
        self._next_remote_id += 1
        worker = WorkerState(worker_id=worker_id, base_url=normalized, source="remote")
        self._workers[worker_id] = worker
        self._remote_urls[normalized] = worker
        return worker, True

    def drain_remote(self, worker_id: str) -> WorkerState:
        """同步停止远程节点的新预占，保留已有执行和资源访问使用的对象。"""
        worker = self.get(worker_id)
        if worker.source != "remote":
            raise KeyError(worker_id)
        worker.state = "draining"
        return worker

    def remove_drained_workers(self, route_count: Callable[[str], int]) -> None:
        """仅删除无预占、请求和任何路由引用的远程 draining 节点。"""
        for worker in self.workers:
            if (
                worker.source == "remote"
                and worker.state == "draining"
                and worker.active_jobs == 0
                and worker.active_requests == 0
                and route_count(worker.worker_id) == 0
            ):
                del self._workers[worker.worker_id]
                del self._remote_urls[worker.base_url]

    async def start(self) -> None:
        """启动本地 worker、刷新全部能力并开启后台健康检查。"""
        self._build_workers()
        for worker in self.workers:
            if worker.local_worker is None:
                continue
            try:
                await worker.local_worker.start(self.client)
                worker.base_url = worker.local_worker.base_url
                worker.generation = 1
            except Exception as exc:
                worker.healthy = False
                worker.last_error = str(exc)
                logger.exception("Failed to start router worker %s", worker.worker_id)
        await self.refresh_all()
        if self.settings.worker_refresh_interval_seconds > 0:
            self._monitor_task = asyncio.create_task(self._monitor_loop(), name="mineru-v1-router-worker-monitor")

    async def close(self) -> None:
        """停止健康任务、本地 worker 和共享 HTTP client。"""
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
            self._monitor_task = None
        for worker in self.workers:
            if worker.local_worker is not None:
                await worker.local_worker.stop()
        await self.client.aclose()

    async def _monitor_loop(self) -> None:
        """按固定间隔刷新 worker 健康和能力状态。"""
        while True:
            await asyncio.sleep(self.settings.worker_refresh_interval_seconds)
            await self.refresh_all()

    async def refresh_all(self) -> None:
        """并发刷新全部 worker 的 V1 health、tiers 与 models。"""
        await asyncio.gather(*(self.refresh(worker) for worker in self.workers))

    def notify_engine_dead(self, worker: WorkerState) -> None:
        """收到明确 engine_dead 后立即剔除；仅托管进程由 Router 负责恢复。"""
        worker.healthy = False
        worker.last_error = "engine_dead"
        worker.restart_requested = worker.local_worker is not None

    async def refresh(self, worker: WorkerState) -> None:
        """序列化单 worker 的健康刷新，确保同一 generation 只替换一次。"""
        if self._workers.get(worker.worker_id) is not worker:
            return
        reference = worker.acquire_request()
        try:
            async with worker.refresh_lock:
                await self._refresh_worker(worker)
        finally:
            reference.release()

    async def _refresh_worker(self, worker: WorkerState) -> None:
        """复用启停超时重建死亡进程，启动失败按 5/10/20/30 秒退避。"""
        if worker.local_worker is not None:
            process = worker.local_worker.process
            if worker.restart_requested or process is None or process.poll() is not None:
                worker.healthy = False
                if time.monotonic() < worker.retry_at:
                    return
                try:
                    replacing = worker.generation > 0 and (process is not None or worker.restart_requested)
                    if replacing:
                        worker.base_url = ""
                        worker.tier_metadata.clear()
                        worker.models.clear()
                        worker.features.clear()
                        await worker.local_worker.stop()
                        if self._on_local_worker_replaced is not None:
                            await self._on_local_worker_replaced(worker)
                        worker.active_jobs = 0
                        worker.generation += 1
                        worker.restart_requested = False
                    await worker.local_worker.start(self.client)
                    worker.base_url = worker.local_worker.base_url
                    worker.generation = max(worker.generation, 1)
                    worker.restart_failures = 0
                    worker.retry_at = 0.0
                except Exception as exc:
                    worker.last_error = str(exc)
                    delay = (5, 10, 20, 30)[min(worker.restart_failures, 3)]
                    worker.restart_failures += 1
                    worker.retry_at = time.monotonic() + delay
                    return
        try:
            health, tiers, models = await asyncio.gather(
                self.client.get(f"{worker.base_url}/v1/health"),
                self.client.get(f"{worker.base_url}/v1/tiers"),
                self.client.get(f"{worker.base_url}/v1/models"),
            )
            if health.status_code == 503 and (health.json().get("error") or {}).get("code") == "engine_dead":
                self.notify_engine_dead(worker)
                return
            health.raise_for_status()
            tiers.raise_for_status()
            models.raise_for_status()
            health_payload, tier_payload, model_payload = health.json(), tiers.json(), models.json()
            # 先完整读取并校验，失败时保留上一份合法快照，避免局部更新污染管理响应。
            tier_metadata = _read_tier_metadata(tier_payload)
            features = dict(health_payload.get("features") or {})
            model_metadata = [dict(item) for item in model_payload.get("data", [])]
            worker.features = features
            worker.tier_metadata = tier_metadata
            worker.models = model_metadata
            worker.healthy = health_payload.get("status") == "ok"
            worker.last_error = None if worker.healthy else health.text
        except Exception as exc:
            worker.healthy = False
            worker.last_error = str(exc)

    def healthy_workers(self) -> list[WorkerState]:
        """仅返回允许新任务接入的健康节点；旧任务仍可通过 get 访问 draining。"""
        return [worker for worker in self.workers if worker.state == "active" and worker.healthy and worker.base_url]

    def available_tiers(self) -> set[Tier]:
        """聚合全部健康 worker 暴露的请求 tier。"""
        return {tier for worker in self.healthy_workers() for tier in worker.tiers}

    def select(
        self,
        *,
        tier: Tier | None,
        required_sources: set[str] | None = None,
        preferred_worker_id: str | None = None,
        required_task_outputs: set[str] | None = None,
        selection_kind: Literal["upload", "job"] = "job",
    ) -> WorkerState | None:
        """按能力和负载选取目标，文件归属仅在最低负载候选中优先。"""
        eligible = self.capable_workers(
            tier=tier,
            required_sources=required_sources,
            required_task_outputs=required_task_outputs,
        )
        if not eligible:
            return None
        loads = {
            worker.worker_id: (worker.active_jobs / max(worker.max_concurrent_jobs, 1), worker.active_jobs)
            for worker in eligible
        }
        minimum = min(loads.values())
        eligible = sorted((worker for worker in eligible if loads[worker.worker_id] == minimum), key=lambda w: w.worker_id)
        if preferred_worker_id is not None:
            preferred = next((worker for worker in eligible if worker.worker_id == preferred_worker_id), None)
            if preferred is not None:
                self._selection_cursors[selection_kind] = preferred.worker_id
                return preferred
        previous = self._selection_cursors.get(selection_kind, "")
        selected = next((worker for worker in eligible if worker.worker_id > previous), eligible[0])
        self._selection_cursors[selection_kind] = selected.worker_id
        return selected

    def capable_workers(
        self,
        *,
        tier: Tier | None,
        required_sources: set[str] | None = None,
        required_task_outputs: set[str] | None = None,
    ) -> list[WorkerState]:
        """只读过滤完整能力组合，供预检复用；不改变轮询游标或预占负载。"""
        return [
            worker
            for worker in self.healthy_workers()
            if (tier is None or tier in worker.tiers)
            and (required_sources or set()).issubset(set(worker.features.get("sources") or []))
            and (required_task_outputs or set()).issubset(set(worker.features.get("task_output_formats") or []))
        ]

    def reserve_job(
        self,
        *,
        tier: Tier | None,
        required_sources: set[str],
        preferred_worker_id: str | None = None,
        required_task_outputs: set[str] | None = None,
    ) -> JobReservation | None:
        """在第一次网络等待前同步选择并预占解析 worker。"""
        worker = self.select(
            tier=tier,
            required_sources=required_sources,
            preferred_worker_id=preferred_worker_id,
            required_task_outputs=required_task_outputs,
        )
        if worker is None:
            return None
        worker.active_jobs += 1
        return JobReservation(worker, worker.generation)


__all__ = [
    "JobReservation",
    "accelerator_device_count",
    "ManagedLocalWorker",
    "RouterSettings",
    "WorkerPool",
    "WorkerRequest",
    "WorkerState",
    "normalize_base_url",
    "parse_local_gpus",
    "worker_base_url",
    "worker_connect_host",
]
