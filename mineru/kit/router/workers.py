# Copyright (c) Opendatalab. All rights reserved.
"""V1 Router 的 upstream 能力发现与本地 api-server 生命周期。"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import httpx
from loguru import logger

from ...model.runtime.device import get_device
from ...parser.process_control import ManagedProcessControl
from ...types import SERVER_TIERS, ServerTier, Tier
from ...utils.stdio import utf8_subprocess_env

DEFAULT_WORKER_REFRESH_INTERVAL_SECONDS = 2.0
DEFAULT_WORKER_STARTUP_TIMEOUT_SECONDS = 300.0
DEFAULT_WORKER_SHUTDOWN_TIMEOUT_SECONDS = 10.0


def normalize_base_url(url: str) -> str:
    """规范化 upstream base URL 并拒绝空地址。"""
    normalized = str(url).strip().rstrip("/")
    if not normalized:
        raise ValueError("Router upstream URL must not be empty")
    return normalized


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

    for env_name in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
        configured = os.getenv(env_name)
        if configured:
            normalized_config = configured.strip().lower()
            if normalized_config == "all":
                break
            devices = [
                item.strip()
                for item in configured.split(",")
                if item.strip() and item.strip().lower() not in {"-1", "none", "void"}
            ]
            if devices:
                return devices
            return [None]

    device = get_device().split(":", 1)[0]
    if device in {"cpu", "mps"}:
        return [None]
    device_count = accelerator_device_count(device)
    return [str(index) for index in range(device_count)] if device_count > 0 else ["0"]


def visible_device_env_name() -> str:
    """根据当前设备族返回本地 worker 的可见设备环境变量名。"""
    return "ASCEND_RT_VISIBLE_DEVICES" if get_device().split(":", 1)[0] == "npu" else "CUDA_VISIBLE_DEVICES"


@dataclass(frozen=True)
class RouterSettings:
    """保存 Router 服务和其托管 api-server worker 的显式配置。"""

    upstream_urls: tuple[str, ...] = ()
    local_gpus: str = "auto"
    worker_host: str = "127.0.0.1"
    worker_tier: ServerTier = "standard"
    worker_concurrency: int = 1
    preload_models: bool = False
    worker_refresh_interval_seconds: float = DEFAULT_WORKER_REFRESH_INTERVAL_SECONDS

    def __post_init__(self) -> None:
        """校验 worker tier、并发数和刷新间隔。"""
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
        )

    def apply_to_env(self) -> None:
        """把 Router 配置写入 reload 子进程可读取的环境变量。"""
        os.environ["MINERU_ROUTER_UPSTREAM_URLS_JSON"] = json.dumps(list(self.upstream_urls))
        os.environ["MINERU_ROUTER_LOCAL_GPUS"] = self.local_gpus
        os.environ["MINERU_ROUTER_WORKER_HOST"] = self.worker_host
        os.environ["MINERU_ROUTER_WORKER_TIER"] = self.worker_tier
        os.environ["MINERU_ROUTER_WORKER_CONCURRENCY"] = str(self.worker_concurrency)
        os.environ["MINERU_ROUTER_PRELOAD_MODELS"] = "1" if self.preload_models else "0"


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
    tiers: set[Tier] = field(default_factory=set)
    models: list[dict[str, Any]] = field(default_factory=list)
    features: dict[str, Any] = field(default_factory=dict)
    active_jobs: int = 0
    max_concurrent_jobs: int = 1
    last_error: str | None = None


class WorkerPool:
    """维护 Router worker 集合并按能力、负载和 affinity 选择目标。"""

    def __init__(
        self,
        settings: RouterSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """按显式 upstream 与本地 GPU 配置创建 worker 状态。"""
        self.settings = settings
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0), transport=transport)
        self._workers: dict[str, WorkerState] = {}
        self._monitor_task: asyncio.Task[None] | None = None
        self._built = False

    def _build_workers(self) -> None:
        """在 lifespan 启动阶段解析设备并创建 remote/local worker 状态。"""
        if self._built:
            return
        for index, url in enumerate(
            dict.fromkeys(normalize_base_url(item) for item in self.settings.upstream_urls),
            start=1,
        ):
            worker_id = f"remote-{index}"
            self._workers[worker_id] = WorkerState(worker_id=worker_id, base_url=url, source="remote")
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

    async def start(self) -> None:
        """启动本地 worker、刷新全部能力并开启后台健康检查。"""
        self._build_workers()
        for worker in self.workers:
            if worker.local_worker is None:
                continue
            try:
                await worker.local_worker.start(self.client)
                worker.base_url = worker.local_worker.base_url
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

    async def refresh(self, worker: WorkerState) -> None:
        """刷新单个 worker；本地进程退出时先尝试重启。"""
        if worker.local_worker is not None:
            process = worker.local_worker.process
            if process is None or process.poll() is not None:
                try:
                    await worker.local_worker.start(self.client)
                    worker.base_url = worker.local_worker.base_url
                except Exception as exc:
                    worker.healthy = False
                    worker.last_error = str(exc)
                    return
        try:
            health, tiers, models = await asyncio.gather(
                self.client.get(f"{worker.base_url}/v1/health"),
                self.client.get(f"{worker.base_url}/v1/tiers"),
                self.client.get(f"{worker.base_url}/v1/models"),
            )
            health.raise_for_status()
            tiers.raise_for_status()
            models.raise_for_status()
            health_payload = health.json()
            tier_payload = tiers.json()
            model_payload = models.json()
            worker.features = dict(health_payload.get("features") or {})
            worker.tiers = {cast(Tier, item["id"]) for item in tier_payload.get("data", []) if item.get("id")}
            worker.models = [dict(item) for item in model_payload.get("data", [])]
            worker.healthy = health_payload.get("status") == "ok"
            worker.last_error = None if worker.healthy else health.text
        except Exception as exc:
            worker.healthy = False
            worker.last_error = str(exc)

    def healthy_workers(self) -> list[WorkerState]:
        """返回当前健康且具有 base URL 的 worker。"""
        return [worker for worker in self.workers if worker.healthy and worker.base_url]

    def available_tiers(self) -> set[Tier]:
        """聚合全部健康 worker 暴露的请求 tier。"""
        return {tier for worker in self.healthy_workers() for tier in worker.tiers}

    def select(
        self,
        *,
        tier: Tier | None,
        required_sources: set[str] | None = None,
        affinity_key: str | None = None,
        preferred_worker_id: str | None = None,
    ) -> WorkerState | None:
        """按 tier、source、preferred worker、affinity 和活动任务数选择目标。"""
        required = required_sources or set()
        eligible = [
            worker
            for worker in self.healthy_workers()
            if (tier is None or tier in worker.tiers) and required.issubset(set(worker.features.get("sources") or []))
        ]
        if not eligible:
            return None
        if preferred_worker_id is not None:
            preferred = next((worker for worker in eligible if worker.worker_id == preferred_worker_id), None)
            if preferred is not None:
                return preferred
        if affinity_key:
            ordered = sorted(eligible, key=lambda worker: worker.worker_id)
            digest = hashlib.sha256(affinity_key.encode("utf-8")).digest()
            return ordered[int.from_bytes(digest[:8], "big") % len(ordered)]
        return min(
            eligible,
            key=lambda worker: (
                worker.active_jobs / max(worker.max_concurrent_jobs, 1),
                worker.active_jobs,
                worker.worker_id,
            ),
        )

    def mark_job_started(self, worker_id: str) -> None:
        """增加指定 worker 的活动任务计数。"""
        self.get(worker_id).active_jobs += 1

    def mark_job_finished(self, worker_id: str) -> None:
        """安全减少指定 worker 的活动任务计数。"""
        worker = self.get(worker_id)
        worker.active_jobs = max(0, worker.active_jobs - 1)


__all__ = [
    "accelerator_device_count",
    "ManagedLocalWorker",
    "RouterSettings",
    "WorkerPool",
    "WorkerState",
    "normalize_base_url",
    "parse_local_gpus",
    "worker_base_url",
    "worker_connect_host",
]
