# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import uuid
from contextlib import ExitStack, asynccontextmanager, suppress
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import click
import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger
from starlette.background import BackgroundTask
from starlette.datastructures import UploadFile as StarletteUploadFile

from mineru.cli.api_client import (
    LOCAL_API_CLEANUP_RETRIES,
    LOCAL_API_CLEANUP_RETRY_INTERVAL_SECONDS,
    LOCAL_API_STARTUP_TIMEOUT_SECONDS,
    TASK_RESULT_TIMEOUT_SECONDS,
    TASK_STATUS_POLL_INTERVAL_SECONDS,
    build_managed_process_popen_kwargs,
    build_http_timeout,
    build_result_download_timeout,
    find_free_port,
    normalize_base_url,
    stop_managed_process,
    strip_local_api_network_args,
    response_detail,
)
from mineru.cli.api_protocol import API_PROTOCOL_VERSION
from mineru.cli.common import normalize_upload_filename
from mineru.cli.public_http_client_policy import (
    configure_public_http_client_policy,
    is_public_bind_host,
    validate_public_http_client_request,
    warn_if_public_http_client_policy as _warn_if_public_http_client_policy,
)
from mineru.cli.vlm_preload import build_local_api_cli_args
from mineru.version import __version__

TASK_PENDING = "pending"
TASK_PROCESSING = "processing"
TASK_COMPLETED = "completed"
TASK_FAILED = "failed"
TASK_TERMINAL_STATES = {TASK_COMPLETED, TASK_FAILED}
DEFAULT_TASK_RETENTION_SECONDS = 24 * 60 * 60
DEFAULT_TASK_CLEANUP_INTERVAL_SECONDS = 5 * 60
FILE_PARSE_TASK_ID_HEADER = "X-MinerU-Task-Id"
FILE_PARSE_TASK_STATUS_HEADER = "X-MinerU-Task-Status"
FILE_PARSE_TASK_STATUS_URL_HEADER = "X-MinerU-Task-Status-Url"
FILE_PARSE_TASK_RESULT_URL_HEADER = "X-MinerU-Task-Result-Url"
HEALTH_ENDPOINT = "/health"
TASKS_ENDPOINT = "/tasks"
SOURCE_LOCAL = "local"
SOURCE_REMOTE = "remote"
LOCAL_GPU_AUTO = "auto"
LOCAL_GPU_NONE = "none"
HTTP_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
UPSTREAM_FAILURE_THRESHOLD = 3
WORKER_REFRESH_INTERVAL_SECONDS = 2.0
WORKER_HEALTH_FAILURE_RESTART_THRESHOLD = 5
MIN_HEALTHY_PROCESSING_WINDOW_SIZE = 1
MINERU_ROUTER_PUBLIC_BIND_EXPOSED_ENV = "MINERU_ROUTER_PUBLIC_BIND_EXPOSED"
MINERU_ROUTER_ALLOW_PUBLIC_HTTP_CLIENT_ENV = "MINERU_ROUTER_ALLOW_PUBLIC_HTTP_CLIENT"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def env_flag_enabled(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def get_int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    if value < minimum:
        return default
    return value


def _parse_json_object_response(
    response: httpx.Response,
    payload_name: str,
) -> dict[str, Any]:
    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"{payload_name} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{payload_name} must be a JSON object")
    return payload


def get_task_retention_seconds() -> int:
    return get_int_env(
        "MINERU_API_TASK_RETENTION_SECONDS",
        DEFAULT_TASK_RETENTION_SECONDS,
        minimum=0,
    )


def get_task_cleanup_interval_seconds() -> int:
    return get_int_env(
        "MINERU_API_TASK_CLEANUP_INTERVAL_SECONDS",
        DEFAULT_TASK_CLEANUP_INTERVAL_SECONDS,
        minimum=1,
    )


def is_task_terminal(status: str) -> bool:
    return status in TASK_TERMINAL_STATES


def warn_if_public_http_client_policy(host: str, allow_public_http_client: bool) -> None:
    _warn_if_public_http_client_policy(
        service_name="router",
        host=host,
        allow_public_http_client=allow_public_http_client,
    )


def cleanup_path(path: str) -> None:
    try:
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)
    except FileNotFoundError:
        return
    except Exception as exc:
        logger.warning("Failed to clean up {}: {}", path, exc)


def cleanup_temporary_directory(temp_dir: tempfile.TemporaryDirectory[str]) -> None:
    last_error: Exception | None = None
    for attempt in range(LOCAL_API_CLEANUP_RETRIES):
        try:
            temp_dir.cleanup()
            return
        except FileNotFoundError:
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < LOCAL_API_CLEANUP_RETRIES:
                import time

                time.sleep(LOCAL_API_CLEANUP_RETRY_INTERVAL_SECONDS)

    if last_error is not None:
        logger.warning("Failed to clean up temporary directory {}: {}", temp_dir.name, last_error)


def parse_json_env(name: str, default: Sequence[str] = ()) -> tuple[str, ...]:
    raw = os.getenv(name)
    if not raw:
        return tuple(default)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Invalid {} value: {}", name, raw)
        return tuple(default)
    if not isinstance(payload, list) or not all(isinstance(item, str) for item in payload):
        logger.warning("Invalid {} value: {}", name, raw)
        return tuple(default)
    return tuple(payload)


def resolve_connect_host(host: str) -> str:
    if host in {"0.0.0.0", "::"}:
        return "127.0.0.1"
    return host


def reserve_unique_local_ports(count: int) -> list[int]:
    """一次性占用并释放多个本地端口，降低并行启动 worker 时的端口重复风险。"""
    if count <= 0:
        return []

    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sock.listen(1)
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def normalize_local_device_type(device: str | None) -> str:
    """将 get_device() 返回值规范化为基础设备类型。"""
    if not device:
        return "cuda"
    return str(device).strip().lower().split(":", 1)[0]


def get_local_device_type() -> str:
    """懒加载读取当前设备类型，避免 router 导入阶段提前加载 torch。"""
    try:
        from mineru.utils.config_reader import get_device

        return normalize_local_device_type(get_device())
    except Exception as exc:
        logger.warning("Failed to resolve local device type, fallback to cuda: {}", exc)
        return "cuda"


def get_local_device_visible_env_name() -> str:
    """根据当前设备类型选择本地 worker 的可见设备环境变量。"""
    if get_local_device_type() == "npu":
        return "ASCEND_RT_VISIBLE_DEVICES"
    return "CUDA_VISIBLE_DEVICES"


def _parse_visible_devices_env(env_name: str) -> list[str] | None:
    """解析显式配置的可见设备列表，未配置时返回 None。"""
    configured_visible_devices = os.getenv(env_name)
    if configured_visible_devices is None:
        return None
    return [
        item.strip()
        for item in configured_visible_devices.split(",")
        if item.strip()
    ]


def _detect_cuda_devices() -> list[str]:
    """自动探测当前 CUDA 可见设备编号。"""
    try:
        import torch  # type: ignore
    except ImportError:
        return []
    if not torch.cuda.is_available():
        return []
    return [str(index) for index in range(torch.cuda.device_count())]


def _detect_npu_devices() -> list[str]:
    """自动探测当前 Ascend NPU 可见设备编号。"""
    try:
        import torch_npu  # type: ignore
    except ImportError:
        return []
    if not torch_npu.npu.is_available():
        return []
    return [str(index) for index in range(torch_npu.npu.device_count())]


def detect_visible_local_devices() -> list[str]:
    """探测当前设备类型对应的可见本地设备编号。"""
    visible_devices_env_name = get_local_device_visible_env_name()
    configured_visible_devices = _parse_visible_devices_env(visible_devices_env_name)
    if configured_visible_devices is not None:
        return configured_visible_devices

    if visible_devices_env_name == "ASCEND_RT_VISIBLE_DEVICES":
        return _detect_npu_devices()
    return _detect_cuda_devices()


def parse_local_gpus(local_gpus: str) -> list[str | None]:
    value = local_gpus.strip().lower()
    if value == LOCAL_GPU_NONE:
        return []
    if value == LOCAL_GPU_AUTO:
        detected = detect_visible_local_devices()
        if detected:
            return detected
        return [None]

    resolved: list[str | None] = []
    for item in local_gpus.split(","):
        normalized = item.strip()
        if not normalized:
            continue
        resolved.append(normalized)
    return resolved


@dataclass(frozen=True)
class RouterSettings:
    upstream_urls: tuple[str, ...] = ()
    local_gpus: str = LOCAL_GPU_AUTO
    worker_host: str = "127.0.0.1"
    enable_vlm_preload: bool = False
    worker_extra_args: tuple[str, ...] = ()
    task_retention_seconds: int = DEFAULT_TASK_RETENTION_SECONDS
    task_cleanup_interval_seconds: int = DEFAULT_TASK_CLEANUP_INTERVAL_SECONDS
    worker_refresh_interval_seconds: float = WORKER_REFRESH_INTERVAL_SECONDS

    @classmethod
    def from_env(cls) -> "RouterSettings":
        return cls(
            upstream_urls=parse_json_env("MINERU_ROUTER_UPSTREAM_URLS_JSON"),
            local_gpus=os.getenv("MINERU_ROUTER_LOCAL_GPUS", LOCAL_GPU_AUTO),
            worker_host=os.getenv("MINERU_ROUTER_WORKER_HOST", "127.0.0.1"),
            enable_vlm_preload=env_flag_enabled(
                "MINERU_ROUTER_ENABLE_VLM_PRELOAD",
                default=False,
            ),
            worker_extra_args=parse_json_env("MINERU_ROUTER_WORKER_ARGS_JSON"),
            task_retention_seconds=get_task_retention_seconds(),
            task_cleanup_interval_seconds=get_task_cleanup_interval_seconds(),
        )


@dataclass
class StagedUpload:
    field_name: str
    upload_name: str
    content_type: str
    path: str


@dataclass
class MultipartPayload:
    temp_dir: str
    fields: list[tuple[str, str]]
    uploads: list[StagedUpload]

    def cleanup(self) -> None:
        cleanup_path(self.temp_dir)

    def get_field_value(self, name: str) -> Optional[str]:
        for key, value in reversed(self.fields):
            if key == name:
                return value
        return None


@dataclass
class RouterTaskRecord:
    task_id: str
    upstream_server_id: str
    upstream_task_id: str
    upstream_base_url: str
    backend: str
    file_names: list[str]
    created_at: str
    status: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    queued_ahead: int | None = None
    upstream_error_count: int = 0

    def to_status_payload(self, request: Request) -> dict[str, Any]:
        payload = {
            "task_id": self.task_id,
            "status": self.status,
            "backend": self.backend,
            "file_names": self.file_names,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "status_url": str(request.url_for("get_router_task_status", task_id=self.task_id)),
            "result_url": str(request.url_for("get_router_task_result", task_id=self.task_id)),
        }
        if self.queued_ahead is not None:
            payload["queued_ahead"] = self.queued_ahead
        return payload


@dataclass
class ManagedLocalServer:
    server_id: str
    worker_host: str
    gpu: str | None
    enable_vlm_preload: bool
    extra_cli_args: tuple[str, ...]
    connect_host: str = field(init=False)
    base_url: str | None = None
    process: subprocess.Popen[bytes] | None = None
    process_group_id: int | None = None
    temp_dir: tempfile.TemporaryDirectory[str] | None = None

    def __post_init__(self) -> None:
        self.connect_host = resolve_connect_host(self.worker_host)

    def is_running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    async def start(self, client: httpx.AsyncClient, port: int | None = None) -> None:
        if self.is_running():
            return

        self.temp_dir = tempfile.TemporaryDirectory(prefix=f"{self.server_id}-")
        output_root = Path(self.temp_dir.name) / "output"
        output_root.mkdir(parents=True, exist_ok=True)

        resolved_port = port if port is not None else find_free_port()
        remaining_cli_args = strip_local_api_network_args(self.extra_cli_args)
        worker_cli_args = build_local_api_cli_args(
            remaining_cli_args,
            enable_vlm_preload=self.enable_vlm_preload,
        )
        self.base_url = f"http://{self.connect_host}:{resolved_port}"
        env = os.environ.copy()
        env["MINERU_API_OUTPUT_ROOT"] = str(output_root)
        env["MINERU_API_DISABLE_ACCESS_LOG"] = "1"
        if self.gpu is not None:
            env[get_local_device_visible_env_name()] = str(self.gpu)

        command = [
            sys.executable,
            "-m",
            "mineru.cli.fast_api",
            "--host",
            self.worker_host,
            "--port",
            str(resolved_port),
            *worker_cli_args,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=os.getcwd(),
            env=env,
            **build_managed_process_popen_kwargs(),
        )
        self.process_group_id = self.process.pid

        try:
            await self.wait_until_ready(client)
        except Exception:
            self.stop()
            raise

    async def wait_until_ready(
        self,
        client: httpx.AsyncClient,
        timeout_seconds: float = LOCAL_API_STARTUP_TIMEOUT_SECONDS,
    ) -> None:
        assert self.base_url is not None
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        last_error: str | None = None
        while asyncio.get_running_loop().time() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(f"Local worker {self.server_id} exited before becoming healthy")
            try:
                response = await client.get(f"{self.base_url}{HEALTH_ENDPOINT}")
                if response.status_code == 200:
                    return
                last_error = response_detail(response)
            except httpx.HTTPError as exc:
                last_error = str(exc)
            await asyncio.sleep(TASK_STATUS_POLL_INTERVAL_SECONDS)

        message = f"Timed out waiting for local worker {self.server_id} to become healthy"
        if last_error:
            message = f"{message}: {last_error}"
        raise RuntimeError(message)

    async def restart(self, client: httpx.AsyncClient) -> None:
        self.stop()
        await self.start(client)

    def stop(self) -> None:
        process = self.process
        process_group_id = self.process_group_id
        self.process = None
        self.process_group_id = None
        try:
            if process is not None or process_group_id is not None:
                stop_managed_process(
                    process,
                    process_group_id=process_group_id,
                    shutdown_timeout_seconds=5,
                    use_stdin_shutdown_watcher=False,
                )
        finally:
            temp_dir = self.temp_dir
            self.temp_dir = None
            self.base_url = None
            if temp_dir is not None:
                cleanup_temporary_directory(temp_dir)


@dataclass
class WorkerState:
    server_id: str
    source: str
    base_url: str
    gpu: str | None
    local_server: ManagedLocalServer | None = None
    healthy: bool = False
    queued_tasks: int = 0
    processing_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    max_concurrent_requests: int = 0
    processing_window_size: int = MIN_HEALTHY_PROCESSING_WINDOW_SIZE
    last_error: Optional[str] = None
    last_checked_at: Optional[str] = None
    pending_assignments: int = 0
    consecutive_health_failures: int = 0

    def score(self) -> float:
        denominator = max(1, self.max_concurrent_requests)
        numerator = self.queued_tasks + self.processing_tasks + self.pending_assignments
        return numerator / denominator

    def snapshot(self) -> dict[str, Any]:
        return {
            "server_id": self.server_id,
            "base_url": self.base_url,
            "source": self.source,
            "healthy": self.healthy,
            "gpu": self.gpu,
            "queued_tasks": self.queued_tasks,
            "processing_tasks": self.processing_tasks,
            "completed_tasks": self.completed_tasks,
            "failed_tasks": self.failed_tasks,
            "max_concurrent_requests": self.max_concurrent_requests,
            "processing_window_size": self.processing_window_size,
            "last_error": self.last_error,
            "last_checked_at": self.last_checked_at,
        }


class WorkerPool:
    def __init__(
        self,
        settings: RouterSettings,
        client: httpx.AsyncClient,
        *,
        randomizer: Optional[random.Random] = None,
    ):
        self.settings = settings
        self.client = client
        self.randomizer = randomizer or random.Random()
        self._selection_lock = asyncio.Lock()
        self._monitor_task: asyncio.Task[Any] | None = None
        self._servers: dict[str, WorkerState] = {}
        self._build_servers()

    def _build_servers(self) -> None:
        for index, url in enumerate(dict.fromkeys(normalize_base_url(item) for item in self.settings.upstream_urls), start=1):
            server_id = f"remote-{index}"
            self._servers[server_id] = WorkerState(
                server_id=server_id,
                source=SOURCE_REMOTE,
                base_url=url,
                gpu=None,
            )

        local_specs = parse_local_gpus(self.settings.local_gpus)
        for index, gpu in enumerate(local_specs, start=1):
            if gpu is None:
                server_id = "local-cpu-1"
            else:
                server_id = f"local-gpu-{gpu}"
            local_server = ManagedLocalServer(
                server_id=server_id,
                worker_host=self.settings.worker_host,
                gpu=gpu,
                enable_vlm_preload=self.settings.enable_vlm_preload,
                extra_cli_args=self.settings.worker_extra_args,
            )
            self._servers[server_id] = WorkerState(
                server_id=server_id,
                source=SOURCE_LOCAL,
                base_url="",
                gpu=gpu,
                local_server=local_server,
            )

    @property
    def servers(self) -> list[WorkerState]:
        return list(self._servers.values())

    async def start(self) -> None:
        local_servers = [
            server for server in self.servers if server.local_server is not None
        ]
        local_ports = reserve_unique_local_ports(len(local_servers))
        await asyncio.gather(
            *(
                self._start_local_server(server, port)
                for server, port in zip(local_servers, local_ports)
            )
        )

        await self.refresh_all()
        if self._monitor_task is None or self._monitor_task.done():
            self._monitor_task = asyncio.create_task(self._monitor_loop(), name="mineru-router-worker-monitor")

    async def _start_local_server(self, server: WorkerState, port: int) -> None:
        """启动单个本地 worker，并把启动失败限制在该 worker 状态内。"""
        if server.local_server is None:
            return
        try:
            await server.local_server.start(self.client, port=port)
            server.base_url = normalize_base_url(server.local_server.base_url or "")
        except Exception as exc:
            server.healthy = False
            server.last_error = str(exc)
            server.last_checked_at = utc_now_iso()

    async def shutdown(self) -> None:
        if self._monitor_task is not None:
            self._monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._monitor_task
            self._monitor_task = None
        for server in self.servers:
            if server.local_server is not None:
                server.local_server.stop()

    async def _monitor_loop(self) -> None:
        while True:
            await asyncio.sleep(self.settings.worker_refresh_interval_seconds)
            await self.refresh_all()

    async def refresh_all(self) -> None:
        for server in self.servers:
            await self._refresh_server(server)

    def _update_server_from_health_payload(
        self,
        server: WorkerState,
        payload: dict[str, Any],
    ) -> None:
        protocol_version = payload.get("protocol_version")
        if protocol_version != API_PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported protocol_version={protocol_version}, expected {API_PROTOCOL_VERSION}"
            )

        server.queued_tasks = int(payload.get("queued_tasks", 0))
        server.processing_tasks = int(payload.get("processing_tasks", 0))
        server.completed_tasks = int(payload.get("completed_tasks", 0))
        server.failed_tasks = int(payload.get("failed_tasks", 0))
        server.max_concurrent_requests = int(payload.get("max_concurrent_requests", 0))
        if server.max_concurrent_requests <= 0:
            raise ValueError("max_concurrent_requests must be a positive integer")
        server.processing_window_size = max(
            MIN_HEALTHY_PROCESSING_WINDOW_SIZE,
            int(
                payload.get(
                    "processing_window_size",
                    MIN_HEALTHY_PROCESSING_WINDOW_SIZE,
                )
            ),
        )
        server.healthy = payload.get("status") == "healthy"
        server.last_error = (
            None if server.healthy else json.dumps(payload, ensure_ascii=False)
        )
        server.consecutive_health_failures = (
            0 if server.healthy else server.consecutive_health_failures + 1
        )

    async def _refresh_server(self, server: WorkerState) -> None:
        if server.local_server is not None:
            if not server.local_server.is_running():
                try:
                    await server.local_server.restart(self.client)
                    server.base_url = normalize_base_url(server.local_server.base_url or "")
                except Exception as exc:
                    server.healthy = False
                    server.last_error = str(exc)
                    server.last_checked_at = utc_now_iso()
                    server.consecutive_health_failures += 1
                    return
            elif server.local_server.base_url is not None:
                server.base_url = normalize_base_url(server.local_server.base_url)

        if not server.base_url:
            server.healthy = False
            server.last_error = "Upstream base_url is not configured"
            server.last_checked_at = utc_now_iso()
            server.consecutive_health_failures += 1
            return

        try:
            response = await self.client.get(f"{server.base_url}{HEALTH_ENDPOINT}")
        except httpx.HTTPError as exc:
            server.healthy = False
            server.last_error = str(exc)
            server.last_checked_at = utc_now_iso()
            server.consecutive_health_failures += 1
            if (
                server.local_server is not None
                and server.consecutive_health_failures
                >= WORKER_HEALTH_FAILURE_RESTART_THRESHOLD
            ):
                await self._restart_local_server(server)
            return

        server.last_checked_at = utc_now_iso()
        if response.status_code != 200:
            server.healthy = False
            server.last_error = response_detail(response)
            server.consecutive_health_failures += 1
            if (
                server.local_server is not None
                and server.consecutive_health_failures
                >= WORKER_HEALTH_FAILURE_RESTART_THRESHOLD
            ):
                await self._restart_local_server(server)
            return

        try:
            payload = _parse_json_object_response(response, "health payload")
            self._update_server_from_health_payload(server, payload)
        except (TypeError, ValueError) as exc:
            server.healthy = False
            server.last_error = f"Invalid health payload: {exc}"
            server.consecutive_health_failures += 1
            if (
                server.local_server is not None
                and server.consecutive_health_failures
                >= WORKER_HEALTH_FAILURE_RESTART_THRESHOLD
            ):
                await self._restart_local_server(server)
            return

    async def _restart_local_server(self, server: WorkerState) -> None:
        local_server = server.local_server
        if local_server is None:
            return
        try:
            await local_server.restart(self.client)
            server.base_url = normalize_base_url(local_server.base_url or "")
            response = await self.client.get(f"{server.base_url}{HEALTH_ENDPOINT}")
            if response.status_code == 200:
                server.last_checked_at = utc_now_iso()
                payload = _parse_json_object_response(response, "health payload")
                self._update_server_from_health_payload(server, payload)
        except (TypeError, ValueError) as exc:
            server.healthy = False
            server.last_error = f"Invalid health payload: {exc}"
            server.last_checked_at = utc_now_iso()
        except Exception as exc:
            server.healthy = False
            server.last_error = str(exc)
            server.last_checked_at = utc_now_iso()

    async def acquire_submission_server(self, excluded_server_ids: Optional[set[str]] = None) -> WorkerState | None:
        excluded = excluded_server_ids or set()
        async with self._selection_lock:
            candidates = [
                server
                for server in self.servers
                if server.healthy and server.server_id not in excluded
            ]
            if not candidates:
                return None

            randomized = list(candidates)
            self.randomizer.shuffle(randomized)
            randomized.sort(
                key=lambda item: (
                    item.score(),
                    item.pending_assignments,
                    0 if item.source == SOURCE_LOCAL else 1,
                )
            )
            selected = randomized[0]
            selected.pending_assignments += 1
            return selected

    async def release_submission_server(self, server_id: str) -> None:
        async with self._selection_lock:
            server = self._servers.get(server_id)
            if server is None:
                return
            if server.pending_assignments > 0:
                server.pending_assignments -= 1

    async def mark_submission_failure(self, server_id: str, error: str) -> None:
        async with self._selection_lock:
            server = self._servers.get(server_id)
            if server is None:
                return
            server.healthy = False
            server.last_error = error
            server.last_checked_at = utc_now_iso()

    def get_server(self, server_id: str) -> WorkerState | None:
        return self._servers.get(server_id)

    def health_payload(self) -> tuple[bool, dict[str, Any]]:
        servers = [server.snapshot() for server in self.servers]
        healthy_servers = [server for server in self.servers if server.healthy]
        payload = {
            "status": "healthy" if healthy_servers else "unhealthy",
            "version": __version__,
            "protocol_version": API_PROTOCOL_VERSION,
            "queued_tasks": sum(server.queued_tasks for server in self.servers),
            "processing_tasks": sum(server.processing_tasks for server in self.servers),
            "completed_tasks": sum(server.completed_tasks for server in self.servers),
            "failed_tasks": sum(server.failed_tasks for server in self.servers),
            "max_concurrent_requests": sum(
                server.max_concurrent_requests for server in healthy_servers
            ),
            "processing_window_size": min(
                (server.processing_window_size for server in healthy_servers),
                default=MIN_HEALTHY_PROCESSING_WINDOW_SIZE,
            ),
            "servers": servers,
        }
        if not healthy_servers:
            payload["error"] = "No healthy upstream MinerU API servers are available"
        return bool(healthy_servers), payload


class RouterTaskRegistry:
    def __init__(
        self,
        *,
        task_retention_seconds: int,
        cleanup_interval_seconds: int,
    ):
        self.task_retention_seconds = task_retention_seconds
        self.cleanup_interval_seconds = cleanup_interval_seconds
        self._tasks: dict[str, RouterTaskRecord] = {}
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task[Any] | None = None

    async def start(self) -> None:
        if self.task_retention_seconds <= 0:
            return
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop(), name="mineru-router-task-cleanup")

    async def shutdown(self) -> None:
        if self._cleanup_task is not None:
            self._cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None

    async def _cleanup_loop(self) -> None:
        while True:
            await asyncio.sleep(self.cleanup_interval_seconds)
            await self.cleanup_expired_tasks()

    async def register(
        self,
        *,
        upstream_server_id: str,
        upstream_base_url: str,
        upstream_task_id: str,
        backend: str,
        file_names: list[str],
        created_at: str,
        status: str,
        started_at: Optional[str],
        completed_at: Optional[str],
        error: Optional[str],
        queued_ahead: int | None,
    ) -> RouterTaskRecord:
        task = RouterTaskRecord(
            task_id=str(uuid.uuid4()),
            upstream_server_id=upstream_server_id,
            upstream_task_id=upstream_task_id,
            upstream_base_url=upstream_base_url,
            backend=backend,
            file_names=file_names,
            created_at=created_at,
            status=status,
            started_at=started_at,
            completed_at=completed_at,
            error=error,
            queued_ahead=queued_ahead,
        )
        async with self._lock:
            self._tasks[task.task_id] = task
        return task

    async def get(self, task_id: str) -> RouterTaskRecord | None:
        async with self._lock:
            return self._tasks.get(task_id)

    async def update_from_upstream_payload(
        self,
        task_id: str,
        payload: dict[str, Any],
    ) -> RouterTaskRecord | None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = str(payload.get("status", task.status))
            task.backend = str(payload.get("backend", task.backend))
            file_names = payload.get("file_names")
            if isinstance(file_names, list) and all(isinstance(item, str) for item in file_names):
                task.file_names = list(file_names)
            task.created_at = str(payload.get("created_at", task.created_at))
            task.started_at = payload.get("started_at") if payload.get("started_at") is None else str(payload.get("started_at"))
            task.completed_at = payload.get("completed_at") if payload.get("completed_at") is None else str(payload.get("completed_at"))
            task.error = payload.get("error") if payload.get("error") is None else str(payload.get("error"))
            queued_ahead = payload.get("queued_ahead")
            task.queued_ahead = queued_ahead if isinstance(queued_ahead, int) else None
            task.upstream_error_count = 0
            return task

    async def increment_upstream_error(
        self,
        task_id: str,
        error: str,
    ) -> RouterTaskRecord | None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.upstream_error_count += 1
            if task.upstream_error_count >= UPSTREAM_FAILURE_THRESHOLD:
                task.status = TASK_FAILED
                task.error = error
                task.completed_at = utc_now_iso()
            return task

    async def mark_failed(self, task_id: str, error: str) -> RouterTaskRecord | None:
        async with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return None
            task.status = TASK_FAILED
            task.error = error
            task.completed_at = utc_now_iso()
            return task

    async def cleanup_expired_tasks(self) -> int:
        if self.task_retention_seconds <= 0:
            return 0

        now = datetime.now(timezone.utc)
        async with self._lock:
            expired_ids = [
                task_id
                for task_id, task in self._tasks.items()
                if self._is_task_expired(task, now)
            ]
            for task_id in expired_ids:
                self._tasks.pop(task_id, None)
        return len(expired_ids)

    def _is_task_expired(self, task: RouterTaskRecord, now: datetime) -> bool:
        if task.status not in TASK_TERMINAL_STATES or not task.completed_at:
            return False
        try:
            completed_at = datetime.fromisoformat(task.completed_at)
        except ValueError:
            return False
        if completed_at.tzinfo is None:
            completed_at = completed_at.replace(tzinfo=timezone.utc)
        return (now - completed_at).total_seconds() >= self.task_retention_seconds


def warn_if_router_preload_ignored(settings: RouterSettings) -> None:
    if settings.enable_vlm_preload and not parse_local_gpus(settings.local_gpus):
        logger.warning(
            "Ignoring --enable-vlm-preload because mineru-router is not launching any local mineru-api workers."
        )


async def startup_router_state(app: FastAPI, settings: RouterSettings) -> None:
    http_client = httpx.AsyncClient(
        timeout=build_http_timeout(),
        follow_redirects=True,
    )
    worker_pool = WorkerPool(settings, http_client)
    registry = RouterTaskRegistry(
        task_retention_seconds=settings.task_retention_seconds,
        cleanup_interval_seconds=settings.task_cleanup_interval_seconds,
    )

    try:
        await registry.start()
        await worker_pool.start()
        healthy, payload = worker_pool.health_payload()
        if not healthy:
            raise RuntimeError(
                payload.get(
                    "error",
                    "No healthy upstream MinerU API servers are available",
                )
            )
    except Exception:
        await registry.shutdown()
        await worker_pool.shutdown()
        await http_client.aclose()
        raise

    app.state.http_client = http_client
    app.state.worker_pool = worker_pool
    app.state.router_task_registry = registry


async def shutdown_router_state(app: FastAPI) -> None:
    registry = getattr(app.state, "router_task_registry", None)
    worker_pool = getattr(app.state, "worker_pool", None)
    http_client = getattr(app.state, "http_client", None)

    if registry is not None:
        await registry.shutdown()
    if worker_pool is not None:
        await worker_pool.shutdown()
    if http_client is not None:
        await http_client.aclose()

    app.state.router_task_registry = None
    app.state.worker_pool = None
    app.state.http_client = None


class UpstreamSubmissionUnavailable(RuntimeError):
    pass


class UpstreamSubmissionRejected(RuntimeError):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def build_upload_destination(upload_dir: str, filename: str) -> Path:
    destination = Path(upload_dir) / filename
    if not destination.exists():
        return destination

    base_name = Path(filename).stem
    suffix = Path(filename).suffix
    index = 2
    while True:
        candidate = Path(upload_dir) / f"{base_name}__upload_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


async def stage_multipart_request(request: Request) -> MultipartPayload:
    temp_dir = tempfile.mkdtemp(prefix="mineru-router-request-")
    uploads: list[StagedUpload] = []
    fields: list[tuple[str, str]] = []

    try:
        form = await request.form()
        for key, value in form.multi_items():
            if isinstance(value, StarletteUploadFile):
                original_name = value.filename or f"upload-{uuid.uuid4()}"
                filename = normalize_upload_filename(original_name)
                destination = build_upload_destination(temp_dir, filename)
                with open(destination, "wb") as handle:
                    while True:
                        chunk = await value.read(1 << 20)
                        if not chunk:
                            break
                        handle.write(chunk)
                uploads.append(
                    StagedUpload(
                        field_name=key,
                        upload_name=original_name,
                        content_type=value.content_type or "application/octet-stream",
                        path=str(destination),
                    )
                )
                await value.close()
            else:
                fields.append((key, str(value)))
    except Exception:
        cleanup_path(temp_dir)
        raise

    return MultipartPayload(temp_dir=temp_dir, fields=fields, uploads=uploads)


def parse_submit_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("MinerU upstream returned an invalid submit payload")
    task_id = payload.get("task_id")
    status = payload.get("status")
    backend = payload.get("backend")
    created_at = payload.get("created_at")
    if not isinstance(task_id, str) or not isinstance(status, str) or not isinstance(backend, str):
        raise ValueError("MinerU upstream returned an invalid submit payload")
    if created_at is not None and not isinstance(created_at, str):
        raise ValueError("MinerU upstream returned an invalid submit payload")
    return {
        "task_id": task_id,
        "status": status,
        "backend": backend,
        "file_names": payload.get("file_names", []),
        "created_at": created_at or utc_now_iso(),
        "started_at": payload.get("started_at"),
        "completed_at": payload.get("completed_at"),
        "error": payload.get("error"),
        "queued_ahead": payload.get("queued_ahead") if isinstance(payload.get("queued_ahead"), int) else None,
    }


def submit_payload_to_upstream_sync(
    base_url: str,
    payload: MultipartPayload,
) -> dict[str, Any]:
    with ExitStack() as stack, httpx.Client(
        timeout=build_http_timeout(),
        follow_redirects=True,
    ) as client:
        multipart = [
            (
                field_name,
                (None, field_value),
            )
            for field_name, field_value in payload.fields
        ]
        multipart.extend(
            [
                (
                    upload.field_name,
                    (
                        upload.upload_name,
                        stack.enter_context(open(upload.path, "rb")),
                        upload.content_type,
                    ),
                )
                for upload in payload.uploads
            ]
        )
        try:
            response = client.post(
                f"{base_url}{TASKS_ENDPOINT}",
                files=multipart,
            )
        except httpx.HTTPError as exc:
            raise UpstreamSubmissionUnavailable(str(exc)) from exc

    if response.status_code == 202:
        try:
            submit_response = _parse_json_object_response(response, "submit payload")
            return parse_submit_response(submit_response)
        except ValueError as exc:
            raise UpstreamSubmissionUnavailable(f"Invalid submit payload: {exc}") from exc
    if response.status_code in HTTP_RETRYABLE_STATUS_CODES:
        raise UpstreamSubmissionUnavailable(
            f"{response.status_code} {response_detail(response)}"
        )
    raise UpstreamSubmissionRejected(response.status_code, response_detail(response))


async def submit_payload_to_upstream(
    base_url: str,
    payload: MultipartPayload,
) -> dict[str, Any]:
    return await asyncio.to_thread(submit_payload_to_upstream_sync, base_url, payload)


async def submit_router_task(
    request: Request,
    payload: MultipartPayload,
) -> RouterTaskRecord:
    validate_public_http_client_request(
        public_bind_exposed=bool(
            getattr(request.app.state, "public_bind_exposed", False)
        ),
        allow_public_http_client=bool(
            getattr(request.app.state, "allow_public_http_client", False)
        ),
        backend=payload.get_field_value("backend") or "",
        server_url=payload.get_field_value("server_url"),
    )
    worker_pool: WorkerPool = request.app.state.worker_pool
    registry: RouterTaskRegistry = request.app.state.router_task_registry
    attempted_servers: set[str] = set()
    last_error: str | None = None

    while True:
        server = await worker_pool.acquire_submission_server(excluded_server_ids=attempted_servers)
        if server is None:
            if last_error is None:
                raise HTTPException(status_code=503, detail="No healthy upstream MinerU API servers are available")
            raise HTTPException(status_code=503, detail=last_error)

        try:
            upstream_payload = await submit_payload_to_upstream(server.base_url, payload)
            file_names = upstream_payload["file_names"]
            normalized_file_names = (
                list(file_names)
                if isinstance(file_names, list) and all(isinstance(item, str) for item in file_names)
                else []
            )
            return await registry.register(
                upstream_server_id=server.server_id,
                upstream_base_url=server.base_url,
                upstream_task_id=upstream_payload["task_id"],
                backend=upstream_payload["backend"],
                file_names=normalized_file_names,
                created_at=upstream_payload["created_at"],
                status=upstream_payload["status"],
                started_at=upstream_payload["started_at"] if isinstance(upstream_payload["started_at"], str) else None,
                completed_at=upstream_payload["completed_at"] if isinstance(upstream_payload["completed_at"], str) else None,
                error=upstream_payload["error"] if isinstance(upstream_payload["error"], str) else None,
                queued_ahead=upstream_payload["queued_ahead"],
            )
        except UpstreamSubmissionRejected as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
        except UpstreamSubmissionUnavailable as exc:
            attempted_servers.add(server.server_id)
            last_error = f"Failed to submit task via {server.server_id}: {exc}"
            await worker_pool.mark_submission_failure(server.server_id, str(exc))
        finally:
            await worker_pool.release_submission_server(server.server_id)


async def fetch_router_task_status(
    request: Request,
    task: RouterTaskRecord,
) -> RouterTaskRecord:
    if is_task_terminal(task.status):
        return task

    registry: RouterTaskRegistry = request.app.state.router_task_registry
    client: httpx.AsyncClient = request.app.state.http_client
    url = f"{task.upstream_base_url}{TASKS_ENDPOINT}/{task.upstream_task_id}"
    try:
        response = await client.get(url)
    except httpx.HTTPError as exc:
        updated = await registry.increment_upstream_error(task.task_id, str(exc))
        if updated is None:
            raise HTTPException(status_code=404, detail="Task not found") from exc
        return updated

    if response.status_code != 200:
        error = f"{response.status_code} {response_detail(response)}"
        updated = await registry.increment_upstream_error(task.task_id, error)
        if updated is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return updated

    try:
        status_payload = _parse_json_object_response(response, "task status payload")
    except ValueError as exc:
        updated = await registry.increment_upstream_error(
            task.task_id,
            f"Invalid task status payload: {exc}",
        )
        if updated is None:
            raise HTTPException(status_code=404, detail="Task not found")
        return updated

    updated = await registry.update_from_upstream_payload(task.task_id, status_payload)
    if updated is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return updated


async def wait_for_router_task_terminal_state(
    request: Request,
    task: RouterTaskRecord,
    timeout_seconds: float = TASK_RESULT_TIMEOUT_SECONDS,
) -> RouterTaskRecord:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    current_task = task
    while asyncio.get_running_loop().time() < deadline:
        current_task = await fetch_router_task_status(request, current_task)
        if is_task_terminal(current_task.status):
            return current_task
        await asyncio.sleep(TASK_STATUS_POLL_INTERVAL_SECONDS)

    raise HTTPException(
        status_code=504,
        detail=f"Timed out waiting for result of task {task.task_id}",
    )


async def proxy_router_task_result(
    request: Request,
    task: RouterTaskRecord,
) -> Response:
    client: httpx.AsyncClient = request.app.state.http_client
    result_url = f"{task.upstream_base_url}{TASKS_ENDPOINT}/{task.upstream_task_id}/result"
    try:
        upstream_response = await client.send(
            client.build_request(
                "GET",
                result_url,
                timeout=build_result_download_timeout(),
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await request.app.state.router_task_registry.increment_upstream_error(task.task_id, str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if upstream_response.status_code != 200:
        body = await upstream_response.aread()
        await upstream_response.aclose()
        detail = body.decode("utf-8", errors="replace").strip() or upstream_response.reason_phrase
        await request.app.state.router_task_registry.increment_upstream_error(
            task.task_id,
            f"{upstream_response.status_code} {detail}",
        )
        raise HTTPException(
            status_code=upstream_response.status_code,
            detail=detail,
        )

    content_type = upstream_response.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await upstream_response.aread()
        await upstream_response.aclose()
        return Response(
            content=body,
            status_code=200,
            media_type="application/json",
        )

    headers: dict[str, str] = {}
    content_disposition = upstream_response.headers.get("content-disposition")
    if content_disposition:
        headers["content-disposition"] = content_disposition
    return StreamingResponse(
        upstream_response.aiter_bytes(),
        status_code=200,
        media_type=content_type or "application/octet-stream",
        headers=headers,
        background=BackgroundTask(upstream_response.aclose),
    )


async def build_sync_router_task_result_response(
    request: Request,
    task: RouterTaskRecord,
) -> Response:
    client: httpx.AsyncClient = request.app.state.http_client
    result_url = f"{task.upstream_base_url}{TASKS_ENDPOINT}/{task.upstream_task_id}/result"
    try:
        upstream_response = await client.send(
            client.build_request(
                "GET",
                result_url,
                timeout=build_result_download_timeout(),
            ),
            stream=True,
        )
    except httpx.HTTPError as exc:
        await request.app.state.router_task_registry.increment_upstream_error(task.task_id, str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if upstream_response.status_code != 200:
        body = await upstream_response.aread()
        await upstream_response.aclose()
        detail = body.decode("utf-8", errors="replace").strip() or upstream_response.reason_phrase
        await request.app.state.router_task_registry.increment_upstream_error(
            task.task_id,
            f"{upstream_response.status_code} {detail}",
        )
        raise HTTPException(status_code=upstream_response.status_code, detail=detail)

    sync_headers = build_sync_task_headers(task, request)
    content_type = upstream_response.headers.get("content-type", "")
    if "application/json" in content_type:
        try:
            await upstream_response.aread()
            payload_data = _parse_json_object_response(
                upstream_response,
                "task result payload",
            )
        except ValueError as exc:
            detail = f"Invalid task result payload: {exc}"
            await request.app.state.router_task_registry.increment_upstream_error(
                task.task_id,
                detail,
            )
            raise HTTPException(status_code=502, detail=detail) from exc
        finally:
            await upstream_response.aclose()

        merged_payload = {
            **task.to_status_payload(request),
            "backend": payload_data.get("backend", task.backend),
            "version": payload_data.get("version", __version__),
            "results": payload_data.get("results", {}),
        }
        return JSONResponse(status_code=200, content=merged_payload, headers=sync_headers)

    headers: dict[str, str] = {}
    content_disposition = upstream_response.headers.get("content-disposition")
    if content_disposition:
        headers["content-disposition"] = content_disposition
    return StreamingResponse(
        upstream_response.aiter_bytes(),
        status_code=200,
        media_type=content_type or "application/octet-stream",
        headers={**headers, **sync_headers},
        background=BackgroundTask(upstream_response.aclose),
    )


def build_sync_task_headers(task: RouterTaskRecord, request: Request) -> dict[str, str]:
    payload = task.to_status_payload(request)
    return {
        FILE_PARSE_TASK_ID_HEADER: task.task_id,
        FILE_PARSE_TASK_STATUS_HEADER: task.status,
        FILE_PARSE_TASK_STATUS_URL_HEADER: payload["status_url"],
        FILE_PARSE_TASK_RESULT_URL_HEADER: payload["result_url"],
    }


def create_app(settings: RouterSettings | None = None) -> FastAPI:
    resolved_settings = settings or RouterSettings.from_env()
    enable_docs = env_flag_enabled("MINERU_API_ENABLE_FASTAPI_DOCS", default=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await startup_router_state(app, resolved_settings)
        try:
            yield
        finally:
            await shutdown_router_state(app)

    app = FastAPI(
        openapi_url="/openapi.json" if enable_docs else None,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        lifespan=lifespan,
    )
    app.state.router_settings = resolved_settings
    configure_public_http_client_policy(
        app,
        public_bind_exposed=env_flag_enabled(
            MINERU_ROUTER_PUBLIC_BIND_EXPOSED_ENV,
            default=False,
        ),
        allow_public_http_client=env_flag_enabled(
            MINERU_ROUTER_ALLOW_PUBLIC_HTTP_CLIENT_ENV,
            default=False,
        ),
    )
    app.add_middleware(GZipMiddleware, minimum_size=1000)

    @app.post(path="/tasks", status_code=202)
    async def submit_parse_task(http_request: Request):
        payload = await stage_multipart_request(http_request)
        try:
            router_task = await submit_router_task(http_request, payload)
        finally:
            payload.cleanup()
        response_payload = router_task.to_status_payload(http_request)
        response_payload["message"] = "Task submitted successfully"
        return JSONResponse(status_code=202, content=response_payload)

    @app.get(path="/tasks/{task_id}", name="get_router_task_status")
    async def get_router_task_status(task_id: str, request: Request):
        registry: RouterTaskRegistry = request.app.state.router_task_registry
        task = await registry.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")
        task = await fetch_router_task_status(request, task)
        return task.to_status_payload(request)

    @app.get(path="/tasks/{task_id}/result", name="get_router_task_result")
    async def get_router_task_result(task_id: str, request: Request):
        registry: RouterTaskRegistry = request.app.state.router_task_registry
        task = await registry.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found")

        task = await fetch_router_task_status(request, task)
        if task.status in {TASK_PENDING, TASK_PROCESSING}:
            return JSONResponse(
                status_code=202,
                content={
                    **task.to_status_payload(request),
                    "message": "Task result is not ready yet",
                },
            )
        if task.status == TASK_FAILED:
            return JSONResponse(
                status_code=409,
                content={
                    **task.to_status_payload(request),
                    "message": "Task execution failed",
                },
            )
        return await proxy_router_task_result(request, task)

    @app.post(path="/file_parse", status_code=200)
    async def file_parse(request: Request):
        payload = await stage_multipart_request(request)
        try:
            router_task = await submit_router_task(request, payload)
        finally:
            payload.cleanup()

        router_task = await wait_for_router_task_terminal_state(request, router_task)
        if router_task.status == TASK_FAILED:
            return JSONResponse(
                status_code=409,
                content={
                    **router_task.to_status_payload(request),
                    "message": "Task execution failed",
                },
            )

        return await build_sync_router_task_result_response(request, router_task)

    @app.get(path="/health")
    async def health_check(request: Request):
        worker_pool: WorkerPool = request.app.state.worker_pool
        healthy, payload = worker_pool.health_payload()
        if healthy:
            return payload
        return JSONResponse(status_code=503, content=payload)

    return app


app = create_app()


@click.command(
    context_settings=dict(ignore_unknown_options=True, allow_extra_args=True)
)
@click.pass_context
@click.option("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
@click.option("--port", default=8002, type=int, help="Server port (default: 8002)")
@click.option("--reload", is_flag=True, help="Enable auto-reload (development mode)")
@click.option(
    "--allow-public-http-client",
    is_flag=True,
    help=(
        "Allow *-http-client backends and server_url even when binding the router "
        "to 0.0.0.0 or ::."
    ),
)
@click.option(
    "--upstream-url",
    "upstream_urls",
    multiple=True,
    help="Existing MinerU FastAPI base URL. Repeat to add multiple upstream servers.",
)
@click.option(
    "--local-gpus",
    default=LOCAL_GPU_AUTO,
    help="Local GPU workers to launch: auto, none, or CSV such as 0,1,2.",
)
@click.option(
    "--worker-host",
    default="127.0.0.1",
    help="Host for router-managed mineru-api workers (default: 127.0.0.1).",
)
@click.option(
    "--enable-vlm-preload",
    "enable_vlm_preload",
    type=bool,
    default=False,
    help="Preload the local VLM model in router-managed mineru-api workers.",
)
def main(
    ctx: click.Context,
    host: str,
    port: int,
    reload: bool,
    allow_public_http_client: bool,
    upstream_urls: tuple[str, ...],
    local_gpus: str,
    worker_host: str,
    enable_vlm_preload: bool,
):
    settings = RouterSettings(
        upstream_urls=tuple(upstream_urls),
        local_gpus=local_gpus,
        worker_host=worker_host,
        enable_vlm_preload=enable_vlm_preload,
        worker_extra_args=tuple(ctx.args),
        task_retention_seconds=get_task_retention_seconds(),
        task_cleanup_interval_seconds=get_task_cleanup_interval_seconds(),
    )
    public_bind_exposed = is_public_bind_host(host)
    warn_if_router_preload_ignored(settings)
    configure_public_http_client_policy(
        app,
        public_bind_exposed=public_bind_exposed,
        allow_public_http_client=allow_public_http_client,
    )
    os.environ["MINERU_ROUTER_UPSTREAM_URLS_JSON"] = json.dumps(list(settings.upstream_urls))
    os.environ["MINERU_ROUTER_LOCAL_GPUS"] = settings.local_gpus
    os.environ["MINERU_ROUTER_WORKER_HOST"] = settings.worker_host
    os.environ["MINERU_ROUTER_ENABLE_VLM_PRELOAD"] = (
        "1" if settings.enable_vlm_preload else "0"
    )
    os.environ["MINERU_ROUTER_WORKER_ARGS_JSON"] = json.dumps(list(settings.worker_extra_args))
    os.environ[MINERU_ROUTER_PUBLIC_BIND_EXPOSED_ENV] = (
        "1" if public_bind_exposed else "0"
    )
    os.environ[MINERU_ROUTER_ALLOW_PUBLIC_HTTP_CLIENT_ENV] = (
        "1" if allow_public_http_client else "0"
    )
    warn_if_public_http_client_policy(host, allow_public_http_client)

    access_log = not env_flag_enabled("MINERU_API_DISABLE_ACCESS_LOG")
    print(f"Start MinerU Router Service: http://{host}:{port}")
    print(f"API documentation: http://{host}:{port}/docs")

    if reload:
        uvicorn.run(
            "mineru.cli.router:app",
            host=host,
            port=port,
            reload=True,
            access_log=access_log,
        )
        return

    configured_app = create_app(settings)
    uvicorn.run(
        configured_app,
        host=host,
        port=port,
        reload=False,
        access_log=access_log,
    )


if __name__ == "__main__":
    main()
