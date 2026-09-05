"""Gradio 使用的 MinerU V1 能力发现与本地服务生命周期。"""

from __future__ import annotations

import atexit
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, cast

import httpx

from ...parser.api_client import ApiJobStatus, MinerUApiParser, should_trust_env_for_url
from ...parser.base import ParseResult
from ...parser.page_range import normalize_page_range_input
from ...parser.process_control import ManagedProcessControl
from ...types import SERVER_TIERS, TIERS, ServerTier, Tier
from ...utils.stdio import utf8_subprocess_env
from .status import (
    STATUS_CHECKING_SERVER,
    STATUS_COMPLETED,
    STATUS_DOWNLOADING_RESULT,
    STATUS_PREPARING_REQUEST,
    STATUS_PROCESSING_ON_SERVER,
    STATUS_PROCESSING_OUTPUT,
    STATUS_QUEUED_ON_SERVER,
    STATUS_SUBMITTING_TASK,
)


class V1ArtifactError(RuntimeError):
    """表示 Gradio 访问 V1 API 或读取解析产物时的稳定错误。"""

    def __init__(self, message: str, *, code: str = "v1_api_error") -> None:
        """保存可供状态面板展示的错误消息和协议错误码。"""
        self.code = code
        super().__init__(message)


@dataclass(frozen=True)
class V1ServerCapabilities:
    """记录一次 V1 server 能力发现结果，供 UI 构造控件使用。"""

    base_url: str
    tiers: tuple[str, ...]
    output_formats: tuple[str, ...]
    sources: tuple[str, ...]


def normalize_v1_base_url(value: str) -> str:
    """规范化 V1 API base URL，并拒绝空地址和缺少 scheme 的输入。"""
    normalized = str(value).strip().rstrip("/")
    if not normalized:
        raise V1ArtifactError("V1 API URL must not be empty", code="invalid_api_url")
    if "://" not in normalized:
        normalized = f"http://{normalized}"
    try:
        parsed = httpx.URL(normalized)
    except (httpx.InvalidURL, ValueError) as exc:
        raise V1ArtifactError(f"Invalid V1 API URL: {value}", code="invalid_api_url") from exc
    if parsed.scheme not in {"http", "https"} or not parsed.host:
        raise V1ArtifactError(f"Unsupported V1 API URL: {value}", code="invalid_api_url")
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        raise V1ArtifactError("V1 API URL must not contain credentials, query, or fragment", code="invalid_api_url")
    return str(parsed).rstrip("/")


def _error_detail(response: httpx.Response) -> tuple[str, str]:
    """从 V1 error envelope 提取稳定 code/message，避免把整段 HTML 返回给用户。"""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        code = str(error.get("code") or f"http_{response.status_code}")
        message = str(error.get("message") or response.text[:500] or "V1 API request failed")
        return code, message
    return f"http_{response.status_code}", response.text[:500] or "V1 API request failed"


def _check_json_response(response: httpx.Response, *, endpoint: str) -> dict[str, Any]:
    """校验 V1 JSON 响应并将协议错误转换为 Gradio client 异常。"""
    if response.status_code >= 400:
        code, message = _error_detail(response)
        raise V1ArtifactError(f"{endpoint}: {message}", code=code)
    try:
        payload = response.json()
    except ValueError as exc:
        raise V1ArtifactError(f"{endpoint} did not return JSON", code="invalid_response") from exc
    if not isinstance(payload, dict):
        raise V1ArtifactError(f"{endpoint} returned a non-object JSON payload", code="invalid_response")
    return payload


class V1ArtifactClient:
    """复用 `MinerUApiParser` 的 V1 请求流程并提供 Gradio 能力发现。"""

    def __init__(
        self,
        *,
        api_url: str,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """创建不保存文档内容的 V1 client；API Key 只保存在当前进程内存。"""
        self.base_url = normalize_v1_base_url(api_url)
        self.api_key = api_key if api_key is not None else os.environ.get("MINERU_API_KEY")
        self._transport = transport
        self._capabilities: V1ServerCapabilities | None = None

    @property
    def capabilities(self) -> V1ServerCapabilities | None:
        """返回最近一次能力发现快照。"""
        return self._capabilities

    async def discover(self) -> V1ServerCapabilities:
        """调用 `/v1/health` 与 `/v1/tiers`，校验 Gradio 所需的 V1 能力。"""
        trust_env = should_trust_env_for_url(self.base_url)
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(30, connect=10),
                follow_redirects=True,
                trust_env=trust_env,
                transport=self._transport,
            ) as client:
                health_response = await client.get(f"{self.base_url}/v1/health", headers=self._headers())
                health = _check_json_response(health_response, endpoint="GET /v1/health")
                tiers_response = await client.get(f"{self.base_url}/v1/tiers", headers=self._headers())
                tiers_payload = _check_json_response(tiers_response, endpoint="GET /v1/tiers")
        except V1ArtifactError:
            raise
        except httpx.HTTPError as exc:
            raise V1ArtifactError(f"Unable to reach V1 API at {self.base_url}: {exc}", code="server_unreachable") from exc

        features = health.get("features")
        if health.get("status") != "ok":
            raise V1ArtifactError("V1 health response did not report status=ok", code="invalid_response")
        if not isinstance(features, dict):
            raise V1ArtifactError("V1 health response is missing features", code="invalid_response")
        output_formats = _string_tuple(features.get("output_formats"), "features.output_formats")
        sources = _string_tuple(features.get("sources"), "features.sources")
        if "zip" not in output_formats:
            raise V1ArtifactError(
                "The configured V1 API server does not advertise zip output, which is required by Gradio.",
                code="unsupported_output_format",
            )
        if "file_id" not in sources:
            raise V1ArtifactError(
                "The configured V1 API server does not advertise file_id uploads, which are required by Gradio.",
                code="unsupported_source",
            )

        raw_tiers = tiers_payload.get("data")
        if not isinstance(raw_tiers, list):
            raise V1ArtifactError("V1 tiers response is missing data", code="invalid_response")
        tiers: list[str] = []
        for item in raw_tiers:
            tier_id = item.get("id") if isinstance(item, dict) else None
            if isinstance(tier_id, str) and tier_id in TIERS and tier_id not in tiers:
                tiers.append(tier_id)
        if not tiers:
            raise V1ArtifactError("The V1 API server did not advertise a supported parsing tier", code="tier_unavailable")

        capabilities = V1ServerCapabilities(
            base_url=self.base_url,
            tiers=tuple(tiers),
            output_formats=output_formats,
            sources=sources,
        )
        self._capabilities = capabilities
        return capabilities

    async def parse_file(
        self,
        path: Path,
        *,
        tier: str,
        page_range: str,
        ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
        status_callback: Callable[[str], None] | None = None,
    ) -> ParseResult:
        """通过 V1 API 解析单个文件，并返回带图片/模型输出的 `ParseResult`。"""
        page_range = normalize_page_range_input(page_range)
        if not path.is_file():
            raise FileNotFoundError(path)
        if tier not in TIERS:
            raise V1ArtifactError(f"Unsupported parse tier: {tier}", code="invalid_request")

        emit = status_callback or (lambda _message: None)
        emit(STATUS_PREPARING_REQUEST)
        if self._capabilities is None:
            emit(STATUS_CHECKING_SERVER)
            await self.discover()
        if self._capabilities is None or tier not in self._capabilities.tiers:
            raise V1ArtifactError(f"Tier '{tier}' is not available in the configured V1 API server", code="tier_unavailable")

        emit(STATUS_SUBMITTING_TASK)
        parser = MinerUApiParser(
            api_url=self.base_url,
            api_key=self.api_key,
            tier=cast(Tier, tier),
            ocr_mode=ocr_mode,
            include_images=True,
            include_model_output=True,
        )

        def on_job_status(status: ApiJobStatus) -> None:
            """将真实 V1 状态映射到卡片阶段，服务端完成仅代表开始下载。"""
            messages: dict[ApiJobStatus, str] = {
                "queued": STATUS_QUEUED_ON_SERVER,
                "running": STATUS_PROCESSING_ON_SERVER,
                "completed": STATUS_DOWNLOADING_RESULT,
                "partial": STATUS_DOWNLOADING_RESULT,
                "failed": "Failed: server task failed",
                "canceled": "Failed: server task canceled",
            }
            emit(messages[status])

        try:
            result = await parser.parse_async(path, page_range=page_range, status_callback=on_job_status)
        except Exception as exc:
            emit(f"Failed: {exc}")
            if isinstance(exc, (FileNotFoundError, V1ArtifactError)):
                raise
            raise V1ArtifactError(str(exc), code=getattr(exc, "code", "parse_failed")) from exc
        return result

    def _headers(self) -> dict[str, str]:
        """生成 V1 请求鉴权头，不把 API Key 写入日志或持久化状态。"""
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}


class GradioArtifactClient:
    """合并已发现的服务能力，并将缺失的 Flash 请求路由到本地 V1 服务。"""

    def __init__(self, primary: V1ArtifactClient, *, local_flash: V1ArtifactClient | None = None) -> None:
        """根据独立能力快照建立明确的档位映射，不修改任何服务声明。"""
        capabilities = primary.capabilities
        if capabilities is None:
            raise ValueError("Discover the primary V1 API capabilities before constructing the Gradio client")
        self._clients_by_tier = dict.fromkeys(capabilities.tiers, primary)
        if local_flash is not None and "flash" not in self._clients_by_tier:
            if local_flash.capabilities is None or "flash" not in local_flash.capabilities.tiers:
                raise V1ArtifactError("The local V1 API server did not advertise Flash", code="tier_unavailable")
            self._clients_by_tier["flash"] = local_flash
        self._capabilities = replace(capabilities, tiers=tuple(tier for tier in TIERS if tier in self._clients_by_tier))

    @property
    def capabilities(self) -> V1ServerCapabilities:
        """返回供界面使用的有效档位集合，主服务地址和协议能力保持原值。"""
        return self._capabilities

    async def parse_file(
        self,
        path: Path,
        *,
        tier: str,
        page_range: str,
        ocr_mode: Literal["auto", "txt", "ocr"] = "auto",
        status_callback: Callable[[str], None] | None = None,
    ) -> ParseResult:
        """仅向该档位对应的服务提交文件，复用原有状态、取消和产物流程。"""
        if tier not in TIERS:
            raise V1ArtifactError(f"Unsupported parse tier: {tier}", code="invalid_request")
        client = self._clients_by_tier.get(tier)
        if client is None:
            raise V1ArtifactError(f"Tier '{tier}' is not available in Gradio", code="tier_unavailable")
        return await client.parse_file(
            path, tier=tier, page_range=page_range, ocr_mode=ocr_mode, status_callback=status_callback
        )


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    """把能力字段严格收敛为字符串元组。"""
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise V1ArtifactError(f"V1 health response field {field_name} is invalid", code="invalid_response")
    return tuple(value)


def _reserve_local_port() -> int:
    """在 loopback 上申请一个临时 TCP 端口，供托管 API server 使用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class ManagedLocalApiServer:
    """管理 Gradio 自动启动的独立 V1 API server 子进程。"""

    def __init__(
        self,
        *,
        tier: ServerTier = "standard",
        concurrency: int = 1,
        language: str = "ch",
        disable_image_analysis: bool = False,
        preload_models: bool = False,
        api_key: str | None = None,
    ) -> None:
        """保存显式启动配置，但不在构造阶段创建目录或启动进程。"""
        if tier not in SERVER_TIERS:
            raise ValueError(f"Unsupported API server tier: {tier}")
        if concurrency <= 0:
            raise ValueError("API server concurrency must be positive")
        self.tier = tier
        self.concurrency = concurrency
        self.language = language
        self.disable_image_analysis = disable_image_analysis
        self.preload_models = preload_models
        self.api_key = api_key
        self.base_url: str | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self._control: ManagedProcessControl | None = None
        self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
        self._atexit_registered = False

    def start(self) -> str:
        """启动 `mineru-kit api-server` 并等待 `/v1/health` 返回成功。"""
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Managed V1 API server is already running")
        if self.process is not None or self._temp_dir is not None or self._control is not None:
            self.stop()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mineru-kit-gradio-api-")
        upload_dir = Path(self._temp_dir.name) / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        port = _reserve_local_port()
        self.base_url = f"http://127.0.0.1:{port}"

        control = ManagedProcessControl.create()
        control.start_accepting()
        environment = utf8_subprocess_env()
        environment.update(control.child_env())
        command = self._command(port, upload_dir)
        try:
            self.process = subprocess.Popen(
                command,
                cwd=os.getcwd(),
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=None,
                stderr=None,
                **_new_session_kwargs(),
            )
        except Exception:
            control.close()
            self._cleanup_temp_dir()
            raise
        self._control = control
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True
        try:
            self._wait_until_ready()
        except Exception:
            self.stop()
            raise
        return self.base_url

    def stop(self) -> None:
        """请求子进程优雅退出，并在超时后清理进程和临时目录。"""
        process = self.process
        control = self._control
        self.process = None
        self._control = None
        try:
            if process is not None and process.poll() is None:
                if control is not None:
                    control.request_shutdown(3.0)
                try:
                    process.wait(timeout=8.0)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=3.0)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        finally:
            if control is not None:
                control.close()
            if self._atexit_registered:
                try:
                    atexit.unregister(self.stop)
                except Exception:
                    pass
                self._atexit_registered = False
            self.base_url = None
            self._cleanup_temp_dir()

    def __enter__(self) -> "ManagedLocalApiServer":
        """进入上下文时启动本地 API server。"""
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        """退出上下文时停止本地 API server。"""
        self.stop()

    def _command(self, port: int, upload_dir: Path) -> list[str]:
        """构造不包含 Gradio 参数的正式 API server 子进程命令。"""
        return [
            sys.executable,
            "-m",
            "mineru.kit.main",
            "api-server",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--upload-dir",
            str(upload_dir),
            "--tier",
            self.tier,
            "--concurrency",
            str(self.concurrency),
            "--language",
            self.language,
            *(["--disable-image-analysis"] if self.disable_image_analysis else []),
            *(["--preload-models"] if self.preload_models else []),
            *(["--api-key", self.api_key] if self.api_key else []),
        ]

    def _wait_until_ready(self) -> None:
        """轮询本地 health，区分进程提前退出和服务未就绪。"""
        if self.base_url is None:
            raise RuntimeError("Managed V1 API server has no base URL")
        timeout_seconds = _startup_timeout_seconds()
        deadline = time.monotonic() + timeout_seconds
        last_error = ""
        with httpx.Client(timeout=httpx.Timeout(3, connect=1), trust_env=False) as client:
            while time.monotonic() < deadline:
                if self.process is not None and self.process.poll() is not None:
                    raise V1ArtifactError("Managed V1 API server exited before becoming ready", code="server_start_failed")
                try:
                    response = client.get(f"{self.base_url}/v1/health")
                    if response.status_code < 400:
                        return
                    error_code, last_error = _error_detail(response)
                    if response.status_code < 500 or error_code.startswith("model_preload_"):
                        raise V1ArtifactError(last_error, code=error_code)
                except httpx.HTTPError as exc:
                    last_error = str(exc)
                time.sleep(0.1)
        suffix = f": {last_error}" if last_error else ""
        raise V1ArtifactError(f"Timed out waiting for managed V1 API server{suffix}", code="server_start_timeout")

    def _cleanup_temp_dir(self) -> None:
        """清理托管 server 的上传临时目录。"""
        if self._temp_dir is None:
            return
        self._temp_dir.cleanup()
        self._temp_dir = None


def _startup_timeout_seconds() -> float:
    """读取本地 API 启动超时环境变量，并对非法值使用安全默认值。"""
    raw = os.environ.get("MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS", "300")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 300.0
    return max(1.0, value)


def _new_session_kwargs() -> dict[str, Any]:
    """返回跨平台隔离托管子进程的 Popen 参数。"""
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return {"creationflags": creationflags} if creationflags else {}
    return {"start_new_session": True}


__all__ = [
    "GradioArtifactClient",
    "ManagedLocalApiServer",
    "STATUS_CHECKING_SERVER",
    "STATUS_COMPLETED",
    "STATUS_DOWNLOADING_RESULT",
    "STATUS_PREPARING_REQUEST",
    "STATUS_PROCESSING_ON_SERVER",
    "STATUS_PROCESSING_OUTPUT",
    "STATUS_SUBMITTING_TASK",
    "V1ArtifactClient",
    "V1ArtifactError",
    "V1ServerCapabilities",
    "normalize_v1_base_url",
]
