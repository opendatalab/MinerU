import asyncio
import atexit
import json
import mimetypes
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Optional, Sequence

import click
import httpx
from loguru import logger

from mineru.cli.api_protocol import (
    API_PROTOCOL_VERSION,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
)
from mineru.utils.config_reader import (
    get_max_concurrent_requests as read_max_concurrent_requests,
)

HEALTH_ENDPOINT = "/health"
TASKS_ENDPOINT = "/tasks"
TASK_STATUS_POLL_INTERVAL_SECONDS = 1.0
TASK_RESULT_TIMEOUT_SECONDS = 3600
LOCAL_API_SHUTDOWN_TIMEOUT_SECONDS = 10
LOCAL_API_CLEANUP_RETRIES = 8
LOCAL_API_CLEANUP_RETRY_INTERVAL_SECONDS = 0.25


def get_float_env(name: str, default: float, minimum: float = 0.0) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        resolved = float(value)
    except ValueError:
        logger.warning(
            "Invalid {} value: {}. Expected a number, using default {}.",
            name,
            value,
            default,
        )
        return default
    if resolved < minimum:
        logger.warning(
            "Invalid {} value: {}. Expected a number >= {}, using default {}.",
            name,
            value,
            minimum,
            default,
        )
        return default
    return resolved


def get_local_api_startup_timeout_seconds(default: float = 300.0) -> float:
    return get_float_env(
        "MINERU_LOCAL_API_STARTUP_TIMEOUT_SECONDS",
        default,
        minimum=1.0,
    )


LOCAL_API_STARTUP_TIMEOUT_SECONDS = get_local_api_startup_timeout_seconds()


@dataclass(frozen=True)
class UploadAsset:
    path: Path
    upload_name: str


@dataclass(frozen=True)
class ServerHealth:
    base_url: str
    max_concurrent_requests: int
    processing_window_size: int


@dataclass(frozen=True)
class SubmitResponse:
    task_id: str
    status_url: str
    result_url: str
    file_names: tuple[str, ...] = ()
    queued_ahead: int | None = None


@dataclass(frozen=True)
class TaskStatusSnapshot:
    status: str
    queued_ahead: int | None = None


class LocalAPIServer:
    def __init__(self, extra_cli_args: Sequence[str] = ()):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="mineru-api-client-")
        self.temp_root = Path(self.temp_dir.name)
        self.output_root = self.temp_root / "output"
        self.base_url: str | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self._atexit_registered = False
        self.extra_cli_args = tuple(extra_cli_args)
        # On Windows, the temporary FastAPI child process can stall during parsing
        # startup when launched with stdin=PIPE and an EOF-based shutdown watcher.
        # Use explicit process termination there instead of stdin-driven shutdown.
        self._use_stdin_shutdown_watcher = os.name != "nt"

    def start(self) -> str:
        if self.process is not None:
            raise RuntimeError("Local API server is already running")

        resolved_port = find_free_port()
        remaining_cli_args = strip_local_api_network_args(self.extra_cli_args)
        self.base_url = f"http://127.0.0.1:{resolved_port}"
        env = os.environ.copy()
        env["MINERU_API_OUTPUT_ROOT"] = str(self.output_root)
        env["MINERU_API_MAX_CONCURRENT_REQUESTS"] = str(
            read_max_concurrent_requests(default=DEFAULT_MAX_CONCURRENT_REQUESTS)
        )
        env["MINERU_API_DISABLE_ACCESS_LOG"] = "1"
        if self._use_stdin_shutdown_watcher:
            env["MINERU_API_SHUTDOWN_ON_STDIN_EOF"] = "1"
            stdin_target = subprocess.PIPE
        else:
            env.pop("MINERU_API_SHUTDOWN_ON_STDIN_EOF", None)
            stdin_target = subprocess.DEVNULL
        self.output_root.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "mineru.cli.fast_api",
            "--host",
            "127.0.0.1",
            "--port",
            str(resolved_port),
            *remaining_cli_args,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=os.getcwd(),
            env=env,
            stdin=stdin_target,
        )

        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True
        return self.base_url

    def stop(self) -> None:
        process = self.process
        self.process = None
        try:
            if process is not None and process.poll() is None:
                if self._use_stdin_shutdown_watcher:
                    if process.stdin is not None and not process.stdin.closed:
                        process.stdin.close()
                    try:
                        process.wait(timeout=LOCAL_API_SHUTDOWN_TIMEOUT_SECONDS)
                        return
                    except subprocess.TimeoutExpired:
                        logger.debug(
                            "Local mineru-api did not stop after stdin EOF within {}s. Falling back to SIGTERM.",
                            LOCAL_API_SHUTDOWN_TIMEOUT_SECONDS,
                        )
                else:
                    logger.debug(
                        "Stopping local mineru-api with process termination on Windows."
                    )

                process.terminate()
                try:
                    process.wait(timeout=LOCAL_API_SHUTDOWN_TIMEOUT_SECONDS)
                    return
                except subprocess.TimeoutExpired:
                    pass
                process.kill()
                process.wait(timeout=LOCAL_API_SHUTDOWN_TIMEOUT_SECONDS)
        finally:
            if self._atexit_registered:
                try:
                    atexit.unregister(self.stop)
                except Exception:
                    pass
                self._atexit_registered = False
            self._cleanup_temp_dir()

    def _cleanup_temp_dir(self) -> None:
        last_error: Exception | None = None
        for attempt in range(LOCAL_API_CLEANUP_RETRIES):
            try:
                self.temp_dir.cleanup()
                return
            except FileNotFoundError:
                return
            except Exception as exc:
                last_error = exc
                if attempt + 1 < LOCAL_API_CLEANUP_RETRIES:
                    time.sleep(LOCAL_API_CLEANUP_RETRY_INTERVAL_SECONDS)

        if last_error is not None:
            logger.warning(
                "Failed to clean up temporary MinerU API directory {}: {}. "
                "You can remove it manually after processes release any open handles.",
                self.temp_root,
                last_error,
            )


class ReusableLocalAPIServer:
    def __init__(self, extra_cli_args: Sequence[str] = ()):
        self._lock = threading.Lock()
        self._server: LocalAPIServer | None = None
        self._extra_cli_args = tuple(extra_cli_args)

    def configure(self, extra_cli_args: Sequence[str]) -> None:
        with self._lock:
            self._extra_cli_args = tuple(extra_cli_args)
            server = self._server
            if server is None:
                return
            if server.process is not None and server.process.poll() is None:
                return
            self._server = None

    def ensure_started(self) -> tuple[LocalAPIServer, bool]:
        with self._lock:
            server = self._server
            if server is not None and server.process is not None and server.process.poll() is None:
                return server, False

            if server is not None:
                server.stop()

            server = LocalAPIServer(extra_cli_args=self._extra_cli_args)
            server.start()
            self._server = server
            return server, True

    def stop(self) -> None:
        with self._lock:
            server = self._server
            self._server = None
        if server is not None:
            server.stop()


def build_http_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=10, read=60, write=300, pool=30)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def strip_local_api_network_args(extra_cli_args: Sequence[str]) -> tuple[str, ...]:
    remaining_args: list[str] = []
    i = 0

    while i < len(extra_cli_args):
        arg = extra_cli_args[i]
        if arg == "--host":
            next_index = i + 1
            if next_index < len(extra_cli_args) and not extra_cli_args[next_index].startswith("--"):
                i += 2
            else:
                i += 1
            continue

        if arg == "--port":
            next_index = i + 1
            if next_index < len(extra_cli_args) and not extra_cli_args[next_index].startswith("--"):
                i += 2
            else:
                i += 1
            continue

        if arg.startswith("--host="):
            i += 1
            continue

        if arg.startswith("--port="):
            i += 1
            continue

        remaining_args.append(arg)
        i += 1

    return tuple(remaining_args)


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def resolve_effective_max_concurrent_requests(
    local_max: int,
    server_max: int,
) -> int:
    if local_max <= 0 or server_max <= 0:
        raise ValueError(
            "local_max and server_max must both be positive integers"
        )
    return min(local_max, server_max)


def response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except Exception:
        text = response.text.strip()
        return text or response.reason_phrase

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        error = payload.get("error")
        if isinstance(error, str):
            return error
        message = payload.get("message")
        if isinstance(message, str):
            return message
    return json.dumps(payload, ensure_ascii=False)


def validate_server_health_payload(payload: dict, base_url: str) -> ServerHealth:
    status = payload.get("status")
    if status != "healthy":
        raise click.ClickException(
            f"MinerU API at {base_url} is not healthy: {json.dumps(payload, ensure_ascii=False)}"
        )

    protocol_version = payload.get("protocol_version")
    if protocol_version != API_PROTOCOL_VERSION:
        raise click.ClickException(
            f"MinerU API at {base_url} returned protocol_version={protocol_version}, "
            f"expected {API_PROTOCOL_VERSION}"
        )

    max_concurrent_requests = payload.get("max_concurrent_requests")
    processing_window_size = payload.get("processing_window_size")
    if not isinstance(max_concurrent_requests, int) or max_concurrent_requests <= 0:
        raise click.ClickException(
            f"MinerU API at {base_url} did not return a valid positive max_concurrent_requests"
        )
    if not isinstance(processing_window_size, int):
        raise click.ClickException(
            f"MinerU API at {base_url} did not return a valid processing_window_size"
        )

    return ServerHealth(
        base_url=base_url,
        max_concurrent_requests=max_concurrent_requests,
        processing_window_size=max(1, processing_window_size),
    )


async def fetch_server_health(
    client: httpx.AsyncClient,
    base_url: str,
) -> ServerHealth:
    response = await client.get(f"{base_url}{HEALTH_ENDPOINT}")
    if response.status_code != 200:
        raise click.ClickException(
            f"Failed to query MinerU API health from {base_url}: "
            f"{response.status_code} {response_detail(response)}"
        )
    return validate_server_health_payload(response.json(), base_url)


async def wait_for_local_api_ready(
    client: httpx.AsyncClient,
    local_server: LocalAPIServer,
    timeout_seconds: float = LOCAL_API_STARTUP_TIMEOUT_SECONDS,
) -> ServerHealth:
    assert local_server.base_url is not None
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_error: str | None = None

    while asyncio.get_running_loop().time() < deadline:
        process = local_server.process
        if process is not None and process.poll() is not None:
            raise click.ClickException(
                "Local mineru-api exited before becoming healthy."
            )
        try:
            return await fetch_server_health(client, local_server.base_url)
        except click.ClickException as exc:
            last_error = str(exc)
        except httpx.HTTPError as exc:
            last_error = str(exc)
        await asyncio.sleep(TASK_STATUS_POLL_INTERVAL_SECONDS)

    message = "Timed out waiting for local mineru-api to become healthy."
    if last_error:
        message = f"{message} {last_error}"
    raise click.ClickException(message)


def build_parse_request_form_data(
    lang_list: Sequence[str],
    backend: str,
    parse_method: str,
    formula_enable: bool,
    table_enable: bool,
    server_url: Optional[str],
    start_page_id: int,
    end_page_id: Optional[int],
    *,
    return_md: bool,
    return_middle_json: bool,
    return_model_output: bool,
    return_content_list: bool,
    return_images: bool,
    response_format_zip: bool,
    return_original_file: bool,
) -> dict[str, str | list[str]]:
    effective_lang_list = list(lang_list) or ["ch"]
    data: dict[str, str | list[str]] = {
        "lang_list": effective_lang_list,
        "backend": backend,
        "parse_method": parse_method,
        "formula_enable": str(formula_enable).lower(),
        "table_enable": str(table_enable).lower(),
        "return_md": str(return_md).lower(),
        "return_middle_json": str(return_middle_json).lower(),
        "return_model_output": str(return_model_output).lower(),
        "return_content_list": str(return_content_list).lower(),
        "return_images": str(return_images).lower(),
        "response_format_zip": str(response_format_zip).lower(),
        "return_original_file": str(return_original_file).lower(),
        "start_page_id": str(start_page_id),
        "end_page_id": str(99999 if end_page_id is None else end_page_id),
    }
    if server_url:
        data["server_url"] = server_url
    return data


async def submit_parse_task(
    base_url: str,
    upload_assets: Sequence[UploadAsset],
    form_data: dict[str, str | list[str]],
) -> SubmitResponse:
    return await asyncio.to_thread(
        submit_parse_task_sync,
        base_url,
        upload_assets,
        form_data,
    )


def submit_parse_task_sync(
    base_url: str,
    upload_assets: Sequence[UploadAsset],
    form_data: dict[str, str | list[str]],
) -> SubmitResponse:
    with httpx.Client(timeout=build_http_timeout(), follow_redirects=True) as sync_client:
        with ExitStack() as stack:
            files = []
            for upload_asset in upload_assets:
                mime_type = (
                    mimetypes.guess_type(upload_asset.upload_name)[0]
                    or "application/octet-stream"
                )
                file_handle = stack.enter_context(open(upload_asset.path, "rb"))
                files.append(
                    (
                        "files",
                        (
                            upload_asset.upload_name,
                            file_handle,
                            mime_type,
                        ),
                    )
                )

            response = sync_client.post(
                f"{base_url}{TASKS_ENDPOINT}",
                data=form_data,
                files=files,
            )

    if response.status_code != 202:
        raise click.ClickException(
            f"Failed to submit parsing task: "
            f"{response.status_code} {response_detail(response)}"
        )

    payload = response.json()
    task_id = payload.get("task_id")
    status_url = payload.get("status_url")
    result_url = payload.get("result_url")
    file_names = payload.get("file_names")
    queued_ahead = payload.get("queued_ahead")
    if (
        not isinstance(task_id, str)
        or not isinstance(status_url, str)
        or not isinstance(result_url, str)
    ):
        raise click.ClickException("MinerU API returned an invalid task payload")

    normalized_file_names: tuple[str, ...] = ()
    if isinstance(file_names, list) and all(isinstance(name, str) for name in file_names):
        normalized_file_names = tuple(file_names)
    if not isinstance(queued_ahead, int):
        queued_ahead = None

    return SubmitResponse(
        task_id=task_id,
        status_url=status_url,
        result_url=result_url,
        file_names=normalized_file_names,
        queued_ahead=queued_ahead,
    )


async def wait_for_task_result(
    client: httpx.AsyncClient,
    submit_response: SubmitResponse,
    task_label: str,
    *,
    status_callback: Optional[Callable[[str], None]] = None,
    status_snapshot_callback: Optional[Callable[[TaskStatusSnapshot], None]] = None,
    timeout_seconds: float = TASK_RESULT_TIMEOUT_SECONDS,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        try:
            response = await client.get(submit_response.status_url)
        except httpx.ReadTimeout:
            logger.warning(
                "Timed out while polling task status for {} (task_id={}). "
                "This can happen during cold start; retrying until the task deadline.",
                task_label,
                submit_response.task_id,
            )
            await asyncio.sleep(TASK_STATUS_POLL_INTERVAL_SECONDS)
            continue
        if response.status_code != 200:
            raise click.ClickException(
                f"Failed to query task status for {task_label}: "
                f"{response.status_code} {response_detail(response)}"
            )

        payload = response.json()
        status = payload.get("status")
        if status in {"pending", "processing"}:
            queued_ahead = payload.get("queued_ahead")
            if not isinstance(queued_ahead, int):
                queued_ahead = None
            if status_snapshot_callback is not None:
                status_snapshot_callback(
                    TaskStatusSnapshot(
                        status=status,
                        queued_ahead=queued_ahead,
                    )
                )
            if status_callback is not None:
                status_callback(status)
            await asyncio.sleep(TASK_STATUS_POLL_INTERVAL_SECONDS)
            continue
        if status == "completed":
            return
        raise click.ClickException(
            f"Task {submit_response.task_id} failed for {task_label}: "
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    raise click.ClickException(
        f"Timed out waiting for result of task {submit_response.task_id} "
        f"for {task_label}"
    )


async def download_result_zip(
    client: httpx.AsyncClient,
    submit_response: SubmitResponse,
    task_label: str,
) -> Path:
    response = await client.get(submit_response.result_url)
    if response.status_code != 200:
        raise click.ClickException(
            f"Failed to download result ZIP for task {submit_response.task_id}: "
            f"{response.status_code} {response_detail(response)}"
        )
    content_type = response.headers.get("content-type", "")
    if "application/zip" not in content_type:
        raise click.ClickException(
            f"Expected a ZIP result for {task_label}, "
            f"got content-type={content_type or 'unknown'}"
        )

    zip_fd, zip_path = tempfile.mkstemp(suffix=".zip", prefix="mineru_cli_result_")
    os.close(zip_fd)
    Path(zip_path).write_bytes(response.content)
    return Path(zip_path)


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_root = output_dir.resolve()

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        for member in zip_file.infolist():
            member_path = PurePosixPath(member.filename)
            if member_path.is_absolute() or ".." in member_path.parts:
                raise click.ClickException(
                    f"Refusing to extract unsafe ZIP entry: {member.filename}"
                )
            target_path = (output_root / Path(*member_path.parts)).resolve()
            if target_path != output_root and output_root not in target_path.parents:
                raise click.ClickException(
                    f"Refusing to extract unsafe ZIP entry: {member.filename}"
                )

            if member.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue

            target_path.parent.mkdir(parents=True, exist_ok=True)
            with zip_file.open(member, "r") as source, open(target_path, "wb") as handle:
                handle.write(source.read())
