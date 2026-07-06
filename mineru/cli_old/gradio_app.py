# Copyright (c) Opendatalab. All rights reserved.

import atexit
import asyncio
import html as html_lib
import httpx
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import click
import gradio as gr
from gradio_pdf import PDF
from loguru import logger

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
# 检测 Gradio 版本，用于兼容 Gradio 5 和 Gradio 6
_gradio_major_version = int(gr.__version__.split('.')[0])
IS_GRADIO_6 = _gradio_major_version >= 6

log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()  # 移除默认handler
logger.add(sys.stderr, level=log_level)  # 添加新handler

from mineru.cli_old.common import (
    image_suffixes,
    normalize_task_stem,
    office_suffixes,
    pdf_suffixes,
    read_fn,
)
from mineru.cli_old.visualization import (
    VISUALIZATION_FINISHED,
    VisualizationJob,
    run_visualization_job,
)
from mineru.cli_old import api_client as _api_client
from mineru.data.data_reader_writer.filebase import FileBasedDataWriter
from mineru.parser.api_client import MinerUApiParser
from mineru.parser.base import ParseResult
from mineru.parser.tier import ParserRuntimeOptions, runtime_options_for_tier
from mineru.render import render_content_list, render_markdown, render_structured_content
from mineru.utils.backend_options import (
    CANONICAL_HYBRID_ENGINE,
    LOCAL_HYBRID_EFFORT,
    MAX_HYBRID_EFFORT,
)
from mineru.utils.image_payload import validate_image_sidecar_path
from mineru.utils.ocr_language import PUBLIC_OCR_LANGUAGE_CHOICES, validate_public_ocr_lang
from mineru.utils.pdf_document import PDFDocument
from mineru.utils.pdf_page_id import parse_page_range
from mineru.utils.pdfium_guard import safe_rewrite_pdf_bytes_with_pdfium_result
from mineru.version import __version__

_gradio_local_api_server: "ReusableGradioV1APIServer"


@dataclass(frozen=True)
class GradioConcurrencyWaitSnapshot:
    limit: int
    active: int
    waiting: int
    ahead: int


@dataclass
class _LimiterState:
    semaphore: asyncio.Semaphore
    active: int = 0
    waiters: list[object] = field(default_factory=list)


class GradioRequestConcurrencyLimiter:
    def __init__(self):
        self._lock = threading.Lock()
        self._states: dict[int, _LimiterState] = {}

    def _get_state(self, limit: int):
        if limit <= 0:
            return None
        with self._lock:
            state = self._states.get(limit)
            if state is None:
                state = _LimiterState(semaphore=asyncio.Semaphore(limit))
                self._states[limit] = state
            return state

    def _build_wait_snapshot(
        self,
        state: _LimiterState,
        limit: int,
        wait_token: object,
    ) -> GradioConcurrencyWaitSnapshot | None:
        if wait_token not in state.waiters:
            return None

        return GradioConcurrencyWaitSnapshot(
            limit=limit,
            active=state.active,
            waiting=len(state.waiters),
            ahead=state.waiters.index(wait_token),
        )

    def _remove_waiter(self, state: _LimiterState, wait_token: object) -> None:
        if wait_token in state.waiters:
            state.waiters.remove(wait_token)

    async def _cleanup_acquire_interruption(
        self,
        state: _LimiterState,
        acquire_task: asyncio.Task[bool],
        wait_token: object,
        should_wait: bool,
    ) -> None:
        if not acquire_task.done():
            acquire_task.cancel()
            await asyncio.gather(acquire_task, return_exceptions=True)
        elif not acquire_task.cancelled():
            try:
                acquired = acquire_task.result()
            except Exception:
                acquired = False
            if acquired:
                state.semaphore.release()

        if should_wait:
            with self._lock:
                self._remove_waiter(state, wait_token)

    @asynccontextmanager
    async def acquire(
        self,
        limit: int,
        on_wait: Callable[[GradioConcurrencyWaitSnapshot], None] | None = None,
    ):
        state = self._get_state(limit)
        if state is None:
            yield
            return

        wait_token = object()
        should_wait = False
        snapshot = None
        with self._lock:
            if state.active >= limit or state.waiters:
                state.waiters.append(wait_token)
                should_wait = True
                snapshot = self._build_wait_snapshot(state, limit, wait_token)

        acquire_task: asyncio.Task[bool] = asyncio.create_task(state.semaphore.acquire())
        last_wait_ahead = None
        if should_wait and on_wait is not None and snapshot is not None:
            on_wait(snapshot)
            last_wait_ahead = snapshot.ahead

        try:
            if should_wait:
                while True:
                    done, _ = await asyncio.wait(
                        {acquire_task},
                        timeout=STATUS_TIMER_INTERVAL_SECONDS,
                    )
                    if acquire_task in done:
                        acquire_task.result()
                        break

                    if on_wait is None:
                        continue

                    with self._lock:
                        snapshot = self._build_wait_snapshot(state, limit, wait_token)

                    if snapshot is None or snapshot.ahead == last_wait_ahead:
                        continue

                    on_wait(snapshot)
                    last_wait_ahead = snapshot.ahead
            else:
                await acquire_task
        except BaseException:
            await self._cleanup_acquire_interruption(
                state=state,
                acquire_task=acquire_task,
                wait_token=wait_token,
                should_wait=should_wait,
            )
            raise

        with self._lock:
            if should_wait:
                self._remove_waiter(state, wait_token)
            state.active += 1
        try:
            yield
        finally:
            with self._lock:
                state.active = max(0, state.active - 1)
            state.semaphore.release()


_gradio_request_concurrency_limiter = GradioRequestConcurrencyLimiter()

STATUS_BOX_AUTOSCROLL_JS = """
(value) => {
    const scrollToBottom = () => {
        const textarea = document.querySelector(".convert-status-box textarea");
        if (!textarea) {
            return;
        }
        textarea.scrollTop = textarea.scrollHeight;
    };

    requestAnimationFrame(() => {
        scrollToBottom();
        requestAnimationFrame(scrollToBottom);
    });

    return [];
}
"""

STATUS_TIMER_INTERVAL_SECONDS = 0.1
STATUS_QUEUE_ANIMATION_INTERVAL_SECONDS = 1.0
STATUS_QUEUE_ANIMATION_MAX_DOTS = 10

STATUS_PREPARING_REQUEST = "Preparing request..."
STATUS_CHECKING_SERVER = "Checking server status..."
STATUS_SUBMITTING_TASK = "Submitting task..."
STATUS_DOWNLOADING_RESULT = "Task completed, downloading result..."
STATUS_PROCESSING_OUTPUT = "Preparing outputs..."
STATUS_COMPLETED = "Completed"
STATUS_QUEUED_ON_SERVER = "Queued on server"
STATUS_PROCESSING_ON_SERVER = "Processing on server"
STATUS_QUEUED_LOCALLY_PREFIX = "Queued locally:"

DEFAULT_GRADIO_TIER = "pro"
GRADIO_TIER_CHOICES = ("standard", "pro")
GRADIO_REMOTE_BACKEND = "hybrid-http-client"
STATUS_STEP_DEFINITIONS = [
    ("status_step_prepare", STATUS_PREPARING_REQUEST),
    ("status_step_check", STATUS_CHECKING_SERVER),
    ("status_step_submit", STATUS_SUBMITTING_TASK),
    ("status_step_queue", STATUS_QUEUED_ON_SERVER),
    ("status_step_process", STATUS_PROCESSING_ON_SERVER),
    ("status_step_download", STATUS_DOWNLOADING_RESULT),
    ("status_step_outputs", STATUS_PROCESSING_OUTPUT),
    ("status_step_done", STATUS_COMPLETED),
]


@dataclass(frozen=True)
class GradioV1ServerHealth:
    """记录 Gradio 当前绑定的 v1 API server 地址和可用 tier。"""

    base_url: str
    tier_choices: tuple[str, ...]
    max_concurrent_requests: int = 1


@dataclass(frozen=True)
class GradioV1PersistedOutput:
    """记录 v1 解析结果落盘后供 Gradio UI 继续消费的路径。"""

    file_name: str
    local_md_dir: Path
    archive_zip_path: Path
    preview_pdf_path: Path | None


class GradioV1LocalAPIServer:
    """管理 Gradio 内置启动的 v1 parser API server 子进程。"""

    def __init__(self, extra_cli_args=()):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="mineru-v1-api-client-")
        self.temp_root = Path(self.temp_dir.name)
        self.upload_dir = self.temp_root / "uploads"
        self.base_url: str | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self.extra_cli_args = tuple(extra_cli_args)
        self._managed_process_group_id: int | None = None
        self._atexit_registered = False

    def start(self) -> str:
        """启动本地 v1 parser API server，并返回可探测的 base URL。"""
        if self.process is not None:
            raise RuntimeError("Local v1 API server is already running")

        resolved_port = _api_client.find_free_port()
        self.base_url = f"http://127.0.0.1:{resolved_port}"
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        cli_args = (
            "--host",
            "127.0.0.1",
            "--port",
            str(resolved_port),
            "--upload-dir",
            str(self.upload_dir),
            *_api_client.strip_local_api_network_args(self.extra_cli_args),
        )
        command = [
            sys.executable,
            "-m",
            "mineru.parser.api_server",
            *cli_args,
        ]
        self.process = subprocess.Popen(
            command,
            cwd=os.getcwd(),
            env=os.environ.copy(),
            stdin=subprocess.PIPE,
            **_api_client.build_managed_process_popen_kwargs(),
        )
        self._managed_process_group_id = self.process.pid
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True
        return self.base_url

    def stop(self) -> None:
        """停止本地 v1 parser API server，并清理临时目录。"""
        process = self.process
        process_group_id = self._managed_process_group_id
        self.process = None
        self._managed_process_group_id = None
        try:
            _api_client.stop_managed_process(
                process,
                process_group_id=process_group_id,
                shutdown_timeout_seconds=_api_client.LOCAL_API_SHUTDOWN_TIMEOUT_SECONDS,
                use_stdin_shutdown_watcher=True,
            )
        finally:
            if self._atexit_registered:
                try:
                    atexit.unregister(self.stop)
                except Exception:
                    pass
                self._atexit_registered = False
            self.temp_dir.cleanup()


class ReusableGradioV1APIServer:
    """复用 Gradio 本地 v1 API server，避免每次请求重复冷启动。"""

    def __init__(self, extra_cli_args=()):
        self._lock = threading.Lock()
        self._server: GradioV1LocalAPIServer | None = None
        self._extra_cli_args = tuple(extra_cli_args)

    def configure(self, extra_cli_args) -> None:
        """记录启动参数；若现有进程已退出，则下次请求会按新参数重启。"""
        with self._lock:
            self._extra_cli_args = tuple(extra_cli_args)
            server = self._server
            if server is None:
                return
            if server.process is not None and server.process.poll() is None:
                return
            server.stop()
            self._server = None

    def ensure_started(self) -> tuple[GradioV1LocalAPIServer, bool]:
        """确保本地 v1 API server 存活，返回 server 实例和是否刚启动。"""
        with self._lock:
            server = self._server
            if server is not None and server.process is not None and server.process.poll() is None:
                return server, False
            if server is not None:
                server.stop()
            server = GradioV1LocalAPIServer(extra_cli_args=self._extra_cli_args)
            server.start()
            self._server = server
            return server, True

    def stop(self) -> None:
        """停止当前复用的本地 v1 API server。"""
        with self._lock:
            server = self._server
            self._server = None
        if server is not None:
            server.stop()


_gradio_local_api_server = ReusableGradioV1APIServer()


def extract_v1_tier_choices(payload) -> tuple[str, ...]:
    """从 /v1/tiers 响应中提取 Gradio 支持的 tier，保留服务端顺序并去重。"""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        raise click.ClickException("MinerU v1 API /v1/tiers returned an invalid payload.")

    choices: list[str] = []
    for item in data:
        tier_id = item.get("id") if isinstance(item, dict) else None
        if tier_id in GRADIO_TIER_CHOICES and tier_id not in choices:
            choices.append(tier_id)
    if not choices:
        supported = ", ".join(GRADIO_TIER_CHOICES)
        raise click.ClickException(
            f"MinerU v1 API did not advertise any supported tier. Supported Gradio tiers: {supported}."
        )
    return tuple(choices)


def default_v1_gradio_tier(tier_choices: tuple[str, ...]) -> str:
    """选择 Gradio 默认 tier；v1 server 返回顺序即默认优先级。"""
    if not tier_choices:
        raise click.ClickException("MinerU v1 API did not advertise any supported tier.")
    return tier_choices[0]


async def fetch_v1_tier_choices(http_client, base_url: str) -> tuple[str, ...]:
    """调用 v1 /tiers 接口并转换成 Gradio Dropdown choices。"""
    try:
        response = await http_client.get(f"{base_url}/v1/tiers")
    except httpx.HTTPError as exc:
        raise click.ClickException(
            f"Failed to query MinerU v1 API tiers from {base_url}: {exc}"
        ) from exc
    if response.status_code != 200:
        raise click.ClickException(
            f"Failed to query MinerU v1 API tiers from {base_url}: "
            f"{response.status_code} {_api_client.response_detail(response)}"
        )
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"MinerU v1 API /v1/tiers from {base_url} did not return JSON."
        ) from exc
    return extract_v1_tier_choices(payload)


async def wait_for_v1_api_ready(
    http_client,
    local_server: GradioV1LocalAPIServer,
    timeout_seconds: float = _api_client.LOCAL_API_STARTUP_TIMEOUT_SECONDS,
) -> GradioV1ServerHealth:
    """等待本地 v1 API server 返回 /v1/tiers，避免 Gradio 页面先于服务可用。"""
    assert local_server.base_url is not None
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    last_error: str | None = None
    while asyncio.get_running_loop().time() < deadline:
        process = local_server.process
        if process is not None and process.poll() is not None:
            raise click.ClickException("Local MinerU v1 API exited before becoming ready.")
        try:
            choices = await fetch_v1_tier_choices(http_client, local_server.base_url)
            return GradioV1ServerHealth(base_url=local_server.base_url, tier_choices=choices)
        except click.ClickException as exc:
            last_error = str(exc)
        except httpx.HTTPError as exc:
            last_error = str(exc)
        await asyncio.sleep(_api_client.TASK_STATUS_POLL_INTERVAL_SECONDS)

    message = "Timed out waiting for local MinerU v1 API to become ready."
    if last_error:
        message = f"{message} {last_error}"
    raise click.ClickException(message)


async def resolve_v1_server_health(http_client, api_url: str | None) -> GradioV1ServerHealth:
    """解析 Gradio 当前应使用的 v1 API server，并返回该 server 的 tier 能力。"""
    if api_url:
        base_url = _api_client.normalize_base_url(api_url)
        return GradioV1ServerHealth(
            base_url=base_url,
            tier_choices=await fetch_v1_tier_choices(http_client, base_url),
        )

    local_server, started_now = _gradio_local_api_server.ensure_started()
    if started_now:
        logger.info(f"Started local MinerU v1 API at {local_server.base_url}")
    return await wait_for_v1_api_ready(http_client, local_server)


def resolve_gradio_v1_startup_health(api_url: str | None) -> GradioV1ServerHealth:
    """同步探测 v1 API server tier 能力，供 Gradio 构建 Dropdown 前使用。"""

    async def _resolve() -> GradioV1ServerHealth:
        async with httpx.AsyncClient(
            timeout=_api_client.build_http_timeout(),
            follow_redirects=True,
        ) as http_client:
            return await resolve_v1_server_health(http_client, api_url)

    return asyncio.run(_resolve())


def build_v1_gradio_page_range(end_pages: int | float | None) -> str:
    """将 Gradio 旧 max_pages 控件转换成 v1 API 使用的 1-based page_range。"""
    try:
        page_limit = int(end_pages) if end_pages is not None else 0
    except (TypeError, ValueError):
        page_limit = 0
    if page_limit <= 0:
        return ""
    return f"1~{page_limit}"


def normalize_mineru_locale(locale):
    """统一自定义 HTML 的语言归一规则：中文使用 zh，其他语言降级为英文。"""
    normalized = str(locale or "").strip().lower()
    if normalized.startswith("zh"):
        return "zh"
    return "en"


def resolve_i18n_text(i18n, key, locale=None):
    """按指定语言读取自定义 HTML 需要的纯文本文案，避免直接渲染 Gradio I18nData 元数据。"""
    if i18n is None:
        return key
    translations = getattr(i18n, "translations", None)
    if translations:
        preferred_locale = normalize_mineru_locale(
            locale or os.getenv("MINERU_GRADIO_DEFAULT_LOCALE", "zh")
        )
        preferred_text = translations.get(preferred_locale, {}).get(key)
        if preferred_text is not None:
            return preferred_text
        fallback_text = translations.get("en", {}).get(key)
        if fallback_text is not None:
            return fallback_text
        return key
    return i18n(key)


def translate_ui(i18n, key, locale=None):
    """按目标语言读取纯文本文案，用作自定义 HTML 的初始渲染内容。"""
    return resolve_i18n_text(i18n, key, locale)


def resolve_request_locale(request):
    """根据 Gradio 请求头推断浏览器语言，避免流式状态面板高频刷新时闪回默认语言。"""
    headers = getattr(request, "headers", None) or {}
    if not hasattr(headers, "get"):
        return None

    accept_language = headers.get("accept-language") or headers.get("Accept-Language")
    if not accept_language:
        return None

    language_candidates = []
    for order, raw_item in enumerate(str(accept_language).split(",")):
        parts = [part.strip() for part in raw_item.split(";") if part.strip()]
        if not parts:
            continue
        quality = 1.0
        for parameter in parts[1:]:
            name, separator, value = parameter.partition("=")
            if separator and name.strip().lower() == "q":
                try:
                    quality = float(value)
                except ValueError:
                    quality = 0.0
                break
        language_candidates.append((-quality, order, parts[0]))

    if not language_candidates:
        return None
    _, _, preferred_language = min(language_candidates)
    return normalize_mineru_locale(preferred_language)


def build_client_i18n_attrs(i18n, key):
    """为自定义 HTML 输出中英文文案属性，交给前端按浏览器语言切换。"""
    attrs = [f'data-mineru-i18n-key="{html_lib.escape(key, quote=True)}"']
    for locale in ("en", "zh"):
        text = resolve_i18n_text(i18n, key, locale)
        attrs.append(f'data-mineru-i18n-{locale}="{html_lib.escape(text, quote=True)}"')
    return " ".join(attrs)


def render_client_i18n_text(i18n, key, locale=None):
    """生成可被前端重新本地化的文本节点，避免 header/status 在英文环境固定成中文。"""
    return (
        f"<span {build_client_i18n_attrs(i18n, key)}>"
        f"{html_lib.escape(translate_ui(i18n, key, locale))}"
        "</span>"
    )


def resolve_gradio_runtime_options(tier: str | None, use_remote_server: bool = False) -> ParserRuntimeOptions:
    """根据 Gradio 公开 tier 与远端开关派生内部 backend/effort，保持旧提交链路兼容。"""
    backend = GRADIO_REMOTE_BACKEND if use_remote_server else CANONICAL_HYBRID_ENGINE
    return runtime_options_for_tier(tier or DEFAULT_GRADIO_TIER, backend=backend)


def select_tier_info_key(tier_choice: object) -> str:
    """根据 Gradio tier 选择说明文案的 i18n key。"""
    if tier_choice == "standard":
        return "tier_info_standard"
    if tier_choice == "pro":
        return "tier_info_pro"
    return "tier_info_default"


def select_force_ocr_info_key(tier_choice: object) -> str:
    """根据 Gradio tier 选择强制 OCR 说明；当前公开 tier 均走 Hybrid 文案。"""
    if tier_choice in GRADIO_TIER_CHOICES:
        return "force_ocr_info_hybrid"
    return "force_ocr_info"


def frontend_managed_initial_visibility(is_visible: bool):
    """转换前端托管显隐控件的初始状态；hidden 会保留 DOM 挂载避免状态丢失。"""
    return True if is_visible else "hidden"


def resolve_status_step_index(status_lines):
    """根据现有状态日志推断步骤面板中当前应高亮的步骤索引。"""
    if not status_lines:
        return -1, False
    if status_lines[-1].startswith("Failed:"):
        return len(STATUS_STEP_DEFINITIONS) - 1, True
    if any(line.startswith(STATUS_COMPLETED) for line in status_lines):
        return len(STATUS_STEP_DEFINITIONS) - 1, False

    for index in range(len(STATUS_STEP_DEFINITIONS) - 1, -1, -1):
        _, marker = STATUS_STEP_DEFINITIONS[index]
        if marker == STATUS_QUEUED_ON_SERVER:
            if any(StatusPanelState.is_queue_message(line) for line in status_lines):
                return index, False
            continue
        if any(line.startswith(marker) for line in status_lines):
            return index, False
    return 0, False


def render_status_steps_html(status_text, i18n, locale=None):
    """把流式状态日志渲染为步骤式状态面板，底层日志格式保持不变。"""
    status_lines = [line for line in str(status_text or "").splitlines() if line]
    current_index, is_failed = resolve_status_step_index(status_lines)
    latest_status = (
        status_lines[-1]
        if status_lines
        else render_client_i18n_text(i18n, "status_idle_hint", locale)
    )

    step_items = []
    for index, (label_key, _) in enumerate(STATUS_STEP_DEFINITIONS):
        classes = ["status-step"]
        if is_failed and index == current_index:
            classes.extend(["is-active", "is-error"])
            label = render_client_i18n_text(i18n, "status_step_failed", locale)
        elif index < current_index or (
            current_index == len(STATUS_STEP_DEFINITIONS) - 1 and not is_failed
        ):
            classes.append("is-done")
            label = render_client_i18n_text(i18n, label_key, locale)
        elif index == current_index:
            classes.append("is-active")
            label = render_client_i18n_text(i18n, label_key, locale)
        else:
            classes.append("is-pending")
            label = render_client_i18n_text(i18n, label_key, locale)
        step_items.append(
            f'<div class="{" ".join(classes)}">'
            f'<span class="status-dot"></span>'
            f'<span class="status-label">{label}</span>'
            "</div>"
        )

    title_key = "status_idle_title" if not status_lines else "status_latest"
    return (
        '<div class="status-steps-panel">'
        f'<div class="status-panel-title">'
        f'{render_client_i18n_text(i18n, title_key, locale)}'
        f'</div>'
        f'<div class="status-steps-list">{"".join(step_items)}</div>'
        f'<div class="status-latest">'
        f'{latest_status if not status_lines else html_lib.escape(latest_status)}'
        f'</div>'
        "</div>"
    )


RESOURCE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources')


def load_resource_text(resource_name):
    """读取 mineru/resources 下的文本资源，集中管理 Gradio 静态片段。"""
    resource_path = os.path.join(RESOURCE_DIR, resource_name)
    with open(resource_path, mode='r', encoding='utf-8') as resource_file:
        return resource_file.read()


APP_CSS = load_resource_text('gradio_app.css')
APP_JS = load_resource_text('gradio_app.js')

# Gradio 6 的 js 参数在部分托管环境里只注入函数文本，使用 head 包装确保页面加载后主动执行。
APP_HEAD = f"""
<script>
(() => {{
    const installMineruAdvancedPopover = {APP_JS};
    if (document.readyState === "loading") {{
        document.addEventListener("DOMContentLoaded", installMineruAdvancedPopover, {{ once: true }});
    }} else {{
        installMineruAdvancedPopover();
    }}
}})();
</script>
"""


@dataclass
class StatusPanelState:
    lines: list[str] = field(default_factory=list)
    processing_index: int | None = None
    processing_started_at: float | None = None
    last_processing_elapsed_seconds: float | None = None
    queue_index: int | None = None
    queue_started_at: float | None = None
    queue_base_message: str | None = None

    def append(self, message: str) -> bool:
        if not message:
            return False

        if self.is_queue_message(message):
            self.finalize_processing()
            return self.update_queue(message)

        if message == STATUS_PROCESSING_ON_SERVER:
            self.finalize_queue()
            return self.start_processing()

        self.finalize_processing()
        self.finalize_queue()
        if message == STATUS_COMPLETED:
            message = format_completed_status(self.last_processing_elapsed_seconds)
        if not self.lines or self.lines[-1] != message:
            self.lines.append(message)
            return True
        return False

    def start_processing(self) -> bool:
        if self.processing_started_at is not None:
            return self.tick_processing()

        self.processing_started_at = time.monotonic()
        self.last_processing_elapsed_seconds = 0.0
        self.processing_index = len(self.lines)
        self.lines.append(format_processing_status(0.0))
        return True

    def tick_processing(self) -> bool:
        if self.processing_started_at is None or self.processing_index is None:
            return False

        elapsed_seconds = max(0.0, time.monotonic() - self.processing_started_at)
        self.last_processing_elapsed_seconds = elapsed_seconds
        updated = format_processing_status(elapsed_seconds)
        if self.lines[self.processing_index] != updated:
            self.lines[self.processing_index] = updated
            return True
        return False

    def finalize_processing(self) -> bool:
        if self.processing_started_at is None or self.processing_index is None:
            return False

        self.tick_processing()
        self.processing_started_at = None
        self.processing_index = None
        return True

    def update_queue(self, message: str) -> bool:
        if (
            self.queue_index is None
            or self.queue_started_at is None
            or self.queue_base_message is None
        ):
            self.queue_started_at = time.monotonic()
            self.queue_index = len(self.lines)
            self.queue_base_message = message
            self.lines.append(format_queue_status(message, 0.0))
            return True

        self.queue_base_message = message
        updated = format_queue_status(
            message,
            max(0.0, time.monotonic() - self.queue_started_at),
        )
        if self.lines[self.queue_index] != updated:
            self.lines[self.queue_index] = updated
            return True
        return False

    def tick_queue(self) -> bool:
        if (
            self.queue_index is None
            or self.queue_started_at is None
            or self.queue_base_message is None
        ):
            return False

        updated = format_queue_status(
            self.queue_base_message,
            max(0.0, time.monotonic() - self.queue_started_at),
        )
        if self.lines[self.queue_index] != updated:
            self.lines[self.queue_index] = updated
            return True
        return False

    def finalize_queue(self) -> bool:
        if (
            self.queue_index is None
            or self.queue_started_at is None
            or self.queue_base_message is None
        ):
            return False

        self.tick_queue()
        self.queue_index = None
        self.queue_started_at = None
        self.queue_base_message = None
        return True

    def tick(self) -> bool:
        if self.is_processing:
            return self.tick_processing()
        if self.is_queueing:
            return self.tick_queue()
        return False

    @property
    def is_processing(self) -> bool:
        return self.processing_started_at is not None

    @property
    def is_queueing(self) -> bool:
        return self.queue_started_at is not None

    @property
    def animation_interval_seconds(self) -> float | None:
        if self.is_processing:
            return STATUS_TIMER_INTERVAL_SECONDS
        if self.is_queueing:
            return STATUS_QUEUE_ANIMATION_INTERVAL_SECONDS
        return None

    @staticmethod
    def is_queue_message(message: str) -> bool:
        return (
            message.startswith(STATUS_QUEUED_LOCALLY_PREFIX)
            or message.startswith(STATUS_QUEUED_ON_SERVER)
        )

    def render(self) -> str:
        return "\n".join(self.lines)


def format_failed_status(error: Exception | str) -> str:
    return f"Failed: {error}"


def format_processing_status(elapsed_seconds: float) -> str:
    return f"{STATUS_PROCESSING_ON_SERVER} ({elapsed_seconds:.1f}s)"


def format_completed_status(elapsed_seconds: float | None) -> str:
    """生成完成状态文案，保留服务端解析阶段最终耗时。"""
    if elapsed_seconds is None:
        return STATUS_COMPLETED
    return f"{STATUS_COMPLETED} ({elapsed_seconds:.1f}s)"


def format_queue_status(base_message: str, elapsed_seconds: float) -> str:
    dots = "." * (
        (int(max(0.0, elapsed_seconds)) % STATUS_QUEUE_ANIMATION_MAX_DOTS) + 1
    )
    return f"{base_message}{dots}"


def format_concurrency_wait_message(snapshot: GradioConcurrencyWaitSnapshot) -> str:
    return f"{STATUS_QUEUED_LOCALLY_PREFIX} {snapshot.ahead} request(s) ahead"


def format_remote_status_message(
    status_snapshot: _api_client.TaskStatusSnapshot | str,
) -> str:
    if isinstance(status_snapshot, _api_client.TaskStatusSnapshot):
        status = status_snapshot.status
        queued_ahead = status_snapshot.queued_ahead
    else:
        status = status_snapshot
        queued_ahead = None

    if status == "pending":
        if queued_ahead is not None:
            return f"{STATUS_QUEUED_ON_SERVER}: {queued_ahead} request(s) ahead"
        return STATUS_QUEUED_ON_SERVER
    if status == "processing":
        return STATUS_PROCESSING_ON_SERVER
    if status == "completed":
        return STATUS_COMPLETED
    if status == "failed":
        return format_failed_status("server task failed")
    return f"Task status: {status}"


def compress_directory_to_zip(directory_path, output_zip_path):
    """压缩指定目录到一个 ZIP 文件。

    :param directory_path: 要压缩的目录路径
    :param output_zip_path: 输出的 ZIP 文件路径
    """
    try:
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:

            # 遍历目录中的所有文件和子目录
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    # 构建完整的文件路径
                    file_path = os.path.join(root, file)
                    # 计算相对路径
                    arcname = os.path.relpath(file_path, directory_path)
                    # 添加文件到 ZIP 文件
                    zipf.write(file_path, arcname)
        return 0
    except Exception as e:
        logger.exception(e)
        return -1


GRADIO_PREVIEW_IMAGE_SUFFIXES = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
}
GRADIO_PREVIEW_EXTERNAL_SRC_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")


def _resolve_gradio_preview_image_path(src, image_dir_path):
    """将 Markdown/HTML 中的图片路径解析成 Gradio 文件路由需要的本地完整路径。"""
    image_src = str(src).strip()
    if not image_src:
        return None
    if Path(image_src).suffix.lower() not in GRADIO_PREVIEW_IMAGE_SUFFIXES:
        return None

    resolved_path = (Path(image_dir_path) / image_src).resolve(strict=False)
    if not resolved_path.is_file():
        logger.warning(f"Skip missing Gradio preview image: {resolved_path}")
        return None
    return str(resolved_path)


def replace_image_with_gradio_file_urls(markdown_text, image_dir_path):
    """将 Gradio 预览中的本地图片路径改写为 HTTP 文件链接，不修改导出的 Markdown 文件。"""
    if not isinstance(markdown_text, str) or not image_dir_path:
        return markdown_text

    def _path_to_public_url(image_src):
        image_path = _resolve_gradio_preview_image_path(image_src, image_dir_path)
        if not image_path:
            return None
        return f"/gradio_api/file={quote(image_path, safe='/:')}"

    # 匹配 Markdown 正文图片 ![alt](path)，只替换可安全访问的本地图片路径。
    def replace_md(match):
        alt_text = match.group("alt")
        image_src = match.group("src")
        public_url = _path_to_public_url(image_src)
        if public_url:
            return f"![{alt_text}]({public_url})"
        return match.group(0)

    result = re.sub(
        r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)",
        replace_md,
        markdown_text,
    )

    # 匹配 HTML 表格内的 <img src="path">，跳过 data/http/blob 等已有可访问地址。
    def replace_html_src(match):
        prefix = match.group("prefix")
        quote_char = match.group("quote")
        image_src = match.group("src")
        public_url = _path_to_public_url(image_src)
        if public_url:
            return f"{prefix}{quote_char}{public_url}{quote_char}"
        return match.group(0)

    result = re.sub(
        r"(?P<prefix><img\b[^>]*?\bsrc\s*=\s*)(?P<quote>[\"'])(?P<src>[^\"']+)(?P=quote)",
        replace_html_src,
        result,
        flags=re.IGNORECASE,
    )

    return result


def read_gradio_content_list_json(local_md_dir, file_name):
    """读取本次 Gradio 解析目录中的 legacy content_list JSON，失败时返回空字符串。"""
    content_list_path = Path(local_md_dir) / f"{file_name}_content_list.json"
    if not content_list_path.is_file():
        logger.warning(
            f"Content list JSON not found for Gradio preview: {content_list_path}"
        )
        return ""
    try:
        return content_list_path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning(
            f"Failed to read Gradio content list JSON: {content_list_path}, error={exc}"
        )
        return ""


def _middle_json_backend_for_gradio(parse_result: ParseResult, backend: str) -> str:
    """推断 Gradio 落盘 middle_json 的 backend 标识，优先使用解析结果自身标记。"""
    for page in parse_result.pages:
        if page._backend:
            return page._backend
    return "hybrid" if str(backend).startswith("hybrid") else backend


def _write_v1_gradio_images(parse_result: ParseResult, local_image_dir: Path) -> None:
    """将 v1 API 返回的图片 sidecar 写入 Gradio legacy images 目录。"""
    writer = FileBasedDataWriter(str(local_image_dir))
    for image_path, image_bytes in parse_result.images().items():
        writer.write(validate_image_sidecar_path(image_path), image_bytes)


def _build_v1_gradio_origin_file_bytes(source_path: Path, page_range: str) -> bytes:
    """按 v1 解析范围生成 Gradio 本地 origin 文件字节；非 PDF 保持原样。"""
    source_bytes = source_path.read_bytes()
    if source_path.suffix.lower().lstrip(".") != "pdf" or not page_range.strip():
        return source_bytes

    try:
        with PDFDocument(source_bytes) as pdf_doc:
            page_count = pdf_doc.page_count
            page_indices = parse_page_range(page_range, page_count)
        if not page_indices:
            logger.warning(f"Gradio origin PDF page_range selects no pages: {page_range}")
            return source_bytes
        if page_indices == list(range(page_count)):
            return source_bytes

        rewrite_result = safe_rewrite_pdf_bytes_with_pdfium_result(
            source_bytes,
            page_indices=page_indices,
        )
        if rewrite_result.used_original:
            logger.warning(
                f"Gradio origin PDF rewrite fell back to original PDF for page_range: {page_range}"
            )
        return rewrite_result.pdf_bytes or source_bytes
    except Exception as exc:
        logger.warning(
            f"Failed to build Gradio page-ranged origin PDF for {source_path}: {exc}; using original PDF."
        )
        return source_bytes


def _generate_v1_gradio_layout_preview(
    *,
    file_suffix: str,
    file_name: str,
    local_md_dir: Path,
    backend: str,
) -> None:
    """为 v1 Gradio 本地结果同步生成 layout 可视化 PDF，失败时保留 origin fallback。"""
    if file_suffix != "pdf":
        return

    result = run_visualization_job(
        VisualizationJob(
            document_stem=file_name,
            backend=backend,
            parse_method="auto",
            parse_dir=local_md_dir,
            draw_span=False,
        )
    )
    if result.status != VISUALIZATION_FINISHED:
        logger.warning(f"Skipping Gradio layout preview for {file_name}: {result.message}")


def persist_v1_gradio_result(
    *,
    parse_result: ParseResult,
    file_path: str,
    extract_root: Path,
    archive_zip_path: Path,
    backend: str,
    effort: str,
    page_range: str,
) -> GradioV1PersistedOutput:
    """把 v1 ParseResult 保存成旧 Gradio UI 预览、JSON 面板和 ZIP 下载所需文件。"""
    source_path = Path(file_path)
    file_name = normalize_task_stem(source_path.stem)
    file_suffix = source_path.suffix.lower().lstrip(".")
    local_md_dir = extract_root / file_name
    local_image_dir = local_md_dir / "images"
    local_md_dir.mkdir(parents=True, exist_ok=True)
    local_image_dir.mkdir(parents=True, exist_ok=True)

    image_dir = local_image_dir.name
    render_pages = parse_result.export_pages()
    _write_v1_gradio_images(parse_result, local_image_dir)

    (local_md_dir / f"{file_name}.md").write_text(
        render_markdown(render_pages, image_dir),
        encoding="utf-8",
    )
    (local_md_dir / f"{file_name}_content_list.json").write_text(
        json.dumps(render_content_list(render_pages, image_dir), ensure_ascii=False, indent=4),
        encoding="utf-8",
    )
    (local_md_dir / f"{file_name}_structured_content.json").write_text(
        json.dumps(render_structured_content(render_pages, image_dir), ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    middle_json = parse_result.to_dict()
    middle_json["_backend"] = _middle_json_backend_for_gradio(parse_result, backend)
    middle_json["_version_name"] = __version__
    (local_md_dir / f"{file_name}_middle.json").write_text(
        json.dumps(middle_json, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    origin_suffix = file_suffix if file_suffix in office_suffixes else "pdf"
    origin_bytes = _build_v1_gradio_origin_file_bytes(source_path, page_range)
    (local_md_dir / f"{file_name}_origin.{origin_suffix}").write_bytes(origin_bytes)

    _generate_v1_gradio_layout_preview(
        file_suffix=file_suffix,
        file_name=file_name,
        local_md_dir=local_md_dir,
        backend=backend,
    )

    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info("Compression successful")
    else:
        logger.error("Compression failed")

    preview_pdf_path = None if file_suffix in office_suffixes else resolve_preview_pdf_path(str(local_md_dir), file_name)
    return GradioV1PersistedOutput(
        file_name=file_name,
        local_md_dir=local_md_dir,
        archive_zip_path=archive_zip_path,
        preview_pdf_path=Path(preview_pdf_path).resolve() if preview_pdf_path else None,
    )


def _escape_latex_html_chars_for_gradio(content):
    """转义公式内部会被 Gradio HTML 解析链路误判的尖括号，保留 LaTeX 对齐用 &。"""
    return content.replace("<", "&lt;").replace(">", "&gt;")


def escape_latex_blocks_for_gradio_preview(markdown_text, latex_delimiters):
    """根据当前 LaTeX 分隔符，仅转义公式内容，避免影响公式外 Markdown/HTML。"""
    if not markdown_text or not latex_delimiters:
        return markdown_text

    delimiter_pairs = []
    for delimiter in latex_delimiters:
        left = delimiter.get("left")
        right = delimiter.get("right")
        if left and right:
            delimiter_pairs.append((left, right))
    delimiter_pairs.sort(key=lambda pair: len(pair[0]), reverse=True)
    if not delimiter_pairs:
        return markdown_text

    result = []
    position = 0
    text_length = len(markdown_text)
    while position < text_length:
        matched_pair = None
        for left, right in delimiter_pairs:
            if markdown_text.startswith(left, position):
                matched_pair = (left, right)
                break

        if matched_pair is None:
            result.append(markdown_text[position])
            position += 1
            continue

        left, right = matched_pair
        content_start = position + len(left)
        content_end = markdown_text.find(right, content_start)
        if content_end == -1:
            # 未闭合的分隔符保持原样，并继续扫描后续可能闭合的公式块。
            result.append(markdown_text[position])
            position += 1
            continue

        result.append(left)
        result.append(
            _escape_latex_html_chars_for_gradio(markdown_text[content_start:content_end])
        )
        result.append(right)
        position = content_end + len(right)

    return "".join(result)


def prepare_markdown_for_gradio_preview(markdown_text, latex_delimiters):
    """准备传给 gr.Markdown 的预览文本；原始 Markdown 文件内容不在这里改写。"""
    if not isinstance(markdown_text, str):
        return markdown_text
    return escape_latex_blocks_for_gradio_preview(markdown_text, latex_delimiters)


def normalize_language(language):
    if '(' in language and ')' in language:
        language = language.split('(')[0].strip()
    return validate_public_ocr_lang(language)


def resolve_parse_method(file_path, is_ocr, tier=DEFAULT_GRADIO_TIER):
    """根据文件类型和 OCR 开关解析旧 multipart parse_method，tier 只保留签名语义。"""
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    if file_suffix in office_suffixes:
        return "auto"
    return "ocr" if is_ocr else "auto"


def is_image_analysis_option_visible(tier: object) -> bool:
    """判断 Gradio 图片分析开关是否应展示；当前公开 tier 不暴露 extra_high。"""
    if tier not in GRADIO_TIER_CHOICES:
        return False
    try:
        runtime = resolve_gradio_runtime_options(str(tier), use_remote_server=False)
    except ValueError:
        return False
    return runtime.effort == MAX_HYBRID_EFFORT


def is_ocr_language_option_visible(tier: object) -> bool:
    """判断 OCR 语言选项是否展示；只有 standard 对应的 Hybrid medium 暴露语言输入。"""
    if tier not in GRADIO_TIER_CHOICES:
        return False
    try:
        runtime = resolve_gradio_runtime_options(str(tier), use_remote_server=False)
    except ValueError:
        return False
    return runtime.effort == LOCAL_HYBRID_EFFORT


def is_force_ocr_option_visible(tier: object) -> bool:
    """判断强制 OCR 开关是否展示；公开 tier 均基于 Hybrid，保留强制 OCR。"""
    return tier in GRADIO_TIER_CHOICES


def should_use_client_side_output_generation(client_side_output_generation):
    """判断当前 Gradio 任务是否需要在客户端生成最终输出。"""
    return client_side_output_generation


def create_gradio_run_paths(file_path, output_root="./output"):
    """生成 Gradio 单次任务目录，统一返回绝对路径以便文件预览组件访问。"""
    run_id = f"{time.strftime('%y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe_stem(Path(file_path).stem)}"
    output_root_path = Path(output_root).expanduser().resolve()
    run_root = output_root_path / "gradio" / run_id
    extract_root = run_root / "result"
    archive_zip_path = run_root / f"{safe_stem(Path(file_path).stem)}.zip"
    return run_root, extract_root, archive_zip_path


def build_gradio_allowed_paths(output_root="./output"):
    """生成 Gradio 可公开访问目录，确保预览 HTTP 图片链接能被 /gradio_api/file= 读取。"""
    allowed_paths = []
    for item in os.environ.get("GRADIO_ALLOWED_PATHS", "").split(","):
        item = item.strip()
        if item:
            allowed_paths.append(item)

    output_path = str(Path(output_root).resolve())
    if output_path not in allowed_paths:
        allowed_paths.append(output_path)
    return allowed_paths


async def resolve_server_health(http_client, api_url):
    """兼容旧调用名：当前 Gradio 健康检查统一走 v1 /tiers 探测。"""
    return await resolve_v1_server_health(http_client, api_url)


async def ensure_local_api_ready_for_gradio_startup(
    timeout_seconds: float = _api_client.LOCAL_API_STARTUP_TIMEOUT_SECONDS,
):
    """兼容旧调用名：等待本地 v1 API server 就绪。"""
    local_server, started_now = _gradio_local_api_server.ensure_started()
    if started_now:
        logger.info(f"Started local MinerU v1 API at {local_server.base_url}")

    async with httpx.AsyncClient(
        timeout=_api_client.build_http_timeout(),
        follow_redirects=True,
    ) as http_client:
        return await wait_for_v1_api_ready(
            http_client,
            local_server,
            timeout_seconds=timeout_seconds,
        )


def maybe_prepare_local_api_for_gradio_startup(
    *,
    api_url: str | None,
    enable_vlm_preload: bool,
):
    """保留旧函数入口；v1 server 已在构建 Dropdown 前完成探测。"""
    return None


def resolve_gradio_max_concurrent_requests(api_url, server_health):
    if api_url is None:
        return server_health.max_concurrent_requests

    return _api_client.resolve_effective_max_concurrent_requests(
        local_max=_api_client.read_max_concurrent_requests(
            default=_api_client.DEFAULT_MAX_CONCURRENT_REQUESTS
        ),
        server_max=server_health.max_concurrent_requests,
    )


async def _run_to_markdown_job(
    file_path,
    end_pages=10,
    is_ocr=False,
    formula_enable=True,
    table_enable=True,
    image_analysis=True,
    tier=DEFAULT_GRADIO_TIER,
    language="ch",
    use_remote_server=False,
    url=None,
    api_url=None,
    client_side_output_generation=False,
    status_callback: Callable[[str], None] | None = None,
):
    if file_path is None:
        return "", "", "", None, None

    # v1 API 当前只公开 tier 和 page_range；旧 UI 参数保留用于前端兼容，不写入未公开 payload。
    del is_ocr, formula_enable, table_enable, image_analysis, language, url, client_side_output_generation

    def emit_status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    file_path = str(file_path)
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    runtime = resolve_gradio_runtime_options(tier, use_remote_server=use_remote_server)
    backend = runtime.backend
    effort = runtime.effort
    page_range = build_v1_gradio_page_range(end_pages)
    run_root, extract_root, archive_zip_path = create_gradio_run_paths(file_path)
    run_root.mkdir(parents=True, exist_ok=True)

    async with httpx.AsyncClient(
        timeout=_api_client.build_http_timeout(),
        follow_redirects=True,
    ) as http_client:
        emit_status(STATUS_PREPARING_REQUEST)
        emit_status(STATUS_CHECKING_SERVER)
        server_health = await resolve_v1_server_health(http_client, api_url)
        effective_max_concurrent_requests = resolve_gradio_max_concurrent_requests(
            api_url=api_url,
            server_health=server_health,
        )
        async with _gradio_request_concurrency_limiter.acquire(
            effective_max_concurrent_requests,
            on_wait=lambda snapshot: emit_status(
                format_concurrency_wait_message(snapshot)
            ),
        ):
            emit_status(STATUS_SUBMITTING_TASK)
            parser = MinerUApiParser(api_url=server_health.base_url, tier=runtime.tier)
            parse_result = await parser.parse_async(
                file_path,
                page_range=page_range,
            )
            emit_status(STATUS_DOWNLOADING_RESULT)
            emit_status(STATUS_PROCESSING_OUTPUT)
            persisted_output = await asyncio.to_thread(
                persist_v1_gradio_result,
                parse_result=parse_result,
                file_path=file_path,
                extract_root=extract_root,
                archive_zip_path=archive_zip_path,
                backend=backend,
                effort=effort,
                page_range=page_range,
            )

    file_name = persisted_output.file_name
    local_md_dir = persisted_output.local_md_dir
    preview_pdf_path = persisted_output.preview_pdf_path

    md_path = local_md_dir / f"{file_name}.md"
    txt_content = md_path.read_text(encoding="utf-8")
    md_content = replace_image_with_gradio_file_urls(txt_content, local_md_dir)
    content_list_json = read_gradio_content_list_json(local_md_dir, file_name)

    if file_suffix in office_suffixes:
        preview_pdf_path = None

    emit_status(STATUS_COMPLETED)
    return md_content, txt_content, content_list_json, str(archive_zip_path), str(preview_pdf_path) if preview_pdf_path else None


async def stream_to_markdown(
    file_path,
    end_pages=10,
    is_ocr=False,
    formula_enable=True,
    table_enable=True,
    image_analysis=True,
    tier=DEFAULT_GRADIO_TIER,
    language="ch",
    use_remote_server=False,
    url=None,
    api_url=None,
    client_side_output_generation=False,
):
    status_state = StatusPanelState()
    job_task: asyncio.Task | None = None
    queue_get_task: asyncio.Task | None = None
    timer_task: asyncio.Task | None = None
    yield status_state.render(), None, "", "", "", gr.skip()

    if file_path is None:
        return

    status_queue: asyncio.Queue[str] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def enqueue_status(message: str) -> None:
        loop.call_soon_threadsafe(status_queue.put_nowait, message)

    try:
        job_task = asyncio.create_task(
            _run_to_markdown_job(
                file_path=file_path,
                end_pages=end_pages,
                is_ocr=is_ocr,
                formula_enable=formula_enable,
                table_enable=table_enable,
                image_analysis=image_analysis,
                tier=tier,
                language=language,
                use_remote_server=use_remote_server,
                url=url,
                api_url=api_url,
                client_side_output_generation=client_side_output_generation,
                status_callback=enqueue_status,
            )
        )

        while True:
            if job_task.done() and status_queue.empty():
                status_state.finalize_processing()
                status_state.finalize_queue()
                break

            queue_get_task = asyncio.create_task(status_queue.get())
            wait_tasks: set[asyncio.Task] = {job_task, queue_get_task}
            timer_task = None
            animation_interval = status_state.animation_interval_seconds
            if animation_interval is not None:
                timer_task = asyncio.create_task(
                    asyncio.sleep(animation_interval)
                )
                wait_tasks.add(timer_task)

            done, pending = await asyncio.wait(
                wait_tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

            if queue_get_task in done:
                message = queue_get_task.result()
                if status_state.append(message):
                    yield status_state.render(), None, "", "", "", gr.skip()
            elif timer_task is not None and timer_task in done:
                if status_state.tick():
                    yield status_state.render(), None, "", "", "", gr.skip()
            else:
                queue_get_task.cancel()
                await asyncio.gather(queue_get_task, return_exceptions=True)

            for pending_task in pending:
                if pending_task is job_task:
                    continue
                pending_task.cancel()
                await asyncio.gather(pending_task, return_exceptions=True)
            queue_get_task = None
            timer_task = None

        while not status_queue.empty():
            status_state.append(status_queue.get_nowait())
    except Exception as exc:
        status_state.append(format_failed_status(exc))
        yield status_state.render(), None, "", "", "", gr.skip()
        raise
    finally:
        for task in (queue_get_task, timer_task, job_task):
            if task is None or task.done():
                continue
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        md_content, txt_content, content_list_json, archive_zip_path, preview_pdf_path = await job_task
    except Exception as exc:
        status_state.append(format_failed_status(exc))
        yield status_state.render(), None, "", "", "", gr.skip()
        raise

    status_state.append(STATUS_COMPLETED)
    yield (
        status_state.render(),
        archive_zip_path,
        md_content,
        txt_content,
        content_list_json,
        preview_pdf_path,
    )


def resolve_preview_pdf_path(local_md_dir, file_name):
    layout_pdf_path = os.path.join(local_md_dir, file_name + '_layout.pdf')
    if os.path.exists(layout_pdf_path):
        return layout_pdf_path

    origin_pdf_path = os.path.join(local_md_dir, file_name + '_origin.pdf')
    if os.path.exists(origin_pdf_path):
        logger.warning(
            f"Layout preview PDF not found for {file_name}, "
            f"falling back to origin PDF: {origin_pdf_path}"
        )
        return origin_pdf_path

    logger.warning(f"No preview PDF found for {file_name} under {local_md_dir}")
    return None


latex_delimiters_type_a = [
    {'left': '$$', 'right': '$$', 'display': True},
    {'left': '$', 'right': '$', 'display': False},
]
latex_delimiters_type_b = [
    {'left': '\\(', 'right': '\\)', 'display': False},
    {'left': '\\[', 'right': '\\]', 'display': True},
]
latex_delimiters_type_all = latex_delimiters_type_a + latex_delimiters_type_b

header_template = load_resource_text('gradio_header.html')

HEADER_I18N_PLACEHOLDERS = {
    "{{HEADER_TITLE}}": "header_title",
    "{{HEADER_SUBTITLE}}": "header_subtitle",
    "{{HEADER_SUPPORT_TEXT}}": "header_support_text",
    "{{HEADER_STARS_ALT}}": "header_stars_alt",
    "{{HEADER_CODE_LINK}}": "header_code_link",
    "{{HEADER_MODEL_LINK}}": "header_model_link",
    "{{HEADER_PAPER_LINK}}": "header_paper_link",
    "{{HEADER_HOMEPAGE_LINK}}": "header_homepage_link",
    "{{HEADER_DOWNLOAD_LINK}}": "header_download_link",
}
HEADER_GRADIO_VERSION_CLASS_PLACEHOLDER = "{{HEADER_GRADIO_VERSION_CLASS}}"


def render_header_html(i18n):
    """渲染支持 i18n 的顶部 Header，保留静态模板中的样式和链接。"""
    rendered_header = header_template
    for placeholder, translation_key in HEADER_I18N_PLACEHOLDERS.items():
        if translation_key == "header_stars_alt":
            replacement = html_lib.escape(translate_ui(i18n, translation_key), quote=True)
        else:
            replacement = render_client_i18n_text(i18n, translation_key)
        rendered_header = rendered_header.replace(
            placeholder,
            replacement,
        )
    rendered_header = rendered_header.replace(
        HEADER_GRADIO_VERSION_CLASS_PLACEHOLDER,
        " mineru-gradio6-header" if IS_GRADIO_6 else "",
    )
    return rendered_header

all_lang = list(PUBLIC_OCR_LANGUAGE_CHOICES)


def safe_stem(file_path):
    stem = Path(file_path).stem
    # 只保留字母、数字、下划线和点，其他字符替换为下划线
    return re.sub(r'[^\w.]', '_', stem)


def to_pdf(file_path):

    if file_path is None:
        return None

    pdf_bytes = read_fn(file_path)

    # unique_filename = f'{uuid.uuid4()}.pdf'
    unique_filename = f'{safe_stem(file_path)}.pdf'

    # 构建完整的文件路径
    tmp_file_path = os.path.join(os.path.dirname(file_path), unique_filename)

    # 将字节数据写入文件
    with open(tmp_file_path, 'wb') as tmp_pdf_file:
        tmp_pdf_file.write(pdf_bytes)

    return tmp_file_path


def to_pdf_preview(file_path):
    """用于 PDF 预览的转换函数，office 文件不支持预览，返回 None。"""
    if file_path is None:
        return None
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    if file_suffix in office_suffixes:
        return None
    return to_pdf(file_path)


def build_gradio_file_public_url(file_path, request: gr.Request):
    """根据当前 Gradio 请求构建上传文件的外部访问 URL，兼容本地和反代环境。"""
    headers = getattr(request, "headers", None) or {}
    host = (
        headers.get('x-forwarded-host')
        or headers.get('host', 'localhost:7860')
    )
    proto = headers.get('x-forwarded-proto', 'http')
    return f"{proto}://{host}/gradio_api/file={quote(str(file_path), safe='/:')}"


def build_short_gradio_file_url(public_url, file_path):
    """生成页面展示用的短链接文本，保留站点前缀和文件名尾部以避免长路径撑破预览区。"""
    base_url = public_url.split("/gradio_api/file=", 1)[0]
    file_name = Path(file_path).name
    file_suffix = Path(file_name).suffix
    file_stem = Path(file_name).stem
    short_file_name = f"{file_stem[-12:]}{file_suffix}" if file_stem else file_name
    return f"{base_url}/....{short_file_name}"


def build_office_preview_html(file_path, request: gr.Request, i18n=None):
    """生成 Office 在线预览 HTML，并提示该预览依赖外部 Microsoft 服务访问。"""
    public_url = build_gradio_file_public_url(file_path, request)
    short_public_url = build_short_gradio_file_url(public_url, file_path)
    viewer_url = (
        "https://view.officeapps.live.com/op/embed.aspx?src="
        f"{quote(public_url, safe='')}"
    )
    title = render_client_i18n_text(i18n, "office_preview_title")
    notice = render_client_i18n_text(i18n, "office_preview_notice")
    source_link = render_client_i18n_text(i18n, "office_preview_source_link")
    ignore_once = render_client_i18n_text(i18n, "office_preview_ignore_once")
    ignore_forever = render_client_i18n_text(i18n, "office_preview_ignore_forever")
    return (
        '<div class="office-preview-shell">'
        '<div class="office-preview-notice">'
        '<div class="office-preview-copy">'
        f"<strong>{title}</strong>"
        f"<span>{notice}</span>"
        '<div class="office-preview-source-link">'
        f"{source_link}: "
        f"{html_lib.escape(short_public_url)}"
        "</div>"
        "</div>"
        '<div class="office-preview-actions">'
        f'<button type="button" class="office-preview-ignore-once">{ignore_once}</button>'
        f'<button type="button" class="office-preview-ignore-forever">{ignore_forever}</button>'
        "</div>"
        "</div>"
        f'<iframe class="office-preview-frame" src="{html_lib.escape(viewer_url)}" '
        'frameborder="0"></iframe>'
        "</div>"
    )


def update_file_options_html(file_path, request: gr.Request, i18n=None):
    """处理文件上传第一阶段：根据文件类型更新 options_group 和 office_html。
    将 doc_show（gradio_pdf.PDF）的更新拆分到独立的 .then() 事件中，
    以规避 gradio_pdf 0.0.24 在 Gradio 6 中对 value=None 处理不当导致的
    整个事件 processing 状态卡死的兼容性问题。
    """
    if file_path is None:
        return (
            gr.update(visible=True),             # options_group - 恢复显示
            gr.update(value="", visible=False),  # office_html - 隐藏
        )

    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    is_office = file_suffix in office_suffixes

    if is_office:
        html_content = build_office_preview_html(file_path, request, i18n)
        return (
            gr.update(visible=False),                    # options_group - 隐藏
            gr.update(value=html_content, visible=True), # office_html - 显示
        )
    else:
        return (
            gr.update(visible=True),             # options_group - 显示
            gr.update(value="", visible=False),  # office_html - 隐藏
        )


def update_doc_show(file_path):
    """处理文件上传第二阶段：单独更新 doc_show（gradio_pdf.PDF）组件。
    对 office 文件仅改变 visible，避免传递 value=None 触发
    gradio_pdf 0.0.24 在 Gradio 6 中无法完成的加载周期。
    """
    if file_path is None:
        # 无文件时恢复显示并清空（clear 按钮路径）
        return gr.update(value=None, visible=True)

    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    is_office = file_suffix in office_suffixes

    if is_office:
        # 仅隐藏，不改变 value，避免触发 gradio_pdf 加载周期导致事件 pending 卡死
        return gr.update(visible=False)
    else:
        pdf_path = to_pdf_preview(file_path)
        return gr.update(value=pdf_path, visible=True)


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.option(
    '--enable-example',
    'example_enable',
    type=bool,
    help="Enable example files for input."
         "The example files to be input need to be placed in the `examples` folder within the directory where the command is currently executed.",
    default=True,
)
@click.option(
    '--enable-http-client',
    'http_client_enable',
    type=bool,
    help="Enable http-client backend to link openai-compatible servers.",
    default=False,
)
@click.option(
    '--enable-api',
    'api_enable',
    type=bool,
    help="Enable gradio API for serving the application.",
    default=True,
)
@click.option(
    '--max-convert-pages',
    'max_convert_pages',
    type=int,
    help="Set the maximum number of pages to convert from PDF to Markdown.",
    default=1000,
)
@click.option(
    '--server-name',
    'server_name',
    type=str,
    help="Set the server name for the Gradio app.",
    default=None,
)
@click.option(
    '--server-port',
    'server_port',
    type=int,
    help="Set the server port for the Gradio app.",
    default=None,
)
@click.option(
    '--api-url',
    'api_url',
    type=str,
    help="MinerU FastAPI base URL. If omitted, gradio starts a reusable local mineru-api service.",
    default=None,
)
@click.option(
    '--enable-vlm-preload',
    'enable_vlm_preload',
    type=bool,
    help="Preload the local VLM model when gradio starts a local mineru-api service.",
    default=False,
)
@click.option(
    '--client-side-output-generation',
    'client_side_output_generation',
    type=bool,
    help="Generate markdown and content lists locally from server-returned middle json.",
    default=False,
)
@click.option(
    '--latex-delimiters-type',
    'latex_delimiters_type',
    type=click.Choice(['a', 'b', 'all']),
    help="Set the type of LaTeX delimiters to use in Markdown rendering:"
         "'a' for type '$', 'b' for type '()[]', 'all' for both types.",
    default='all',
)
def main(ctx,
        example_enable,
        http_client_enable,
        api_enable, max_convert_pages,
        server_name, server_port, api_url, enable_vlm_preload,
        client_side_output_generation, latex_delimiters_type, **kwargs
):

    # 创建 i18n 实例，支持中英文
    i18n = gr.I18n(
        en={
            "upload_file": "Select or paste a file to upload\nPDF, image, DOCX, PPTX, or XLSX",
            "header_title": "MinerU 3: Document Extraction Demo",
            "header_subtitle": "Open-source document extraction for PDF, DOCX, PPTX, XLSX, and images to Markdown and JSON.",
            "header_support_text": "If you found our project helpful, please give us a ⭐️ to support us!",
            "header_stars_alt": "stars",
            "header_code_link": "Code",
            "header_model_link": "Model",
            "header_paper_link": "Paper",
            "header_homepage_link": "Homepage",
            "header_download_link": "Download",
            "max_pages": "Max convert pages",
            "tier": "Tier",
            "tier_info_default": "Select the parsing tier.",
            "tier_info_standard": "Standard tier uses fast local Hybrid parsing.",
            "tier_info_pro": "Pro tier uses higher quality Hybrid parsing.",
            "use_remote_server": "Use remote server",
            "use_remote_server_info": "Send parsing requests to an OpenAI-compatible server.",
            "server_url": "Server URL",
            "server_url_info": "OpenAI-compatible server URL.",
            "recognition_options": "**Recognition Options:**",
            "advanced_options": "Advanced options",
            "table_enable": "Enable table recognition",
            "table_info": "If disabled, tables will be shown as images.",
            "image_analysis_enable": "Enable image analysis",
            "image_analysis_info": "If disabled, image/chart blocks will keep layout positions but skip VLM image/chart analysis.",
            "formula_label_vlm": "Enable display formula recognition",
            "formula_label_hybrid": "Enable inline formula recognition",
            "formula_info_vlm": "If disabled, display formulas will be shown as images.",
            "formula_info_hybrid": "If disabled, inline formulas will not be detected or parsed.",
            "ocr_language": "OCR Language",
            "ocr_language_info": "Select the OCR language for Hybrid medium local OCR and table OCR.",
            "force_ocr": "Force enable OCR",
            "force_ocr_info": "Enable only if the result is extremely poor. Hybrid medium requires correct OCR language.",
            "force_ocr_info_hybrid": "Enable only if the result is extremely poor.",
            "convert": "Convert",
            "clear": "Clear",
            "doc_preview": "Document preview",
            "examples": "Examples:",
            "convert_status": "Conversion Status",
            "convert_result": "Convert result",
            "result_file": "Result file",
            "md_rendering": "Markdown rendering",
            "md_text": "Markdown text",
            "content_list_json": "JSON Content List",
            "status_idle_title": "Waiting",
            "status_idle_hint": "Upload a file and start conversion.",
            "status_latest": "Latest status",
            "status_step_prepare": "Prepare",
            "status_step_check": "Check service",
            "status_step_submit": "Submit",
            "status_step_queue": "Queue",
            "status_step_process": "Parse",
            "status_step_download": "Download",
            "status_step_outputs": "Build outputs",
            "status_step_done": "Done",
            "status_step_failed": "Failed",
            "office_preview_title": "Office online preview",
            "office_preview_notice": "This preview requires the current file to be reachable by Microsoft Office Online. Conversion does not depend on this preview.",
            "office_preview_source_link": "File url",
            "office_preview_ignore_once": "Dismiss",
            "office_preview_ignore_forever": "Always dismiss",
        },
        zh={
            "upload_file": "请选择或粘贴要上传的文件\nPDF、图片、DOCX、PPTX 或 XLSX",
            "header_title": "MinerU 3：文档提取演示",
            "header_subtitle": "开源文档提取工具，支持将 PDF、DOCX、PPTX、XLSX 和图片转换为 Markdown 与 JSON。",
            "header_support_text": "如果我们的项目对你有帮助，请点亮 ⭐️ 支持我们！",
            "header_stars_alt": "GitHub 星标",
            "header_code_link": "代码",
            "header_model_link": "模型",
            "header_paper_link": "论文",
            "header_homepage_link": "主页",
            "header_download_link": "下载",
            "max_pages": "最大转换页数",
            "tier": "解析 tier",
            "tier_info_default": "选择文档解析 tier。",
            "tier_info_standard": "standard 使用快速本地 Hybrid 解析。",
            "tier_info_pro": "pro 使用更高质量的 Hybrid 解析。",
            "use_remote_server": "使用远端服务",
            "use_remote_server_info": "将解析请求发送到 OpenAI 兼容服务。",
            "server_url": "服务器地址",
            "server_url_info": "OpenAI 兼容服务器地址。",
            "recognition_options": "**识别选项：**",
            "advanced_options": "高级选项",
            "table_enable": "启用表格识别",
            "table_info": "禁用后，表格将显示为图片。",
            "image_analysis_enable": "启用图片分析",
            "image_analysis_info": "禁用后，图片/图表块仍保留版面位置，但跳过 VLM 图片/图表分析。",
            "formula_label_vlm": "启用行间公式识别",
            "formula_label_hybrid": "启用行内公式识别",
            "formula_info_vlm": "禁用后，行间公式将显示为图片。",
            "formula_info_hybrid": "禁用后，行内公式将不会被检测或解析。",
            "ocr_language": "OCR 语言",
            "ocr_language_info": "为 Hybrid medium 的本地 OCR 和表格 OCR 选择语言。",
            "force_ocr": "强制启用 OCR",
            "force_ocr_info": "仅在识别效果极差时启用；Hybrid medium 需选择正确的 OCR 语言。",
            "force_ocr_info_hybrid": "仅在识别效果极差时启用。",
            "convert": "转换",
            "clear": "清除",
            "doc_preview": "文档预览",
            "examples": "示例：",
            "convert_status": "转换状态",
            "convert_result": "转换结果",
            "result_file": "结果文件",
            "md_rendering": "Markdown 渲染",
            "md_text": "Markdown 文本",
            "content_list_json": "JSON 内容列表",
            "status_idle_title": "等待任务",
            "status_idle_hint": "上传文件后开始转换。",
            "status_latest": "最新状态",
            "status_step_prepare": "准备请求",
            "status_step_check": "检查服务",
            "status_step_submit": "提交任务",
            "status_step_queue": "排队",
            "status_step_process": "解析中",
            "status_step_download": "下载结果",
            "status_step_outputs": "整理输出",
            "status_step_done": "完成",
            "status_step_failed": "失败",
            "office_preview_title": "Office 在线预览",
            "office_preview_notice": "该预览需要当前文件可被 Microsoft 在线预览服务访问，转换不依赖该预览。",
            "office_preview_source_link": "文件链接",
            "office_preview_ignore_once": "忽略",
            "office_preview_ignore_forever": "不再提示",
        },
    )

    # 根据 tier 获取公式识别标签（闭包函数以支持 i18n）
    def get_formula_label(tier_choice):
        return i18n("formula_label_hybrid")

    def get_formula_info(tier_choice):
        return i18n("formula_info_hybrid") if tier_choice in GRADIO_TIER_CHOICES else ""

    def get_tier_info(tier_choice):
        return i18n(select_tier_info_key(tier_choice))

    def get_force_ocr_info(tier_choice):
        """根据 tier 返回强制 OCR 控件说明，当前公开 tier 统一使用 Hybrid 文案。"""
        return i18n(select_force_ocr_info_key(tier_choice))

    # 更新界面函数
    def update_interface(tier_choice, use_remote_server):
        """根据 tier 和远端开关更新 Gradio 联动控件，保持前端显隐规则集中。"""
        use_remote_server = bool(use_remote_server)
        formula_label_update = gr.update(label=get_formula_label(tier_choice), info=get_formula_info(tier_choice))
        tier_info_update = gr.update(info=get_tier_info(tier_choice))
        force_ocr_update = gr.update(info=get_force_ocr_info(tier_choice))
        image_analysis_update = gr.update(visible=is_image_analysis_option_visible(tier_choice))
        language_update = gr.update(visible=is_ocr_language_option_visible(tier_choice))
        client_options_update = gr.update(visible=use_remote_server)
        ocr_options_update = gr.update(visible=is_force_ocr_option_visible(tier_choice))

        return (
            client_options_update,
            ocr_options_update,
            force_ocr_update,
            formula_label_update,
            tier_info_update,
            image_analysis_update,
            language_update,
        )

    del kwargs
    if enable_vlm_preload:
        logger.warning(
            "Ignoring --enable-vlm-preload because Gradio now launches the v1 parser API server."
        )
    _gradio_local_api_server.configure(ctx.args)
    startup_health = resolve_gradio_v1_startup_health(api_url=api_url)
    tier_choices = startup_health.tier_choices

    if latex_delimiters_type == 'a':
        latex_delimiters = latex_delimiters_type_a
    elif latex_delimiters_type == 'b':
        latex_delimiters = latex_delimiters_type_b
    elif latex_delimiters_type == 'all':
        latex_delimiters = latex_delimiters_type_all
    else:
        raise ValueError(f"Invalid latex delimiters type: {latex_delimiters_type}.")


    async def convert_to_markdown_stream(
        file_path,
        end_pages=10,
        is_ocr=False,
        formula_enable=True,
        table_enable=True,
        image_analysis=True,
        tier=DEFAULT_GRADIO_TIER,
        language="ch",
        use_remote_server=False,
        url=None,
        request: gr.Request = None,
    ):
        request_locale = resolve_request_locale(request)
        async for update in stream_to_markdown(
            file_path=file_path,
            end_pages=end_pages,
            is_ocr=is_ocr,
            formula_enable=formula_enable,
            table_enable=table_enable,
            image_analysis=image_analysis,
            tier=tier,
            language=language,
            use_remote_server=use_remote_server,
            url=url,
            api_url=api_url,
            client_side_output_generation=client_side_output_generation,
        ):
            update = (
                render_status_steps_html(update[0], i18n, locale=request_locale),
                update[1],
                prepare_markdown_for_gradio_preview(update[2], latex_delimiters),
                *update[3:],
            )
            yield update

    suffixes = [f".{suffix}" for suffix in pdf_suffixes + image_suffixes + office_suffixes]
    _blocks_kwargs = {} if IS_GRADIO_6 else {"css": APP_CSS, "js": APP_JS}
    with gr.Blocks(**_blocks_kwargs) as demo:
        gr.HTML(render_header_html(i18n))
        with gr.Row(elem_classes=["mineru-workspace-row"]):
            with gr.Column(variant='panel', scale=2, min_width=280, elem_classes=["mineru-control-column"]):
                input_file = gr.File(
                    label=i18n("upload_file"),
                    file_types=suffixes,
                    elem_classes=["mineru-upload-file"],
                )
                preferred_tier = default_v1_gradio_tier(tier_choices)
                with gr.Group(elem_classes=["mineru-backend-options-block"]):
                    tier = gr.Dropdown(
                        list(tier_choices),
                        label=i18n("tier"),
                        value=preferred_tier,
                        info=get_tier_info(preferred_tier),
                        elem_classes=["mineru-tier-select"],
                    )
                    use_remote_server = gr.Checkbox(
                        label=i18n("use_remote_server"),
                        value=False,
                        visible=http_client_enable,
                        info=i18n("use_remote_server_info"),
                        elem_classes=["mineru-remote-server-toggle"],
                    )
                    with gr.Row(
                        visible=frontend_managed_initial_visibility(False),
                        elem_classes=["mineru-client-options"],
                    ) as client_options:
                        url = gr.Textbox(
                            label=i18n("server_url"),
                            value='http://localhost:30000',
                            placeholder='http://localhost:30000',
                            info=i18n("server_url_info"),
                        )
                # 下面这些选项在上传 office 文件时会被自动隐藏
                with gr.Group() as options_group:
                    max_pages = gr.Slider(1, max_convert_pages, max_convert_pages, step=1, label=i18n("max_pages"))
                    advanced_bu = gr.Button(
                        i18n("advanced_options"),
                        size="sm",
                        elem_classes=["mineru-advanced-open"],
                    )
                with gr.Row(elem_classes=["mineru-actions"]):
                    change_bu = gr.Button(i18n("convert"), variant="primary", scale=1, min_width=0)
                    clear_bu = gr.ClearButton(value=i18n("clear"), scale=1, min_width=0)
                output_file = gr.File(
                    label=i18n("convert_result"),
                    interactive=False,
                    elem_classes=["mineru-result-file"],
                )
                status_panel = gr.HTML(
                    value=render_status_steps_html("", i18n),
                    label=i18n("convert_status"),
                    elem_classes=["mineru-status-panel"],
                )

            _doc_preview_label = "doc preview" if IS_GRADIO_6 else i18n("doc_preview")
            # preview_content_height 约束文档预览/Markdown 的内容区；gradio_pdf 的 height
            # 实际是单页 canvas 高度，需要单独扣除 label 和分页器占用，避免上传 PDF 后撑高预览块。
            preview_content_height = 775
            pdf_preview_page_height = 720
            with gr.Column(variant='panel', scale=4, min_width=340, elem_classes=["mineru-preview-pane"]):
                doc_show = PDF(
                    label=_doc_preview_label,
                    interactive=False,
                    visible=True,
                    height=pdf_preview_page_height,
                )
                office_html = gr.HTML(
                    value="",
                    visible=False,
                    min_height=preview_content_height,
                    elem_classes=["mineru-office-preview-html"],
                )

            with gr.Column(variant='panel', scale=4, min_width=340, elem_classes=["mineru-markdown-pane"]):
                _md_copy_kwargs = {"buttons": ["copy"]} if IS_GRADIO_6 else {"show_copy_button": True}
                _textarea_copy_kwargs = {"buttons": ["copy"]} if IS_GRADIO_6 else {"show_copy_button": True}
                with gr.Tabs(elem_classes=["mineru-markdown-tabs"]):
                    with gr.Tab(i18n("md_rendering")):
                        md = gr.Markdown(
                            label=i18n("md_rendering"),
                            height=preview_content_height,
                            elem_classes=["mineru-markdown-output"],
                            latex_delimiters=latex_delimiters,
                            line_breaks=True,
                            **_md_copy_kwargs
                        )
                    with gr.Tab(i18n("md_text")):
                        md_text = gr.Code(
                            lines=28,
                            language="markdown",
                            label=i18n("md_text"),
                            interactive=False,
                            wrap_lines=True,
                            show_label=False,
                            elem_classes=["mineru-markdown-text"],
                        )
                    with gr.Tab(i18n("content_list_json")):
                        content_list_json = gr.Code(
                            lines=28,
                            language="json",
                            label=i18n("content_list_json"),
                            interactive=False,
                            wrap_lines=True,
                            show_label=False,
                            elem_classes=["mineru-content-list-json"],
                        )

        if example_enable:
            example_root = os.path.join(os.getcwd(), 'examples')
            if os.path.exists(example_root):
                example_files = [
                    os.path.join(example_root, _) for _ in os.listdir(example_root)
                    if _.endswith(tuple(suffixes))
                ]
                if example_files:
                    with gr.Accordion(i18n("examples"), open=True, elem_classes=["mineru-examples-panel"]):
                        gr.Examples(
                            examples=example_files,
                            inputs=input_file,
                            elem_id="mineru-example-files",
                            label=None,
                        )

        with gr.Column(elem_classes=["mineru-advanced-popover"]):
            with gr.Column(elem_classes=["mineru-advanced-card"]):
                with gr.Group():
                    table_enable = gr.Checkbox(label=i18n("table_enable"), value=True, info=i18n("table_info"))
                    formula_enable = gr.Checkbox(label=get_formula_label(preferred_tier), value=True, info=get_formula_info(preferred_tier))
                    image_analysis = gr.Checkbox(
                        label=i18n("image_analysis_enable"),
                        value=True,
                        visible=frontend_managed_initial_visibility(
                            is_image_analysis_option_visible(preferred_tier)
                        ),
                        info=i18n("image_analysis_info"),
                        elem_classes=["mineru-image-analysis-option"],
                    )
                with gr.Column(elem_classes=["mineru-force-ocr-option"]) as ocr_options:
                    with gr.Group():
                        with gr.Column(elem_classes=["mineru-ocr-language-options"]):
                            language = gr.Dropdown(
                                all_lang,
                                label=i18n("ocr_language"),
                                value=all_lang[0],
                                info=i18n("ocr_language_info"),
                                visible=frontend_managed_initial_visibility(
                                    is_ocr_language_option_visible(preferred_tier)
                                ),
                            )
                        is_ocr = gr.Checkbox(
                            label=i18n("force_ocr"),
                            value=False,
                            info=i18n(select_force_ocr_info_key(preferred_tier)),
                        )

        # 添加事件处理
        _private_api_kwargs = (
            {"api_visibility": "private", "queue": False}
            if IS_GRADIO_6
            else {"api_name": False, "queue": False}
        )
        tier.change(
            fn=update_interface,
            inputs=[tier, use_remote_server],
            outputs=[client_options, ocr_options, is_ocr, formula_enable, tier, image_analysis, language],
            **_private_api_kwargs
        )
        use_remote_server.change(
            fn=update_interface,
            inputs=[tier, use_remote_server],
            outputs=[client_options, ocr_options, is_ocr, formula_enable, tier, image_analysis, language],
            **_private_api_kwargs
        )
        # 添加demo.load事件，在页面加载时触发一次界面更新
        demo.load(
            fn=update_interface,
            inputs=[tier, use_remote_server],
            outputs=[client_options, ocr_options, is_ocr, formula_enable, tier, image_analysis, language],
            **_private_api_kwargs
        )
        clear_bu.add([input_file, md, doc_show, md_text, content_list_json, output_file, is_ocr, office_html, status_panel])

        def reset_primary_ui():
            """清除主界面状态。高级气泡由前端点击外部逻辑自动收起。"""
            return (
                gr.update(visible=True),
                gr.update(value=None, visible=True),
                gr.update(value="", visible=False),
                gr.update(value=render_status_steps_html("", i18n)),
                gr.update(value=""),
            )

        # 清除按钮额外重置 UI 可见性（ClearButton 不一定触发 input_file.change）
        clear_bu.click(
            fn=reset_primary_ui,
            inputs=[],
            outputs=[options_group, doc_show, office_html, status_panel, content_list_json],
            **_private_api_kwargs
        )

        def update_file_options_html_for_ui(file_path, request: gr.Request):
            """绑定当前 i18n 的文件上传 UI 更新函数，避免事件签名暴露额外参数。"""
            return update_file_options_html(file_path, request, i18n)

        # 第一阶段：快速更新 options_group 和 office_html，不涉及 gradio_pdf 组件
        # 第二阶段（.then）：单独更新 doc_show，使 office_html 的 processing 遮罩
        # 在第一阶段完成后立即消失，规避 gradio_pdf 0.0.24 与 Gradio 6 的兼容性问题。
        input_file.change(
            fn=update_file_options_html_for_ui,
            inputs=input_file,
            outputs=[options_group, office_html],
            **_private_api_kwargs
        ).then(
            fn=update_doc_show,
            inputs=input_file,
            outputs=[doc_show],
            **_private_api_kwargs
        )
        _to_md_api_kwargs = (
            {
                "api_visibility": "public" if api_enable else "private",
                "queue": True,
                "show_progress": "hidden",
            }
            if IS_GRADIO_6
            else {
                "api_name": "to_markdown" if api_enable else False,
                "queue": True,
                "show_progress": "hidden",
            }
        )
        change_bu.click(
            fn=convert_to_markdown_stream,
            inputs=[input_file, max_pages, is_ocr, formula_enable, table_enable, image_analysis, tier, language, use_remote_server, url],
            outputs=[status_panel, output_file, md, md_text, content_list_json, doc_show],
            **_to_md_api_kwargs
        )

    demo.queue(default_concurrency_limit=None)

    if IS_GRADIO_6:
        footer_links = ["gradio", "settings"]
        if api_enable:
            footer_links.append("api")
        _launch_kwargs = {"footer_links": footer_links, "css": APP_CSS, "head": APP_HEAD}
    else:
        _launch_kwargs = {"show_api": api_enable}
    maybe_prepare_local_api_for_gradio_startup(
        api_url=api_url,
        enable_vlm_preload=enable_vlm_preload,
    )
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        i18n=i18n,
        allowed_paths=build_gradio_allowed_paths(),
        **_launch_kwargs,
    )


if __name__ == '__main__':
    main()
