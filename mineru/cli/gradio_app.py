# Copyright (c) Opendatalab. All rights reserved.

import base64
import asyncio
import httpx
import os
import re
import sys
import threading
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

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

from mineru.cli.common import (
    docx_suffixes,
    image_suffixes,
    normalize_task_stem,
    office_suffixes,
    pdf_suffixes,
    read_fn,
)
from mineru.cli import api_client as _api_client
from mineru.cli.output_paths import resolve_parse_dir
from mineru.cli.vlm_preload import resolve_gradio_local_api_cli_args
from mineru.cli.visualization import VisualizationJob, run_visualization_job

_gradio_local_api_server = _api_client.ReusableLocalAPIServer()


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


@dataclass
class StatusPanelState:
    lines: list[str] = field(default_factory=list)
    processing_index: int | None = None
    processing_started_at: float | None = None
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
        if not self.lines or self.lines[-1] != message:
            self.lines.append(message)
            return True
        return False

    def start_processing(self) -> bool:
        if self.processing_started_at is not None:
            return self.tick_processing()

        self.processing_started_at = time.monotonic()
        self.processing_index = len(self.lines)
        self.lines.append(format_processing_status(0.0))
        return True

    def tick_processing(self) -> bool:
        if self.processing_started_at is None or self.processing_index is None:
            return False

        updated = format_processing_status(
            max(0.0, time.monotonic() - self.processing_started_at)
        )
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


def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def replace_image_with_base64(markdown_text, image_dir_path):
    # MIME类型映射
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
        '.svg': 'image/svg+xml',
    }

    def _path_to_data_uri(relative_path):
        file_ext = os.path.splitext(relative_path)[1].lower()
        if file_ext not in mime_types:
            return None
        try:
            full_path = os.path.join(image_dir_path, relative_path)
            base64_image = image_to_base64(full_path)
            return f'data:{mime_types[file_ext]};base64,{base64_image}'
        except Exception as e:
            logger.warning(f"Failed to convert image {relative_path} to base64: {e}")
            return None

    # 匹配Markdown中的图片标签 ![...](path)
    def replace_md(match):
        relative_path = match.group(1)
        data_uri = _path_to_data_uri(relative_path)
        if data_uri:
            return f'![{relative_path}]({data_uri})'
        return match.group(0)

    result = re.sub(r'\!\[(?:[^\]]*)\]\(([^)]+)\)', replace_md, markdown_text)

    # 匹配HTML表格中的 <img src="path"> (跳过已有的data: URI)
    def replace_html_src(match):
        relative_path = match.group(1)
        data_uri = _path_to_data_uri(relative_path)
        if data_uri:
            return f'src="{data_uri}"'
        return match.group(0)

    result = re.sub(r'src="(?!data:)([^"]+)"', replace_html_src, result)

    return result


def normalize_language(language):
    if '(' in language and ')' in language:
        return language.split('(')[0].strip()
    return language


def resolve_parse_method(file_path, is_ocr, backend):
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    if file_suffix in office_suffixes:
        return "auto"
    if backend.startswith("vlm"):
        return "auto"
    return "ocr" if is_ocr else "auto"


def create_gradio_run_paths(file_path, output_root="./output"):
    run_id = f"{time.strftime('%y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}_{safe_stem(Path(file_path).stem)}"
    run_root = Path(output_root) / "gradio" / run_id
    extract_root = run_root / "result"
    archive_zip_path = run_root / f"{safe_stem(Path(file_path).stem)}.zip"
    return run_root, extract_root, archive_zip_path


def build_gradio_upload_name(file_path):
    path = Path(file_path)
    return f"{normalize_task_stem(path.stem)}{path.suffix}"


def resolve_result_file_name(submit_response, extract_root, file_path):
    if submit_response.file_names:
        return submit_response.file_names[0]

    candidate_dirs = sorted(path.name for path in Path(extract_root).iterdir() if path.is_dir())
    if len(candidate_dirs) == 1:
        return candidate_dirs[0]
    return normalize_task_stem(Path(file_path).stem)


async def resolve_server_health(http_client, api_url):
    if api_url:
        return await _api_client.fetch_server_health(
            http_client,
            _api_client.normalize_base_url(api_url),
        )

    local_server, started_now = _gradio_local_api_server.ensure_started()
    if started_now:
        logger.info(f"Started local mineru-api at {local_server.base_url}")
    return await _api_client.wait_for_local_api_ready(http_client, local_server)


async def ensure_local_api_ready_for_gradio_startup(
    timeout_seconds: float = _api_client.LOCAL_API_STARTUP_TIMEOUT_SECONDS,
):
    local_server, started_now = _gradio_local_api_server.ensure_started()
    if started_now:
        logger.info(f"Started local mineru-api at {local_server.base_url}")

    async with httpx.AsyncClient(
        timeout=_api_client.build_http_timeout(),
        follow_redirects=True,
    ) as http_client:
        return await _api_client.wait_for_local_api_ready(
            http_client,
            local_server,
            timeout_seconds=timeout_seconds,
        )


def maybe_prepare_local_api_for_gradio_startup(
    *,
    api_url: str | None,
    enable_vlm_preload: bool,
):
    if api_url is not None or not enable_vlm_preload:
        return None

    try:
        return asyncio.run(ensure_local_api_ready_for_gradio_startup())
    except Exception:
        _gradio_local_api_server.stop()
        raise


def resolve_gradio_max_concurrent_requests(api_url, server_health):
    if api_url is None:
        return server_health.max_concurrent_requests

    return _api_client.resolve_effective_max_concurrent_requests(
        local_max=_api_client.read_max_concurrent_requests(
            default=_api_client.DEFAULT_MAX_CONCURRENT_REQUESTS
        ),
        server_max=server_health.max_concurrent_requests,
    )


def maybe_generate_local_preview(extract_root, file_name, file_suffix, backend, parse_method):
    if file_suffix in office_suffixes:
        return None

    parse_dir = resolve_parse_dir(
        extract_root,
        file_name,
        backend,
        parse_method,
        allow_office_fallback=True,
    )
    visualization_job = VisualizationJob(
        document_stem=file_name,
        backend=backend,
        parse_method=parse_method,
        parse_dir=parse_dir,
        draw_span=backend.startswith("pipeline"),
    )
    result = run_visualization_job(visualization_job)
    if result.status != "finished":
        logger.warning(
            f"Skipping visualization for {visualization_job.document_stem}: {result.message}"
        )
    return resolve_preview_pdf_path(parse_dir, file_name)


async def _run_to_markdown_job(
    file_path,
    end_pages=10,
    is_ocr=False,
    formula_enable=True,
    table_enable=True,
    language="ch",
    backend="pipeline",
    url=None,
    api_url=None,
    status_callback: Callable[[str], None] | None = None,
):
    if file_path is None:
        return "", "", None, None

    def emit_status(message: str) -> None:
        if status_callback is not None:
            status_callback(message)

    normalized_language = normalize_language(language)
    file_path = str(file_path)
    file_suffix = Path(file_path).suffix.lower().lstrip('.')
    parse_method = resolve_parse_method(file_path, is_ocr, backend)
    run_root, extract_root, archive_zip_path = create_gradio_run_paths(file_path)
    run_root.mkdir(parents=True, exist_ok=True)

    form_data = _api_client.build_parse_request_form_data(
        lang_list=[normalized_language],
        backend=backend,
        parse_method=parse_method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        server_url=url,
        start_page_id=0,
        end_page_id=end_pages - 1,
        return_md=True,
        return_middle_json=True,
        return_model_output=True,
        return_content_list=True,
        return_images=True,
        response_format_zip=True,
        return_original_file=True,
    )
    upload_assets = [
        _api_client.UploadAsset(
            path=Path(file_path),
            upload_name=build_gradio_upload_name(file_path),
        )
    ]

    async with httpx.AsyncClient(
        timeout=_api_client.build_http_timeout(),
        follow_redirects=True,
    ) as http_client:
        emit_status(STATUS_PREPARING_REQUEST)
        emit_status(STATUS_CHECKING_SERVER)
        server_health = await resolve_server_health(http_client, api_url)
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
            submit_response = await _api_client.submit_parse_task(
                base_url=server_health.base_url,
                upload_assets=upload_assets,
                form_data=form_data,
            )
            emit_status(f"Task submitted：task_id={submit_response.task_id}")

            last_task_snapshot = None

            def handle_task_status(
                status_snapshot: _api_client.TaskStatusSnapshot,
            ) -> None:
                nonlocal last_task_snapshot
                if status_snapshot == last_task_snapshot:
                    return
                last_task_snapshot = status_snapshot
                emit_status(format_remote_status_message(status_snapshot))

            await _api_client.wait_for_task_result(
                client=http_client,
                submit_response=submit_response,
                task_label=Path(file_path).name,
                status_snapshot_callback=handle_task_status,
            )
            emit_status(STATUS_DOWNLOADING_RESULT)
            result_zip_path = await _api_client.download_result_zip(
                client=http_client,
                submit_response=submit_response,
                task_label=Path(file_path).name,
            )

    try:
        _api_client.safe_extract_zip(result_zip_path, extract_root)
    finally:
        result_zip_path.unlink(missing_ok=True)

    file_name = resolve_result_file_name(submit_response, extract_root, file_path)
    local_md_dir = resolve_parse_dir(
        extract_root,
        file_name,
        backend,
        parse_method,
        allow_office_fallback=True,
    )
    preview_pdf_path = maybe_generate_local_preview(
        extract_root=extract_root,
        file_name=file_name,
        file_suffix=file_suffix,
        backend=backend,
        parse_method=parse_method,
    )

    emit_status(STATUS_PROCESSING_OUTPUT)
    zip_archive_success = compress_directory_to_zip(local_md_dir, archive_zip_path)
    if zip_archive_success == 0:
        logger.info('Compression successful')
    else:
        logger.error('Compression failed')

    md_path = Path(local_md_dir) / f"{file_name}.md"
    with open(md_path, 'r', encoding='utf-8') as f:
        txt_content = f.read()
    md_content = replace_image_with_base64(txt_content, local_md_dir)

    if file_suffix in office_suffixes:
        preview_pdf_path = None

    emit_status(STATUS_COMPLETED)
    return md_content, txt_content, str(archive_zip_path), preview_pdf_path


async def stream_to_markdown(
    file_path,
    end_pages=10,
    is_ocr=False,
    formula_enable=True,
    table_enable=True,
    language="ch",
    backend="pipeline",
    url=None,
    api_url=None,
):
    status_state = StatusPanelState()
    job_task: asyncio.Task | None = None
    queue_get_task: asyncio.Task | None = None
    timer_task: asyncio.Task | None = None
    yield status_state.render(), None, "", "", gr.skip()

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
                language=language,
                backend=backend,
                url=url,
                api_url=api_url,
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
                    yield status_state.render(), None, "", "", gr.skip()
            elif timer_task is not None and timer_task in done:
                if status_state.tick():
                    yield status_state.render(), None, "", "", gr.skip()
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
        yield status_state.render(), None, "", "", gr.skip()
        raise
    finally:
        for task in (queue_get_task, timer_task, job_task):
            if task is None or task.done():
                continue
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    try:
        md_content, txt_content, archive_zip_path, preview_pdf_path = await job_task
    except Exception as exc:
        status_state.append(format_failed_status(exc))
        yield status_state.render(), None, "", "", gr.skip()
        raise

    status_state.append(STATUS_COMPLETED)
    yield (
        status_state.render(),
        archive_zip_path,
        md_content,
        txt_content,
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

header_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'resources', 'header.html')
with open(header_path, mode='r', encoding='utf-8') as header_file:
    header = header_file.read()

other_lang = [
    'ch (Chinese, English, Chinese Traditional)',
    'ch_lite (Chinese, English, Chinese Traditional, Japanese)',
    'ch_server (Chinese, English, Chinese Traditional, Japanese)',
    'en (English)',
    'korean (Korean, English)',
    'japan (Chinese, English, Chinese Traditional, Japanese)',
    'chinese_cht (Chinese, English, Chinese Traditional, Japanese)',
    'ta (Tamil, English)',
    'te (Telugu, English)',
    'ka (Kannada)',
    'el (Greek, English)',
    'th (Thai, English)'
]
add_lang = [
    'latin (French, German, Afrikaans, Italian, Spanish, Bosnian, Portuguese, Czech, Welsh, Danish, Estonian, Irish, Croatian, Uzbek, Hungarian, Serbian (Latin), Indonesian, Occitan, Icelandic, Lithuanian, Maori, Malay, Dutch, Norwegian, Polish, Slovak, Slovenian, Albanian, Swedish, Swahili, Tagalog, Turkish, Latin, Azerbaijani, Kurdish, Latvian, Maltese, Pali, Romanian, Vietnamese, Finnish, Basque, Galician, Luxembourgish, Romansh, Catalan, Quechua)',
    'arabic (Arabic, Persian, Uyghur, Urdu, Pashto, Kurdish, Sindhi, Balochi, English)',
    'east_slavic (Russian, Belarusian, Ukrainian, English)',
    'cyrillic (Russian, Belarusian, Ukrainian, Serbian (Cyrillic), Bulgarian, Mongolian, Abkhazian, Adyghe, Kabardian, Avar, Dargin, Ingush, Chechen, Lak, Lezgin, Tabasaran, Kazakh, Kyrgyz, Tajik, Macedonian, Tatar, Chuvash, Bashkir, Malian, Moldovan, Udmurt, Komi, Ossetian, Buryat, Kalmyk, Tuvan, Sakha, Karakalpak, English)',
    'devanagari (Hindi, Marathi, Nepali, Bihari, Maithili, Angika, Bhojpuri, Magahi, Santali, Newari, Konkani, Sanskrit, Haryanvi, English)'
]
all_lang = [*other_lang, *add_lang]


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


def update_file_options_html(file_path, request: gr.Request):
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
        # 构建可公开访问的文件 URL，供 Microsoft 在线预览使用
        host = (request.headers.get('x-forwarded-host')
                or request.headers.get('host', 'localhost:7860'))
        proto = request.headers.get('x-forwarded-proto', 'http')
        base_url = f"{proto}://{host}"
        public_url = f"{base_url}/gradio_api/file={file_path}"
        viewer_url = f"https://view.officeapps.live.com/op/embed.aspx?src={public_url}"
        html_content = (
            f'<iframe src="{viewer_url}" '
            f'width="100%" height="960px" frameborder="0" '
            f'style="border: none;"></iframe>'
        )
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
         "The example files to be input need to be placed in the `example` folder within the directory where the command is currently executed.",
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
        server_name, server_port, api_url, enable_vlm_preload, latex_delimiters_type, **kwargs
):

    # 创建 i18n 实例，支持中英文
    i18n = gr.I18n(
        en={
            "upload_file": "Please select a file to upload (PDF, image, DOCX, PPTX, or XLSX)",
            "max_pages": "Max convert pages",
            "backend": "Backend",
            "server_url": "Server URL",
            "server_url_info": "OpenAI-compatible server URL for http-client backend.",
            "recognition_options": "**Recognition Options:**",
            "table_enable": "Enable table recognition",
            "table_info": "If disabled, tables will be shown as images.",
            "formula_label_vlm": "Enable display formula recognition",
            "formula_label_pipeline": "Enable formula recognition",
            "formula_label_hybrid": "Enable inline formula recognition",
            "formula_info_vlm": "If disabled, display formulas will be shown as images.",
            "formula_info_pipeline": "If disabled, display formulas will be shown as images, and inline formulas will not be detected or parsed.",
            "formula_info_hybrid": "If disabled, inline formulas will not be detected or parsed.",
            "ocr_language": "OCR Language",
            "ocr_language_info": "Select the OCR language for image-based PDFs and images.",
            "force_ocr": "Force enable OCR",
            "force_ocr_info": "Enable only if the result is extremely poor. Requires correct OCR language.",
            "convert": "Convert",
            "clear": "Clear",
            "doc_preview": "Document preview",
            "examples": "Examples:",
            "convert_status": "Conversion Status",
            "convert_result": "Convert result",
            "md_rendering": "Markdown rendering",
            "md_text": "Markdown text",
            "backend_info_vlm": "High-precision parsing via VLM, supports Chinese and English documents only.",
            "backend_info_pipeline": "Traditional Multi-model pipeline parsing, supports multiple languages, hallucination-free.",
            "backend_info_hybrid": "High-precision hybrid parsing, supports multiple languages.",
            "backend_info_default": "Select the backend engine for document parsing.",
        },
        zh={
            "upload_file": "请选择要上传的文件（PDF、图片、DOCX、PPTX 或 XLSX）",
            "max_pages": "最大转换页数",
            "backend": "解析后端",
            "server_url": "服务器地址",
            "server_url_info": "http-client 后端的 OpenAI 兼容服务器地址。",
            "recognition_options": "**识别选项：**",
            "table_enable": "启用表格识别",
            "table_info": "禁用后，表格将显示为图片。",
            "formula_label_vlm": "启用行间公式识别",
            "formula_label_pipeline": "启用公式识别",
            "formula_label_hybrid": "启用行内公式识别",
            "formula_info_vlm": "禁用后，行间公式将显示为图片。",
            "formula_info_pipeline": "禁用后，行间公式将显示为图片，行内公式将不会被检测或解析。",
            "formula_info_hybrid": "禁用后，行内公式将不会被检测或解析。",
            "ocr_language": "OCR 语言",
            "ocr_language_info": "为扫描版 PDF 和图片选择 OCR 语言。",
            "force_ocr": "强制启用 OCR",
            "force_ocr_info": "仅在识别效果极差时启用，需选择正确的 OCR 语言。",
            "convert": "转换",
            "clear": "清除",
            "doc_preview": "文档预览",
            "examples": "示例：",
            "convert_status": "转换状态",
            "convert_result": "转换结果",
            "md_rendering": "Markdown 渲染",
            "md_text": "Markdown 文本",
            "backend_info_vlm": "多模态大模型高精度解析，仅支持中英文文档。",
            "backend_info_pipeline": "传统多模型管道解析，支持多语言，无幻觉。",
            "backend_info_hybrid": "高精度混合解析，支持多语言。",
            "backend_info_default": "选择文档解析的后端引擎。",
        },
    )

    # 根据后端类型获取公式识别标签（闭包函数以支持 i18n）
    def get_formula_label(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("formula_label_vlm")
        elif backend_choice == "pipeline":
            return i18n("formula_label_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("formula_label_hybrid")
        else:
            return i18n("formula_label_pipeline")

    def get_formula_info(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("formula_info_vlm")
        elif backend_choice == "pipeline":
            return i18n("formula_info_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("formula_info_hybrid")
        else:
            return ""

    def get_backend_info(backend_choice):
        if backend_choice.startswith("vlm"):
            return i18n("backend_info_vlm")
        elif backend_choice == "pipeline":
            return i18n("backend_info_pipeline")
        elif backend_choice.startswith("hybrid"):
            return i18n("backend_info_hybrid")
        else:
            return i18n("backend_info_default")

    # 更新界面函数
    def update_interface(backend_choice):
        formula_label_update = gr.update(label=get_formula_label(backend_choice), info=get_formula_info(backend_choice))
        backend_info_update = gr.update(info=get_backend_info(backend_choice))
        if "http-client" in backend_choice:
            client_options_update = gr.update(visible=True)
        else:
            client_options_update = gr.update(visible=False)
        if "vlm" in backend_choice:
            ocr_options_update = gr.update(visible=False)
        else:
            ocr_options_update = gr.update(visible=True)

        return client_options_update, ocr_options_update, formula_label_update, backend_info_update


    del kwargs
    _gradio_local_api_server.configure(
        resolve_gradio_local_api_cli_args(
            ctx.args,
            api_url=api_url,
            enable_vlm_preload=enable_vlm_preload,
        )
    )

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
        language="ch",
        backend="pipeline",
        url=None,
    ):
        async for update in stream_to_markdown(
            file_path=file_path,
            end_pages=end_pages,
            is_ocr=is_ocr,
            formula_enable=formula_enable,
            table_enable=table_enable,
            language=language,
            backend=backend,
            url=url,
            api_url=api_url,
        ):
            yield update

    suffixes = [f".{suffix}" for suffix in pdf_suffixes + image_suffixes + office_suffixes]
    with gr.Blocks() as demo:
        gr.HTML(header)
        with gr.Row():
            with gr.Column(variant='panel', scale=5):
                with gr.Row():
                    input_file = gr.File(label=i18n("upload_file"), file_types=suffixes)
                # 下面这些选项在上传 office 文件时会被自动隐藏
                with gr.Group() as options_group:
                    with gr.Row():
                        max_pages = gr.Slider(1, max_convert_pages, max_convert_pages, step=1, label=i18n("max_pages"))
                    with gr.Row():
                        drop_list = ["pipeline", "vlm-auto-engine", "hybrid-auto-engine"]
                        preferred_option = "hybrid-auto-engine"
                        if http_client_enable:
                            drop_list.extend(["vlm-http-client", "hybrid-http-client"])
                        backend = gr.Dropdown(drop_list, label=i18n("backend"), value=preferred_option, info=get_backend_info(preferred_option))
                    with gr.Row(visible=False) as client_options:
                        url = gr.Textbox(label=i18n("server_url"), value='http://localhost:30000', placeholder='http://localhost:30000', info=i18n("server_url_info"))
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            gr.Markdown(i18n("recognition_options"))
                            table_enable = gr.Checkbox(label=i18n("table_enable"), value=True, info=i18n("table_info"))
                            formula_enable = gr.Checkbox(label=get_formula_label(preferred_option), value=True, info=get_formula_info(preferred_option))
                        with gr.Column() as ocr_options:
                            language = gr.Dropdown(all_lang, label=i18n("ocr_language"), value='ch (Chinese, English, Chinese Traditional)', info=i18n("ocr_language_info"))
                            is_ocr = gr.Checkbox(label=i18n("force_ocr"), value=False, info=i18n("force_ocr_info"))
                with gr.Row():
                    change_bu = gr.Button(i18n("convert"))
                    clear_bu = gr.ClearButton(value=i18n("clear"))
                _doc_preview_label = "doc preview" if IS_GRADIO_6 else i18n("doc_preview")
                doc_show = PDF(label=_doc_preview_label, interactive=False, visible=True, height=800)
                office_html = gr.HTML(value="", visible=False)
                if example_enable:
                    example_root = os.path.join(os.getcwd(), 'examples')
                    if os.path.exists(example_root):
                        gr.Examples(
                            label=i18n("examples"),
                            examples=[os.path.join(example_root, _) for _ in os.listdir(example_root) if
                                      _.endswith(tuple(suffixes))],
                            inputs=input_file
                        )

            with gr.Column(variant='panel', scale=5):
                status_box = gr.TextArea(
                    label=i18n("convert_status"),
                    value="",
                    lines=4,
                    max_lines=4,
                    interactive=False,
                    autoscroll=True,
                    elem_classes=["convert-status-box"],
                )
                output_file = gr.File(label=i18n("convert_result"), interactive=False)
                with gr.Blocks():
                    with gr.Tab(i18n("md_rendering")):
                        _md_copy_kwargs = {"buttons": ["copy"]} if IS_GRADIO_6 else {"show_copy_button": True}
                        md = gr.Markdown(
                            label=i18n("md_rendering"),
                            height=1200,
                            latex_delimiters=latex_delimiters,
                            line_breaks=True,
                            **_md_copy_kwargs
                        )
                    with gr.Tab(i18n("md_text")):
                        _textarea_copy_kwargs = {"buttons": ["copy"]} if IS_GRADIO_6 else {"show_copy_button": True}
                        md_text = gr.TextArea(
                            lines=45,
                            label=i18n("md_text"),
                            **_textarea_copy_kwargs
                        )

        # 添加事件处理
        _private_api_kwargs = (
            {"api_visibility": "private", "queue": False}
            if IS_GRADIO_6
            else {"api_name": False, "queue": False}
        )
        backend.change(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, formula_enable, backend],
            **_private_api_kwargs
        )
        # 添加demo.load事件，在页面加载时触发一次界面更新
        demo.load(
            fn=update_interface,
            inputs=[backend],
            outputs=[client_options, ocr_options, formula_enable, backend],
            **_private_api_kwargs
        )
        status_box.change(
            fn=None,
            inputs=[status_box],
            outputs=[],
            js=STATUS_BOX_AUTOSCROLL_JS,
            **_private_api_kwargs
        )
        clear_bu.add([input_file, md, doc_show, md_text, output_file, is_ocr, office_html, status_box])

        # 清除按钮额外重置 UI 可见性（ClearButton 不一定触发 input_file.change）
        clear_bu.click(
            fn=lambda: (
                gr.update(visible=True),
                gr.update(value=None, visible=True),
                gr.update(value="", visible=False),
                gr.update(value=""),
            ),
            inputs=[],
            outputs=[options_group, doc_show, office_html, status_box],
            **_private_api_kwargs
        )

        # 第一阶段：快速更新 options_group 和 office_html，不涉及 gradio_pdf 组件
        # 第二阶段（.then）：单独更新 doc_show，使 office_html 的 processing 遮罩
        # 在第一阶段完成后立即消失，规避 gradio_pdf 0.0.24 与 Gradio 6 的兼容性问题。
        input_file.change(
            fn=update_file_options_html,
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
            inputs=[input_file, max_pages, is_ocr, formula_enable, table_enable, language, backend, url],
            outputs=[status_box, output_file, md, md_text, doc_show],
            **_to_md_api_kwargs
        )

    demo.queue(default_concurrency_limit=None)

    if IS_GRADIO_6:
        footer_links = ["gradio", "settings"]
        if api_enable:
            footer_links.append("api")
        _launch_kwargs = {"footer_links": footer_links}
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
        **_launch_kwargs,
    )


if __name__ == '__main__':
    main()
