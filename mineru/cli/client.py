# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import os
import sys
import threading
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, Optional, TextIO

import click
import httpx
import pypdfium2 as pdfium
from loguru import logger

from mineru.cli.api_protocol import (
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_PROCESSING_WINDOW_SIZE,
)
from mineru.utils.config_reader import (
    get_max_concurrent_requests as read_max_concurrent_requests,
)
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.utils.pdf_page_id import get_end_page_id
from mineru.utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)

from mineru.version import __version__
from mineru.cli.common import (
    HybridDependencyError,
    ensure_backend_dependencies,
    image_suffixes,
    office_suffixes,
    pdf_suffixes,
    uniquify_task_stems,
)
from mineru.cli import api_client as _api_client
from mineru.cli.output_paths import resolve_parse_dir
from mineru.cli.visualization import (
    VisualizationJob,
    run_visualization_job,
)

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()

@dataclass(frozen=True)
class InputDocument:
    path: Path
    suffix: str
    stem: str
    effective_pages: int
    order: int


@dataclass
class PlannedTask:
    index: int
    documents: list[InputDocument]
    total_pages: int


@dataclass
class TaskExecutionProgress:
    total_tasks: int
    total_pages: int
    completed_tasks: int
    completed_pages: int
    lock: asyncio.Lock


@dataclass
class VisualizationContext:
    executor: ProcessPoolExecutor
    futures: list[Future]


ServerHealth = _api_client.ServerHealth
SubmitResponse = _api_client.SubmitResponse
LocalAPIServer = _api_client.LocalAPIServer


@dataclass(frozen=True)
class TaskFailure:
    task_index: int
    document_stems: tuple[str, ...]
    message: str


@dataclass
class LiveTaskStatusState:
    task_index: int
    task_id: str
    status: str
    queued_ahead: int | None = None
    frame_step: int = 0


class LiveAwareStderrSink:
    def __init__(self, stream: TextIO):
        self.stream = stream
        self.lock = threading.RLock()
        self.renderer: LiveTaskStatusRenderer | None = None

    def set_renderer(self, renderer: "LiveTaskStatusRenderer | None") -> None:
        with self.lock:
            self.renderer = renderer

    def isatty(self) -> bool:
        return bool(getattr(self.stream, "isatty", lambda: False)())

    def write(self, message: str) -> None:
        with self.lock:
            renderer = self.renderer
            if renderer is not None:
                renderer.clear_locked()
            self.stream.write(message)
            self.stream.flush()
            if renderer is not None:
                renderer.render_locked()

    def flush(self) -> None:
        self.stream.flush()

    def stop(self) -> None:
        self.flush()


class LiveTaskStatusRenderer:
    BAR_WIDTH = 12
    RUNNER_WIDTH = 4
    ACTIVE_STATUSES = {"pending", "processing"}

    def __init__(self, sink: LiveAwareStderrSink):
        self.sink = sink
        self._rendered_line_count = 0
        self._task_states: dict[str, LiveTaskStatusState] = {}

    def register_task(
        self,
        task: PlannedTask,
        task_id: str,
        queued_ahead: int | None = None,
    ) -> None:
        with self.sink.lock:
            self._task_states[task_id] = LiveTaskStatusState(
                task_index=task.index,
                task_id=task_id,
                status="pending",
                queued_ahead=queued_ahead,
            )
            self.render_locked()

    def update_status(
        self,
        task_id: str,
        status_update: _api_client.TaskStatusSnapshot | str,
    ) -> None:
        with self.sink.lock:
            state = self._task_states.get(task_id)
            if state is None:
                return
            if isinstance(status_update, _api_client.TaskStatusSnapshot):
                status = status_update.status
                queued_ahead = (
                    status_update.queued_ahead if status == "pending" else None
                )
            else:
                status = status_update
                queued_ahead = None

            state.status = status
            state.queued_ahead = queued_ahead
            if status in self.ACTIVE_STATUSES:
                state.frame_step += 1
            self.render_locked()

    def remove_task(self, task_id: str) -> None:
        with self.sink.lock:
            if self._task_states.pop(task_id, None) is None:
                return
            self.render_locked()

    def close(self) -> None:
        with self.sink.lock:
            self._task_states.clear()
            self.clear_locked()

    def clear_locked(self) -> None:
        if self._rendered_line_count <= 0:
            return

        self.sink.stream.write(f"\x1b[{self._rendered_line_count}A\r")
        for index in range(self._rendered_line_count):
            self.sink.stream.write("\x1b[2K")
            if index + 1 < self._rendered_line_count:
                self.sink.stream.write("\x1b[1B\r")
        if self._rendered_line_count > 1:
            self.sink.stream.write(f"\x1b[{self._rendered_line_count - 1}A\r")
        self.sink.stream.flush()
        self._rendered_line_count = 0

    def render_locked(self) -> None:
        self.clear_locked()
        lines = self._build_render_lines_locked()
        if lines:
            self.sink.stream.write("\n".join(lines))
            self.sink.stream.write("\n")
            self.sink.stream.flush()
        self._rendered_line_count = len(lines)

    def _build_render_lines_locked(self) -> list[str]:
        states = sorted(
            self._task_states.values(),
            key=lambda state: (state.task_index, state.task_id),
        )
        return [
            self._build_render_line_locked(state)
            for state in states
        ]

    @staticmethod
    def _build_render_line_locked(state: LiveTaskStatusState) -> str:
        parts = [f"status={state.status}"]
        if state.status == "pending" and state.queued_ahead is not None:
            parts.append(f"ahead={state.queued_ahead}")
        parts.append(f"task_id={state.task_id}")
        return f"{LiveTaskStatusRenderer._build_bar(state.frame_step)} {' | '.join(parts)}"

    @classmethod
    def _build_bar(cls, frame_step: int) -> str:
        cells = [" "] * cls.BAR_WIDTH
        runner_start = frame_step % cls.BAR_WIDTH
        for offset in range(cls.RUNNER_WIDTH):
            position = (runner_start + offset) % cls.BAR_WIDTH
            cells[position] = "="
        head_position = (runner_start + cls.RUNNER_WIDTH - 1) % cls.BAR_WIDTH
        cells[head_position] = ">"
        return f"[{''.join(cells)}]"


def create_live_task_status_renderer(
    api_url: Optional[str],
) -> Optional[LiveTaskStatusRenderer]:
    if api_url is None or not _stderr_sink.isatty():
        _stderr_sink.set_renderer(None)
        return None

    renderer = LiveTaskStatusRenderer(_stderr_sink)
    _stderr_sink.set_renderer(renderer)
    return renderer


_stderr_sink = LiveAwareStderrSink(sys.stderr)
logger.remove()
logger.add(_stderr_sink, level=log_level)


def build_http_timeout() -> httpx.Timeout:
    return _api_client.build_http_timeout()

def find_free_port() -> int:
    return _api_client.find_free_port()


def normalize_base_url(url: str) -> str:
    return _api_client.normalize_base_url(url)


def format_task_label(task: PlannedTask) -> str:
    doc_names = ", ".join(doc.stem for doc in task.documents)
    return f"task#{task.index} [{doc_names}]"


def format_task_log_label(task: PlannedTask, max_documents: int = 3) -> str:
    if len(task.documents) <= max_documents:
        return format_task_label(task)

    visible_names = ", ".join(doc.stem for doc in task.documents[:max_documents])
    hidden_count = len(task.documents) - max_documents
    return f"task#{task.index} [{visible_names}, +{hidden_count} more]"


def format_count(value: int, singular: str, plural: Optional[str] = None) -> str:
    unit = singular if value == 1 else (plural or f"{singular}s")
    return f"{value} {unit}"


def format_task_submission_message(
    task: PlannedTask,
    progress: TaskExecutionProgress,
) -> str:
    return (
        f"Submitting batch {task.index}/{progress.total_tasks} | "
        f"{format_count(len(task.documents), 'document')}, "
        f"{format_count(task.total_pages, 'page')} in this batch | "
        f"{format_count(progress.total_pages, 'page')} total | "
        f"{format_task_log_label(task)}"
    )


def format_task_completion_message(
    task: PlannedTask,
    progress: TaskExecutionProgress,
    completed_tasks: int,
    completed_pages: int,
) -> str:
    batch_word = "batch" if progress.total_tasks == 1 else "batches"
    page_word = "page" if progress.total_pages == 1 else "pages"
    return (
        f"Completed batch {task.index}/{progress.total_tasks} | "
        f"Processed {completed_pages}/{progress.total_pages} {page_word} | "
        f"{completed_tasks} of {progress.total_tasks} {batch_word} finished | "
        f"{format_task_log_label(task)}"
    )


def build_task_execution_progress(
    planned_tasks: list[PlannedTask],
) -> TaskExecutionProgress:
    return TaskExecutionProgress(
        total_tasks=len(planned_tasks),
        total_pages=sum(task.total_pages for task in planned_tasks),
        completed_tasks=0,
        completed_pages=0,
        lock=asyncio.Lock(),
    )


async def mark_task_completed(
    progress: TaskExecutionProgress,
    completed_pages: int,
) -> tuple[int, int]:
    async with progress.lock:
        progress.completed_tasks += 1
        progress.completed_pages += completed_pages
        return progress.completed_tasks, progress.completed_pages


def create_visualization_context() -> Optional[VisualizationContext]:
    try:
        return VisualizationContext(
            executor=ProcessPoolExecutor(max_workers=1),
            futures=[],
        )
    except Exception as exc:
        logger.warning(f"Failed to start visualization worker process: {exc}")
        return None


def build_visualization_jobs(
    planned_task: PlannedTask,
    output_dir: Path,
    backend: str,
    parse_method: str,
) -> list[VisualizationJob]:
    draw_span = backend.startswith("pipeline")
    return [
        VisualizationJob(
            document_stem=document.stem,
            backend=backend,
            parse_method=parse_method,
            parse_dir=resolve_parse_dir(
                output_dir,
                document.stem,
                backend,
                parse_method,
                is_office=document.suffix in office_suffixes,
            ),
            draw_span=draw_span,
        )
        for document in planned_task.documents
    ]


def log_visualization_future_result(
    job: VisualizationJob,
    future: Future,
) -> None:
    try:
        result = future.result()
    except Exception as exc:
        logger.warning(f"Skipping visualization for {job.document_stem}: {exc}")
        return

    if result.status != "finished":
        logger.warning(
            f"Skipping visualization for {result.document_stem}: {result.message}"
        )


def queue_visualization_jobs(
    visualization_context: Optional[VisualizationContext],
    jobs: list[VisualizationJob],
    planned_task: PlannedTask,
) -> int:
    if visualization_context is None or not jobs:
        return 0

    del planned_task
    queued_jobs = 0
    for job in jobs:
        try:
            future = visualization_context.executor.submit(run_visualization_job, job)
        except Exception as exc:
            logger.warning(f"Skipping visualization for {job.document_stem}: {exc}")
            continue

        future.add_done_callback(
            lambda completed_future, job=job: log_visualization_future_result(
                job,
                completed_future,
            )
        )
        visualization_context.futures.append(future)
        queued_jobs += 1

    return queued_jobs


async def wait_for_visualization_jobs(
    visualization_context: Optional[VisualizationContext],
) -> None:
    if visualization_context is None:
        return

    try:
        if visualization_context.futures:
            await asyncio.gather(
                *(
                    asyncio.wrap_future(future)
                    for future in visualization_context.futures
                ),
                return_exceptions=True,
            )
    finally:
        visualization_context.executor.shutdown(wait=True)


def response_detail(response: httpx.Response) -> str:
    return _api_client.response_detail(response)


def validate_server_health_payload(payload: dict, base_url: str) -> ServerHealth:
    return _api_client.validate_server_health_payload(payload, base_url)


async def fetch_server_health(
    client: httpx.AsyncClient,
    base_url: str,
) -> ServerHealth:
    return await _api_client.fetch_server_health(client, base_url)


async def wait_for_local_api_ready(
    client: httpx.AsyncClient,
    local_server: LocalAPIServer,
    timeout_seconds: float = _api_client.LOCAL_API_STARTUP_TIMEOUT_SECONDS,
) -> ServerHealth:
    return await _api_client.wait_for_local_api_ready(
        client,
        local_server,
        timeout_seconds=timeout_seconds,
    )


def probe_pdf_effective_pages(
    path: Path,
    start_page_id: int,
    end_page_id: Optional[int],
) -> int:
    pdf_doc = open_pdfium_document(pdfium.PdfDocument, str(path))
    try:
        page_count = get_pdfium_document_page_count(pdf_doc)
    finally:
        close_pdfium_document(pdf_doc)

    if page_count <= 0:
        raise click.ClickException(f"PDF has no pages: {path}")

    effective_end_page_id = get_end_page_id(end_page_id, page_count)
    if start_page_id > effective_end_page_id:
        raise click.ClickException(
            f"Requested page range is empty for PDF {path}: "
            f"start={start_page_id}, end={end_page_id}"
        )
    return effective_end_page_id - start_page_id + 1


def collect_input_documents(
    input_path: Path,
    start_page_id: int,
    end_page_id: Optional[int],
) -> list[InputDocument]:
    documents: list[Path]
    if input_path.is_dir():
        documents = [path for path in sorted(input_path.glob("*")) if path.is_file()]
    else:
        documents = [input_path]

    collected: list[InputDocument] = []
    for order, path in enumerate(documents):
        suffix = guess_suffix_by_path(path)
        if suffix not in pdf_suffixes + image_suffixes + office_suffixes:
            continue

        if suffix in pdf_suffixes:
            effective_pages = probe_pdf_effective_pages(
                path,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )
        else:
            effective_pages = 1

        collected.append(
            InputDocument(
                path=path,
                suffix=suffix,
                stem=path.stem,
                effective_pages=effective_pages,
                order=order,
            )
        )

    if not collected:
        raise click.ClickException(f"No supported documents found under {input_path}")

    normalized_stems, renamed_stems = uniquify_task_stems(
        [document.stem for document in collected]
    )
    if renamed_stems:
        rename_details = ", ".join(
            f"{document.path.name} -> {effective_stem}"
            for document, effective_stem in zip(collected, normalized_stems)
            if document.stem != effective_stem
        )
        logger.warning(
            f"Normalized duplicate document stems within this run: {rename_details}"
        )
        return [
            InputDocument(
                path=document.path,
                suffix=document.suffix,
                stem=effective_stem,
                effective_pages=document.effective_pages,
                order=document.order,
            )
            for document, effective_stem in zip(collected, normalized_stems)
        ]

    return collected


def plan_pipeline_tasks(
    documents: list[InputDocument],
    processing_window_size: int,
) -> list[PlannedTask]:
    bins: list[PlannedTask] = []
    sorted_docs = sorted(
        documents,
        key=lambda doc: (-doc.effective_pages, doc.order),
    )

    for document in sorted_docs:
        if document.effective_pages > processing_window_size:
            bins.append(
                PlannedTask(
                    index=len(bins) + 1,
                    documents=[document],
                    total_pages=document.effective_pages,
                )
            )
            continue

        candidates = [
            task
            for task in bins
            if task.total_pages + document.effective_pages <= processing_window_size
        ]
        if candidates:
            selected = min(candidates, key=lambda task: (task.total_pages, task.index))
            selected.documents.append(document)
            selected.total_pages += document.effective_pages
            continue

        bins.append(
            PlannedTask(
                index=len(bins) + 1,
                documents=[document],
                total_pages=document.effective_pages,
            )
        )

    for index, task in enumerate(bins, start=1):
        task.index = index
    return bins


def plan_tasks(
    documents: list[InputDocument],
    backend: str,
    processing_window_size: int,
) -> list[PlannedTask]:
    if backend == "pipeline":
        return plan_pipeline_tasks(documents, processing_window_size)
    return [
        PlannedTask(index=index, documents=[document], total_pages=document.effective_pages)
        for index, document in enumerate(documents, start=1)
    ]


def build_request_form_data(
    lang: str,
    backend: str,
    method: str,
    formula_enable: bool,
    table_enable: bool,
    server_url: Optional[str],
    start_page_id: int,
    end_page_id: Optional[int],
) -> dict[str, str | list[str]]:
    return _api_client.build_parse_request_form_data(
        lang_list=[lang],
        backend=backend,
        parse_method=method,
        formula_enable=formula_enable,
        table_enable=table_enable,
        server_url=server_url,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
        return_md=True,
        return_middle_json=True,
        return_model_output=True,
        return_content_list=True,
        return_images=True,
        response_format_zip=True,
        return_original_file=True,
    )


async def submit_task(
    client: httpx.AsyncClient,
    base_url: str,
    planned_task: PlannedTask,
    form_data: dict[str, str | list[str]],
) -> SubmitResponse:
    del client
    upload_assets = [
        _api_client.UploadAsset(
            path=document.path,
            upload_name=f"{document.stem}{document.path.suffix}",
        )
        for document in planned_task.documents
    ]
    return await _api_client.submit_parse_task(
        base_url=base_url,
        upload_assets=upload_assets,
        form_data=form_data,
    )


async def wait_for_task_result(
    client: httpx.AsyncClient,
    submit_response: SubmitResponse,
    planned_task: PlannedTask,
    live_renderer: Optional[LiveTaskStatusRenderer] = None,
    timeout_seconds: float = _api_client.TASK_RESULT_TIMEOUT_SECONDS,
) -> None:
    return await _api_client.wait_for_task_result(
        client=client,
        submit_response=submit_response,
        task_label=format_task_label(planned_task),
        status_snapshot_callback=(
            None
            if live_renderer is None
            else lambda snapshot: live_renderer.update_status(
                submit_response.task_id,
                snapshot,
            )
        ),
        timeout_seconds=timeout_seconds,
    )


async def download_result_zip(
    client: httpx.AsyncClient,
    submit_response: SubmitResponse,
    planned_task: PlannedTask,
) -> Path:
    return await _api_client.download_result_zip(
        client=client,
        submit_response=submit_response,
        task_label=format_task_label(planned_task),
    )


def safe_extract_zip(zip_path: Path, output_dir: Path) -> None:
    _api_client.safe_extract_zip(zip_path, output_dir)


def resolve_submit_concurrency(max_concurrent_requests: int, task_count: int) -> int:
    if max_concurrent_requests <= 0:
        raise ValueError("max_concurrent_requests must be a positive integer")
    return max(1, min(max_concurrent_requests, task_count))


def resolve_effective_max_concurrent_requests(
    local_max: int,
    server_max: int,
) -> int:
    return _api_client.resolve_effective_max_concurrent_requests(
        local_max=local_max,
        server_max=server_max,
    )


async def execute_planned_tasks(
    planned_tasks: list[PlannedTask],
    concurrency: int,
    task_runner: Callable[[PlannedTask], Awaitable[None]],
) -> list[TaskFailure]:
    queue: asyncio.Queue[PlannedTask | None] = asyncio.Queue()
    failures: list[TaskFailure] = []

    for task in planned_tasks:
        await queue.put(task)
    for _ in range(concurrency):
        await queue.put(None)

    async def worker() -> None:
        while True:
            planned_task = await queue.get()
            try:
                if planned_task is None:
                    return
                await task_runner(planned_task)
            except Exception as exc:
                assert planned_task is not None
                failures.append(
                    TaskFailure(
                        task_index=planned_task.index,
                        document_stems=tuple(doc.stem for doc in planned_task.documents),
                        message=str(exc),
                    )
                )
            finally:
                queue.task_done()

    workers = [asyncio.create_task(worker()) for _ in range(concurrency)]
    await queue.join()
    await asyncio.gather(*workers, return_exceptions=True)
    return failures


async def run_planned_task(
    client: httpx.AsyncClient,
    server_health: ServerHealth,
    planned_task: PlannedTask,
    progress: TaskExecutionProgress,
    backend: str,
    parse_method: str,
    visualization_context: Optional[VisualizationContext],
    form_data: dict[str, str],
    output_dir: Path,
    live_renderer: Optional[LiveTaskStatusRenderer] = None,
) -> None:
    logger.info(format_task_submission_message(planned_task, progress))
    submit_response = await submit_task(
        client=client,
        base_url=server_health.base_url,
        planned_task=planned_task,
        form_data=form_data,
    )
    if live_renderer is not None:
        live_renderer.register_task(
            planned_task,
            submit_response.task_id,
            queued_ahead=submit_response.queued_ahead,
        )
    try:
        await wait_for_task_result(
            client=client,
            submit_response=submit_response,
            planned_task=planned_task,
            live_renderer=live_renderer,
        )
    finally:
        if live_renderer is not None:
            live_renderer.remove_task(submit_response.task_id)
    zip_path = await download_result_zip(
        client=client,
        submit_response=submit_response,
        planned_task=planned_task,
    )
    try:
        safe_extract_zip(zip_path, output_dir)
    finally:
        zip_path.unlink(missing_ok=True)
    completed_tasks, completed_pages = await mark_task_completed(
        progress,
        planned_task.total_pages,
    )
    logger.info(
        format_task_completion_message(
            planned_task,
            progress,
            completed_tasks,
            completed_pages,
        )
    )
    try:
        visualization_jobs = build_visualization_jobs(
            planned_task,
            output_dir,
            backend,
            parse_method,
        )
    except Exception as exc:
        logger.warning(
            f"Skipping visualization for {format_task_log_label(planned_task)}: {exc}"
        )
    else:
        queue_visualization_jobs(
            visualization_context,
            visualization_jobs,
            planned_task,
        )


async def run_orchestrated_cli(
    input_path: Path,
    output_dir: Path,
    method: str,
    backend: str,
    lang: str,
    server_url: Optional[str],
    api_url: Optional[str],
    start_page_id: int,
    end_page_id: Optional[int],
    formula_enable: bool,
    table_enable: bool,
    extra_cli_args: tuple[str, ...] = (),
) -> None:
    if start_page_id < 0:
        raise click.ClickException("--start must be greater than or equal to 0")
    if end_page_id is not None and end_page_id < 0:
        raise click.ClickException("--end must be greater than or equal to 0")
    if api_url is None:
        try:
            ensure_backend_dependencies(backend)
        except HybridDependencyError as exc:
            raise click.ClickException(str(exc)) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    documents = collect_input_documents(
        input_path=input_path,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )

    timeout = build_http_timeout()
    local_server: LocalAPIServer | None = None
    visualization_context: Optional[VisualizationContext] = None
    live_renderer: Optional[LiveTaskStatusRenderer] = None
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http_client:
        try:
            if api_url is None:
                local_server = LocalAPIServer(extra_cli_args=extra_cli_args)
                base_url = local_server.start()
                logger.info(f"Started local mineru-api at {base_url}")
                server_health = await wait_for_local_api_ready(http_client, local_server)
                effective_max_concurrent_requests = (
                    server_health.max_concurrent_requests
                )
            else:
                server_health = await fetch_server_health(
                    http_client,
                    normalize_base_url(api_url),
                )
                effective_max_concurrent_requests = (
                    resolve_effective_max_concurrent_requests(
                        read_max_concurrent_requests(
                            default=DEFAULT_MAX_CONCURRENT_REQUESTS
                        ),
                        server_health.max_concurrent_requests,
                    )
                )
                live_renderer = create_live_task_status_renderer(api_url)

            planned_tasks = plan_tasks(
                documents=documents,
                backend=backend,
                processing_window_size=server_health.processing_window_size
                if backend == "pipeline"
                else DEFAULT_PROCESSING_WINDOW_SIZE,
            )
            progress = build_task_execution_progress(planned_tasks)
            concurrency = resolve_submit_concurrency(
                effective_max_concurrent_requests,
                len(planned_tasks),
            )
            form_data = build_request_form_data(
                lang=lang,
                backend=backend,
                method=method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                server_url=server_url,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
            )
            visualization_context = create_visualization_context()
            failures = await execute_planned_tasks(
                planned_tasks=planned_tasks,
                concurrency=concurrency,
                task_runner=lambda planned_task: run_planned_task(
                    client=http_client,
                    server_health=server_health,
                    planned_task=planned_task,
                    progress=progress,
                    backend=backend,
                    parse_method=method,
                    visualization_context=visualization_context,
                    form_data=form_data,
                    output_dir=output_dir,
                    live_renderer=live_renderer,
                ),
            )
            if failures:
                details = "\n".join(
                    f"- task#{failure.task_index} ({', '.join(failure.document_stems)}): {failure.message}"
                    for failure in sorted(failures, key=lambda item: item.task_index)
                )
                raise click.ClickException(
                    f"{len(failures)} task(s) failed while processing documents:\n{details}"
                )
        finally:
            try:
                if local_server is not None:
                    local_server.stop()
            finally:
                try:
                    await wait_for_visualization_jobs(visualization_context)
                finally:
                    if live_renderer is not None:
                        live_renderer.close()
                        _stderr_sink.set_renderer(None)


@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
@click.pass_context
@click.version_option(__version__, "--version", "-v", help="display the version and exit")
@click.option(
    "-p",
    "--path",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="local filepath or directory. support pdf, image, docx, pptx, xlsx files",
)
@click.option(
    "-o",
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="output local directory",
)
@click.option(
    "--api-url",
    "api_url",
    type=str,
    default=None,
    help="MinerU FastAPI base URL. If omitted, mineru starts a temporary local mineru-api service.",
)
@click.option(
    "-m",
    "--method",
    "method",
    type=click.Choice(["auto", "txt", "ocr"]),
    default="auto",
    help="""\b
    the method for parsing pdf:
      auto: Automatically determine the method based on the file type.
      txt: Use text extraction method.
      ocr: Use OCR method for image-based PDFs.
    Without method specified, 'auto' will be used by default.
    Adapted only for the case where the backend is set to 'pipeline' and 'hybrid-*'.""",
)
@click.option(
    "-b",
    "--backend",
    "backend",
    type=click.Choice(
        [
            "pipeline",
            "vlm-http-client",
            "hybrid-http-client",
            "vlm-auto-engine",
            "hybrid-auto-engine",
        ]
    ),
    default="hybrid-auto-engine",
    help="""\b
    the backend for parsing pdf:
      pipeline: More general.
      vlm-auto-engine: High accuracy via local computing power.
      vlm-http-client: High accuracy via remote computing power(client suitable for openai-compatible servers).
      hybrid-auto-engine: Next-generation high accuracy solution via local computing power.
      hybrid-http-client: High accuracy but requires a little local computing power(client suitable for openai-compatible servers).
    Without method specified, hybrid-auto-engine will be used by default.""",
)
@click.option(
    "-l",
    "--lang",
    "lang",
    type=click.Choice(
        [
            "ch",
            "ch_server",
            "ch_lite",
            "en",
            "korean",
            "japan",
            "chinese_cht",
            "ta",
            "te",
            "ka",
            "th",
            "el",
            "latin",
            "arabic",
            "east_slavic",
            "cyrillic",
            "devanagari",
        ]
    ),
    default="ch",
    help="""
    Input the languages in the pdf (if known) to improve OCR accuracy.
    Without languages specified, 'ch' will be used by default.
    Adapted only for the case where the backend is set to 'pipeline' and 'hybrid-*'.
    """,
)
@click.option(
    "-u",
    "--url",
    "server_url",
    type=str,
    default=None,
    help="""
    When the backend is `<vlm/hybrid>-http-client`, you need to specify the server_url, for example:`http://127.0.0.1:30000`
    """,
)
@click.option(
    "-s",
    "--start",
    "start_page_id",
    type=int,
    default=0,
    help="The starting page for PDF parsing, beginning from 0.",
)
@click.option(
    "-e",
    "--end",
    "end_page_id",
    type=int,
    default=None,
    help="The ending page for PDF parsing, beginning from 0.",
)
@click.option(
    "-f",
    "--formula",
    "formula_enable",
    type=bool,
    default=True,
    help="Enable formula parsing. Default is True. ",
)
@click.option(
    "-t",
    "--table",
    "table_enable",
    type=bool,
    default=True,
    help="Enable table parsing. Default is True. ",
)
def main(
    ctx: click.Context,
    input_path: Path,
    output_dir: Path,
    api_url: Optional[str],
    method: str,
    backend: str,
    lang: str,
    server_url: Optional[str],
    start_page_id: int,
    end_page_id: Optional[int],
    formula_enable: bool,
    table_enable: bool,
) -> None:
    asyncio.run(
        run_orchestrated_cli(
            input_path=input_path,
            output_dir=output_dir,
            method=method,
            backend=backend,
            lang=lang,
            server_url=server_url,
            api_url=api_url,
            start_page_id=start_page_id,
            end_page_id=end_page_id,
            formula_enable=formula_enable,
            table_enable=table_enable,
            extra_cli_args=tuple(ctx.args),
        )
    )


if __name__ == "__main__":
    main()
