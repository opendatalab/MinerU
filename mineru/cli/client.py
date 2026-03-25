# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import atexit
import json
import mimetypes
import os
import socket
import subprocess
import sys
import tempfile
import time
import zipfile
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Awaitable, Callable, Optional

import click
import httpx
import pypdfium2 as pdfium
from loguru import logger

from mineru.cli.api_protocol import (
    API_PROTOCOL_VERSION,
    DEFAULT_PROCESSING_WINDOW_SIZE,
)
from mineru.utils.guess_suffix_or_lang import guess_suffix_by_path
from mineru.utils.pdf_page_id import get_end_page_id
from mineru.utils.pdfium_guard import (
    close_pdfium_document,
    get_pdfium_document_page_count,
    open_pdfium_document,
)

from ..version import __version__
from .common import image_suffixes, office_suffixes, pdf_suffixes

os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"
log_level = os.getenv("MINERU_LOG_LEVEL", "INFO").upper()
logger.remove()
logger.add(sys.stderr, level=log_level)

HEALTH_ENDPOINT = "/health"
TASKS_ENDPOINT = "/tasks"
TASK_STATUS_POLL_INTERVAL_SECONDS = 1.0
TASK_RESULT_TIMEOUT_SECONDS = 600
LOCAL_API_STARTUP_TIMEOUT_SECONDS = 30
LOCAL_API_CLEANUP_RETRIES = 8
LOCAL_API_CLEANUP_RETRY_INTERVAL_SECONDS = 0.25
LOCAL_API_MAX_CONCURRENT_REQUESTS = 3


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


@dataclass(frozen=True)
class TaskFailure:
    task_index: int
    document_stems: tuple[str, ...]
    message: str


def build_http_timeout() -> httpx.Timeout:
    return httpx.Timeout(connect=10, read=60, write=300, pool=30)


class LocalAPIServer:
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory(prefix="mineru-api-client-")
        self.temp_root = Path(self.temp_dir.name)
        self.output_root = self.temp_root / "output"
        self.base_url: str | None = None
        self.process: subprocess.Popen[bytes] | None = None
        self._atexit_registered = False

    def start(self) -> str:
        if self.process is not None:
            raise RuntimeError("Local API server is already running")

        port = find_free_port()
        self.base_url = f"http://127.0.0.1:{port}"
        env = os.environ.copy()
        env["MINERU_API_OUTPUT_ROOT"] = str(self.output_root)
        env["MINERU_API_MAX_CONCURRENT_REQUESTS"] = str(
            LOCAL_API_MAX_CONCURRENT_REQUESTS
        )
        env["MINERU_API_DISABLE_ACCESS_LOG"] = "1"
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mineru.cli.fast_api",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=os.getcwd(),
            env=env,
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
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
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

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def normalize_base_url(url: str) -> str:
    return url.rstrip("/")


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
    if not isinstance(max_concurrent_requests, int):
        raise click.ClickException(
            f"MinerU API at {base_url} did not return a valid max_concurrent_requests"
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

    stem_to_paths: dict[str, list[Path]] = {}
    for document in collected:
        stem_to_paths.setdefault(document.stem, []).append(document.path)
    duplicate_stems = {
        stem: paths for stem, paths in stem_to_paths.items() if len(paths) > 1
    }
    if duplicate_stems:
        details = "; ".join(
            f"{stem}: {', '.join(str(path) for path in paths)}"
            for stem, paths in sorted(duplicate_stems.items())
        )
        raise click.ClickException(
            f"Duplicate output stems detected. Rename inputs to avoid collisions: {details}"
        )

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
) -> dict[str, str]:
    data = {
        "lang_list": lang,
        "backend": backend,
        "parse_method": method,
        "formula_enable": str(formula_enable).lower(),
        "table_enable": str(table_enable).lower(),
        "return_md": "true",
        "return_middle_json": "true",
        "return_model_output": "true",
        "return_content_list": "true",
        "return_images": "true",
        "response_format_zip": "true",
        "return_original_file": "true",
        "start_page_id": str(start_page_id),
        "end_page_id": str(99999 if end_page_id is None else end_page_id),
    }
    if server_url:
        data["server_url"] = server_url
    return data


async def submit_task(
    client: httpx.AsyncClient,
    base_url: str,
    planned_task: PlannedTask,
    form_data: dict[str, str],
) -> SubmitResponse:
    return await asyncio.to_thread(
        submit_task_sync,
        base_url,
        planned_task,
        form_data,
    )


def submit_task_sync(
    base_url: str,
    planned_task: PlannedTask,
    form_data: dict[str, str],
) -> SubmitResponse:
    with httpx.Client(timeout=build_http_timeout(), follow_redirects=True) as sync_client:
        with ExitStack() as stack:
            files = []
            for document in planned_task.documents:
                mime_type = mimetypes.guess_type(document.path.name)[0] or "application/octet-stream"
                file_handle = stack.enter_context(open(document.path, "rb"))
                files.append(
                    (
                        "files",
                        (
                            document.path.name,
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
            f"Failed to submit {format_task_label(planned_task)}: "
            f"{response.status_code} {response_detail(response)}"
        )

    payload = response.json()
    task_id = payload.get("task_id")
    status_url = payload.get("status_url")
    result_url = payload.get("result_url")
    if (
        not isinstance(task_id, str)
        or not isinstance(status_url, str)
        or not isinstance(result_url, str)
    ):
        raise click.ClickException(
            f"MinerU API returned an invalid task payload for {format_task_label(planned_task)}"
        )
    return SubmitResponse(
        task_id=task_id,
        status_url=status_url,
        result_url=result_url,
    )


async def wait_for_task_result(
    client: httpx.AsyncClient,
    submit_response: SubmitResponse,
    planned_task: PlannedTask,
    timeout_seconds: float = TASK_RESULT_TIMEOUT_SECONDS,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_seconds
    while asyncio.get_running_loop().time() < deadline:
        response = await client.get(submit_response.status_url)
        if response.status_code != 200:
            raise click.ClickException(
                f"Failed to query task status for {format_task_label(planned_task)}: "
                f"{response.status_code} {response_detail(response)}"
            )

        payload = response.json()
        status = payload.get("status")
        if status in {"pending", "processing"}:
            await asyncio.sleep(TASK_STATUS_POLL_INTERVAL_SECONDS)
            continue
        if status == "completed":
            return
        raise click.ClickException(
            f"Task {submit_response.task_id} failed for {format_task_label(planned_task)}: "
            f"{json.dumps(payload, ensure_ascii=False)}"
        )

    raise click.ClickException(
        f"Timed out waiting for result of task {submit_response.task_id} "
        f"for {format_task_label(planned_task)}"
    )


async def download_result_zip(
    client: httpx.AsyncClient,
    submit_response: SubmitResponse,
    planned_task: PlannedTask,
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
            f"Expected a ZIP result for {format_task_label(planned_task)}, "
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


def resolve_submit_concurrency(max_concurrent_requests: int, task_count: int) -> int:
    if max_concurrent_requests <= 0:
        return max(1, task_count)
    return max(1, min(max_concurrent_requests, task_count))


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
    form_data: dict[str, str],
    output_dir: Path,
) -> None:
    logger.info(format_task_submission_message(planned_task, progress))
    submit_response = await submit_task(
        client=client,
        base_url=server_health.base_url,
        planned_task=planned_task,
        form_data=form_data,
    )
    await wait_for_task_result(
        client=client,
        submit_response=submit_response,
        planned_task=planned_task,
    )
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
) -> None:
    if start_page_id < 0:
        raise click.ClickException("--start must be greater than or equal to 0")
    if end_page_id is not None and end_page_id < 0:
        raise click.ClickException("--end must be greater than or equal to 0")

    output_dir.mkdir(parents=True, exist_ok=True)
    documents = collect_input_documents(
        input_path=input_path,
        start_page_id=start_page_id,
        end_page_id=end_page_id,
    )

    timeout = build_http_timeout()
    local_server: LocalAPIServer | None = None
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as http_client:
        try:
            if api_url is None:
                local_server = LocalAPIServer()
                base_url = local_server.start()
                logger.info(f"Started local mineru-api at {base_url}")
                server_health = await wait_for_local_api_ready(http_client, local_server)
            else:
                server_health = await fetch_server_health(
                    http_client,
                    normalize_base_url(api_url),
                )

            planned_tasks = plan_tasks(
                documents=documents,
                backend=backend,
                processing_window_size=server_health.processing_window_size
                if backend == "pipeline"
                else DEFAULT_PROCESSING_WINDOW_SIZE,
            )
            progress = build_task_execution_progress(planned_tasks)
            concurrency = resolve_submit_concurrency(
                server_health.max_concurrent_requests,
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
            failures = await execute_planned_tasks(
                planned_tasks=planned_tasks,
                concurrency=concurrency,
                task_runner=lambda planned_task: run_planned_task(
                    client=http_client,
                    server_health=server_health,
                    planned_task=planned_task,
                    progress=progress,
                    form_data=form_data,
                    output_dir=output_dir,
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
            if local_server is not None:
                local_server.stop()


@click.command()
@click.version_option(__version__, "--version", "-v", help="display the version and exit")
@click.option(
    "-p",
    "--path",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="local filepath or directory. support pdf, png, jpg, jpeg files",
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
        )
    )


if __name__ == "__main__":
    main()
