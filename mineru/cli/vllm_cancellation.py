# Copyright (c) Opendatalab. All rights reserved.
import asyncio
import inspect
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from threading import RLock
from typing import Any, Iterator

from loguru import logger


def print_cancel_event(event: str, **fields: Any) -> None:
    field_text = " ".join(
        f"{key}={value}" for key, value in fields.items() if value is not None
    )
    message = f"[MINERU_CANCEL] event={event}"
    if field_text:
        message = f"{message} {field_text}"
    print(message, flush=True)


class VllmRequestCancelled(asyncio.CancelledError):
    pass


@dataclass(frozen=True)
class VllmRequestContext:
    registry: "VllmCancellationRegistry"
    task_id: str
    file_name: str


_current_context: ContextVar[VllmRequestContext | None] = ContextVar(
    "mineru_vllm_request_context",
    default=None,
)
_current_http_request_id: ContextVar[str | None] = ContextVar(
    "mineru_http_vllm_request_id",
    default=None,
)


@contextmanager
def vllm_request_context(
    registry: "VllmCancellationRegistry | None",
    task_id: str | None,
    file_name: str | None,
) -> Iterator[None]:
    if registry is None or task_id is None or file_name is None:
        yield
        return
    token = _current_context.set(
        VllmRequestContext(
            registry=registry,
            task_id=task_id,
            file_name=file_name,
        )
    )
    try:
        yield
    finally:
        _current_context.reset(token)


@dataclass
class RegisteredVllmRequest:
    task_id: str
    file_name: str
    request_id: str
    abort_handle: Any


class VllmCancellationRegistry:
    def __init__(self) -> None:
        self._lock = RLock()
        self._requests: dict[str, RegisteredVllmRequest] = {}
        self._requests_by_task_file: dict[tuple[str, str], set[str]] = {}
        self._cancelled_files: set[tuple[str, str]] = set()
        self._cancelled_tasks: set[str] = set()

    def mark_task_cancel_requested(self, task_id: str) -> None:
        with self._lock:
            self._cancelled_tasks.add(task_id)
        logger.info(f"Cancel requested for mineru_task_id={task_id}")
        print_cancel_event("cancel_requested", task_id=task_id, file="*")

    def mark_file_cancel_requested(self, task_id: str, file_name: str) -> None:
        with self._lock:
            self._cancelled_files.add((task_id, file_name))
        logger.info(f"Cancel requested for mineru_task_id={task_id}, file={file_name}")
        print_cancel_event("cancel_requested", task_id=task_id, file=file_name)

    def is_task_cancel_requested(self, task_id: str) -> bool:
        with self._lock:
            return task_id in self._cancelled_tasks

    def is_file_cancel_requested(self, task_id: str, file_name: str) -> bool:
        with self._lock:
            return task_id in self._cancelled_tasks or (task_id, file_name) in self._cancelled_files

    def register_request(
        self,
        task_id: str,
        file_name: str,
        request_id: str,
        abort_handle: Any,
    ) -> None:
        with self._lock:
            if task_id in self._cancelled_tasks or (task_id, file_name) in self._cancelled_files:
                raise VllmRequestCancelled(
                    f"VLLM request blocked because file was cancelled: task_id={task_id}, file={file_name}"
                )
            self._requests[request_id] = RegisteredVllmRequest(
                task_id=task_id,
                file_name=file_name,
                request_id=request_id,
                abort_handle=abort_handle,
            )
            self._requests_by_task_file.setdefault((task_id, file_name), set()).add(request_id)
        logger.info(
            "Registered VLLM request: "
            f"mineru_task_id={task_id}, file={file_name}, vllm_request_id={request_id}"
        )

    def unregister_request(self, request_id: str) -> None:
        with self._lock:
            record = self._requests.pop(request_id, None)
            if record is None:
                return
            request_ids = self._requests_by_task_file.get((record.task_id, record.file_name))
            if request_ids is not None:
                request_ids.discard(request_id)
                if not request_ids:
                    self._requests_by_task_file.pop((record.task_id, record.file_name), None)
        logger.info(
            "Unregistered VLLM request: "
            f"mineru_task_id={record.task_id}, file={record.file_name}, vllm_request_id={request_id}"
        )

    def get_active_request_ids(self, task_id: str, file_name: str | None = None) -> list[str]:
        with self._lock:
            if file_name is not None:
                return sorted(self._requests_by_task_file.get((task_id, file_name), set()))
            return sorted(
                request_id
                for request_id, record in self._requests.items()
                if record.task_id == task_id
            )

    async def abort_requests(self, task_id: str, file_name: str | None = None) -> list[str]:
        request_ids = self.get_active_request_ids(task_id, file_name)
        aborted: list[str] = []
        for request_id in request_ids:
            with self._lock:
                record = self._requests.get(request_id)
            if record is None:
                continue
            abort = getattr(record.abort_handle, "abort", None)
            if not callable(abort):
                logger.warning(
                    "Cannot abort VLLM request because abort() is unavailable: "
                    f"mineru_task_id={record.task_id}, file={record.file_name}, "
                    f"vllm_request_id={request_id}"
                )
                continue
            result = abort(request_id)
            if inspect.isawaitable(result):
                await result
            aborted.append(request_id)
            self.unregister_request(request_id)
            logger.info(
                "Aborted VLLM request: "
                f"mineru_task_id={record.task_id}, file={record.file_name}, "
                f"vllm_request_id={request_id}"
            )
        return aborted

    def build_file_status(self, task_id: str, file_names: list[str]) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {
                file_name: {
                    "cancel_requested": task_id in self._cancelled_tasks
                    or (task_id, file_name) in self._cancelled_files,
                    "active_vllm_request_ids": sorted(
                        self._requests_by_task_file.get((task_id, file_name), set())
                    ),
                }
                for file_name in file_names
            }

    def cleanup_task(self, task_id: str) -> None:
        with self._lock:
            for request_id, record in list(self._requests.items()):
                if record.task_id == task_id:
                    self._requests.pop(request_id, None)
                    self._requests_by_task_file.get((record.task_id, record.file_name), set()).discard(request_id)
            for key, request_ids in list(self._requests_by_task_file.items()):
                if key[0] == task_id or not request_ids:
                    self._requests_by_task_file.pop(key, None)
            self._cancelled_tasks.discard(task_id)
            for key in list(self._cancelled_files):
                if key[0] == task_id:
                    self._cancelled_files.discard(key)


def patch_vllm_async_llm_for_cancellation(vllm_async_llm: Any) -> None:
    if vllm_async_llm is None or getattr(vllm_async_llm, "_mineru_cancel_patch", False):
        return

    original_generate = vllm_async_llm.generate

    def generate_with_tracking(*args: Any, **kwargs: Any):
        request_id = kwargs.get("request_id")
        context = _current_context.get()
        if context is None or not isinstance(request_id, str):
            return original_generate(*args, **kwargs)

        context.registry.register_request(
            task_id=context.task_id,
            file_name=context.file_name,
            request_id=request_id,
            abort_handle=vllm_async_llm,
        )

        async def tracked_iterator():
            try:
                async for output in original_generate(*args, **kwargs):
                    yield output
            finally:
                context.registry.unregister_request(request_id)

        return tracked_iterator()

    vllm_async_llm.generate = generate_with_tracking
    vllm_async_llm._mineru_cancel_patch = True
    logger.info("Enabled VLLM request cancellation tracking on AsyncLLM")


class HttpVllmRequestAbortHandle:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._lock = RLock()
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    def register_task(self, request_id: str, task: asyncio.Task[Any] | None) -> None:
        if task is None:
            return
        with self._lock:
            self._tasks[request_id] = task

    def unregister_task(self, request_id: str) -> None:
        with self._lock:
            self._tasks.pop(request_id, None)

    async def abort(self, request_id: str) -> None:
        with self._lock:
            task = self._tasks.get(request_id)
        if task is not None and not task.done():
            task.cancel()
            logger.info(f"Cancelled local HTTP VLLM request task: vllm_request_id={request_id}")
        logger.info(
            "Skipped VLLM HTTP server-side cancel because /v1/chat/completions "
            f"request_id is not a /v1/responses response_id: vllm_request_id={request_id}"
        )


def patch_http_vlm_client_for_cancellation(vlm_client: Any) -> None:
    if vlm_client is None or getattr(vlm_client, "_mineru_cancel_patch", False):
        return
    if not all(hasattr(vlm_client, attr) for attr in ("aio_predict", "build_request_body")):
        return
    if not hasattr(vlm_client, "chat_url"):
        return

    original_aio_predict = vlm_client.aio_predict
    original_build_request_body = vlm_client.build_request_body
    abort_handle = HttpVllmRequestAbortHandle(vlm_client)

    def build_request_body_with_request_id(*args: Any, **kwargs: Any):
        request_body = original_build_request_body(*args, **kwargs)
        request_id = _current_http_request_id.get()
        if request_id:
            request_body["request_id"] = request_id
        return request_body

    async def aio_predict_with_tracking(*args: Any, **kwargs: Any):
        context = _current_context.get()
        if context is None:
            return await original_aio_predict(*args, **kwargs)

        short_task_id = "".join(
            char for char in context.task_id if char.isascii() and char.isalnum()
        )[:12] or "task"
        request_id = f"mineru-{short_task_id}-{uuid.uuid4().hex[:12]}"
        current_task = asyncio.current_task()
        abort_handle.register_task(request_id, current_task)
        context.registry.register_request(
            task_id=context.task_id,
            file_name=context.file_name,
            request_id=request_id,
            abort_handle=abort_handle,
        )
        token = _current_http_request_id.set(request_id)
        try:
            return await original_aio_predict(*args, **kwargs)
        finally:
            _current_http_request_id.reset(token)
            abort_handle.unregister_task(request_id)
            context.registry.unregister_request(request_id)

    vlm_client.build_request_body = build_request_body_with_request_id
    vlm_client.aio_predict = aio_predict_with_tracking
    vlm_client._mineru_cancel_patch = True
    logger.info("Enabled VLLM request cancellation tracking on HTTP VLM client")
