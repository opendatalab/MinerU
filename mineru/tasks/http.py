"""Tasks 的 HTTP 输入、响应序列化与路由，生命周期交给共享服务管理。"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any
from urllib.parse import quote

from docvortex.document.page_range import normalize_page_range_input
from docvortex.errors import InvalidRequestError
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.requests import ClientDisconnect

from .contracts import (
    CancelTaskResponse,
    CanceledTaskFile,
    CreateTaskRequest,
    CreateTaskResponse,
    CreatedTaskFile,
    TaskDetailResponse,
    TaskFileDetail,
    TaskFileInput,
    TaskProgress,
    TaskUploadLink,
    UploadFileResponse,
    UploadTaskOptions,
)
from .errors import invalid_task_request
from .execution import TaskBackend
from .service import TASK_UPLOAD_TTL, TaskRecord, TaskStore


async def _read_submission(request: Request) -> tuple[UploadTaskOptions, list[TaskFileInput]]:
    """只读取 JSON 清单，严格拒绝表单、直传字节和 URL 查询参数。"""
    if request.query_params:
        invalid_task_request("POST parameters must be supplied in the JSON body, not the URL")
    media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if media_type != "application/json":
        invalid_task_request("Use application/json", status_code=415)
    try:
        body = CreateTaskRequest.model_validate(await request.json())
        inputs = body.files
        options = UploadTaskOptions.model_validate(body.model_dump(exclude={"files"}))
        options.page_range = normalize_page_range_input(options.page_range)
        options.output_formats = list(dict.fromkeys(options.output_formats))
        for index, item in enumerate(inputs):
            try:
                item.page_range = normalize_page_range_input(
                    item.page_range if item.page_range is not None else options.page_range
                )
            except InvalidRequestError as exc:
                invalid_task_request(str(exc), code=exc.code, param=f"files.{index}.page_range")
        return options, inputs
    except ClientDisconnect:
        invalid_task_request("Request interrupted", code="request_interrupted")
    except InvalidRequestError as exc:
        invalid_task_request(str(exc), code=exc.code, param=exc.param)
    except ValidationError as exc:
        location = exc.errors()[0]["loc"]
        invalid_task_request("Invalid request: " + str(exc), param=".".join(map(str, location)) or None)
    except ValueError as exc:
        invalid_task_request(str(exc))


def _retention_lease(request: Request) -> int | None:
    """读取 Router 的有界结果租约，避免父任务结束前 worker 提前清理子文件。"""
    value = request.headers.get("x-mineru-task-lease-seconds")
    if value is None:
        return None
    try:
        seconds = int(value)
        if not 0 <= seconds <= TASK_UPLOAD_TTL:
            raise ValueError
    except ValueError:
        invalid_task_request("Task result lease must be between 0 and 3600 seconds")
    return seconds


def _url(request: Request, path: str) -> str:
    """生成包含反向代理 root_path 的本部署地址，客户端无需自行拼接路径。"""
    return str(request.base_url).rstrip("/") + path


def _upload_descriptor(record: TaskRecord, index: int, request: Request) -> TaskUploadLink | None:
    """上传描述仅表示当前可执行的动作，不将服务端准备误判为文件就绪。"""
    item, state = record.inputs[index], record.file_states[index]
    if (
        item.url is None
        and state.upload is None
        and not state.upload_active
        and state.status == "waiting-file"
        and record.status != "canceled"
        and record.expires_at > time.time()
    ):
        return TaskUploadLink(
            url=_url(request, f"/v1/tasks/{record.id}/uploads/{quote(item.file_id, safe='')}"),
            headers={"Content-Type": "application/octet-stream"},
            expires_at=record.expires_at,
        )
    return None


def _file_payload(record: TaskRecord, index: int, request: Request) -> dict[str, Any]:
    """从内部执行记录提取文件信息，各响应模型显式限定自身公开字段。"""
    item, state = record.inputs[index], record.file_states[index]
    payload: dict[str, Any] = {
        "file_id": item.file_id,
        "file_name": item.file_name,
        "bytes": item.bytes,
        "url": item.url,
        "status": state.status,
        "upload": _upload_descriptor(record, index, request),
    }
    if state.execution is not None:
        payload["error"] = state.execution.error
        if state.status == "completed":
            payload["parse"] = state.execution.parse
        if (metadata := state.execution.input_metadata) is not None:
            payload["bytes"] = metadata.bytes
        if state.status == "completed" and state.execution.result_file_id:
            if state.execution.result_file_id != item.file_id:
                raise ValueError("Result file identity differs from the reserved task file ID")
            payload["result_url"] = _url(request, f"/v1/files/{quote(item.file_id, safe='')}/content")
    if state.error is not None:
        payload["error"] = state.error
    return payload


def create_task_payload(record: TaskRecord, request: Request) -> dict[str, Any]:
    """创建响应只确认服务端文件标识和上传入口，不依赖 GET 详情模型。"""
    return CreateTaskResponse(
        task_id=record.id,
        status=record.status,
        files=[
            CreatedTaskFile(
                file_id=item.file_id,
                file_name=item.file_name,
                status=state.status,
                upload=_upload_descriptor(record, index, request),
            )
            for index, (item, state) in enumerate(zip(record.inputs, record.file_states))
        ],
    ).model_dump(exclude_none=True)


def task_payload(record: TaskRecord, request: Request) -> dict[str, Any]:
    """查询返回当前完整详情，递归省略空值且保留进度中的零。"""
    starts = [
        state.execution.started_at for state in record.file_states if state.execution is not None and state.execution.started_at
    ]
    return TaskDetailResponse(
        task_id=record.id,
        status=record.status,
        created_at=record.created_at,
        started_at=min(starts) if starts else None,
        finished_at=record.finished_at,
        output_formats=record.options.output_formats,
        progress=TaskProgress(
            completed=sum(state.status == "completed" for state in record.file_states),
            failed=sum(state.status == "failed" for state in record.file_states),
            total=len(record.inputs),
        ),
        files=[TaskFileDetail.model_validate(_file_payload(record, index, request)) for index in range(len(record.inputs))],
        error=record.error,
    ).model_dump(exclude_none=True)


def cancel_task_payload(record: TaskRecord, request: Request) -> dict[str, Any]:
    """取消响应保留全部逐文件终态和成功产物，不携带上传或其他详情字段。"""
    return CancelTaskResponse(
        task_id=record.id,
        status=record.status,
        finished_at=record.finished_at,
        files=[CanceledTaskFile.model_validate(_file_payload(record, index, request)) for index in range(len(record.inputs))],
    ).model_dump(exclude_none=True)


def _request_schema() -> dict[str, Any]:
    """OpenAPI 使用与运行时相同的 JSON 清单和格式约束。"""
    schema = CreateTaskRequest.model_json_schema()
    definitions = schema.pop("$defs", {})

    def expand(value: Any) -> Any:
        """内联局部模型引用，避免 OpenAPI 把 JSON Schema 的 $defs 当成全局路径。"""
        if isinstance(value, dict):
            if "$ref" in value:
                return expand(definitions[value["$ref"].rsplit("/", 1)[-1]])
            return {key: expand(item) for key, item in value.items()}
        if isinstance(value, list):
            return [expand(item) for item in value]
        return value

    return {"requestBody": {"required": True, "content": {"application/json": {"schema": expand(schema)}}}}


def build_task_router(factory: Callable[[Request], TaskBackend]) -> APIRouter:
    """两类部署使用同一任务协议，下载交给已有 Files 入口。"""
    router = APIRouter(prefix="/v1", tags=["Tasks"])

    @router.post("/tasks", status_code=202, response_model=CreateTaskResponse, openapi_extra=_request_schema())
    async def create_task(request: Request) -> Response:
        """先验证 JSON 再建立稳定任务 ID，HTTP 结束后逐文件工作独立继续。"""
        backend = factory(request)
        store: TaskStore = request.app.state.task_store
        try:
            options, inputs = await _read_submission(request)
            record = await store.create(backend, options, inputs, lease_seconds=_retention_lease(request) or 0)
        except BaseException:
            backend.release_uploads(rollback=True)
            raise
        return JSONResponse(create_task_payload(record, request), status_code=202)

    @router.get("/tasks/{task_id}", response_model=TaskDetailResponse)
    async def get_task(task_id: str, request: Request) -> Response:
        """读取逐文件状态与结果地址，上传链接截止时间保持不变。"""
        store: TaskStore = request.app.state.task_store
        record = store.lookup(task_id, factory(request).owner_scope)
        store.update_lease(record, _retention_lease(request))
        await store.refresh(record)
        return JSONResponse(task_payload(record, request))

    @router.put(
        "/tasks/{task_id}/uploads/{file_id}",
        response_model=UploadFileResponse,
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
            }
        },
    )
    async def upload_task_file(task_id: str, file_id: str, request: Request) -> Response:
        """每条上传链接只接收一个文件，该文件就绪后立即提交执行。"""
        store: TaskStore = request.app.state.task_store
        record = store.lookup(task_id, factory(request).owner_scope)
        lease_seconds = _retention_lease(request)
        store.update_lease(record, lease_seconds)
        try:
            skipped = await store.upload(
                record,
                file_id,
                chunks=request.stream(),
                content_length=request.headers.get("content-length"),
                mime_type=request.headers.get("content-type", "application/octet-stream"),
            )
        except ClientDisconnect:
            invalid_task_request("Upload interrupted; retry this file on the same task", code="upload_interrupted")
        # 大文件转传可超过租约长度；受理完成时续租，保护紧接着发布的结果。
        store.update_lease(record, lease_seconds)
        return JSONResponse(
            UploadFileResponse(
                task_id=record.id,
                file_id=file_id,
                status=next(state.status for item, state in zip(record.inputs, record.file_states) if item.file_id == file_id),
            ).model_dump(),
            headers={"Connection": "close"} if skipped and request.scope.get("http_version") == "1.1" else None,
        )

    @router.post("/tasks/{task_id}/cancel", response_model=CancelTaskResponse)
    async def cancel_task(task_id: str, request: Request) -> Response:
        """统一取消等待上传与正在执行的任务，不重建后台作业。"""
        store: TaskStore = request.app.state.task_store
        record = store.lookup(task_id, factory(request).owner_scope)
        await store.refresh(record)
        await store.cancel(record, wait=False)
        return JSONResponse(cancel_task_payload(record, request))

    return router


__all__ = ["build_task_router", "cancel_task_payload", "create_task_payload", "task_payload"]
