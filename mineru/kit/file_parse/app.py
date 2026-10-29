"""单文件同步 HTTP 包装层；成功返回附件，超时返回可恢复的请求链接。"""

from __future__ import annotations

import asyncio
import hashlib
import os
import re
import tempfile
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ValidationError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import ClientDisconnect
from starlette.types import ASGIApp, Receive, Scope, Send

from ...task_types import TASK_OUTPUT_FORMATS
from ...types import TIERS
from .models import GatewayError, ParseOptions, Record
from .service import FileParseService
from .settings import FileParseSettings

REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
MIME_TYPES = {
    "zip": "application/zip",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "html": "text/html",
}


class StatusResponse(BaseModel):
    """查询与超时响应共享结构，创建尚未确认时 task_id 为空。"""

    request_id: str
    task_id: str | None
    status: str
    status_url: str
    result_url: str | None
    output_formats: list[str]
    error: dict[str, Any] | None


class ReceiveTimeoutMiddleware:
    """对接收请求体的每次等待设置独立空闲超时。"""

    def __init__(self, app: ASGIApp, timeout: float) -> None:
        """记录传输预算，不限制正常持续传输大文件的总时长。"""
        self.app = app
        self.timeout = timeout

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """仅约束 POST 请求体，响应等待期间的断连观察不应用传输超时。"""
        body_finished = False

        async def timed_receive() -> dict[str, Any]:
            """在最后一块请求体到达后恢复普通 receive 行为。"""
            nonlocal body_finished
            try:
                if body_finished:
                    return await receive()
                message = await asyncio.wait_for(receive(), self.timeout)
            except asyncio.TimeoutError as exc:
                raise GatewayError("upload_receive_timeout", "Receiving the file timed out", 408) from exc
            if message["type"] == "http.request" and not message.get("more_body", False):
                body_finished = True
            return message

        await self.app(scope, timed_receive if scope["type"] == "http" and scope.get("method") == "POST" else receive, send)


class TemporaryFileResponse(FileResponse):
    """无论发送完成、客户端断连还是取消，都释放已验证的结果临时文件。"""

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """用 finally 兜底清理，避免只依赖成功发送后的后台回调。"""
        try:
            await super().__call__(scope, receive, send)
        finally:
            Path(self.path).unlink(missing_ok=True)


def _request_schema() -> dict[str, Any]:
    """为手动严格解析的 multipart 请求提供完整 OpenAPI 描述。"""
    properties = ParseOptions.model_json_schema()["properties"]
    properties.update(
        {
            "file": {"type": "string", "format": "binary"},
            "request_id": {"type": "string", "pattern": REQUEST_ID.pattern},
            "wait_timeout": {"type": "integer", "minimum": 0, "maximum": 3600, "default": 3600},
        }
    )
    properties["tier"] = {
        "title": "Tier",
        "type": "string",
        "enum": list(TIERS),
        "description": (
            "Optional PDF/image tier. Omit this multipart field to let the server choose "
            "standard, basic, flash, then advanced; "
            "advanced is automatic only when it is the sole available tier. Explicit unsupported PDF/image tiers are rejected. "
            "Supported native documents always use flash. Empty strings and the literal string 'null' are invalid."
        ),
    }
    properties["output_formats"] = {
        "type": "array",
        "items": {"type": "string", "enum": list(TASK_OUTPUT_FORMATS)},
        "minItems": 1,
        "default": ["default"],
    }
    return {
        "requestBody": {
            "required": True,
            "content": {
                "multipart/form-data": {
                    "schema": {"type": "object", "properties": properties, "required": ["file"], "additionalProperties": False},
                    "encoding": {"output_formats": {"style": "form", "explode": True}},
                }
            },
        }
    }


async def _receive_file(request: Request, service: FileParseService) -> tuple[Record, int]:
    """严格接收单文件 multipart 并持久化源字节，完整接受后才启动上游工作。"""
    if request.query_params:
        raise GatewayError("invalid_request", "Parameters must be supplied as multipart fields", 400)
    if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
        raise GatewayError("unsupported_media_type", "Expected multipart/form-data", 415)
    allowed = {"file", "request_id", "wait_timeout", "tier", "ocr_mode", "page_range", "output_formats"}
    source: Path | None = None
    accepted = False
    try:
        async with request.form(max_files=1, max_fields=32, max_part_size=65536) as form:
            if set(form) - allowed or any(len(form.getlist(key)) != 1 for key in form if key != "output_formats"):
                raise GatewayError("invalid_request", "Unknown or repeated scalar multipart fields", 400)
            upload = form.get("file")
            if not isinstance(upload, UploadFile) or any(
                not isinstance(value, str) for key, value in form.multi_items() if key != "file"
            ):
                raise GatewayError("invalid_request", "Exactly one file and textual parameters are required", 400)
            filename = (upload.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
            if filename in {"", ".", ".."} or len(filename) > 1024 or any(ord(char) < 32 for char in filename):
                raise GatewayError("invalid_request", "A valid filename is required", 400)
            request_id = str(form.get("request_id", uuid.uuid4().hex))
            if not REQUEST_ID.fullmatch(request_id):
                raise GatewayError("invalid_request", "Invalid request_id", 400)
            request.state.request_id = request_id
            try:
                wait_timeout = int(str(form.get("wait_timeout", "3600")))
                if not 0 <= wait_timeout <= 3600:
                    raise ValueError("wait_timeout out of range")
                options = ParseOptions.model_validate(
                    {
                        **{key: form[key] for key in ("tier", "ocr_mode", "page_range") if key in form},
                        "output_formats": form.getlist("output_formats") if "output_formats" in form else ["default"],
                    }
                )
            except (ValueError, ValidationError) as exc:
                raise GatewayError("invalid_request", "Invalid parsing parameters or wait_timeout", 400) from exc
            descriptor, name = tempfile.mkstemp(prefix="source-", dir=service.store.sources)
            source = Path(name)
            size, hasher = 0, hashlib.sha256()
            with os.fdopen(descriptor, "wb") as destination:
                while chunk := await upload.read(1024 * 1024):
                    size += len(chunk)
                    hasher.update(chunk)
                    await asyncio.to_thread(destination.write, chunk)
                destination.flush()
                await asyncio.to_thread(os.fsync, destination.fileno())
            if size == 0:
                raise GatewayError("invalid_request", "The file must not be empty", 400)
            operation = asyncio.create_task(service.accept(request_id, filename, size, hasher.hexdigest(), options, source))
            try:
                record = await asyncio.shield(operation)
            except asyncio.CancelledError:
                await operation
                accepted = True
                raise
            accepted = True
            return record, wait_timeout
    except HTTPException as exc:
        raise GatewayError("invalid_request", "Invalid multipart upload or more than one file", 400) from exc
    except ClientDisconnect as exc:
        raise GatewayError("upload_interrupted", "Client disconnected before the file was accepted", 400) from exc
    except OSError as exc:
        raise GatewayError("storage_error", "Cannot retain the uploaded file", 500) from exc
    finally:
        if source is not None and not accepted:
            source.unlink(missing_ok=True)


async def _attachment(service: FileParseService, record: Record) -> TemporaryFileResponse:
    """获取已验证产物并设置可直接下载的文件名、类型和任务标识。"""
    try:
        path = await service.artifact(record)
    except OSError as exc:
        raise GatewayError("storage_error", "Cannot save the downloaded result", 500) from exc
    formats = record.options["output_formats"]
    extension = formats[0] if len(formats) == 1 and formats[0] in {"docx", "html"} else "zip"
    task = record.task
    assert task is not None
    return TemporaryFileResponse(
        path,
        filename=f"{Path(record.filename).stem}.{extension}",
        media_type=MIME_TYPES[extension],
        headers={"X-Request-ID": record.request_id, "X-Task-ID": task.task_id},
    )


async def _finish_request(request: Request, service: FileParseService, record: Record, timeout: int) -> Response:
    """等待预算只包围排队、解析及导出，结果下载使用独立传输预算。"""
    if await service.wait(record, timeout):
        return await _attachment(service, record)
    payload = service.payload(record, str(request.base_url))
    headers = {"Location": payload["status_url"], "X-Request-ID": record.request_id, "Retry-After": "2"}
    if payload["task_id"] is not None:
        headers["X-Task-ID"] = payload["task_id"]
    return JSONResponse(payload, status_code=202, headers=headers)


async def _until_disconnect(request: Request, operation: asyncio.Task[Response]) -> Response:
    """客户端离开时只停止该连接的等待或下载，不传播到托管上传任务。"""

    async def disconnected() -> None:
        """在请求体读取完毕后等待 ASGI 的真实断连消息。"""
        while (await request.receive())["type"] != "http.disconnect":
            pass

    watcher = asyncio.create_task(disconnected())
    try:
        done, _ = await asyncio.wait({operation, watcher}, return_when=asyncio.FIRST_COMPLETED)
        if operation in done:
            return await operation
        return Response(status_code=499)
    finally:
        watcher.cancel()
        if not operation.done():
            operation.cancel()
        await asyncio.gather(watcher, operation, return_exceptions=True)


def create_app(settings: FileParseSettings, *, transport: httpx.AsyncBaseTransport | None = None) -> FastAPI:
    """构建独立包装服务，生命周期内创建数据库和 HTTP 客户端。"""

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """启动恢复、后台观察和关闭清理均由应用生命周期管理。"""
        service = FileParseService(settings, transport)
        application.state.service = service
        try:
            await service.start()
            yield
        finally:
            await service.close()

    application = FastAPI(title="MinerU File Parse", version="1", lifespan=lifespan)
    application.add_middleware(ReceiveTimeoutMiddleware, timeout=settings.transfer_timeout)

    @application.exception_handler(GatewayError)
    async def gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        """错误响应保留已接受请求的查询链接，但不暴露上游凭据及签名链接。"""
        payload: dict[str, Any] = {}
        service = request.app.state.service
        request_id = getattr(request.state, "request_id", None)
        record = service.records.get(request_id)
        if record is not None:
            payload.update(service.payload(record, str(request.base_url)))
        payload["error"] = exc.detail()
        return JSONResponse(payload, status_code=exc.status_code)

    @application.post(
        "/v1/file_parse",
        openapi_extra=_request_schema(),
        response_model=None,
        response_class=Response,
        responses={
            200: {
                "description": "Result attachment",
                "content": {mime: {"schema": {"type": "string", "format": "binary"}} for mime in MIME_TYPES.values()},
            },
            202: {"model": StatusResponse, "description": "Accepted; query status_url to continue"},
        },
    )
    async def file_parse(request: Request) -> Response:
        """接收一个文件并同步等待，调用方可通过 request_id 恢复已有任务。"""
        service = request.app.state.service
        record, timeout = await _receive_file(request, service)
        return await _until_disconnect(request, asyncio.create_task(_finish_request(request, service, record, timeout)))

    @application.get("/v1/file_parse/{request_id}", response_model=StatusResponse)
    async def status(request_id: str, request: Request) -> dict[str, Any]:
        """查询已有请求，创建结果未知时返回明确状态而不发起新的任务。"""
        request.state.request_id = request_id
        service = request.app.state.service
        record = service.get(request_id)
        if record.task is not None:
            await service.refresh(record)
        return service.payload(record, str(request.base_url))

    @application.get("/v1/file_parse/{request_id}/result", response_class=FileResponse)
    async def result(request_id: str, request: Request) -> Response:
        """使用当前有效的上游链接获取结果，断连时释放本次下载临时文件。"""
        request.state.request_id = request_id
        service = request.app.state.service
        record = service.get(request_id)
        if record.task is None:
            service.raise_record_error(record)
        return await _until_disconnect(request, asyncio.create_task(_attachment(service, record)))

    return application


__all__ = ["StatusResponse", "create_app"]
