"""只通过共同 Tasks HTTP 契约工作的流式客户端。"""

from __future__ import annotations

import asyncio
import json
import time
import zipfile
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx
from pydantic import ValidationError

from ...task_types import task_status_url
from ...tasks.contracts import ErrorResponse
from .models import GatewayError, Record, RemoteCreatedTask, RemoteFile, RemoteTask
from .settings import FileParseSettings, origin

CHUNK_BYTES = 1024 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
REDIRECTS = frozenset({301, 302, 303, 307, 308})


def _is_task_creation_rejected(status: int, body: bytes) -> bool:
    """仅识别 Tasks 明确未接单的档位错误，畸形或混入任务字段的响应仍视为未知。"""
    if status != 503:
        return False
    try:
        response = ErrorResponse.model_validate_json(body)
    except ValidationError:
        return False
    return response.error.code == "quality_tier_unavailable"


async def file_chunks(path: Path) -> AsyncIterator[bytes]:
    """从磁盘分块读取上传内容，不在事件循环中执行大块文件 I/O。"""
    with path.open("rb") as source:
        while chunk := await asyncio.to_thread(source.read, CHUNK_BYTES):
            yield chunk


def validate_artifact(path: Path, formats: list[str]) -> None:
    """校验下载的基本格式和 ZIP 完整性，不解压文件或重新生成产物。"""
    try:
        if path.stat().st_size == 0:
            raise ValueError("Empty artifact")
        if formats == ["html"]:
            with path.open("r", encoding="utf-8-sig") as source:
                prefix = source.read(65536).lower()
            if "<html" not in prefix:
                raise ValueError("Expected an HTML document")
        else:
            with zipfile.ZipFile(path) as archive:
                if archive.testzip() is not None:
                    raise ValueError("Corrupt ZIP member")
                if formats == ["docx"] and "word/document.xml" not in archive.namelist():
                    raise ValueError("Expected a DOCX document")
    except (ValueError, UnicodeError, zipfile.BadZipFile, RuntimeError) as exc:
        raise GatewayError("invalid_result", "Upstream returned an invalid or incomplete result", retryable=True) from exc


class TaskClient:
    """所有请求显式选择认证；创建任务始终只发送一次。"""

    def __init__(self, settings: FileParseSettings, transport: httpx.AsyncBaseTransport | None = None) -> None:
        """创建轻量连接池，测试可以注入传输而不替换业务实现。"""
        self.settings = settings
        self.origin = origin(settings.api_url)
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.request_timeout, connect=settings.connect_timeout),
            follow_redirects=False,
            transport=transport,
        )
        self.transfer_timeout = httpx.Timeout(settings.transfer_timeout, connect=settings.connect_timeout)
        self.retry_delay = 1.0

    async def close(self) -> None:
        """释放上游连接池，停止服务时不发送任务取消请求。"""
        await self.client.aclose()

    def resolve(self, url: str, *, base: str | None = None, metadata: bool = False) -> str:
        """保留签名查询串，元数据查询必须留在已配置的 API 同源地址。"""
        resolved = urljoin(base or self.settings.api_url + "/", url)
        try:
            target = origin(resolved)
        except ValueError as exc:
            raise GatewayError("upstream_protocol_error", "Upstream returned an invalid HTTP URL") from exc
        if metadata and target != self.origin:
            raise GatewayError("upstream_protocol_error", "Task status URL must use the configured API origin")
        return resolved

    def headers(self, url: str, supplied: Mapping[str, str] | None = None) -> dict[str, str]:
        """只在 API 同源请求中添加密钥，保留存储服务显式要求的上传请求头。"""
        try:
            headers = httpx.Headers(supplied or {})
        except (UnicodeError, ValueError) as exc:
            raise GatewayError("upstream_protocol_error", "Upstream returned invalid upload headers") from exc
        if any("\r" in key + value or "\n" in key + value for key, value in headers.items()):
            raise GatewayError("upstream_protocol_error", "Upstream returned invalid upload headers")
        token = f"Bearer {self.settings.api_key}" if self.settings.api_key else None
        if origin(url) == self.origin:
            if token is not None and "authorization" not in headers:
                headers["Authorization"] = token
        elif token is not None and headers.get("Authorization") == token:
            raise GatewayError("upstream_protocol_error", "Refusing to forward the API key to another origin")
        return dict(headers)

    async def _read_metadata(self, response: httpx.Response) -> bytes:
        """限制元数据响应的总大小，避免意外把产物当成 JSON 缓冲。"""
        content = bytearray()
        async for chunk in response.aiter_bytes():
            content.extend(chunk)
            if len(content) > MAX_JSON_BYTES:
                raise GatewayError("upstream_protocol_error", "Upstream metadata response is too large")
        return bytes(content)

    def _http_error(self, status: int, body: bytes) -> GatewayError:
        """保留上游结构化错误码，同时区分配置认证失败和可重试响应。"""
        code, message = "upstream_http_error", f"Upstream returned HTTP {status}"
        try:
            payload = json.loads(body)
            detail = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(detail, dict):
                code = str(detail.get("code") or code)
                message = str(detail.get("message") or message)
        except (ValueError, UnicodeError):
            pass
        return GatewayError(code, message, retryable=status in RETRYABLE_HTTP, upstream_status=status)

    def _snapshot(self, body: bytes, expected: RemoteTask | None, formats: list[str]) -> RemoteTask:
        """校验任务 ID、单文件映射和输出格式，拒绝不匹配的任务响应。"""
        try:
            task = RemoteTask.model_validate({**json.loads(body), "details_loaded": True})
            if expected is not None and task.task_id != expected.task_id:
                raise ValueError("Task identity changed")
            if list(dict.fromkeys(task.output_formats)) != formats:
                raise ValueError("Task output formats changed")
            if expected is not None and (
                task.files[0].file_id != expected.files[0].file_id
                or task.files[0].file_name != expected.files[0].file_name
                or task.files[0].bytes != expected.files[0].bytes
            ):
                raise ValueError("Task input identity changed")
            for item in task.files:
                if item.upload is not None:
                    item.upload.url = self.resolve(item.upload.url)
            if task.files[0].result_url:
                task.files[0].result_url = self.resolve(task.files[0].result_url)
            return task
        except (ValidationError, ValueError, TypeError) as exc:
            raise GatewayError("upstream_protocol_error", "Upstream returned inconsistent task metadata") from exc

    def _created_snapshot(self, body: bytes, request: dict[str, Any]) -> RemoteTask:
        """用创建回执和本地请求构造恢复上下文，查询详情之前即可安全持久化 ID。"""
        try:
            created = RemoteCreatedTask.model_validate_json(body)
            item = created.files[0]
            source = request["files"][0]
            if item.file_name != source["file_name"]:
                raise ValueError("Created task input identity changed")
            if item.upload is not None:
                item.upload.url = self.resolve(item.upload.url)
            return RemoteTask(
                task_id=created.task_id,
                status=created.status,
                files=[RemoteFile(**item.model_dump(), bytes=source["bytes"])],
                output_formats=request["output_formats"],
                details_loaded=False,
            )
        except (ValidationError, ValueError) as exc:
            raise GatewayError("upstream_protocol_error", "Upstream returned inconsistent creation metadata") from exc

    async def _json_request(
        self,
        method: str,
        url: str,
        formats: list[str],
        *,
        expected: RemoteTask | None = None,
        payload: dict[str, Any] | None = None,
    ) -> RemoteTask:
        """安全查询有界重试，创建响应丢失或内容不可用则标记结果不确定。"""
        for attempt in range(1 if method == "POST" else 3):
            try:
                async with self.client.stream(method, url, headers=self.headers(url), json=payload) as response:
                    body = await self._read_metadata(response)
                    if response.status_code not in ({202} if method == "POST" else {200}):
                        error = self._http_error(response.status_code, body)
                        if (
                            method == "POST"
                            and (response.status_code >= 500 or response.status_code == 408)
                            and not _is_task_creation_rejected(response.status_code, body)
                        ):
                            raise GatewayError("submission_unknown", "Task creation outcome is unknown; not resubmitting")
                        raise error
                if method == "POST":
                    assert payload is not None
                    return self._created_snapshot(body, payload)
                return self._snapshot(body, expected, formats)
            except (httpx.HTTPError, GatewayError) as exc:
                error = (
                    exc
                    if isinstance(exc, GatewayError)
                    else GatewayError("upstream_unreachable", "Cannot communicate with the upstream service", retryable=True)
                )
                if method == "POST":
                    if isinstance(exc, httpx.HTTPError) or error.code == "upstream_protocol_error":
                        error = GatewayError("submission_unknown", "Task creation outcome is unknown; not resubmitting")
                    raise error from exc
                if attempt == 2 or not error.retryable:
                    raise error from exc
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        raise AssertionError("unreachable")

    async def create(self, record: Record) -> RemoteTask:
        """只提交一条清单，hash 用于上游缓存预检及上传完整性检查。"""
        options = {key: value for key, value in record.options.items() if value is not None}
        payload = {
            **options,
            "files": [
                {
                    "file_name": record.filename,
                    "bytes": record.size,
                    "sha256sum": record.sha256sum,
                }
            ],
        }
        return await self._json_request(
            "POST", self.settings.api_url + "/v1/tasks", record.options["output_formats"], payload=payload
        )

    async def query(self, task: RemoteTask) -> RemoteTask:
        """保留部署前缀并按稳定任务 ID 查询，失败不会触发创建操作。"""
        return await self._json_request(
            "GET", task_status_url(self.settings.api_url, task.task_id), task.output_formats, expected=task
        )

    async def upload(self, task: RemoteTask, path: Path, size: int) -> RemoteTask:
        """文件准备期间只轮询，明确出现上传目标后才发送，最多重试三次。"""
        attempts = 0
        uploaded = False
        last_error: GatewayError | None = None
        while task.files[0].status == "waiting-file":
            link = task.files[0].upload
            if link is None or uploaded:
                await asyncio.sleep(self.settings.poll_interval)
                task = await self.query(task)
                continue
            if attempts >= 3:
                raise last_error or GatewayError("upload_failed", "Upload retry budget exhausted")
            if link.expires_at <= time.time():
                raise GatewayError("upload_expired", "The task upload link has expired", 410)
            attempts += 1
            try:
                headers = self.headers(link.url, link.headers)
                headers["content-length"] = str(size)
                async with self.client.stream(
                    "PUT", link.url, headers=headers, content=file_chunks(path), timeout=self.transfer_timeout
                ) as response:
                    body = await self._read_metadata(response)
                    if not 200 <= response.status_code < 300:
                        raise self._http_error(response.status_code, body)
                uploaded = True
                task = await self.query(task)
            except (httpx.HTTPError, GatewayError) as exc:
                last_error = (
                    exc
                    if isinstance(exc, GatewayError)
                    else GatewayError("upload_interrupted", "The task upload was interrupted", retryable=True)
                )
                task = await self.query(task)
                if task.files[0].status != "waiting-file":
                    return task
                if task.files[0].upload is None:
                    continue
                if attempts == 3 or not last_error.retryable:
                    raise last_error from exc
                await asyncio.sleep(self.retry_delay * attempts)
        return task

    async def _download_once(self, url: str, path: Path) -> None:
        """逐跳重新决定认证并流式写盘，在完整接收前不向调用者发送成功响应。"""
        for _ in range(6):
            async with self.client.stream(
                "GET",
                url,
                headers=self.headers(url),
                timeout=self.transfer_timeout,
            ) as response:
                if response.status_code in REDIRECTS:
                    location = response.headers.get("location")
                    if not location:
                        raise GatewayError("upstream_protocol_error", "Download redirect has no Location")
                    url = self.resolve(location, base=url)
                    continue
                if response.status_code != 200:
                    raise self._http_error(response.status_code, await self._read_metadata(response))
                size = 0
                with path.open("wb") as target:
                    async for chunk in response.aiter_bytes(CHUNK_BYTES):
                        await asyncio.to_thread(target.write, chunk)
                        size += len(chunk)
                length = response.headers.get("content-length")
                if length is not None and response.headers.get("content-encoding", "identity") == "identity":
                    try:
                        complete = int(length) == size
                    except ValueError:
                        complete = False
                    if not complete:
                        raise GatewayError("download_incomplete", "Result download is incomplete", retryable=True)
                return
        raise GatewayError("upstream_protocol_error", "Too many result download redirects")

    async def download(self, task: RemoteTask, path: Path) -> RemoteTask:
        """下载错误或签名失效时查询原任务，再对可恢复错误执行有界重试。"""
        for attempt in range(3):
            if task.files[0].status != "completed" or not task.files[0].result_url:
                raise GatewayError("result_not_ready", "The result is not ready", 409)
            try:
                await self._download_once(task.files[0].result_url, path)
                await asyncio.to_thread(validate_artifact, path, task.output_formats)
                return task
            except (httpx.HTTPError, GatewayError) as exc:
                error = (
                    exc
                    if isinstance(exc, GatewayError)
                    else GatewayError("download_interrupted", "Result download was interrupted", retryable=True)
                )
                if attempt == 2 or (not error.retryable and error.upstream_status not in {401, 403, 404}):
                    raise error from exc
                task = await self.query(task)
                await asyncio.sleep(self.retry_delay * (attempt + 1))
        raise AssertionError("unreachable")


__all__ = ["TaskClient", "validate_artifact"]
