# Copyright (c) Opendatalab. All rights reserved.
"""DocumentParser implementation for the MinerU v1 REST API.

``MinerUApiParser`` extends ``DocumentParser`` and talks to a v1 API
server (local or cloud).  It exposes API tier semantics directly and
does not expose parser backend selection.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import html
import io
import ipaddress
import json
import logging
import math
import os
import random
import re
import time
import tempfile
import zipfile
from collections.abc import Callable, Iterator, AsyncIterator
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Literal, cast
from urllib.parse import unquote, urlparse, urljoin

import httpx
from docvortex.assets import validate_image_sidecar_path

from ..types import PageInfo, Tier, validate_tier
from .base import DocumentParser, ParseResult
from .page_range import normalize_page_range_input

_POLL_INTERVAL_SECONDS = 1
_POLL_MAX_ATTEMPTS = 3600
_TERMINAL_JOB_STATUSES = {"completed", "partial", "failed", "canceled"}
_TRANSPORT_MAX_ATTEMPTS = 3
_TRANSPORT_RETRY_BASE_DELAY_SECONDS = 0.25
_TRANSPORT_RETRY_MAX_DELAY_SECONDS = 2.0
_HTTP_RETRY_MAX_DELAY_SECONDS = 60.0
_RETRYABLE_HTTP_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

_VISUAL_HTML_BLOCK_TYPES = frozenset({"image_body", "table_body", "chart_body"})
_HTML_IMAGE_SOURCE_RE = re.compile(
    r"(?P<prefix><img\b(?:[^>\"']|\"[^\"]*\"|'[^']*')*?\s+src\s*=\s*)"
    r"(?:(?P<quote>[\"'])(?P<quoted>.*?)(?P=quote)|(?P<unquoted>[^\s>]+))",
    re.IGNORECASE | re.DOTALL,
)

logger = logging.getLogger("mineru.api_client")

_RequestMethod = Literal["GET", "POST", "PUT"]
ApiTaskStatus = Literal["waiting_upload", "queued", "running", "completed", "partial", "failed", "canceled"]
ApiJobStatus = ApiTaskStatus
_JOB_STATUSES = {"waiting_upload", "queued", "running", *_TERMINAL_JOB_STATUSES}


def _notify_job_status(job: dict[str, Any], callback: Callable[[ApiJobStatus], None] | None) -> None:
    """通知观察者真实任务状态，隔离回调错误并忽略未知的扩展状态。"""
    status = job.get("status")
    if callback is None or not isinstance(status, str) or status not in _JOB_STATUSES:
        return
    try:
        callback(cast(ApiJobStatus, status))
    except Exception:
        logger.exception("Parse job status callback failed")


def _notify_job_duration(job: dict[str, Any], callback: Callable[[float], None] | None) -> None:
    """下载前通知单文件既有 duration_ms；沿用含导出打包的语义，并隔离缺失值和回调错误。"""
    if callback is None or job.get("status") not in {"completed", "partial"}:
        return
    files = job.get("files")
    if not isinstance(files, list) or len(files) != 1 or not isinstance(files[0], dict):
        return
    info = files[0].get("parse")
    value = info.get("duration_ms") if isinstance(info, dict) else None
    if type(value) not in (int, float):
        return
    try:
        duration = float(value)
    except OverflowError:
        return
    if not math.isfinite(duration) or duration < 0:
        return
    try:
        callback(duration)
    except Exception:
        logger.exception("Parse job duration callback failed")


class _APITransportError(Exception):
    def __init__(self, *, stage: str, method: _RequestMethod, attempts: int, cause: httpx.TransportError) -> None:
        self.stage = stage
        self.method = method
        self.attempts = attempts
        self.cause = cause
        self.timed_out = isinstance(cause, httpx.TimeoutException)
        super().__init__(f"API transport failed during {stage} after {attempts} attempt(s) ({type(cause).__name__})")


class MinerUApiParser(DocumentParser):
    """通过唯一 Tasks 协议解析文件；同机也先使用 hash 预检再按需上传。"""

    DEFAULT_API_URL = "https://mineru.net/api"

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        tier: Tier | None = None,
        ocr_mode: Literal["auto", "txt", "ocr"] | None = None,
        include_images: bool = False,
        include_model_output: bool = False,
    ) -> None:
        """只保存显式配置，不在构造时联网或读取文件。"""
        if ocr_mode is not None and ocr_mode not in ("auto", "txt", "ocr"):
            raise ValueError(f"Unsupported OCR mode '{ocr_mode}'")
        self._base_url = (api_url or os.environ.get("MINERU_API_URL") or self.DEFAULT_API_URL).rstrip("/")
        self._api_key = (api_key if api_key is not None else os.environ.get("MINERU_API_KEY")) or None
        self._trust_env = should_trust_env_for_url(self._base_url)
        self.tier = validate_tier(tier) if tier is not None else None
        self.ocr_mode = ocr_mode
        self.include_images = include_images
        self.include_model_output = include_model_output

    def _headers(self) -> dict[str, str]:
        """生成仅用于 API 同源地址的认证头。"""
        return {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}

    def _resolve(self, url: str, *, metadata: bool = False, base: str | None = None) -> str:
        """解析链接并保留签名；元数据查询不得转向其他来源。"""
        result = urljoin(base or self._base_url + "/", url)
        _http_origin(result)
        if metadata and _http_origin(result) != _http_origin(self._base_url):
            raise _V1APIError("invalid_response", "Task metadata URL must have the API origin")
        return result

    @staticmethod
    def _check(response: httpx.Response) -> dict[str, Any]:
        """保留结构化服务错误及远端认证错误，拒绝非对象响应。"""
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not isinstance(payload, dict):
            raise _V1APIError("invalid_response", "API response must be a JSON object")
        if response.status_code >= 400 or _remote_auth_message(payload):
            _raise_for_http_error(response.status_code, payload, response.text)
        if not payload:
            raise _V1APIError("invalid_response", "API returned an empty or invalid JSON response")
        return payload

    def _payload(self, path: Path, page_range: str) -> dict[str, Any]:
        """固定本次源文件身份，任何部署都使用相同的清单和默认结果包。"""
        if not path.is_file():
            raise FileNotFoundError(path)
        payload: dict[str, Any] = {
            "files": [
                {
                    "client_file_id": "file",
                    "file_name": path.name,
                    "bytes": path.stat().st_size,
                    "sha256sum": _sha256_file(path),
                    "page_range": page_range,
                }
            ],
            "output_formats": ["default"],
            "include_model_output": self.include_model_output,
        }
        if self.tier is not None:
            payload["tier"] = self.tier
        if self.ocr_mode is not None:
            payload["ocr_mode"] = self.ocr_mode
        return payload

    def _snapshot(self, response: httpx.Response, expected: dict[str, Any] | None = None) -> dict[str, Any]:
        """校验稳定任务身份、唯一文件映射和产物选项，不依赖文件名或数组位置。"""
        task = self._check(response)
        files = task.get("files")
        if (
            not isinstance(task.get("task_id"), str)
            or not task["task_id"]
            or not isinstance(task.get("status"), str)
            or task.get("status") not in _JOB_STATUSES
            or task.get("output_formats") != ["default"]
            or task.get("include_model_output", False) != self.include_model_output
            or not isinstance(files, list)
            or len(files) != 1
            or not isinstance(files[0], dict)
            or files[0].get("client_file_id") != "file"
            or (expected is not None and task["task_id"] != expected["task_id"])
        ):
            raise _V1APIError("invalid_response", "Inconsistent Tasks response")
        item = files[0]
        if item.get("upload_status") not in ("pending", "ready", "expired") or item.get("status") not in (
            "waiting_upload",
            "queued",
            "running",
            "completed",
            "failed",
            "canceled",
        ):
            raise _V1APIError("invalid_response", "Invalid task file state")
        if not isinstance(task.get("status_url"), str):
            raise _V1APIError("invalid_response", "Task status URL is missing")
        task["status_url"] = self._resolve(task["status_url"], metadata=True)
        if files[0].get("status") == "completed" and not isinstance(files[0].get("result_url"), str):
            raise _V1APIError("invalid_response", "Completed file is missing result_url")
        return task

    def parse(
        self,
        path: str | Path,
        *,
        page_range: str = "",
        status_callback: Callable[[ApiTaskStatus], None] | None = None,
        duration_callback: Callable[[float], None] | None = None,
    ) -> ParseResult:
        """创建单文件任务、按需上传并下载完整 ZIP，回调对应真实服务端阶段。"""
        file_path = Path(path)
        payload = self._payload(file_path, normalize_page_range_input(page_range))
        with httpx.Client(timeout=httpx.Timeout(120, connect=30), trust_env=self._trust_env, follow_redirects=False) as cli:
            response = _request_once(
                cli,
                "POST",
                f"{self._base_url}/v1/tasks",
                stage="task submission",
                headers=self._headers(),
                json_body=payload,
            )
            task = self._snapshot(response)
            _notify_job_status(task, status_callback)
            task = self._upload_input(cli, task, file_path, status_callback, size=payload["files"][0]["bytes"])
            for _ in range(_POLL_MAX_ATTEMPTS):
                if task["status"] in _TERMINAL_JOB_STATUSES:
                    break
                time.sleep(_POLL_INTERVAL_SECONDS)
                task = self._query(cli, task)
                _notify_job_status(task, status_callback)
            if task["status"] not in _TERMINAL_JOB_STATUSES:
                raise _V1APIError("timeout", f"Task {task['task_id']} did not complete within timeout")
            _raise_for_terminal_job_error(task)
            if task["files"][0].get("status") != "completed":
                raise _V1APIError("parse_failed", "Task did not complete its input file")
            _notify_job_duration(task, duration_callback)
            with tempfile.TemporaryDirectory(prefix="mineru-api-client-") as directory:
                target = Path(directory) / "result.zip"
                self._download_task(cli, task, target)
                return _parse_result_from_zip_bytes(
                    target, include_images=self.include_images, include_model_output=self.include_model_output
                )

    def _query(self, cli: httpx.Client, task: dict[str, Any]) -> dict[str, Any]:
        """按服务返回的同源状态链接查询，保留元数据请求的有限重试。"""
        response = _request_with_retry(
            cli,
            "GET",
            task["status_url"],
            stage="task polling",
            headers=self._headers(),
        )
        return self._snapshot(response, task)

    def _upload_input(
        self,
        cli: httpx.Client,
        task: dict[str, Any],
        path: Path,
        callback: Callable[[ApiTaskStatus], None] | None,
        *,
        size: int,
    ) -> dict[str, Any]:
        """首次上传失败先核实是否已确认，再重新打开文件流；缓存命中完全不传字节。"""
        for attempt in range(1, _TRANSPORT_MAX_ATTEMPTS + 1):
            if task["files"][0].get("upload_status") != "pending":
                return task
            links = task.get("uploads", [])
            link = next((item for item in links if isinstance(item, dict) and item.get("client_file_id") == "file"), None)
            if not link or link.get("upload_method") != "PUT":
                raise _V1APIError("invalid_response", "Missing task upload link")
            url = self._resolve(link["upload_url"])
            headers = _same_origin_upload_headers(self._base_url, url, link.get("upload_headers", {}), self._headers())
            headers = dict(httpx.Headers(headers))
            headers.pop("transfer-encoding", None)
            headers["content-length"] = str(size)
            try:
                response = _request_once(
                    cli,
                    "PUT",
                    url,
                    stage="upload content",
                    headers=headers,
                    content=_file_chunks(path),
                )
                if response.status_code >= 400:
                    self._check(response)
            except (_APITransportError, _V1APIError) as exc:
                task = self._query(cli, task)
                _notify_job_status(task, callback)
                if task["files"][0].get("upload_status") != "pending":
                    return task
                if isinstance(exc, _V1APIError) and response.status_code not in {408, 429, 500, 502, 503, 504}:
                    raise
                if attempt == _TRANSPORT_MAX_ATTEMPTS:
                    raise
                time.sleep(_transport_retry_delay(attempt))
                continue
            task = self._query(cli, task)
            _notify_job_status(task, callback)
            if task["files"][0].get("upload_status") != "pending":
                return task
        raise _V1APIError("upload_failed", "Upload was not confirmed")

    def _download_task(self, cli: httpx.Client, task: dict[str, Any], target: Path) -> None:
        """流式下载和校验完整包；重定向逐跳选择认证，签名失效后重新查询任务。"""
        for attempt in range(1, _TRANSPORT_MAX_ATTEMPTS + 1):
            url = self._resolve(task["files"][0]["result_url"])
            try:
                for _ in range(6):
                    headers = _same_origin_upload_headers(self._base_url, url, {}, self._headers())
                    with cli.stream("GET", url, headers=headers, follow_redirects=False) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise _V1APIError("invalid_response", "Download redirect is missing Location")
                            url = self._resolve(location, base=url)
                            continue
                        if response.status_code >= 400:
                            response.read()
                            self._check(response)
                        received = 0
                        with target.open("wb") as output:
                            for chunk in response.iter_bytes():
                                output.write(chunk)
                                received += len(chunk)
                        _check_download_length(response, received)
                    with zipfile.ZipFile(target) as archive:
                        if archive.testzip() is not None:
                            raise zipfile.BadZipFile("Corrupt ZIP member")
                    return
                raise _V1APIError("invalid_response", "Too many download redirects")
            except (httpx.TransportError, zipfile.BadZipFile, _V1APIError) as exc:
                target.unlink(missing_ok=True)
                if attempt == _TRANSPORT_MAX_ATTEMPTS:
                    if isinstance(exc, httpx.TransportError):
                        raise _APITransportError(stage="output download", method="GET", attempts=attempt, cause=exc) from exc
                    if isinstance(exc, zipfile.BadZipFile):
                        raise _V1APIError("invalid_zip_output", "Invalid or incomplete result ZIP") from exc
                    raise
                time.sleep(_transport_retry_delay(attempt))
                task = self._query(cli, task)
                _raise_for_terminal_job_error(task)

    def get_usage(self) -> dict[str, Any]:
        """查询现有用量接口，保持同步包装与 Doclib 的返回结构。"""
        with httpx.Client(timeout=httpx.Timeout(30, connect=10), trust_env=self._trust_env) as cli:
            response = _request_with_retry(
                cli,
                "GET",
                f"{self._base_url}/v1/usage",
                stage="usage query",
                headers=self._headers(),
            )
            return self._check(response)

    async def parse_async(
        self,
        path: str | Path,
        *,
        page_range: str = "",
        status_callback: Callable[[ApiTaskStatus], None] | None = None,
        duration_callback: Callable[[float], None] | None = None,
    ) -> ParseResult:
        """创建单文件任务、按需上传并下载完整 ZIP，回调对应真实服务端阶段。"""
        file_path = Path(path)
        payload = await asyncio.to_thread(self._payload, file_path, normalize_page_range_input(page_range))
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(120, connect=30), trust_env=self._trust_env, follow_redirects=False
        ) as cli:
            response = await _async_request_once(
                cli,
                "POST",
                f"{self._base_url}/v1/tasks",
                stage="task submission",
                headers=self._headers(),
                json_body=payload,
            )
            task = self._snapshot(response)
            _notify_job_status(task, status_callback)
            task = await self._async_upload_input(cli, task, file_path, status_callback, size=payload["files"][0]["bytes"])
            for _ in range(_POLL_MAX_ATTEMPTS):
                if task["status"] in _TERMINAL_JOB_STATUSES:
                    break
                await asyncio.sleep(_POLL_INTERVAL_SECONDS)
                task = await self._async_query(cli, task)
                _notify_job_status(task, status_callback)
            if task["status"] not in _TERMINAL_JOB_STATUSES:
                raise _V1APIError("timeout", f"Task {task['task_id']} did not complete within timeout")
            _raise_for_terminal_job_error(task)
            if task["files"][0].get("status") != "completed":
                raise _V1APIError("parse_failed", "Task did not complete its input file")
            _notify_job_duration(task, duration_callback)
            with tempfile.TemporaryDirectory(prefix="mineru-api-client-") as directory:
                target = Path(directory) / "result.zip"
                await self._async_download_task(cli, task, target)
                return _parse_result_from_zip_bytes(
                    target, include_images=self.include_images, include_model_output=self.include_model_output
                )

    async def _async_query(self, cli: httpx.AsyncClient, task: dict[str, Any]) -> dict[str, Any]:
        """按服务返回的同源状态链接查询，保留元数据请求的有限重试。"""
        response = await _async_request_with_retry(
            cli,
            "GET",
            task["status_url"],
            stage="task polling",
            headers=self._headers(),
        )
        return self._snapshot(response, task)

    async def _async_upload_input(
        self,
        cli: httpx.AsyncClient,
        task: dict[str, Any],
        path: Path,
        callback: Callable[[ApiTaskStatus], None] | None,
        *,
        size: int,
    ) -> dict[str, Any]:
        """首次上传失败先核实是否已确认，再重新打开文件流；缓存命中完全不传字节。"""
        for attempt in range(1, _TRANSPORT_MAX_ATTEMPTS + 1):
            if task["files"][0].get("upload_status") != "pending":
                return task
            links = task.get("uploads", [])
            link = next((item for item in links if isinstance(item, dict) and item.get("client_file_id") == "file"), None)
            if not link or link.get("upload_method") != "PUT":
                raise _V1APIError("invalid_response", "Missing task upload link")
            url = self._resolve(link["upload_url"])
            headers = _same_origin_upload_headers(self._base_url, url, link.get("upload_headers", {}), self._headers())
            headers = dict(httpx.Headers(headers))
            headers.pop("transfer-encoding", None)
            headers["content-length"] = str(size)
            try:
                response = await _async_request_once(
                    cli,
                    "PUT",
                    url,
                    stage="upload content",
                    headers=headers,
                    content=_async_file_chunks(path),
                )
                if response.status_code >= 400:
                    self._check(response)
            except (_APITransportError, _V1APIError) as exc:
                task = await self._async_query(cli, task)
                _notify_job_status(task, callback)
                if task["files"][0].get("upload_status") != "pending":
                    return task
                if isinstance(exc, _V1APIError) and response.status_code not in {408, 429, 500, 502, 503, 504}:
                    raise
                if attempt == _TRANSPORT_MAX_ATTEMPTS:
                    raise
                await asyncio.sleep(_transport_retry_delay(attempt))
                continue
            task = await self._async_query(cli, task)
            _notify_job_status(task, callback)
            if task["files"][0].get("upload_status") != "pending":
                return task
        raise _V1APIError("upload_failed", "Upload was not confirmed")

    async def _async_download_task(self, cli: httpx.AsyncClient, task: dict[str, Any], target: Path) -> None:
        """流式下载和校验完整包；重定向逐跳选择认证，签名失效后重新查询任务。"""
        for attempt in range(1, _TRANSPORT_MAX_ATTEMPTS + 1):
            url = self._resolve(task["files"][0]["result_url"])
            try:
                for _ in range(6):
                    headers = _same_origin_upload_headers(self._base_url, url, {}, self._headers())
                    async with cli.stream("GET", url, headers=headers, follow_redirects=False) as response:
                        if response.status_code in {301, 302, 303, 307, 308}:
                            location = response.headers.get("location")
                            if not location:
                                raise _V1APIError("invalid_response", "Download redirect is missing Location")
                            url = self._resolve(location, base=url)
                            continue
                        if response.status_code >= 400:
                            await response.aread()
                            self._check(response)
                        received = 0
                        with target.open("wb") as output:
                            async for chunk in response.aiter_bytes():
                                await asyncio.to_thread(output.write, chunk)
                                received += len(chunk)
                        _check_download_length(response, received)
                    with zipfile.ZipFile(target) as archive:
                        if archive.testzip() is not None:
                            raise zipfile.BadZipFile("Corrupt ZIP member")
                    return
                raise _V1APIError("invalid_response", "Too many download redirects")
            except (httpx.TransportError, zipfile.BadZipFile, _V1APIError) as exc:
                target.unlink(missing_ok=True)
                if attempt == _TRANSPORT_MAX_ATTEMPTS:
                    if isinstance(exc, httpx.TransportError):
                        raise _APITransportError(stage="output download", method="GET", attempts=attempt, cause=exc) from exc
                    if isinstance(exc, zipfile.BadZipFile):
                        raise _V1APIError("invalid_zip_output", "Invalid or incomplete result ZIP") from exc
                    raise
                await asyncio.sleep(_transport_retry_delay(attempt))
                task = await self._async_query(cli, task)
                _raise_for_terminal_job_error(task)

    async def get_usage_async(self) -> dict[str, Any]:
        """查询现有用量接口，保持同步包装与 Doclib 的返回结构。"""
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10), trust_env=self._trust_env) as cli:
            response = await _async_request_with_retry(
                cli,
                "GET",
                f"{self._base_url}/v1/usage",
                stage="usage query",
                headers=self._headers(),
            )
            return self._check(response)


def _request_once(
    client: httpx.Client,
    method: _RequestMethod,
    url: str,
    *,
    stage: str,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
    content: bytes | Iterator[bytes] | AsyncIterator[bytes] | None = None,
) -> httpx.Response:
    return _request(
        client,
        method,
        url,
        stage=stage,
        headers=headers,
        json_body=json_body,
        content=content,
        max_attempts=1,
    )


def _request_with_retry(
    client: httpx.Client,
    method: _RequestMethod,
    url: str,
    *,
    stage: str,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
    content: bytes | Iterator[bytes] | AsyncIterator[bytes] | None = None,
) -> httpx.Response:
    return _request(
        client,
        method,
        url,
        stage=stage,
        headers=headers,
        json_body=json_body,
        content=content,
        max_attempts=_TRANSPORT_MAX_ATTEMPTS,
    )


def _request(
    client: httpx.Client,
    method: _RequestMethod,
    url: str,
    *,
    stage: str,
    headers: dict[str, str],
    json_body: dict[str, Any] | None,
    content: bytes | Iterator[bytes] | AsyncIterator[bytes] | None,
    max_attempts: int,
) -> httpx.Response:
    """仅对安全 GET 重试临时 HTTP 错误，创建与上传沿用各自的一次请求边界。"""
    for attempt in range(1, max_attempts + 1):
        try:
            response = _send_request(client, method, url, headers=headers, json_body=json_body, content=content)
            delay = _http_retry_delay(response, attempt) if method == "GET" and attempt < max_attempts else None
            if delay is None:
                return response
            response.close()
            time.sleep(delay)
        except httpx.TransportError as exc:
            if attempt >= max_attempts:
                raise _APITransportError(stage=stage, method=method, attempts=attempt, cause=exc) from exc
            logger.warning(
                "Retrying API transport request: stage=%s method=%s next_attempt=%s/%s error=%s",
                stage,
                method,
                attempt + 1,
                max_attempts,
                type(exc).__name__,
            )
            time.sleep(_transport_retry_delay(attempt))
    raise AssertionError("API transport retry loop exited unexpectedly")


async def _async_request_once(
    client: httpx.AsyncClient,
    method: _RequestMethod,
    url: str,
    *,
    stage: str,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
    content: bytes | Iterator[bytes] | AsyncIterator[bytes] | None = None,
) -> httpx.Response:
    return await _async_request(
        client,
        method,
        url,
        stage=stage,
        headers=headers,
        json_body=json_body,
        content=content,
        max_attempts=1,
    )


async def _async_request_with_retry(
    client: httpx.AsyncClient,
    method: _RequestMethod,
    url: str,
    *,
    stage: str,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
    content: bytes | Iterator[bytes] | AsyncIterator[bytes] | None = None,
) -> httpx.Response:
    return await _async_request(
        client,
        method,
        url,
        stage=stage,
        headers=headers,
        json_body=json_body,
        content=content,
        max_attempts=_TRANSPORT_MAX_ATTEMPTS,
    )


async def _async_request(
    client: httpx.AsyncClient,
    method: _RequestMethod,
    url: str,
    *,
    stage: str,
    headers: dict[str, str],
    json_body: dict[str, Any] | None,
    content: bytes | Iterator[bytes] | AsyncIterator[bytes] | None,
    max_attempts: int,
) -> httpx.Response:
    """仅对安全 GET 重试临时 HTTP 错误，创建与上传沿用各自的一次请求边界。"""
    for attempt in range(1, max_attempts + 1):
        try:
            response = await _async_send_request(client, method, url, headers=headers, json_body=json_body, content=content)
            delay = _http_retry_delay(response, attempt) if method == "GET" and attempt < max_attempts else None
            if delay is None:
                return response
            await response.aclose()
            await asyncio.sleep(delay)
        except httpx.TransportError as exc:
            if attempt >= max_attempts:
                raise _APITransportError(stage=stage, method=method, attempts=attempt, cause=exc) from exc
            logger.warning(
                "Retrying API transport request: stage=%s method=%s next_attempt=%s/%s error=%s",
                stage,
                method,
                attempt + 1,
                max_attempts,
                type(exc).__name__,
            )
            await asyncio.sleep(_transport_retry_delay(attempt))
    raise AssertionError("API transport retry loop exited unexpectedly")


def _send_request(
    client: httpx.Client,
    method: _RequestMethod,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any] | None,
    content: bytes | Iterator[bytes] | AsyncIterator[bytes] | None,
) -> httpx.Response:
    if method == "GET":
        return client.get(url, headers=headers)
    if method == "PUT":
        return client.put(url, headers=headers, content=content or b"")
    if json_body is None:
        return client.post(url, headers=headers)
    return client.post(url, headers=headers, json=json_body)


async def _async_send_request(
    client: httpx.AsyncClient,
    method: _RequestMethod,
    url: str,
    *,
    headers: dict[str, str],
    json_body: dict[str, Any] | None,
    content: bytes | Iterator[bytes] | AsyncIterator[bytes] | None,
) -> httpx.Response:
    if method == "GET":
        return await client.get(url, headers=headers)
    if method == "PUT":
        return await client.put(url, headers=headers, content=content or b"")
    if json_body is None:
        return await client.post(url, headers=headers)
    return await client.post(url, headers=headers, json=json_body)


def _transport_retry_delay(attempt: int) -> float:
    exponential = min(
        _TRANSPORT_RETRY_MAX_DELAY_SECONDS,
        _TRANSPORT_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
    )
    return exponential * (0.5 + random.random())


def _http_retry_delay(response: httpx.Response, attempt: int) -> float | None:
    """解析秒数或 HTTP 日期的 Retry-After；超过等待预算时返回原错误，不提前重试。"""
    if response.status_code not in _RETRYABLE_HTTP_STATUSES:
        return None
    value = response.headers.get("retry-after", "").strip()
    if value.isascii() and value.isdigit():
        # 避免超长数字转换，并保留服务端要求的最短等待时间。
        seconds = value.lstrip("0") or "0"
        delay = float(seconds) if len(seconds) <= 3 else _HTTP_RETRY_MAX_DELAY_SECONDS + 1
    else:
        try:
            stamp = parsedate_to_datetime(value)
            if stamp.tzinfo is None:
                raise ValueError("Retry-After date must have a timezone")
            delay = max(0.0, stamp.timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            delay = _transport_retry_delay(attempt)
    return delay if delay <= _HTTP_RETRY_MAX_DELAY_SECONDS else None


def _check_download_length(response: httpx.Response, received: int) -> None:
    """未压缩响应必须完整接收声明的长度，不能交付恰好仍可解压的截断包。"""
    raw = response.headers.get("content-length")
    if raw is None or response.headers.get("content-encoding", "identity") != "identity":
        return
    try:
        complete = int(raw) == received
    except ValueError:
        complete = False
    if not complete:
        raise httpx.RemoteProtocolError("Incomplete result download")


def _file_chunks(path: Path) -> Iterator[bytes]:
    """每次传输重新打开文件，避免重试复用已消费的流。"""
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            yield chunk


async def _async_file_chunks(path: Path) -> AsyncIterator[bytes]:
    """异步上传逐块读取文件，不把整个输入缓冲到内存。"""
    with path.open("rb") as source:
        while chunk := await asyncio.to_thread(source.read, 1024 * 1024):
            yield chunk


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _extract_model_output_from_archive(archive: zipfile.ZipFile) -> Any | None:
    """从已打开的 zip 读取原始模型输出。"""
    names = set(archive.namelist())
    local_name = "model_output.json"
    if local_name in names:
        model_output_name = local_name
    else:
        legacy_names = sorted(name for name in names if Path(name).name.endswith("_model.json"))
        if not legacy_names:
            return None
        if len(legacy_names) > 1:
            available = ", ".join(legacy_names)
            raise _V1APIError("invalid_model_output", f"ZIP output contained multiple model outputs: {available}")
        model_output_name = legacy_names[0]

    from ..types import ModelJson

    raw = archive.read(model_output_name)
    try:
        model = ModelJson.from_json(raw)
        from ..integrations.docvortex import validate_mineru_metadata

        validate_mineru_metadata(model)
        return model
    except (ValueError, TypeError) as exc:
        raise _V1APIError("invalid_model_output", f"Invalid Model JSON document: {exc}") from exc


def _parse_result_from_zip_bytes(
    zip_bytes: bytes | Path,
    *,
    include_images: bool,
    include_model_output: bool,
) -> ParseResult:
    """解析自包含 zip 包：读取 middle_json、图片 sidecar 和可选 model_output。"""
    try:
        with zipfile.ZipFile(zip_bytes if isinstance(zip_bytes, Path) else io.BytesIO(zip_bytes)) as archive:
            mid_json = _read_middle_json_from_zip(archive)
            if include_images:
                images = _read_image_sidecars_from_zip(archive, mid_json)
                if images:
                    _inline_image_sidecars(mid_json, images)
            result = _parse_result_from_middle_json(mid_json)
            if include_model_output:
                model_output = _extract_model_output_from_archive(archive)
                if model_output is not None:
                    result._model_output = model_output
            return result
    except zipfile.BadZipFile as exc:
        raise _V1APIError("invalid_zip_output", "zip output is not a valid ZIP archive") from exc
    except ValueError as exc:
        raise _V1APIError("invalid_image_sidecar", str(exc)) from exc


def _middle_json_zip_candidates() -> list[str]:
    """列出当前客户端支持的 API zip middle_json 文件名。"""
    return [
        "middle_json.json",
        "layout.json",
    ]


def _read_middle_json_from_zip(archive: zipfile.ZipFile) -> dict[str, Any]:
    """从本地 api_server 或 remote server 的 zip 中读取 middle_json。"""
    names = set(archive.namelist())
    for candidate in _middle_json_zip_candidates():
        if candidate not in names:
            continue
        raw = archive.read(candidate)
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise _V1APIError("invalid_middle_json_output", f"middle_json output is not valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise _V1APIError(
                "invalid_middle_json_output",
                "middle_json output must be a JSON object with pages or pdf_info",
            )
        return loaded
    available = ", ".join(sorted(names)) or "none"
    raise _V1APIError("missing_middle_json_output", f"ZIP output did not contain middle_json; available entries: {available}")


def _collect_image_paths_from_middle_json(value: Any) -> set[str]:
    """收集直接图片与视觉 HTML 图片引用，保留代码正文中的图片字面量。"""
    paths: set[str] = set()
    if isinstance(value, dict):
        content = value.get("content")
        if value.get("type") in _VISUAL_HTML_BLOCK_TYPES and isinstance(content, str):
            for match in _HTML_IMAGE_SOURCE_RE.finditer(content):
                if (path := _markup_image_path(match)) is not None:
                    paths.add(path)
        for key, item in value.items():
            if key in {"image_path", "img_path"} and isinstance(item, str) and item:
                paths.add(item)
            else:
                paths.update(_collect_image_paths_from_middle_json(item))
    elif isinstance(value, list):
        for item in value:
            paths.update(_collect_image_paths_from_middle_json(item))
    return paths


def _markup_image_path(match: re.Match[str]) -> str | None:
    """解释 HTML 图片的相对素材路径，已有 data URI 和网络图片不读取。"""
    source = html.unescape(match.group("quoted") if match.group("quote") else match.group("unquoted")).strip()
    if not source or source.startswith("#") or urlparse(source).scheme.lower() in {"data", "http", "https"}:
        return None
    return validate_image_sidecar_path(unquote(source))


def _read_image_sidecars_from_zip(archive: zipfile.ZipFile, mid_json: dict[str, Any]) -> dict[str, bytes]:
    """从 zip 读取 middle_json 实际引用的图片 sidecar，并复用统一路径安全校验。"""
    archive_names = set(archive.namelist())
    images: dict[str, bytes] = {}
    for image_path in sorted(_collect_image_paths_from_middle_json(mid_json)):
        safe_image_path = validate_image_sidecar_path(image_path)
        candidates = [safe_image_path]
        if not safe_image_path.startswith("images/"):
            candidates.append(f"images/{safe_image_path}")
        for candidate in candidates:
            if candidate in archive_names:
                images[safe_image_path] = archive.read(candidate)
                break
        else:
            raise _V1APIError("invalid_image_sidecar", f"Missing image sidecar in ZIP: {safe_image_path}")
    return images


_IMAGE_DATA_URI_SUBTYPES: dict[str, str] = {
    "jpg": "jpeg",
    "jpeg": "jpeg",
    "png": "png",
    "gif": "gif",
    "webp": "webp",
    "bmp": "bmp",
    "tiff": "tiff",
    "svg": "svg+xml",
}


def _encode_image_data_uri(image_bytes: bytes, image_path: str) -> str:
    """把图片字节编码为 data URI，MIME subtype 从 sidecar 扩展名推断。"""
    ext = Path(image_path).suffix.lstrip(".").lower()
    subtype = _IMAGE_DATA_URI_SUBTYPES.get(ext)
    if subtype is None:
        raise _V1APIError("invalid_image_sidecar", f"Unsupported image sidecar extension: {image_path}")
    return f"data:image/{subtype};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def _inline_image_sidecars(mid_json: Any, images: dict[str, bytes]) -> None:
    """递归把 middle_json 中引用的图片 sidecar 路径替换为内联 data URI。"""
    if isinstance(mid_json, dict):
        image_path = mid_json.get("image_path") or mid_json.get("img_path")
        if isinstance(image_path, str) and image_path:
            safe_path = validate_image_sidecar_path(image_path)
            image_bytes = images.get(safe_path)
            if image_bytes is None and not safe_path.startswith("images/"):
                image_bytes = images.get(f"images/{safe_path}")
            if image_bytes is not None:
                mid_json["image_base64"] = _encode_image_data_uri(image_bytes, safe_path)
                mid_json.pop("image_path", None)
                mid_json.pop("img_path", None)
        content = mid_json.get("content")
        if mid_json.get("type") in _VISUAL_HTML_BLOCK_TYPES and isinstance(content, str):

            def replace_source(match: re.Match[str]) -> str:
                """只替换 src 的值，保留 HTML 属性、空白与引号。"""
                path = _markup_image_path(match)
                if path is None:
                    return match.group(0)
                data_uri = _encode_image_data_uri(images[path], path)
                group = "quoted" if match.group("quote") else "unquoted"
                start, end = match.span(group)
                original = match.group(0)
                return original[: start - match.start()] + data_uri + original[end - match.start() :]

            mid_json["content"] = _HTML_IMAGE_SOURCE_RE.sub(replace_source, content)
        for value in mid_json.values():
            _inline_image_sidecars(value, images)
    elif isinstance(mid_json, list):
        for item in mid_json:
            _inline_image_sidecars(item, images)


def _same_origin_upload_headers(
    base_url: str,
    upload_url: str,
    upload_headers: object,
    auth_headers: dict[str, str],
) -> dict[str, str]:
    """仅为 API 同源上传合并鉴权头，避免向外部预签名地址泄露密钥。"""
    if not isinstance(upload_headers, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in upload_headers.items()
    ):
        raise _V1APIError("invalid_response", "upload_headers must be a string mapping")
    headers = dict(upload_headers)
    if _http_origin(base_url) == _http_origin(upload_url):
        headers.update(auth_headers)
    elif any(key.lower() == "authorization" and value.strip() in auth_headers.values() for key, value in headers.items()):
        raise _V1APIError("invalid_response", "Refusing to forward the API key to another origin")
    return headers


def _http_origin(url: str) -> tuple[str, str, int | None]:
    """将 HTTP URL 规范为 scheme、host、effective-port 三元组。"""
    parsed = urlparse(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise _V1APIError("invalid_response", "Invalid HTTP URL in API response") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
    ):
        raise _V1APIError("invalid_response", "Invalid HTTP URL in API response")
    effective_port = port if port is not None else (443 if parsed.scheme == "https" else 80)
    return parsed.scheme.lower(), parsed.hostname.lower(), effective_port


def should_trust_env_for_url(url: str) -> bool:
    return not _is_local_network_url(url)


def _is_local_network_url(url: str) -> bool:
    parsed = urlparse(url if "://" in url else f"http://{url}")
    host = parsed.hostname
    if not host:
        return False
    normalized = host.lower()
    if normalized == "localhost" or normalized.endswith(".localhost"):
        return True
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return False
    return address.is_loopback or address.is_private or address.is_link_local


def _check_download_response(r: httpx.Response) -> None:
    if r.status_code < 400:
        return
    preview = r.text[:500]
    raise _V1APIError("download_failed", f"HTTP {r.status_code}: {preview}")


def _pages_from_middle_json(mid_json: dict[str, Any] | None) -> list[PageInfo]:
    if mid_json is None:
        return []
    return _parse_result_from_middle_json(mid_json).pages


def _parse_result_from_middle_json(mid_json: dict[str, Any]) -> ParseResult:
    """严格恢复共享 Middle JSON 2.0，历史缓存兼容仅属于 Doclib 读取边界。"""
    if not isinstance(mid_json, dict):
        raise _V1APIError("invalid_middle_json_output", "middle_json output must be a JSON object")
    try:
        return ParseResult.from_dict(mid_json)
    except ValueError as exc:
        raise _V1APIError("invalid_middle_json_output", str(exc)) from exc


def _raise_for_terminal_job_error(job: dict[str, Any]) -> None:
    if job.get("status") not in ("failed", "canceled"):
        return
    error = job.get("error")
    if isinstance(error, dict):
        code = str(error.get("code") or job.get("status"))
        message = str(error.get("message") or error)
        param = str(error["param"]) if error.get("param") is not None else None
        raise _V1APIError(code, message, param=param)

    file_error = _first_failed_file_error(job)
    if file_error is not None:
        code = str(file_error.get("code") or job.get("status"))
        message = str(file_error.get("message") or file_error)
        param = str(file_error["param"]) if file_error.get("param") is not None else None
        raise _V1APIError(code, message, param=param)

    raise _V1APIError(
        str(job.get("status")), f"Parse task {job.get('task_id', '<unknown>')} ended with status {job.get('status')}"
    )


def _raise_for_http_error(status_code: int, data: dict[str, Any], text: str) -> None:
    err = _structured_error(data)
    if err is not None:
        raise _V1APIError(
            str(err.get("code") or "unknown"),
            str(err.get("message") or err),
            param=str(err["param"]) if err.get("param") is not None else None,
        )

    remote_message = _remote_auth_message(data)
    if status_code == 401 or remote_message is not None:
        message = remote_message or "API key invalid or remote authentication failed."
        raise _V1APIError(
            "invalid_api_key",
            f"Remote authentication failed: {message}",
            param="parse_server.remote.api_key",
        )

    raise _V1APIError("http_error", f"HTTP {status_code}: {text[:500]}")


def _structured_error(data: dict[str, Any]) -> dict[str, Any] | None:
    error = data.get("error")
    if isinstance(error, dict):
        return error
    return None


def _remote_auth_message(data: dict[str, Any]) -> str | None:
    msg_code = data.get("msgCode")
    msg = data.get("msg")
    # Official API auth failures may use the legacy msgCode/msg payload instead of {"error": ...}.
    if msg_code == "A0202":
        return str(msg or "user authenticate failed")
    if isinstance(msg, str) and "authenticate failed" in msg.lower():
        return msg
    return None


def _first_failed_file_error(job: dict[str, Any]) -> dict[str, Any] | None:
    files = job.get("files")
    if not isinstance(files, list):
        return None
    for file_result in files:
        if not isinstance(file_result, dict):
            continue
        if file_result.get("status") != "failed":
            continue
        error = file_result.get("error")
        if isinstance(error, dict):
            return error
    return None


class _V1APIError(Exception):
    def __init__(self, code: str, message: str, param: str | None = None) -> None:
        self.code = code
        self.message = message
        self.param = param
        super().__init__(f"[{code}] {message}")
