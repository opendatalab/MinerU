# Copyright (c) Opendatalab. All rights reserved.
"""DocumentParser implementation for the MinerU v1 REST API.

``MinerUApiParser`` extends ``DocumentParser`` and talks to a v1 API
server (local or cloud).  It exposes API tier semantics directly and
does not expose parser backend selection.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import ipaddress
import json
import logging
import os
import random
import time
import zipfile
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlparse

import httpx

from ..filetypes import mime_type_for_extension
from ..types import PageInfo, Tier, validate_tier
from ..utils.image_payload import validate_image_sidecar_path
from .base import DocumentParser, ParseResult

_POLL_INTERVAL_SECONDS = 1
_POLL_MAX_ATTEMPTS = 3600
_TERMINAL_JOB_STATUSES = {"completed", "partial", "failed", "canceled"}
_TRANSPORT_MAX_ATTEMPTS = 3
_TRANSPORT_RETRY_BASE_DELAY_SECONDS = 0.25
_TRANSPORT_RETRY_MAX_DELAY_SECONDS = 2.0

logger = logging.getLogger("mineru.api_client")

_RequestMethod = Literal["GET", "POST", "PUT"]


class _APITransportError(Exception):
    def __init__(self, *, stage: str, method: _RequestMethod, attempts: int, cause: httpx.TransportError) -> None:
        self.stage = stage
        self.method = method
        self.attempts = attempts
        self.cause = cause
        self.timed_out = isinstance(cause, httpx.TimeoutException)
        super().__init__(f"API transport failed during {stage} after {attempts} attempt(s) ({type(cause).__name__})")


class MinerUApiParser(DocumentParser):
    """Parser that delegates to a MinerU v1 API server.

    Works with local-server, LAN, and cloud (mineru.net) deployments::

        # local deployment — uses ``local`` source only when the server advertises it.
        # Start the local server with --allow-local-source to skip upload.
        parser = MinerUApiParser(api_url="http://localhost:8000", tier="medium")

        # cloud (remote)
        parser = MinerUApiParser(
            api_url="https://mineru.net/api", api_key="sk_...",
            tier="high",
        )

        result = parser.parse("report.pdf")
        print(result.markdown())

    Constructor parameters:

    - ``tier`` → v1 ``tier`` (``"flash"`` / ``"medium"`` / ``"high"`` / ``"xhigh"``); ``None`` omits the field
    - ``page_range`` → per-file v1 ``page_range``
    """

    DEFAULT_API_URL = "https://mineru.net/api"

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        tier: Tier | None = None,
        include_images: bool = False,
        include_model_output: bool = False,
    ) -> None:
        self._base_url = (api_url or os.environ.get("MINERU_API_URL") or self.DEFAULT_API_URL).rstrip("/")
        self._api_key = (api_key if api_key is not None else os.environ.get("MINERU_API_KEY")) or None
        self._local = _is_local_network_url(self._base_url)
        self._trust_env = should_trust_env_for_url(self._base_url)
        self._source_features: set[str] | None = None
        self.tier = validate_tier(tier) if tier is not None else None
        self.include_images = include_images
        self.include_model_output = include_model_output

    # ── DocumentParser interface ─────────────────────────────────────

    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        payload = self._build_payload(self._build_source(file_path), page_range)
        job = self._do_parse(payload)
        return self._build_result(job, file_path.name)

    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        payload = self._build_payload(await self._async_build_source(file_path), page_range)
        job = await self._async_do_parse(payload)
        return await self._async_build_result(job, file_path.name)

    # ── payload construction ─────────────────────────────────────────

    def _build_source(self, file_path: Path) -> dict[str, Any]:
        if self._supports_local_source():
            source = {"type": "local", "path": str(file_path)}
        else:
            # upload flow: create → PUT → complete → file_id
            file_id = self._upload(file_path)
            source = {"type": "file_id", "file_id": file_id}
        return source

    async def _async_build_source(self, file_path: Path) -> dict[str, Any]:
        if await self._async_supports_local_source():
            source = {"type": "local", "path": str(file_path)}
        else:
            file_id = await self._async_upload(file_path)
            source = {"type": "file_id", "file_id": file_id}
        return source

    def _build_payload(self, source: dict[str, Any], page_range: str) -> dict[str, Any]:
        file_entry: dict[str, Any] = {"source": source}
        if page_range:
            file_entry["page_range"] = page_range

        payload: dict[str, Any] = {
            "files": [file_entry],
            "output_formats": self._output_formats(),
        }
        if self.tier is not None:
            payload["tier"] = self.tier
        return payload

    def _supports_local_source(self) -> bool:
        if not self._local:
            return False
        return "local" in self._get_source_features()

    async def _async_supports_local_source(self) -> bool:
        if not self._local:
            return False
        return "local" in await self._async_get_source_features()

    def _get_source_features(self) -> set[str]:
        if self._source_features is not None:
            return self._source_features
        with httpx.Client(timeout=httpx.Timeout(30, connect=10), trust_env=self._trust_env) as cli:
            r = _request_with_retry(
                cli,
                "GET",
                f"{self._base_url}/v1/health",
                stage="health discovery",
                headers=self._headers(),
            )
            health = self._check(r)
        self._source_features = _extract_feature_list(health, "sources")
        return self._source_features

    async def _async_get_source_features(self) -> set[str]:
        if self._source_features is not None:
            return self._source_features
        async with httpx.AsyncClient(timeout=httpx.Timeout(30, connect=10), trust_env=self._trust_env) as cli:
            r = await _async_request_with_retry(
                cli,
                "GET",
                f"{self._base_url}/v1/health",
                stage="health discovery",
                headers=self._headers(),
            )
            health = self._check(r)
        self._source_features = _extract_feature_list(health, "sources")
        return self._source_features

    def _output_formats(self) -> list[str]:
        if self.include_model_output or self.include_images:
            return ["zip"]
        return ["middle_json"]

    # ── HTTP helpers ─────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    @staticmethod
    def _is_localhost(base: str) -> bool:
        return _is_local_network_url(base)

    @staticmethod
    def _check(r: Any) -> dict[str, Any]:
        data: dict[str, Any] = {}
        try:
            loaded = r.json()
            if isinstance(loaded, dict):
                data = loaded
        except Exception:
            data = {}
        if r.status_code >= 400:
            _raise_for_http_error(r.status_code, data, str(getattr(r, "text", "")))
        if not data:
            return {}  # e.g. OSS PUT returns 200 with empty body
        if "error" in data:
            err = data["error"]
            if isinstance(err, dict):
                raise _V1APIError(
                    str(err.get("code") or "unknown"),
                    str(err.get("message") or err),
                    param=str(err["param"]) if err.get("param") is not None else None,
                )
            raise _V1APIError("unknown", str(err))
        return data

    # ── upload ───────────────────────────────────────────────────────

    def _upload(self, file_path: Path) -> str:
        size = file_path.stat().st_size
        sha = _sha256_file(file_path)

        with httpx.Client(timeout=httpx.Timeout(120, connect=30), trust_env=self._trust_env) as cli:
            r = _request_once(
                cli,
                "POST",
                f"{self._base_url}/v1/uploads",
                stage="upload creation",
                headers=self._headers(),
                json_body={
                    "filename": file_path.name,
                    "bytes": size,
                    "mime_type": mime_type_for_extension(file_path),
                    "purpose": "parse",
                    "sha256sum": sha,
                },
            )
            resp = self._check(r)
            if resp.get("status") == "completed":
                return resp["file"]["id"]

            upload_url = resp["upload_url"]
            upload_headers = resp.get("upload_headers", {})
            if upload_url.startswith("/"):
                upload_url = f"{self._base_url}{upload_url}"
            with file_path.open("rb") as fh:
                r2 = _request_with_retry(
                    cli,
                    "PUT",
                    upload_url,
                    stage="upload content",
                    headers=upload_headers,
                    content=fh.read(),
                )
            self._check(r2)

            resp3 = self._complete_upload(cli, str(resp["id"]))
            return resp3["file"]["id"]

    async def _async_upload(self, file_path: Path) -> str:
        size = file_path.stat().st_size
        sha = _sha256_file(file_path)

        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30), trust_env=self._trust_env) as cli:
            r = await _async_request_once(
                cli,
                "POST",
                f"{self._base_url}/v1/uploads",
                stage="upload creation",
                headers=self._headers(),
                json_body={
                    "filename": file_path.name,
                    "bytes": size,
                    "mime_type": mime_type_for_extension(file_path),
                    "purpose": "parse",
                    "sha256sum": sha,
                },
            )
            resp = self._check(r)
            if resp.get("status") == "completed":
                return resp["file"]["id"]

            upload_url = resp["upload_url"]
            upload_headers = resp.get("upload_headers", {})
            if upload_url.startswith("/"):
                upload_url = f"{self._base_url}{upload_url}"
            data = file_path.read_bytes()
            r2 = await _async_request_with_retry(
                cli,
                "PUT",
                upload_url,
                stage="upload content",
                headers=upload_headers,
                content=data,
            )
            self._check(r2)

            resp3 = await self._async_complete_upload(cli, str(resp["id"]))
            return resp3["file"]["id"]

    def _complete_upload(self, cli: httpx.Client, upload_id: str) -> dict[str, Any]:
        complete_url = f"{self._base_url}/v1/uploads/{upload_id}/complete"
        last_transport_error: _APITransportError | None = None
        for attempt in range(1, _TRANSPORT_MAX_ATTEMPTS + 1):
            try:
                response = _request_once(
                    cli,
                    "POST",
                    complete_url,
                    stage="upload completion",
                    headers=self._headers(),
                )
                return self._check(response)
            except _APITransportError as exc:
                last_transport_error = exc
            except _V1APIError as exc:
                if exc.code != "upload_already_terminal":
                    raise

            upload = self._get_upload(cli, upload_id)
            if upload.get("status") == "completed":
                return upload
            if attempt < _TRANSPORT_MAX_ATTEMPTS:
                time.sleep(_transport_retry_delay(attempt))

        if last_transport_error is not None:
            raise last_transport_error
        raise _V1APIError("upload_not_ready", f"Upload {upload_id} did not reach completed status")

    async def _async_complete_upload(self, cli: httpx.AsyncClient, upload_id: str) -> dict[str, Any]:
        complete_url = f"{self._base_url}/v1/uploads/{upload_id}/complete"
        last_transport_error: _APITransportError | None = None
        for attempt in range(1, _TRANSPORT_MAX_ATTEMPTS + 1):
            try:
                response = await _async_request_once(
                    cli,
                    "POST",
                    complete_url,
                    stage="upload completion",
                    headers=self._headers(),
                )
                return self._check(response)
            except _APITransportError as exc:
                last_transport_error = exc
            except _V1APIError as exc:
                if exc.code != "upload_already_terminal":
                    raise

            upload = await self._async_get_upload(cli, upload_id)
            if upload.get("status") == "completed":
                return upload
            if attempt < _TRANSPORT_MAX_ATTEMPTS:
                await asyncio.sleep(_transport_retry_delay(attempt))

        if last_transport_error is not None:
            raise last_transport_error
        raise _V1APIError("upload_not_ready", f"Upload {upload_id} did not reach completed status")

    def _get_upload(self, cli: httpx.Client, upload_id: str) -> dict[str, Any]:
        response = _request_with_retry(
            cli,
            "GET",
            f"{self._base_url}/v1/uploads/{upload_id}",
            stage="upload status",
            headers=self._headers(),
        )
        return self._check(response)

    async def _async_get_upload(self, cli: httpx.AsyncClient, upload_id: str) -> dict[str, Any]:
        response = await _async_request_with_retry(
            cli,
            "GET",
            f"{self._base_url}/v1/uploads/{upload_id}",
            stage="upload status",
            headers=self._headers(),
        )
        return self._check(response)

    # ── parse execution ──────────────────────────────────────────────

    def _do_parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=httpx.Timeout(120, connect=30), trust_env=self._trust_env) as cli:
            r = _request_once(
                cli,
                "POST",
                f"{self._base_url}/v1/parse/jobs",
                stage="job submission",
                headers=self._headers(),
                json_body=payload,
            )
            job = self._check(r)
            if job.get("status") not in _TERMINAL_JOB_STATUSES:
                job = self._poll(cli, job["job_id"])
        return job

    async def _async_do_parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30), trust_env=self._trust_env) as cli:
            r = await _async_request_once(
                cli,
                "POST",
                f"{self._base_url}/v1/parse/jobs",
                stage="job submission",
                headers=self._headers(),
                json_body=payload,
            )
            job = self._check(r)
            if job.get("status") not in _TERMINAL_JOB_STATUSES:
                job = await self._async_poll(cli, job["job_id"])
        return job

    def _poll(self, cli: Any, job_id: str) -> dict[str, Any]:
        for _ in range(_POLL_MAX_ATTEMPTS):  # max 1 hour
            time.sleep(_POLL_INTERVAL_SECONDS)
            r = _request_with_retry(
                cli,
                "GET",
                f"{self._base_url}/v1/parse/jobs/{job_id}",
                stage="job polling",
                headers=self._headers(),
            )
            job = self._check(r)
            if job.get("status") in _TERMINAL_JOB_STATUSES:
                return job
        raise _V1APIError("timeout", f"Job {job_id} did not complete within timeout")

    async def _async_poll(self, cli: Any, job_id: str) -> dict[str, Any]:
        for _ in range(_POLL_MAX_ATTEMPTS):
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            r = await _async_request_with_retry(
                cli,
                "GET",
                f"{self._base_url}/v1/parse/jobs/{job_id}",
                stage="job polling",
                headers=self._headers(),
            )
            job = self._check(r)
            if job.get("status") in _TERMINAL_JOB_STATUSES:
                return job
        raise _V1APIError("timeout", f"Job {job_id} did not complete within timeout")

    # ── build ParseResult ────────────────────────────────────────────

    def _build_result(self, job: dict[str, Any], file_name: str) -> ParseResult:
        return _parse_result_from_job(job, file_name, self)

    async def _async_build_result(self, job: dict[str, Any], file_name: str) -> ParseResult:
        return await _async_parse_result_from_job(job, file_name, self)


# ═══════════════════════════════════════════════════════════════════════
#  helpers
# ═══════════════════════════════════════════════════════════════════════


def _request_once(
    client: httpx.Client,
    method: _RequestMethod,
    url: str,
    *,
    stage: str,
    headers: dict[str, str],
    json_body: dict[str, Any] | None = None,
    content: bytes | None = None,
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
    content: bytes | None = None,
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
    content: bytes | None,
    max_attempts: int,
) -> httpx.Response:
    for attempt in range(1, max_attempts + 1):
        try:
            return _send_request(client, method, url, headers=headers, json_body=json_body, content=content)
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
    content: bytes | None = None,
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
    content: bytes | None = None,
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
    content: bytes | None,
    max_attempts: int,
) -> httpx.Response:
    for attempt in range(1, max_attempts + 1):
        try:
            return await _async_send_request(client, method, url, headers=headers, json_body=json_body, content=content)
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
    content: bytes | None,
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
    content: bytes | None,
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _extract_feature_list(health: dict[str, Any], name: str) -> set[str]:
    features = health.get("features")
    if not isinstance(features, dict):
        return set()
    values = features.get(name)
    if not isinstance(values, list):
        return set()
    return {item for item in values if isinstance(item, str)}


def _parse_result_from_job(job: dict[str, Any], file_name: str, parser: MinerUApiParser) -> ParseResult:
    _raise_for_terminal_job_error(job)
    files = job.get("files", [])
    outputs: dict[str, Any] = {}
    if files and files[0].get("output_files"):
        outputs = files[0]["output_files"]

    if _uses_zip_output(parser):
        return _parse_result_from_zip_output(parser, outputs)

    mid_json = _download_json(parser, outputs)
    result = _parse_result_from_middle_json(mid_json)
    return result


async def _async_parse_result_from_job(job: dict[str, Any], file_name: str, parser: MinerUApiParser) -> ParseResult:
    _raise_for_terminal_job_error(job)
    files = job.get("files", [])
    outputs: dict[str, Any] = {}
    if files and files[0].get("output_files"):
        outputs = files[0]["output_files"]

    if _uses_zip_output(parser):
        return _parse_result_from_zip_bytes(
            await _async_download_zip_output(parser, outputs),
            include_images=parser.include_images,
            include_model_output=parser.include_model_output,
        )

    mid_json = await _async_download_json(parser, outputs)
    result = _parse_result_from_middle_json(mid_json)
    return result


def _uses_zip_output(parser: MinerUApiParser) -> bool:
    return parser.include_model_output or parser.include_images


def _extract_model_output_from_archive(archive: zipfile.ZipFile) -> Any | None:
    """从已打开的 zip 读取原始模型输出。"""
    names = set(archive.namelist())
    local_name = "model_output.json"
    if local_name in names:
        model_output_name = local_name
    else:
        staging_names = sorted(name for name in names if Path(name).name.endswith("_model.json"))
        if not staging_names:
            return None
        if len(staging_names) > 1:
            available = ", ".join(staging_names)
            raise _V1APIError("invalid_model_output", f"ZIP output contained multiple model outputs: {available}")
        model_output_name = staging_names[0]

    raw = archive.read(model_output_name)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _V1APIError("invalid_model_output", f"model output JSON is not valid JSON: {exc}") from exc


def _download_zip_output(parser: MinerUApiParser, outputs: dict[str, Any]) -> bytes:
    """下载 zip-only 模式的唯一结果包；缺失 zip 时给出明确错误。"""
    zip_ref = outputs.get("zip")
    if not isinstance(zip_ref, dict):
        available = ", ".join(sorted(outputs)) or "none"
        raise _V1APIError("missing_zip_output", f"Parse job did not return zip output; available outputs: {available}")
    return _download_bytes(parser, zip_ref)


async def _async_download_zip_output(parser: MinerUApiParser, outputs: dict[str, Any]) -> bytes:
    """异步下载 zip-only 模式的唯一结果包；缺失 zip 时给出明确错误。"""
    zip_ref = outputs.get("zip")
    if not isinstance(zip_ref, dict):
        available = ", ".join(sorted(outputs)) or "none"
        raise _V1APIError("missing_zip_output", f"Parse job did not return zip output; available outputs: {available}")
    return await _async_download_bytes(parser, zip_ref)


def _parse_result_from_zip_output(parser: MinerUApiParser, outputs: dict[str, Any]) -> ParseResult:
    """从 v1 API 自包含 zip 输出恢复 ParseResult，避免逐个下载 middle_json 和图片。"""
    return _parse_result_from_zip_bytes(
        _download_zip_output(parser, outputs),
        include_images=parser.include_images,
        include_model_output=parser.include_model_output,
    )


def _parse_result_from_zip_bytes(
    zip_bytes: bytes,
    *,
    include_images: bool,
    include_model_output: bool,
) -> ParseResult:
    """解析自包含 zip 包：读取 middle_json、图片 sidecar 和可选 model_output。"""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            mid_json = _read_middle_json_from_zip(archive)
            result = _parse_result_from_middle_json(mid_json)
            if include_images:
                images = _read_image_sidecars_from_zip(archive, mid_json)
                if images:
                    result.attach_export_images(images)
            if include_model_output:
                model_output = _extract_model_output_from_archive(archive)
                if model_output is not None:
                    result._model_output = model_output
            return result
    except zipfile.BadZipFile as exc:
        raise _V1APIError("invalid_zip_output", "zip output is not a valid ZIP archive") from exc


def _middle_json_zip_candidates() -> list[str]:
    """列出当前客户端支持的 API zip middle_json 文件名。"""
    return [
        "middle_json.json",
        "layout.json",
    ]


def _read_middle_json_from_zip(archive: zipfile.ZipFile) -> dict[str, Any]:
    """从本地 api_server 或 staging remote server 的 zip 中读取 middle_json。"""
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
    """递归收集 middle_json 中引用的图片 sidecar 路径。"""
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"image_path", "img_path"} and isinstance(item, str) and item:
                paths.add(item)
            else:
                paths.update(_collect_image_paths_from_middle_json(item))
    elif isinstance(value, list):
        for item in value:
            paths.update(_collect_image_paths_from_middle_json(item))
    return paths


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
    return images


def _download_json(parser: MinerUApiParser, outputs: dict[str, Any]) -> dict[str, Any]:
    ref = outputs.get("middle_json")
    if not ref:
        available = ", ".join(sorted(outputs)) or "none"
        raise _V1APIError(
            "missing_middle_json_output",
            f"Parse job did not return middle_json output; available outputs: {available}",
        )
    raw = _download_bytes(parser, ref)
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _V1APIError("invalid_middle_json_output", f"middle_json output is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise _V1APIError("invalid_middle_json_output", "middle_json output must be a JSON object with pages or pdf_info")
    return loaded


async def _async_download_json(parser: MinerUApiParser, outputs: dict[str, Any]) -> dict[str, Any]:
    ref = outputs.get("middle_json")
    if not ref:
        available = ", ".join(sorted(outputs)) or "none"
        raise _V1APIError(
            "missing_middle_json_output",
            f"Parse job did not return middle_json output; available outputs: {available}",
        )
    raw = await _async_download_bytes(parser, ref)
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _V1APIError("invalid_middle_json_output", f"middle_json output is not valid JSON: {exc}") from exc
    if not isinstance(loaded, dict):
        raise _V1APIError("invalid_middle_json_output", "middle_json output must be a JSON object with pages or pdf_info")
    return loaded


def _download_bytes(parser: MinerUApiParser, ref: dict[str, Any]) -> bytes:
    file_id = ref.get("file_id")
    if not file_id:
        raise _V1APIError("invalid_response", "No file_id in output reference")

    with httpx.Client(timeout=httpx.Timeout(120, connect=30), follow_redirects=True, trust_env=parser._trust_env) as cli:
        r = _request_with_retry(
            cli,
            "GET",
            f"{parser._base_url}/v1/files/{file_id}/content",
            stage="output download",
            headers=parser._headers(),
        )
        _check_download_response(r)
        return r.content


async def _async_download_bytes(parser: MinerUApiParser, ref: dict[str, Any]) -> bytes:
    file_id = ref.get("file_id")
    if not file_id:
        raise _V1APIError("invalid_response", "No file_id in output reference")

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120, connect=30), follow_redirects=True, trust_env=parser._trust_env
    ) as cli:
        r = await _async_request_with_retry(
            cli,
            "GET",
            f"{parser._base_url}/v1/files/{file_id}/content",
            stage="output download",
            headers=parser._headers(),
        )
        _check_download_response(r)
        return r.content


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
    if isinstance(mid_json, dict):
        pages = mid_json.get("pages")
        if pages is not None:
            return ParseResult.from_dict(mid_json)

        pdf_info = mid_json.get("pdf_info")
        if isinstance(pdf_info, list):
            # Staging JSON output may use the older pdf_info field instead of ParseResult pages.
            compat_payload = dict(mid_json)
            compat_payload["pages"] = pdf_info
            return ParseResult.from_dict(compat_payload)
    raise _V1APIError("invalid_middle_json_output", "middle_json output must contain a list field named pages or pdf_info")


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
        str(job.get("status")), f"Parse job {job.get('job_id', '<unknown>')} ended with status {job.get('status')}"
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
    # Staging auth failures still use the legacy msgCode/msg payload instead of {"error": ...}.
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
