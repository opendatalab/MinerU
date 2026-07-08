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
import os
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from ..types import PageInfo, Tier, validate_tier
from ..utils.image_payload import validate_image_sidecar_path
from .base import DocumentParser, ParseResult

_POLL_INTERVAL_SECONDS = 1
_POLL_MAX_ATTEMPTS = 3600
_TERMINAL_JOB_STATUSES = {"completed", "partial", "failed", "canceled"}


class MinerUApiParser(DocumentParser):
    """Parser that delegates to a MinerU v1 API server.

    Works with local-server, LAN, and cloud (mineru.net) deployments::

        # local deployment — uses ``local`` source, no upload needed
        parser = MinerUApiParser(api_url="http://localhost:8000/api", tier="medium")

        # cloud (remote)
        parser = MinerUApiParser(
            api_url="https://mineru.net/api", api_key="sk_...",
            tier="high",
        )

        result = parser.parse("report.pdf")
        print(result.markdown())

    Constructor parameters:

    - ``tier`` → v1 ``tier`` (``"flash"`` / ``"medium"`` / ``"high"`` / ``"extra_high"``); ``None`` omits the field
    - ``page_range`` → per-file v1 ``page_range``
    """

    DEFAULT_API_URL = "https://mineru.net/api"

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        tier: Tier | None = None,
        include_model_output: bool = False,
        zip_output_only: bool = False,
    ) -> None:
        self._base_url = (api_url or os.environ.get("MINERU_API_URL") or self.DEFAULT_API_URL).rstrip("/")
        self._api_key = (api_key if api_key is not None else os.environ.get("MINERU_API_KEY")) or None
        self._local = _is_local_network_url(self._base_url)
        self._trust_env = should_trust_env_for_url(self._base_url)
        self.tier = validate_tier(tier) if tier is not None else None
        self.include_model_output = include_model_output
        self.zip_output_only = zip_output_only

    # ── DocumentParser interface ─────────────────────────────────────

    def parse(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        payload = self._build_payload(file_path, page_range)
        job = self._do_parse(payload)
        return self._build_result(job, file_path.name)

    async def parse_async(self, path: str | Path, *, page_range: str = "") -> ParseResult:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        payload = self._build_payload(file_path, page_range)
        job = await self._async_do_parse(payload)
        return await self._async_build_result(job, file_path.name)

    # ── payload construction ─────────────────────────────────────────

    def _build_payload(self, file_path: Path, page_range: str) -> dict[str, Any]:
        source: dict[str, Any]
        if self._local:
            source = {"type": "local", "path": str(file_path)}
        else:
            # upload flow: create → PUT → complete → file_id
            file_id = self._upload(file_path)
            source = {"type": "file_id", "file_id": file_id}

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

    def _output_formats(self) -> list[str]:
        if "staging" in self._base_url:
            # Staging still exposes the non-standard "json" output instead of middle_json/images.
            return ["json"]
        if self.zip_output_only:
            return ["zip"]
        formats = ["middle_json", "images"]
        if self.include_model_output:
            formats.append("zip")
        return formats

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
            r = cli.post(
                f"{self._base_url}/v1/uploads",
                headers=self._headers(),
                json={
                    "filename": file_path.name,
                    "bytes": size,
                    "mime_type": _mime_type(file_path),
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
                r2 = cli.put(upload_url, content=fh.read(), headers=upload_headers)
            self._check(r2)

            r3 = cli.post(
                f"{self._base_url}/v1/uploads/{resp['id']}/complete",
                headers=self._headers(),
            )
            resp3 = self._check(r3)
            return resp3["file"]["id"]

    async def _async_upload(self, file_path: Path) -> str:
        size = file_path.stat().st_size
        sha = _sha256_file(file_path)

        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30), trust_env=self._trust_env) as cli:
            r = await cli.post(
                f"{self._base_url}/v1/uploads",
                headers=self._headers(),
                json={
                    "filename": file_path.name,
                    "bytes": size,
                    "mime_type": _mime_type(file_path),
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
            r2 = await cli.put(upload_url, content=data, headers=upload_headers)
            self._check(r2)

            r3 = await cli.post(
                f"{self._base_url}/v1/uploads/{resp['id']}/complete",
                headers=self._headers(),
            )
            resp3 = self._check(r3)
            return resp3["file"]["id"]

    # ── parse execution ──────────────────────────────────────────────

    def _do_parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=httpx.Timeout(120, connect=30), trust_env=self._trust_env) as cli:
            r = cli.post(f"{self._base_url}/v1/parse/jobs", headers=self._headers(), json=payload)
            job = self._check(r)
            if job.get("status") not in _TERMINAL_JOB_STATUSES:
                # poll if wait timed out
                job = self._poll(cli, job["job_id"])
        return job

    async def _async_do_parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30), trust_env=self._trust_env) as cli:
            r = await cli.post(f"{self._base_url}/v1/parse/jobs", headers=self._headers(), json=payload)
            job = self._check(r)
            if job.get("status") not in _TERMINAL_JOB_STATUSES:
                job = await self._async_poll(cli, job["job_id"])
        return job

    def _poll(self, cli: Any, job_id: str) -> dict[str, Any]:
        for _ in range(_POLL_MAX_ATTEMPTS):  # max 1 hour
            time.sleep(_POLL_INTERVAL_SECONDS)
            r = cli.get(f"{self._base_url}/v1/parse/jobs/{job_id}", headers=self._headers())
            job = self._check(r)
            if job.get("status") in _TERMINAL_JOB_STATUSES:
                return job
        raise _V1APIError("timeout", f"Job {job_id} did not complete within timeout")

    async def _async_poll(self, cli: Any, job_id: str) -> dict[str, Any]:
        for _ in range(_POLL_MAX_ATTEMPTS):
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)
            r = await cli.get(f"{self._base_url}/v1/parse/jobs/{job_id}", headers=self._headers())
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".pdf": "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        ".html": "text/html",
        ".htm": "text/html",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".jp2": "image/jp2",
        ".webp": "image/webp",
        ".gif": "image/gif",
        ".bmp": "image/bmp",
        ".tiff": "image/tiff",
    }.get(ext, "application/octet-stream")


def _parse_result_from_job(job: dict[str, Any], file_name: str, parser: MinerUApiParser) -> ParseResult:
    _raise_for_terminal_job_error(job)
    files = job.get("files", [])
    outputs: dict[str, Any] = {}
    if files and files[0].get("output_files"):
        outputs = files[0]["output_files"]

    if parser.zip_output_only and "staging" not in parser._base_url:
        return _parse_result_from_zip_output(parser, outputs, file_name)

    mid_json = _download_json(parser, outputs)
    result = _parse_result_from_middle_json(mid_json)
    images = _download_image_sidecars(parser, outputs)
    if images or "images" in outputs:
        result.attach_export_images(images)
    model_output = _download_model_output_from_zip(parser, outputs, file_name)
    if model_output is not None:
        result._model_output = model_output
    return result


async def _async_parse_result_from_job(job: dict[str, Any], file_name: str, parser: MinerUApiParser) -> ParseResult:
    _raise_for_terminal_job_error(job)
    files = job.get("files", [])
    outputs: dict[str, Any] = {}
    if files and files[0].get("output_files"):
        outputs = files[0]["output_files"]

    if parser.zip_output_only and "staging" not in parser._base_url:
        return _parse_result_from_zip_bytes(
            await _async_download_zip_output(parser, outputs),
            file_name,
            include_model_output=parser.include_model_output,
        )

    mid_json = await _async_download_json(parser, outputs)
    result = _parse_result_from_middle_json(mid_json)
    images = await _async_download_image_sidecars(parser, outputs)
    if images or "images" in outputs:
        result.attach_export_images(images)
    model_output = await _async_download_model_output_from_zip(parser, outputs, file_name)
    if model_output is not None:
        result._model_output = model_output
    return result


def _download_model_output_from_zip(parser: MinerUApiParser, outputs: dict[str, Any], file_name: str) -> Any | None:
    """按需从 v1 zip 输出中提取原始模型输出；缺少目标文件时保持兼容返回 None。"""
    if not parser.include_model_output:
        return None
    zip_ref = outputs.get("zip")
    if not isinstance(zip_ref, dict):
        return None
    return _extract_model_output_from_zip(_download_bytes(parser, zip_ref), file_name)


async def _async_download_model_output_from_zip(
    parser: MinerUApiParser,
    outputs: dict[str, Any],
    file_name: str,
) -> Any | None:
    """异步按需下载 v1 zip 并提取原始模型输出；缺少目标文件时保持兼容返回 None。"""
    if not parser.include_model_output:
        return None
    zip_ref = outputs.get("zip")
    if not isinstance(zip_ref, dict):
        return None
    return _extract_model_output_from_zip(await _async_download_bytes(parser, zip_ref), file_name)


def _extract_model_output_from_zip(zip_bytes: bytes, file_name: str) -> Any | None:
    """从 API server zip 中读取 `{stem}_model_output.json`，没有该文件则不影响普通解析。"""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            return _extract_model_output_from_archive(archive, file_name)
    except zipfile.BadZipFile as exc:
        raise _V1APIError("invalid_model_output", "model output zip is not a valid ZIP archive") from exc


def _extract_model_output_from_archive(archive: zipfile.ZipFile, file_name: str) -> Any | None:
    """从已打开的 zip 读取原始模型输出，供普通 zip 和 zip-only 解析路径复用。"""
    model_output_name = f"{Path(file_name).stem}_model_output.json"
    if model_output_name not in archive.namelist():
        return None
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


def _parse_result_from_zip_output(parser: MinerUApiParser, outputs: dict[str, Any], file_name: str) -> ParseResult:
    """从 v1 API 自包含 zip 输出恢复 ParseResult，避免逐个下载 middle_json 和图片。"""
    return _parse_result_from_zip_bytes(
        _download_zip_output(parser, outputs),
        file_name,
        include_model_output=parser.include_model_output,
    )


def _parse_result_from_zip_bytes(zip_bytes: bytes, file_name: str, *, include_model_output: bool) -> ParseResult:
    """解析自包含 zip 包：读取 middle_json、图片 sidecar 和可选 model_output。"""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            mid_json = _read_middle_json_from_zip(archive, file_name)
            result = _parse_result_from_middle_json(mid_json)
            images = _read_image_sidecars_from_zip(archive, mid_json)
            if images:
                result.attach_export_images(images)
            if include_model_output:
                model_output = _extract_model_output_from_archive(archive, file_name)
                if model_output is not None:
                    result._model_output = model_output
            return result
    except zipfile.BadZipFile as exc:
        raise _V1APIError("invalid_zip_output", "zip output is not a valid ZIP archive") from exc


def _middle_json_zip_candidates(file_name: str) -> list[str]:
    """列出 API zip 和本地 Gradio zip 中可能出现的 middle_json 文件名。"""
    stem = Path(file_name).stem
    return [
        f"{stem}.middle_json",
        f"{stem}_middle.json",
        f"{stem}.middle_json.json",
        "middle_json.json",
    ]


def _read_middle_json_from_zip(archive: zipfile.ZipFile, file_name: str) -> dict[str, Any]:
    """从 zip 中读取 middle_json，兼容 API server 和本地保存包的命名。"""
    names = set(archive.namelist())
    for candidate in _middle_json_zip_candidates(file_name):
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
        if safe_image_path in archive_names:
            images[safe_image_path] = archive.read(safe_image_path)
    return images


def _download_image_sidecars(parser: MinerUApiParser, outputs: dict[str, Any]) -> dict[str, bytes]:
    """下载 API 返回的图片 sidecar，并按 middle_json 中的 image_path 建立字节映射。"""
    images: dict[str, bytes] = {}
    image_refs = outputs.get("images")
    if not isinstance(image_refs, list):
        return images
    for ref in image_refs:
        if not isinstance(ref, dict):
            continue
        img_path = ref.get("path")
        if not isinstance(img_path, str) or not img_path:
            continue
        safe_img_path = validate_image_sidecar_path(img_path)
        images[safe_img_path] = _download_bytes(parser, ref)
    return images


async def _async_download_image_sidecars(parser: MinerUApiParser, outputs: dict[str, Any]) -> dict[str, bytes]:
    """异步下载 API 返回的图片 sidecar，并按 image_path 建立字节映射。"""
    images: dict[str, bytes] = {}
    image_refs = outputs.get("images")
    if not isinstance(image_refs, list):
        return images
    for ref in image_refs:
        if not isinstance(ref, dict):
            continue
        img_path = ref.get("path")
        if not isinstance(img_path, str) or not img_path:
            continue
        safe_img_path = validate_image_sidecar_path(img_path)
        images[safe_img_path] = await _async_download_bytes(parser, ref)
    return images


def _download_json(parser: MinerUApiParser, outputs: dict[str, Any]) -> dict[str, Any]:
    # Staging returns output_files.json; the standard local v1 API returns output_files.middle_json.
    ref = outputs.get("middle_json") or outputs.get("json")
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
    # Staging returns output_files.json; the standard local v1 API returns output_files.middle_json.
    ref = outputs.get("middle_json") or outputs.get("json")
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
        r = cli.get(f"{parser._base_url}/v1/files/{file_id}/content", headers=parser._headers())
        _check_download_response(r)
        return r.content


async def _async_download_bytes(parser: MinerUApiParser, ref: dict[str, Any]) -> bytes:
    file_id = ref.get("file_id")
    if not file_id:
        raise _V1APIError("invalid_response", "No file_id in output reference")

    async with httpx.AsyncClient(
        timeout=httpx.Timeout(120, connect=30), follow_redirects=True, trust_env=parser._trust_env
    ) as cli:
        r = await cli.get(f"{parser._base_url}/v1/files/{file_id}/content", headers=parser._headers())
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
