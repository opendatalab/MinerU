# Copyright (c) Opendatalab. All rights reserved.
"""DocumentParser implementation for the MinerU v1 REST API.

``MinerUApiParser`` extends ``DocumentParser`` and talks to a v1 API
server (local or cloud).  It exposes API tier semantics directly and
does not expose parser backend selection.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from ..types import PageInfo, Tier
from .base import DocumentParser, ParseResult


class MinerUApiParser(DocumentParser):
    """Parser that delegates to a MinerU v1 API server.

    Works with local-server, LAN, and cloud (mineru.net) deployments::

        # local deployment — uses ``local`` source, no upload needed
        parser = MinerUApiParser(api_url="http://localhost:8000/api", tier="standard")

        # cloud (remote)
        parser = MinerUApiParser(
            api_url="https://mineru.net/api", api_key="sk_...",
            tier="pro",
        )

        result = parser.parse("report.pdf")
        print(result.markdown())

    Constructor parameters:

    - ``tier`` → v1 ``tier`` (``"standard"`` / ``"pro"``); ``None`` omits the field
    - ``page_range`` → per-file v1 ``page_range``
    """

    def __init__(
        self,
        *,
        api_url: str | None = None,
        api_key: str | None = None,
        tier: Tier | None = None,
    ) -> None:
        api_url = api_url or os.environ.get("MINERU_API_URL", "https://mineru.net/api")
        self._base = api_url.rstrip("/")
        self._api_key = api_key
        self._local = self._is_localhost(self._base)
        self.tier = tier

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
        if "staging" in self._base:
            return ["json"]
        return ["middle_json", "images"]

    # ── HTTP helpers ─────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        return h

    @staticmethod
    def _is_localhost(base: str) -> bool:
        return any(
            base.startswith(p)
            for p in (
                "http://localhost",
                "http://127.",
                "http://192.168.",
                "http://10.",
                "http://172.",
            )
        )

    @staticmethod
    def _check(r: Any) -> dict[str, Any]:
        if r.status_code >= 400:
            raise _V1APIError("http_error", f"HTTP {r.status_code}: {r.text[:500]}")
        try:
            data = r.json()
        except Exception:
            return {}  # e.g. OSS PUT returns 200 with empty body
        if "error" in data:
            err = data["error"]
            raise _V1APIError(err.get("code", "unknown"), err.get("message", str(err)))
        if "detail" in data and isinstance(data["detail"], dict) and "error" in data["detail"]:
            err = data["detail"]["error"]
            raise _V1APIError(err.get("code", "unknown"), err.get("message", str(err)))
        return data

    # ── upload ───────────────────────────────────────────────────────

    def _upload(self, file_path: Path) -> str:
        size = file_path.stat().st_size
        sha = _sha256_file(file_path)

        with httpx.Client(timeout=httpx.Timeout(120, connect=30)) as cli:
            r = cli.post(
                f"{self._base}/v1/uploads",
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
                upload_url = f"{self._base}{upload_url}"
            with file_path.open("rb") as fh:
                r2 = cli.put(upload_url, content=fh.read(), headers=upload_headers)
            self._check(r2)

            r3 = cli.post(
                f"{self._base}/v1/uploads/{resp['id']}/complete",
                headers=self._headers(),
            )
            resp3 = self._check(r3)
            return resp3["file"]["id"]

    async def _async_upload(self, file_path: Path) -> str:
        size = file_path.stat().st_size
        sha = _sha256_file(file_path)

        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30)) as cli:
            r = await cli.post(
                f"{self._base}/v1/uploads",
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
                upload_url = f"{self._base}{upload_url}"
            data = file_path.read_bytes()
            r2 = await cli.put(upload_url, content=data, headers=upload_headers)
            self._check(r2)

            r3 = await cli.post(
                f"{self._base}/v1/uploads/{resp['id']}/complete",
                headers=self._headers(),
            )
            resp3 = self._check(r3)
            return resp3["file"]["id"]

    # ── parse execution ──────────────────────────────────────────────

    def _do_parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        with httpx.Client(timeout=httpx.Timeout(120, connect=30)) as cli:
            r = cli.post(f"{self._base}/v1/parse/jobs", headers=self._headers(), json=payload)
            job = self._check(r)
            if job.get("status") not in ("completed", "partial", "failed", "canceled"):
                # poll if wait timed out
                job = self._poll(cli, job["job_id"])
        return job

    async def _async_do_parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30)) as cli:
            r = await cli.post(f"{self._base}/v1/parse/jobs", headers=self._headers(), json=payload)
            job = self._check(r)
            if job.get("status") not in ("completed", "partial", "failed", "canceled"):
                job = await self._async_poll(cli, job["job_id"])
        return job

    def _poll(self, cli: Any, job_id: str) -> dict[str, Any]:
        delay = 2
        for _ in range(150):  # max 5 min
            time.sleep(delay)
            r = cli.get(f"{self._base}/v1/parse/jobs/{job_id}", headers=self._headers())
            job = self._check(r)
            if job.get("status") in ("completed", "partial", "failed", "canceled"):
                return job
            delay = min(delay * 2, 30)
        raise _V1APIError("timeout", f"Job {job_id} did not complete within timeout")

    async def _async_poll(self, cli: Any, job_id: str) -> dict[str, Any]:
        delay = 2
        for _ in range(150):
            await asyncio.sleep(delay)
            r = await cli.get(f"{self._base}/v1/parse/jobs/{job_id}", headers=self._headers())
            job = self._check(r)
            if job.get("status") in ("completed", "partial", "failed", "canceled"):
                return job
            delay = min(delay * 2, 30)
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

    mid_json = _download_json(parser, outputs)
    result = ParseResult.from_dict(mid_json)
    images = _download_image_sidecars(parser, outputs)
    if images or "images" in outputs:
        result.attach_export_images(images)
    return result


async def _async_parse_result_from_job(job: dict[str, Any], file_name: str, parser: MinerUApiParser) -> ParseResult:
    _raise_for_terminal_job_error(job)
    files = job.get("files", [])
    outputs: dict[str, Any] = {}
    if files and files[0].get("output_files"):
        outputs = files[0]["output_files"]

    mid_json = await _async_download_json(parser, outputs)
    result = ParseResult.from_dict(mid_json)
    images = await _async_download_image_sidecars(parser, outputs)
    if images or "images" in outputs:
        result.attach_export_images(images)
    return result


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
        images[img_path] = _download_bytes(parser, ref)
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
        images[img_path] = await _async_download_bytes(parser, ref)
    return images


def _download_json(parser: MinerUApiParser, outputs: dict[str, Any]) -> dict[str, Any]:
    ref = outputs.get("middle_json") or outputs.get("json") or outputs.get("json_")
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
        raise _V1APIError("invalid_middle_json_output", "middle_json output must be a JSON object with pages")
    return loaded


async def _async_download_json(parser: MinerUApiParser, outputs: dict[str, Any]) -> dict[str, Any]:
    ref = outputs.get("middle_json") or outputs.get("json") or outputs.get("json_")
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
        raise _V1APIError("invalid_middle_json_output", "middle_json output must be a JSON object with pages")
    return loaded


def _download_bytes(parser: MinerUApiParser, ref: dict[str, Any]) -> bytes:
    # TEMP(2026-06-12): staging server may return {"url": "..."} instead of
    # {"file_id": "...", "bytes": ...}. Keep this branch until staging aligns.
    if ref.get("url"):
        with httpx.Client(timeout=httpx.Timeout(120, connect=30), follow_redirects=True) as cli:
            r = cli.get(ref["url"])
            _check_download_response(r)
            return r.content

    file_id = ref.get("file_id")
    if not file_id:
        raise _V1APIError("invalid_response", "No file_id or url in output reference")

    with httpx.Client(timeout=httpx.Timeout(120, connect=30), follow_redirects=True) as cli:
        r = cli.get(f"{parser._base}/v1/files/{file_id}/content", headers=parser._headers())
        _check_download_response(r)
        return r.content


async def _async_download_bytes(parser: MinerUApiParser, ref: dict[str, Any]) -> bytes:
    # TEMP(2026-06-12): staging server may return {"url": "..."} instead of
    # {"file_id": "...", "bytes": ...}. Keep this branch until staging aligns.
    if ref.get("url"):
        async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30), follow_redirects=True) as cli:
            r = await cli.get(ref["url"])
            _check_download_response(r)
            return r.content

    file_id = ref.get("file_id")
    if not file_id:
        raise _V1APIError("invalid_response", "No file_id or url in output reference")

    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30), follow_redirects=True) as cli:
        r = await cli.get(f"{parser._base}/v1/files/{file_id}/content", headers=parser._headers())
        _check_download_response(r)
        return r.content


def _check_download_response(r: httpx.Response) -> None:
    if r.status_code < 400:
        return
    preview = r.text[:500]
    raise _V1APIError("download_failed", f"HTTP {r.status_code}: {preview}")


def _pages_from_middle_json(mid_json: dict[str, Any] | None) -> list[PageInfo]:
    if mid_json is None:
        return []
    if isinstance(mid_json, dict):
        pages = mid_json.get("pages")
        if isinstance(pages, list):
            return [PageInfo.from_dict(raw) for raw in pages if isinstance(raw, dict)]
    raise _V1APIError("invalid_middle_json_output", "middle_json output must contain a list field named pages")


def _raise_for_terminal_job_error(job: dict[str, Any]) -> None:
    if job.get("status") not in ("failed", "canceled"):
        return
    error = job.get("error")
    if isinstance(error, dict):
        code = str(error.get("code") or job.get("status"))
        message = str(error.get("message") or error)
        raise _V1APIError(code, message)
    raise _V1APIError(
        str(job.get("status")), f"Parse job {job.get('job_id', '<unknown>')} ended with status {job.get('status')}"
    )


class _V1APIError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
