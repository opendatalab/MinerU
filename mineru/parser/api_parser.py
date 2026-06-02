# Copyright (c) Opendatalab. All rights reserved.
"""DocumentParser implementation for the MinerU v1 REST API.

``MinerUApiParser`` extends ``DocumentParser`` and talks to a v1 API
server (local or cloud).  All :class:`DocumentParser` constructor
parameters are mapped to the v1 API job payload automatically.
"""

from __future__ import annotations

import hashlib
import json as _json
from pathlib import Path
from typing import Any

from .base import DocumentParser
from .parse_result import ParseResult
from .types import PageInfo


class MinerUApiParser(DocumentParser):
    """Parser that delegates to a MinerU v1 API server.

    Works with local-server, LAN, and cloud (mineru.net) deployments::

        # local deployment — uses ``local`` source, no upload needed
        parser = MinerUApiParser(api_url="http://localhost:8000/api", backend="pipeline")

        # cloud (paid)
        parser = MinerUApiParser(
            api_url="https://mineru.net/api", token="msk_...",
            backend="vlm",
        )

        result = parser.parse("report.pdf")
        print(result.markdown())

    Constructor parameters (matching :class:`DocumentParser`):

    - ``backend`` → v1 ``preset`` (``"pipeline"`` / ``"vlm"`` / ``"auto"``)
    - ``method`` → v1 ``ocr`` (``"auto"`` / ``"ocr"`` / ``"txt"``)
    - ``lang`` → v1 ``language``
    - ``formula_enable`` / ``table_enable`` / ``image_analysis`` → mapped directly
    - ``start_page_id`` / ``end_page_id`` → v1 ``page_range``
    - ``return_md`` / ``return_middle_json`` / ``return_content_list`` /
      ``return_images`` → v1 ``output_formats``
    """

    def __init__(
        self,
        *,
        api_url: str,
        token: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._base = api_url.rstrip("/")
        self._token = token
        self._local = self._is_localhost(self._base)

    # ── DocumentParser interface ─────────────────────────────────────

    def parse(self, path: str | Path) -> ParseResult:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        payload = self._build_payload(file_path)
        job = self._do_parse(payload)
        return self._build_result(job, file_path.name)

    async def parse_async(self, path: str | Path) -> ParseResult:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(file_path)

        payload = self._build_payload(file_path)
        job = await self._async_do_parse(payload)
        return await self._async_build_result(job, file_path.name)

    # ── payload construction ─────────────────────────────────────────

    def _build_payload(self, file_path: Path) -> dict[str, Any]:
        source: dict[str, Any]
        if self._local:
            source = {"type": "local", "path": str(file_path)}
        else:
            # upload flow: create → PUT → complete → file_id
            file_id = self._upload(file_path)
            source = {"type": "file_id", "file_id": file_id}

        options: dict[str, Any] = {
            "language": self.lang,
            "ocr": self._ocr_mode(),
            "formula": self.formula_enable,
            "table": self.table_enable,
            "image_analysis": self.image_analysis,
        }
        page_range = self._page_range_str()
        if page_range:
            options["page_range"] = page_range

        return {
            "files": [{"source": source, "options": options}],
            "preset": self._preset(),
            "output_formats": self._output_formats(),
        }

    def _preset(self) -> str:
        if self.backend == "pipeline":
            return "pipeline"
        if self.backend.startswith("vlm"):
            return "vlm"
        if self.backend.startswith("html"):
            return "html"
        return "auto"

    def _ocr_mode(self) -> str:
        if self.method == "ocr":
            return "true"
        if self.method == "txt":
            return "txt"
        return "auto"

    def _page_range_str(self) -> str | None:
        if self.start_page_id == 0 and self.end_page_id is None:
            return None
        start = self.start_page_id + 1  # 0-based → 1-based
        end = self.end_page_id or ""
        return f"{start}-{end}".rstrip("-")

    def _output_formats(self) -> list[str]:
        return ["json"]

    # ── HTTP helpers ─────────────────────────────────────────────────

    def _headers(self) -> dict[str, str]:
        h: dict[str, str] = {}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
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
        data = r.json()
        if "error" in data:
            err = data["error"]
            raise _V1APIError(err.get("code", "unknown"), err.get("message", str(err)))
        if "detail" in data and isinstance(data["detail"], dict) and "error" in data["detail"]:
            err = data["detail"]["error"]
            raise _V1APIError(err.get("code", "unknown"), err.get("message", str(err)))
        return data

    # ── upload ───────────────────────────────────────────────────────

    def _upload(self, file_path: Path) -> str:
        import httpx

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
            if upload_url.startswith("/"):
                upload_url = f"{self._base}{upload_url}"
            with file_path.open("rb") as fh:
                r2 = cli.put(upload_url, content=fh.read())
            self._check(r2)

            r3 = cli.post(f"{self._base}/v1/uploads/{resp['id']}/complete", headers=self._headers())
            resp3 = self._check(r3)
            return resp3["file"]["id"]

    async def _async_upload(self, file_path: Path) -> str:
        import httpx

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
            if upload_url.startswith("/"):
                upload_url = f"{self._base}{upload_url}"
            data = file_path.read_bytes()
            r2 = await cli.put(upload_url, content=data)
            self._check(r2)

            r3 = await cli.post(f"{self._base}/v1/uploads/{resp['id']}/complete", headers=self._headers())
            resp3 = self._check(r3)
            return resp3["file"]["id"]

    # ── parse execution ──────────────────────────────────────────────

    def _do_parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        payload["wait"] = 600
        with httpx.Client(timeout=httpx.Timeout(660, connect=30)) as cli:
            r = cli.post(f"{self._base}/v1/parse/jobs", headers=self._headers(), json=payload)
            job = self._check(r)
        if job.get("status") not in ("completed", "partial", "failed"):
            # poll if wait timed out
            job = self._poll(cli, job["job_id"])
        return job

    async def _async_do_parse(self, payload: dict[str, Any]) -> dict[str, Any]:
        import httpx

        payload["wait"] = 600
        async with httpx.AsyncClient(timeout=httpx.Timeout(660, connect=30)) as cli:
            r = await cli.post(f"{self._base}/v1/parse/jobs", headers=self._headers(), json=payload)
            job = self._check(r)
        if job.get("status") not in ("completed", "partial", "failed"):
            job = await self._async_poll(cli, job["job_id"])
        return job

    def _poll(self, cli: Any, job_id: str) -> dict[str, Any]:
        import time

        delay = 2
        for _ in range(150):  # max 5 min
            time.sleep(delay)
            r = cli.get(f"{self._base}/v1/parse/jobs/{job_id}", headers=self._headers())
            job = self._check(r)
            if job.get("status") in ("completed", "partial", "failed"):
                return job
            delay = min(delay * 2, 30)
        raise _V1APIError("timeout", f"Job {job_id} did not complete within timeout")

    async def _async_poll(self, cli: Any, job_id: str) -> dict[str, Any]:
        import asyncio

        delay = 2
        for _ in range(150):
            await asyncio.sleep(delay)
            r = await cli.get(f"{self._base}/v1/parse/jobs/{job_id}", headers=self._headers())
            job = self._check(r)
            if job.get("status") in ("completed", "partial", "failed"):
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
    files = job.get("files", [])
    outputs: dict[str, Any] = {}
    if files and files[0].get("output_files"):
        outputs = files[0]["output_files"]

    mid_json = _download_json(parser, outputs)

    from mineru.version import __version__

    return ParseResult(
        pages=_pages_from_middle_json(mid_json),
        _backend=parser.backend,
        _version_name=f"v1-api-{__version__}",
        _file_name=file_name,
    )


async def _async_parse_result_from_job(job: dict[str, Any], file_name: str, parser: MinerUApiParser) -> ParseResult:
    files = job.get("files", [])
    outputs: dict[str, Any] = {}
    if files and files[0].get("output_files"):
        outputs = files[0]["output_files"]

    mid_json = await _async_download_json(parser, outputs)

    from mineru.version import __version__

    return ParseResult(
        pages=_pages_from_middle_json(mid_json),
        _backend=parser.backend,
        _version_name=f"v1-api-{__version__}",
        _file_name=file_name,
    )


def _download_json(parser: MinerUApiParser, outputs: dict[str, Any]) -> dict[str, Any] | None:
    ref = outputs.get("json") or outputs.get("json_")
    if not ref or not ref.get("file_id"):
        return None
    try:
        raw = _download_bytes(parser, ref["file_id"])
        return _json.loads(raw)
    except Exception:
        return None


async def _async_download_json(parser: MinerUApiParser, outputs: dict[str, Any]) -> dict[str, Any] | None:
    ref = outputs.get("json") or outputs.get("json_")
    if not ref or not ref.get("file_id"):
        return None
    try:
        raw = await _async_download_bytes(parser, ref["file_id"])
        return _json.loads(raw)
    except Exception:
        return None


def _download_bytes(parser: MinerUApiParser, file_id: str) -> bytes:
    import httpx

    with httpx.Client(timeout=httpx.Timeout(120, connect=30)) as cli:
        r = cli.get(f"{parser._base}/v1/files/{file_id}/content", headers=parser._headers())
        parser._check(r)
        return r.content


async def _async_download_bytes(parser: MinerUApiParser, file_id: str) -> bytes:
    import httpx

    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=30)) as cli:
        r = await cli.get(f"{parser._base}/v1/files/{file_id}/content", headers=parser._headers())
        parser._check(r)
        return r.content


def _pages_from_middle_json(mid_json: dict[str, Any] | list[Any] | None) -> list[Any]:
    if mid_json is None:
        return []
    if isinstance(mid_json, list):
        return [PageInfo.from_dict(raw) for raw in mid_json if isinstance(raw, dict)]
    if isinstance(mid_json, dict):
        pdf_info = mid_json.get("pdf_info", [])
        if isinstance(pdf_info, list):
            return [PageInfo.from_dict(raw) for raw in pdf_info if isinstance(raw, dict)]
        if isinstance(pdf_info, dict):
            raw_pages = pdf_info.get("preproc_blocks", [])
            return [
                PageInfo(page_idx=i, page_size=(raw.get("width", 0), raw.get("height", 0)) if isinstance(raw, dict) else (0, 0))
                for i, raw in enumerate(raw_pages)
            ]
    return []


class _V1APIError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"[{code}] {message}")
