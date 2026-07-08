# Copyright (c) Opendatalab. All rights reserved.
"""MinerU v1 REST API — Pydantic models and FastAPI route definitions.

The API surface defined here matches the NEW-API.md specification.
All route handlers are stubs (raise 501 Not Implemented) — only the
contract (models, path/query params, response schemas, status codes)
is defined.  Business logic belongs in downstream integration modules.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import pathlib
import secrets
import shutil
import sys
import tempfile
import threading
import time
import zipfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncIterator, Callable, Literal, NoReturn
from urllib.parse import urlparse

import click
import httpx
import uvicorn
from fastapi import APIRouter, Body, Depends, FastAPI, HTTPException, Path, Query, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..types import Tier, validate_tier
from ..utils.backend_options import DEFAULT_HYBRID_EFFORT
from ..utils.image_payload import validate_image_sidecar_path
from ..utils.ocr_language import PUBLIC_OCR_LANGUAGES, validate_public_ocr_lang
from ..version import __version__
from . import parse_async
from .tier import (
    ParserRuntimeOptions,
    TierDependencyError,
    ensure_tier_runtime_dependencies,
    runtime_options_for_tier,
)

_API_SERVER_TIERS: tuple[Tier, ...] = ("flash", "medium", "high", "extra_high")
_DEFAULT_API_SERVER_TIER: Tier = "high"
_API_SERVER_LANGUAGES = PUBLIC_OCR_LANGUAGES
_MANAGED_PARSE_SERVER_ENV = "MINERU_MANAGED_PARSE_SERVER"
_MAX_INLINE_BYTES_DEFAULT = 1024 * 1024
_LOCAL_PARSE_OUTPUT_FORMATS: tuple[OutputFormat, ...] = (
    "markdown",
    "middle_json",
    "content_list",
    "structured_content",
    "zip",
)
_BASE_PARSE_SOURCES: tuple[SourceType, ...] = ("file_id", "url", "inline")
logger = logging.getLogger("mineru.parser.api_server")


def _sanitize_surrogates(value: str) -> str:
    return "".join("\ufffd" if 0xD800 <= ord(ch) <= 0xDFFF else ch for ch in value)


def _sanitize_json_for_utf8(value: object) -> object:
    if isinstance(value, str):
        return _sanitize_surrogates(value)
    if isinstance(value, list):
        return [_sanitize_json_for_utf8(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_json_for_utf8(item) for item in value]
    if isinstance(value, dict):
        return {_sanitize_json_for_utf8(key): _sanitize_json_for_utf8(item) for key, item in value.items()}
    return value


def _text_utf8_bytes(value: str) -> bytes:
    return _sanitize_surrogates(value).encode("utf-8")


def _json_utf8_bytes(value: object) -> bytes:
    return json.dumps(_sanitize_json_for_utf8(value), ensure_ascii=False).encode("utf-8")


class ParseServerStartupError(RuntimeError):
    """Raised when the parse server cannot start because of local setup."""


# ── literal type aliases ────────────────────────────────────────────

JobStatus = Literal["queued", "running", "completed", "partial", "failed", "canceled"]
"""Job lifecycle states."""

FileStatus = Literal["queued", "running", "completed", "failed"]
"""Per-file status within a job."""

UploadStatus = Literal["pending", "completed", "cancelled", "expired"]
"""Upload lifecycle states."""

OutputFormat = Literal[
    "markdown",
    "middle_json",
    "content_list",
    "structured_content",
    "html",
    "latex",
    "docx",
    "zip",
]
"""Output artifact formats."""

OutputFormatReqToken = Literal["html", "latex", "docx"]
"""Output formats that require a valid API key."""

SourceType = Literal["file_id", "url", "inline", "local"]
"""File source types."""

AccessLevel = Literal["anonymous", "registered"]
"""Account access level determined by API Key."""

FilePurpose = Literal["parse", "parse_output", "input_image"]
"""File purpose: source files, parse artifacts, or chat/response input images."""

# ── helper ───────────────────────────────────────────────────────────


def _env_flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _install_managed_parse_server_stdin_watcher(server: Any) -> threading.Thread | None:
    if not _env_flag(_MANAGED_PARSE_SERVER_ENV, default=False):
        return None

    def _watch_stdin_for_eof() -> None:
        stdin_stream = getattr(sys.stdin, "buffer", sys.stdin)
        try:
            stdin_stream.read()
        except Exception:
            return
        server.should_exit = True

    watcher = threading.Thread(
        target=_watch_stdin_for_eof,
        name="mineru-managed-parse-server-stdin-sentinel",
        daemon=True,
    )
    watcher.start()
    return watcher


# ═══════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════

_PYDANTIC_CONFIG = ConfigDict(extra="forbid")


# ── Error ────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: str
    code: str | None = None
    message: str
    param: str | None = None


class ErrorResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    error: ErrorDetail


class ApiServerError(Exception):
    """Structured API error raised by parser API business logic."""

    def __init__(self, status_code: int, error: ErrorDetail) -> None:
        super().__init__(error.message)
        self.status_code = status_code
        self.error = error


def _raise_api_error(
    status_code: int,
    *,
    error_type: str,
    code: str | None,
    message: str,
    param: str | None = None,
) -> NoReturn:
    raise ApiServerError(
        status_code,
        ErrorDetail(
            type=error_type,
            code=code,
            message=message,
            param=param,
        ),
    )


def _error_response(status_code: int, error: ErrorDetail) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error).model_dump(by_alias=True),
    )


def _error_type_for_status(status_code: int) -> str:
    if status_code == 401:
        return "authentication_error"
    if status_code == 403:
        return "permission_error"
    if 400 <= status_code < 500:
        return "invalid_request_error"
    return "api_error"


def _validation_error_param(exc: RequestValidationError) -> str | None:
    for err in exc.errors():
        loc = err.get("loc", ())
        parts = [str(part) for part in loc if part not in ("body", "query", "path")]
        if parts:
            return ".".join(parts)
    return None


def _validation_error_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Invalid request."
    first = errors[0]
    loc = first.get("loc", ())
    parts = [str(part) for part in loc if part not in ("body", "query", "path")]
    field = ".".join(parts)
    msg = str(first.get("msg", "Invalid value"))
    if field:
        return f"Invalid request: {field}: {msg}"
    return f"Invalid request: {msg}"


def _http_exception_error(exc: HTTPException) -> ErrorDetail:
    detail = exc.detail
    if isinstance(detail, dict):
        candidate = detail.get("error")
        if isinstance(candidate, dict):
            return ErrorDetail.model_validate(candidate)
        nested = detail.get("detail")
        if isinstance(nested, dict) and isinstance(nested.get("error"), dict):
            return ErrorDetail.model_validate(nested["error"])
    message = str(detail) if detail else "HTTP error"
    return ErrorDetail(
        type=_error_type_for_status(exc.status_code),
        code="api_error" if exc.status_code >= 500 else "invalid_request",
        message=message,
    )


# ── Health ───────────────────────────────────────────────────────────


class ModelHealthStatus(BaseModel):
    model_config = _PYDANTIC_CONFIG
    pipeline: str = "ok"
    vlm: str = "ok"
    html: str = "ok"


class HealthFeatures(BaseModel):
    model_config = _PYDANTIC_CONFIG
    webhook: bool = False
    output_formats: list[OutputFormat] = Field(default_factory=lambda: list(_LOCAL_PARSE_OUTPUT_FORMATS))
    sources: list[SourceType] = Field(default_factory=lambda: list(_BASE_PARSE_SOURCES))


class HealthResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    status: str = "ok"
    version: str
    parser_version: str | None = None
    models: ModelHealthStatus | None = None
    features: HealthFeatures = Field(default_factory=HealthFeatures)


# ── Models API ───────────────────────────────────────────────────────


class ModelInfo(BaseModel):
    model_config = _PYDANTIC_CONFIG
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "mineru"
    description: str | None = None


class ModelListResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ── Tiers API ────────────────────────────────────────────────────────


class TierInfo(BaseModel):
    model_config = _PYDANTIC_CONFIG
    id: Tier
    description: str
    current_model: str | None = None


class TierListResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    object: Literal["list"] = "list"
    data: list[TierInfo]


# ── Upload ───────────────────────────────────────────────────────────


class ExpiresAfter(BaseModel):
    model_config = _PYDANTIC_CONFIG
    anchor: Literal["created_at"] = "created_at"
    seconds: int = Field(default=3600, ge=3600, le=2592000)


class CreateUploadRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    filename: str = Field(min_length=1)
    bytes: int = Field(gt=0)
    mime_type: str = Field(min_length=1)
    purpose: Literal["parse", "input_image"] = "parse"
    sha256sum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    expires_after: ExpiresAfter | None = None


class CompleteUploadRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    sha256sum: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")


class UploadResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    id: str
    object: Literal["upload"] = "upload"
    bytes: int
    created_at: int
    expires_at: int
    filename: str
    purpose: FilePurpose = "parse"
    mime_type: str
    sha256sum: str | None = None
    status: UploadStatus
    upload_url: str | None = None
    upload_method: Literal["PUT"] | None = None
    upload_headers: dict[str, str] | None = None
    file: FileObjectModel | None = None


# ── File ─────────────────────────────────────────────────────────────


class FileObjectModel(BaseModel):
    """A file resource (source or parse output)."""

    model_config = _PYDANTIC_CONFIG
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    expires_at: int | None = None
    filename: str
    purpose: FilePurpose
    sha256sum: str | None = None


class FileDeletionResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    id: str
    object: Literal["file"] = "file"
    deleted: bool = True


class FileListResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    object: Literal["list"] = "list"
    data: list[FileObjectModel]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


# ── Job sources ──────────────────────────────────────────────────────


class FileIdSource(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["file_id"] = "file_id"
    file_id: str


class UrlSource(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["url"] = "url"
    url: str


class InlineSource(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["inline"] = "inline"
    name: str = Field(min_length=1)
    data: str = Field(min_length=1)


class LocalSource(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["local"] = "local"
    path: str = Field(min_length=1)


FileSource = Annotated[
    FileIdSource | UrlSource | InlineSource | LocalSource,
    Field(discriminator="type"),
]


class JobFileEntry(BaseModel):
    model_config = _PYDANTIC_CONFIG
    source: FileSource
    page_range: str | None = None


class CallbackConfig(BaseModel):
    model_config = _PYDANTIC_CONFIG
    url: str
    secret: str | None = None


class CreateJobRequest(BaseModel):
    model_config = _PYDANTIC_CONFIG
    files: list[JobFileEntry] = Field(min_length=1)
    tier: Tier | None = None
    output_formats: list[OutputFormat] = ["markdown"]
    callback: CallbackConfig | None = None


# ── Job responses ────────────────────────────────────────────────────


class JobLinks(BaseModel):
    model_config = _PYDANTIC_CONFIG
    self: str
    cancel: str


class JobProgress(BaseModel):
    model_config = _PYDANTIC_CONFIG
    completed: int = 0
    failed: int = 0
    total: int = 0


class FileParseInfo(BaseModel):
    model_config = _PYDANTIC_CONFIG
    model_used: str | None = None
    duration_ms: int | None = None
    parser_version: str | None = None


class OutputFileRef(BaseModel):
    model_config = _PYDANTIC_CONFIG
    file_id: str
    bytes: int


class OutputFiles(BaseModel):
    """Per-file output artifacts. Content is downloaded via the Files API."""

    model_config = _PYDANTIC_CONFIG
    markdown: OutputFileRef | None = None
    middle_json: OutputFileRef | None = None
    content_list: OutputFileRef | None = None
    structured_content: OutputFileRef | None = None
    html: OutputFileRef | None = None
    latex: OutputFileRef | None = None
    docx: OutputFileRef | None = None
    zip: OutputFileRef | None = None


class JobFileResult(BaseModel):
    model_config = _PYDANTIC_CONFIG
    file_id: str | None = None
    name: str
    page_range: str
    status: FileStatus
    parse: FileParseInfo | None = None
    output_files: OutputFiles | None = None
    error: ErrorDetail | None = None


class JobAsyncResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    job_id: str
    status: JobStatus
    created_at: str  # ISO-8601 UTC
    started_at: str | None = None
    finished_at: str | None = None
    tier: Tier
    output_formats: list[OutputFormat]
    access_level: AccessLevel
    progress: JobProgress | None = None
    files: list[JobFileResult]
    links: JobLinks


class JobListItem(BaseModel):
    model_config = _PYDANTIC_CONFIG
    job_id: str
    status: JobStatus
    created_at: str
    file_count: int


class JobListResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    object: Literal["list"] = "list"
    data: list[JobListItem]
    first_id: str | None = None
    last_id: str | None = None
    has_more: bool = False


class JobCancelResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    job_id: str
    status: Literal["canceled"] = "canceled"
    canceled_at: str


# ── Usage ────────────────────────────────────────────────────────────


class UsageBillingPeriod(BaseModel):
    model_config = _PYDANTIC_CONFIG
    start: str  # ISO-8601
    end: str | None = None


class UsageCurrent(BaseModel):
    model_config = _PYDANTIC_CONFIG
    pages_processed: int = 0
    files_processed: int = 0
    jobs_created: int = 0


class UsageLimits(BaseModel):
    model_config = _PYDANTIC_CONFIG
    max_pages_per_file: int = 1000
    max_file_size_bytes: int = 209715200
    max_files_per_job: int = 100
    max_concurrent_jobs: int = 1
    max_file_retention_days: int | None = None


class UsageResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    object: Literal["usage"] = "usage"
    access_level: AccessLevel
    billing_period: UsageBillingPeriod
    current: UsageCurrent
    limits: UsageLimits


# ═══════════════════════════════════════════════════════════════════════
#  STORE
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class _UploadRecord:
    id: str
    filename: str
    bytes: int
    mime_type: str
    created_at: int
    expires_at: int
    purpose: FilePurpose = "parse"
    sha256sum: str | None = None
    status: UploadStatus = "pending"
    file_id: str | None = None  # set on complete


@dataclass
class _FileRecord:
    id: str
    filename: str
    bytes: int
    created_at: int
    purpose: FilePurpose
    sha256sum: str | None = None
    expires_at: int | None = None
    upload_id: str | None = None


class FileStore:
    """Content-addressed blob storage with in-memory metadata."""

    def __init__(self, root: pathlib.Path) -> None:
        self._root = root
        self._blobs = root / "blobs"
        self._blobs.mkdir(parents=True, exist_ok=True)
        self._uploads: dict[str, _UploadRecord] = {}
        self._files: dict[str, _FileRecord] = {}

    # ── ID generation ────────────────────────────────────────────────

    @staticmethod
    def _new_upload_id() -> str:
        return "upload_" + secrets.token_hex(12)

    @staticmethod
    def _new_file_id() -> str:
        return "file-" + secrets.token_hex(12)

    # ── blob helpers ─────────────────────────────────────────────────

    @staticmethod
    def _blob_path(sha256hex: str) -> pathlib.PurePosixPath:
        return pathlib.PurePosixPath(sha256hex[:2]) / sha256hex[2:]

    def _blob_abs(self, sha256hex: str) -> pathlib.Path:
        return self._blobs / self._blob_path(sha256hex)

    def blob_exists(self, sha256hex: str) -> bool:
        return self._blob_abs(sha256hex).is_file()

    def store_blob(self, data: bytes, *, sha256hex: str) -> None:
        p = self._blob_abs(sha256hex)
        if not p.is_file():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    def read_blob(self, sha256hex: str) -> bytes:
        p = self._blob_abs(sha256hex)
        if not p.is_file():
            raise FileNotFoundError(sha256hex)
        return p.read_bytes()

    # ── uploads ──────────────────────────────────────────────────────

    def create_upload(self, req: CreateUploadRequest, *, base_url: str = "") -> UploadResponse:
        now = int(time.time())
        expires_seconds = req.expires_after.seconds if req.expires_after else 3600

        # sha256sum dedup
        if req.sha256sum and self.blob_exists(req.sha256sum):
            file_id = self._new_file_id()
            self._files[file_id] = _FileRecord(
                id=file_id,
                filename=req.filename,
                bytes=req.bytes,
                created_at=now,
                purpose=req.purpose,
                sha256sum=req.sha256sum,
                expires_at=now + expires_seconds,
            )
            upload_id = self._new_upload_id()
            self._uploads[upload_id] = _UploadRecord(
                id=upload_id,
                filename=req.filename,
                bytes=req.bytes,
                mime_type=req.mime_type,
                created_at=now,
                expires_at=now + expires_seconds,
                purpose=req.purpose,
                sha256sum=req.sha256sum,
                status="completed",
                file_id=file_id,
            )
            return self._make_upload_response(self._uploads[upload_id], base_url=base_url, expose_upload_url=True)

        upload_id = self._new_upload_id()
        self._uploads[upload_id] = _UploadRecord(
            id=upload_id,
            filename=req.filename,
            bytes=req.bytes,
            mime_type=req.mime_type,
            created_at=now,
            expires_at=now + expires_seconds,
            purpose=req.purpose,
            sha256sum=req.sha256sum,
        )
        return self._make_upload_response(self._uploads[upload_id], base_url=base_url, expose_upload_url=True)

    def get_upload(self, upload_id: str) -> _UploadRecord:
        rec = self._uploads.get(upload_id)
        if rec is None:
            _raise_api_error(
                404,
                error_type="invalid_request_error",
                code="upload_not_found",
                message=f"Upload {upload_id} not found",
            )
        return rec

    def _read_upload_data(self, upload_id: str) -> bytes:
        p = self._blobs / "_uploads" / upload_id
        if not p.is_file():
            _raise_api_error(
                409,
                error_type="invalid_request_error",
                code="upload_not_ready",
                message="Upload bytes not yet received",
            )
        return p.read_bytes()

    def complete_upload(self, upload_id: str, sha256hex: str | None) -> UploadResponse:
        rec = self.get_upload(upload_id)
        if rec.status == "completed":
            _raise_api_error(
                409,
                error_type="invalid_request_error",
                code="upload_already_terminal",
                message="Upload is already completed",
            )
        if rec.status in ("cancelled", "expired"):
            _raise_api_error(
                409,
                error_type="invalid_request_error",
                code="upload_already_terminal",
                message=f"Upload is {rec.status}",
            )

        # compute sha256 from uploaded data if not provided
        data = self._read_upload_data(upload_id)
        actual_sha = hashlib.sha256(data).hexdigest()

        if sha256hex and sha256hex != actual_sha:
            _raise_api_error(
                400,
                error_type="invalid_request_error",
                code="file_hash_mismatch",
                message="SHA-256 mismatch",
            )
        if rec.sha256sum and rec.sha256sum != actual_sha:
            _raise_api_error(
                400,
                error_type="invalid_request_error",
                code="file_hash_mismatch",
                message="SHA-256 mismatch",
            )

        # move from upload blob to content-addressed blob
        sha = sha256hex or actual_sha
        if not self.blob_exists(sha):
            self.store_blob(data, sha256hex=sha)
        # remove temp upload blob
        (self._blobs / "_uploads" / upload_id).unlink(missing_ok=True)

        now = int(time.time())
        file_id = self._new_file_id()
        self._files[file_id] = _FileRecord(
            id=file_id,
            filename=rec.filename,
            bytes=rec.bytes,
            created_at=now,
            purpose=rec.purpose,
            sha256sum=sha,
            expires_at=rec.expires_at,
            upload_id=upload_id,
        )
        rec.status = "completed"
        rec.file_id = file_id
        rec.sha256sum = sha
        return self._make_upload_response(rec)

    def cancel_upload(self, upload_id: str) -> UploadResponse:
        rec = self.get_upload(upload_id)
        if rec.status in ("completed", "cancelled", "expired"):
            _raise_api_error(
                409,
                error_type="invalid_request_error",
                code="upload_already_terminal",
                message=f"Upload is {rec.status}",
            )
        rec.status = "cancelled"
        return self._make_upload_response(rec)

    def store_upload_data(self, upload_id: str, data: bytes) -> None:
        """Store raw bytes for an upload (before complete)."""
        rec = self.get_upload(upload_id)
        if rec.status != "pending":
            _raise_api_error(
                409,
                error_type="invalid_request_error",
                code="upload_already_terminal",
                message=f"Upload is {rec.status}",
            )
        # store temporarily under upload_id in blobs dir
        p = self._blobs / "_uploads" / upload_id
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def _make_upload_response(
        self, rec: _UploadRecord, *, base_url: str = "", expose_upload_url: bool = False
    ) -> UploadResponse:
        resp = UploadResponse(
            id=rec.id,
            bytes=rec.bytes,
            created_at=rec.created_at,
            expires_at=rec.expires_at,
            filename=rec.filename,
            mime_type=rec.mime_type,
            purpose=rec.purpose,
            sha256sum=rec.sha256sum,
            status=rec.status,
        )
        if expose_upload_url and rec.status == "pending":
            resp.upload_url = f"{base_url}/v1/uploads/{rec.id}/content"
            resp.upload_method = "PUT"
            resp.upload_headers = {"Content-Type": rec.mime_type}
        elif rec.status == "completed" and rec.file_id and rec.file_id in self._files:
            resp.file = self._make_file_object(self._files[rec.file_id])
        return resp

    # ── files ────────────────────────────────────────────────────────

    def create_file_for_output(self, filename: str, data: bytes, *, sha256hex: str | None = None) -> str:
        """Store parse-output data and return a file_id."""
        sha = sha256hex or hashlib.sha256(data).hexdigest()
        self.store_blob(data, sha256hex=sha)
        now = int(time.time())
        file_id = self._new_file_id()
        self._files[file_id] = _FileRecord(
            id=file_id,
            filename=filename,
            bytes=len(data),
            created_at=now,
            purpose="parse_output",
            sha256sum=sha,
        )
        return file_id

    def get_file(self, file_id: str) -> _FileRecord:
        rec = self._files.get(file_id)
        if rec is None:
            _raise_api_error(
                404,
                error_type="invalid_request_error",
                code="file_not_found",
                message=f"File {file_id} not found",
            )
        return rec

    @staticmethod
    def _make_file_object(rec: _FileRecord) -> FileObjectModel:
        return FileObjectModel(
            id=rec.id,
            bytes=rec.bytes,
            created_at=rec.created_at,
            expires_at=rec.expires_at,
            filename=rec.filename,
            purpose=rec.purpose,
            sha256sum=rec.sha256sum,
        )

    def read_file_data(self, file_id: str) -> bytes:
        rec = self.get_file(file_id)
        if rec.purpose != "parse_output":
            _raise_api_error(
                403,
                error_type="permission_error",
                code="feature_requires_api_key",
                message="Source files cannot be downloaded",
            )
        if rec.sha256sum is None:
            _raise_api_error(
                500,
                error_type="api_error",
                code="internal_error",
                message="File has no sha256sum",
            )
        return self.read_blob(rec.sha256sum)

    def delete_file(self, file_id: str) -> None:
        if file_id not in self._files:
            _raise_api_error(
                404,
                error_type="invalid_request_error",
                code="file_not_found",
                message=f"File {file_id} not found",
            )
        del self._files[file_id]

    def list_files(
        self,
        *,
        after: str | None,
        limit: int,
        order: str,
        purpose: FilePurpose | None,
    ) -> FileListResponse:
        recs = list(self._files.values())
        # filter
        if purpose is not None:
            recs = [r for r in recs if r.purpose == purpose]
        # sort by created_at
        reverse = order == "desc"
        recs.sort(key=lambda r: r.created_at, reverse=reverse)
        # cursor
        start = 0
        if after is not None:
            for i, r in enumerate(recs):
                if r.id == after:
                    start = i + 1
                    break
        page = recs[start : start + limit]
        return FileListResponse(
            data=[self._make_file_object(r) for r in page],
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
            has_more=(start + limit) < len(recs),
        )

    # ── defaults → app state ─────────────────────────────────────────

    def install(self, app_state: Any) -> None:
        app_state.file_store = self


def _build_self_contained_zip_output(result: Any, output_stem: str) -> bytes:
    """构建自包含解析 zip，确保只请求 zip 时也能恢复 middle_json、图片和 model_output。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{output_stem}.markdown", _text_utf8_bytes(result.markdown() or ""))
        zf.writestr(f"{output_stem}.middle_json", _json_utf8_bytes(result.to_dict(skip_defaults=True)))
        zf.writestr(f"{output_stem}.content_list", _json_utf8_bytes(result.content_list()))
        zf.writestr(f"{output_stem}.structured_content", _json_utf8_bytes(result.structured_content()))

        for img_path, img_bytes in sorted(result.images().items()):
            safe_img_path = validate_image_sidecar_path(img_path)
            zf.writestr(safe_img_path, img_bytes)

        model_output = getattr(result, "_model_output", None)
        if model_output is not None:
            zf.writestr(
                f"{output_stem}_model_output.json",
                _text_utf8_bytes(
                    json.dumps(
                        _sanitize_json_for_utf8(model_output),
                        ensure_ascii=False,
                        indent=4,
                    )
                ),
            )
    return buf.getvalue()


# ═══════════════════════════════════════════════════════════════════════
#  DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════


def _get_store(request: Request) -> FileStore:
    return request.app.state.file_store


def _get_job_store(request: Request) -> JobStore:
    return request.app.state.job_store


def _resolve_access_level(request: Request) -> AccessLevel:
    """Return ``"registered"`` when the server has an API key configured
    (the middleware already validated it), ``"anonymous"`` otherwise."""
    return (
        "registered" if request.app.state.api_key else "anonymous"
    )  # ═══════════════════════════════════════════════════════════════════════


#  JOB STORE
# ═══════════════════════════════════════════════════════════════════════

_SUPPORTED_SUFFIXES: dict[str, str] = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".jp2": "image",
    ".webp": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".docx": "docx",
    ".pptx": "pptx",
    ".xlsx": "xlsx",
    ".html": "html",
    ".htm": "html",
}

_OUTPUT_FORMATS_LOCAL = set(_LOCAL_PARSE_OUTPUT_FORMATS)


def _parse_sources_for_server(*, allow_local_source: bool) -> list[SourceType]:
    sources = list(_BASE_PARSE_SOURCES)
    if allow_local_source:
        sources.append("local")
    return sources


def _decode_inline_data(data: str) -> bytes:
    try:
        return base64.b64decode(data, validate=True)
    except Exception as exc:
        raise ValueError("inline source data must be valid base64") from exc


def _validate_source_policy(
    source: FileSource,
    *,
    allow_local_source: bool,
    max_inline_bytes: int,
    allow_http_source: bool,
) -> None:
    if isinstance(source, UrlSource):
        scheme = urlparse(source.url).scheme.lower()
        if scheme == "https":
            return
        if scheme == "http" and allow_http_source:
            return
        if scheme == "http":
            raise ValueError("url source must use https unless --allow-http-source is enabled")
        raise ValueError("url source must use http or https")
    if isinstance(source, InlineSource):
        data = _decode_inline_data(source.data)
        if len(data) > max_inline_bytes:
            raise ValueError(f"inline source exceeds max_inline_bytes ({max_inline_bytes})")
        return
    if isinstance(source, LocalSource):
        if not allow_local_source:
            raise ValueError("local source is disabled; enable --allow-local-source")
        return


def _validate_job_source_policy(
    req: CreateJobRequest,
    *,
    allow_local_source: bool,
    max_inline_bytes: int,
    allow_http_source: bool,
) -> None:
    for index, entry in enumerate(req.files):
        try:
            _validate_source_policy(
                entry.source,
                allow_local_source=allow_local_source,
                max_inline_bytes=max_inline_bytes,
                allow_http_source=allow_http_source,
            )
        except ValueError as exc:
            _raise_api_error(
                400,
                error_type="invalid_request_error",
                code="invalid_request",
                message=str(exc),
                param=f"files.{index}.source",
            )

@dataclass
class _JobRecord:
    id: str
    status: JobStatus = "queued"
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    tier: Tier = _DEFAULT_API_SERVER_TIER
    output_formats: list[OutputFormat] = None  # type: ignore[assignment]
    progress: JobProgress | None = None
    files: list[JobFileResult] = None  # type: ignore[assignment]
    links: JobLinks | None = None

    def __post_init__(self) -> None:
        if self.files is None:
            self.files = []
        if self.output_formats is None:
            self.output_formats = ["markdown"]
        if self.progress is None:
            self.progress = JobProgress()


class JobStore:
    def __init__(self, concurrency: int = 1) -> None:
        self._jobs: dict[str, _JobRecord] = {}
        self._semaphore = asyncio.Semaphore(max(1, concurrency))
        self._started_at = JobStore._now()

    @staticmethod
    def _new_job_id() -> str:
        return "job_" + secrets.token_hex(12)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    def create(self, req: CreateJobRequest, file_store: FileStore) -> _JobRecord:
        if req.tier is None:
            raise ValueError("CreateJobRequest.tier must be resolved before job creation")
        job_id = self._new_job_id()
        now = self._now()
        rec = _JobRecord(
            id=job_id,
            status="queued",
            created_at=now,
            tier=req.tier,
            output_formats=req.output_formats,
            links=JobLinks(
                self=f"/v1/parse/jobs/{job_id}",
                cancel=f"/v1/parse/jobs/{job_id}",
            ),
        )
        for entry in req.files:
            name = _source_name(entry.source, file_store)
            rec.files.append(
                JobFileResult(
                    file_id=entry.source.file_id if isinstance(entry.source, FileIdSource) else None,
                    name=name,
                    page_range=entry.page_range or "",
                    status="queued",
                )
            )
        rec.progress.total = len(rec.files)
        self._jobs[job_id] = rec
        return rec

    def get(self, job_id: str) -> _JobRecord:
        rec = self._jobs.get(job_id)
        if rec is None:
            _raise_api_error(
                404,
                error_type="invalid_request_error",
                code="job_not_found",
                message=f"Job {job_id} not found",
            )
        return rec

    def cancel(self, job_id: str) -> _JobRecord:
        rec = self.get(job_id)
        if rec.status in ("completed", "partial", "failed", "canceled"):
            _raise_api_error(
                409,
                error_type="invalid_request_error",
                code="job_already_terminal",
                message=f"Job is {rec.status}",
            )
        rec.status = "canceled"
        return rec

    def list_jobs(
        self,
        *,
        status_filter: str | None,
        limit: int,
        after: str | None,
        created_after: str | None,
        order: str = "desc",
    ) -> JobListResponse:
        recs = list(self._jobs.values())
        if status_filter:
            allowed = set(status_filter.split(","))
            recs = [r for r in recs if r.status in allowed]
        if created_after:
            recs = [r for r in recs if r.created_at >= created_after]
        reverse = order == "desc"
        recs.sort(key=lambda r: r.created_at, reverse=reverse)
        start = 0
        if after:
            for i, r in enumerate(recs):
                if r.id == after:
                    start = i + 1
                    break
        page = recs[start : start + limit]
        return JobListResponse(
            data=[
                JobListItem(
                    job_id=r.id,
                    status=r.status,
                    created_at=r.created_at,
                    file_count=len(r.files),
                )
                for r in page
            ],
            first_id=page[0].id if page else None,
            last_id=page[-1].id if page else None,
            has_more=(start + limit) < len(recs),
        )

    def build_response(self, rec: _JobRecord, access_level: AccessLevel = "registered") -> JobAsyncResponse:
        return JobAsyncResponse(
            job_id=rec.id,
            status=rec.status,
            created_at=rec.created_at,
            started_at=rec.started_at,
            finished_at=rec.finished_at,
            tier=rec.tier,
            output_formats=rec.output_formats,
            access_level=access_level,
            progress=rec.progress,
            files=rec.files,
            links=rec.links or JobLinks(self="", cancel=""),
        )

    def usage(self, access_level: AccessLevel) -> UsageResponse:
        completed = sum(1 for j in self._jobs.values() if j.status in ("completed", "partial"))
        processed_page_count = sum(
            sum(_count_pages_in_range(fr.page_range) for fr in j.files if fr.status == "completed") for j in self._jobs.values()
        )
        return UsageResponse(
            access_level=access_level,
            billing_period=UsageBillingPeriod(start=self._started_at),
            current=UsageCurrent(
                pages_processed=processed_page_count,
                files_processed=completed,
                jobs_created=len(self._jobs),
            ),
            limits=UsageLimits(
                max_concurrent_jobs=self._semaphore._value,
            ),
        )

    def install(self, app_state: Any) -> None:
        app_state.job_store = self


# ── source helpers ───────────────────────────────────────────────────


def _source_name(source: FileSource, file_store: FileStore | None = None) -> str:
    if isinstance(source, FileIdSource) and file_store is not None:
        try:
            return file_store.get_file(source.file_id).filename
        except Exception:
            pass
        return source.file_id
    if isinstance(source, UrlSource):
        return source.url.rsplit("/", 1)[-1].split("?")[0] or "unknown"
    if isinstance(source, InlineSource):
        return source.name
    if isinstance(source, LocalSource):
        return pathlib.Path(source.path).name
    return "unknown"


def _compact_page_numbers(page_numbers: list[int]) -> str:
    if not page_numbers:
        return ""
    ordered = sorted(set(page_numbers))
    ranges: list[str] = []
    start = prev = ordered[0]
    for page in ordered[1:]:
        if page == prev + 1:
            prev = page
            continue
        ranges.append(str(start) if start == prev else f"{start}~{prev}")
        start = prev = page
    ranges.append(str(start) if start == prev else f"{start}~{prev}")
    return ",".join(ranges)


def _page_range_from_result_pages(pages: list[Any]) -> str:
    page_numbers: list[int] = []
    for i, page in enumerate(pages):
        page_idx = getattr(page, "page_idx", i)
        if isinstance(page_idx, int):
            page_numbers.append(page_idx + 1)
    return _compact_page_numbers(page_numbers)


def _count_pages_in_range(page_range: str) -> int:
    total = 0
    for part in page_range.split(","):
        token = part.strip()
        if not token:
            continue
        if "~" not in token:
            total += 1
            continue
        left, right = token.split("~", 1)
        try:
            start = int(left)
            end = int(right)
        except ValueError:
            total += 1
            continue
        total += max(0, end - start + 1)
    return total


async def _extract_bytes(
    source: FileSource,
    file_store: FileStore,
    *,
    url_timeout: int = 60,
    allow_local_source: bool = False,
    max_inline_bytes: int = _MAX_INLINE_BYTES_DEFAULT,
    allow_http_source: bool = False,
) -> bytes:
    _validate_source_policy(
        source,
        allow_local_source=allow_local_source,
        max_inline_bytes=max_inline_bytes,
        allow_http_source=allow_http_source,
    )
    if isinstance(source, FileIdSource):
        rec = file_store.get_file(source.file_id)
        if rec.sha256sum is None:
            raise ValueError("File has no content")
        return file_store.read_blob(rec.sha256sum)
    if isinstance(source, UrlSource):
        async with httpx.AsyncClient(timeout=url_timeout) as cli:
            r = await cli.get(source.url)
            r.raise_for_status()
            return r.content
    if isinstance(source, InlineSource):
        return _decode_inline_data(source.data)
    if isinstance(source, LocalSource):
        return pathlib.Path(source.path).expanduser().resolve(strict=False).read_bytes()
    raise ValueError(f"Unknown source type: {type(source)}")


def _suffix_type(filename: str) -> str:
    ext = pathlib.Path(filename).suffix.lower()
    return _SUPPORTED_SUFFIXES.get(ext, "")


async def _run_job(
    rec: _JobRecord,
    req: CreateJobRequest,
    file_store: FileStore,
    *,
    server_backend: str,
    language: str,
    ocr_mode: str,
    image_analysis: bool,
    effort: str = DEFAULT_HYBRID_EFFORT,
    url_timeout: int = 60,
    allow_local_source: bool = False,
    max_inline_bytes: int = _MAX_INLINE_BYTES_DEFAULT,
    allow_http_source: bool = False,
) -> None:
    rec.status = "running"
    rec.started_at = JobStore._now()

    with tempfile.TemporaryDirectory(prefix="mineru_job_") as tmpdir:
        for i, entry in enumerate(req.files):
            if rec.status == "canceled":
                break
            fr = rec.files[i]
            try:
                file_started = time.monotonic()
                data = await _extract_bytes(
                    entry.source,
                    file_store,
                    url_timeout=url_timeout,
                    allow_local_source=allow_local_source,
                    max_inline_bytes=max_inline_bytes,
                    allow_http_source=allow_http_source,
                )
                stype = _suffix_type(fr.name)
                if not stype:
                    raise ValueError(f"Unsupported file type: {fr.name}")

                # write to temp file for parsers that require a path
                suffix = pathlib.Path(fr.name).suffix or ".pdf"
                tmp_path = pathlib.Path(tmpdir) / f"input_{i}{suffix}"
                tmp_path.write_bytes(data)

                page_range = entry.page_range or ""

                result = await parse_async(
                    str(tmp_path),
                    tier=rec.tier,
                    backend=server_backend,
                    language=language,
                    ocr_mode=ocr_mode,
                    effort=effort,
                    disable_image_analysis=not image_analysis,
                    page_range=page_range,
                )

                # collect outputs
                out_formats = set(rec.output_formats)
                output_files = OutputFiles()

                for fmt in (
                    "markdown",
                    "middle_json",
                    "content_list",
                    "structured_content",
                ):
                    if fmt not in out_formats:
                        continue
                    if fmt == "markdown":
                        md = result.markdown()
                        content_bytes = _text_utf8_bytes(md or "")
                        sha = hashlib.sha256(content_bytes).hexdigest()
                        file_store.store_blob(content_bytes, sha256hex=sha)
                        fid = file_store.create_file_for_output(f"{fr.name}.md", content_bytes, sha256hex=sha)
                        output_files.markdown = OutputFileRef(file_id=fid, bytes=len(content_bytes))
                    elif fmt == "middle_json":
                        mj = _json_utf8_bytes(result.to_dict(skip_defaults=True))
                        sha = hashlib.sha256(mj).hexdigest()
                        file_store.store_blob(mj, sha256hex=sha)
                        fid = file_store.create_file_for_output(f"{fr.name}.middle.json", mj, sha256hex=sha)
                        output_files.middle_json = OutputFileRef(file_id=fid, bytes=len(mj))
                    elif fmt == "content_list":
                        cl = _json_utf8_bytes(result.content_list())
                        sha = hashlib.sha256(cl).hexdigest()
                        file_store.store_blob(cl, sha256hex=sha)
                        fid = file_store.create_file_for_output(f"{fr.name}.content_list.json", cl, sha256hex=sha)
                        output_files.content_list = OutputFileRef(file_id=fid, bytes=len(cl))
                    elif fmt == "structured_content":
                        cl2 = _json_utf8_bytes(result.structured_content())
                        sha = hashlib.sha256(cl2).hexdigest()
                        file_store.store_blob(cl2, sha256hex=sha)
                        fid = file_store.create_file_for_output(f"{fr.name}.structured_content.json", cl2, sha256hex=sha)
                        output_files.structured_content = OutputFileRef(file_id=fid, bytes=len(cl2))

                # zip
                if "zip" in out_formats:
                    zip_bytes = _build_self_contained_zip_output(result, pathlib.Path(fr.name).stem)
                    zip_sha = hashlib.sha256(zip_bytes).hexdigest()
                    file_store.store_blob(zip_bytes, sha256hex=zip_sha)
                    zip_fid = file_store.create_file_for_output(f"{fr.name}.zip", zip_bytes, sha256hex=zip_sha)
                    output_files.zip = OutputFileRef(file_id=zip_fid, bytes=len(zip_bytes))

                fr.status = "completed"
                fr.page_range = _page_range_from_result_pages(result.pages)
                fr.output_files = output_files

                fr.parse = FileParseInfo(
                    model_used=None,
                    duration_ms=int((time.monotonic() - file_started) * 1000),
                    parser_version=__version__,
                )
                fr.file_id = file_store.create_file_for_output(fr.name, data)
                rec.progress.completed += 1

            except Exception as exc:
                logger.exception(
                    "Parse-server job file failed: job_id=%s file=%r tier=%s page_range=%r",
                    rec.id,
                    fr.name,
                    rec.tier,
                    fr.page_range,
                )
                fr.status = "failed"
                fr.error = ErrorDetail(type="engine_error", code="parse_failed", message=str(exc))
                rec.progress.failed += 1

    if rec.status != "canceled":
        if rec.progress.failed == rec.progress.total:
            rec.status = "failed"
        elif rec.progress.failed > 0:
            rec.status = "partial"
        else:
            rec.status = "completed"
    rec.finished_at = JobStore._now()


# ═══════════════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════════════

_router = APIRouter(prefix="/v1")

# Reusable error response maps
_ERR_400: dict[int | str, dict[str, Any]] = {400: {"model": ErrorResponse}}
_ERR_401: dict[int | str, dict[str, Any]] = {401: {"model": ErrorResponse}}
_ERR_403: dict[int | str, dict[str, Any]] = {403: {"model": ErrorResponse}}
_ERR_404: dict[int | str, dict[str, Any]] = {404: {"model": ErrorResponse}}
_ERR_409: dict[int | str, dict[str, Any]] = {409: {"model": ErrorResponse}}
_ERR_413: dict[int | str, dict[str, Any]] = {413: {"model": ErrorResponse}}
_ERR_429: dict[int | str, dict[str, Any]] = {429: {"model": ErrorResponse}}


# ── Health ───────────────────────────────────────────────────────────


@_router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses={503: {"model": ErrorResponse}},
    tags=["Health"],
)
async def get_health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        version=__version__,
        parser_version=__version__,
        models=ModelHealthStatus(),
        features=HealthFeatures(sources=_parse_sources_for_server(allow_local_source=request.app.state.allow_local_source)),
    )


# ── Models ───────────────────────────────────────────────────────────


@_router.get(
    "/models",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Models"],
)
async def list_models(request: Request) -> ModelListResponse:
    """List all available parsing models."""
    model_ids: list[str] = request.app.state.model_ids
    now = int(time.time())
    return ModelListResponse(
        data=[ModelInfo(id=mid, created=now) for mid in model_ids],
    )


@_router.get(
    "/models/{model}",
    response_model=ModelInfo,
    status_code=status.HTTP_200_OK,
    responses={**_ERR_404, **_ERR_403},
    tags=["Models"],
)
async def get_model(
    request: Request,
    model: str = Path(description="Model ID"),
) -> ModelInfo:
    """Retrieve a single model by ID."""
    model_ids: list[str] = request.app.state.model_ids
    if model not in model_ids:
        _raise_api_error(
            404,
            error_type="invalid_request_error",
            code="model_not_found",
            message=f"Model '{model}' not found",
        )
    now = int(time.time())
    return ModelInfo(id=model, created=now)


# ── Tiers ────────────────────────────────────────────────────────────


@_router.get(
    "/tiers",
    response_model=TierListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Tiers"],
)
async def list_tiers(request: Request) -> TierListResponse:
    """List all available parser tiers."""
    tiers: list[dict[str, Any]] = request.app.state.tiers
    return TierListResponse(
        data=[TierInfo(**p) for p in tiers],
    )


# ── Uploads ──────────────────────────────────────────────────────────


@_router.post(
    "/uploads",
    response_model=None,
    status_code=status.HTTP_200_OK,
    responses={200: {"model": UploadResponse}, **_ERR_400, **_ERR_413},
    tags=["Uploads"],
)
async def create_upload(
    body: CreateUploadRequest,
    request: Request,
    store: FileStore = Depends(_get_store),
) -> Response:
    """Create an Upload."""
    base = str(request.base_url).rstrip("/")
    resp = store.create_upload(body, base_url=base)
    if resp.status == "completed":
        return JSONResponse(content=resp.model_dump(by_alias=True), status_code=200)
    return JSONResponse(content=resp.model_dump(by_alias=True), status_code=200)


@_router.get(
    "/uploads/{upload_id}",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    responses={**_ERR_404},
    tags=["Uploads"],
)
async def get_upload(
    upload_id: str = Path(description="Upload ID (upload_...)"),
    store: FileStore = Depends(_get_store),
) -> UploadResponse:
    """Retrieve an Upload by ID."""
    return store._make_upload_response(store.get_upload(upload_id))


@_router.put(
    "/uploads/{upload_id}/content",
    response_model=None,
    status_code=status.HTTP_200_OK,
    responses={**_ERR_404, **_ERR_409},
    tags=["Uploads"],
)
async def upload_content(
    upload_id: str = Path(description="Upload ID"),
    data: bytes = Body(media_type="application/octet-stream"),
    store: FileStore = Depends(_get_store),
) -> Response:
    """Upload raw bytes for an upload (OpenAI-compatible)."""
    store.store_upload_data(upload_id, data)
    return Response(status_code=200)


@_router.post(
    "/uploads/{upload_id}/complete",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    responses={**_ERR_400, **_ERR_404, **_ERR_409},
    tags=["Uploads"],
)
async def complete_upload(
    upload_id: str = Path(description="Upload ID"),
    body: CompleteUploadRequest | None = None,
    store: FileStore = Depends(_get_store),
) -> UploadResponse:
    """Complete an upload, creating the File object."""
    return store.complete_upload(upload_id, body.sha256sum if body else None)


@_router.post(
    "/uploads/{upload_id}/cancel",
    response_model=UploadResponse,
    status_code=status.HTTP_200_OK,
    responses={**_ERR_404, **_ERR_409},
    tags=["Uploads"],
)
async def cancel_upload(
    upload_id: str = Path(description="Upload ID"),
    store: FileStore = Depends(_get_store),
) -> UploadResponse:
    """Cancel an in-progress upload."""
    return store.cancel_upload(upload_id)


# ── Files ────────────────────────────────────────────────────────────


@_router.get(
    "/files",
    response_model=FileListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Files"],
)
async def list_files(
    after: str | None = Query(default=None, description="Cursor: last_id from previous page"),
    limit: int = Query(default=100, ge=1, le=1000),
    order: Literal["asc", "desc"] = Query(default="desc"),
    purpose: FilePurpose | None = Query(default=None),
    store: FileStore = Depends(_get_store),
) -> FileListResponse:
    """List files for the current tenant."""
    return store.list_files(after=after, limit=limit, order=order, purpose=purpose)


@_router.get(
    "/files/{file_id}",
    response_model=FileObjectModel,
    status_code=status.HTTP_200_OK,
    responses={**_ERR_404},
    tags=["Files"],
)
async def get_file(
    file_id: str = Path(description="File ID (file-...)"),
    store: FileStore = Depends(_get_store),
) -> FileObjectModel:
    """Retrieve file metadata."""
    return store._make_file_object(store.get_file(file_id))


@_router.get(
    "/files/{file_id}/content",
    response_model=None,
    status_code=status.HTTP_200_OK,
    responses={
        200: {"description": "File content"},
        302: {"description": "Redirect to CDN (cloud mode)"},
        **_ERR_404,
        **_ERR_403,
    },
    tags=["Files"],
)
async def get_file_content(
    file_id: str = Path(description="File ID"),
    store: FileStore = Depends(_get_store),
) -> Response:
    """Download parse-output content."""
    data = store.read_file_data(file_id)
    return Response(content=data, media_type="application/octet-stream")


@_router.delete(
    "/files/{file_id}",
    response_model=FileDeletionResponse,
    status_code=status.HTTP_200_OK,
    responses={**_ERR_404},
    tags=["Files"],
)
async def delete_file(
    file_id: str = Path(description="File ID"),
    store: FileStore = Depends(_get_store),
) -> FileDeletionResponse:
    """Delete a file from the current tenant view."""
    store.delete_file(file_id)
    return FileDeletionResponse(id=file_id, deleted=True)


# ── Parse Jobs ───────────────────────────────────────────────────────


@_router.post(
    "/parse/jobs",
    response_model=None,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        202: {"model": JobAsyncResponse},
        **_ERR_400,
        **_ERR_403,
        **_ERR_429,
    },
    tags=["Jobs"],
)
async def create_job(
    body: CreateJobRequest,
    request: Request,
    file_store: FileStore = Depends(_get_store),
    job_store: JobStore = Depends(_get_job_store),
    access_level: AccessLevel = Depends(_resolve_access_level),
) -> Response:
    """Create a parse job."""
    if body.callback is not None:
        _raise_api_error(
            400,
            error_type="invalid_request_error",
            code="invalid_request",
            message=(
                "Webhook callback is not supported by this Local Parse Server. Use polling via GET /v1/parse/jobs/{job_id}."
            ),
        )

    # validate output formats — advanced formats require API key
    for fmt in body.output_formats:
        if fmt in ("html", "latex", "docx") and access_level == "anonymous":
            _raise_api_error(
                403,
                error_type="permission_error",
                code="feature_requires_api_key",
                message=f"Output format '{fmt}' requires an API key",
            )
        if fmt not in _OUTPUT_FORMATS_LOCAL:
            _raise_api_error(
                400,
                error_type="invalid_request_error",
                code="invalid_request",
                message=f"Unknown output format: {fmt}",
            )

    # 按请求 tier 选择启动时预先解析好的 runtime，避免所有 job 共享默认 effort。
    body.tier = body.tier or request.app.state.default_tier
    runtime_options: dict[Tier, ParserRuntimeOptions] = request.app.state.tier_runtime_options
    runtime = runtime_options.get(body.tier)
    if runtime is None:
        _raise_api_error(
            400,
            error_type="invalid_request_error",
            code="invalid_request",
            message=f"Tier '{body.tier}' not available in this server",
        )

    _validate_job_source_policy(
        body,
        allow_local_source=request.app.state.allow_local_source,
        max_inline_bytes=request.app.state.max_inline_bytes,
        allow_http_source=request.app.state.allow_http_source,
    )

    rec = job_store.create(body, file_store)
    backend = runtime.backend
    url_timeout_val: int = request.app.state.url_timeout
    allow_local_source_val: bool = request.app.state.allow_local_source
    max_inline_bytes_val: int = request.app.state.max_inline_bytes
    allow_http_source_val: bool = request.app.state.allow_http_source
    language_val: str = request.app.state.language
    ocr_mode_val: str = request.app.state.ocr_mode
    effort_val = runtime.effort
    image_analysis_val: bool = request.app.state.image_analysis

    # async — fire and forget
    async def _bg_run() -> None:
        async with job_store._semaphore:
            await _run_job(
                rec,
                body,
                file_store,
                server_backend=backend,
                language=language_val,
                ocr_mode=ocr_mode_val,
                effort=effort_val,
                image_analysis=image_analysis_val,
                url_timeout=url_timeout_val,
                allow_local_source=allow_local_source_val,
                max_inline_bytes=max_inline_bytes_val,
                allow_http_source=allow_http_source_val,
            )

    asyncio.create_task(_bg_run())
    return JSONResponse(content=job_store.build_response(rec).model_dump(by_alias=True), status_code=202)


@_router.get(
    "/parse/jobs/{job_id}",
    response_model=JobAsyncResponse,
    status_code=status.HTTP_200_OK,
    responses={**_ERR_404},
    tags=["Jobs"],
)
async def get_job(
    job_id: str = Path(description="Job ID (job_...)"),
    job_store: JobStore = Depends(_get_job_store),
) -> JobAsyncResponse:
    """Retrieve a job's current status and results."""
    return job_store.build_response(job_store.get(job_id))


@_router.get(
    "/parse/jobs",
    response_model=JobListResponse,
    status_code=status.HTTP_200_OK,
    tags=["Jobs"],
)
async def list_jobs(
    status_filter: str | None = Query(default=None, alias="status", description="Filter by status (comma-separated)"),
    limit: int = Query(default=20, ge=1, le=100, description="Items per page"),
    after: str | None = Query(default=None, description="Pagination cursor (last_id from previous page)"),
    order: Literal["asc", "desc"] = Query(default="desc", description="Sort by created_at"),
    created_after: str | None = Query(default=None, description="ISO-8601 lower bound for created_at"),
    job_store: JobStore = Depends(_get_job_store),
) -> JobListResponse:
    """List jobs for the current tenant."""
    return job_store.list_jobs(
        status_filter=status_filter,
        limit=limit,
        after=after,
        order=order,
        created_after=created_after,
    )


@_router.delete(
    "/parse/jobs/{job_id}",
    response_model=JobCancelResponse,
    status_code=status.HTTP_200_OK,
    responses={**_ERR_404, **_ERR_409},
    tags=["Jobs"],
)
async def cancel_job(
    job_id: str = Path(description="Job ID"),
    job_store: JobStore = Depends(_get_job_store),
) -> JobCancelResponse:
    """Cancel a queued or running job."""
    rec = job_store.cancel(job_id)
    return JobCancelResponse(job_id=rec.id, status="canceled", canceled_at=JobStore._now())


# ── Usage ─────────────────────────────────────────────────────────────


@_router.get(
    "/usage",
    response_model=UsageResponse,
    status_code=status.HTTP_200_OK,
    tags=["Usage"],
)
async def get_usage(
    request: Request,
    job_store: JobStore = Depends(_get_job_store),
    access_level: AccessLevel = Depends(_resolve_access_level),
) -> UsageResponse:
    """Query current usage and limits."""
    return job_store.usage(access_level)


# ═══════════════════════════════════════════════════════════════════════
#  APP FACTORY
# ═══════════════════════════════════════════════════════════════════════


def _build_v1_router() -> APIRouter:
    """Build and return the /v1 router."""
    return _router


_OCR_MODES = ("auto", "txt", "ocr")


def _normalize_server_tiers(tier: Tier | list[Tier] | tuple[Tier, ...] | None) -> list[Tier]:
    """规范化 API server 启动 tier 列表，保留用户顺序并拒绝旧 standard/pro tier。"""
    raw_tiers: list[Tier]
    if tier is None:
        raw_tiers = list(_API_SERVER_TIERS)
    elif isinstance(tier, str):
        raw_tiers = [validate_tier(tier)]
    else:
        raw_tiers = [validate_tier(item) for item in list(tier)] or list(_API_SERVER_TIERS)

    tiers: list[Tier] = []
    for item in raw_tiers:
        if item not in tiers:
            tiers.append(item)
    return tiers


def _runtime_options_for_server_tiers(tiers: list[Tier]) -> dict[Tier, ParserRuntimeOptions]:
    """为每个启动 tier 生成独立 runtime，API server 只允许 tier 决定 backend/effort。"""
    return {tier: runtime_options_for_tier(tier) for tier in tiers}


def _model_ids_and_tiers_for_server_tier(tier: Tier) -> tuple[list[str], list[dict[str, Any]]]:
    if tier == "flash":
        return ["MinerU-Flash"], [
            {
                "id": "flash",
                "description": "Fast local text extraction.",
                "current_model": "flash",
            },
        ]
    if tier == "medium":
        return ["Hybrid-Medium", "MinerU-HTML"], [
            {
                "id": "medium",
                "description": "Hybrid medium parsing with local lightweight models.",
                "current_model": "hybrid-medium",
            },
        ]
    if tier == "extra_high":
        return ["MinerU2.5-Pro-2604-1.2B", "MinerU-HTML"], [
            {
                "id": "extra_high",
                "description": "Hybrid maximum-accuracy parsing.",
                "current_model": "MinerU2.5-Pro-2604-1.2B",
            },
        ]

    return ["MinerU2.5-Pro-2604-1.2B", "MinerU-HTML"], [
        {
            "id": "high",
            "description": "Hybrid high-accuracy parsing.",
            "current_model": "MinerU2.5-Pro-2604-1.2B",
        },
    ]


def _model_ids_and_tiers_for_server_tiers(tiers: list[Tier]) -> tuple[list[str], list[dict[str, Any]]]:
    """聚合多个启动 tier 的模型和 tier metadata，并对重复 model id 保序去重。"""
    model_ids: list[str] = []
    tier_infos: list[dict[str, Any]] = []
    for tier in tiers:
        tier_model_ids, single_tier_infos = _model_ids_and_tiers_for_server_tier(tier)
        for model_id in tier_model_ids:
            if model_id not in model_ids:
                model_ids.append(model_id)
        tier_infos.extend(single_tier_infos)
    return model_ids, tier_infos


def _preflight_tier_dependencies(tier: Tier) -> None:
    try:
        ensure_tier_runtime_dependencies(tier)
    except TierDependencyError as exc:
        raise ParseServerStartupError(str(exc)) from exc


def _dependency_tier_for_runtime(runtime: ParserRuntimeOptions) -> Tier:
    """根据实际 runtime 判断依赖预检 tier。"""
    return runtime.tier


def _preflight_runtime_dependencies(runtime_options: dict[Tier, ParserRuntimeOptions]) -> None:
    """对多 tier server 的 runtime 依赖做去重预检，避免重复导入检查。"""
    checked_tiers: set[Tier] = set()
    for runtime in runtime_options.values():
        dependency_tier = _dependency_tier_for_runtime(runtime)
        if dependency_tier in checked_tiers:
            continue
        checked_tiers.add(dependency_tier)
        _preflight_tier_dependencies(dependency_tier)


def create_app(
    *,
    upload_dir: str = "",
    tier: Tier | list[Tier] | tuple[Tier, ...] | None = None,
    concurrency: int = 1,
    url_timeout: int = 60,
    allow_local_source: bool = False,
    max_inline_bytes: int = _MAX_INLINE_BYTES_DEFAULT,
    allow_http_source: bool = False,
    api_key: str | None = None,
    language: str = "ch",
    ocr_mode: str = "auto",
    image_analysis: bool = True,
) -> FastAPI:
    """Create a FastAPI application implementing the MinerU v1 REST API.

    Parameters
    ----------
    upload_dir:
        Directory for uploaded files and parse artifacts.
    tier:
        Server parsing tier. ``"flash"`` selects flash parsing; ``"medium"``,
        ``"high"`` and ``"extra_high"`` map to same-name Hybrid efforts.
    concurrency:
        Maximum concurrent parse jobs (default 1).
    url_timeout:
        Timeout in seconds for downloading url sources (default 60).
    allow_local_source:
        Whether ``local`` sources may read paths visible to the server process.
    max_inline_bytes:
        Maximum decoded bytes accepted for ``inline`` sources.
    allow_http_source:
        Whether ``url`` sources may use plain HTTP. HTTPS is always allowed.
    api_key:
        Optional API key.  When set, clients must pass ``Authorization: Bearer <key>``
        to access list endpoints and advanced output formats.
    language:
        Hybrid medium OCR language hint; accepted by other efforts for compatibility.
    ocr_mode:
        PDF OCR/text extraction mode for Hybrid backends.
    image_analysis:
        Whether image analysis is enabled for Hybrid backends.
    """
    upload_dir = upload_dir or ""
    tier_input = None if tier in ((), []) else tier
    server_tiers = _normalize_server_tiers(tier_input)
    tier_runtime_options = _runtime_options_for_server_tiers(server_tiers)
    server_tiers = list(tier_runtime_options)
    default_tier = _DEFAULT_API_SERVER_TIER if _DEFAULT_API_SERVER_TIER in tier_runtime_options else server_tiers[0]
    default_runtime = tier_runtime_options[default_tier]
    tier = default_tier
    backend = default_runtime.backend
    effort = default_runtime.effort
    _preflight_runtime_dependencies(tier_runtime_options)
    _api_key: str | None = api_key or None
    _upload_dir = pathlib.Path(upload_dir) if upload_dir else pathlib.Path(tempfile.mkdtemp(prefix="mineru_"))
    _upload_dir.mkdir(parents=True, exist_ok=True)
    if max_inline_bytes < 0:
        raise ValueError("max_inline_bytes must be non-negative")
    language = validate_public_ocr_lang(language)

    _model_ids, _tiers = _model_ids_and_tiers_for_server_tiers(server_tiers)

    @asynccontextmanager
    async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.upload_dir = _upload_dir
        application.state.tier = tier
        application.state.default_tier = default_tier
        application.state.backend = backend
        application.state.tier_runtime_options = tier_runtime_options
        application.state.model_ids = _model_ids
        application.state.tiers = _tiers
        application.state.concurrency = concurrency
        application.state.url_timeout = url_timeout
        application.state.allow_local_source = allow_local_source
        application.state.max_inline_bytes = max_inline_bytes
        application.state.allow_http_source = allow_http_source
        application.state.api_key = _api_key
        application.state.language = language
        application.state.ocr_mode = ocr_mode
        application.state.effort = effort
        application.state.image_analysis = image_analysis
        yield
        if not upload_dir and _upload_dir.exists():
            shutil.rmtree(_upload_dir, ignore_errors=True)

    enable_docs = _env_flag("MINERU_API_ENABLE_FASTAPI_DOCS", default=True)

    application = FastAPI(
        title="MinerU API",
        version="1.0.0",
        openapi_url="/openapi.json" if enable_docs else None,
        docs_url="/docs" if enable_docs else None,
        redoc_url="/redoc" if enable_docs else None,
        lifespan=_lifespan,
    )
    application.state.upload_dir = _upload_dir
    application.state.tier = tier
    application.state.default_tier = default_tier
    application.state.backend = backend
    application.state.tier_runtime_options = tier_runtime_options
    application.state.model_ids = _model_ids
    application.state.tiers = _tiers
    application.state.concurrency = concurrency
    application.state.url_timeout = url_timeout
    application.state.allow_local_source = allow_local_source
    application.state.max_inline_bytes = max_inline_bytes
    application.state.allow_http_source = allow_http_source
    application.state.api_key = _api_key
    application.state.language = language
    application.state.ocr_mode = ocr_mode
    application.state.effort = effort
    application.state.image_analysis = image_analysis
    FileStore(_upload_dir).install(application.state)
    JobStore(concurrency=concurrency).install(application.state)
    application.add_middleware(GZipMiddleware, minimum_size=1000)

    @application.exception_handler(ApiServerError)
    async def _api_server_error_handler(request: Request, exc: ApiServerError) -> JSONResponse:
        return _error_response(exc.status_code, exc.error)

    @application.exception_handler(RequestValidationError)
    async def _request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            400,
            ErrorDetail(
                type="invalid_request_error",
                code="invalid_request",
                message=_validation_error_message(exc),
                param=_validation_error_param(exc),
            ),
        )

    @application.exception_handler(HTTPException)
    async def _http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
        return _error_response(exc.status_code, _http_exception_error(exc))

    @application.exception_handler(Exception)
    async def _unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("Unhandled parse API error", exc_info=(type(exc), exc, exc.__traceback__))
        return _error_response(
            500,
            ErrorDetail(
                type="api_error",
                code="internal_error",
                message="Internal server error",
            ),
        )

    _PUBLIC_PATHS = frozenset({"/v1/health", "/v1/models", "/v1/tiers"})
    _PUBLIC_PREFIXES = ("/openapi", "/docs", "/redoc")

    async def _auth_middleware(request: Request, call_next: Callable[[Request], Any]) -> Any:
        path = request.url.path
        if path in _PUBLIC_PATHS or any(path.startswith(p) for p in _PUBLIC_PREFIXES) or path.startswith("/v1/models/"):
            return await call_next(request)
        api_key = (request.app.state.api_key or "").strip()
        if not api_key:
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        if not auth.startswith("Bearer ") or auth[7:] != api_key:
            return _error_response(
                401,
                ErrorDetail(
                    type="authentication_error",
                    code="invalid_api_key",
                    message="Invalid or missing API key",
                ),
            )
        return await call_next(request)

    application.middleware("http")(_auth_middleware)
    application.include_router(_build_v1_router())
    return application


# ── CLI ──────────────────────────────────────────────────────────────


@click.command()
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8000, type=int, help="Server port")
@click.option(
    "--upload-dir",
    default="",
    help="Upload directory (default: auto-created temp dir)",
    type=click.Path(file_okay=False, writable=True, resolve_path=True),
)
@click.option(
    "--tier",
    multiple=True,
    type=click.Choice(list(_API_SERVER_TIERS)),
    help=(
        "Server parsing tier; repeat to expose multiple tiers. "
        "Defaults to flash, medium, high, and extra_high; requests without tier default to high."
    ),
)
@click.option(
    "--concurrency",
    default=1,
    type=int,
    help="Maximum concurrent parse jobs (default: 1)",
)
@click.option(
    "--url-timeout",
    default=60,
    type=int,
    help="Timeout in seconds for url source downloads (default: 60)",
)
@click.option(
    "--allow-local-source",
    is_flag=True,
    help="Allow local sources to read any path visible to the server process.",
)
@click.option(
    "--max-inline-bytes",
    default=_MAX_INLINE_BYTES_DEFAULT,
    type=int,
    help=f"Maximum decoded bytes for inline sources (default: {_MAX_INLINE_BYTES_DEFAULT})",
)
@click.option(
    "--allow-http-source",
    is_flag=True,
    help="Allow url sources to use plain HTTP. HTTPS is always allowed.",
)
@click.option(
    "--language",
    default="ch",
    type=str,
    metavar="[" + "|".join(_API_SERVER_LANGUAGES) + "]",
    help="Hybrid medium OCR language hint; accepted by other efforts for compatibility.",
)
@click.option(
    "--ocr-mode",
    default="auto",
    type=click.Choice(_OCR_MODES),
    help="PDF OCR/text extraction mode. Applies to hybrid-* backends.",
)
@click.option("--disable-image-analysis", is_flag=True, help="Disable image analysis for Hybrid backends.")
@click.option(
    "--api-key",
    default=None,
    type=str,
    help="Optional API key. When set, clients must pass Authorization: Bearer <key> to access protected endpoints.",
)
def main(
    host: str,
    port: int,
    upload_dir: str,
    tier: tuple[Tier, ...],
    concurrency: int,
    url_timeout: int,
    allow_local_source: bool,
    max_inline_bytes: int,
    allow_http_source: bool,
    language: str,
    ocr_mode: str,
    disable_image_analysis: bool,
    api_key: str | None,
) -> None:
    """Start the MinerU v1 REST API server."""
    try:
        application = create_app(
            upload_dir=upload_dir,
            tier=tier or None,
            concurrency=concurrency,
            url_timeout=url_timeout,
            allow_local_source=allow_local_source,
            max_inline_bytes=max_inline_bytes,
            allow_http_source=allow_http_source,
            api_key=api_key,
            language=language,
            ocr_mode=ocr_mode,
            image_analysis=not disable_image_analysis,
        )
    except ParseServerStartupError as exc:
        raise click.ClickException(str(exc)) from None
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    config = uvicorn.Config(
        application,
        host=host,
        port=port,
    )
    server = uvicorn.Server(config)
    _install_managed_parse_server_stdin_watcher(server)
    server.run()


if __name__ == "__main__":
    main()


__all__ = ["create_app", "main"]
