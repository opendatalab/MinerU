# Copyright (c) Opendatalab. All rights reserved.
"""MinerU V1 解析服务：上传资源、统一后台任务与 HTTP 接口。"""

from __future__ import annotations

import asyncio
import base64
import hashlib
from importlib.resources import files as package_resources
import logging
import math
import os
import pathlib
import re
import secrets
import shutil
import tempfile
import threading
import time
from collections.abc import Collection
from contextlib import asynccontextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from typing import Annotated, Any, AsyncIterator, Awaitable, Callable, Literal, TypedDict
from urllib.parse import urlparse

import click
import httpx
import uvicorn
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Path, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.openapi.docs import get_redoc_html, get_swagger_ui_html, get_swagger_ui_oauth2_redirect_html
from fastapi_offline import FastAPIOffline
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer
from filelock import FileLock, Timeout as FileLockTimeout
from ..utils.retention import resolve_retention_seconds, RETENTION_SCAN_INTERVAL_SECONDS
from pydantic import BaseModel, ConfigDict, Field
from starlette.datastructures import State
from starlette.types import Receive, Scope, Send

from ..config import VlmConfig
from ..config import get_config
from ..errors import MineruError
from ..integrations.docvortex import MinerUMetadata
from ..filetypes import (
    PARSEABLE_EXTENSIONS,
    batch_effective_parse_tier,
    file_type_for_extension,
    is_flash_only_parse_extension,
)
from ..types import DeploymentTier, PageInfo, Tier, select_default_task_tier
from ..utils.async_utils import drain_future, run_sync
from ..model.runtime.ownership import RuntimeOwner, runtime_owner
from ..model.vlm.errors import is_fatal_engine_error
from ..utils.logger import configure_global_log_level
from ..utils.stdio import configure_standard_streams
from ..version import __version__
from . import parse_async
from .page_range import (
    PAGE_RANGE_DESCRIPTION,
    normalize_page_range_input,
)
from .page_range import (
    count_pages_in_range as _count_pages_in_range,
)
from .page_range import (
    format_page_range as _compact_page_numbers,
)
from .process_control import ManagedProcessControlWatcher
from .server_tiers import model_tier_for_server_tiers, parse_server_tier, resolve_server_tiers
from .tier import (
    ParserRuntimeOptions,
    TierDependencyError,
    ensure_tier_runtime_dependencies,
    runtime_options_for_tier,
)
from ..task_types import TASK_OUTPUT_FORMATS, TaskFileState, TaskOutputFormat
from ..tasks.contracts import MAX_TASK_FILES, ErrorDetail, ErrorResponse, FileParseInfo
from ..tasks.errors import ApiServerError, raise_api_error
from ..tasks.execution import TaskExecution, TaskInputMetadata, execution_timestamp, new_execution_id
from ..tasks.http import build_task_router
from ..tasks.service import TaskStore

_MAX_URL_BYTES_DEFAULT = 209715200

logger = logging.getLogger("mineru.parser.api_server")


class ParseServerStartupError(RuntimeError):
    """Raised when the parse server cannot start because of local setup."""


# ── literal type aliases ────────────────────────────────────────────

AccessLevel = Literal["anonymous", "registered"]
"""Account access level determined by API Key."""

FilePurpose = Literal["parse", "parse_output"]
"""文件用途限定为解析输入或产物。"""

# ── helper ───────────────────────────────────────────────────────────


def _env_flag(name: str, default: bool = True) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


# ═══════════════════════════════════════════════════════════════════════
#  MODELS
# ═══════════════════════════════════════════════════════════════════════

_PYDANTIC_CONFIG = ConfigDict(extra="forbid")


# ── Error ────────────────────────────────────────────────────────────


def _error_response(status_code: int, error: ErrorDetail) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content=ErrorResponse(error=error).model_dump(by_alias=True, exclude_none=True),
    )


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


def _validation_error_code(exc: RequestValidationError) -> str:
    for err in exc.errors():
        loc = tuple(err.get("loc", ()))
        if err.get("type") == "union_tag_invalid" and "source" in loc:
            return "unsupported_source"
    return "invalid_request"


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
        code="api_error" if exc.status_code >= 500 else "invalid_request",
        message=message,
    )


# ── Health ───────────────────────────────────────────────────────────


class HealthFeatures(BaseModel):
    model_config = _PYDANTIC_CONFIG
    webhook: bool = False
    task_output_formats: list[TaskOutputFormat] = Field(default_factory=lambda: list(TASK_OUTPUT_FORMATS))
    sources: list[Literal["upload", "url"]] = Field(default_factory=lambda: ["upload", "url"])


class HealthResponse(BaseModel):
    model_config = _PYDANTIC_CONFIG
    status: str = "ok"
    version: str
    features: HealthFeatures = Field(default_factory=HealthFeatures)


# ── Models API ───────────────────────────────────────────────────────

# /v1/models 只广告一个虚拟模型：id 随服务版本走，解析质量由请求 tier 选择，
# 不对应任何可单独调用的引擎。
_VIRTUAL_MODEL_ID = f"mineru-{__version__}"
_VIRTUAL_MODEL_DESCRIPTION = (
    "MinerU virtual model. PDF/image quality is selected per request via tier; native documents always use flash. "
    "this id identifies the server version, not a specific engine."
)


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


class TierInfoData(TypedDict):
    id: Tier
    description: str
    current_model: str


# ── Upload ───────────────────────────────────────────────────────────


# ── File ─────────────────────────────────────────────────────────────


class FileObjectModel(BaseModel):
    """A file resource (source or parse output)."""

    model_config = _PYDANTIC_CONFIG
    id: str
    object: Literal["file"] = "file"
    bytes: int
    created_at: int
    expires_at: int | None = None
    file_name: str
    purpose: FilePurpose
    sha256sum: str | None = None


# ── Job sources ──────────────────────────────────────────────────────


class _StoredSource(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["file_id"] = "file_id"
    file_id: str


class _UrlSource(BaseModel):
    model_config = _PYDANTIC_CONFIG
    type: Literal["url"] = "url"
    url: str


_ExecutionSource = Annotated[
    _StoredSource | _UrlSource,
    Field(discriminator="type"),
]


class _ExecutionRequest(BaseModel):
    """本地单文件执行请求，公开结果身份与内部源文件引用分别保存。"""

    model_config = _PYDANTIC_CONFIG
    source: _ExecutionSource
    file_id: str
    file_name: str
    bytes: int | None = None
    sha256sum: str | None = None
    page_range: str | None = Field(default=None, description=PAGE_RANGE_DESCRIPTION, examples=["1-5,8,r3-r1", "all"])
    tier: Tier | None = None
    ocr_mode: Literal["auto", "txt", "ocr"] = "auto"
    output_formats: list[TaskOutputFormat] = Field(default_factory=lambda: ["default"])


# ── Job responses ────────────────────────────────────────────────────


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
    max_pages_per_file: int | None = None
    max_file_size_bytes: int | None = None
    max_files_per_job: int = MAX_TASK_FILES
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
class _FileRecord:
    id: str
    filename: str
    bytes: int
    created_at: int
    purpose: FilePurpose
    sha256sum: str | None = None
    expires_at: float | None = None
    job_id: str | None = None


class FileStore:
    """Content-addressed blob storage with in-memory metadata."""

    def __init__(self, root: pathlib.Path, retention_seconds: int = 86400) -> None:
        self._lock = threading.RLock()
        self.retention_seconds = retention_seconds
        self._pins: dict[str, int] = {}
        self._job_output_pins: dict[str, set[str]] = {}
        self._owner_lock = FileLock(str(root / ".mineru-service.lock"))
        self._root = root
        self._blobs = root / "blobs"
        self._blobs.mkdir(parents=True, exist_ok=True)
        self._files: dict[str, _FileRecord] = {}
        self._allocated_file_ids: set[str] = set()
        self._reserved_output_ids: set[str] = set()
        self._task_file_ids: set[str] = set()

    def claim_directory(self) -> None:
        """服务启动时独占上传目录，避免多个进程的内存元数据互相误删。"""
        try:
            self._owner_lock.acquire(timeout=0)
        except FileLockTimeout as exc:
            raise RuntimeError("Upload directory is already owned by another API service") from exc

    def release_directory(self) -> None:
        """关闭服务时释放目录占用，元数据仍只在本进程内维护。"""
        self._owner_lock.release()

    def acquire_inputs(self, file_ids: list[str]) -> dict[str, str]:
        """提交时一次性校验输入并取得 blob 租约，文件视图到期不影响排队任务。"""
        with self._lock:
            records = {file_id: self.get_file(file_id) for file_id in file_ids}
            values = {file_id: rec.sha256sum for file_id, rec in records.items() if rec.sha256sum is not None}
            for sha in values.values():
                self._pins[sha] = self._pins.get(sha, 0) + 1
            return values

    def release_inputs(self, values: dict[str, str]) -> None:
        """最后一个排队或运行任务退出后才释放共享输入引用。"""
        with self._lock:
            for sha in values.values():
                self._pins[sha] -= 1
                if not self._pins[sha]:
                    self._pins.pop(sha)
            values.clear()

    def pin_cached_blob(self, sha: str) -> pathlib.Path | None:
        """哈希复用检查和取得传输引用保持原子性，过期视图不能被新请求复用。"""
        with self._lock:
            now = time.time()
            if not any(
                rec.sha256sum == sha and (rec.expires_at is None or rec.expires_at > now) for rec in self._files.values()
            ):
                return None
            path = self._blob_abs(sha)
            if not path.is_file():
                return None
            self._pins[sha] = self._pins.get(sha, 0) + 1
            return path

    def unpin_cached_blob(self, sha: str) -> None:
        """请求提交或校验失败后释放哈希预检的引用。"""
        with self._lock:
            self._pins[sha] -= 1
            if not self._pins[sha]:
                self._pins.pop(sha)

    def finalize_outputs(self, job_id: str, finished: float) -> None:
        """产物从任务清理完成时计时，保留亚秒精度，避免短保留期被向下取整。"""
        with self._lock:
            for rec in self._files.values():
                if rec.job_id == job_id:
                    rec.expires_at = finished + self.retention_seconds if self.retention_seconds else None
            for sha in self._job_output_pins.pop(job_id, set()):
                self._pins[sha] -= 1
                if not self._pins[sha]:
                    self._pins.pop(sha)

    def collect_expired(self) -> None:
        """回收过期视图和无引用受管 blob，启动时也可清理上次遗留的旧文件。"""
        if not self.retention_seconds:
            return
        with self._lock:
            now = time.time()
            for file_id, rec in tuple(self._files.items()):
                if rec.expires_at is not None and rec.expires_at <= now:
                    self._files.pop(file_id)
                    if file_id not in self._task_file_ids:
                        self._allocated_file_ids.discard(file_id)
            referenced = {rec.sha256sum for rec in self._files.values()} | set(self._pins)
            for prefix in self._blobs.iterdir():
                if prefix.is_symlink() or not prefix.is_dir() or not re.fullmatch(r"[0-9a-f]{2}", prefix.name):
                    continue
                for blob in prefix.iterdir():
                    sha = prefix.name + blob.name
                    if (
                        blob.is_symlink()
                        or not blob.is_file()
                        or not re.fullmatch(r"[0-9a-f]{62}(?:\.[0-9a-f]{16})?", blob.name)
                    ):
                        continue
                    if sha in referenced:
                        continue
                    # 刚写入但尚未注册的文件及当前扫描期间的写入受锁和年龄门槛保护。
                    if blob.stat().st_mtime + self.retention_seconds <= now:
                        blob.unlink(missing_ok=True)

    # ── ID generation ────────────────────────────────────────────────

    @staticmethod
    def _new_file_id() -> str:
        """生成不携带存储位置或内容信息的随机文件标识。"""
        return "file-" + secrets.token_hex(12)

    def _allocate_file_id(self) -> str:
        """检查源文件、产物和仍被任务持有的 ID，避免覆盖活跃资源。"""
        with self._lock:
            while True:
                file_id = self._new_file_id()
                if file_id not in self._allocated_file_ids and file_id not in self._files:
                    self._allocated_file_ids.add(file_id)
                    return file_id

    def reserve_file_id(self, *, task_owned: bool = True) -> str:
        """只预留公共产物 ID，完成前不建立可查询的 Files 对象。"""
        with self._lock:
            file_id = self._allocate_file_id()
            self._reserved_output_ids.add(file_id)
            if task_owned:
                self._task_file_ids.add(file_id)
            return file_id

    def forget_file_id(self, file_id: str) -> None:
        """任务记录过期后回收 ID 索引，仍存在的 Files 资源继续阻止占用。"""
        with self._lock:
            self._task_file_ids.discard(file_id)
            self._reserved_output_ids.discard(file_id)
            if file_id not in self._files:
                self._allocated_file_ids.discard(file_id)

    def discard_file_id(self, file_id: str) -> None:
        """取消未发布的产物及其预留，迟到写出不得重新注册该 ID。"""
        with self._lock:
            self._reserved_output_ids.discard(file_id)
            record = self._files.get(file_id)
            if record is not None and record.purpose == "parse_output":
                self._files.pop(file_id)

    def _store_path(self, path: pathlib.Path, sha256sum: str) -> pathlib.Path:
        """在持有存储锁时物化共享字节，源文件和产物复用相同的原子写入。"""
        target = self._blob_abs(sha256sum)
        if not target.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + "." + secrets.token_hex(8))
            try:
                shutil.copyfile(path, temporary)
                os.replace(temporary, target)
            finally:
                temporary.unlink(missing_ok=True)
        return target

    # ── blob helpers ─────────────────────────────────────────────────

    @staticmethod
    def _blob_path(sha256hex: str) -> pathlib.PurePosixPath:
        return pathlib.PurePosixPath(sha256hex[:2]) / sha256hex[2:]

    def _blob_abs(self, sha256hex: str) -> pathlib.Path:
        return self._blobs / self._blob_path(sha256hex)

    def blob_exists(self, sha256hex: str) -> bool:
        with self._lock:
            return self._blob_abs(sha256hex).is_file()

    def store_blob(self, data: bytes, *, sha256hex: str) -> None:
        """原子写入已校验的内容，避免并发读取尚未写完的 blob。"""
        with self._lock:
            p = self._blob_abs(sha256hex)
            if not p.is_file():
                p.parent.mkdir(parents=True, exist_ok=True)
                temporary = p.with_name(p.name + "." + secrets.token_hex(8))
                try:
                    temporary.write_bytes(data)
                    os.replace(temporary, p)
                finally:
                    temporary.unlink(missing_ok=True)

    def register_source_file(
        self, filename: str, path: pathlib.Path, sha256sum: str, file_id: str | None = None, *, keep_alive: bool = False
    ) -> str:
        """从已校验的暂存文件原子注册源字节及独立的文件名视图。"""
        with self._lock:
            file_id = file_id or self._allocate_file_id()
            existing = self._files.get(file_id)
            if (
                file_id in self._task_file_ids
                or file_id in self._reserved_output_ids
                or (
                    existing is not None
                    and (existing.purpose != "parse" or existing.filename != filename or existing.sha256sum != sha256sum)
                )
            ):
                raise ValueError("File ID already belongs to another resource")
            self._allocated_file_ids.add(file_id)
            target = self._store_path(path, sha256sum)
            now = int(time.time())
            self._files[file_id] = _FileRecord(
                id=file_id,
                filename=filename,
                bytes=target.stat().st_size,
                created_at=now,
                purpose="parse",
                sha256sum=sha256sum,
                expires_at=now + self.retention_seconds if self.retention_seconds and not keep_alive else None,
            )
            return file_id

    def finalize_source_views(self, file_ids: list[str]) -> None:
        """准备完成后恢复源文件视图的正常保留期，执行所需字节另由 Job 租约保护。"""
        with self._lock:
            expires_at = int(time.time()) + self.retention_seconds if self.retention_seconds else None
            for file_id in file_ids:
                record = self._files.get(file_id)
                if record is not None:
                    record.expires_at = expires_at

    def create_file_for_output_path(self, filename: str, path: pathlib.Path, *, job_id: str, file_id: str | None = None) -> str:
        """把完整产物原子登记到预分配 ID，已取消或已消费的预留不能再次写入。"""
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                hasher.update(chunk)
        sha = hasher.hexdigest()
        with self._lock:
            file_id = file_id or self.reserve_file_id(task_owned=False)
            if file_id not in self._reserved_output_ids or file_id in self._files:
                raise ValueError("Output file ID is not reserved or has already been published")
            target = self._store_path(path, sha)
            self._files[file_id] = _FileRecord(
                id=file_id,
                filename=filename,
                bytes=target.stat().st_size,
                created_at=int(time.time()),
                purpose="parse_output",
                sha256sum=sha,
                job_id=job_id,
            )
            self._reserved_output_ids.remove(file_id)
            pins = self._job_output_pins.setdefault(job_id, set())
            if sha not in pins:
                pins.add(sha)
                self._pins[sha] = self._pins.get(sha, 0) + 1
            return file_id

    def pin_output_file(self, file_id: str) -> tuple[pathlib.Path, str, str]:
        """检查产物权限并取得下载租约，防止传输途中被保留策略删除。"""
        with self._lock:
            rec = self.get_file(file_id)
            if rec.purpose != "parse_output":
                raise_api_error(
                    403,
                    code="feature_requires_api_key",
                    message="Source files cannot be downloaded",
                )
            if rec.sha256sum is None or not self._blob_abs(rec.sha256sum).is_file():
                raise_api_error(404, code="file_not_found", message="Output expired")
            self._pins[rec.sha256sum] = self._pins.get(rec.sha256sum, 0) + 1
            return self._blob_abs(rec.sha256sum), rec.filename, rec.sha256sum

    def read_blob(self, sha256hex: str) -> bytes:
        with self._lock:
            p = self._blob_abs(sha256hex)
            if not p.is_file():
                raise FileNotFoundError(sha256hex)
            return p.read_bytes()

    def create_file_for_output(
        self, filename: str, data: bytes, *, sha256hex: str | None = None, job_id: str | None = None
    ) -> str:
        """Store parse-output data and return a file_id."""
        with self._lock:
            sha = sha256hex or hashlib.sha256(data).hexdigest()
            self.store_blob(data, sha256hex=sha)
            now = int(time.time())
            file_id = self._allocate_file_id()
            self._files[file_id] = _FileRecord(
                id=file_id,
                filename=filename,
                bytes=len(data),
                created_at=now,
                purpose="parse_output",
                sha256sum=sha,
                job_id=job_id,
                expires_at=(now + self.retention_seconds) if self.retention_seconds and job_id is None else None,
            )
            if job_id is not None:
                pins = self._job_output_pins.setdefault(job_id, set())
                if sha not in pins:
                    pins.add(sha)
                    self._pins[sha] = self._pins.get(sha, 0) + 1
            return file_id

    def get_file(self, file_id: str) -> _FileRecord:
        with self._lock:
            rec = self._files.get(file_id)
            if rec is None or (rec.expires_at is not None and rec.expires_at <= time.time()):
                raise_api_error(
                    404,
                    code="file_not_found",
                    message=f"File {file_id} not found",
                )
            return rec

    @staticmethod
    def _make_file_object(rec: _FileRecord) -> FileObjectModel:
        """公开时间戳保持整秒并向上取整，内部过期判断使用未取整的期限。"""
        return FileObjectModel(
            id=rec.id,
            bytes=rec.bytes,
            created_at=rec.created_at,
            expires_at=math.ceil(rec.expires_at) if rec.expires_at is not None else None,
            file_name=rec.filename,
            purpose=rec.purpose,
            sha256sum=rec.sha256sum,
        )

    def read_file_data(self, file_id: str) -> bytes:
        with self._lock:
            rec = self.get_file(file_id)
            if rec.purpose != "parse_output":
                raise_api_error(
                    403,
                    code="feature_requires_api_key",
                    message="Source files cannot be downloaded",
                )
            if rec.sha256sum is None:
                raise_api_error(
                    500,
                    code="internal_error",
                    message="File has no sha256sum",
                )
            return self.read_blob(rec.sha256sum)

    def delete_file(self, file_id: str) -> None:
        with self._lock:
            if file_id not in self._files:
                raise_api_error(
                    404,
                    code="file_not_found",
                    message=f"File {file_id} not found",
                )
            del self._files[file_id]
            if file_id not in self._task_file_ids:
                self._allocated_file_ids.discard(file_id)

    # ── defaults → app state ─────────────────────────────────────────

    def install(self, app_state: State) -> None:
        app_state.file_store = self


# ═══════════════════════════════════════════════════════════════════════
#  DEPENDENCIES
# ═══════════════════════════════════════════════════════════════════════


def _get_store(request: Request) -> FileStore:
    return request.app.state.file_store


def _get_execution_store(request: Request) -> ExecutionStore:
    """读取当前应用持有的单文件执行存储。"""
    return request.app.state.execution_store


def _resolve_access_level(request: Request) -> AccessLevel:
    """Return ``"registered"`` when the server has an API key configured
    (the middleware already validated it), ``"anonymous"`` otherwise."""
    return "registered" if request.app.state.api_key else "anonymous"


# ═══════════════════════════════════════════════════════════════════════
#  JOB STORE
# ═══════════════════════════════════════════════════════════════════════


def _validate_source_policy(source: _ExecutionSource, *, allow_http_source: bool) -> None:
    """执行层仅接受已校验的源文件引用或遵守服务策略的 URL。"""
    if isinstance(source, _StoredSource):
        return
    scheme = urlparse(source.url).scheme.lower()
    if scheme == "https" or (scheme == "http" and allow_http_source):
        return
    raise ValueError("url source must use https unless --allow-http-source is enabled")


def _validate_execution_source_policy(req: _ExecutionRequest, *, allow_http_source: bool) -> None:
    """校验单个来源并保留 Tasks 对外错误参数的位置约定。"""
    try:
        _validate_source_policy(req.source, allow_http_source=allow_http_source)
    except ValueError as exc:
        message = str(exc)
        code = "unsupported_source" if "source" in message else "invalid_request"
        raise_api_error(400, code=code, message=message, param="files.0.source")


@dataclass
class _ExecutionRecord:
    """保存一次单文件执行的状态和资源租约，批次进度只由父 Task 聚合。"""

    id: str
    request: _ExecutionRequest
    status: TaskFileState = "pending"
    created_at: str = ""
    started_at: str | None = None
    finished_at: str | None = None
    page_range: str = ""
    parse: FileParseInfo | None = None
    error: ErrorDetail | None = None
    input_blobs: dict[str, str] = field(default_factory=dict)
    input_metadata: TaskInputMetadata | None = None
    prepared_input: _PreparedInput | None = None
    file_store: FileStore | None = None
    usage_counted: bool = False
    result_file_id: str | None = None
    retention_held: bool = False
    retention_epoch: float | None = None


def _discard_unfinished_output(rec: _ExecutionRecord) -> None:
    """失败或取消时撤销预留结果，阻止尚在排空的后台线程发布迟到产物。"""
    if rec.file_store is not None and rec.status != "completed":
        rec.file_store.discard_file_id(rec.request.file_id)


class ExecutionStore:
    """管理本地单文件执行、并发名额及结果保留期。"""

    def __init__(self, concurrency: int = 1, retention_seconds: int = 86400) -> None:
        """初始化本应用执行状态、准备限流与解析信号量。"""
        self.retention_seconds = retention_seconds
        self._jobs_created = 0
        self._files_processed = 0
        self._pages_processed = 0
        self._executions: dict[str, _ExecutionRecord] = {}
        self._concurrency = max(1, concurrency)
        self._semaphore = asyncio.Semaphore(self._concurrency)
        self._preparation_semaphore = asyncio.Semaphore(self._concurrency)
        self._started_at = execution_timestamp()
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._completion_events: dict[str, asyncio.Event] = {}
        self._closing = False
        self.runtime_owner = RuntimeOwner()

    def fail_engine(self, message: str) -> None:
        """标记本应用未结束执行的引擎错误，并沿原边界取消和排空任务。"""
        if self.runtime_owner.engine_error is None:
            self.runtime_owner.engine_error = message
        current = asyncio.current_task()
        for execution_id, task in tuple(self._tasks.items()):
            rec = self._executions[execution_id]
            if task.done() or rec.status == "canceled":
                continue
            rec.status = "failed"
            rec.error = ErrorDetail(code="engine_dead", message=message)
            _discard_unfinished_output(rec)
            if task is not current and not task.cancelling():
                task.cancel()

    def require_ready(self) -> None:
        """健康和提交入口共用引擎死亡状态及现有错误封装。"""
        message = self.runtime_owner.engine_error
        if message is not None:
            self.fail_engine(message)
            raise_api_error(503, code="engine_dead", message=message)

    def start_task(
        self,
        rec: _ExecutionRecord,
        operation: Callable[[], Awaitable[None]],
        *,
        prepare: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """保存真实后台任务及所属应用租约，排队取消也不会开始解析。"""
        self.require_ready()
        if self._closing:
            raise_api_error(503, code="server_shutting_down", message="Server is shutting down")
        self._completion_events[rec.id] = asyncio.Event()

        async def run_owned() -> None:
            """准备输入不占解析名额，等待解析时仍限制准备任务数以控制内存。"""
            token = runtime_owner.set(self.runtime_owner)
            acquired = False
            try:
                if prepare is not None:
                    async with self._preparation_semaphore:
                        if rec.status not in {"canceled", "failed"}:
                            await prepare()
                        if rec.status != "pending":
                            return
                        # 取得解析名额后才允许下一份输入准备，避免待解析字节无限堆积。
                        await self._semaphore.acquire()
                        acquired = True
                else:
                    await self._semaphore.acquire()
                    acquired = True
                if rec.status == "pending":
                    await operation()
            except asyncio.CancelledError:
                if rec.status != "failed":
                    rec.status = "canceled"
                raise
            finally:
                rec.prepared_input = None
                if acquired:
                    self._semaphore.release()
                runtime_owner.reset(token)

        def finished(task: asyncio.Task[None]) -> None:
            """回收句柄并读取异常，覆盖任务首次执行前就被取消的路径。"""
            self._tasks.pop(rec.id, None)
            rec.finished_at = execution_timestamp()
            if not rec.retention_held:
                if rec.retention_epoch is None:
                    rec.retention_epoch = time.time()
                if rec.file_store is not None:
                    rec.file_store.finalize_outputs(rec.id, rec.retention_epoch)
            if rec.file_store is not None:
                rec.file_store.release_inputs(rec.input_blobs)
            if not task.cancelled():
                error = task.exception()
                if error is not None:
                    logger.error("Parse job task failed: %s", rec.id, exc_info=(type(error), error, error.__traceback__))
                    if rec.status != "canceled":
                        rec.status = "failed"
            _discard_unfinished_output(rec)
            self._record_usage(rec)
            self._completion_events[rec.id].set()

        task = asyncio.create_task(run_owned(), name=f"mineru-job-{rec.id}")
        self._tasks[rec.id] = task
        task.add_done_callback(finished)

    async def wait_for_terminal(self, job_id: str, timeout: float) -> _ExecutionRecord:
        """只等待完成通知，超时或等待方取消不会传播到后台解析任务。"""
        rec = self.get(job_id)
        event = self._completion_events.get(job_id)
        if event is not None:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
        return rec

    async def shutdown(self) -> None:
        """停止接单并等待所有任务清理，再释放应用持有的推理运行时。"""
        self._closing = True
        tasks = list(self._tasks.items())
        for job_id, task in tasks:
            if not task.done():
                rec = self._executions[job_id]
                if rec.status not in {"canceled", "failed"}:
                    rec.status = "canceled"
                if not task.cancelling():
                    task.cancel()
        try:
            await drain_future(asyncio.gather(*(task for _, task in tasks), return_exceptions=True))
        finally:
            from ..model.runtime.ownership import release_runtime_owner

            await run_sync(release_runtime_owner, self.runtime_owner)

    def create(self, req: _ExecutionRequest, file_store: FileStore, *, execution_id: str | None = None) -> _ExecutionRecord:
        """登记一个已确定档位的文件执行，持有源字节引用而不创建内部批次。"""
        if req.tier is None:
            raise ValueError("_ExecutionRequest.tier must be resolved before execution creation")
        execution_id = execution_id or new_execution_id()
        if execution_id in self._executions:
            raise_api_error(409, code="job_already_exists", message="Job already exists")
        rec = _ExecutionRecord(
            id=execution_id,
            request=req,
            status="waiting-file" if isinstance(req.source, _UrlSource) else "pending",
            created_at=execution_timestamp(),
            page_range=req.page_range or "",
            file_store=file_store,
            input_blobs=file_store.acquire_inputs([req.source.file_id] if isinstance(req.source, _StoredSource) else []),
        )
        self._executions[execution_id] = rec
        self._jobs_created += 1
        return rec

    def collect_expired(self) -> None:
        """按清理完成时间删除终态任务、事件及其文件视图，运行任务不受 TTL 影响。"""
        if not self.retention_seconds:
            return
        now = time.time()
        for job_id, rec in tuple(self._executions.items()):
            if job_id in self._tasks or rec.finished_at is None or rec.retention_held:
                continue
            finished = rec.retention_epoch or datetime.fromisoformat(rec.finished_at.replace("Z", "+00:00")).timestamp()
            if finished + self.retention_seconds <= now:
                self._record_usage(rec)
                self._executions.pop(job_id)
                self._completion_events.pop(job_id, None)

    def release_task_result(self, job_id: str, finished: float) -> None:
        """父 Task 收尾后才开始子作业与产物的保留期，不更改真实完成时间。"""
        rec = self._executions.get(job_id)
        if rec is None:
            return
        rec.retention_held = False
        rec.retention_epoch = finished
        if rec.file_store is not None:
            rec.file_store.finalize_outputs(job_id, finished)

    def get(self, job_id: str) -> _ExecutionRecord:
        """读取仍在保留期内的执行，未知标识沿用已有错误合同。"""
        self.collect_expired()
        rec = self._executions.get(job_id)
        if rec is None:
            raise_api_error(
                404,
                code="job_not_found",
                message=f"Job {job_id} not found",
            )
        return rec

    def cancel(self, job_id: str) -> _ExecutionRecord:
        """取消尚未完成的执行并撤销结果预留，不强制中断同步工作。"""
        rec = self.get(job_id)
        if rec.status in ("completed", "failed", "canceled"):
            raise_api_error(
                409,
                code="job_already_terminal",
                message=f"Job is {rec.status}",
            )
        rec.status = "canceled"
        _discard_unfinished_output(rec)
        task = self._tasks.get(job_id)
        if task is not None and not task.done():
            task.cancel()
        return rec

    def snapshot(self, rec: _ExecutionRecord) -> TaskExecution:
        """返回独立的有类型快照，公开文件身份始终取预留结果 ID。"""
        return TaskExecution(
            execution_id=rec.id,
            file_id=rec.request.file_id,
            file_name=rec.request.file_name,
            status=rec.status,
            created_at=rec.created_at,
            started_at=rec.started_at,
            finished_at=rec.finished_at,
            page_range=rec.page_range,
            parse=rec.parse.model_copy(deep=True) if rec.parse is not None else None,
            result_file_id=rec.result_file_id,
            error=rec.error.model_copy(deep=True) if rec.error is not None else None,
            input_metadata=rec.input_metadata,
        )

    def _record_usage(self, rec: _ExecutionRecord) -> None:
        """每个执行只计入一次成功文件和实际页面，用量不随记录到期倒退。"""
        if not rec.usage_counted:
            if rec.status == "completed":
                self._files_processed += 1
                self._pages_processed += _count_pages_in_range(rec.page_range)
            rec.usage_counted = True

    def usage(self, access_level: AccessLevel) -> UsageResponse:
        """保留公开 jobs_created 与并发上限字段，按真实单文件执行累计用量。"""
        pending = [rec for rec in self._executions.values() if not rec.usage_counted and rec.status == "completed"]
        return UsageResponse(
            access_level=access_level,
            billing_period=UsageBillingPeriod(start=self._started_at),
            current=UsageCurrent(
                pages_processed=self._pages_processed + sum(_count_pages_in_range(rec.page_range) for rec in pending),
                files_processed=self._files_processed + len(pending),
                jobs_created=self._jobs_created,
            ),
            limits=UsageLimits(
                max_concurrent_jobs=self._concurrency,
                max_file_retention_days=(self.retention_seconds + 86399) // 86400 if self.retention_seconds else None,
            ),
        )

    def install(self, app_state: State) -> None:
        """把执行存储显式安装到当前应用实例。"""
        app_state.execution_store = self


# ── source helpers ───────────────────────────────────────────────────


def _execution_can_default_to_flash(req: _ExecutionRequest) -> bool:
    """原生格式或等待下载后识别类型的无扩展名 URL 可使用 Flash 回退。"""
    return is_flash_only_parse_extension(req.file_name) or (
        isinstance(req.source, _UrlSource) and not pathlib.Path(req.file_name).suffix
    )


def _execution_requires_flash(req: _ExecutionRequest) -> bool:
    """按清单声明的逻辑文件名判断单文件是否必须使用原生解析。"""
    return is_flash_only_parse_extension(req.file_name)


def _page_range_from_result_pages(pages: list[PageInfo]) -> str:
    page_numbers: list[int] = []
    for i, page in enumerate(pages):
        page_idx = page.page_idx if page.page_idx is not None else i
        if isinstance(page_idx, int):
            page_numbers.append(page_idx + 1)
    return _compact_page_numbers(page_numbers)


@dataclass(frozen=True, slots=True)
class _ExtractedSource:
    """保存 API 输入字节、响应类型及 HTML 来源上下文。"""

    data: bytes
    source_uri: str | None = None
    local_resource_root: pathlib.Path | None = None
    transport_encoding: str | None = None
    media_type: str | None = None


@dataclass(frozen=True, slots=True)
class _PreparedInput:
    """暂存已校验的 URL 输入及准备耗时，等待解析队列不计入文件执行耗时。"""

    source: _ExtractedSource
    duration_seconds: float


def _validate_url_input(rec: _ExecutionRecord, extracted: _ExtractedSource) -> None:
    """下载完成后核对声明大小与哈希，实际输入元数据直接归属于本次执行。"""
    data = extracted.data
    if not data:
        raise MineruError("invalid_request", "Empty files are not supported")
    if rec.request.bytes is not None and len(data) != rec.request.bytes:
        raise MineruError("bytes_mismatch", "Declared size does not match downloaded bytes")
    if rec.request.sha256sum is not None and hashlib.sha256(data).hexdigest() != rec.request.sha256sum:
        raise MineruError("file_hash_mismatch", "SHA-256 mismatch")
    rec.input_metadata = TaskInputMetadata(bytes=len(data))


async def _prepare_execution_input(
    rec: _ExecutionRecord,
    file_store: FileStore,
    *,
    url_timeout: int,
    allow_http_source: bool,
    max_url_bytes: int,
) -> None:
    """在准备限流内下载一个 URL，保留来源及耗时，完成后再等待解析名额。"""
    if rec.status in {"canceled", "failed"} or not isinstance(rec.request.source, _UrlSource):
        return
    rec.status = "waiting-file"
    rec.started_at = rec.started_at or execution_timestamp()
    started = time.monotonic()
    try:
        extracted = await _extract_bytes(
            rec.request.source,
            file_store,
            url_timeout=url_timeout,
            allow_http_source=allow_http_source,
            max_url_bytes=max_url_bytes,
        )
        _validate_url_input(rec, extracted)
        rec.prepared_input = _PreparedInput(extracted, time.monotonic() - started)
        if rec.status not in {"canceled", "failed"}:
            rec.status = "pending"
    except Exception as exc:
        logger.exception("Parse-server input preparation failed: job_id=%s file=%r", rec.id, rec.request.file_name)
        if rec.status in {"canceled", "failed"}:
            return
        rec.status = "failed"
        rec.error = (
            ErrorDetail(code=exc.code, message=str(exc), param=exc.param)
            if isinstance(exc, MineruError)
            else ErrorDetail(code="parse_failed", message=str(exc))
        )


async def _extract_bytes(
    source: _ExecutionSource,
    file_store: FileStore,
    *,
    url_timeout: int = 60,
    allow_http_source: bool = False,
    max_url_bytes: int = _MAX_URL_BYTES_DEFAULT,
    input_blobs: dict[str, str] | None = None,
) -> _ExtractedSource:
    """按既有来源策略读取字节，并保留不会由临时文件替代的来源上下文。"""
    _validate_source_policy(
        source,
        allow_http_source=allow_http_source,
    )
    if isinstance(source, _StoredSource):
        sha = (input_blobs or {}).get(source.file_id)
        if sha is None:
            sha = file_store.get_file(source.file_id).sha256sum
        if sha is None:
            raise ValueError("File has no content")
        return _ExtractedSource(file_store.read_blob(sha))
    if isinstance(source, _UrlSource):
        async with httpx.AsyncClient(timeout=url_timeout) as cli:
            # 流式读取并在超过 max_url_bytes 时中止，避免无上限响应被整包缓冲进内存。
            async with cli.stream("GET", source.url) as r:
                r.raise_for_status()
                chunks: list[bytes] = []
                received = 0
                async for chunk in r.aiter_bytes():
                    received += len(chunk)
                    if received > max_url_bytes:
                        raise ValueError(f"url source exceeds max_url_bytes ({max_url_bytes})")
                    chunks.append(chunk)
                return _ExtractedSource(
                    b"".join(chunks),
                    source_uri=str(r.url),
                    transport_encoding=r.charset_encoding,
                    media_type=(r.headers.get("content-type") or "").partition(";")[0].strip().casefold() or None,
                )
    raise ValueError(f"Unknown source type: {type(source)}")


def _suffix_type(filename: str) -> str:
    ext = pathlib.Path(filename).suffix.lower().lstrip(".")
    if ext not in PARSEABLE_EXTENSIONS:
        return ""
    file_type = file_type_for_extension(ext)
    return "" if file_type == "unknown" else file_type


async def _run_execution(
    rec: _ExecutionRecord,
    file_store: FileStore,
    *,
    image_analysis: bool,
    url_timeout: int = 60,
    allow_http_source: bool = False,
    max_url_bytes: int = _MAX_URL_BYTES_DEFAULT,
    vlm_config: VlmConfig | None = None,
    execution_store: ExecutionStore | None = None,
) -> None:
    """执行唯一输入并发布同一公开文件身份的产物，取消时排空已有同步阶段。"""
    if rec.status in {"canceled", "failed"}:
        return
    rec.status = "running"
    rec.started_at = rec.started_at or execution_timestamp()
    req = rec.request
    prepared = rec.prepared_input
    rec.prepared_input = None

    with tempfile.TemporaryDirectory(prefix="mineru_job_") as tmpdir:
        try:
            file_started = time.monotonic() - (prepared.duration_seconds if prepared is not None else 0)
            extracted = (
                prepared.source
                if prepared is not None
                else await _extract_bytes(
                    req.source,
                    file_store,
                    url_timeout=url_timeout,
                    allow_http_source=allow_http_source,
                    max_url_bytes=max_url_bytes,
                    input_blobs=rec.input_blobs,
                )
            )
            data = extracted.data
            if prepared is None and isinstance(req.source, _UrlSource):
                _validate_url_input(rec, extracted)
            suffix = pathlib.Path(req.file_name).suffix
            stype = _suffix_type(req.file_name)
            if not stype and not suffix and isinstance(req.source, _UrlSource) and extracted.media_type == "text/html":
                stype = "html"
                suffix = ".html"
            if not stype:
                raise ValueError(f"Unsupported file type: {req.file_name}")

            suffix = suffix or ".pdf"
            tmp_path = pathlib.Path(tmpdir) / f"input_0{suffix}"
            tmp_path.write_bytes(data)
            effective_tier = batch_effective_parse_tier(req.tier, suffix)
            source_context = None
            if stype == "html":
                from docvortex.document.contracts import HtmlSourceContext

                source_context = HtmlSourceContext(
                    source_uri=extracted.source_uri,
                    local_resource_root=extracted.local_resource_root,
                    transport_encoding=extracted.transport_encoding,
                )
            result = await parse_async(
                str(tmp_path),
                tier=effective_tier,
                ocr_mode=req.ocr_mode,
                image_analysis=image_analysis,
                page_range=req.page_range or "",
                source_context=source_context,
                vlm_config=vlm_config,
            )
            if rec.status in {"canceled", "failed"}:
                return
            actual_metadata = MinerUMetadata.model_validate(result.middle_json.extensions.get("mineru"))
            from .task_outputs import build_task_output

            try:
                artifact = await run_sync(build_task_output, result, req.output_formats, pathlib.Path(tmpdir) / "outputs_0")
                if rec.status in {"canceled", "failed"}:
                    return
                rec.result_file_id = await run_sync(
                    file_store.create_file_for_output_path,
                    pathlib.Path(req.file_name).stem + artifact.suffix,
                    artifact,
                    job_id=rec.id,
                    file_id=req.file_id,
                )
                if rec.status in {"canceled", "failed"}:
                    file_store.discard_file_id(rec.result_file_id)
                    return
            except Exception as exc:
                raise MineruError("result_packaging_failed", str(exc)) from exc

            rec.page_range = _page_range_from_result_pages(result.pages)
            rec.parse = FileParseInfo(
                tier=actual_metadata.tier,
                duration_ms=int((time.monotonic() - file_started) * 1000),
                parser_version=__version__,
            )
            rec.status = "completed"
        except Exception as exc:
            if execution_store is not None and is_fatal_engine_error(exc):
                execution_store.fail_engine(str(exc))
                raise
            if rec.status in {"canceled", "failed"}:
                return
            logger.exception(
                "Parse-server job file failed: job_id=%s file=%r tier=%s page_range=%r",
                rec.id,
                req.file_name,
                req.tier,
                rec.page_range,
            )
            rec.status = "failed"
            rec.error = (
                ErrorDetail(code=exc.code, message=str(exc), param=exc.param)
                if isinstance(exc, MineruError)
                else ErrorDetail(code="parse_failed", message=str(exc))
            )
        finally:
            rec.finished_at = execution_timestamp()


# ═══════════════════════════════════════════════════════════════════════
#  ROUTER
# ═══════════════════════════════════════════════════════════════════════

_router = APIRouter(prefix="/v1")

# OpenAPI tag metadata: drives the group order and descriptions on /docs.
_OPENAPI_TAGS: list[dict[str, str]] = [
    {"name": "Tasks", "description": "JSON task creation, per-file uploads, details and cancellation receipts."},
    {"name": "Health", "description": "Liveness and server readiness."},
    {"name": "Tiers", "description": "Parsing tiers and runtime capability advertised by this server."},
    {"name": "Models", "description": "VLM models available for parsing."},
    {"name": "Files", "description": "Inspect saved files and download completed parsing artifacts."},
    {"name": "Usage", "description": "Remote service quota and usage reporting."},
]

_FASTAPI_DESCRIPTION = """\
Parse documents into structured Middle JSON via self-hosted or remote capacity.

Submit a JSON files array to `/v1/tasks`, upload missing files to their one-hour URLs,
then poll the task and download each completed file's `result_url`. Each file runs
as soon as it is ready; other files may still be waiting for upload. File
entries can instead provide a source `url`, with optional `bytes`; workers download
these inputs in the background. Choose one or more of default/docx/html/latex in
`output_formats`. Task status contains metadata only, never inline result bytes.
Health and usage
endpoints report server readiness and quota state. Tiers (`flash`, `basic`,
`standard`, `advanced`) select the quality/speed trade-off; the set advertised
by `/v1/tiers` is what this server can actually serve.
"""

# Reusable error response maps
_ErrorResponseMap = dict[int | str, dict[str, type[BaseModel]]]
_ERR_400: _ErrorResponseMap = {400: {"model": ErrorResponse}}
_ERR_401: _ErrorResponseMap = {401: {"model": ErrorResponse}}
_ERR_403: _ErrorResponseMap = {403: {"model": ErrorResponse}}
_ERR_404: _ErrorResponseMap = {404: {"model": ErrorResponse}}
_ERR_409: _ErrorResponseMap = {409: {"model": ErrorResponse}}
_ERR_413: _ErrorResponseMap = {413: {"model": ErrorResponse}}
_ERR_429: _ErrorResponseMap = {429: {"model": ErrorResponse}}
_ERR_503: _ErrorResponseMap = {503: {"model": ErrorResponse}}


# ── Health ───────────────────────────────────────────────────────────


def _require_model_preload_ready(request: Request) -> None:
    request.app.state.execution_store.require_ready()
    preload_error: _ModelPreloadError | None = request.app.state.model_preload_error
    if preload_error is None:
        return
    raise_api_error(
        503,
        code=preload_error.code,
        message=preload_error.message,
    )


@_router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    responses={503: {"model": ErrorResponse}},
    tags=["Health"],
)
async def get_health(request: Request) -> HealthResponse:
    """Health check endpoint."""
    _require_model_preload_ready(request)
    return HealthResponse(
        version=__version__,
        features=HealthFeatures(),
    )


# ── Models ───────────────────────────────────────────────────────────


@_router.get(
    "/models",
    response_model=ModelListResponse,
    status_code=status.HTTP_200_OK,
    responses=_ERR_503,
    tags=["Models"],
)
async def list_models(request: Request) -> ModelListResponse:
    """List all available parsing models."""
    _require_model_preload_ready(request)
    model_ids: list[str] = request.app.state.model_ids
    registered_at: int = request.app.state.model_registered_at
    return ModelListResponse(
        data=[ModelInfo(id=mid, created=registered_at, description=_VIRTUAL_MODEL_DESCRIPTION) for mid in model_ids],
    )


@_router.get(
    "/models/{model}",
    response_model=ModelInfo,
    status_code=status.HTTP_200_OK,
    responses={**_ERR_404, **_ERR_403, **_ERR_503},
    tags=["Models"],
)
async def get_model(
    request: Request,
    model: str = Path(description="Model ID"),
) -> ModelInfo:
    """Retrieve a single model by ID."""
    _require_model_preload_ready(request)
    model_ids: list[str] = request.app.state.model_ids
    if model not in model_ids:
        raise_api_error(
            404,
            code="model_not_found",
            message=f"Model '{model}' not found",
        )
    return ModelInfo(id=model, created=request.app.state.model_registered_at, description=_VIRTUAL_MODEL_DESCRIPTION)


# ── Tiers ────────────────────────────────────────────────────────────


@_router.get(
    "/tiers",
    response_model=TierListResponse,
    status_code=status.HTTP_200_OK,
    responses=_ERR_503,
    tags=["Tiers"],
)
async def list_tiers(request: Request) -> TierListResponse:
    """发布 PDF/图片可选档位；原生文档的固定 Flash 能力不由该列表控制。"""
    _require_model_preload_ready(request)
    tiers: list[TierInfoData] = request.app.state.tiers
    return TierListResponse(
        data=[TierInfo(**p) for p in tiers],
    )


# ── Files ────────────────────────────────────────────────────────────


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


class _OutputFileResponse(FileResponse):
    """把文件租约绑定到 ASGI 响应，正常结束和断连都释放引用。"""

    def __init__(self, store: FileStore, file_id: str) -> None:
        """在响应头发送之前完成权限检查，并固定实际文件路径。"""
        path, filename, sha = store.pin_output_file(file_id)
        self._store = store
        self._sha = sha
        try:
            super().__init__(
                path,
                filename=filename,
                media_type={
                    ".zip": "application/zip",
                    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    ".html": "text/html",
                }.get(pathlib.Path(filename).suffix.lower(), "application/octet-stream"),
            )
        except BaseException:
            store.unpin_cached_blob(sha)
            raise

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """让文件响应负责分块和 Range，同时确保取消不会泄漏租约。"""
        try:
            await super().__call__(scope, receive, send)
        finally:
            self._store.unpin_cached_blob(self._sha)


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
    """直接流式发送已保存的产物，下载期间持有文件租约。"""
    return _OutputFileResponse(store, file_id)


# ── 私有解析执行 ─────────────────────────────────────────────────────


async def submit_execution(
    body: _ExecutionRequest,
    request: Request,
    file_store: FileStore,
    execution_store: ExecutionStore,
    *,
    execution_id: str | None = None,
) -> _ExecutionRecord:
    """校验单文件执行请求并接入唯一并发队列，不再创建内部批次或按序号映射。"""
    try:
        body.page_range = normalize_page_range_input(body.page_range) or None
    except MineruError as exc:
        raise_api_error(400, code=exc.code, message=str(exc), param="files.0.page_range")
    _require_model_preload_ready(request)
    default_tier: Tier | None = request.app.state.default_tier
    can_default_to_flash = _execution_can_default_to_flash(body)
    native_document = _execution_requires_flash(body)
    if native_document:
        body.tier = "flash"
    if body.tier is None and default_tier is None:
        if can_default_to_flash:
            body.tier = "flash"
        else:
            raise_api_error(
                503,
                code="quality_tier_unavailable",
                message=(
                    "No basic, standard, or advanced tier available in this server. "
                    "Pass tier='flash' explicitly for flash parsing."
                ),
            )
    else:
        body.tier = body.tier or default_tier
    runtime_options: dict[Tier, ParserRuntimeOptions] = request.app.state.tier_runtime_options
    # 公开 runtime 表仅约束 PDF/图片；原生文档始终走独立 Flash 路径。
    runtime = runtime_options_for_tier("flash") if native_document else runtime_options.get(body.tier)
    if runtime is None and can_default_to_flash:
        runtime = runtime_options_for_tier("flash")
    if runtime is None:
        raise_api_error(
            400,
            code="invalid_request",
            message=f"Tier '{body.tier}' not available in this server",
        )
    _validate_execution_source_policy(body, allow_http_source=request.app.state.allow_http_source)
    if execution_store._closing:
        raise_api_error(503, code="server_shutting_down", message="Server is shutting down")
    rec = execution_store.create(body, file_store, execution_id=execution_id)
    rec.retention_held = True
    url_timeout_val: int = request.app.state.url_timeout
    allow_http_source_val: bool = request.app.state.allow_http_source
    max_url_bytes_val: int = request.app.state.max_url_bytes
    image_analysis_val: bool = request.app.state.image_analysis
    vlm_config_val: VlmConfig = request.app.state.vlm_config

    async def _bg_run() -> None:
        """运行单文件内容，排队、资源租约和取消排空仍由执行存储管理。"""
        await _run_execution(
            rec,
            file_store,
            image_analysis=image_analysis_val,
            url_timeout=url_timeout_val,
            allow_http_source=allow_http_source_val,
            max_url_bytes=max_url_bytes_val,
            vlm_config=vlm_config_val,
            execution_store=execution_store,
        )

    async def _bg_prepare() -> None:
        """沿用来源配置准备 URL 输入，完整下载并校验后才进入解析队列。"""
        await _prepare_execution_input(
            rec,
            file_store,
            url_timeout=url_timeout_val,
            allow_http_source=allow_http_source_val,
            max_url_bytes=max_url_bytes_val,
        )

    try:
        execution_store.start_task(rec, _bg_run, prepare=_bg_prepare if isinstance(body.source, _UrlSource) else None)
    except BaseException:
        file_store.release_inputs(rec.input_blobs)
        execution_store._executions.pop(rec.id, None)
        execution_store._jobs_created -= 1
        raise
    return rec


# ── Usage ─────────────────────────────────────────────────────────────


@_router.get(
    "/usage",
    response_model=UsageResponse,
    status_code=status.HTTP_200_OK,
    tags=["Usage"],
)
async def get_usage(
    request: Request,
    execution_store: ExecutionStore = Depends(_get_execution_store),
    access_level: AccessLevel = Depends(_resolve_access_level),
) -> UsageResponse:
    """Query current usage and limits."""
    return execution_store.usage(access_level)


# ═══════════════════════════════════════════════════════════════════════
#  APP FACTORY
# ═══════════════════════════════════════════════════════════════════════


def _build_v1_router() -> APIRouter:
    """Build and return the /v1 router."""
    return _router


def _runtime_options_for_server_tiers(tiers: Collection[Tier]) -> dict[Tier, ParserRuntimeOptions]:
    """为明确开放的请求 tier 生成独立 runtime，执行 effort 仍由 tier 决定。"""
    return {tier: runtime_options_for_tier(tier) for tier in tiers}


def _tiers_for_server_tier(tier: Tier) -> list[TierInfoData]:
    """单档请求 tier 的展示 metadata；current_model 统一指向唯一虚拟模型。"""
    if tier == "flash":
        return [
            {
                "id": "flash",
                "description": "Fast local text extraction.",
                "current_model": _VIRTUAL_MODEL_ID,
            },
        ]
    if tier == "basic":
        return [
            {
                "id": "basic",
                "description": "Basic parsing with local lightweight models.",
                "current_model": _VIRTUAL_MODEL_ID,
            },
        ]
    if tier == "advanced":
        return [
            {
                "id": "advanced",
                "description": "Advanced parsing for difficult documents.",
                "current_model": _VIRTUAL_MODEL_ID,
            },
        ]

    return [
        {
            "id": "standard",
            "description": "Standard parsing for most documents.",
            "current_model": _VIRTUAL_MODEL_ID,
        },
    ]


def _tiers_for_server_tiers(tiers: Collection[Tier]) -> list[TierInfoData]:
    """按精确集合发布请求 tier metadata。"""
    return [tier_info for tier in tiers for tier_info in _tiers_for_server_tier(tier)]


def _preflight_tier_dependencies(tier: DeploymentTier | None, vlm_config: VlmConfig | None = None) -> None:
    """远程 VLM 仅预检本地 Hybrid 依赖，避免要求安装本地 VLM serving 引擎。"""
    if tier is None:
        return
    settings = vlm_config if vlm_config is not None else get_config().model.vlm
    if tier == "standard":
        settings.validate_environment()
    try:
        ensure_tier_runtime_dependencies(tier, vlm_config=settings)
    except TierDependencyError as exc:
        raise ParseServerStartupError(str(exc)) from exc


@dataclass(frozen=True)
class _ModelPreloadResult:
    tier: DeploymentTier
    engine: str


@dataclass(frozen=True)
class _ModelPreloadError:
    code: str
    message: str


_MODEL_PRELOAD_DEVICE_ERROR_PATTERNS = (
    "cuda is not available",
    "device is not available",
    "failed to infer device type",
    "no available accelerator",
    "no available gpu",
    "unsupported device",
    "only supported on macos",
)


def _classify_model_preload_error(exc: Exception) -> tuple[str, str]:
    message = str(exc).strip() or type(exc).__name__
    if is_fatal_engine_error(exc):
        return "engine_dead", message
    if isinstance(exc, (ModuleNotFoundError, ImportError)):
        return "model_preload_dependency_missing", message
    if isinstance(exc, FileNotFoundError):
        return "model_preload_files_missing", message

    normalized_message = message.lower()
    if any(pattern in normalized_message for pattern in _MODEL_PRELOAD_DEVICE_ERROR_PATTERNS):
        return "model_preload_device_unavailable", message
    return "model_preload_failed", message


def _preload_local_models() -> None:
    # TODO: fix import
    from ..model.runtime.contracts import AtomicModelName
    from ..model.runtime.hybrid import HybridLocalModelContextSingleton

    context = HybridLocalModelContextSingleton().get_model()
    manager = context.atom_model_manager
    manager.get_atom_model(atom_model_name=AtomicModelName.TableOrientationCls)
    manager.get_atom_model(atom_model_name=AtomicModelName.TableCls)
    # 表格模型语言与 HybridLocalModelContext 运行时的缓存键保持一致，确保预热的模型可被复用。
    manager.get_atom_model(atom_model_name=AtomicModelName.WirelessTable, lang="ch")
    manager.get_atom_model(atom_model_name=AtomicModelName.WiredTable, lang="ch")
    manager.get_atom_model(atom_model_name=AtomicModelName.OCR, lang="seal")


def _preload_server_models(tier: DeploymentTier, *, vlm_config: VlmConfig | None = None) -> _ModelPreloadResult:
    """使用与实际解析相同的 VLM 客户端，并预加载本地 Hybrid 模型。"""
    if tier == "basic":
        _preload_local_models()
        return _ModelPreloadResult(tier=tier, engine="hybrid-local")

    from ..model.vlm.client import get_vlm_predictor

    _predictor, engine = get_vlm_predictor(vlm_config)
    _preload_local_models()
    return _ModelPreloadResult(tier=tier, engine=engine)


_SHUTDOWN_FORCE_EXIT_SEC = 30.0


def _start_force_exit_timer(
    seconds: float = _SHUTDOWN_FORCE_EXIT_SEC,
    *,
    exit_fn: Callable[[int], None] = os._exit,
) -> threading.Timer:
    """优雅关闭的兜底强杀：托管子进程不应比父进程存活更久。

    uvicorn 的 ``should_exit`` 是优雅语义，会等待 in-flight 请求完成；父进程
    （doclib）死亡时正在执行的 parse 请求会让托管子进程以孤儿形态存活整个请求
    时长并继续占用端口。控制通道触发的关闭因此附带强杀上限。
    """
    timer = threading.Timer(seconds, exit_fn, args=(1,))
    timer.daemon = True
    timer.start()
    return timer


def create_app(
    *,
    upload_dir: str = "",
    tiers: Collection[Tier] | None = None,
    concurrency: int = 1,
    url_timeout: int = 60,
    max_url_bytes: int = _MAX_URL_BYTES_DEFAULT,
    allow_http_source: bool = False,
    api_key: str | None = None,
    image_analysis: bool = True,
    preload_models: bool = False,
    vlm_config: VlmConfig | None = None,
    retention_seconds: int | None = None,
) -> FastAPI:
    """Create a FastAPI application implementing the MinerU v1 REST API.

    Parameters
    ----------
    upload_dir:
        Directory for uploaded files and parse artifacts.
    tiers:
        Exact non-empty set of PDF/image request tiers. ``None`` recommends
        a capability ceiling from the effective small-model backend and hardware.
        Torch validates acceleration without OCR calibration; ONNX may calibrate
        before finalizing a preliminary ceiling of Basic or higher on Windows/Linux.
        Native documents always use Flash independently of this set.
    concurrency:
        Maximum concurrent parse jobs (default 1).
    url_timeout:
        Timeout in seconds for downloading url sources (default 60).
    max_url_bytes:
        Maximum bytes downloaded for ``url`` sources; larger responses are
        aborted mid-download instead of being buffered into memory.
    allow_http_source:
        Whether ``url`` sources may use plain HTTP. HTTPS is always allowed.
    api_key:
        Optional API key.  When set, clients must pass ``Authorization: Bearer <key>``
        to access list endpoints and advanced output formats.
    image_analysis:
        Whether image analysis is enabled for Hybrid backends.
    preload_models:
        Initialize the VLM client and local Hybrid models at startup. Ignored for Flash.
    vlm_config:
        Complete VLM connection configuration override; defaults to global model.vlm settings.
    """
    retention_seconds = resolve_retention_seconds(retention_seconds)
    upload_dir = upload_dir or ""
    vlm_config = (vlm_config if vlm_config is not None else get_config().model.vlm).model_copy(deep=True)
    server_tiers = resolve_server_tiers(tiers, vlm_config=vlm_config)
    model_tier = model_tier_for_server_tiers(server_tiers)
    tier_runtime_options = _runtime_options_for_server_tiers(server_tiers)
    default_tier = select_default_task_tier(server_tiers)
    _preflight_tier_dependencies(model_tier, vlm_config)
    _api_key: str | None = api_key or None
    _upload_dir = pathlib.Path(upload_dir) if upload_dir else pathlib.Path(tempfile.mkdtemp(prefix="mineru_"))
    _upload_dir.mkdir(parents=True, exist_ok=True)
    if max_url_bytes < 0:
        raise ValueError("max_url_bytes must be non-negative")

    _tiers = _tiers_for_server_tiers(server_tiers)
    _model_registered_at = int(time.time())
    _preload_tier = model_tier if preload_models else None

    @asynccontextmanager
    async def _lifespan(application: FastAPI) -> AsyncIterator[None]:
        application.state.upload_dir = _upload_dir
        application.state.server_tiers = server_tiers
        application.state.model_tier = model_tier
        application.state.default_tier = default_tier
        application.state.tier_runtime_options = tier_runtime_options
        application.state.model_ids = [_VIRTUAL_MODEL_ID]
        application.state.model_registered_at = _model_registered_at
        application.state.tiers = _tiers
        application.state.concurrency = concurrency
        application.state.url_timeout = url_timeout
        application.state.max_url_bytes = max_url_bytes
        application.state.allow_http_source = allow_http_source
        application.state.api_key = _api_key
        application.state.image_analysis = image_analysis
        application.state.preload_models = preload_models
        application.state.vlm_config = vlm_config
        application.state.model_preload_error = None
        execution_store: ExecutionStore = application.state.execution_store
        file_store: FileStore = application.state.file_store
        file_store.claim_directory()
        application.state.task_store.start()
        collection_task: asyncio.Task[None] | None = None

        async def collect_loop() -> None:
            """每五分钟扫描，先清理终态元数据，再释放无引用文件。"""
            while True:
                await asyncio.sleep(RETENTION_SCAN_INTERVAL_SECONDS)
                execution_store.collect_expired()
                await run_sync(file_store.collect_expired)

        if retention_seconds:
            collection_task = asyncio.create_task(collect_loop(), name="mineru-api-retention")
        token = runtime_owner.set(execution_store.runtime_owner)
        try:
            file_store.collect_expired()
            if _preload_tier is not None:
                logger.info("Initializing VLM client and local models for startup tier %s", _preload_tier)
                try:
                    preload_result = await run_sync(
                        _preload_server_models,
                        _preload_tier,
                        vlm_config=vlm_config,
                    )
                except Exception as exc:
                    error_code, error_msg = _classify_model_preload_error(exc)
                    if error_code == "engine_dead":
                        execution_store.fail_engine(error_msg)
                    application.state.model_preload_error = _ModelPreloadError(
                        code=error_code,
                        message=error_msg,
                    )
                    logger.exception("Model initialization failed for startup tier %s", _preload_tier)
                else:
                    logger.info(
                        "Model initialization completed for startup tier %s (engine=%s)",
                        _preload_tier,
                        preload_result.engine,
                    )
            yield
        finally:
            runtime_owner.reset(token)
            if collection_task is not None:
                collection_task.cancel()
                await asyncio.gather(collection_task, return_exceptions=True)
            try:
                await application.state.task_store.shutdown()
                await execution_store.shutdown()
            finally:
                file_store.release_directory()
                if not upload_dir and _upload_dir.exists():
                    shutil.rmtree(_upload_dir, ignore_errors=True)

    enable_docs = _env_flag("MINERU_API_ENABLE_FASTAPI_DOCS", default=True)

    app_factory = FastAPIOffline if enable_docs else FastAPI
    application = app_factory(
        **({"static_url": "/docs/assets/vendor"} if enable_docs else {}),
        title="MinerU API",
        description=_FASTAPI_DESCRIPTION,
        version="1.0.0",
        openapi_tags=_OPENAPI_TAGS,
        dependencies=[Depends(HTTPBearer(auto_error=False, description="Bearer API key when configured"))],
        openapi_url="/openapi.json" if enable_docs else None,
        docs_url=None,
        redoc_url=None,
        lifespan=_lifespan,
    )
    if enable_docs:

        @application.get("/docs", include_in_schema=False)
        async def _offline_swagger(request: Request) -> Response:
            """使用安装包内资源，并为反向代理补全 OAuth 回调的路径前缀。"""
            root = request.scope.get("root_path", "").rstrip("/")
            return get_swagger_ui_html(
                openapi_url=f"{root}/openapi.json",
                title="MinerU API - Swagger UI",
                swagger_js_url=f"{root}/docs/assets/vendor/swagger-ui-bundle.js",
                swagger_css_url=f"{root}/docs/assets/vendor/swagger-ui.css",
                swagger_favicon_url=f"{root}/docs/assets/vendor/favicon.png",
                oauth2_redirect_url=f"{root}/docs/oauth2-redirect",
            )

        @application.get("/redoc", include_in_schema=False)
        async def _offline_redoc(request: Request) -> Response:
            """ReDoc 使用本地脚本、图标和字体，不请求远程品牌素材。"""
            root = request.scope.get("root_path", "").rstrip("/")
            return get_redoc_html(
                openapi_url=f"{root}/openapi.json",
                title="MinerU API - ReDoc",
                redoc_js_url=f"{root}/docs/assets/redoc.standalone.js",
                redoc_favicon_url=f"{root}/docs/assets/vendor/favicon.png",
                with_google_fonts=False,
            )

        @application.get("/docs/assets/redoc.standalone.js", include_in_schema=False)
        async def _offline_redoc_script() -> Response:
            """沿用安装包内 ReDoc，替换其硬编码 CDN 图标为同包内嵌图标。"""
            resources = package_resources("fastapi_offline").joinpath("static")
            script = resources.joinpath("redoc.standalone.js").read_text(encoding="utf-8")
            icon = base64.b64encode(resources.joinpath("favicon.png").read_bytes()).decode("ascii")
            script = script.replace("https://cdn.redoc.ly/redoc/logo-mini.svg", f"data:image/png;base64,{icon}")
            return Response(script, media_type="application/javascript")

        @application.get("/docs/oauth2-redirect", include_in_schema=False)
        async def _offline_oauth_redirect() -> Response:
            """保留 Swagger 的本地 OAuth 授权回调页。"""
            return get_swagger_ui_oauth2_redirect_html()

    application.state.upload_dir = _upload_dir
    application.state.server_tiers = server_tiers
    application.state.model_tier = model_tier
    application.state.default_tier = default_tier
    application.state.tier_runtime_options = tier_runtime_options
    application.state.model_ids = [_VIRTUAL_MODEL_ID]
    application.state.model_registered_at = _model_registered_at
    application.state.tiers = _tiers
    application.state.concurrency = concurrency
    application.state.url_timeout = url_timeout
    application.state.max_url_bytes = max_url_bytes
    application.state.allow_http_source = allow_http_source
    application.state.api_key = _api_key
    application.state.image_analysis = image_analysis
    application.state.preload_models = preload_models
    application.state.vlm_config = vlm_config
    application.state.model_preload_error = None
    FileStore(_upload_dir, retention_seconds=retention_seconds).install(application.state)
    ExecutionStore(concurrency=concurrency, retention_seconds=retention_seconds).install(application.state)
    application.add_middleware(GZipMiddleware, minimum_size=1000)

    @application.exception_handler(ApiServerError)
    async def _api_server_error_handler(request: Request, exc: ApiServerError) -> JSONResponse:
        return _error_response(exc.status_code, exc.error)

    @application.exception_handler(RequestValidationError)
    async def _request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error_response(
            400,
            ErrorDetail(
                code=_validation_error_code(exc),
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
                code="internal_error",
                message="Internal server error",
            ),
        )

    _PUBLIC_PATHS = frozenset({"/v1/health", "/v1/models", "/v1/tiers"})
    _PUBLIC_PREFIXES = ("/openapi", "/docs", "/redoc")

    async def _auth_middleware(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        path = request.url.path
        root = request.scope.get("root_path", "").rstrip("/")
        if root and path.startswith(root + "/"):
            path = path[len(root) :]
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
                    code="invalid_api_key",
                    message="Invalid or missing API key",
                ),
            )
        return await call_next(request)

    application.middleware("http")(_auth_middleware)
    application.include_router(_build_v1_router())
    from .task_backend import ApiTaskBackend

    application.state.task_store = TaskStore(retention_seconds)
    application.include_router(build_task_router(ApiTaskBackend))
    return application


# ── CLI ──────────────────────────────────────────────────────────────


def _build_server_log_config(log_level: str) -> dict[str, Any]:
    """为服务日志构造独立配置，保留 root、模型日志和进度条的既有设置。"""
    log_config = deepcopy(uvicorn.config.LOGGING_CONFIG)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "uvicorn.asgi"):
        log_config["loggers"].setdefault(name, {})["level"] = uvicorn.config.LOG_LEVELS[log_level]
    log_config["loggers"][logger.name] = {
        "handlers": ["default"],
        "level": uvicorn.config.LOG_LEVELS[log_level],
        "propagate": False,
    }
    return log_config


@click.command()
@click.option(
    "--retention-seconds",
    default=None,
    type=click.IntRange(min=0),
    envvar="MINERU_API_RETENTION_SECONDS",
    help="Terminal resource retention in seconds (default: 86400; 0 disables collection)",
)
@click.option("--host", default="127.0.0.1", help="Server host")
@click.option("--port", default=8000, type=int, help="Server port")
@click.option(
    "--log-level",
    default=None,
    type=click.Choice(["critical", "error", "warning", "info", "debug", "trace"], case_sensitive=False),
    help=(
        "API service log level: critical, error, warning, info, debug, trace; default: global log.level. "
        "Server-scope only (Uvicorn startup, HTTP access, ASGI, and service logs); "
        "the Loguru default sink always follows global log.level."
    ),
)
@click.option(
    "--upload-dir",
    default="",
    help="Upload directory (default: auto-created temp dir)",
    type=click.Path(file_okay=False, writable=True, resolve_path=True),
)
@click.option(
    "--tier",
    default="auto",
    type=str,
    help="PDF/image tier ceiling: flash, basic, standard, advanced, or auto (default). Native documents always use Flash.",
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
    "--max-url-bytes",
    default=_MAX_URL_BYTES_DEFAULT,
    type=int,
    help=f"Maximum bytes downloaded for url sources (default: {_MAX_URL_BYTES_DEFAULT})",
)
@click.option(
    "--allow-http-source",
    is_flag=True,
    help="Allow url sources to use plain HTTP. HTTPS is always allowed.",
)
@click.option(
    "--disable-image-analysis",
    is_flag=True,
    help="Disable image analysis for Hybrid backends.",
)
@click.option(
    "--preload-models",
    is_flag=True,
    help="Initialize VLM client and local Hybrid models at startup. Ignored for Flash.",
)
@click.option("--vlm-server-url", default=None, type=str, help="Remote VLM URL; empty value selects local VLM.")
@click.option("--vlm-api-key", default=None, type=str, help="Bearer key for the remote VLM server.")
@click.option("--vlm-model", default=None, type=str, help="Remote VLM model name; empty value enables discovery.")
@click.option(
    "--vlm-http-timeout", default=None, type=click.IntRange(min=1), help="VLM HTTP timeout in seconds (default: 600)."
)
@click.option(
    "--vlm-max-concurrency", default=None, type=click.IntRange(min=1), help="VLM inference concurrency (default: 100)."
)
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
    tier: str,
    concurrency: int,
    url_timeout: int,
    max_url_bytes: int,
    allow_http_source: bool,
    disable_image_analysis: bool,
    preload_models: bool,
    api_key: str | None,
    vlm_server_url: str | None,
    vlm_api_key: str | None,
    vlm_model: str | None,
    vlm_http_timeout: int | None,
    vlm_max_concurrency: int | None,
    log_level: str | None,
    retention_seconds: int | None = None,
) -> None:
    """合并显式 VLM 参数后启动 MinerU v1 REST API 服务，不修改全局配置。"""
    configure_standard_streams()
    global_log_level = configure_global_log_level()
    server_log_level = (log_level if log_level is not None else global_log_level).lower()
    shutdown_requested = threading.Event()
    server_ref: list[uvicorn.Server | None] = [None]

    def _request_shutdown() -> None:
        first_request = not shutdown_requested.is_set()
        shutdown_requested.set()
        server = server_ref[0]
        if server is not None:
            server.should_exit = True
        if first_request:
            _start_force_exit_timer()

    try:
        control_watcher = ManagedProcessControlWatcher.from_environment(_request_shutdown)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from None
    if control_watcher is not None:
        control_watcher.start()
    try:
        vlm_overrides = {
            "server_url": vlm_server_url,
            "api_key": vlm_api_key,
            "model": vlm_model,
            "http_timeout": vlm_http_timeout,
            "max_concurrency": vlm_max_concurrency,
        }
        vlm_settings = VlmConfig.model_validate(
            {
                **get_config().model.vlm.model_dump(),
                **{key: value for key, value in vlm_overrides.items() if value is not None},
            }
        )
        application = create_app(
            upload_dir=upload_dir,
            tiers=parse_server_tier(tier),
            concurrency=concurrency,
            url_timeout=url_timeout,
            max_url_bytes=max_url_bytes,
            allow_http_source=allow_http_source,
            api_key=api_key,
            image_analysis=not disable_image_analysis,
            preload_models=preload_models,
            vlm_config=vlm_settings,
            retention_seconds=retention_seconds,
        )
    except ParseServerStartupError as exc:
        if control_watcher is not None:
            control_watcher.close()
        raise click.ClickException(str(exc)) from None
    except ValueError as exc:
        if control_watcher is not None:
            control_watcher.close()
        raise click.ClickException(str(exc)) from exc

    config = uvicorn.Config(
        application,
        host=host,
        port=port,
        log_level=server_log_level,
        log_config=_build_server_log_config(server_log_level),
    )
    server = uvicorn.Server(config)
    server_ref[0] = server
    if shutdown_requested.is_set():
        server.should_exit = True
    try:
        server.run()
    finally:
        if control_watcher is not None:
            control_watcher.close()


if __name__ == "__main__":
    main()


__all__ = ["create_app", "main"]
