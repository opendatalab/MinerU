# Copyright (c) Opendatalab. All rights reserved.
"""Router 对外资源标识与 upstream 路由信息的内存注册表。"""

from __future__ import annotations

import builtins
import hashlib
import secrets
import tempfile
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Literal, cast, overload

from ...utils.retention import resolve_retention_seconds
from .route_state import ExecutionRouteState, ResultLease

ResourceKind = Literal["file", "job"]

_PUBLIC_ID_PREFIXES: dict[ResourceKind, str] = {
    "file": "file-",
    "job": "job_",
}


def utc_now_iso() -> str:
    """返回与 V1 API 一致的 UTC ISO-8601 时间。"""
    return datetime.fromtimestamp(time.time(), timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(kw_only=True)
class ResourceRoute(ABC):
    """记录一个 Router 公共资源在具体 upstream 中的真实标识。"""

    public_id: str
    owner_scope: str
    worker_id: str
    upstream_id: str
    created_at: str = field(default_factory=utc_now_iso)
    payload: dict[str, Any] | None = None
    pins: int = 0
    lease: ResultLease = field(default_factory=ResultLease)

    @property
    @abstractmethod
    def kind(self) -> ResourceKind:
        """资源种类由具体路由固定，注册后不能更改并破坏正反索引。"""
        raise NotImplementedError


@dataclass(kw_only=True)
class FileRoute(ResourceRoute):
    """保存文件映射及其原始响应缓存，内部生命周期字段不进入响应字典。"""

    @property
    def kind(self) -> Literal["file"]:
        """文件路由在全部索引和查询中保持固定种类。"""
        return "file"


@dataclass(kw_only=True)
class ExecutionRoute(ResourceRoute):
    """私有单文件执行映射始终具有完整的执行状态和明确的终态时间。"""

    state: ExecutionRouteState
    finalized_at: str | None = None

    @property
    def kind(self) -> Literal["job"]:
        """执行路由沿用内部 job 索引，公开 Task 协议保持不变。"""
        return "job"


@dataclass(frozen=True)
class StoredSourceFile:
    """记录 Router 私有暂存输入的路径、大小、哈希和媒体类型。"""

    path: Path
    bytes: int
    sha256sum: str
    mime_type: str


class SourceFileStore:
    """在 Router 临时目录中保存可供 cross-worker 重传的输入字节。"""

    def __init__(self, retention_seconds: int = 86400) -> None:
        """创建由当前 Router 进程独占并在关闭时清理的临时目录。"""
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mineru-v1-router-sources-")
        self._root = Path(self._temp_dir.name)
        self.retention_seconds = retention_seconds
        self._file_expiry: dict[str, float | None] = {}
        self._active_references: set[str] = set()
        self._uploads: dict[str, StoredSourceFile] = {}
        self._files: dict[str, StoredSourceFile] = {}
        self._bound_uploads: set[str] = set()
        self._file_scopes: dict[str, str] = {}
        self._hash_files: dict[tuple[str, str], set[str]] = {}
        self._pins: dict[Path, int] = {}

    async def stage_upload(
        self,
        upload_id: str,
        chunks: AsyncIterator[bytes],
        *,
        mime_type: str,
        max_bytes: int | None = None,
    ) -> StoredSourceFile:
        """流式写入私有暂存输入，并计算实际大小与 SHA256。"""
        if upload_id in self._bound_uploads:
            raise ValueError(f"Upload {upload_id} is already bound to a completed file")
        path = self._root / "uploads" / upload_id
        path.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        byte_count = 0
        try:
            with path.open("wb") as output:
                async for chunk in chunks:
                    if not chunk:
                        continue
                    byte_count += len(chunk)
                    if max_bytes is not None and byte_count > max_bytes:
                        raise ValueError("file_too_large")
                    output.write(chunk)
                    hasher.update(chunk)
        except BaseException:
            path.unlink(missing_ok=True)
            self._uploads.pop(upload_id, None)
            raise
        stored = StoredSourceFile(
            path=path,
            bytes=byte_count,
            sha256sum=hasher.hexdigest(),
            mime_type=mime_type,
        )
        previous = self._uploads.get(upload_id)
        self._uploads[upload_id] = stored
        if previous is not None and previous.path != path:
            previous.path.unlink(missing_ok=True)
        return stored

    def bind_file(
        self, upload_id: str, file_id: str, *, owner_scope: str = "anonymous", expires_at: int | None = None
    ) -> StoredSourceFile:
        """把已完成上传的暂存输入绑定到 Router 私有源引用。"""
        stored = self._uploads.pop(upload_id)
        self.bind_source(file_id, stored, owner_scope=owner_scope, expires_at=expires_at)
        self._bound_uploads.add(upload_id)
        return stored

    def bind_source(self, file_id: str, stored: StoredSourceFile, *, owner_scope: str, expires_at: int | None = None) -> None:
        """增加私有源缓存引用，并建立调用方隔离的哈希索引。"""
        previous = self._files.get(file_id)
        if previous is not None:
            previous_scope = self._file_scopes.get(file_id, "anonymous")
            previous_key = (previous_scope, previous.sha256sum)
            ids = self._hash_files.get(previous_key, set())
            ids.discard(file_id)
            if not ids:
                self._hash_files.pop(previous_key, None)
        self._file_expiry[file_id] = (
            expires_at
            if expires_at is not None
            else (int(time.time()) + self.retention_seconds if self.retention_seconds else None)
        )
        self._files[file_id] = stored
        self._file_scopes[file_id] = owner_scope
        self._hash_files.setdefault((owner_scope, stored.sha256sum), set()).add(file_id)
        if previous is not None:
            self._release_path(previous.path)

    def hold_reference(self, reference_id: str) -> None:
        """准备和执行阶段固定源引用，父任务清理前不开始保留期。"""
        if reference_id not in self._files:
            raise KeyError(reference_id)
        self._active_references.add(reference_id)
        self._file_expiry[reference_id] = None

    def release_reference(self, reference_id: str, finished: float) -> None:
        """首次解除任务引用时开始缓存保留期，重复释放不延长期限。"""
        if reference_id not in self._active_references:
            return
        self._active_references.remove(reference_id)
        self._file_expiry[reference_id] = finished + self.retention_seconds if self.retention_seconds else None

    def collect_expired(self) -> None:
        """独立回收过期私有引用，传输中的字节仍由 pin 保护。"""
        now = time.time()
        for reference_id, deadline in tuple(self._file_expiry.items()):
            if reference_id not in self._active_references and deadline is not None and deadline <= now:
                self.delete_file(reference_id)

    def find_hash(self, owner_scope: str, sha256sum: str) -> StoredSourceFile | None:
        """只查当前调用方仍持有的源字节，不使用 worker 的哈希声明。"""
        for file_id in sorted(self._hash_files.get((owner_scope, sha256sum), set())):
            stored = self._files.get(file_id)
            deadline = self._file_expiry.get(file_id)
            if stored is not None and (deadline is None or deadline > time.time()) and stored.path.is_file():
                return stored
        return None

    @contextmanager
    def pin(self, stored: StoredSourceFile) -> Iterator[None]:
        """在跨 worker 传输期间保留字节，直到最后一个文件或传输引用退出。"""
        self._pins[stored.path] = self._pins.get(stored.path, 0) + 1
        try:
            yield
        finally:
            self._pins[stored.path] -= 1
            if not self._pins[stored.path]:
                self._pins.pop(stored.path)
            self._release_path(stored.path)

    def _release_path(self, path: Path) -> None:
        """仅删除没有上传、文件和传输引用的暂存路径。"""
        if self._pins.get(path):
            return
        if any(stored.path == path for stored in (*self._uploads.values(), *self._files.values())):
            return
        path.unlink(missing_ok=True)

    def find_file(self, file_id: str) -> StoredSourceFile | None:
        """按私有源引用读取缓存输入，不存在时返回 None。"""
        return self._files.get(file_id)

    def discard_upload(self, upload_id: str) -> None:
        """删除取消或失败 Upload 的私有暂存输入。"""
        if upload_id in self._bound_uploads:
            self._bound_uploads.discard(upload_id)
            return
        stored = self._uploads.pop(upload_id, None)
        if stored is not None:
            self._release_path(stored.path)

    def delete_file(self, file_id: str) -> None:
        """删除私有源引用，只有最后一个引用及传输租约释放后才删除字节。"""
        self._active_references.discard(file_id)
        self._file_expiry.pop(file_id, None)
        stored = self._files.pop(file_id, None)
        if stored is not None:
            scope = self._file_scopes.pop(file_id, "anonymous")
            key = (scope, stored.sha256sum)
            files = self._hash_files.get(key, set())
            files.discard(file_id)
            if not files:
                self._hash_files.pop(key, None)
            self._release_path(stored.path)

    def close(self) -> None:
        """清理当前 Router 进程的全部暂存输入。"""
        self._uploads.clear()
        self._files.clear()
        self._bound_uploads.clear()
        self._file_scopes.clear()
        self._file_expiry.clear()
        self._active_references.clear()
        self._hash_files.clear()
        self._pins.clear()
        self._temp_dir.cleanup()


async def stored_file_chunks(path: Path, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    """按固定块大小异步迭代一个 Router 私有暂存文件。"""
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            yield chunk


class ResourceRegistry:
    """维护当前 Router 进程的文件与私有执行路由。"""

    def __init__(self, retention_seconds: int | None = None) -> None:
        """初始化按公共标识和 upstream 标识建立的双向索引。"""
        self.retention_seconds = resolve_retention_seconds(retention_seconds)
        self._by_public: dict[ResourceKind, dict[str, ResourceRoute]] = {
            "file": {},
            "job": {},
        }
        self._by_upstream: dict[tuple[ResourceKind, str, str, str], ResourceRoute] = {}
        self._retired_usage: dict[str, dict[str, int]] = {}
        self._allocated_file_ids: set[str] = set()
        self._reserved_file_ids: set[str] = set()
        self._task_file_ids: set[str] = set()

    def reserve_file_id(self, *, task_owned: bool = True) -> str:
        """在本层预留公共文件 ID，完成前不登记可访问的 Files 路由。"""
        while True:
            file_id = self._new_public_id("file")
            if file_id not in self._allocated_file_ids and file_id not in self._by_public["file"]:
                self._allocated_file_ids.add(file_id)
                self._reserved_file_ids.add(file_id)
                if task_owned:
                    self._task_file_ids.add(file_id)
                return file_id

    def forget_file_id(self, file_id: str) -> None:
        """任务记录过期后回收占用索引，仍存在的公开资源保持唯一性。"""
        self._task_file_ids.discard(file_id)
        self._reserved_file_ids.discard(file_id)
        if file_id not in self._by_public["file"]:
            self._allocated_file_ids.discard(file_id)

    def is_file_reserved(self, file_id: str) -> bool:
        """区分仍可发布的产物预留与已经取消、删除或过期的资源。"""
        return file_id in self._reserved_file_ids

    def discard_file_id(self, file_id: str) -> None:
        """撤销失败或取消文件的预留与结果映射，保留已用 ID 防止迟到结果复活。"""
        self._reserved_file_ids.discard(file_id)
        self.remove("file", file_id)

    def register_file(
        self,
        *,
        owner_scope: str,
        worker_id: str,
        upstream_id: str,
        public_id: str | None = None,
    ) -> FileRoute:
        """注册文件映射，复用同一 upstream 标识时保留租约、传输固定和响应缓存。"""
        existing = self.find_upstream("file", owner_scope, worker_id, upstream_id)
        if existing is not None:
            if public_id is not None and public_id != existing.public_id:
                raise ValueError("Upstream resource is already bound to another public ID")
            return existing
        route = FileRoute(
            public_id=public_id or self.reserve_file_id(task_owned=False),
            owner_scope=owner_scope,
            worker_id=worker_id,
            upstream_id=upstream_id,
        )
        self._insert_route(route)
        return route

    def register_execution(
        self,
        *,
        owner_scope: str,
        worker_id: str,
        upstream_id: str,
        state: ExecutionRouteState,
        payload: dict[str, Any] | None = None,
    ) -> ExecutionRoute:
        """执行快照与路由同时登记，禁止先发布缺少身份和预占归属的执行记录。"""
        existing = self.find_upstream("job", owner_scope, worker_id, upstream_id)
        if existing is not None:
            if state.execution.execution_id != existing.public_id:
                raise ValueError("Upstream resource is already bound to another public ID")
            existing.state = state
            if payload is not None:
                existing.payload = dict(payload)
            return existing
        route = ExecutionRoute(
            public_id=state.execution.execution_id,
            owner_scope=owner_scope,
            worker_id=worker_id,
            upstream_id=upstream_id,
            created_at=state.execution.created_at,
            state=state,
            payload=dict(payload) if payload is not None else None,
        )
        self._insert_route(route)
        return route

    def _insert_route(self, route: ResourceRoute) -> None:
        """共用身份冲突、文件预留消费和正反索引写入，状态由具体注册入口构造。"""
        if route.public_id in self._by_public[route.kind]:
            raise ValueError("Public resource ID is already occupied")
        if route.kind == "file":
            if route.public_id in self._allocated_file_ids and route.public_id not in self._reserved_file_ids:
                raise ValueError("File reservation has already been consumed or discarded")
            self._allocated_file_ids.add(route.public_id)
            self._reserved_file_ids.discard(route.public_id)
        self._by_public[route.kind][route.public_id] = route
        self._by_upstream[(route.kind, route.owner_scope, route.worker_id, route.upstream_id)] = route

    @contextmanager
    def pin_route(self, route: ResourceRoute) -> Iterator[None]:
        """资源流式写入或复制期间保留路由；到期后新的请求仍不可访问。"""
        route.pins += 1
        try:
            yield
        finally:
            route.pins -= 1

    def is_expired(self, route: ResourceRoute) -> bool:
        """优先沿用上游显式期限；任务从清理完成时间开始计算保留期限。"""
        if route.lease.held:
            until = route.lease.retain_until
            return until is not None and until <= time.time()
        payload = route.payload or {}
        if isinstance(route, ExecutionRoute):
            if payload.get("status") not in {"completed", "partial", "failed", "canceled"}:
                return False
            if not self.retention_seconds:
                return False
            if route.finalized_at is None:
                route.finalized_at = utc_now_iso()
            origin = route.finalized_at
        else:
            explicit = payload.get("expires_at")
            if isinstance(explicit, (int, float)):
                return explicit <= time.time()
            if not self.retention_seconds:
                return False
            origin = route.created_at
        try:
            stamp = datetime.fromisoformat(str(origin).replace("Z", "+00:00")).timestamp()
        except ValueError:
            return False
        return stamp + self.retention_seconds <= time.time()

    def collect_expired(self, sources: SourceFileStore) -> None:
        """同步清理路由和源缓存，运行或待回收副本仍持有的文件暂不物理删除。"""
        protected: set[str] = set()
        for route in self._by_public["job"].values():
            payload = route.payload or {}
            if payload.get("status") not in {"completed", "partial", "failed", "canceled"}:
                protected.update(file.get("file_id") for file in payload.get("files", []) if file.get("file_id"))
        for kind, routes in self._by_public.items():
            for public_id, route in tuple(routes.items()):
                if not self.is_expired(route) or public_id in protected or route.pins or route.lease.held:
                    continue
                self.remove(kind, public_id)
                if kind == "file":
                    sources.delete_file(public_id)

    @overload
    def get(self, kind: Literal["file"], public_id: str) -> FileRoute:
        """文件查询返回具有文件生命周期字段的映射。"""
        ...

    @overload
    def get(self, kind: Literal["job"], public_id: str) -> ExecutionRoute:
        """执行查询返回带有强制执行状态的映射。"""
        ...

    def get(self, kind: ResourceKind, public_id: str) -> ResourceRoute:
        """按公共标识读取资源路由，不存在时抛出 KeyError。"""
        return self._by_public[kind][public_id]

    @overload
    def find(self, kind: Literal["file"], public_id: str) -> FileRoute | None:
        """按文件身份返回有效文件映射。"""
        ...

    @overload
    def find(self, kind: Literal["job"], public_id: str) -> ExecutionRoute | None:
        """按执行身份返回有效执行映射。"""
        ...

    def find(self, kind: ResourceKind, public_id: str) -> ResourceRoute | None:
        """按公共标识读取资源路由，不存在时返回 None。"""
        route = self._by_public[kind].get(public_id)
        return route if route is not None and not self.is_expired(route) else None

    @overload
    def find_upstream(self, kind: Literal["file"], owner_scope: str, worker_id: str, upstream_id: str) -> FileRoute | None:
        """上游文件别名始终映射到文件记录。"""
        ...

    @overload
    def find_upstream(self, kind: Literal["job"], owner_scope: str, worker_id: str, upstream_id: str) -> ExecutionRoute | None:
        """上游执行别名始终映射到有类型的执行记录。"""
        ...

    def find_upstream(
        self,
        kind: ResourceKind,
        owner_scope: str,
        worker_id: str,
        upstream_id: str,
    ) -> ResourceRoute | None:
        """按 worker 与 upstream 标识读取已有公共映射。"""
        return self._by_upstream.get((kind, owner_scope, worker_id, upstream_id))

    def alias_upstream(self, route: ResourceRoute, *, worker_id: str, upstream_id: str) -> None:
        """把同一公共资源在另一 worker 中的复制标识绑定到现有记录。"""
        self._by_upstream[(route.kind, route.owner_scope, worker_id, upstream_id)] = route

    def remove_upstream_alias(
        self,
        kind: ResourceKind,
        *,
        owner_scope: str,
        worker_id: str,
        upstream_id: str,
    ) -> None:
        """删除 cross-worker 副本对应的反向 alias，不影响公共资源主映射。"""
        self._by_upstream.pop((kind, owner_scope, worker_id, upstream_id), None)

    @overload
    def list(self, kind: Literal["file"], *, owner_scope: str | None = None, include_expired: bool = False) -> list[FileRoute]:
        """文件枚举不暴露执行专用状态。"""
        ...

    @overload
    def list(
        self, kind: Literal["job"], *, owner_scope: str | None = None, include_expired: bool = False
    ) -> list[ExecutionRoute]:
        """执行枚举可直接访问执行快照、预占及私有认证状态。"""
        ...

    def list(
        self, kind: ResourceKind, *, owner_scope: str | None = None, include_expired: bool = False
    ) -> list[FileRoute] | list[ExecutionRoute]:
        """按注册顺序返回指定类型、可选调用方 scope 的资源路由。"""
        routes = [route for route in self._by_public[kind].values() if include_expired or not self.is_expired(route)]
        if owner_scope is not None:
            routes = [route for route in routes if route.owner_scope == owner_scope]
        # 两个显式注册入口保证各 kind 的索引只包含其对应的具体路由类型。
        return cast(list[FileRoute], routes) if kind == "file" else cast(list[ExecutionRoute], routes)

    def remove(self, kind: ResourceKind, public_id: str) -> ResourceRoute | None:
        """删除公共资源及其反向索引，并返回被删除的记录。"""
        route = self._by_public[kind].pop(public_id, None)
        if kind == "file" and public_id not in self._task_file_ids and public_id not in self._reserved_file_ids:
            self._allocated_file_ids.discard(public_id)
        if route is not None:
            if kind == "job":
                totals = self._retired_usage.setdefault(route.owner_scope, {"jobs_created": 0, "files_processed": 0})
                for usage_key, count in self._job_usage(route).items():
                    totals[usage_key] += count
            for upstream_key, mapped_route in tuple(self._by_upstream.items()):
                if mapped_route is route:
                    self._by_upstream.pop(upstream_key)
        return route

    def remove_worker(self, worker_id: str) -> builtins.list[ResourceRoute]:
        """删除指定 worker generation 拥有的公共路由及相关反向 aliases。"""
        removed: list[ResourceRoute] = []
        for routes in self._by_public.values():
            for public_id, route in list(routes.items()):
                if route.worker_id != worker_id or (isinstance(route, ExecutionRoute) and route.state.upstream_lost):
                    continue
                removed.append(route)
                self.remove(route.kind, public_id)
        removed_route_ids = {id(route) for route in removed}
        for key, route in list(self._by_upstream.items()):
            if key[2] == worker_id or id(route) in removed_route_ids:
                self._by_upstream.pop(key, None)
        return removed

    def worker_route_count(self, worker_id: str) -> int:
        """计算正反索引引用的不同路由，已过期但尚未回收或仍有租约的记录也阻止注销。"""
        route_ids = {
            id(route) for routes in self._by_public.values() for route in routes.values() if route.worker_id == worker_id
        }
        route_ids.update(id(route) for key, route in self._by_upstream.items() if key[2] == worker_id)
        return len(route_ids)

    def usage(self, owner_scope: str) -> dict[str, int]:
        """按调用方聚合存活与已移除任务的累计用量，到期隐藏不影响统计。"""
        totals = {"jobs_created": 0, "files_processed": 0, **self._retired_usage.get(owner_scope, {})}
        for route in self.list("job", owner_scope=owner_scope, include_expired=True):
            for key, value in self._job_usage(route).items():
                totals[key] += value
        return totals

    @staticmethod
    def _job_usage(route: ResourceRoute) -> dict[str, int]:
        """统计一个已创建任务和成功文件，保持 Router 原有逐文件统计语义。"""
        payload = route.payload or {}
        completed = payload.get("status") in {"completed", "partial"}
        return {
            "jobs_created": 1,
            "files_processed": sum(
                1
                for file in payload.get("files") or []
                if completed and isinstance(file, dict) and file.get("status") == "completed"
            ),
        }

    @staticmethod
    def _new_public_id(kind: ResourceKind) -> str:
        """生成保持 V1 前缀约定的随机 Router 公共标识。"""
        return _PUBLIC_ID_PREFIXES[kind] + secrets.token_hex(12)


__all__ = [
    "ExecutionRoute",
    "FileRoute",
    "ResourceKind",
    "ResourceRegistry",
    "ResourceRoute",
    "SourceFileStore",
    "StoredSourceFile",
    "stored_file_chunks",
    "utc_now_iso",
]
