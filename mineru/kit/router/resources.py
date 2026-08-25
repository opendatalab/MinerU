# Copyright (c) Opendatalab. All rights reserved.
"""Router 对外资源标识与 upstream 路由信息的内存注册表。"""

from __future__ import annotations

import hashlib
import secrets
import tempfile
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

ResourceKind = Literal["upload", "file", "job"]

_PUBLIC_ID_PREFIXES: dict[ResourceKind, str] = {
    "upload": "upload_",
    "file": "file-",
    "job": "job_",
}


def utc_now_iso() -> str:
    """返回与 V1 API 一致的 UTC ISO-8601 时间。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class ResourceRoute:
    """记录一个 Router 公共资源在具体 upstream 中的真实标识。"""

    kind: ResourceKind
    public_id: str
    owner_scope: str
    worker_id: str
    upstream_id: str
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StoredSourceFile:
    """记录 Router 私有暂存输入的路径、大小、哈希和媒体类型。"""

    path: Path
    bytes: int
    sha256sum: str
    mime_type: str


@dataclass(frozen=True)
class CopiedInputFile:
    """记录一个 Job 为 cross-worker 执行创建的目标 worker 输入副本。"""

    source_public_id: str
    owner_scope: str
    worker_id: str
    upstream_file_id: str


class SourceFileStore:
    """在 Router 临时目录中保存可供 cross-worker 重传的输入字节。"""

    def __init__(self) -> None:
        """创建由当前 Router 进程独占并在关闭时清理的临时目录。"""
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mineru-v1-router-sources-")
        self._root = Path(self._temp_dir.name)
        self._uploads: dict[str, StoredSourceFile] = {}
        self._files: dict[str, StoredSourceFile] = {}
        self._bound_uploads: set[str] = set()

    async def stage_upload(
        self,
        upload_id: str,
        chunks: AsyncIterator[bytes],
        *,
        mime_type: str,
    ) -> StoredSourceFile:
        """流式写入一个公共 Upload 的输入字节，并计算实际大小与 SHA256。"""
        if upload_id in self._bound_uploads:
            raise ValueError(f"Upload {upload_id} is already bound to a completed file")
        path = self._root / "uploads" / upload_id
        path.parent.mkdir(parents=True, exist_ok=True)
        hasher = hashlib.sha256()
        byte_count = 0
        with path.open("wb") as output:
            async for chunk in chunks:
                if not chunk:
                    continue
                output.write(chunk)
                hasher.update(chunk)
                byte_count += len(chunk)
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

    def bind_file(self, upload_id: str, file_id: str) -> StoredSourceFile:
        """把已完成 Upload 的暂存输入绑定到 Router 公共 File。"""
        stored = self._uploads.pop(upload_id)
        self._files[file_id] = stored
        self._bound_uploads.add(upload_id)
        return stored

    def is_bound_upload(self, upload_id: str) -> bool:
        """判断 Upload 的暂存路径是否已经绑定到完成后的公共 File。"""
        return upload_id in self._bound_uploads

    def find_upload(self, upload_id: str) -> StoredSourceFile | None:
        """读取公共 Upload 对应的私有暂存输入，不存在时返回 None。"""
        return self._uploads.get(upload_id)

    def find_file(self, file_id: str) -> StoredSourceFile | None:
        """读取公共 File 对应的私有暂存输入，不存在时返回 None。"""
        return self._files.get(file_id)

    def discard_upload(self, upload_id: str) -> None:
        """删除取消或失败 Upload 的私有暂存输入。"""
        if upload_id in self._bound_uploads:
            return
        stored = self._uploads.pop(upload_id, None)
        if stored is not None:
            stored.path.unlink(missing_ok=True)

    def delete_file(self, file_id: str) -> None:
        """删除公共 File 绑定的私有暂存输入。"""
        stored = self._files.pop(file_id, None)
        if stored is not None:
            stored.path.unlink(missing_ok=True)

    def close(self) -> None:
        """清理当前 Router 进程的全部暂存输入。"""
        self._uploads.clear()
        self._files.clear()
        self._bound_uploads.clear()
        self._temp_dir.cleanup()


async def stored_file_chunks(path: Path, chunk_size: int = 1024 * 1024) -> AsyncIterator[bytes]:
    """按固定块大小异步迭代一个 Router 私有暂存文件。"""
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            yield chunk


class ResourceRegistry:
    """维护当前 Router 进程创建或发现的 uploads、files 与 jobs。"""

    def __init__(self) -> None:
        """初始化按公共标识和 upstream 标识建立的双向索引。"""
        self._by_public: dict[ResourceKind, dict[str, ResourceRoute]] = {
            "upload": {},
            "file": {},
            "job": {},
        }
        self._by_upstream: dict[tuple[ResourceKind, str, str, str], ResourceRoute] = {}

    def register(
        self,
        kind: ResourceKind,
        *,
        owner_scope: str,
        worker_id: str,
        upstream_id: str,
        metadata: dict[str, Any] | None = None,
        public_id: str | None = None,
    ) -> ResourceRoute:
        """注册资源并复用同一 worker/upstream 标识已有的公共映射。"""
        upstream_key = (kind, owner_scope, worker_id, upstream_id)
        existing = self._by_upstream.get(upstream_key)
        if existing is not None:
            if metadata is not None:
                existing.metadata = dict(metadata)
            return existing

        resolved_public_id = public_id or self._new_public_id(kind)
        route = ResourceRoute(
            kind=kind,
            public_id=resolved_public_id,
            owner_scope=owner_scope,
            worker_id=worker_id,
            upstream_id=upstream_id,
            metadata=dict(metadata or {}),
        )
        self._by_public[kind][resolved_public_id] = route
        self._by_upstream[upstream_key] = route
        return route

    def get(self, kind: ResourceKind, public_id: str) -> ResourceRoute:
        """按公共标识读取资源路由，不存在时抛出 KeyError。"""
        return self._by_public[kind][public_id]

    def find(self, kind: ResourceKind, public_id: str) -> ResourceRoute | None:
        """按公共标识读取资源路由，不存在时返回 None。"""
        return self._by_public[kind].get(public_id)

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

    def list(self, kind: ResourceKind, *, owner_scope: str | None = None) -> list[ResourceRoute]:
        """按注册顺序返回指定类型、可选调用方 scope 的资源路由。"""
        routes = list(self._by_public[kind].values())
        if owner_scope is None:
            return routes
        return [route for route in routes if route.owner_scope == owner_scope]

    def remove(self, kind: ResourceKind, public_id: str) -> ResourceRoute | None:
        """删除公共资源及其反向索引，并返回被删除的记录。"""
        route = self._by_public[kind].pop(public_id, None)
        if route is not None:
            self._by_upstream.pop((kind, route.owner_scope, route.worker_id, route.upstream_id), None)
        return route

    @staticmethod
    def _new_public_id(kind: ResourceKind) -> str:
        """生成保持 V1 前缀约定的随机 Router 公共标识。"""
        return _PUBLIC_ID_PREFIXES[kind] + secrets.token_hex(12)


__all__ = [
    "CopiedInputFile",
    "ResourceKind",
    "ResourceRegistry",
    "ResourceRoute",
    "SourceFileStore",
    "StoredSourceFile",
    "stored_file_chunks",
    "utc_now_iso",
]
