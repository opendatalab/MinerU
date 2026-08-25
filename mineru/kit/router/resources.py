# Copyright (c) Opendatalab. All rights reserved.
"""Router 对外资源标识与 upstream 路由信息的内存注册表。"""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
    worker_id: str
    upstream_id: str
    created_at: str = field(default_factory=utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)


class ResourceRegistry:
    """维护当前 Router 进程创建或发现的 uploads、files 与 jobs。"""

    def __init__(self) -> None:
        """初始化按公共标识和 upstream 标识建立的双向索引。"""
        self._by_public: dict[ResourceKind, dict[str, ResourceRoute]] = {
            "upload": {},
            "file": {},
            "job": {},
        }
        self._by_upstream: dict[tuple[ResourceKind, str, str], ResourceRoute] = {}

    def register(
        self,
        kind: ResourceKind,
        *,
        worker_id: str,
        upstream_id: str,
        metadata: dict[str, Any] | None = None,
        public_id: str | None = None,
    ) -> ResourceRoute:
        """注册资源并复用同一 worker/upstream 标识已有的公共映射。"""
        upstream_key = (kind, worker_id, upstream_id)
        existing = self._by_upstream.get(upstream_key)
        if existing is not None:
            if metadata is not None:
                existing.metadata = dict(metadata)
            return existing

        resolved_public_id = public_id or self._new_public_id(kind)
        route = ResourceRoute(
            kind=kind,
            public_id=resolved_public_id,
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

    def find_upstream(self, kind: ResourceKind, worker_id: str, upstream_id: str) -> ResourceRoute | None:
        """按 worker 与 upstream 标识读取已有公共映射。"""
        return self._by_upstream.get((kind, worker_id, upstream_id))

    def alias_upstream(self, route: ResourceRoute, *, worker_id: str, upstream_id: str) -> None:
        """把同一公共资源在另一 worker 中的复制标识绑定到现有记录。"""
        self._by_upstream[(route.kind, worker_id, upstream_id)] = route

    def list(self, kind: ResourceKind) -> list[ResourceRoute]:
        """按注册顺序返回指定类型的全部资源路由。"""
        return list(self._by_public[kind].values())

    def remove(self, kind: ResourceKind, public_id: str) -> ResourceRoute | None:
        """删除公共资源及其反向索引，并返回被删除的记录。"""
        route = self._by_public[kind].pop(public_id, None)
        if route is not None:
            self._by_upstream.pop((kind, route.worker_id, route.upstream_id), None)
        return route

    @staticmethod
    def _new_public_id(kind: ResourceKind) -> str:
        """生成保持 V1 前缀约定的随机 Router 公共标识。"""
        return _PUBLIC_ID_PREFIXES[kind] + secrets.token_hex(12)


__all__ = ["ResourceKind", "ResourceRegistry", "ResourceRoute", "utc_now_iso"]
