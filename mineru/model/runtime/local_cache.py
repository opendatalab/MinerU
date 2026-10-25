# Copyright (c) Opendatalab. All rights reserved.
"""本地模型缓存的显式依赖租约，保护共享上下文及嵌套原子模型。"""

from __future__ import annotations

import gc
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from .ownership import RuntimeOwner, current_runtime_owner

LOCAL_MODEL_CACHE_LOCK = threading.RLock()


@dataclass(eq=False)
class LocalModelResource:
    """记录缓存对象及应用、父模型和子模型之间的真实持有关系。"""

    cache: dict[object, LocalModelResource]
    key: object
    value: Any = None
    owners: set[RuntimeOwner] = field(default_factory=set)
    parents: set[LocalModelResource] = field(default_factory=set)
    dependencies: set[LocalModelResource] = field(default_factory=set)
    pinned: bool = False
    closed: bool = False


_current_resource: ContextVar[LocalModelResource | None] = ContextVar("mineru_local_model_resource", default=None)


def current_local_resource() -> LocalModelResource | None:
    """让新建 Hybrid 上下文保留其缓存节点，供后续惰性加载登记依赖。"""
    return _current_resource.get()


@contextmanager
def local_resource_scope(resource: LocalModelResource | None) -> Iterator[None]:
    """将后续原子模型借用登记为指定上下文或模型的依赖。"""
    token = _current_resource.set(resource)
    try:
        yield
    finally:
        _current_resource.reset(token)


def _collect_unused(resource: LocalModelResource, discarded: list[object]) -> None:
    """在缓存锁内移除无人持有节点，延迟到锁外才释放对象强引用。"""
    if resource.closed or resource.pinned or resource.owners or resource.parents:
        return
    resource.closed = True
    if resource.cache.get(resource.key) is resource:
        del resource.cache[resource.key]
    if resource.value is not None:
        discarded.append(resource.value)
        resource.value = None
    dependencies = tuple(resource.dependencies)
    resource.dependencies.clear()
    for dependency in dependencies:
        dependency.parents.discard(resource)
        _collect_unused(dependency, discarded)


def get_local_model(cache: dict[object, LocalModelResource], key: object, factory: Callable[[], Any]) -> Any:
    """沿现有缓存键构造或复用模型，初始化失败时回滚其未被共享的依赖。"""
    discarded: list[object] = []
    try:
        with LOCAL_MODEL_CACHE_LOCK:
            owner = current_runtime_owner()
            parent = _current_resource.get()
            if parent is not None and parent.closed:
                raise RuntimeError("Local model context is closed")
            resource = cache.get(key)
            if resource is None:
                resource = LocalModelResource(cache, key)
                try:
                    with local_resource_scope(resource):
                        resource.value = factory()
                except BaseException:
                    _collect_unused(resource, discarded)
                    raise
                cache[key] = resource
            if parent is not None:
                parent.dependencies.add(resource)
                resource.parents.add(parent)
            elif owner is None:
                resource.pinned = True
            else:
                resource.owners.add(owner)
            return resource.value
    finally:
        # ORT/Torch 小模型没有统一 close API；释放宿主持有引用即可让各自对象回收资源。
        discarded.clear()


def release_local_owner(cache: dict[object, LocalModelResource], owner: RuntimeOwner) -> None:
    """撤销应用租约并递归释放无其它持有者的模型，垃圾回收不持有初始化锁。"""
    owner.close()
    discarded: list[object] = []
    with LOCAL_MODEL_CACHE_LOCK:
        for resource in tuple(cache.values()):
            resource.owners.discard(owner)
            _collect_unused(resource, discarded)
    if discarded:
        discarded.clear()
        gc.collect()


__all__ = [
    "LOCAL_MODEL_CACHE_LOCK",
    "LocalModelResource",
    "current_local_resource",
    "get_local_model",
    "local_resource_scope",
    "release_local_owner",
]
