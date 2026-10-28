# Copyright (c) Opendatalab. All rights reserved.
"""应用与文档的模型租约，统一协调共享推理资源的释放。"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar


class RuntimeOwner:
    """标识应用持有的模型租约，并等待已接纳的文档自然完成。"""

    def __init__(self) -> None:
        """初始化关闭状态、引擎错误及受保护的在途文档计数。"""
        self.closed = False
        self.engine_error: str | None = None
        self._documents = 0
        self._initializations = 0
        self._condition = threading.Condition()

    def acquire_document(self) -> None:
        """接纳新文档；关闭开始后不允许新的模型使用范围进入。"""
        with self._condition:
            if self.closed:
                raise RuntimeError("Runtime owner is closed")
            self._documents += 1

    def release_document(self) -> None:
        """文档全部清理后归还租约，唤醒等待应用资源回收的线程。"""
        with self._condition:
            self._documents -= 1
            self._condition.notify_all()

    def close(self) -> None:
        """拒绝新文档并等待既有文档退出，不强停同步模型工作。"""
        lease = _document_lease.get()
        if lease is not None and lease.active and lease.owner is self:
            raise RuntimeError("Cannot release a runtime owner from its active document")
        initialization = _initialization_lease.get()
        if initialization is not None and initialization.active and initialization.owner is self:
            raise RuntimeError("Cannot release a runtime owner from its active initialization")
        with self._condition:
            self.closed = True
            while self._documents or self._initializations:
                self._condition.wait()

    def acquire_initialization(self) -> None:
        """接纳一次模型获取或初始化，使应用关闭等待构造及失败清理完整结束。"""
        with self._condition:
            if current_runtime_owner() is not self:
                raise RuntimeError("Runtime initialization owner does not match the current context")
            self._initializations += 1

    def release_initialization(self) -> None:
        """构造结果已发布或清理完毕后归还初始化租约。"""
        with self._condition:
            self._initializations -= 1
            self._condition.notify_all()


class _RuntimeLease:
    """让复制到工作线程的上下文共享同一个有效期标记。"""

    def __init__(self, owner: RuntimeOwner) -> None:
        """保留接纳当前操作的应用身份，并标记操作尚未退出。"""
        self.owner = owner
        self.active = True


runtime_owner: ContextVar[RuntimeOwner | None] = ContextVar("mineru_runtime_owner", default=None)
_document_lease: ContextVar[_RuntimeLease | None] = ContextVar("mineru_runtime_document", default=None)
_initialization_lease: ContextVar[_RuntimeLease | None] = ContextVar("mineru_runtime_initialization", default=None)


def current_runtime_owner() -> RuntimeOwner | None:
    """校验借用权限；关闭中的应用仅允许仍有效的既有文档惰性加载模型。"""
    owner = runtime_owner.get()
    lease = _document_lease.get()
    if lease is not None and lease.owner is owner and not lease.active:
        raise RuntimeError("Runtime document lease has ended")
    initialization = _initialization_lease.get()
    if initialization is not None and initialization.owner is owner and not initialization.active:
        raise RuntimeError("Runtime initialization lease has ended")
    if owner is not None and owner.closed and not (lease is not None and lease.owner is owner and lease.active):
        raise RuntimeError("Runtime owner is closed")
    return owner


@contextmanager
def runtime_document_scope() -> Iterator[None]:
    """为整份文档保留应用模型，异步线程复制上下文时也沿用同一租约。"""
    owner = runtime_owner.get()
    if owner is None:
        yield
        return
    owner.acquire_document()
    lease = _RuntimeLease(owner)
    token = _document_lease.set(lease)
    try:
        yield
    finally:
        lease.active = False
        _document_lease.reset(token)
        owner.release_document()


@contextmanager
def runtime_initialization_scope() -> Iterator[None]:
    """先校验应用再解析模型路径；构造期间关闭应用时，等待清理而不发布迟到实例。"""
    owner = current_runtime_owner()
    if owner is None:
        yield
        return
    owner.acquire_initialization()
    lease = _RuntimeLease(owner)
    token = _initialization_lease.set(lease)
    try:
        yield
    finally:
        lease.active = False
        _initialization_lease.reset(token)
        owner.release_initialization()


def release_runtime_owner(owner: RuntimeOwner) -> None:
    """排空已接纳文档后释放所有模型池，某个池失败也继续清理其余资源。"""
    owner.close()
    from .hybrid import AtomModelSingleton, HybridLocalModelContextSingleton
    from ..vlm.runtime import ModelSingleton

    first_error: BaseException | None = None
    for release in (
        ModelSingleton().release_owner,
        HybridLocalModelContextSingleton().release_owner,
        AtomModelSingleton().release_owner,
    ):
        try:
            release(owner)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
    if first_error is not None:
        raise first_error


__all__ = [
    "RuntimeOwner",
    "current_runtime_owner",
    "release_runtime_owner",
    "runtime_document_scope",
    "runtime_initialization_scope",
    "runtime_owner",
]
