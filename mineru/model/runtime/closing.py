# Copyright (c) Opendatalab. All rights reserved.
"""模型关闭回调的上下文，防止清理过程中等待正在关闭的资源。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

_runtime_closing: ContextVar[bool] = ContextVar("mineru_runtime_closing", default=False)


@contextmanager
def runtime_close_scope() -> Iterator[None]:
    """标记当前关闭调用链，调度到异步运行时后仍保留回调身份。"""
    token = _runtime_closing.set(True)
    try:
        yield
    finally:
        _runtime_closing.reset(token)


def in_runtime_close_scope() -> bool:
    """判断当前是否处于资源关闭回调，供缓存入口拒绝阻塞式重入。"""
    return _runtime_closing.get()


__all__ = ["in_runtime_close_scope", "runtime_close_scope"]
