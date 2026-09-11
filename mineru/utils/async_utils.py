# Copyright (c) Opendatalab. All rights reserved.
"""异步编排中的取消与同步资源生命周期边界。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


async def drain_future(future: asyncio.Future[T]) -> T:
    """即使重复取消也等待已有任务结束；仅用于必须完成的资源清理。"""
    while not future.done():
        try:
            await asyncio.shield(future)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break
    return future.result()


async def run_sync(function: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """在线程中执行同步阶段，取消后等待线程完成再允许调用方释放资源。"""
    work = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(work)
    except asyncio.CancelledError:
        try:
            await drain_future(work)
        except BaseException:
            pass
        raise


__all__ = ["drain_future", "run_sync"]
