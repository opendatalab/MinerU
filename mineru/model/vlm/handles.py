# Copyright (c) Opendatalab. All rights reserved.
"""宿主持有的 VLM 客户端与确定的后端关闭动作。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING

from ...utils.async_utils import drain_future, run_sync
from ..runtime.closing import runtime_close_scope
from ..runtime.ownership import RuntimeOwner

if TYPE_CHECKING:
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.structs import ContentBlock, ExtractResult
    from PIL.Image import Image


class RuntimeHandle:
    """将客户端、串行执行锁和创建时确定的资源释放动作放在同一宿主对象中。"""

    def __init__(
        self,
        predictor: MinerUClient,
        *,
        backend: str,
        close_engine: Callable[[], None] | None = None,
    ) -> None:
        """保存已创建资源，不修改第三方客户端的属性或推测其内部结构。"""
        self.backend = backend
        self.owners: set[RuntimeOwner] = set()
        self.pinned = False
        self.execution_lock = threading.Lock() if backend == "mlx-engine" else None
        self._predictor: MinerUClient | None = predictor
        self._close_engine = close_engine
        self._condition = threading.Condition()
        self._active = 0
        self._closing = False
        self._closed = threading.Event()

    @property
    def predictor(self) -> MinerUClient:
        """只向托管异步运行时交付仍有效的底层客户端。"""
        if self._predictor is None:
            raise RuntimeError("VLM runtime is closed")
        return self._predictor

    @property
    def is_closed(self) -> bool:
        """让缓存识别正在关闭或已经关闭的宿主句柄。"""
        return self._closing

    @contextmanager
    def _use(self, *, acquire_lock: bool = True) -> Iterator[MinerUClient]:
        """等待同步推理自然返回后才允许关闭引擎，MLX 使用宿主串行锁。"""
        with self._condition:
            if self._closing:
                raise RuntimeError("VLM runtime is shutting down")
            self._active += 1
            predictor = self.predictor
        try:
            if self.execution_lock is None or not acquire_lock:
                yield predictor
            else:
                with self.execution_lock:
                    yield predictor
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    @asynccontextmanager
    async def _async_use(self) -> AsyncIterator[MinerUClient]:
        """可取消地等待串行锁，排队期间取消不得让迟到工作线程开始推理。"""
        lock = self.execution_lock
        if lock is not None:
            while True:
                with self._condition:
                    if self._closing:
                        raise RuntimeError("VLM runtime is shutting down")
                    if lock.acquire(blocking=False):
                        break
                await asyncio.sleep(0.01)
        try:
            with self._use(acquire_lock=False) as predictor:
                yield predictor
        finally:
            if lock is not None:
                lock.release()

    def batch_extract_with_layout(
        self,
        images: list[Image],
        blocks_list: Sequence[Sequence[ContentBlock | dict]],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """同步抽取期间持有客户端，关闭操作等待本次调用完整退出。"""
        with self._use() as predictor:
            return predictor.batch_extract_with_layout(
                images, blocks_list, not_extract_list=not_extract_list, image_analysis=image_analysis
            )

    def batch_two_step_extract(
        self,
        images: list[Image],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """同步两阶段推理沿用同一资源使用范围，不强制中断 CPU 执行。"""
        with self._use() as predictor:
            return predictor.batch_two_step_extract(images, not_extract_list=not_extract_list, image_analysis=image_analysis)

    async def aio_batch_extract_with_layout(
        self,
        images: list[Image],
        blocks_list: Sequence[Sequence[ContentBlock | dict]],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """同步后端的异步调用等待工作线程排空，取消不会提前关闭资源。"""
        async with self._async_use() as predictor:
            return await run_sync(
                predictor.batch_extract_with_layout,
                images,
                blocks_list,
                not_extract_list=not_extract_list,
                image_analysis=image_analysis,
            )

    async def aio_batch_two_step_extract(
        self,
        images: list[Image],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """在线程中运行同步两阶段推理，并保留取消后的资源清理边界。"""
        async with self._async_use() as predictor:
            return await run_sync(
                predictor.batch_two_step_extract, images, not_extract_list=not_extract_list, image_analysis=image_analysis
            )

    def _begin_close(self) -> bool:
        """只让一个调用方执行关闭，其余调用方等待同一个完成信号。"""
        with self._condition:
            if self._closing:
                return False
            self._closing = True
            return True

    def _wait_for_calls(self) -> None:
        """在缓存锁之外等待所有同步调用退出。"""
        with self._condition:
            while self._active:
                self._condition.wait()

    def _finish_close(self, first_error: BaseException | None = None) -> None:
        """执行唯一确定的后端关闭动作，并在失败时同样释放宿主强引用。"""
        close_engine, self._close_engine = self._close_engine, None
        try:
            if close_engine is not None:
                with runtime_close_scope():
                    close_engine()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        finally:
            self._predictor = None
            self._closed.set()
        if first_error is not None:
            raise first_error

    def shutdown(self) -> None:
        """同步后端排空调用后关闭；HTTP 句柄的连接关闭仍交给其公开异步接口。"""
        if not self._begin_close():
            self._closed.wait()
            return
        error: BaseException | None = None
        try:
            with runtime_close_scope():
                self._wait_for_calls()
                if self.backend == "http-client":
                    asyncio.run(self.predictor.aclose())
        except BaseException as exc:
            error = exc
        self._finish_close(error)

    async def _close_client(self) -> None:
        """在受保护的任务中排空推理并关闭客户端，等待异常也不跳过连接清理。"""
        error: BaseException | None = None
        with runtime_close_scope():
            try:
                await run_sync(self._wait_for_calls)
            except BaseException as exc:
                error = exc
            try:
                await self.predictor.aclose()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    async def aclose(self) -> None:
        """取消时仍排空客户端关闭，再释放引擎并传播最早的取消或清理异常。"""
        if not self._begin_close():
            await run_sync(self._closed.wait)
            return
        work = asyncio.create_task(self._close_client())
        error: BaseException | None = None
        try:
            await asyncio.shield(work)
        except BaseException as exc:
            error = exc
            try:
                await drain_future(work)
            except BaseException:
                pass
        self._finish_close(error)


__all__ = ["RuntimeHandle"]
