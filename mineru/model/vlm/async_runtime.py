# Copyright (c) Opendatalab. All rights reserved.
"""在固定事件循环中复用 VLM 客户端，并跨线程桥接同步、异步请求。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
from collections.abc import Awaitable, Callable, Sequence
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any, TypeVar

from ...utils.async_utils import drain_future, run_sync

if TYPE_CHECKING:
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.structs import ContentBlock, ExtractResult
    from PIL.Image import Image

T = TypeVar("T")


class RuntimeOwner:
    """标识应用持有的模型租约；独立解析使用进程级缓存租约。"""

    def __init__(self) -> None:
        """记录关闭状态，阻止已关闭应用重新取得模型。"""
        self.closed = False


runtime_owner: ContextVar[RuntimeOwner | None] = ContextVar("mineru_vlm_runtime_owner", default=None)


class _Submission:
    """分离请求结果与清理完成信号，避免 Future 取消被误认为推理退出。"""

    def __init__(self) -> None:
        """所有任务字段只在所属事件循环线程中读写。"""
        self.result: concurrent.futures.Future[Any] = concurrent.futures.Future()
        self.finished: concurrent.futures.Future[None] = concurrent.futures.Future()
        self.task: asyncio.Task[Any] | None = None
        self.cancel_requested = False

    def cancel(self) -> None:
        """只向底层任务发送一次取消，重复调用不打断已开始的清理。"""
        if not self.cancel_requested:
            self.cancel_requested = True
            if self.task is not None:
                self.task.cancel()


class AsyncVlmPredictor:
    """只暴露文档抽取接口，统一持有原生异步客户端、并发额度与事件循环。"""

    def __init__(
        self,
        factory: Callable[[], MinerUClient],
        dispose: Callable[[MinerUClient], None],
        *,
        backend: str,
        max_concurrency: int,
    ) -> None:
        """启动专属线程；模型构造在该线程的运行中事件循环内执行。"""
        if max_concurrency < 1:
            raise ValueError("VLM max_concurrency must be positive")
        self.backend = backend
        self.max_concurrency = max_concurrency
        self.owners: set[RuntimeOwner] = set()
        self.pinned = False
        self._factory = factory
        self._dispose = dispose
        self._ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._state_lock = threading.Lock()
        self._closing = False
        self._shutdown_lock = threading.Lock()
        self._submissions: set[_Submission] = set()
        self._thread = threading.Thread(target=self._serve, name=f"mineru-{backend}", daemon=True)
        self._thread.start()

    async def _initialize(self) -> None:
        """在固定循环创建引擎和共享请求信号量。"""
        self._predictor = self._factory()
        self._semaphore = asyncio.Semaphore(self.max_concurrency)

    def _serve(self) -> None:
        """运行独立事件循环，退出前回收异步生成器及线程池工作。"""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._initialize())
            self._ready.set_result(None)
            self._loop.run_forever()
        except BaseException as exc:
            if not self._ready.done():
                self._ready.set_exception(exc)
        finally:
            pending = asyncio.all_tasks(self._loop)
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.run_until_complete(self._loop.shutdown_default_executor())
            self._loop.close()

    @property
    def is_closed(self) -> bool:
        """供缓存边界识别已经关闭或启动失败的旧代理。"""
        return self._closing or not self._thread.is_alive()

    def wait_ready(self) -> None:
        """等待惰性构造完成，并原样传播初始化错误。"""
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot synchronously wait for the VLM runtime from its own thread")
        self._ready.result()

    def _submit(self, operation: Callable[[], Awaitable[T]]) -> _Submission:
        """在线程安全边界接收请求，关闭开始后拒绝新增请求。"""
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot submit a blocking bridge from the VLM runtime thread")
        submission = _Submission()

        async def execute() -> T:
            """在所属循环执行调用级操作，不占用等待线程。"""
            return await operation()

        def finish(task: asyncio.Task[T]) -> None:
            """底层任务退出后才发布结果和清理完成信号。"""
            try:
                submission.result.set_result(task.result())
            except BaseException as exc:
                submission.result.set_exception(exc)
            finally:
                self._submissions.discard(submission)
                submission.finished.set_result(None)

        def start() -> None:
            """登记任务及完成回调，覆盖首次执行前就被取消的情况。"""
            self._submissions.add(submission)
            submission.task = self._loop.create_task(execute())
            submission.task.add_done_callback(finish)
            if submission.cancel_requested:
                submission.task.cancel()

        with self._state_lock:
            if self._closing:
                raise RuntimeError("VLM runtime is shutting down")
            self._loop.call_soon_threadsafe(start)
        return submission

    def _cancel_submission(self, submission: _Submission) -> None:
        """请求已经清理完成时不再访问可能已关闭的所属循环。"""
        if submission.finished.done():
            return
        try:
            self._loop.call_soon_threadsafe(submission.cancel)
        except RuntimeError:
            if not submission.finished.done():
                raise

    def _call(self, operation: Callable[[], Awaitable[T]]) -> T:
        """同步等待同一异步引擎；中断后等待实际请求清理。"""
        submission = self._submit(operation)
        try:
            return submission.result.result()
        except BaseException:
            self._cancel_submission(submission)
            submission.finished.result()
            raise

    async def _acall(self, operation: Callable[[], Awaitable[T]]) -> T:
        """桥接调用方循环，取消后等待引擎清理完成而非仅取消 Future。"""
        submission = self._submit(operation)
        result = asyncio.wrap_future(submission.result)
        try:
            return await asyncio.shield(result)
        except asyncio.CancelledError:
            self._cancel_submission(submission)
            await drain_future(asyncio.wrap_future(submission.finished))
            try:
                await drain_future(result)
            except BaseException:
                pass
            raise

    async def _extract_with_layout(
        self,
        images: list[Image],
        blocks_list: Sequence[Sequence[ContentBlock | dict]],
        not_extract_list: list[str] | None,
        image_analysis: bool | None,
    ) -> list[ExtractResult]:
        """所有文档共用同一请求信号量执行外部布局抽取。"""
        return await self._predictor.aio_batch_extract_with_layout(
            images,
            blocks_list,
            semaphore=self._semaphore,
            not_extract_list=not_extract_list,
            image_analysis=image_analysis,
        )

    def batch_extract_with_layout(
        self,
        images: list[Image],
        blocks_list: Sequence[Sequence[ContentBlock | dict]],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """同步入口通过常驻循环复用原生异步外部布局抽取。"""
        return self._call(lambda: self._extract_with_layout(images, blocks_list, not_extract_list, image_analysis))

    async def aio_batch_extract_with_layout(
        self,
        images: list[Image],
        blocks_list: Sequence[Sequence[ContentBlock | dict]],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """异步入口直接等待原生异步外部布局抽取。"""
        return await self._acall(lambda: self._extract_with_layout(images, blocks_list, not_extract_list, image_analysis))

    async def _two_step(
        self,
        images: list[Image],
        not_extract_list: list[str] | None,
        image_analysis: bool | None,
    ) -> list[ExtractResult]:
        """版面与内容推理共用跨文档并发额度。"""
        return await self._predictor.aio_batch_two_step_extract(
            images,
            semaphore=self._semaphore,
            not_extract_list=not_extract_list,
            image_analysis=image_analysis,
        )

    def batch_two_step_extract(
        self,
        images: list[Image],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """同步等待共享异步引擎完成两阶段抽取。"""
        return self._call(lambda: self._two_step(images, not_extract_list, image_analysis))

    async def aio_batch_two_step_extract(
        self,
        images: list[Image],
        *,
        not_extract_list: list[str] | None = None,
        image_analysis: bool | None = None,
    ) -> list[ExtractResult]:
        """异步等待共享引擎完成两阶段抽取。"""
        return await self._acall(lambda: self._two_step(images, not_extract_list, image_analysis))

    async def _close(self) -> None:
        """停止在途推理并等待清理，然后关闭自有客户端和引擎。"""
        submissions = list(self._submissions)
        for submission in submissions:
            submission.cancel()
        await asyncio.gather(*(asyncio.wrap_future(item.finished) for item in submissions))
        try:
            await self._predictor.aclose()
        finally:
            self._dispose(self._predictor)

    def shutdown(self) -> None:
        """幂等关闭运行时，等待线程退出后允许重新初始化。"""
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot shut down the VLM runtime from its own thread")
        with self._shutdown_lock:
            if not self._thread.is_alive():
                return
            try:
                self.wait_ready()
            except BaseException:
                self._thread.join()
                return
            with self._state_lock:
                self._closing = True
                work = asyncio.run_coroutine_threadsafe(self._close(), self._loop)
            try:
                work.result()
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join()

    async def aclose(self) -> None:
        """异步关闭时同样等待关闭线程完成，保证重复取消不会遗留运行时。"""
        await run_sync(self.shutdown)


__all__ = ["AsyncVlmPredictor", "RuntimeOwner", "runtime_owner"]
