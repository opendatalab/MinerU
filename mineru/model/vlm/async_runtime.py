# Copyright (c) Opendatalab. All rights reserved.
"""在固定事件循环中复用 VLM 客户端，并跨线程桥接同步、异步请求。"""

from __future__ import annotations

import asyncio
import concurrent.futures
import threading
import time
from collections.abc import Awaitable, Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from ...utils.async_utils import drain_future, run_sync
from ..runtime import ownership
from .errors import EngineDeadError, is_fatal_engine_error
from .handles import RuntimeHandle
from .lifecycle import RuntimeLifecycle
from .text_recognition import TextRecognitionResult, raise_unrecoverable_text_error, validate_text_result

if TYPE_CHECKING:
    from mineru_vl_utils.structs import ContentBlock, ExtractResult
    from PIL.Image import Image

T = TypeVar("T")


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
        factory: Callable[[], RuntimeHandle],
        *,
        backend: str,
        max_concurrency: int,
    ) -> None:
        """启动专属线程；模型构造在该线程的运行中事件循环内执行。"""
        if max_concurrency < 1:
            raise ValueError("VLM max_concurrency must be positive")
        self.backend = backend
        self.max_concurrency = max_concurrency
        self.lifecycle = RuntimeLifecycle(backend)
        self._factory = factory
        self._ready: concurrent.futures.Future[None] = concurrent.futures.Future()
        self._state_lock = threading.Lock()
        self._probe_lock = threading.Lock()
        self._health_submission: _Submission | None = None
        self._shutdown_lock = threading.Lock()
        self._submissions: set[_Submission] = set()
        self._thread = threading.Thread(target=self._serve, name=f"mineru-{backend}", daemon=True)
        self._thread.start()

    async def _initialize(self) -> None:
        """在固定循环创建引擎和共享请求信号量。"""
        self._runtime_handle = self._factory()
        self.lifecycle = self._runtime_handle.lifecycle
        try:
            self._predictor = self._runtime_handle.predictor
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
        except BaseException:
            await self._runtime_handle.aclose()
            raise

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
            if self._ready.done() and self._ready.exception() is None and not self.lifecycle.closing:
                self.lifecycle.mark_fatal(EngineDeadError("VLM runtime event loop stopped unexpectedly"))
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
        return self.lifecycle.closing or (self._ready.done() and self._ready.exception() is not None)

    @property
    def owners(self) -> set[ownership.RuntimeOwner]:
        """代理与内部句柄共用宿主集合，避免失效通知遗漏一层。"""
        return self.lifecycle.owners

    @property
    def pinned(self) -> bool:
        """从共享状态读取独立调用方的固定租约。"""
        return self.lifecycle.pinned

    @property
    def supports_health_probe(self) -> bool:
        """仅已成功创建并声明健康协议的运行时参与探测。"""
        return (
            self._ready.done()
            and self._ready.exception() is None
            and self._runtime_handle is not None
            and self._runtime_handle.supports_health_probe
        )

    def require_ready(self, *, allow_unavailable: bool = False) -> None:
        """统一缓存和提交边界的判断，异常退出的循环视为明确死亡。"""
        self.lifecycle.require_ready(allow_unavailable=allow_unavailable)
        if not self._thread.is_alive():
            self.lifecycle.mark_fatal(EngineDeadError("VLM runtime event loop is closed"))
            self.lifecycle.raise_if_dead()

    def probe_health(self, timeout: float = 5.0) -> None:
        """由调用方线程限定等待期限，循环阻塞时保留唯一的在途探测。"""
        from mineru_vl_utils.vlm_client.base_client import EngineHealth

        if not self.supports_health_probe:
            return
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot synchronously probe from the VLM runtime thread")
        if timeout <= 0:
            raise ValueError("Health probe timeout must be positive")
        self.lifecycle.raise_if_dead()
        started = time.monotonic()
        if not self._probe_lock.acquire(timeout=timeout):
            self.lifecycle.apply_health(EngineHealth("unavailable", "Another engine health probe is still pending"))
            return
        try:
            if self.lifecycle.closing:
                return
            self.require_ready(allow_unavailable=True)
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                self.lifecycle.apply_health(EngineHealth("unavailable", "Engine health probe deadline elapsed"))
                return
            if self._health_submission is None:
                self._health_submission = self._submit(
                    lambda: self._runtime_handle.aio_probe_health(remaining), health_probe=True
                )
            submission = self._health_submission
            try:
                submission.result.result(timeout=remaining)
            except concurrent.futures.TimeoutError:
                self.lifecycle.apply_health(EngineHealth("unavailable", f"VLM health probe timed out after {timeout:g}s"))
            finally:
                if submission.finished.done():
                    self._health_submission = None
        finally:
            self._probe_lock.release()

    def wait_ready(self) -> None:
        """等待惰性构造完成，并原样传播初始化错误。"""
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot synchronously wait for the VLM runtime from its own thread")
        self._ready.result()

    def _submit(self, operation: Callable[[], Awaitable[T]], *, health_probe: bool = False) -> _Submission:
        """在线程安全边界接收请求，关闭开始后拒绝新增请求。"""
        if threading.current_thread() is self._thread:
            raise RuntimeError("Cannot submit a blocking bridge from the VLM runtime thread")
        submission = _Submission()

        async def execute() -> T:
            """在所属循环执行调用级操作，不占用等待线程。"""
            try:
                self.lifecycle.require_ready(allow_unavailable=health_probe)
                result = await operation()
                self.lifecycle.raise_if_dead()
                return result
            except Exception as exc:
                if self.backend != "http-client" and is_fatal_engine_error(exc):
                    self.lifecycle.mark_fatal(exc)
                raise

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
            self.require_ready(allow_unavailable=health_probe)
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
        self.probe_health()
        submission = self._submit(operation)
        try:
            return submission.result.result()
        except BaseException:
            self._cancel_submission(submission)
            submission.finished.result()
            raise

    async def _acall(self, operation: Callable[[], Awaitable[T]]) -> T:
        """桥接调用方循环，取消后等待引擎清理完成而非仅取消 Future。"""
        if self.supports_health_probe:
            await run_sync(self.probe_health)
        submission = self._submit(operation)
        result = asyncio.wrap_future(submission.result)
        try:
            await asyncio.wait([result])
            return result.result()
        except asyncio.CancelledError:
            self._cancel_submission(submission)
            await drain_future(asyncio.wrap_future(submission.finished))
            try:
                await drain_future(result)
            except BaseException:
                pass
            raise

    async def _recognize_text(self, images: list[Image]) -> list[TextRecognitionResult]:
        """逐请求隔离识别异常，跨文档共享并发额度，取消后排空所有切片。"""

        async def recognize(image: Image) -> TextRecognitionResult:
            """单片失败不会取消同批成功请求，程序错误和引擎故障仍向上传播。"""
            try:
                output = await self._predictor.aio_content_extract(image, type="text", semaphore=self._semaphore)
            except Exception as error:
                raise_unrecoverable_text_error(error)
                return error
            return validate_text_result(output)

        tasks = [asyncio.create_task(recognize(image)) for image in images]
        pending = set(tasks)
        try:
            # wait 不会把父任务取消隐式转发给子任务，避免二次 cancel 打断清理。
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
            return [task.result() for task in tasks]
        except BaseException as primary:
            for task in pending:
                task.cancel()
            results = await drain_future(asyncio.gather(*tasks, return_exceptions=True))
            if isinstance(primary, Exception) and not is_fatal_engine_error(primary):
                for result in results:
                    if isinstance(result, Exception) and is_fatal_engine_error(result):
                        raise result
            raise

    def batch_recognize_text(self, images: list[Image]) -> list[TextRecognitionResult]:
        """同步调用方通过常驻循环复用原生异步切片识别。"""
        return self._call(lambda: self._recognize_text(images))

    async def aio_batch_recognize_text(self, images: list[Image]) -> list[TextRecognitionResult]:
        """异步切片识别沿用运行时取消确认和关闭屏障。"""
        return await self._acall(lambda: self._recognize_text(images))

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
            await self._runtime_handle.aclose()
        finally:
            self._predictor = None
            self._runtime_handle = None

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
                self.lifecycle.begin_close()
                work = asyncio.run_coroutine_threadsafe(self._close(), self._loop)
            try:
                work.result()
            finally:
                self._loop.call_soon_threadsafe(self._loop.stop)
                self._thread.join()

    async def aclose(self) -> None:
        """异步关闭时同样等待关闭线程完成，保证重复取消不会遗留运行时。"""
        await run_sync(self.shutdown)


__all__ = ["AsyncVlmPredictor"]
