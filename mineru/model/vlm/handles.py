# Copyright (c) Opendatalab. All rights reserved.
"""宿主持有的 VLM 客户端与确定的后端关闭动作。"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator, Sequence
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING

from ...utils.async_utils import drain_future, run_sync
from ..runtime.closing import runtime_close_scope
from ..runtime.ownership import RuntimeOwner
from .errors import is_fatal_engine_error
from .lifecycle import RuntimeLifecycle
from .text_recognition import TextRecognitionResult, recognize_text_batch

if TYPE_CHECKING:
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.structs import ContentBlock, ExtractResult
    from mineru_vl_utils.vlm_client.base_client import EngineHealth
    from PIL.Image import Image

logger = logging.getLogger(__name__)


class RuntimeHandle:
    """将客户端、串行执行锁和创建时确定的资源释放动作放在同一宿主对象中。"""

    def __init__(
        self,
        predictor: MinerUClient,
        *,
        backend: str,
        close_engine: Callable[[], None] | None = None,
        probe_engine: Callable[[float], EngineHealth] | None = None,
        aio_probe_engine: Callable[[float], Awaitable[EngineHealth]] | None = None,
        close_health_probe: Callable[[], None] | None = None,
    ) -> None:
        """保存已创建资源，不修改第三方客户端的属性或推测其内部结构。"""
        self.backend = backend
        self.lifecycle = RuntimeLifecycle(backend)
        self.execution_lock = threading.Lock() if backend == "mlx-engine" else None
        self._predictor: MinerUClient | None = predictor
        self._close_engine = close_engine
        self._probe_engine = probe_engine
        self._aio_probe_engine = aio_probe_engine
        self._close_health_probe = close_health_probe
        self._probe_lock = threading.Lock()
        self._condition = self.lifecycle.condition
        self._active = 0
        self._close_started = False
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
        return self.lifecycle.closing

    @property
    def supports_health_probe(self) -> bool:
        """缓存管理者据此开启只读探测，不为未加载模型触发初始化。"""
        return self._probe_engine is not None or self._aio_probe_engine is not None

    @property
    def owners(self) -> set[RuntimeOwner]:
        """向缓存暴露唯一状态中的宿主集合。"""
        return self.lifecycle.owners

    @property
    def pinned(self) -> bool:
        """独立 SDK 或 CLI 固定租约同样属于共享状态。"""
        return self.lifecycle.pinned

    def require_ready(self, *, allow_unavailable: bool = False) -> None:
        """同步和异步运行时共用同一就绪规则。"""
        self.lifecycle.require_ready(allow_unavailable=allow_unavailable)

    async def aio_probe_health(self, timeout: float = 5.0) -> None:
        """由异步代理在所属循环调用，期限由外层跨线程等待同时保障。"""
        if self._aio_probe_engine is None:
            return
        try:
            health = await self._aio_probe_engine(timeout)
        except Exception as error:
            if is_fatal_engine_error(error):
                self.lifecycle.mark_fatal(error)
                raise
            from mineru_vl_utils.vlm_client.base_client import EngineHealth

            health = EngineHealth("unavailable", f"VLM health probe failed: {type(error).__name__}: {error}")
        self.lifecycle.apply_health(health)

    def probe_health(self, timeout: float = 5.0) -> None:
        """执行至多一个宿主探测；超时状态可恢复，明确死亡永不被健康结果覆盖。"""
        from mineru_vl_utils.vlm_client.base_client import EngineHealth

        if self._probe_engine is None:
            return
        if timeout <= 0:
            raise ValueError("Health probe timeout must be positive")
        self.lifecycle.raise_if_dead()
        started = time.monotonic()
        if not self._probe_lock.acquire(timeout=timeout):
            self.lifecycle.apply_health(EngineHealth("unavailable", "Another engine health probe is still pending"))
            return
        active = False
        try:
            with self._condition:
                if self.lifecycle.closing:
                    return
                self.lifecycle.raise_if_dead()
                self._active += 1
                active = True
            try:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError("Engine health probe deadline elapsed while waiting")
                health = self._probe_engine(remaining)
                if health.state not in {"healthy", "unavailable", "dead"}:
                    raise ValueError("Invalid engine health state")
            except Exception as exc:
                if is_fatal_engine_error(exc):
                    self.lifecycle.mark_fatal(exc)
                    raise
                logger.exception("VLM health probe failed: backend=%s", self.backend)
                message = f"VLM health probe failed: {type(exc).__name__}: {exc}"
            else:
                self.lifecycle.apply_health(health)
                return
            self.lifecycle.apply_health(EngineHealth("unavailable", message))
        finally:
            if active:
                with self._condition:
                    self._active -= 1
                    self._condition.notify_all()
            self._probe_lock.release()

    @contextmanager
    def _use(self, *, acquire_lock: bool = True, probe: bool = True) -> Iterator[MinerUClient]:
        """等待同步推理自然返回后才允许关闭引擎，MLX 使用宿主串行锁。"""
        if probe:
            self.probe_health()
        with self._condition:
            self.require_ready()
            self._active += 1
            predictor = self.predictor
        try:
            if self.execution_lock is None or not acquire_lock:
                yield predictor
            else:
                with self.execution_lock:
                    yield predictor
            with self._condition:
                self.lifecycle.raise_if_dead()
        except Exception as exc:
            if is_fatal_engine_error(exc):
                self.lifecycle.mark_fatal(exc)
            raise
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    @asynccontextmanager
    async def _async_use(self) -> AsyncIterator[MinerUClient]:
        """可取消地等待串行锁，排队期间取消不得让迟到工作线程开始推理。"""
        if self._probe_engine is not None:
            await run_sync(self.probe_health)
        lock = self.execution_lock
        if lock is not None:
            while True:
                with self._condition:
                    if self.lifecycle.closing:
                        raise RuntimeError("VLM runtime is shutting down")
                    if lock.acquire(blocking=False):
                        break
                await asyncio.sleep(0.01)
        try:
            with self._use(acquire_lock=False, probe=False) as predictor:
                yield predictor
        finally:
            if lock is not None:
                lock.release()

    def batch_recognize_text(self, images: list[Image]) -> list[TextRecognitionResult]:
        """切片合批及必要的单片补试共享同一次运行时租约与执行锁。"""
        with self._use() as predictor:
            return recognize_text_batch(predictor, images)

    async def aio_batch_recognize_text(self, images: list[Image]) -> list[TextRecognitionResult]:
        """同步引擎在线程内识别切片，取消时等待推理退出再归还图片。"""
        async with self._async_use() as predictor:
            return await run_sync(recognize_text_batch, predictor, images)

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
            if self._close_started:
                return False
            self._close_started = True
            self.lifecycle.begin_close()
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
            self.lifecycle.finish_close()
            self._predictor = None
            self._probe_engine = None
            self._aio_probe_engine = None
            self._close_health_probe = None
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
                if self._close_health_probe is not None:
                    self._close_health_probe()
                elif self.backend in {"http-client", "lmdeploy-engine"}:
                    asyncio.run(self.predictor.aclose())
        except BaseException as exc:
            error = exc
        self._finish_close(error)

    async def _close_client(self) -> None:
        """在受保护的任务中排空推理并关闭客户端，等待异常也不跳过连接清理。"""
        error: BaseException | None = None
        with runtime_close_scope():
            try:
                # 关闭已拒绝新调用；空闲时无需线程池，避免 atexit 在线程池退出后继续提交任务。
                with self._condition:
                    has_active_calls = self._active > 0
                if has_active_calls:
                    await run_sync(self._wait_for_calls)
            except BaseException as exc:
                error = exc
            try:
                if self._close_health_probe is not None:
                    # 与同步关闭使用相同的排空回调，避免客户端局部超时后仍释放 Pipeline。
                    await run_sync(self._close_health_probe)
                else:
                    await self.predictor.aclose()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error

    async def aclose(self) -> None:
        """取消时仍排空客户端关闭，再释放引擎并传播最早的取消或清理异常。"""
        if not self._begin_close():
            if not self._closed.is_set():
                await run_sync(self._closed.wait)
            return
        work = asyncio.create_task(self._close_client())
        error: BaseException | None = None
        try:
            await asyncio.wait([work])
            work.result()
        except BaseException as exc:
            error = exc
            try:
                await drain_future(work)
            except BaseException:
                pass
        self._finish_close(error)


__all__ = ["RuntimeHandle"]
