"""同步阶段取消后正确收尾，不重复报告已经处理的引擎异常。"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import pytest

from mineru.model.vlm.errors import EngineDeadError
from mineru.utils.async_utils import drain_future, run_sync


def test_canceled_sync_failure_is_drained_without_unhandled_exception() -> None:
    """重放真机停机竞态：重复取消后等待线程退出，保留取消而不产生孤立异常日志。"""
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    unhandled: list[dict[str, Any]] = []

    def fail_engine() -> None:
        """模拟在取消之后才观察到引擎死亡的同步推理。"""
        started.set()
        try:
            if not release.wait(5):
                raise TimeoutError("Test did not release the inference thread")
            raise EngineDeadError("engine stopped during inference")
        finally:
            finished.set()

    def record_error(loop: asyncio.AbstractEventLoop, context: dict[str, Any]) -> None:
        """收集事件循环未处理错误，以捕获 Python 3.14 的 shield 回调日志。"""
        unhandled.append(context)

    async def scenario() -> None:
        """只取消等待者，不在工作线程实际完成前结束资源租约。"""
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(record_error)
        task = asyncio.create_task(run_sync(fail_engine))
        try:
            async with asyncio.timeout(5):
                while not started.is_set():
                    await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            assert not finished.is_set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert finished.is_set()
            await asyncio.sleep(0)
            assert not unhandled
        finally:
            release.set()

    asyncio.run(scenario())


def test_sync_and_drain_preserve_uncanceled_failure() -> None:
    """未取消的同步调用和清理等待仍将原始异常对象交给调用方。"""
    error = EngineDeadError("original failure")

    def fail() -> None:
        """抛出固定异常，检查等待方式没有改写错误语义。"""
        raise error

    async def scenario() -> None:
        """分别检查正常等待和显式清理的错误传播。"""
        with pytest.raises(EngineDeadError) as raised:
            await run_sync(fail)
        assert raised.value is error
        work = asyncio.create_task(asyncio.to_thread(fail))
        with pytest.raises(EngineDeadError) as drained:
            await drain_future(work)
        assert drained.value is error

    asyncio.run(scenario())
