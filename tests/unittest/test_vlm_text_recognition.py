"""验证行内 VLM 文本识别的错误隔离、共享并发和取消屏障。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from mineru_vl_utils.vlm_client.base_client import ClientClosedError, RequestError

from mineru.model.vlm.async_runtime import AsyncVlmPredictor
from mineru.model.vlm.errors import EngineDeadError
from mineru.model.vlm.handles import RuntimeHandle
from mineru.model.vlm.text_recognition import recognize_text_batch


def test_local_batch_keeps_successes_and_only_retries_after_batch_failure() -> None:
    """本地正常合批不拆单；批次失败后只逐片补试一次并保留异常落点。"""
    failure = RequestError("bad crop")
    client = SimpleNamespace(batch_content_extract=MagicMock(return_value=["α", None]), content_extract=MagicMock())
    assert recognize_text_batch(client, [1, 2]) == ["α", None]
    client.content_extract.assert_not_called()
    client.batch_content_extract.assert_called_once_with([1, 2], types="text")
    client.batch_content_extract.side_effect = RequestError("batch failed")
    client.content_extract.side_effect = ["success", failure, None]
    assert recognize_text_batch(client, [1, 2, 3]) == ["success", failure, None]
    assert client.content_extract.call_count == 3
    assert [call.kwargs for call in client.content_extract.call_args_list] == [{"type": "text"}] * 3


@pytest.mark.parametrize(
    "error", [EngineDeadError("dead"), ClientClosedError("closed"), RuntimeError("engine stopped"), TypeError("bug")]
)
def test_local_fatal_errors_never_retry(error: Exception) -> None:
    """引擎失效、关闭及程序错误不得重试后伪装成 MFR 回退。"""
    client = SimpleNamespace(batch_content_extract=MagicMock(side_effect=error), content_extract=MagicMock())
    with pytest.raises(type(error)):
        recognize_text_batch(client, [1])
    client.content_extract.assert_not_called()


@pytest.mark.parametrize("outputs", [[], ["x", "y"], [object()]])
def test_local_output_contract_is_strict(outputs: list[Any]) -> None:
    """长度和类型错配立即报错，不能触发掩盖适配器缺陷的逐片补试。"""
    client = SimpleNamespace(batch_content_extract=MagicMock(return_value=outputs), content_extract=MagicMock())
    with pytest.raises((ValueError, TypeError)):
        recognize_text_batch(client, [1])
    client.content_extract.assert_not_called()


class _TextClient:
    """异步文本替身在真实运行时循环里使用共享信号量并延迟清理。"""

    def __init__(self) -> None:
        """记录请求和清理事件，便于验证多个文档及不同入口共享运行时。"""
        self.loop = asyncio.get_running_loop()
        self.active = 0
        self.peak = 0
        self.block = False
        self.entered = threading.Event()
        self.closed = False

    async def aio_content_extract(self, image: int, *, type: str, semaphore: asyncio.Semaphore) -> str:
        """指定切片失败仍释放请求额度，取消时模拟异步资源清理。"""
        assert asyncio.get_running_loop() is self.loop
        assert type == "text"
        async with semaphore:
            self.active += 1
            self.peak = max(self.peak, self.active)
            self.entered.set()
            try:
                await asyncio.sleep(60 if self.block else 0.01)
                if image == -1:
                    raise RequestError("bad crop")
                if image == -2:
                    raise EngineDeadError("dead")
                return str(image)
            finally:
                await asyncio.sleep(0.03)
                self.active -= 1

    async def aclose(self) -> None:
        """只有所有切片完成清理之后才允许关闭客户端。"""
        assert self.active == 0
        self.closed = True


@pytest.fixture
def text_runtime() -> Any:
    """创建原生异步宿主并保证测试结束后关闭线程及客户端。"""
    runtime = AsyncVlmPredictor(
        lambda: RuntimeHandle(_TextClient(), backend="http-client"), backend="http-client", max_concurrency=2
    )
    runtime.wait_ready()
    try:
        yield runtime
    finally:
        runtime.shutdown()


def test_native_text_results_keep_order_errors_and_global_limit(text_runtime: AsyncVlmPredictor) -> None:
    """跨线程同步及异步入口共享额度，单片失败不会取消其他切片。"""

    def invoke(index: int) -> list[Any]:
        """交错同步和异步调用，模拟多文档同时提交。"""
        if index % 2:
            return asyncio.run(text_runtime.aio_batch_recognize_text([index, -1]))
        return text_runtime.batch_recognize_text([index, -1])

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(invoke, range(4)))
    assert [result[0] for result in results] == ["0", "1", "2", "3"]
    assert all(isinstance(result[1], RequestError) for result in results)
    assert text_runtime._predictor.peak == 2
    assert text_runtime._predictor.active == 0


def test_native_repeated_cancellation_drains_all_crops(text_runtime: AsyncVlmPredictor) -> None:
    """重复取消不得使调用方在任何切片仍读取图片时退出。"""
    client = text_runtime._predictor
    client.block = True

    async def run() -> None:
        """发出两次取消并等待跨循环的清理确认。"""
        task = asyncio.create_task(text_runtime.aio_batch_recognize_text([1, 2, 3]))
        assert await asyncio.to_thread(client.entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.005)
        task.cancel()
        assert not task.done()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.active == 0

    asyncio.run(run())
    client.block = False
    assert text_runtime.batch_recognize_text([4]) == ["4"]


def test_native_engine_failure_drains_other_requests(text_runtime: AsyncVlmPredictor) -> None:
    """引擎死亡向上传播，同时等待同批其他请求清理。"""
    with pytest.raises(EngineDeadError):
        text_runtime.batch_recognize_text([-2, 1, 2])
    assert text_runtime._predictor.active == 0


def test_local_shutdown_waits_for_text_request() -> None:
    """本地切片识别在途时关闭必须等待，关闭后拒绝新请求。"""
    entered, release, closed = threading.Event(), threading.Event(), threading.Event()

    def infer(images: list[Any], *, types: str) -> list[str]:
        """模拟持有图片的同步推理，显式释放前不得关闭引擎。"""
        entered.set()
        assert release.wait(2)
        assert not closed.is_set()
        return ["x"]

    handle = RuntimeHandle(SimpleNamespace(batch_content_extract=infer), backend="test", close_engine=closed.set)
    with ThreadPoolExecutor(max_workers=2) as pool:
        work = pool.submit(handle.batch_recognize_text, [1])
        assert entered.wait(2)
        shutting_down = pool.submit(handle.shutdown)
        assert not closed.wait(0.02)
        release.set()
        assert work.result() == ["x"]
        shutting_down.result()
    assert closed.is_set()
    with pytest.raises(RuntimeError, match="shutting down"):
        handle.batch_recognize_text([1])
