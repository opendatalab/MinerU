"""跨同步句柄、常驻循环和缓存验证同一生命周期契约。"""

from __future__ import annotations

import asyncio
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock

import pytest
from mineru_vl_utils.vlm_client.base_client import EngineHealth, EngineUnavailableError

from mineru.model.runtime.ownership import RuntimeOwner, runtime_owner
from mineru.model.vlm.async_runtime import AsyncVlmPredictor
from mineru.model.vlm.errors import EngineDeadError, EngineNotReadyError
from mineru.model.vlm.handles import RuntimeHandle
from mineru.model.vlm.runtime import ModelSingleton


@pytest.fixture
def runtime() -> Iterator[AsyncVlmPredictor]:
    """使用真实常驻循环和可控公开探测回调，不依赖 GPU 或远程服务。"""
    state = SimpleNamespace(error=None, health=EngineHealth("healthy"), calls=0, probes=0, closed=False)

    async def infer(images: list[Any], **kwargs: Any) -> list[Any]:
        """记录实际进入推理的次数，并传播指定错误。"""
        state.calls += 1
        if state.error is not None:
            raise state.error
        return images

    async def probe(timeout: float) -> EngineHealth:
        """确认健康回调运行在同一个专属事件循环。"""
        assert threading.current_thread().name == "mineru-vllm-async-engine"
        state.probes += 1
        return state.health

    async def close() -> None:
        """记录客户端已在所属循环清理。"""
        state.closed = True

    client = SimpleNamespace(aio_batch_two_step_extract=infer, aclose=close, test_state=state)
    instance = AsyncVlmPredictor(
        lambda: RuntimeHandle(client, backend="vllm-async-engine", aio_probe_engine=probe),
        backend="vllm-async-engine",
        max_concurrency=2,
    )
    instance.wait_ready()
    try:
        yield instance
    finally:
        instance.shutdown()
    assert state.closed


def test_async_death_uses_shared_state_and_rejects_cache_borrow(
    runtime: AsyncVlmPredictor, monkeypatch: pytest.MonkeyPatch
) -> None:
    """死亡通知全部宿主，迟到宿主不能从缓存借用旧实例或触发重建。"""
    state = runtime._predictor.test_state
    assert runtime.lifecycle is runtime._runtime_handle.lifecycle
    owners = [RuntimeOwner(), RuntimeOwner()]
    for owner in owners:
        runtime.lifecycle.retain(owner)
    original = EngineUnavailableError("original backend failure")
    state.error = original
    with pytest.raises(EngineUnavailableError) as raised:
        runtime.batch_two_step_extract([1])
    assert raised.value is original
    assert all(owner.engine_error == str(original) for owner in owners)
    state.error = None
    with pytest.raises(EngineDeadError) as raised:
        asyncio.run(runtime.aio_batch_two_step_extract([2]))
    assert raised.value.__cause__ is original
    assert state.calls == 1
    key = ("dead-async",)
    monkeypatch.setattr(ModelSingleton, "_models", {key: runtime})
    newcomer = RuntimeOwner()
    token = runtime_owner.set(newcomer)
    factory = MagicMock()
    try:
        with pytest.raises(EngineDeadError):
            ModelSingleton()._get_or_initialize(key, factory, native_async=True, backend=runtime.backend, max_concurrency=2)
    finally:
        runtime_owner.reset(token)
    factory.assert_not_called()
    assert newcomer not in runtime.owners
    assert newcomer.engine_error == str(original)


def test_blocked_engine_loop_times_out_without_duplicate_probes(runtime: AsyncVlmPredictor) -> None:
    """循环阻塞时超时由外部计时，多次检查不排队创建额外探测。"""
    owner = RuntimeOwner()
    runtime.lifecycle.retain(owner)
    entered, release = threading.Event(), threading.Event()

    def block() -> None:
        """有界阻塞引擎循环，不阻塞测试线程或 API 所属循环。"""
        entered.set()
        assert release.wait(3)

    runtime._loop.call_soon_threadsafe(block)
    assert entered.wait(1)
    started = time.monotonic()
    try:
        for _ in range(3):
            with pytest.raises(EngineNotReadyError):
                runtime.probe_health(0.02)
        assert time.monotonic() - started < 0.5
        assert owner.engine_error is None
        assert owner.engine_unavailable
        assert runtime._predictor.test_state.probes == 0
        with pytest.raises(EngineNotReadyError):
            runtime._submit(lambda: runtime._two_step([1], None, None))
    finally:
        release.set()
    runtime.probe_health(1)
    assert runtime._predictor.test_state.probes == 1
    assert owner.engine_unavailable is None
    assert runtime.batch_two_step_extract([2]) == [2]


def test_late_health_cannot_resurrect_fatal_runtime(runtime: AsyncVlmPredictor) -> None:
    """已在途的健康检查完成时，推理线程记录的死亡状态仍有优先级。"""
    entered, release = threading.Event(), threading.Event()

    async def probe(timeout: float) -> EngineHealth:
        """模拟探测在真正读取结果之前暂停。"""
        entered.set()
        await asyncio.to_thread(release.wait, 2)
        return EngineHealth("healthy")

    runtime._runtime_handle._aio_probe_engine = probe
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(runtime.probe_health, 1)
        assert entered.wait(1)
        runtime.lifecycle.mark_fatal(EngineUnavailableError("dead while probing"))
        release.set()
        with pytest.raises(EngineDeadError):
            future.result(2)
    with pytest.raises(EngineDeadError):
        runtime.require_ready()


def test_close_drains_scheduled_probe_before_engine_release(runtime: AsyncVlmPredictor) -> None:
    """关闭与尚未执行的健康检查交错，也必须退出而不留下循环任务。"""
    entered, release = threading.Event(), threading.Event()

    def block() -> None:
        """在探测提交前暂停引擎循环，制造开始与关闭的竞态。"""
        entered.set()
        release.wait(2)

    runtime._loop.call_soon_threadsafe(block)
    assert entered.wait(1)
    with pytest.raises(EngineNotReadyError):
        runtime.probe_health(0.01)
    with ThreadPoolExecutor(1) as pool:
        closing = pool.submit(runtime.shutdown)
        release.set()
        closing.result(2)
    assert not runtime._thread.is_alive()


def test_mlx_unknown_runtime_failure_does_not_retry_or_poison_instance() -> None:
    """未知 MLX 故障只使当前请求失败，不做单片补试或永久隔离。"""
    error = RuntimeError("unknown Metal execution failure")
    client = SimpleNamespace(batch_content_extract=MagicMock(side_effect=error), content_extract=MagicMock())
    handle = RuntimeHandle(client, backend="mlx-engine")
    owner = RuntimeOwner()
    handle.lifecycle.retain(owner)
    try:
        with pytest.raises(RuntimeError) as raised:
            handle.batch_recognize_text([1])
        assert raised.value is error
        client.content_extract.assert_not_called()
        assert owner.engine_error is None
        client.batch_content_extract.side_effect = None
        client.batch_content_extract.return_value = ["recognized"]
        assert handle.batch_recognize_text([2]) == ["recognized"]
    finally:
        handle.shutdown()


def test_failed_async_initialization_is_closed_without_a_live_instance() -> None:
    """构造失败仍被标记为关闭，不与已成功创建后死亡且需锁存的实例混淆。"""

    def fail() -> RuntimeHandle:
        """模拟模型尚未创建成功时的输入错误。"""
        raise ValueError("initialization failed")

    runtime = AsyncVlmPredictor(fail, backend="vllm-async-engine", max_concurrency=1)
    try:
        with pytest.raises(ValueError, match="initialization failed"):
            runtime.wait_ready()
        assert runtime.is_closed
    finally:
        runtime.shutdown()
    assert not runtime._thread.is_alive()
