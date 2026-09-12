"""验证常驻推理运行时的共享、取消确认和应用租约。"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from mineru.model.vlm.async_runtime import AsyncVlmPredictor, RuntimeOwner, runtime_owner
from mineru.utils.async_utils import run_sync


class _AsyncClient:
    """只实现异步推理的引擎替身，记录真实执行循环与并发峰值。"""

    def __init__(self) -> None:
        """在运行时线程内创建同步观测信号和异步阻塞门。"""
        self.loop = asyncio.get_running_loop()
        self.active = 0
        self.peak = 0
        self.closed = False
        self.block = False
        self.entered = threading.Event()
        self.drained = threading.Event()

    async def aio_batch_extract_with_layout(self, images: list[Any], blocks_list: list[Any], **kwargs: Any) -> list[Any]:
        """逐请求使用传入的共享额度，模拟异步引擎执行与取消后的清理。"""
        assert asyncio.get_running_loop() is self.loop

        async def infer(image: Any) -> Any:
            """模拟一次独立引擎请求，清理期间必须保留并发名额。"""
            async with kwargs["semaphore"]:
                self.active += 1
                self.peak = max(self.peak, self.active)
                self.entered.set()
                try:
                    await asyncio.sleep(60 if self.block else 0.01)
                    return image
                finally:
                    await asyncio.sleep(0.03)
                    self.active -= 1
                    self.drained.set()

        from mineru_vl_utils.vlm_client.utils import gather_tasks

        return await gather_tasks([infer(image) for image in images])

    async def aio_batch_two_step_extract(self, images: list[Any], **kwargs: Any) -> list[Any]:
        """两阶段入口复用同一引擎和请求额度。"""
        return await self.aio_batch_extract_with_layout(images, [], **kwargs)

    def batch_extract_with_layout(self, *args: Any, **kwargs: Any) -> None:
        """一旦误用同步推理立即失败。"""
        raise AssertionError("Synchronous inference must not run")

    async def aclose(self) -> None:
        """关闭只允许发生在所属循环且所有请求已经完成清理。"""
        assert asyncio.get_running_loop() is self.loop
        assert self.active == 0
        self.closed = True


@pytest.fixture
def native_runtime() -> Any:
    """创建可跨线程使用的原生异步运行时并保证用例后关闭。"""
    runtime = AsyncVlmPredictor(_AsyncClient, lambda _: None, backend="test", max_concurrency=2)
    runtime.wait_ready()
    try:
        yield runtime
    finally:
        runtime.shutdown()


def test_sync_async_and_different_loops_share_engine_and_limit(native_runtime: AsyncVlmPredictor) -> None:
    """同步线程和多个事件循环共享一份引擎及跨文档并发额度。"""

    def invoke(index: int) -> list[int]:
        """交错使用同步入口和不同线程中的异步入口。"""
        if index % 2:
            return asyncio.run(native_runtime.aio_batch_two_step_extract([index]))
        return native_runtime.batch_extract_with_layout([index], [[]])

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(invoke, range(8))) == [[i] for i in range(8)]
    assert native_runtime._predictor.peak == 2
    assert native_runtime._predictor.active == 0


def test_repeated_cancel_waits_for_engine_cleanup(native_runtime: AsyncVlmPredictor) -> None:
    """重复取消不能提前释放调用方资源，也不能二次打断底层清理。"""
    client = native_runtime._predictor
    client.block = True

    async def run() -> None:
        """取消请求后再次取消外层任务，清理完成之前不得返回。"""
        task = asyncio.create_task(native_runtime.aio_batch_two_step_extract([1]))
        assert await asyncio.to_thread(client.entered.wait, 2)
        task.cancel()
        await asyncio.sleep(0.005)
        task.cancel()
        assert not task.done()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert client.drained.is_set()
        assert client.active == 0

    asyncio.run(run())
    client.block = False
    assert native_runtime.batch_two_step_extract([2]) == [2]


def test_shutdown_drains_active_requests_and_rejects_new(native_runtime: AsyncVlmPredictor) -> None:
    """关闭时等待在途请求真正退出，然后拒绝旧代理继续提交。"""
    native_runtime._predictor.block = True
    with ThreadPoolExecutor() as executor:
        future = executor.submit(native_runtime.batch_two_step_extract, [1])
        assert native_runtime._predictor.entered.wait(2)
        native_runtime.shutdown()
        with pytest.raises(asyncio.CancelledError):
            future.result()
    assert native_runtime._predictor.closed
    assert not native_runtime._thread.is_alive()
    with pytest.raises(RuntimeError, match="shutting down"):
        native_runtime.batch_two_step_extract([2])


def test_runtime_cannot_synchronously_wait_on_itself(native_runtime: AsyncVlmPredictor) -> None:
    """明确拒绝运行时线程内的同步桥接，避免自我等待死锁。"""

    async def nested() -> None:
        """在所属循环中故意调用同步接口。"""
        native_runtime.batch_two_step_extract([1])

    with pytest.raises(RuntimeError, match="runtime thread"):
        native_runtime._call(nested)


def test_application_owners_share_and_release_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """一个应用退出不关闭其他应用的引擎，最后租约释放后允许重新构造。"""
    from mineru.model.vlm.runtime import ModelSingleton

    monkeypatch.setattr(ModelSingleton, "_models", {})
    monkeypatch.setattr(ModelSingleton, "_create_model", lambda *args, **kwargs: _AsyncClient())
    manager = ModelSingleton()
    first, second = RuntimeOwner(), RuntimeOwner()

    def get(owner: RuntimeOwner | None) -> AsyncVlmPredictor:
        """在指定应用或进程级租约中获取同一远程配置。"""
        token = runtime_owner.set(owner)
        try:
            return manager.get_model("http-client", None, "http://unused", max_concurrency=2)
        finally:
            runtime_owner.reset(token)

    try:
        predictor = get(first)
        assert get(second) is predictor
        manager.release_owner(first)
        assert predictor.batch_two_step_extract([1]) == [1]
        assert not predictor._predictor.closed
        manager.release_owner(second)
        assert predictor._predictor.closed
        replacement = get(None)
        assert replacement is not predictor
        third = RuntimeOwner()
        assert get(third) is replacement
        manager.release_owner(third)
        assert not replacement._predictor.closed
        with pytest.raises(RuntimeError, match="owner is closed"):
            get(first)
    finally:
        manager.shutdown()


def test_cancelled_initialization_is_drained_before_owner_release(monkeypatch: pytest.MonkeyPatch) -> None:
    """取消初始化时先等构造线程结束，再释放缓存和模型句柄。"""
    from mineru.model.vlm.runtime import ModelSingleton

    entered, release = threading.Event(), threading.Event()
    clients = []

    def factory(*args: Any, **kwargs: Any) -> _AsyncClient:
        """模拟无法强行中断的模型初始化。"""
        entered.set()
        assert release.wait(3)
        client = _AsyncClient()
        clients.append(client)
        return client

    monkeypatch.setattr(ModelSingleton, "_models", {})
    monkeypatch.setattr(ModelSingleton, "_create_model", factory)
    owner = RuntimeOwner()
    manager = ModelSingleton()

    async def run() -> None:
        """重复取消等待初始化的调用，确认不会留下迟到的模型。"""
        token = runtime_owner.set(owner)
        try:
            task = asyncio.create_task(run_sync(manager.get_model, "http-client", None, "http://unused"))
            assert await asyncio.to_thread(entered.wait, 2)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            await run_sync(manager.release_owner, owner)
        finally:
            release.set()
            runtime_owner.reset(token)

    asyncio.run(run())
    assert clients[0].closed
    assert not manager._models


def test_failed_initialization_does_not_poison_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """初始化失败应退出线程并允许相同配置重新尝试。"""
    from mineru.model.vlm.runtime import ModelSingleton

    def fail(*args: Any, **kwargs: Any) -> None:
        """模拟模型发现或引擎初始化失败。"""
        raise ValueError("initialization failed")

    monkeypatch.setattr(ModelSingleton, "_models", {})
    monkeypatch.setattr(ModelSingleton, "_create_model", fail)
    manager = ModelSingleton()
    with pytest.raises(ValueError, match="initialization failed"):
        manager.get_model("http-client", None, "http://unused")
    assert not manager._models
    monkeypatch.setattr(ModelSingleton, "_create_model", lambda *args, **kwargs: _AsyncClient())
    try:
        assert manager.get_model("http-client", None, "http://unused").batch_two_step_extract([1]) == [1]
    finally:
        manager.shutdown()


def test_vllm_sync_and_async_names_resolve_to_one_async_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """同步引擎名称在托管边界规范化，防止初始化两套 vLLM 权重。"""
    from mineru.model.vlm import runtime

    calls = []

    def factory(self: Any, backend: str, *args: Any, **kwargs: Any) -> _AsyncClient:
        """记录实际请求构造的后端，并拒绝离线同步引擎。"""
        assert backend == "vllm-async-engine"
        calls.append(backend)
        return _AsyncClient()

    monkeypatch.setattr(runtime.ModelSingleton, "_models", {})
    monkeypatch.setattr(runtime.ModelSingleton, "_create_model", factory)
    monkeypatch.setattr(runtime, "get_device", lambda: "cuda")
    manager = runtime.ModelSingleton()
    try:
        sync = manager.get_model("vllm-engine", "/mock/model", None)
        asynchronous = manager.get_model("vllm-async-engine", "/mock/model", None)
        assert sync is asynchronous
        assert sync.batch_two_step_extract([1]) == [1]
        assert asyncio.run(asynchronous.aio_batch_two_step_extract([2])) == [2]
        assert calls == ["vllm-async-engine"]
    finally:
        manager.shutdown()


def test_real_model_factory_constructs_async_llm_on_owner_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """真实 MinerU 工厂必须在所属循环构造 AsyncLLM，并只关闭一次自有引擎。"""
    import sys
    from types import ModuleType
    from mineru.model.vlm import runtime

    events = []

    class EngineArgs:
        """保留传给 vLLM 的显式模型构造参数。"""

        def __init__(self, **kwargs: object) -> None:
            """不执行 GPU 分配，只记录配置。"""
            self.options = kwargs

    class Engine:
        """仅替换 vLLM 本身，保留 MinerU 的真实加载与关闭链。"""

        @classmethod
        def from_engine_args(cls, args: EngineArgs) -> Any:
            """要求构造发生在运行中的专属循环，而非调用方线程。"""
            events.append(("load", threading.current_thread().name, asyncio.get_running_loop(), args.options))
            return cls()

        def shutdown(self) -> None:
            """记录自有引擎的唯一关闭动作。"""
            events.append(("close",))

    modules = {
        name: ModuleType(name)
        for name in [
            "vllm",
            "vllm.config",
            "vllm.engine",
            "vllm.engine.arg_utils",
            "vllm.v1",
            "vllm.v1.engine",
            "vllm.v1.engine.async_llm",
        ]
    }
    modules["vllm.config"].CompilationConfig = EngineArgs
    modules["vllm.engine.arg_utils"].AsyncEngineArgs = EngineArgs
    modules["vllm.v1.engine.async_llm"].AsyncLLM = Engine
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(runtime.ModelSingleton, "_models", {})
    monkeypatch.setattr(runtime, "get_device", lambda: "cuda")
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.setattr(runtime, "mod_kwargs_by_device_type", lambda kwargs, **options: kwargs)
    monkeypatch.setattr(runtime, "enable_custom_logits_processors", lambda: False)
    monkeypatch.setattr(runtime, "set_default_gpu_memory_utilization", lambda: 0.8)
    monkeypatch.setattr(runtime, "MinerUClient", lambda **kwargs: _AsyncClient())
    manager = runtime.ModelSingleton()
    try:
        predictor = manager.get_model("vllm-engine", "/mock/model", None)
        assert predictor.batch_two_step_extract([1]) == [1]
        assert events[0][0:2] == ("load", "mineru-vllm-async-engine")
        assert events[0][3]["model"] == "/mock/model"
        assert manager.get_model("vllm-async-engine", "/mock/model", None) is predictor
    finally:
        manager.shutdown()
    assert [event[0] for event in events] == ["load", "close"]


def test_owner_cleanup_failure_still_closes_other_runtimes(monkeypatch: pytest.MonkeyPatch) -> None:
    """一个连接关闭失败也必须完成其余应用自有运行时的清理。"""
    from mineru.model.vlm.runtime import ModelSingleton

    clients = []

    async def fail_close() -> None:
        """模拟底层传输关闭异常。"""
        raise RuntimeError("close failed")

    def factory(*args: Any, **kwargs: Any) -> _AsyncClient:
        """仅第一个客户端关闭时失败。"""
        client = _AsyncClient()
        if not clients:
            client.aclose = fail_close
        clients.append(client)
        return client

    monkeypatch.setattr(ModelSingleton, "_models", {})
    monkeypatch.setattr(ModelSingleton, "_create_model", factory)
    owner = RuntimeOwner()
    manager = ModelSingleton()
    token = runtime_owner.set(owner)
    try:
        first = manager.get_model("http-client", None, "http://first")
        second = manager.get_model("http-client", None, "http://second")
    finally:
        runtime_owner.reset(token)
    with pytest.raises(RuntimeError, match="close failed"):
        manager.release_owner(owner)
    assert not manager._models
    assert not first._thread.is_alive()
    assert not second._thread.is_alive()
    assert clients[1].closed


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("async_mode", [False, True])
def test_vllm_runtime_uses_standard_progress_label(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    enabled: bool,
    async_mode: bool,
) -> None:
    """同步桥接和原生异步入口都通过真实高层抽取显示同一文案，不创建嵌套进度。"""
    from PIL import Image
    from mineru_vl_utils import MinerUClient
    from mineru_vl_utils.structs import ContentBlock
    from mineru_vl_utils.vlm_client.vllm_async_engine_client import VllmAsyncEngineVlmClient

    backend = object.__new__(VllmAsyncEngineVlmClient)
    backend.max_concurrency = 2

    async def predict(image: Any, prompt: str = "", **kwargs: Any) -> str:
        """仅替换 GPU 单请求，保持真实异步批处理和布局回填。"""
        return "recognized"

    def create() -> MinerUClient:
        """在运行时线程创建高层客户端，复用已注入的无 GPU 后端。"""
        return MinerUClient(backend="vllm-async-engine", vllm_async_llm=object(), use_tqdm=enabled)

    monkeypatch.setattr(backend, "aio_predict", predict)
    monkeypatch.setattr("mineru_vl_utils.mineru_client.new_vlm_client", lambda **kwargs: backend)
    runtime = AsyncVlmPredictor(create, lambda _: None, backend="vllm-async-engine", max_concurrency=2)
    image = Image.new("RGB", (32, 32))
    blocks = [[ContentBlock("text", [0, 0, 1, 1])]]
    try:
        runtime.wait_ready()
        results = (
            asyncio.run(runtime.aio_batch_extract_with_layout([image], blocks))
            if async_mode
            else runtime.batch_extract_with_layout([image], blocks)
        )
        assert results[0][0].content == "recognized"
        stderr = capsys.readouterr().err
        assert ("VLM Predict" in stderr) is enabled
        assert "External Layout Extraction" not in stderr
        assert "Processed prompts" not in stderr
        if enabled:
            assert "1/1" in stderr
    finally:
        runtime.shutdown()
        image.close()
