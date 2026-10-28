"""验证应用、文档和模型依赖的租约，不加载真实模型权重。"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from contextlib import contextmanager
from contextvars import copy_context
from dataclasses import dataclass
from typing import Any

import pytest

from mineru.config import get_config
from mineru.model.runtime import cpu_threads, hybrid, local_cache, onnx_device
from mineru.model.runtime.ownership import (
    RuntimeOwner,
    current_runtime_owner,
    release_runtime_owner,
    runtime_document_scope,
    runtime_owner,
)
from mineru.model.vlm.handles import RuntimeHandle
from mineru.model.vlm.runtime import ModelSingleton


@dataclass
class _Atom:
    """保留嵌套模型引用的轻量替身，用于验证实际缓存依赖关系。"""

    name: str
    dependency: _Atom | None = None


@contextmanager
def _owned(owner: RuntimeOwner | None) -> Iterator[None]:
    """在明确的应用上下文中调用真实模型缓存。"""
    token = runtime_owner.set(owner)
    try:
        yield
    finally:
        runtime_owner.reset(token)


@pytest.fixture
def local_models(monkeypatch: pytest.MonkeyPatch) -> list[_Atom]:
    """只替换实际模型构造，保留上下文、缓存键和嵌套借用的生产逻辑。"""
    created: list[_Atom] = []
    monkeypatch.setattr(hybrid.AtomModelSingleton, "_models", {})
    monkeypatch.setattr(hybrid.HybridLocalModelContextSingleton, "_models", {})
    monkeypatch.setattr(ModelSingleton, "_models", {})
    monkeypatch.setattr(ModelSingleton, "_closing", {})
    monkeypatch.setattr(ModelSingleton, "_initializing", {})
    monkeypatch.setattr(hybrid, "resolve_small_model_backend", lambda backend=None: backend or "onnx")
    monkeypatch.setattr(hybrid, "get_device", lambda: "cpu")
    monkeypatch.setattr(hybrid, "ocr_det_batch_setting", lambda: True)
    monkeypatch.setattr(cpu_threads, "initialize_torch_threads", lambda: 2)
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", lambda: "cpu")

    def create(model_name: str, **options: Any) -> _Atom:
        """表格模型像生产构造器一样借用不同配置的 OCR，形成真实二级依赖。"""
        dependency = None
        if model_name == "wireless_table":
            dependency = hybrid.AtomModelSingleton().get_atom_model(
                "ocr", small_backend=options["small_backend"], device=options["device"], det_db_unclip_ratio=1.6
            )
        model = _Atom(model_name, dependency)
        created.append(model)
        return model

    monkeypatch.setattr(hybrid, "atom_model_init", create)
    return created


def test_shared_context_keeps_cached_properties_until_last_owner(local_models: list[_Atom]) -> None:
    """A 首次加载惰性模型、B 复用同一上下文后，关闭 A 不得清理 B 的依赖。"""
    first, second = RuntimeOwner(), RuntimeOwner()
    manager = hybrid.HybridLocalModelContextSingleton()
    with _owned(first):
        context = manager.get_model()
        formula = context.mfr_model
        table = context.wireless_table_model
    with _owned(second):
        shared = manager.get_model()
        assert shared is context
        assert shared.mfr_model is formula
        assert shared.wireless_table_model is table
    release_runtime_owner(first)
    assert shared.mfr_model is formula
    assert shared.wireless_table_model.dependency is table.dependency
    assert len(local_models) == 5
    assert len(hybrid.AtomModelSingleton._models) == 5
    release_runtime_owner(second)
    assert not manager._models
    assert not hybrid.AtomModelSingleton._models


def test_distinct_contexts_preserve_shared_atom_dependencies(
    local_models: list[_Atom], monkeypatch: pytest.MonkeyPatch
) -> None:
    """不同 ONNX 配置上下文解析到同一设备时，共享的表格及其 OCR 依赖按最后持有者释放。"""
    first, second = RuntimeOwner(), RuntimeOwner()
    manager = hybrid.HybridLocalModelContextSingleton()
    monkeypatch.setattr(get_config().model, "onnx_device", "auto")
    with _owned(first):
        left = manager.get_model()
        table = left.wireless_table_model
    monkeypatch.setattr(get_config().model, "onnx_device", "cpu")
    with _owned(second):
        right = manager.get_model()
        assert right is not left
        assert right.ocr_model is left.ocr_model
        assert right.wireless_table_model is table
    release_runtime_owner(first)
    assert right.wireless_table_model.dependency is table.dependency
    assert len(local_models) == 4
    assert len(hybrid.AtomModelSingleton._models) == 4
    release_runtime_owner(second)
    assert not manager._models
    assert not hybrid.AtomModelSingleton._models


def test_backend_and_device_changes_release_only_their_context(
    local_models: list[_Atom], monkeypatch: pytest.MonkeyPatch
) -> None:
    """后端或设备切换仍沿原缓存键隔离，关闭一个配置不会回收另一个配置。"""
    first, second = RuntimeOwner(), RuntimeOwner()
    manager = hybrid.HybridLocalModelContextSingleton()
    with _owned(first):
        onnx_context = manager.get_model()
    monkeypatch.setattr(hybrid, "resolve_small_model_backend", lambda backend=None: backend or "torch")
    monkeypatch.setattr(hybrid, "get_device", lambda: "cuda:0")
    with _owned(second):
        torch_context = manager.get_model()
    assert torch_context.ocr_model is not onnx_context.ocr_model
    release_runtime_owner(first)
    assert len(manager._models) == 1
    assert len(hybrid.AtomModelSingleton._models) == 2
    with _owned(second):
        assert manager.get_model() is torch_context
    release_runtime_owner(second)
    assert not manager._models
    assert not hybrid.AtomModelSingleton._models


def test_cli_pin_survives_application_release(local_models: list[_Atom]) -> None:
    """独立 CLI 借用固定进程缓存，后续应用退出不能撤销该固定租约。"""
    manager = hybrid.HybridLocalModelContextSingleton()
    with _owned(None):
        pinned = manager.get_model()
    owner = RuntimeOwner()
    with _owned(owner):
        assert manager.get_model() is pinned
    release_runtime_owner(owner)
    with _owned(None):
        assert manager.get_model() is pinned
    assert len(hybrid.AtomModelSingleton._models) == 2


def test_empty_and_shared_owner_release_do_not_force_full_gc(
    local_models: list[_Atom], monkeypatch: pytest.MonkeyPatch
) -> None:
    """空应用和仍被共享的模型无需全量 GC，最后上下文释放依赖后只回收一次。"""
    collections: list[bool] = []

    def collect() -> int:
        """记录显式全量回收，避免测试本身依赖真实内存回收时机。"""
        collections.append(True)
        return 0

    monkeypatch.setattr(local_cache.gc, "collect", collect)
    release_runtime_owner(RuntimeOwner())
    assert not collections
    first, second = RuntimeOwner(), RuntimeOwner()
    with _owned(first):
        hybrid.HybridLocalModelContextSingleton().get_model()
    with _owned(second):
        hybrid.HybridLocalModelContextSingleton().get_model()
    release_runtime_owner(first)
    assert not collections
    release_runtime_owner(second)
    assert collections == [True]


@pytest.mark.parametrize("shared", [False, True])
def test_failed_context_initialization_rolls_back_unshared_dependencies(
    local_models: list[_Atom], monkeypatch: pytest.MonkeyPatch, shared: bool
) -> None:
    """OCR 已初始化而版面模型失败时，失败上下文及其孤立 OCR 租约必须一并撤销。"""
    create = hybrid.atom_model_init

    def fail_layout(model_name: str, **options: Any) -> _Atom:
        """只让第二个必需阶段失败，真实执行先前依赖的登记与回滚。"""
        if model_name == "layout":
            raise ValueError("layout initialization failed")
        return create(model_name, **options)

    monkeypatch.setattr(hybrid, "atom_model_init", fail_layout)
    existing_owner = RuntimeOwner()
    if shared:
        with _owned(existing_owner):
            hybrid.AtomModelSingleton().get_atom_model("ocr")
    owner = RuntimeOwner()
    with _owned(owner), pytest.raises(ValueError, match="layout initialization failed"):
        hybrid.HybridLocalModelContextSingleton().get_model()
    assert len(local_models) == 1
    assert not hybrid.HybridLocalModelContextSingleton._models
    assert len(hybrid.AtomModelSingleton._models) == int(shared)
    release_runtime_owner(existing_owner)
    assert not hybrid.AtomModelSingleton._models


def test_owner_close_waits_for_document_and_allows_its_lazy_models(local_models: list[_Atom]) -> None:
    """关闭应用时等待已接纳文档继续惰性加载并结束，同时拒绝新文档进入。"""
    owner = RuntimeOwner()
    entered, continue_document, lazy_loaded = threading.Event(), threading.Event(), threading.Event()

    def document() -> None:
        """模拟正在同步执行的文档，在关闭开始后才首次读取公式模型。"""
        with _owned(owner), runtime_document_scope():
            context = hybrid.HybridLocalModelContextSingleton().get_model()
            entered.set()
            assert continue_document.wait(3)
            assert context.mfr_model.name == "mfr"
            lazy_loaded.set()

    with ThreadPoolExecutor(max_workers=2) as executor:
        work = executor.submit(document)
        try:
            assert entered.wait(3)
            closing = executor.submit(release_runtime_owner, owner)
            with pytest.raises(TimeoutError):
                closing.result(timeout=0.05)
            assert owner.closed
            with _owned(owner), pytest.raises(RuntimeError, match="owner is closed"), runtime_document_scope():
                raise AssertionError("New document must not enter")
        finally:
            continue_document.set()
        work.result(timeout=3)
        closing.result(timeout=3)
    assert lazy_loaded.is_set()
    assert len(local_models) == 3
    assert not hybrid.HybridLocalModelContextSingleton._models
    assert not hybrid.AtomModelSingleton._models


def test_owner_release_inside_document_is_rejected_without_deadlock() -> None:
    """文档内部同步释放自己的应用时明确报错，不能等待当前文档自身退出。"""
    owner = RuntimeOwner()
    with _owned(owner), runtime_document_scope():
        with pytest.raises(RuntimeError, match="active document"):
            release_runtime_owner(owner)
        assert not owner.closed
        assert current_runtime_owner() is owner
    owner.close()
    assert owner.closed


def test_expired_document_context_cannot_acquire_models(local_models: list[_Atom]) -> None:
    """已退出文档复制到迟到线程的上下文不能重新借用或初始化模型。"""
    owner = RuntimeOwner()
    with _owned(owner), runtime_document_scope():
        delayed = copy_context()
    with pytest.raises(RuntimeError, match="lease has ended"):
        delayed.run(hybrid.HybridLocalModelContextSingleton().get_model)
    assert not local_models


class _SyncClient:
    """记录同步执行边界的微型客户端，不启动模型或网络连接。"""

    def __init__(self) -> None:
        """创建可以由测试显式放行的同步执行门。"""
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls: list[list[int]] = []

    def batch_two_step_extract(self, images: list[int], **kwargs: Any) -> list[int]:
        """模拟无法被 asyncio 取消强行打断的 CPU 推理。"""
        self.calls.append(images)
        self.entered.set()
        assert self.release.wait(3)
        return images

    async def aclose(self) -> None:
        """为异步托管测试提供客户端关闭接口，本替身没有额外连接资源。"""


def test_sync_backend_owners_share_until_last_application_release(
    local_models: list[_Atom], monkeypatch: pytest.MonkeyPatch
) -> None:
    """同步后端同样按应用租约共享，A 关闭后 B 可继续推理，最后关闭才释放引擎。"""
    closed = threading.Event()
    client = _SyncClient()
    client.release.set()
    monkeypatch.setattr(
        ModelSingleton,
        "_create_model",
        lambda *args, **kwargs: RuntimeHandle(client, backend="transformers", close_engine=closed.set),
    )
    first, second = RuntimeOwner(), RuntimeOwner()
    with _owned(first):
        predictor = ModelSingleton().get_model("transformers", "shared", None)
    with _owned(second):
        assert ModelSingleton().get_model("transformers", "shared", None) is predictor
    release_runtime_owner(first)
    assert not closed.is_set()
    with _owned(second):
        assert predictor.batch_two_step_extract([1]) == [1]
    release_runtime_owner(second)
    assert closed.is_set()
    assert not ModelSingleton._models


def test_sync_runtime_shutdown_waits_for_inference() -> None:
    """宿主句柄关闭前等待同步推理自然完成，不提前调用底层引擎关闭方法。"""
    client = _SyncClient()
    closed = threading.Event()
    handle = RuntimeHandle(client, backend="transformers", close_engine=closed.set)
    with ThreadPoolExecutor(max_workers=2) as executor:
        inference = executor.submit(handle.batch_two_step_extract, [1])
        try:
            assert client.entered.wait(3)
            shutdown = executor.submit(handle.shutdown)
            with pytest.raises(TimeoutError):
                shutdown.result(timeout=0.05)
            assert not closed.is_set()
        finally:
            client.release.set()
        assert inference.result(timeout=3) == [1]
        shutdown.result(timeout=3)
    assert closed.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        _ = handle.predictor


def test_handle_closes_engine_after_client_error_and_preserves_first_error() -> None:
    """客户端和引擎关闭都失败时仍执行两个动作，并保留最早的异常。"""
    closed: list[str] = []

    class Client:
        """只提供可观察关闭错误的客户端替身。"""

        async def aclose(self) -> None:
            """模拟自有连接释放失败。"""
            closed.append("client")
            raise ValueError("client close failed")

    def close_engine() -> None:
        """模拟引擎释放同样失败，不能覆盖前面的客户端异常。"""
        closed.append("engine")
        raise RuntimeError("engine close failed")

    handle = RuntimeHandle(Client(), backend="http-client", close_engine=close_engine)
    with pytest.raises(ValueError, match="client close failed"):
        asyncio.run(handle.aclose())
    assert closed == ["client", "engine"]
    assert handle.is_closed
    with pytest.raises(RuntimeError, match="closed"):
        _ = handle.predictor


def test_mlx_queued_async_cancel_never_starts_delayed_inference() -> None:
    """同步推理占用 MLX 锁时取消异步排队者，不得在锁释放后执行已取消请求。"""
    client = _SyncClient()
    handle = RuntimeHandle(client, backend="mlx-engine")

    async def run() -> None:
        """交错真实句柄的同步与异步入口，并验证取消后下一请求仍能获得锁。"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(handle.batch_two_step_extract, [1])
            try:
                assert await asyncio.to_thread(client.entered.wait, 3)
                queued = asyncio.create_task(handle.aio_batch_two_step_extract([2]))
                await asyncio.sleep(0.03)
                queued.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await queued
                assert client.calls == [[1]]
            finally:
                client.release.set()
            assert await asyncio.wrap_future(first) == [1]
            assert await asyncio.wait_for(handle.aio_batch_two_step_extract([3]), timeout=1) == [3]

    try:
        asyncio.run(run())
        assert client.calls == [[1], [3]]
    finally:
        handle.shutdown()


def test_mlx_running_async_cancel_drains_before_next_sync_call() -> None:
    """异步入口已开始 CPU 推理后，取消必须排空线程再向同步等待者交出 MLX 锁。"""
    client = _SyncClient()
    handle = RuntimeHandle(client, backend="mlx-engine")

    async def run() -> None:
        """取消正在执行的异步调用，并在同一锁上安排后续同步调用。"""
        first = asyncio.create_task(handle.aio_batch_two_step_extract([1]))
        assert await asyncio.to_thread(client.entered.wait, 3)
        with ThreadPoolExecutor(max_workers=1) as executor:
            second = executor.submit(handle.batch_two_step_extract, [2])
            try:
                first.cancel()
                await asyncio.sleep(0.02)
                assert not first.done()
                assert client.calls == [[1]]
            finally:
                client.release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert await asyncio.wait_for(asyncio.wrap_future(second), timeout=1) == [2]

    try:
        asyncio.run(run())
        assert client.calls == [[1], [2]]
    finally:
        handle.shutdown()


def test_cleanup_error_does_not_skip_other_runtimes_or_local_models(
    local_models: list[_Atom], monkeypatch: pytest.MonkeyPatch
) -> None:
    """一个 VLM 关闭失败时仍清理其它 VLM 与全部本地依赖，最终传播最初的异常。"""
    closed: list[str] = []

    def create(self: ModelSingleton, backend: str, path: str, *args: Any, **kwargs: Any) -> RuntimeHandle:
        """为每个配置绑定唯一关闭动作，首个动作故意失败。"""

        def close() -> None:
            """记录关闭顺序并模拟一个后端释放失败。"""
            closed.append(path)
            if path == "first":
                raise ValueError("first close failed")

        return RuntimeHandle(_SyncClient(), backend=backend, close_engine=close)

    monkeypatch.setattr(ModelSingleton, "_create_model", create)
    owner = RuntimeOwner()
    with _owned(owner):
        hybrid.HybridLocalModelContextSingleton().get_model()
        first = ModelSingleton().get_model("transformers", "first", None)
        second = ModelSingleton().get_model("transformers", "second", None)
    with pytest.raises(ValueError, match="first close failed"):
        release_runtime_owner(owner)
    assert closed == ["first", "second"]
    assert first.is_closed and second.is_closed
    assert not ModelSingleton._models
    assert not ModelSingleton._closing
    assert not hybrid.HybridLocalModelContextSingleton._models
    assert not hybrid.AtomModelSingleton._models


def test_backend_close_runs_outside_global_cache_lock(local_models: list[_Atom], monkeypatch: pytest.MonkeyPatch) -> None:
    """关闭回调可等待另一个线程获取不同模型，证明外部关闭不持有全局缓存锁。"""
    first, second = RuntimeOwner(), RuntimeOwner()
    entered = threading.Event()

    def borrow_other() -> None:
        """在另一个应用中构造不同缓存键的模型。"""
        with _owned(second):
            ModelSingleton().get_model("transformers", "other", None)
        entered.set()

    def close() -> None:
        """真实等待另一个线程完成模型借用，持锁实现会在此超时。"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(borrow_other).result(timeout=2)

    def create(self: ModelSingleton, backend: str, path: str, *args: Any, **kwargs: Any) -> RuntimeHandle:
        """只给首个模型安装跨线程关闭回调。"""
        return RuntimeHandle(_SyncClient(), backend=backend, close_engine=close if path == "first" else None)

    monkeypatch.setattr(ModelSingleton, "_create_model", create)
    with _owned(first):
        ModelSingleton().get_model("transformers", "first", None)
    release_runtime_owner(first)
    assert entered.is_set()
    release_runtime_owner(second)


def test_same_cache_key_waits_for_previous_engine_close(local_models: list[_Atom], monkeypatch: pytest.MonkeyPatch) -> None:
    """旧引擎关闭期间同键借用必须等待，不能并行创建第二份权重。"""
    first, second = RuntimeOwner(), RuntimeOwner()
    closing, release_close = threading.Event(), threading.Event()
    created: list[RuntimeHandle] = []

    def close() -> None:
        """让真实缓存的关闭阶段停留在已撤销但资源尚未释放的状态。"""
        closing.set()
        assert release_close.wait(3)

    def create(self: ModelSingleton, backend: str, *args: Any, **kwargs: Any) -> RuntimeHandle:
        """记录同键工厂调用次数，只有首个实例需要阻塞关闭。"""
        handle = RuntimeHandle(_SyncClient(), backend=backend, close_engine=close if not created else None)
        created.append(handle)
        return handle

    def borrow() -> RuntimeHandle:
        """在另一应用的上下文中请求正在关闭的同一模型。"""
        with _owned(second):
            return ModelSingleton().get_model("transformers", "same", None)

    monkeypatch.setattr(ModelSingleton, "_create_model", create)
    with _owned(first):
        initial = ModelSingleton().get_model("transformers", "same", None)
    with ThreadPoolExecutor(max_workers=2) as executor:
        shutdown = executor.submit(release_runtime_owner, first)
        try:
            assert closing.wait(3)
            replacement = executor.submit(borrow)
            with pytest.raises(TimeoutError):
                replacement.result(timeout=0.05)
            assert len(created) == 1
        finally:
            release_close.set()
        shutdown.result(timeout=3)
        assert replacement.result(timeout=3) is not initial
    assert len(created) == 2
    release_runtime_owner(second)


@pytest.mark.parametrize("backend", ["transformers", "http-client"])
def test_same_cache_key_initializes_once_for_concurrent_owners(
    local_models: list[_Atom], monkeypatch: pytest.MonkeyPatch, backend: str
) -> None:
    """同步和原生异步后端的同键并发首次获取只构造一次，并分别保留两个应用租约。"""
    first, second = RuntimeOwner(), RuntimeOwner()
    entered, release = threading.Event(), threading.Event()
    created: list[RuntimeHandle] = []

    def create(self: ModelSingleton, backend: str, *args: Any, **kwargs: Any) -> RuntimeHandle:
        """延迟真实缓存工厂的完成，迫使第二应用等待同一个初始化屏障。"""
        entered.set()
        assert release.wait(3)
        handle = RuntimeHandle(_SyncClient(), backend=backend)
        created.append(handle)
        return handle

    def borrow(owner: RuntimeOwner) -> Any:
        """从指定应用首次请求同一模型键。"""
        with _owned(owner):
            return ModelSingleton().get_model(backend, "same", "https://unused.test")

    monkeypatch.setattr(ModelSingleton, "_create_model", create)
    with ThreadPoolExecutor(max_workers=2) as executor:
        left = executor.submit(borrow, first)
        try:
            assert entered.wait(3)
            right = executor.submit(borrow, second)
            with pytest.raises(TimeoutError):
                right.result(timeout=0.05)
        finally:
            release.set()
        predictor = left.result(timeout=3)
        assert right.result(timeout=3) is predictor
    assert len(created) == 1
    assert not ModelSingleton._initializing
    release_runtime_owner(first)
    assert not predictor.is_closed
    release_runtime_owner(second)
    assert predictor.is_closed
    assert not ModelSingleton._models


def test_document_scope_propagates_to_async_worker_thread(local_models: list[_Atom]) -> None:
    """异步文档在线程准备阶段继续使用同一租约，退出后原线程上下文不再有效。"""
    owner = RuntimeOwner()

    async def run() -> None:
        """沿生产中的 to_thread 传播方式借用模型并捕获线程上下文。"""
        with _owned(owner), runtime_document_scope():
            context = await asyncio.to_thread(hybrid.HybridLocalModelContextSingleton().get_model)
            delayed = await asyncio.to_thread(copy_context)
            assert await asyncio.to_thread(lambda: context.mfr_model.name) == "mfr"
        with pytest.raises(RuntimeError, match="lease has ended"):
            delayed.run(current_runtime_owner)

    asyncio.run(run())
    release_runtime_owner(owner)
    assert not hybrid.AtomModelSingleton._models
