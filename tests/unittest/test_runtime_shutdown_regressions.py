"""用有界独立进程回归运行时关闭再入、失败构造及取消清理，不加载真实模型。"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


_BOOTSTRAP = '''
import asyncio
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

from mineru.model.runtime.ownership import RuntimeOwner, runtime_owner
from mineru.model.vlm import runtime
from mineru.model.vlm.handles import RuntimeHandle

manager = runtime.ModelSingleton()

class Client:
    """不创建网络连接或模型的客户端，记录自有关闭动作。"""

    def __init__(self) -> None:
        """初始化关闭计数。"""
        self.close_calls = 0

    async def aclose(self) -> None:
        """完成一次可观察的连接清理。"""
        self.close_calls += 1

def borrow(owner, backend, model_path, server_url):
    """给真实缓存调用设置明确的应用身份。"""
    token = runtime_owner.set(owner)
    try:
        return manager.get_model(backend, model_path, server_url)
    finally:
        runtime_owner.reset(token)
'''


def _run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    """把死锁上限限定在单个子进程，失败不遗留线程或污染其它模型缓存测试。"""
    try:
        completed = subprocess.run(
            [sys.executable, "-c", _BOOTSTRAP + "\n" + textwrap.dedent(script)],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Runtime shutdown did not complete within 10 seconds: {exc.stdout!r}\n{exc.stderr!r}", pytrace=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    return completed


@pytest.mark.parametrize("warm_executor", [False, True])
def test_interpreter_exit_closes_pinned_async_runtime(warm_executor: bool) -> None:
    """真实解释器退出后仍关闭客户端和引擎，不能向已经停止的线程池提交清理工作。"""
    completed = _run_isolated(
        f"warm_executor = {warm_executor!r}\n"
        + '''
class ExitClient(Client):
    """将进程退出阶段的客户端清理写入标准输出，供父进程验证。"""

    async def aclose(self) -> None:
        """保留异步调度点，验证清理仍在所属循环中完整执行。"""
        await asyncio.sleep(0)
        await super().aclose()
        print("client_closed", flush=True)

def create(self, backend, model_path, server_url, **kwargs):
    """用真实缓存与异步运行时管理轻量客户端，保留实际 atexit 注册流程。"""
    return RuntimeHandle(ExitClient(), backend=backend, close_engine=lambda: print("engine_closed", flush=True))

async def warm_default_executor() -> None:
    """覆盖推理曾经启用过默认线程池的解释器退出路径。"""
    await asyncio.to_thread(lambda: None)

runtime.ModelSingleton._create_model = create
predictor = manager.get_model("http-client", None, "http://example.test")
if warm_executor:
    predictor._call(warm_default_executor)
# 故意不显式调用 shutdown，验证真实解释器退出顺序。
'''
    )
    assert completed.stdout.splitlines() == ["client_closed", "engine_closed"]
    assert "Exception ignored in atexit callback" not in completed.stderr
    assert "cannot schedule new futures" not in completed.stderr
    assert "Exception in thread" not in completed.stderr


def test_sync_close_callback_rejects_another_key_in_its_closing_batch() -> None:
    """A 的关闭回调不能等待同批尚未关闭的 B，同时仍允许另一应用借用未关闭键。"""
    _run_isolated('''
        first_owner, other_owner = RuntimeOwner(), RuntimeOwner()
        closed, rejected = [], []

        def create(self, backend, model_path, server_url, **kwargs):
            """仅第一个模型的关闭动作尝试借用其它缓存键。"""
            def close() -> None:
                """非关闭键仍可复用，当前关闭批次中的键必须及时拒绝。"""
                closed.append(model_path)
                if model_path != "first":
                    return
                assert borrow(other_owner, "transformers", "other", None) is other
                try:
                    borrow(other_owner, "transformers", "second", None)
                except RuntimeError as error:
                    rejected.append(str(error))
                else:
                    raise AssertionError("A close callback acquired another key from its closing batch")
            return RuntimeHandle(Client(), backend=backend, close_engine=close)

        runtime.ModelSingleton._create_model = create
        other = borrow(other_owner, "transformers", "other", None)
        first = borrow(first_owner, "transformers", "first", None)
        second = borrow(first_owner, "transformers", "second", None)
        manager.release_owner(first_owner)
        assert closed == ["first", "second"]
        assert len(rejected) == 1
        assert first.is_closed and second.is_closed and not other.is_closed
        manager.release_owner(other_owner)
        assert closed == ["first", "second", "other"]
        manager.shutdown()
    ''')


def test_async_runtime_close_callback_rejects_its_own_cache_key() -> None:
    """关闭再入保护必须传播到专属异步运行时线程，不能只比较外部关闭线程 ID。"""
    _run_isolated('''
        owner, other_owner = RuntimeOwner(), RuntimeOwner()
        closed, rejected, clients = [], [], []

        def create(self, backend, model_path, server_url, **kwargs):
            """由真实 AsyncVlmPredictor 在所属循环内构造一个轻量 HTTP 句柄。"""
            client = Client()
            clients.append(client)
            def close() -> None:
                """异步运行时关闭回调可借用其它键，但不能等待当前键自己的屏障。"""
                closed.append(backend)
                if backend != "http-client":
                    return
                borrow(other_owner, "transformers", "other", None)
                try:
                    borrow(other_owner, "http-client", None, "http://example.test")
                except RuntimeError as error:
                    rejected.append(str(error))
                else:
                    raise AssertionError("An async close callback acquired its own closing key")
            return RuntimeHandle(client, backend=backend, close_engine=close)

        runtime.ModelSingleton._create_model = create
        predictor = borrow(owner, "http-client", None, "http://example.test")
        manager.release_owner(owner)
        assert predictor.is_closed and closed == ["http-client"]
        assert clients[0].close_calls == 1 and len(rejected) == 1
        manager.release_owner(other_owner)
        assert closed == ["http-client", "transformers"]
        manager.shutdown()
    ''')


@pytest.mark.parametrize("asynchronous", [False, True])
def test_direct_cached_handle_close_rejects_same_key_reentry(asynchronous: bool) -> None:
    """直接关闭仍在缓存中的句柄时，同键再入不能绕过 cached.is_closed 分支的保护。"""
    _run_isolated(
        f"asynchronous = {asynchronous!r}\n"
        + '''
owner, other_owner = RuntimeOwner(), RuntimeOwner()
created, closed, rejected = [], [], []

def create(self, backend, model_path, server_url, **kwargs):
    """首个句柄在其关闭回调中故意重新请求同一缓存键。"""
    identity = len(created)
    def close() -> None:
        """直接句柄关闭未提前撤销缓存，同键请求仍应立即失败。"""
        closed.append(identity)
        if identity != 0:
            return
        try:
            borrow(other_owner, "transformers", "same", None)
        except RuntimeError as error:
            rejected.append(str(error))
        else:
            raise AssertionError("Direct handle close allowed same-key reentry")
    handle = RuntimeHandle(Client(), backend=backend, close_engine=close)
    created.append(handle)
    return handle

runtime.ModelSingleton._create_model = create
handle = borrow(owner, "transformers", "same", None)
if asynchronous:
    asyncio.run(handle.aclose())
else:
    handle.shutdown()
assert handle.is_closed and closed == [0] and len(rejected) == 1
replacement = borrow(other_owner, "transformers", "same", None)
assert replacement is not handle and len(created) == 2
manager.release_owner(owner)
assert not replacement.is_closed
manager.release_owner(other_owner)
assert closed == [0, 1]
manager.shutdown()
'''
    )


def test_close_callback_does_not_wait_for_initializer_that_needs_closing_key() -> None:
    """A 的关闭回调不能等待正在构造且反向等待 A 的 B，避免关闭与初始化互等。"""
    _run_isolated('''
        first_owner, second_owner = RuntimeOwner(), RuntimeOwner()
        b_entered, allow_b = threading.Event(), threading.Event()
        a_handles, b_results, errors, rejected = [], [], [], []

        def create(self, backend, model_path, server_url, **kwargs):
            """让 B 的构造等待 A 已开始关闭，再请求 A 的替代实例。"""
            if model_path == "B":
                b_entered.set()
                assert allow_b.wait(3)
                borrow(second_owner, "transformers", "A", None)
                return RuntimeHandle(Client(), backend=backend)
            identity = len(a_handles)
            def close() -> None:
                """仅旧 A 的关闭回调尝试获取仍在普通构造阶段的 B。"""
                if identity != 0:
                    return
                allow_b.set()
                try:
                    borrow(second_owner, "transformers", "B", None)
                except RuntimeError as error:
                    rejected.append(str(error))
                else:
                    raise AssertionError("A close callback waited for a pending initializer")
            handle = RuntimeHandle(Client(), backend=backend, close_engine=close)
            a_handles.append(handle)
            return handle

        def initialize_b() -> None:
            """在独立应用中执行真实同键初始化等待流程。"""
            try:
                b_results.append(borrow(second_owner, "transformers", "B", None))
            except BaseException as error:
                errors.append(error)

        runtime.ModelSingleton._create_model = create
        initial_a = borrow(first_owner, "transformers", "A", None)
        worker = threading.Thread(target=initialize_b, daemon=True)
        worker.start()
        assert b_entered.wait(2)
        manager.release_owner(first_owner)
        worker.join(2)
        assert not worker.is_alive() and not errors
        assert initial_a.is_closed and len(rejected) == 1
        assert len(a_handles) == 2 and len(b_results) == 1
        assert not a_handles[1].is_closed and not b_results[0].is_closed
        manager.release_owner(second_owner)
        assert a_handles[1].is_closed and b_results[0].is_closed
        manager.shutdown()
    ''')


def test_failed_real_factory_closes_engine_outside_global_cache_lock() -> None:
    """真实工厂的客户端构造失败清理也必须允许其它线程加载不同模型键。"""
    _run_isolated('''
        import sys
        from mineru.model.runtime import llama_device

        owner, other_owner = RuntimeOwner(), RuntimeOwner()
        finished = threading.Event()
        waited, worker_errors, workers = [], [], []

        @contextmanager
        def engine_scope(selection):
            """跳过真实设备环境操作，保留工厂内部的引擎创建顺序。"""
            yield

        def client_factory(**kwargs):
            """只让 llama 高层客户端失败，其它键正常创建轻量客户端。"""
            if kwargs["backend"] == "llama-cpp-engine":
                raise ValueError("client initialization failed")
            return Client()

        class Engine:
            """不读取权重的引擎替身，使用真实工厂调用其关闭动作。"""

            def __init__(self, *args, **kwargs) -> None:
                """接收真实工厂的路径和选项但不分配模型。"""

            def close(self) -> None:
                """在清理回调中等待不同线程完成另一个缓存键的构造。"""
                def load_other() -> None:
                    """给另一应用构造轻量异步客户端，并记录可能的异常。"""
                    try:
                        borrow(other_owner, "http-client", None, "http://other.test")
                    except BaseException as error:
                        worker_errors.append(error)
                    finally:
                        finished.set()
                worker = threading.Thread(target=load_other, daemon=True)
                workers.append(worker)
                worker.start()
                waited.append(finished.wait(2))

        # 使用完整设备计划，让真实线程策略通过后继续验证失败构造的清理与锁边界。
        llama_device.resolve_llama_device = lambda **kwargs: llama_device.LlamaSelection(device="cpu")
        llama_device.llama_engine_scope = engine_scope
        runtime.MinerUClient = client_factory
        sys.modules["mineru_llama_cpp"] = SimpleNamespace(Engine=Engine)
        try:
            borrow(owner, "llama-cpp-engine", "/tmp/fake-model", None)
        except ValueError as error:
            assert str(error) == "client initialization failed"
        else:
            raise AssertionError("The original initialization error was lost")
        assert finished.wait(2)
        for worker in workers:
            worker.join(2)
        manager.release_owner(owner)
        manager.release_owner(other_owner)
        manager.shutdown()
        assert not worker_errors
        assert waited == [True], "Failed initialization cleanup still holds the global model lock"
    ''')


@pytest.mark.parametrize("stage", ["inference_drain", "client_close"])
def test_cancelled_handle_close_drains_client_and_engine_once(stage: str) -> None:
    """等待推理或关闭客户端期间重复取消，都必须完成清理再向调用方传播取消。"""
    _run_isolated(
        f"stage = {stage!r}\n"
        + '''
class BlockingClient(Client):
    """分别阻塞同步推理与异步关闭，使两个取消时序可以稳定重现。"""

    def __init__(self) -> None:
        """在调用方事件循环中创建关闭门，线程只使用线程事件。"""
        super().__init__()
        self.entered = threading.Event()
        self.release = threading.Event()
        self.close_started = asyncio.Event()
        self.finish_close = asyncio.Event()
        self.close_completed = False

    def batch_two_step_extract(self, images, **kwargs):
        """模拟无法被协程取消强制中断的同步模型调用。"""
        self.entered.set()
        assert self.release.wait(3)
        return images

    async def aclose(self) -> None:
        """记录关闭开始与真正完成，不能把一次调用误认为资源已经释放。"""
        self.close_calls += 1
        self.close_started.set()
        await self.finish_close.wait()
        self.close_completed = True

async def check() -> None:
    """重复取消真实句柄的关闭入口，并验证后续幂等关闭不会重复释放。"""
    client = BlockingClient()
    engine_closed = []
    handle = RuntimeHandle(client, backend="http-client", close_engine=lambda: engine_closed.append(True))
    work = None
    if stage == "inference_drain":
        client.finish_close.set()
        work = asyncio.create_task(handle.aio_batch_two_step_extract([1]))
        assert await asyncio.to_thread(client.entered.wait, 2)
    closing = asyncio.create_task(handle.aclose())
    try:
        if stage == "client_close":
            await asyncio.wait_for(client.close_started.wait(), 2)
        else:
            await asyncio.sleep(0.03)
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0.02)
        assert not closing.done(), "Closing returned before the blocked resource finished"
    finally:
        client.release.set()
        client.finish_close.set()
    if work is not None:
        assert await work == [1]
    try:
        await closing
    except asyncio.CancelledError:
        pass
    else:
        raise AssertionError("Caller cancellation was swallowed")
    assert client.close_calls == 1 and client.close_completed
    assert engine_closed == [True] and handle.is_closed
    await handle.aclose()
    assert client.close_calls == 1 and engine_closed == [True]

asyncio.run(check())
'''
    )


def test_closed_owner_is_rejected_before_device_probe_or_model_download() -> None:
    """已关闭应用在任何缺省模型路径解析或设备探测前就必须被拒绝。"""
    _run_isolated("""
        calls = []
        runtime.get_device = lambda: calls.append("device") or "cpu"
        runtime.MINERU_2_5_PRO_2605_1_2B = SimpleNamespace(ensure=lambda: calls.append("ensure") or "/tmp/mock-model")
        owner = RuntimeOwner()
        owner.close()
        try:
            borrow(owner, "vllm-engine", None, None)
        except RuntimeError:
            pass
        else:
            raise AssertionError("A closed owner started model loading")
        assert calls == []
    """)


def test_owner_release_waits_for_inflight_synchronous_initialization() -> None:
    """锁外初始化与应用关闭竞争时，关闭等待构造收尾并禁止迟到发布缓存。"""
    _run_isolated('''
        owner, other_owner = RuntimeOwner(), RuntimeOwner()
        entered, release, close_done = threading.Event(), threading.Event(), threading.Event()
        created, engine_closed, borrow_errors, close_errors = [], [], [], []

        def create(self, backend, model_path, server_url, **kwargs):
            """只阻塞第一个实例，后续应用仍可重新构造同键模型。"""
            identity = len(created)
            handle = RuntimeHandle(Client(), backend=backend, close_engine=lambda: engine_closed.append(identity))
            created.append(handle)
            if identity == 0:
                entered.set()
                assert release.wait(3)
            return handle

        def initialize() -> None:
            """在原应用中发起没有文档范围的预热式模型构造。"""
            try:
                borrow(owner, "transformers", "same", None)
            except BaseException as error:
                borrow_errors.append(error)

        def close() -> None:
            """记录关闭真正返回的时刻，不能只验证缓存最终为空。"""
            try:
                manager.release_owner(owner)
            except BaseException as error:
                close_errors.append(error)
            finally:
                close_done.set()

        runtime.ModelSingleton._create_model = create
        initializing = threading.Thread(target=initialize, daemon=True)
        initializing.start()
        assert entered.wait(2)
        closing = threading.Thread(target=close, daemon=True)
        closing.start()
        try:
            deadline = time.monotonic() + 2
            while not owner.closed and time.monotonic() < deadline:
                time.sleep(0.005)
            assert owner.closed
            assert not close_done.wait(0.05), "Owner release skipped its inflight initializer"
        finally:
            release.set()
        initializing.join(2)
        closing.join(2)
        assert not initializing.is_alive() and not closing.is_alive()
        assert not close_errors
        assert len(borrow_errors) == 1 and isinstance(borrow_errors[0], RuntimeError)
        assert created[0].is_closed and engine_closed == [0]
        replacement = borrow(other_owner, "transformers", "same", None)
        assert replacement is not created[0] and len(created) == 2
        manager.release_owner(other_owner)
        assert engine_closed == [0, 1]
        manager.shutdown()
    ''')
