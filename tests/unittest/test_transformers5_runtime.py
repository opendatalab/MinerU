"""验证新推理栈的惰性加载边界及 LMDeploy Pipeline 生命周期。"""

from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest


def test_public_imports_keep_optional_model_dependencies_lazy() -> None:
    """新进程中的公共入口导入不得尝试加载任何可选推理引擎。"""
    code = """
import importlib.abc
import sys

class RejectModelImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # 显式阻止重依赖，检查公共导入是否越过模型加载边界。
        if fullname.split('.')[0] in {'torch', 'transformers', 'vllm', 'lmdeploy', 'mlx', 'mlx_vlm'}:
            raise AssertionError('Eager optional dependency: ' + fullname)

sys.meta_path.insert(0, RejectModelImports())
import mineru
import mineru.parser
import mineru.cli.main
"""
    process = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2], check=False
    )
    assert process.returncode == 0, process.stderr


def test_runtime_constructs_caches_and_closes_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """模型门面须缓存公开 Pipeline 句柄，并通过其 close 方法释放资源。"""
    from mineru.model.vlm import runtime

    calls: list[dict[str, Any]] = []

    class Pipeline:
        """记录工厂参数和关闭次数的引擎替身。"""

        def __init__(self, model_path: str, **kwargs: Any) -> None:
            """捕获设备配置，不触发实际 GPU 分配。"""
            calls.append({"model_path": model_path, **kwargs})
            self.closed = 0

        def close(self) -> None:
            """模拟公开的资源释放接口。"""
            self.closed += 1

    module = ModuleType("lmdeploy")
    module.PytorchEngineConfig = SimpleNamespace
    module.TurbomindEngineConfig = SimpleNamespace
    module.pipeline = Pipeline
    utilities = ModuleType("lmdeploy.utils")
    utilities.get_logger = logging.getLogger
    monkeypatch.setitem(sys.modules, "lmdeploy", module)
    monkeypatch.setitem(sys.modules, "lmdeploy.utils", utilities)
    monkeypatch.setattr(runtime, "MinerUClient", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(runtime.ModelSingleton, "_models", {})
    singleton = runtime.ModelSingleton()
    kwargs = {"lmdeploy_backend": "pytorch", "lmdeploy_device": "cuda", "cache_max_entry_count": 0.2}
    client = singleton.get_model("lmdeploy-engine", "local-model", None, **kwargs)
    assert singleton.get_model("lmdeploy-engine", "local-model", None, **kwargs) is client
    assert len(calls) == 1
    assert calls[0]["backend_config"].device_type == "cuda"
    assert calls[0]["backend_config"].cache_max_entry_count == 0.2
    assert calls[0]["log_level"] == "ERROR"
    handle = client._mineru_runtime_handles["lmdeploy_engine"]
    runtime._shutdown_runtime_handle(handle)
    assert handle.closed == 1


def test_cancelled_mlx_guard_does_not_leave_a_background_lock_owner() -> None:
    """等待模型锁时取消请求，不得有后台线程随后取得锁并永远阻塞后续请求。"""
    import asyncio
    import threading

    from mineru.model.vlm.runtime import aio_predictor_execution_guard

    lock = threading.Lock()
    predictor = SimpleNamespace(_mineru_execution_lock=lock)

    async def guarded() -> None:
        """模拟进入共享 MLX 模型的推理范围。"""
        async with aio_predictor_execution_guard(predictor):
            await asyncio.sleep(0)

    async def run() -> None:
        """先占用锁，取消等待者，再验证下一请求仍可正常进入。"""
        lock.acquire()
        task = asyncio.create_task(guarded())
        await asyncio.sleep(0.03)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            lock.release()
        await asyncio.wait_for(guarded(), timeout=1)
        assert not lock.locked()

    asyncio.run(run())


def test_cancelled_model_initialization_waits_before_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    """取消加载不能让后台线程在调用方清理之后重新写回模型缓存。"""
    import asyncio
    import threading

    from mineru.model.vlm import runtime

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def load(*args: object, **kwargs: object) -> object:
        """模拟无法被 Future.cancel 中断的模型加载。"""
        entered.set()
        assert release.wait(5)
        finished.set()
        return SimpleNamespace()

    monkeypatch.setattr(runtime.ModelSingleton, "get_model", load)

    async def run() -> None:
        """等待加载开始后取消，确认协程只有在模型加载结束后才退出。"""
        task = asyncio.create_task(runtime._get_model_async("mlx-engine", "local", None))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
        finally:
            release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert finished.is_set()

    asyncio.run(run())
