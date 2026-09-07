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
