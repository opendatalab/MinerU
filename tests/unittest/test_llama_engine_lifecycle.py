"""使用实际 llama.cpp 适配器验证共享运行时接入，不再维护后端死亡判断。"""

import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from mineru_vl_utils.vlm_client.base_client import EngineUnavailableError
from PIL import Image

from mineru.model.runtime.ownership import RuntimeOwner
from mineru.model.vlm import runtime
from mineru.model.vlm.errors import EngineDeadError

native = pytest.importorskip("mineru_llama_cpp")


def test_factory_binds_llama_health_and_notifies_all_owners(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """真实工厂按能力声明绑定探测，空闲死亡通知所有 owner 并拒绝缓存复用。"""
    core = SimpleNamespace(check_health=MagicMock(), generate=MagicMock())

    def initialize(self: native.Engine, *args: object, **kwargs: object) -> None:
        """仅替换加载模型的步骤，其余使用公开 Engine 与生产适配器。"""
        self._closed = False
        self._close_lock = threading.Lock()
        self._core = core
        self._eos_token_str = ""

    monkeypatch.setattr(native.Engine, "__init__", initialize)
    monkeypatch.setattr(runtime, "vlm_model_repo", lambda name: SimpleNamespace(paths={"main": "model", "mmproj": "projector"}))
    pool = runtime.ModelSingleton()
    handle = pool._create_model("llama-cpp-engine", str(tmp_path), None, n_gpu_layers=0)
    owners = [RuntimeOwner(), RuntimeOwner()]
    handle.owners.update(owners)
    key = ("llama-lifecycle",)
    monkeypatch.setattr(runtime.ModelSingleton, "_models", {key: handle})
    original = native.EngineUnavailableError("native loop exited", code="engine_dead", details={"native_type": "logic_error"})
    try:
        assert handle.supports_health_probe
        handle.probe_health()
        core.check_health.side_effect = original
        with pytest.raises(EngineUnavailableError) as captured:
            pool.probe_owner_health(owners[0])
        assert captured.value.__cause__ is original
        assert all(owner.engine_error == "native loop exited" for owner in owners)
        core.check_health.side_effect = None
        with pytest.raises(EngineDeadError):
            handle.batch_recognize_text([Image.new("RGB", (64, 64))])
        factory = MagicMock()
        with pytest.raises(EngineDeadError):
            pool._get_or_initialize(key, factory, native_async=False, backend="llama-cpp-engine", max_concurrency=1)
        factory.assert_not_called()
        core.generate.assert_not_called()
        assert core.check_health.call_count == 2
    finally:
        handle.shutdown()
