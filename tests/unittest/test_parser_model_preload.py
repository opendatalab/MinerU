from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

import mineru.parser.api_server as api_server
from mineru.config import VlmConfig


def test_preload_local_models_initializes_conditional_model_families(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    class _Manager:
        def get_atom_model(self, atom_model_name: str, **kwargs: str) -> object:
            calls.append((atom_model_name, kwargs.get("lang")))
            return object()

    class _Context:
        atom_model_manager = _Manager()

    class _ContextSingleton:
        def get_model(self) -> _Context:
            calls.append(("context", None))
            return _Context()

    fake_runtime = types.ModuleType("mineru.model.runtime.hybrid")
    fake_runtime.HybridLocalModelContextSingleton = _ContextSingleton
    monkeypatch.setitem(sys.modules, "mineru.model.runtime.hybrid", fake_runtime)

    api_server._preload_local_models("ch")

    assert calls == [
        ("context", None),
        ("table_ori_cls", None),
        ("table_cls", None),
        ("wireless_table", "ch"),
        ("wired_table", "ch"),
        ("ocr", "seal"),
    ]


def test_preload_basic_models_initializes_only_local_models(monkeypatch: pytest.MonkeyPatch) -> None:
    local_calls: list[str] = []
    monkeypatch.setattr(api_server, "_preload_local_models", local_calls.append)

    result = api_server._preload_server_models("basic", language="ch")

    assert result == api_server._ModelPreloadResult(tier="basic", engine="hybrid-local")
    assert local_calls == ["ch"]


def test_preload_standard_models_initializes_platform_engine_and_local_models(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[object] = []
    monkeypatch.setattr(api_server, "_preload_local_models", lambda language: calls.append(("local", language)))

    from mineru.model.vlm import selector as engine_utils

    monkeypatch.setattr(engine_utils, "get_vlm_engine", lambda inference_engine, is_async=False: "lmdeploy-engine")

    fake_runtime = types.ModuleType("mineru.model.vlm.runtime")

    class _ModelSingleton:
        def get_model(self, backend: str, model_path: str | None, server_url: str | None) -> object:
            calls.append(("vlm", backend, model_path, server_url))
            return object()

    fake_runtime.ModelSingleton = _ModelSingleton
    monkeypatch.setitem(sys.modules, "mineru.model.vlm.runtime", fake_runtime)

    result = api_server._preload_server_models("standard", language="en")

    assert result == api_server._ModelPreloadResult(tier="standard", engine="lmdeploy-engine")
    assert calls == [("vlm", "lmdeploy-engine", None, None), ("local", "en")]


def test_local_preload_and_parse_reuse_llama_predictor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """本地预加载与解析共用同步引擎缓存，llama 参数循环不能覆盖模型缓存键。"""
    from unittest.mock import MagicMock

    from mineru.model.vlm import runtime, selector
    from mineru.model.vlm.client import get_vlm_predictor

    engine_factory = MagicMock()
    predictor_factory = MagicMock()
    engine_module = types.ModuleType("mineru_llama_cpp")
    engine_module.Engine = engine_factory
    monkeypatch.setitem(sys.modules, "mineru_llama_cpp", engine_module)
    monkeypatch.setattr(runtime.ModelSingleton, "_models", {})
    monkeypatch.setattr(runtime, "MinerUClient", predictor_factory)
    monkeypatch.setattr(type(runtime.MINERU_2_5_PRO_2605_1_2B), "ensure", lambda self: tmp_path)
    engine_selector = MagicMock(return_value="llama-cpp-engine")
    monkeypatch.setattr(selector, "get_vlm_engine", engine_selector)
    monkeypatch.setattr(api_server, "_preload_local_models", lambda language: None)

    api_server._preload_server_models("standard", language="ch", vlm_config=VlmConfig())
    predictor, engine = get_vlm_predictor(VlmConfig())
    assert predictor is predictor_factory.return_value
    assert engine == "llama-cpp-engine"
    predictor_factory.assert_called_once()
    engine_factory.assert_called_once()
    assert all(call.kwargs == {"is_async": False} for call in engine_selector.call_args_list)


@pytest.mark.parametrize(
    ("exc", "expected_code"),
    [
        (ModuleNotFoundError("missing dependency"), "model_preload_dependency_missing"),
        (FileNotFoundError("missing weights"), "model_preload_files_missing"),
        (ValueError("CUDA is not available."), "model_preload_device_unavailable"),
        (RuntimeError("engine boot failed"), "model_preload_failed"),
    ],
)
def test_classify_model_preload_error(exc: Exception, expected_code: str) -> None:
    code, message = api_server._classify_model_preload_error(exc)

    assert code == expected_code
    assert message == str(exc)
