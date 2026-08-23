from __future__ import annotations

import sys
import types

import pytest

import mineru.parser.api_server as api_server


def test_preload_local_models_initializes_conditional_model_families(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str | None]] = []

    class _AtomicModel:
        TableOrientationCls = "table-orientation"
        TableCls = "table-classification"
        WirelessTable = "wireless-table"
        WiredTable = "wired-table"
        OCR = "ocr"

    class _Manager:
        def get_atom_model(self, atom_model_name: str, **kwargs: str) -> object:
            calls.append((atom_model_name, kwargs.get("lang")))
            return object()

    class _Context:
        atom_model_manager = _Manager()

    class _ContextSingleton:
        def get_model(self, lang: str, formula_enable: bool) -> _Context:
            calls.append(("context", lang if formula_enable else None))
            return _Context()

    fake_runtime = types.ModuleType("mineru.backend.local_model_runtime")
    fake_runtime.AtomicModel = _AtomicModel
    fake_runtime.HybridLocalModelContextSingleton = _ContextSingleton
    monkeypatch.setitem(sys.modules, "mineru.backend.local_model_runtime", fake_runtime)

    api_server._preload_local_models("ch")

    assert calls == [
        ("context", "ch"),
        ("table-orientation", None),
        ("table-classification", None),
        ("wireless-table", "ch"),
        ("wired-table", "ch"),
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

    from mineru.utils import engine_utils

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
