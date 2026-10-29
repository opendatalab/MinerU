"""小模型后端统一决定 CPU/GPU；llama 不枚举、匹配或绑定显卡。"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mineru.config import get_config
from mineru.model.runtime import device, llama_device as policy, llama_probe, onnx_device
from mineru.model.runtime.device import TorchDeviceSelection
from mineru.model.runtime.onnx_device import OnnxSelection
from mineru.model.runtime.webgpu import WebGPUDevice


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(policy, "_runtime_environment", None)
    monkeypatch.setattr(device, "resolve_small_model_backend", lambda: "onnx")
    for name in (*policy._BACKEND_ENV, "MINERU_LLAMA_CPP_BACKEND"):
        monkeypatch.delenv(name, raising=False)
    forbidden = Mock(side_effect=AssertionError("llama runtime must not probe devices"))
    monkeypatch.setattr(llama_probe, "probe_devices", forbidden)
    monkeypatch.setattr(llama_probe, "_probe_library", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


def onnx(monkeypatch: pytest.MonkeyPatch, selection: OnnxSelection) -> Mock:
    function = Mock(return_value=selection)
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", function)
    return function


@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize(
    "decision,reason",
    [
        ("gpu", "discrete_gpu"),
        ("gpu", "rec_calibration"),
        ("gpu", "configured_gpu"),
        ("cpu", "rec_calibration"),
        ("cpu", "configured_cpu"),
        ("cpu", "no_hardware_gpu"),
        ("cpu", "selection_failed"),
        ("cpu", "webgpu_architecture_policy"),
    ],
)
def test_automatic_route_consumes_onnx_result(
    monkeypatch: pytest.MonkeyPatch, platform: str, decision: str, reason: str
) -> None:
    monkeypatch.setattr(policy.sys, "platform", platform)
    # CPU/GPU 结论本身即可驱动 llama；不要求携带或匹配另一套显卡库存。
    select = onnx(monkeypatch, OnnxSelection(decision, reason=reason))
    result = policy.resolve_llama_device()
    assert result == policy.LlamaSelection(decision, reason=f"onnx_{reason}")
    select.assert_called_once_with()


def test_repeated_resolution_reuses_onnx_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(get_config().model, "onnx_device", "gpu")
    select = Mock(return_value=OnnxSelection("gpu", reason="configured_gpu"))
    monkeypatch.setattr(onnx_device, "_select_and_save", select)
    assert policy.resolve_llama_device() == policy.resolve_llama_device()
    select.assert_called_once()


def test_offline_resolution_preserves_download_constraint(monkeypatch: pytest.MonkeyPatch) -> None:
    select = onnx(monkeypatch, OnnxSelection(reason="selection_failed"))
    assert policy.resolve_llama_device(allow_download=False).device == "cpu"
    select.assert_called_once_with(allow_download=False)


@pytest.mark.parametrize("accelerated", [False, True])
def test_torch_decision_does_not_use_onnx_or_intel_workaround(monkeypatch: pytest.MonkeyPatch, accelerated: bool) -> None:
    monkeypatch.setattr(device, "resolve_small_model_backend", lambda: "torch")
    monkeypatch.setattr(
        device, "resolve_torch_device", lambda: TorchDeviceSelection("xpu" if accelerated else "cpu", "torch_result")
    )
    select = onnx(monkeypatch, OnnxSelection())
    result = policy.resolve_llama_device()
    assert result == policy.LlamaSelection("gpu" if accelerated else "cpu", reason="torch_result")
    select.assert_not_called()


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize(
    "vendor,chip,affected",
    [
        (0x8086, 0x4680, True),
        (0x8086, 0x3E92, True),
        (0x8086, 0x7D51, True),
        (0x8086, 0xB640, True),
        (0x8086, 0x56A0, False),
        (0x8086, 0x64A0, False),
        (0x8086, 0xFFFF, False),
        (0x10DE, 0x7D51, False),
    ],
)
def test_intel_compatibility_uses_existing_onnx_adapter(
    monkeypatch: pytest.MonkeyPatch, platform: str, vendor: int, chip: int, affected: bool
) -> None:
    monkeypatch.setattr(policy.sys, "platform", platform)
    onnx(monkeypatch, OnnxSelection("gpu", WebGPUDevice(vendor, chip), "rec_calibration", "webgpu"))
    result = policy.resolve_llama_device()
    assert result.environment == (((policy.SERIALIZE_ENV, "1"),) if platform == "linux" and affected else ())
    assert "GGML_VK_VISIBLE_DEVICES" not in dict(result.environment)


@pytest.mark.parametrize("value", ["0", "1"])
def test_existing_serialization_setting_wins(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv(policy.SERIALIZE_ENV, value)
    onnx(monkeypatch, OnnxSelection("gpu", WebGPUDevice(0x8086, 0x7D51), "configured_gpu", "webgpu"))
    result = policy.resolve_llama_device()
    assert not result.environment
    with policy.llama_engine_scope(result):
        assert os.environ[policy.SERIALIZE_ENV] == value
    assert os.environ[policy.SERIALIZE_ENV] == value


@pytest.mark.parametrize(
    "backend,layers,names,expected",
    [
        ("auto", 0, None, "cpu"),
        ("cpu", None, None, "cpu"),
        ("auto", None, ("none",), "cpu"),
        ("vulkan", None, None, "native"),
        ("metal", None, None, "native"),
        ("auto", 99, None, "native"),
        ("auto", -1, None, "native"),
        ("auto", None, ("Vulkan99",), "native"),
    ],
)
def test_explicit_overrides_never_select_or_validate_inventory(
    monkeypatch: pytest.MonkeyPatch, backend: str, layers: int | None, names: tuple[str, ...] | None, expected: str
) -> None:
    monkeypatch.setenv("MINERU_LLAMA_CPP_BACKEND", backend)
    select = onnx(monkeypatch, OnnxSelection())
    result = policy.resolve_llama_device(n_gpu_layers=layers, device_names=names)
    assert result.device == expected and not result.environment
    select.assert_not_called()


def test_invalid_backend_is_rejected_even_with_cpu_layers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINERU_LLAMA_CPP_BACKEND", "invalid")
    with pytest.raises(ValueError, match="MINERU_LLAMA_CPP_BACKEND"):
        policy.resolve_llama_device(n_gpu_layers=0)


@pytest.mark.parametrize("platform", ["darwin", "other"])
def test_other_platforms_keep_native_routing(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    monkeypatch.setattr(policy.sys, "platform", platform)
    select = onnx(monkeypatch, OnnxSelection())
    assert policy.resolve_llama_device() == policy.LlamaSelection(reason="platform_native")
    select.assert_not_called()


def test_explicit_native_initialization_waits_for_temporary_cpu_environment() -> None:
    """自动 CPU 的临时后端变量不能让并发的手动 GPU 构造落到 CPU。"""
    attempted, entered = threading.Event(), threading.Event()

    def initialize_native() -> None:
        attempted.set()
        with policy.llama_engine_scope(policy.LlamaSelection(reason="explicit_gpu")):
            assert "MINERU_LLAMA_CPP_BACKEND" not in os.environ
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with policy.llama_engine_scope(policy.LlamaSelection("cpu")):
            future = executor.submit(initialize_native)
            assert attempted.wait(3)
            assert not entered.wait(0.05)
        future.result(timeout=3)
    assert entered.is_set()


def test_environment_restored_and_cached_vulkan_setting_cannot_change() -> None:
    result = policy.LlamaSelection("gpu", environment=((policy.SERIALIZE_ENV, "1"),))
    with pytest.raises(RuntimeError, match="constructor"):
        with policy.llama_engine_scope(result):
            assert os.environ[policy.SERIALIZE_ENV] == "1"
            raise RuntimeError("constructor")
    assert policy.SERIALIZE_ENV not in os.environ
    with policy.llama_engine_scope(result):
        assert os.environ[policy.SERIALIZE_ENV] == "1"
    with pytest.raises(RuntimeError, match="fresh process"):
        with policy.llama_engine_scope(replace(result, environment=())):
            pass
    assert policy.SERIALIZE_ENV not in os.environ


@pytest.mark.parametrize(
    "chip,forced,calibrated_gpu",
    [(0x4680, False, False), (0x4680, True, False), (0x7D51, False, False), (0x7D51, True, False), (0x7D51, False, True)],
)
@pytest.mark.parametrize("threads", [None, -1, 6])
def test_real_model_factory_applies_plan_at_engine_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chip: int, forced: bool, calibrated_gpu: bool, threads: int | None
) -> None:
    from mineru.model.vlm import runtime

    choice = (
        OnnxSelection("gpu", WebGPUDevice(0x8086, chip), "rec_calibration", "webgpu")
        if calibrated_gpu
        else OnnxSelection(reason="configured_cpu", backend="webgpu")
    )
    onnx(monkeypatch, choice)
    captured = []

    def construct(*args: object, **kwargs: object) -> SimpleNamespace:
        captured.append((kwargs, os.environ.get(policy.SERIALIZE_ENV), os.environ.get("MINERU_LLAMA_CPP_BACKEND")))
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setitem(sys.modules, "mineru_llama_cpp", SimpleNamespace(Engine=construct))
    monkeypatch.setattr(
        runtime, "vlm_model_repo", lambda name: SimpleNamespace(paths={"main": "main.gguf", "mmproj": "mmproj.gguf"})
    )
    monkeypatch.setattr(runtime, "MinerUClient", lambda **kwargs: SimpleNamespace())
    options = {"n_gpu_layers": 99} if forced else {}
    if threads is not None:
        options["n_threads"] = threads
    runtime.ModelSingleton()._create_model("llama-cpp-engine", str(tmp_path), None, **options)
    kwargs, serialize, backend = captured[0]
    use_gpu = forced or calibrated_gpu
    assert kwargs.get("n_gpu_layers") == (99 if forced else None if use_gpu else 0)
    assert kwargs.get("n_threads") == (threads if threads is not None else 4 if use_gpu else None)
    assert serialize == ("1" if calibrated_gpu and not forced else None)
    assert backend == (None if use_gpu else "cpu")
    assert policy.SERIALIZE_ENV not in os.environ and "MINERU_LLAMA_CPP_BACKEND" not in os.environ
