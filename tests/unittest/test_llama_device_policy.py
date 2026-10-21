"""实际可见设备、跨平台集显决策与进程级 Vulkan 环境隔离。"""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mineru.model.runtime import llama_device as policy
from mineru.model.runtime.llama_probe import LlamaDevice, PROBE_MARKER
from mineru.model.runtime.onnx_device import OnnxSelection
from mineru.model.runtime.webgpu import WebGPUDevice
from mineru.model.runtime.windows_gpu import GpuAdapter


def gpu(vendor: int, chip: int, *, integrated: bool = True, index: int = 0) -> LlamaDevice:
    """构造有稳定物理索引的原生设备。"""
    return LlamaDevice(f"Vulkan{index}", f"GPU {vendor:04x}:{chip:04x}", "vulkan", integrated, vendor, chip, index)


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """单测不触发真实设备探测、模型加载或持久化校准。"""
    monkeypatch.setattr(policy, "sys", SimpleNamespace(platform="linux", executable="python"))
    monkeypatch.setattr(policy, "_runtime_environment", None)
    for name in (*policy._BACKEND_ENV, "MINERU_LLAMA_CPP_BACKEND"):
        monkeypatch.delenv(name, raising=False)


def inventory(monkeypatch: pytest.MonkeyPatch, devices: tuple[LlamaDevice, ...]) -> Mock:
    """模拟 ggml 可见设备过滤和过滤后 VulkanN 的重新编号。"""

    def discover(environment: dict[str, str] | None = None) -> tuple[LlamaDevice, ...]:
        if environment is None or "GGML_VK_VISIBLE_DEVICES" not in environment:
            return devices
        indices = [int(i) for i in environment["GGML_VK_VISIBLE_DEVICES"].split(",")]
        return tuple(
            replace(next(d for d in devices if d.vulkan_index == index), name=f"Vulkan{i}") for i, index in enumerate(indices)
        )

    function = Mock(side_effect=discover)
    monkeypatch.setattr(policy, "discover_llama_devices", function)
    return function


def onnx(monkeypatch: pytest.MonkeyPatch, selection: OnnxSelection) -> Mock:
    """替换共享 ONNX 决策，单独核验是否应当调用。"""
    from mineru.model.runtime import onnx_device

    function = Mock(return_value=selection)
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", function)
    return function


@pytest.mark.parametrize(
    "vendor,chip,expected",
    [
        (0x8086, 0x4680, True),
        (0x8086, 0x3E92, True),
        (0x8086, 0x4905, True),
        (0x8086, 0x7D51, False),
        (0x8086, 0x56A0, False),
        (0x8086, 0xE202, False),
        (0x8086, 0xFFFF, False),
        (0x1002, 0x4680, False),
        (0x10DE, 0x4680, False),
    ],
)
def test_explicit_architecture_mapping(vendor: int, chip: int, expected: bool) -> None:
    assert policy.needs_serialized_submissions(gpu(vendor, chip)) is expected


def test_workaround_is_linux_vulkan_only(monkeypatch: pytest.MonkeyPatch) -> None:
    intel = gpu(0x8086, 0x4680)
    assert not policy.needs_serialized_submissions(replace(intel, backend="metal"))
    monkeypatch.setattr(policy.sys, "platform", "win32")
    assert not policy.needs_serialized_submissions(intel)


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("vendor,chip", [(0x10DE, 0x2489), (0x1002, 0x73BF), (0x8086, 0x56A0)])
def test_discrete_gpu_ignores_cpu_onnx_choice(monkeypatch: pytest.MonkeyPatch, platform: str, vendor: int, chip: int) -> None:
    monkeypatch.setattr(policy.sys, "platform", platform)
    intel, discrete = gpu(0x8086, 0x4680), gpu(vendor, chip, integrated=False, index=1)
    inventory(monkeypatch, (intel, discrete))
    calibrate = onnx(monkeypatch, OnnxSelection(reason="configured_cpu"))
    selected = policy.resolve_llama_device()
    assert selected.device == "gpu" and selected.devices[0].device_id == chip
    assert dict(selected.environment) == {"GGML_VK_VISIBLE_DEVICES": "1"}
    assert selected.devices[0].name == "Vulkan0"
    calibrate.assert_not_called()


@pytest.mark.parametrize("platform,backend", [("linux", "webgpu"), ("win32", "dml")])
@pytest.mark.parametrize("decision", ["cpu", "gpu"])
def test_integrated_only_reuses_platform_decision(
    monkeypatch: pytest.MonkeyPatch, platform: str, backend: str, decision: str
) -> None:
    monkeypatch.setattr(policy.sys, "platform", platform)
    device = gpu(0x8086, 0x7D51)
    inventory(monkeypatch, (device,))
    adapter = WebGPUDevice(0x8086, 0x7D51) if backend == "webgpu" else GpuAdapter(0, "luid", "Arc", 0x8086, 0x7D51, True, 0)
    calibrate = onnx(monkeypatch, OnnxSelection(decision, adapter if decision == "gpu" else None, "rec_calibration", backend))
    result = policy.resolve_llama_device()
    assert result.device == decision
    assert policy.SERIALIZE_ENV not in dict(result.environment)
    calibrate.assert_called_once_with()


def test_old_intel_architecture_fallback_and_forced_vulkan(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory(monkeypatch, (gpu(0x8086, 0x4680),))
    calibrate = onnx(monkeypatch, OnnxSelection(reason="webgpu_architecture_policy", backend="webgpu"))
    assert policy.resolve_llama_device().device == "cpu"
    forced = policy.resolve_llama_device(backend="vulkan")
    assert forced.device == "gpu" and dict(forced.environment)[policy.SERIALIZE_ENV] == "1"
    assert calibrate.call_count == 1


def test_failed_or_mismatched_calibration_cannot_select_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory(monkeypatch, (gpu(0x1002, 0x150E),))
    onnx(monkeypatch, OnnxSelection(reason="selection_failed"))
    assert policy.resolve_llama_device().device == "cpu"
    onnx(monkeypatch, OnnxSelection("gpu", WebGPUDevice(0x10DE, 0x2489), "rec_calibration", "webgpu"))
    assert policy.resolve_llama_device().reason == "onnx_gpu_identity_unconfirmed"


def test_no_gpu_and_explicit_cpu_do_not_calibrate(monkeypatch: pytest.MonkeyPatch) -> None:
    discover = inventory(monkeypatch, ())
    calibrate = onnx(monkeypatch, OnnxSelection())
    assert policy.resolve_llama_device(n_gpu_layers=0).device == "cpu"
    discover.assert_not_called()
    assert policy.resolve_llama_device().device == "cpu"
    calibrate.assert_not_called()
    with pytest.raises(RuntimeError, match="unavailable"):
        policy.resolve_llama_device(backend="vulkan")


def test_explicit_target_on_multi_gpu_is_bound_before_workaround(monkeypatch: pytest.MonkeyPatch) -> None:
    nvidia = gpu(0x10DE, 0x2489, integrated=False)
    intel = gpu(0x8086, 0x4680, index=1)
    inventory(monkeypatch, (nvidia, intel))
    calibrate = onnx(monkeypatch, OnnxSelection())
    result = policy.resolve_llama_device(device_names=("Vulkan1",))
    assert result.devices[0].device_id == 0x4680
    assert dict(result.environment) == {policy.SERIALIZE_ENV: "1", "GGML_VK_VISIBLE_DEVICES": "1"}
    assert result.device_aliases == (("Vulkan1", "Vulkan0"),)
    calibrate.assert_not_called()


def test_old_intel_discrete_does_not_slow_other_discrete_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory(monkeypatch, (gpu(0x8086, 0x4905, integrated=False), gpu(0x10DE, 0x2489, integrated=False, index=1)))
    assert policy.resolve_llama_device().devices[0].vendor_id == 0x10DE
    with pytest.raises(ValueError, match="different Vulkan submission policies"):
        policy.resolve_llama_device(device_names=("Vulkan0", "Vulkan1"))


def test_manual_environment_is_preserved_including_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory(monkeypatch, (gpu(0x8086, 0x4680),))
    monkeypatch.setenv(policy.SERIALIZE_ENV, "0")
    result = policy.resolve_llama_device(n_gpu_layers=99)
    assert policy.SERIALIZE_ENV not in dict(result.environment)
    with policy.llama_engine_scope(result):
        assert os.environ[policy.SERIALIZE_ENV] == "0"
    assert os.environ[policy.SERIALIZE_ENV] == "0"
    assert "GGML_VK_VISIBLE_DEVICES" not in os.environ


def test_environment_restored_on_failure_and_cached_backend_change_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    inventory(monkeypatch, (gpu(0x8086, 0x4680),))
    result = policy.resolve_llama_device(n_gpu_layers=99)
    with pytest.raises(ValueError, match="constructor"):
        with policy.llama_engine_scope(result):
            assert os.environ[policy.SERIALIZE_ENV] == "1"
            raise ValueError("constructor")
    assert policy.SERIALIZE_ENV not in os.environ and "GGML_VK_VISIBLE_DEVICES" not in os.environ
    with policy.llama_engine_scope(result):
        assert os.environ[policy.SERIALIZE_ENV] == "1"
    with pytest.raises(RuntimeError, match="fresh process"):
        with policy.llama_engine_scope(replace(result, environment=(("GGML_VK_VISIBLE_DEVICES", "0"),))):
            pass
    assert policy.SERIALIZE_ENV not in os.environ


def test_binding_verification_rejects_wrong_physical_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    old, other = gpu(0x8086, 0x4680), gpu(0x10DE, 0x2489, integrated=False)
    monkeypatch.setattr(policy, "discover_llama_devices", Mock(side_effect=[(old,), (other,)]))
    with pytest.raises(RuntimeError, match="binding changed"):
        policy.resolve_llama_device(backend="vulkan")


def test_probe_protocol_uses_native_inventory_without_parsing_log_names(monkeypatch: pytest.MonkeyPatch) -> None:
    native = [
        {
            "name": "Vulkan0",
            "description": "device",
            "backend": "vulkan",
            "integrated": True,
            "vendor_id": 0x8086,
            "device_id": 0x4680,
            "vulkan_index": 1,
        }
    ]
    run = Mock(return_value=SimpleNamespace(stdout="native log\n" + PROBE_MARKER + json.dumps(native) + "\n"))
    monkeypatch.setattr(policy.subprocess, "run", run)
    assert policy.discover_llama_devices()[0].vulkan_index == 1
    assert run.call_args.kwargs["env"] == dict(os.environ)
    assert "onnxruntime" not in run.call_args.args[0]


def test_other_platforms_do_not_probe_or_calibrate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy.sys, "platform", "darwin")
    probe = inventory(monkeypatch, ())
    assert policy.resolve_llama_device().device == "native"
    probe.assert_not_called()


def test_discovery_error_is_contained(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(policy, "discover_llama_devices", Mock(side_effect=subprocess.CalledProcessError(1, "probe")))
    assert policy.resolve_llama_device().reason == "discovery_unavailable"


@pytest.mark.parametrize("forced", [False, True])
def test_real_model_factory_applies_plan_at_engine_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, forced: bool
) -> None:
    from mineru.model.vlm import runtime

    inventory(monkeypatch, (gpu(0x8086, 0x4680),))
    onnx(monkeypatch, OnnxSelection(reason="webgpu_architecture_policy", backend="webgpu"))
    captured = []

    def construct(*args: object, **kwargs: object) -> SimpleNamespace:
        captured.append((kwargs, os.environ.get(policy.SERIALIZE_ENV), os.environ.get("MINERU_LLAMA_CPP_BACKEND")))
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setitem(sys.modules, "mineru_llama_cpp", SimpleNamespace(Engine=construct))
    monkeypatch.setattr(
        runtime, "vlm_model_repo", lambda name: SimpleNamespace(paths={"main": "main.gguf", "mmproj": "mmproj.gguf"})
    )
    monkeypatch.setattr(runtime, "MinerUClient", lambda **kwargs: SimpleNamespace())
    runtime.ModelSingleton()._create_model("llama-cpp-engine", str(tmp_path), None, **({"n_gpu_layers": 99} if forced else {}))
    kwargs, serialize, backend = captured[0]
    assert kwargs["n_gpu_layers"] == (99 if forced else 0)
    assert serialize == ("1" if forced else None)
    assert backend == (None if forced else "cpu")
    assert policy.SERIALIZE_ENV not in os.environ and "MINERU_LLAMA_CPP_BACKEND" not in os.environ
