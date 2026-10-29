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

from mineru.model.runtime.gpu import classify_d3d12_gpu, with_gpu_classification
from mineru.model.runtime import llama_device as policy
from mineru.model.runtime.llama_probe import LlamaDevice, PROBE_MARKER
from mineru.model.runtime.onnx_device import OnnxSelection
from mineru.model.runtime.webgpu import WebGPUDevice
from mineru.model.runtime.windows_gpu import GpuAdapter


def gpu(vendor: int, chip: int, *, integrated: bool = True, index: int = 0) -> LlamaDevice:
    """构造有稳定物理索引的原生设备。"""
    return LlamaDevice(
        f"Vulkan{index}",
        f"GPU {vendor:04x}:{chip:04x}",
        "vulkan",
        vendor,
        chip,
        index,
        kind="integrated" if integrated else "discrete",
        kind_source="vulkan",
    )


@pytest.fixture(autouse=True)
def isolated(monkeypatch: pytest.MonkeyPatch) -> None:
    """单测不触发真实设备探测、模型加载或持久化校准。"""
    monkeypatch.setattr(policy, "sys", SimpleNamespace(platform="linux", executable="python"))
    monkeypatch.setattr(policy, "_runtime_environment", None)
    from mineru.model.runtime import device

    monkeypatch.setattr(device, "resolve_small_model_backend", lambda: "onnx")
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
        (0x8086, 0x7D51, True),
        (0x8086, 0x7D55, True),
        (0x8086, 0xB640, True),
        (0x8086, 0x4F80, False),
        (0x8086, 0x56A0, False),
        (0x8086, 0x6420, False),
        (0x8086, 0xE202, False),
        (0x8086, 0xB080, False),
        (0x8086, 0xFD80, False),
        (0x8086, 0xFFFF, False),
        (0x1002, 0x4680, False),
        (0x10DE, 0x4680, False),
        (0x1002, 0x7D51, False),
        (0x10DE, 0x7D51, False),
    ],
)
def test_explicit_architecture_mapping(vendor: int, chip: int, expected: bool) -> None:
    assert policy.needs_serialized_submissions(gpu(vendor, chip)) is expected


@pytest.mark.parametrize("chip", [0x4680, 0x7D51, 0xB640])
@pytest.mark.parametrize("platform", ["win32", "darwin"])
def test_workaround_is_linux_vulkan_only(monkeypatch: pytest.MonkeyPatch, chip: int, platform: str) -> None:
    intel = gpu(0x8086, chip)
    assert not policy.needs_serialized_submissions(replace(intel, backend="metal"))
    monkeypatch.setattr(policy.sys, "platform", platform)
    assert not policy.needs_serialized_submissions(intel)


def test_missing_pci_identity_does_not_guess_from_name() -> None:
    intel = gpu(0x8086, 0x7D51)
    assert not policy.needs_serialized_submissions(replace(intel, device_id=None, description="Intel Arc Xe-LPG"))
    assert not policy.needs_serialized_submissions(replace(intel, vendor_id=None))


@pytest.mark.parametrize("platform", ["linux", "win32"])
@pytest.mark.parametrize("vendor,chip", [(0x10DE, 0x2489), (0x1002, 0x73BF), (0x8086, 0x56A0)])
@pytest.mark.parametrize("igpu_chip", [0x4680, 0x7D51])
def test_discrete_gpu_ignores_cpu_onnx_choice(
    monkeypatch: pytest.MonkeyPatch, platform: str, vendor: int, chip: int, igpu_chip: int
) -> None:
    monkeypatch.setattr(policy.sys, "platform", platform)
    intel, discrete = gpu(0x8086, igpu_chip), gpu(vendor, chip, integrated=False, index=1)
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
    adapter = (
        WebGPUDevice(0x8086, 0x7D51)
        if backend == "webgpu"
        else with_gpu_classification(GpuAdapter(0, "luid", "Arc", 0x8086, 0x7D51, 0), classify_d3d12_gpu(True))
    )
    calibrate = onnx(monkeypatch, OnnxSelection(decision, adapter if decision == "gpu" else None, "rec_calibration", backend))
    result = policy.resolve_llama_device()
    assert result.device == decision
    expected = "1" if platform == "linux" and decision == "gpu" else None
    assert dict(result.environment).get(policy.SERIALIZE_ENV) == expected
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


def test_unselected_backend_discrete_does_not_exempt_integrated(monkeypatch: pytest.MonkeyPatch) -> None:
    """其他后端的独显不在自动候选中时，实际选中的 Vulkan 集显仍需复用校准决策。"""
    unavailable = replace(gpu(0x10DE, 1, integrated=False), backend="other")
    integrated = gpu(0x1002, 0x150E, index=1)
    inventory(monkeypatch, (unavailable, integrated))
    calibrate = onnx(monkeypatch, OnnxSelection(reason="configured_cpu"))
    result = policy.resolve_llama_device()
    assert result.device == "cpu" and result.reason == "onnx_configured_cpu"
    calibrate.assert_called_once()


def test_no_gpu_and_explicit_cpu_do_not_calibrate(monkeypatch: pytest.MonkeyPatch) -> None:
    discover = inventory(monkeypatch, ())
    calibrate = onnx(monkeypatch, OnnxSelection())
    assert policy.resolve_llama_device(n_gpu_layers=0).device == "cpu"
    discover.assert_not_called()
    assert policy.resolve_llama_device().device == "cpu"
    calibrate.assert_not_called()
    with pytest.raises(RuntimeError, match="unavailable"):
        policy.resolve_llama_device(backend="vulkan")


@pytest.mark.parametrize("chip", [0x4680, 0x7D51, 0xB640])
@pytest.mark.parametrize("other_vendor,other_chip", [(0x10DE, 0x2489), (0x1002, 0x73BF), (0x8086, 0x56A0)])
def test_explicit_target_on_multi_gpu_is_bound_before_workaround(
    monkeypatch: pytest.MonkeyPatch, chip: int, other_vendor: int, other_chip: int
) -> None:
    discrete = gpu(other_vendor, other_chip, integrated=False)
    intel = gpu(0x8086, chip, index=1)
    inventory(monkeypatch, (discrete, intel))
    calibrate = onnx(monkeypatch, OnnxSelection())
    result = policy.resolve_llama_device(device_names=("Vulkan1",))
    assert result.devices[0].device_id == chip
    assert dict(result.environment) == {policy.SERIALIZE_ENV: "1", "GGML_VK_VISIBLE_DEVICES": "1"}
    assert result.device_aliases == (("Vulkan1", "Vulkan0"),)
    calibrate.assert_not_called()


@pytest.mark.parametrize("chip", [0x4905, 0x7D51])
@pytest.mark.parametrize("other_vendor,other_chip", [(0x10DE, 0x2489), (0x8086, 0x56A0)])
def test_mixed_submission_policies_do_not_slow_other_gpu(
    monkeypatch: pytest.MonkeyPatch, chip: int, other_vendor: int, other_chip: int
) -> None:
    inventory(
        monkeypatch,
        (gpu(0x8086, chip, integrated=chip == 0x7D51), gpu(other_vendor, other_chip, integrated=False, index=1)),
    )
    selected = policy.resolve_llama_device()
    assert selected.devices[0].device_id == other_chip
    assert policy.SERIALIZE_ENV not in dict(selected.environment)
    with pytest.raises(ValueError, match="different Vulkan submission policies"):
        policy.resolve_llama_device(device_names=("Vulkan0", "Vulkan1"))


@pytest.mark.parametrize("chip", [0x4680, 0x7D51, 0x56A0])
def test_manual_environment_is_preserved_including_zero(monkeypatch: pytest.MonkeyPatch, chip: int) -> None:
    inventory(monkeypatch, (gpu(0x8086, chip, integrated=chip != 0x56A0),))
    monkeypatch.setenv(policy.SERIALIZE_ENV, "0")
    result = policy.resolve_llama_device(n_gpu_layers=99)
    assert policy.SERIALIZE_ENV not in dict(result.environment)
    with policy.llama_engine_scope(result):
        assert os.environ[policy.SERIALIZE_ENV] == "0"
    assert os.environ[policy.SERIALIZE_ENV] == "0"
    assert "GGML_VK_VISIBLE_DEVICES" not in os.environ


@pytest.mark.parametrize("chip", [0x4680, 0x7D51])
def test_environment_restored_on_failure_and_cached_backend_change_rejected(monkeypatch: pytest.MonkeyPatch, chip: int) -> None:
    inventory(monkeypatch, (gpu(0x8086, chip),))
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


@pytest.mark.parametrize("chip", [0x4680, 0x7D51])
def test_binding_verification_rejects_wrong_physical_gpu(monkeypatch: pytest.MonkeyPatch, chip: int) -> None:
    intel, other = gpu(0x8086, chip), gpu(0x10DE, 0x2489, integrated=False)
    monkeypatch.setattr(policy, "discover_llama_devices", Mock(side_effect=[(intel,), (other,)]))
    with pytest.raises(RuntimeError, match="binding changed"):
        policy.resolve_llama_device(backend="vulkan")


@pytest.mark.parametrize("chip", [0x4680, 0x7D51])
def test_missing_device_index_cannot_apply_global_workaround(monkeypatch: pytest.MonkeyPatch, chip: int) -> None:
    inventory(monkeypatch, (replace(gpu(0x8086, chip), vulkan_index=None),))
    with pytest.raises(RuntimeError, match="Cannot safely bind"):
        policy.resolve_llama_device(backend="vulkan")
    assert policy.SERIALIZE_ENV not in os.environ


def test_probe_protocol_uses_native_inventory_without_parsing_log_names(monkeypatch: pytest.MonkeyPatch) -> None:
    native = [
        {
            "name": "Vulkan0",
            "description": "device",
            "backend": "vulkan",
            "kind": "integrated",
            "kind_source": "vulkan",
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


@pytest.mark.parametrize(
    "chip,forced,calibrated_gpu",
    [(0x4680, False, False), (0x4680, True, False), (0x7D51, False, False), (0x7D51, True, False), (0x7D51, False, True)],
)
def test_real_model_factory_applies_plan_at_engine_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, chip: int, forced: bool, calibrated_gpu: bool
) -> None:
    from mineru.model.vlm import runtime

    inventory(monkeypatch, (gpu(0x8086, chip),))
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
    runtime.ModelSingleton()._create_model("llama-cpp-engine", str(tmp_path), None, **({"n_gpu_layers": 99} if forced else {}))
    kwargs, serialize, backend = captured[0]
    use_gpu = forced or calibrated_gpu
    assert kwargs.get("n_gpu_layers") == (99 if forced else None if use_gpu else 0)
    assert serialize == ("1" if use_gpu else None)
    assert backend == (None if use_gpu else "cpu")
    assert policy.SERIALIZE_ENV not in os.environ and "MINERU_LLAMA_CPP_BACKEND" not in os.environ
