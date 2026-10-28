"""Intel architecture gates and actual Vulkan restrictions on mixed GPU hosts."""

from __future__ import annotations

import ctypes as ct
from dataclasses import asdict, replace
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mineru.model.runtime.gpu import classify_d3d12_gpu, classify_vulkan_gpu, with_gpu_classification
from mineru.model.runtime import intel_gpu, onnx, onnx_device, vulkan, webgpu


@pytest.fixture(autouse=True)
def fresh_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离运行时和宿主 Vulkan 驱动，让各用例显式声明可见设备。"""
    monkeypatch.setattr(webgpu, "_runtime_pid", None)
    monkeypatch.setattr(webgpu, "_runtime_device", None)
    monkeypatch.setattr(webgpu, "sys", SimpleNamespace(platform="linux"))


@pytest.mark.parametrize(
    "device_id",
    [
        0x0126,
        0x0412,
        0x1616,
        0x1912,
        0x5912,
        0x3E92,
        0x9BC5,
        0x8A52,
        0x4E61,
        0x9A49,
        0x4C8A,
        0x4680,
        0x4690,
        0x4688,
        0x46A6,
        0xA780,
        0xA7A0,
        0x46D1,
        0x46D4,
        0x4905,
    ],
)
def test_old_intel_architectures_are_cpu_only(device_id: int) -> None:
    assert intel_gpu.intel_architecture(device_id) in {"legacy", "xe_lp"}
    assert not intel_gpu.supports_webgpu_architecture(0x8086, device_id)


@pytest.mark.parametrize(
    ("device_id", "architecture"),
    [
        (0x4F80, "xe_hpg"),
        (0x56A0, "xe_hpg"),
        (0x5693, "xe_hpg"),
        (0x7D51, "xe_lpg"),
        (0x7D55, "xe_lpg"),
        (0x7D67, "xe_lpg"),
        (0x7D41, "xe_lpg"),
        (0xB640, "xe_lpg"),
        (0x64A0, "xe2_lpg"),
        (0xE20B, "xe2_hpg"),
        (0xB080, "xe3_lpg"),
        (0xFD80, "xe3_lpg_xs"),
    ],
)
def test_supported_intel_architectures_continue_calibration(device_id: int, architecture: str) -> None:
    assert intel_gpu.intel_architecture(device_id) == architecture
    assert intel_gpu.supports_webgpu_architecture(0x8086, device_id)


@pytest.mark.parametrize("device_id", [-1, 0x10000, 0xFFFF, 0x1234])
def test_unknown_intel_is_not_guessed_from_numeric_age(device_id: int) -> None:
    assert intel_gpu.intel_architecture(device_id) is None
    assert not intel_gpu.supports_webgpu_architecture(0x8086, device_id)


@pytest.mark.parametrize("vendor", [0x1002, 0x10DE])
def test_amd_nvidia_ids_never_use_intel_mapping(vendor: int, monkeypatch: pytest.MonkeyPatch) -> None:
    """AMD/NVIDIA 即使使用相同设备号，也按 Vulkan 类型而非 Intel 架构表分类。"""
    descriptor = webgpu.WebGPUDevice(vendor, 0x4680)
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        lambda: (with_gpu_classification(vulkan.VulkanDevice(vendor, 0x4680, "GPU"), classify_vulkan_gpu(1)),),
    )
    assert intel_gpu.supports_webgpu_architecture(vendor, descriptor.device_id)
    assert webgpu.select_webgpu_device((descriptor,)) == with_gpu_classification(descriptor, classify_vulkan_gpu(1))


@pytest.mark.parametrize("requested", ["auto", "gpu"])
@pytest.mark.parametrize("device_id", [0x4680, 0xA780, 0x9A49, 0x4905, 0xFFFF])
def test_blocked_policy_never_downloads_measures_or_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, requested: str, device_id: int
) -> None:
    from mineru.model.runtime import onnx_calibration
    from mineru.model import registry

    settings = importlib.import_module("mineru.config")
    path = tmp_path / "config.yaml"
    path.write_text(f"model:\n  onnx_device: {requested}\n  source: local\n")
    before = path.read_bytes()
    monkeypatch.setattr(settings.get_config().model, "onnx_device", requested)
    monkeypatch.setattr(settings, "get_config_file_path", lambda: str(path))
    monkeypatch.setattr(settings, "get_config_source", lambda _: "file")
    monkeypatch.setattr(onnx_device, "_selections", {})
    monkeypatch.setattr(onnx_device, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(onnx_device, "discover_webgpu_devices", lambda: (webgpu.WebGPUDevice(0x8086, device_id),))
    download = Mock(side_effect=AssertionError("download"))
    monkeypatch.setattr(registry, "MINERU_4_MODELS_ONNX", SimpleNamespace(ocr_rec=SimpleNamespace(ensure=download)))
    measure = Mock(side_effect=AssertionError("calibration"))
    monkeypatch.setattr(onnx_calibration, "run_calibration", measure)
    selected = onnx_device.resolve_onnx_device()
    assert selected.device == "cpu" and selected.reason == "webgpu_architecture_policy"
    assert path.read_bytes() == before
    assert onnx_device.resolve_onnx_device() is selected
    download.assert_not_called()
    measure.assert_not_called()


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("eligible", [(0x10DE, 0x2489), (0x1002, 0x150E), (0x8086, 0x56A0)])
def test_mixed_host_selects_and_binds_eligible_gpu(
    monkeypatch: pytest.MonkeyPatch, reverse: bool, eligible: tuple[int, int]
) -> None:
    old = webgpu.WebGPUDevice(0x8086, 0x4680)
    good = webgpu.WebGPUDevice(*eligible)
    devices = (good, old) if reverse else (old, good)
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        lambda: (
            with_gpu_classification(vulkan.VulkanDevice(old.vendor_id, old.device_id, "old"), classify_vulkan_gpu(1)),
            with_gpu_classification(vulkan.VulkanDevice(*eligible, "discrete"), classify_vulkan_gpu(2)),
        ),
    )
    selected = webgpu.select_webgpu_device(devices)
    assert selected == with_gpu_classification(webgpu.WebGPUDevice(*eligible, vulkan_device=eligible), classify_vulkan_gpu(2))


@pytest.mark.parametrize("include_old", [False, True])
def test_icd_exclusion_cannot_be_bypassed_by_ort_inventory(monkeypatch: pytest.MonkeyPatch, include_old: bool) -> None:
    """即使 ORT 只报告独显，也不能绕过 Vulkan ICD 对该设备的隐藏。"""
    old = webgpu.WebGPUDevice(0x8086, 0x4680)
    good = webgpu.WebGPUDevice(0x10DE, 0x2489)
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        lambda: (with_gpu_classification(vulkan.VulkanDevice(0x8086, 0x4680, "Intel-only ICD"), classify_vulkan_gpu(1)),),
    )
    assert webgpu.select_webgpu_device((good, old) if include_old else (good,)) is None


@pytest.mark.parametrize("include_old", [False, True])
def test_multiple_good_candidates_prefer_discrete_gpu(monkeypatch: pytest.MonkeyPatch, include_old: bool) -> None:
    """全部架构受支持的多卡机器也应独显优先，不依赖 ORT 的枚举顺序。"""
    old, integrated, discrete = (
        webgpu.WebGPUDevice(0x8086, 0x4680),
        webgpu.WebGPUDevice(0x1002, 0x150E),
        webgpu.WebGPUDevice(0x10DE, 0x2489),
    )
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        lambda: (
            with_gpu_classification(vulkan.VulkanDevice(0x1002, 0x150E, "AMD integrated"), classify_vulkan_gpu(1)),
            with_gpu_classification(vulkan.VulkanDevice(0x10DE, 0x2489, "NVIDIA"), classify_vulkan_gpu(2)),
            with_gpu_classification(vulkan.VulkanDevice(0x8086, 0x4680, "old Intel"), classify_vulkan_gpu(1)),
        ),
    )
    selected = webgpu.select_webgpu_device((old, integrated, discrete) if include_old else (integrated, discrete))
    assert selected == with_gpu_classification(replace(discrete, vulkan_device=(0x10DE, 0x2489)), classify_vulkan_gpu(2))


@pytest.mark.parametrize("duplicate", ["vulkan", "ort"])
def test_ambiguous_device_identity_does_not_confirm_discrete(monkeypatch: pytest.MonkeyPatch, duplicate: str) -> None:
    """同型号多卡或重复 EP 注册无法唯一对应时，不授予独显免校准资格。"""
    descriptor = webgpu.WebGPUDevice(0x10DE, 0x2489)
    physical = with_gpu_classification(vulkan.VulkanDevice(0x10DE, 0x2489, "GPU"), classify_vulkan_gpu(2))
    monkeypatch.setattr(
        vulkan, "enumerate_vulkan_devices", lambda: (physical, physical) if duplicate == "vulkan" else (physical,)
    )
    selected = webgpu.select_webgpu_device((descriptor, descriptor) if duplicate == "ort" else (descriptor,))
    assert selected is not None and selected.kind == "unknown"


@pytest.mark.parametrize("failure", [OSError("loader unavailable"), RuntimeError("enumeration failed")])
def test_unknown_vulkan_type_keeps_calibration_eligible(monkeypatch: pytest.MonkeyPatch, failure: Exception) -> None:
    """没有受限架构时，原生类型查询失败只撤销免校准资格，不猜测独显。"""
    descriptor = webgpu.WebGPUDevice(0x10DE, 0x2489, (("type", "discrete"),))
    monkeypatch.setattr(vulkan, "enumerate_vulkan_devices", Mock(side_effect=failure))
    assert webgpu.select_webgpu_device((descriptor,)) == descriptor
    with pytest.raises(type(failure)):
        webgpu.select_webgpu_device((descriptor, webgpu.WebGPUDevice(0x8086, 0x4680)))


def test_windows_dml_does_not_apply_linux_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from mineru.model.runtime import onnx_calibration
    from mineru.model.runtime.windows_gpu import GpuAdapter
    from mineru.model import registry

    monkeypatch.setattr(onnx_device, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(onnx.ort, "get_available_providers", lambda: ["DmlExecutionProvider"])
    monkeypatch.setattr(
        onnx_device,
        "discover_gpus",
        lambda **_: (with_gpu_classification(GpuAdapter(0, "id", "UHD770", 0x8086, 0x4680, 0), classify_d3d12_gpu(True)),),
    )
    monkeypatch.setattr(onnx_device, "select_webgpu_device", Mock(side_effect=AssertionError("Linux policy")))
    monkeypatch.setattr(registry, "MINERU_4_MODELS_ONNX", SimpleNamespace(ocr_rec=SimpleNamespace(ensure=lambda: "rec.onnx")))
    measure = Mock(return_value={"decision_valid": True, "cpu_elapsed": 1, "gpu_elapsed": 0.5})
    monkeypatch.setattr(onnx_calibration, "run_calibration", measure)
    selected, completed = onnx_device._select("auto", 4, 1)
    assert selected.device == "gpu" and selected.backend == "dml" and completed
    measure.assert_called_once()


@pytest.mark.parametrize("device_id", [0x7D51, 0x7D55, 0x56A0, 0x64A0, 0xE20B, 0xB080])
@pytest.mark.parametrize("device_type", [1, 2, None])
def test_modern_intel_auto_calibrates_according_to_vulkan_type(
    monkeypatch: pytest.MonkeyPatch, device_id: int, device_type: int | None
) -> None:
    """架构白名单只决定兼容性，是否免校准由实际 Vulkan 类型独立决定。"""
    from mineru.model.runtime import onnx_calibration
    from mineru.model import registry

    descriptor = webgpu.WebGPUDevice(0x8086, device_id)
    monkeypatch.setattr(onnx_device, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(onnx_device, "discover_webgpu_devices", lambda: (descriptor,))
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        Mock(
            return_value=(
                with_gpu_classification(
                    vulkan.VulkanDevice(0x8086, device_id, "Intel GPU"), classify_vulkan_gpu(device_type or 1)
                ),
            ),
            side_effect=OSError("type unavailable") if device_type is None else None,
        ),
    )
    monkeypatch.setattr(registry, "MINERU_4_MODELS_ONNX", SimpleNamespace(ocr_rec=SimpleNamespace(ensure=lambda: "rec.onnx")))
    measure = Mock(return_value={"decision_valid": True, "cpu_elapsed": 1, "gpu_elapsed": 0.5})
    monkeypatch.setattr(onnx_calibration, "run_calibration", measure)
    selected, completed = onnx_device._select("auto", 4, 1)
    assert selected.device == "gpu"
    if device_type == 2:
        assert selected.reason == "discrete_gpu" and not completed
        measure.assert_not_called()
    else:
        assert selected.reason == "rec_calibration" and completed
        measure.assert_called_once_with(
            "rec.onnx",
            with_gpu_classification(descriptor, classify_vulkan_gpu(device_type)),
            intra_op_num_threads=4,
            inter_op_num_threads=1,
            threshold=0.8,
        )


@pytest.mark.parametrize("fail", [False, True])
def test_loader_restriction_is_verified_and_restores_environment(monkeypatch: pytest.MonkeyPatch, fail: bool) -> None:
    old, good = (
        with_gpu_classification(vulkan.VulkanDevice(0x8086, 0x4680, "Intel"), classify_vulkan_gpu(1)),
        with_gpu_classification(vulkan.VulkanDevice(0x10DE, 0x2489, "NVIDIA"), classify_vulkan_gpu(2)),
    )
    monkeypatch.setenv("VK_LOADER_VENDOR_ID_FILTER", "0x8086,0x10de")
    monkeypatch.setenv("VK_LOADER_DEVICE_ID_FILTER", "0:65535")
    snapshot = dict(os.environ)

    def visible() -> tuple[vulkan.VulkanDevice, ...]:
        return (good,) if os.environ["VK_LOADER_VENDOR_ID_FILTER"] == "0x10de" else (old, good)

    monkeypatch.setattr(vulkan, "enumerate_vulkan_devices", visible)
    try:
        with vulkan.restrict_vulkan_device(0x10DE, 0x2489):
            assert os.environ["VK_LOADER_DEVICE_ID_FILTER"] == "0x2489"
            if fail:
                raise RuntimeError("session init failure")
    except RuntimeError:
        assert fail
    assert dict(os.environ) == snapshot


def test_older_loader_uses_verified_mesa_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    old, good = (
        with_gpu_classification(vulkan.VulkanDevice(0x8086, 0x4680, "Intel"), classify_vulkan_gpu(1)),
        with_gpu_classification(vulkan.VulkanDevice(0x8086, 0x56A0, "Arc"), classify_vulkan_gpu(2)),
    )
    snapshot = dict(os.environ)

    def visible() -> tuple[vulkan.VulkanDevice, ...]:
        return (good,) if os.environ.get("MESA_VK_DEVICE_SELECT") == "8086:56a0!" else (old, good)

    monkeypatch.setattr(vulkan, "enumerate_vulkan_devices", visible)
    with vulkan.restrict_vulkan_device(0x8086, 0x56A0):
        assert os.environ["MESA_VK_DEVICE_SELECT"] == "8086:56a0!"
    assert dict(os.environ) == snapshot


def test_ignored_filters_fail_before_creating_a_gpu_session(monkeypatch: pytest.MonkeyPatch) -> None:
    old, good = (
        with_gpu_classification(vulkan.VulkanDevice(0x8086, 0x4680, "Intel"), classify_vulkan_gpu(1)),
        with_gpu_classification(vulkan.VulkanDevice(0x10DE, 0x2489, "NVIDIA"), classify_vulkan_gpu(2)),
    )
    monkeypatch.setattr(vulkan, "enumerate_vulkan_devices", lambda: (old, good))
    factory = Mock(side_effect=AssertionError("unsafe native session"))
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    with pytest.raises(RuntimeError, match="Cannot restrict"):
        onnx.create_ort_session("rec.onnx", adapter=webgpu.WebGPUDevice(0x10DE, 0x2489, vulkan_device=(0x10DE, 0x2489)))
    factory.assert_not_called()
    assert webgpu._runtime_pid is None


def test_forbidden_intel_cannot_bypass_policy_through_low_level_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    factory = Mock(side_effect=AssertionError("native init"))
    monkeypatch.setattr(onnx.ort, "InferenceSession", factory)
    with pytest.raises(RuntimeError, match="Intel Xe-LPG"):
        onnx.create_ort_session("rec.onnx", adapter=webgpu.WebGPUDevice(0x8086, 0x4680))
    factory.assert_not_called()


def test_scope_covers_provider_attachment_and_native_session_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    good = with_gpu_classification(vulkan.VulkanDevice(0x10DE, 0x2489, "NVIDIA"), classify_vulkan_gpu(2))
    monkeypatch.setattr(vulkan, "enumerate_vulkan_devices", lambda: (good,))
    snapshot = dict(os.environ)

    def attached(*_: object) -> None:
        assert os.environ["VK_LOADER_DEVICE_ID_FILTER"] == "0x2489"

    def create(*_: object, **kwargs: object) -> SimpleNamespace:
        assert os.environ["VK_LOADER_VENDOR_ID_FILTER"] == "0x10de"
        return SimpleNamespace(get_providers=lambda: [webgpu.WEBGPU_PROVIDER], disable_fallback=Mock())

    monkeypatch.setattr(onnx, "attach_webgpu_provider", attached)
    monkeypatch.setattr(onnx.ort, "InferenceSession", create)
    for _ in range(2):
        onnx.create_ort_session("rec.onnx", adapter=webgpu.WebGPUDevice(0x10DE, 0x2489, vulkan_device=(0x10DE, 0x2489)))
    assert dict(os.environ) == snapshot
    with pytest.raises(RuntimeError, match="cannot change"):
        with webgpu.webgpu_session_scope(webgpu.WebGPUDevice(0x1002, 1)):
            raise AssertionError("rebound context")


def test_bound_descriptor_roundtrips_through_worker_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    from mineru.model.runtime import onnx_calibration

    descriptor = with_gpu_classification(
        webgpu.WebGPUDevice(0x10DE, 0x2489, (("pci_bus_id", "0000:01:00.0"),), (0x10DE, 0x2489)), classify_vulkan_gpu(2)
    )
    stream = io.StringIO(json.dumps({"backend": "webgpu", "adapter": asdict(descriptor), "model_path": "rec.onnx"}))
    monkeypatch.setattr(onnx_calibration.sys, "stdin", stream)
    measure = Mock(return_value={"ok": True})
    monkeypatch.setattr(onnx_calibration, "measure_rec", measure)
    onnx_calibration.main()
    assert measure.call_args.kwargs["adapter"] == descriptor
    assert hash(measure.call_args.kwargs["adapter"]) == hash(descriptor)


@pytest.mark.parametrize("error", [False, True])
@pytest.mark.parametrize("native_type", [0, 1, 2, 3])
def test_vulkan_query_reads_pci_identity_without_creating_logical_device_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch, error: bool, native_type: int
) -> None:
    """原生类型在适配边界统一转换，未知硬件仍保留，而软件 CPU 不进入 GPU 库存。"""

    def create(info: object, allocator: object, instance: object) -> int:
        instance._obj.value = 123
        return 0

    def enumerate_devices(instance: object, count: object, handles: object) -> int:
        if error:
            return -3
        count._obj.value = 2
        if handles is not None:
            handles[0], handles[1] = 100, 200
        return 0

    def properties(handle: object, storage: object) -> None:
        p = storage._obj.prefix
        p.vendorID, p.deviceID, p.deviceType, p.deviceName = (
            (0x10DE, 0x2489, native_type, b"NVIDIA") if handle == 100 else (0x10005, 0, 4, b"llvmpipe")
        )

    lib = SimpleNamespace(
        vkCreateInstance=Mock(side_effect=create),
        vkDestroyInstance=Mock(),
        vkEnumeratePhysicalDevices=Mock(side_effect=enumerate_devices),
        vkGetPhysicalDeviceProperties=Mock(side_effect=properties),
    )
    monkeypatch.setattr(vulkan.ct, "CDLL", lambda name: lib)
    if error:
        with pytest.raises(RuntimeError, match="count devices"):
            vulkan.enumerate_vulkan_devices()
    else:
        assert vulkan.enumerate_vulkan_devices() == (
            with_gpu_classification(vulkan.VulkanDevice(0x10DE, 0x2489, "NVIDIA", 0), classify_vulkan_gpu(native_type)),
        )
    lib.vkDestroyInstance.assert_called_once()
    assert ct.sizeof(vulkan._PropertiesStorage) >= 824
