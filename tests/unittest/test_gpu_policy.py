"""共享 GPU 类型、身份关联及各运行入口的一致性回归。"""

from __future__ import annotations

from dataclasses import asdict, replace
import itertools
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mineru.model.runtime import hardware, llama_probe, vulkan, webgpu, windows_gpu
from mineru.model.runtime.gpu import (
    GPU_KIND_ORDER,
    GpuClassification,
    GpuIdentity,
    GpuKind,
    PciAddress,
    classify_d3d12_gpu,
    classify_macos_gpu,
    classify_vulkan_gpu,
    gpu_identities_match,
    match_gpu_identities,
    with_gpu_classification,
)
from mineru.model.runtime.hardware_types import GpuInfo, RuntimeProbe


@pytest.mark.parametrize(
    ("platform", "classification"),
    [
        ("linux", classify_vulkan_gpu(1)),
        ("linux", classify_vulkan_gpu(2)),
        ("linux", classify_vulkan_gpu(3)),
        ("linux", classify_vulkan_gpu(0)),
        ("win32", classify_d3d12_gpu(True)),
        ("win32", classify_d3d12_gpu(False)),
        ("win32", classify_d3d12_gpu(None)),
    ],
)
def test_native_classification_agrees_across_runtime_and_report(
    monkeypatch: pytest.MonkeyPatch, platform: str, classification: GpuClassification
) -> None:
    """同一卡的 OCR、ggml 可见设备和硬件报告必须采用同一个分类来源。"""
    for module in (llama_probe, webgpu):
        monkeypatch.setattr(module, "sys", SimpleNamespace(platform=platform))
    monkeypatch.delenv("MINERU_LLAMA_CPP_BACKEND", raising=False)
    native = vulkan.VulkanDevice(0x1002, 1, "GPU", 0)
    # Windows 故意提供相反的 Vulkan 类型，确认分类权威仍是 D3D12。
    native = with_gpu_classification(
        native, classification if platform == "linux" else classify_vulkan_gpu(2 if classification.kind != "discrete" else 1)
    )
    monkeypatch.setattr(vulkan, "enumerate_vulkan_devices", lambda: (native,))
    windows = with_gpu_classification(windows_gpu.GpuAdapter(0, "physical", "GPU", 0x1002, 1, 0), classification)
    monkeypatch.setattr(windows_gpu, "discover_gpus", lambda: (windows,))
    monkeypatch.setattr(webgpu, "discover_webgpu_devices", lambda: (webgpu.WebGPUDevice(0x1002, 1),))
    actual = llama_probe._classify_devices((llama_probe.LlamaDevice("Vulkan0", "GPU", "vulkan"),), (native,))
    ocr = (
        windows_gpu.preferred_gpu((windows,))
        if platform == "win32"
        else webgpu.select_webgpu_device(webgpu.discover_webgpu_devices())
    )
    assert ocr is not None
    report = [GpuInfo(name="GPU", device_id="0x1002:0x0001")]
    hardware.merge_gpu_inventory(
        report,
        RuntimeProbe(
            "dxgi" if platform == "win32" else "vulkan", "ok", {"devices": [asdict(windows if platform == "win32" else native)]}
        ),
    )
    for device in (ocr, actual[0], report[0]):
        assert device.kind == classification.kind
        assert device.kind_source == classification.kind_source
        assert device.unified_memory == classification.unified_memory


@pytest.mark.parametrize("order", list(itertools.permutations(GPU_KIND_ORDER)))
def test_windows_kind_order_with_native_tiebreakers(order: tuple[GpuKind, ...]) -> None:
    """Windows 性能排序只能在统一类型优先级之后生效。"""
    windows = tuple(
        windows_gpu.GpuAdapter(index, str(index), kind, 0x1002, index, index, kind=kind) for index, kind in enumerate(order)
    )
    for expected in GPU_KIND_ORDER:
        selected = windows_gpu.preferred_gpu(windows)
        assert selected is not None and selected.kind == expected
        windows = tuple(device for device in windows if device.kind != expected)


def test_identity_matching_requires_mutual_uniqueness() -> None:
    """不能仅凭同型号或重复名称把多个设备合并，已有强身份相异时也不得降级匹配。"""
    card = GpuIdentity(0x10DE, 1, name="GPU")
    assert match_gpu_identities((card,), (card,)) == (0,)
    assert match_gpu_identities((card, card), (card,)) == (None, None)
    assert match_gpu_identities((card,), (card, card)) == (None,)
    assert match_gpu_identities((GpuIdentity(name="GPU"),), (card,)) == (None,)
    assert match_gpu_identities((GpuIdentity(name="GPU"),), (card,), allow_name=True) == (0,)
    assert not gpu_identities_match(replace(card, luid="a"), replace(card, luid="b"))
    assert not gpu_identities_match(replace(card, bus_id="0000:01:00.0"), replace(card, bus_id="0000:02:00.0"))
    assert not gpu_identities_match(card, replace(card, device_id=2), allow_name=True)
    first, second = replace(card, luid="a"), replace(card, luid="b")
    assert match_gpu_identities((first, second), (second, first)) == (1, 0)
    assert gpu_identities_match(GpuIdentity(bus_id="00000000:01:00.0"), GpuIdentity(bus_id="0000:01:00.0"))


@pytest.mark.parametrize("reverse", [False, True])
def test_physical_pci_match_beats_duplicate_name(reverse: bool) -> None:
    """有效 PCI 坐标优先于同名无物理地址的 DXGI 条目，不依赖枚举顺序。"""
    selected = GpuIdentity(name="8060S", pci_address=PciAddress(195, 0, domain=0))
    physical = GpuIdentity(0x1002, 5510, luid="physical", name="8060S", pci_address=PciAddress(195, 0, function=0))
    duplicate = replace(physical, luid="duplicate", pci_address=None)
    targets = (physical, duplicate) if not reverse else (duplicate, physical)
    assert match_gpu_identities((selected,), targets, allow_name=True) == (int(reverse),)
    assert match_gpu_identities((selected, selected), targets, allow_name=True) == (None, None)


def test_partial_pci_address_does_not_guess_domain_or_function() -> None:
    """Torch 不提供 function、Windows 不提供 domain 时，只比较已知坐标并保留多解。"""
    selected = GpuIdentity(name="GPU", pci_address=PciAddress(195, 0, domain=0))
    function_zero = GpuIdentity(name="GPU", bus_id="0000:c3:00.0")
    function_one = replace(function_zero, bus_id="0000:c3:00.1")
    assert match_gpu_identities((selected,), (function_one,), allow_name=True) == (0,)
    assert match_gpu_identities((selected,), (function_zero, function_one), allow_name=True) == (None,)
    for conflict in ("0001:c3:00.0", "0000:c4:00.0", "0000:c3:01.0"):
        assert not gpu_identities_match(selected, replace(function_zero, bus_id=conflict), allow_name=True)
    assert not gpu_identities_match(replace(selected, luid="one"), replace(function_zero, luid="two"), allow_name=True)


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_ambiguous_native_mapping_remains_unknown_everywhere(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """同型号多卡无法关联到运行时和报告时，不能被标为确定独显。"""
    monkeypatch.setattr(llama_probe, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(webgpu, "sys", SimpleNamespace(platform="linux"))
    physical = tuple(vulkan.VulkanDevice(0x10DE, 1, "GPU", index, kind="discrete", kind_source="vulkan") for index in range(2))
    raw = tuple(llama_probe.LlamaDevice(f"Vulkan{index}", "GPU", "vulkan") for index in range(2))
    monkeypatch.setattr(vulkan, "enumerate_vulkan_devices", lambda: physical)
    monkeypatch.setattr(
        windows_gpu,
        "discover_gpus",
        lambda: tuple(
            windows_gpu.GpuAdapter(index, str(index), "GPU", 0x10DE, 1, index, kind="discrete", kind_source="d3d12")
            for index in range(2)
        ),
    )
    result = llama_probe._classify_devices(raw, physical)
    assert all(device.kind == "unknown" for device in result)
    ocr = webgpu.select_webgpu_device((webgpu.WebGPUDevice(0x10DE, 1),))
    assert ocr is not None and ocr.kind == "unknown"
    report = [GpuInfo(name="GPU", device_id="0x10de:0x0001") for _ in range(2)]
    hardware.merge_gpu_inventory(report, RuntimeProbe("vulkan", "ok", {"devices": [asdict(device) for device in physical]}))
    assert len(report) == 2 and all(device.kind == "unknown" for device in report)


@pytest.mark.parametrize("luid", ["selected", "other", "missing", None])
@pytest.mark.parametrize("reverse", [False, True])
def test_windows_vulkan_luid_disambiguates_same_model_dxgi_adapters(
    monkeypatch: pytest.MonkeyPatch, luid: str | None, reverse: bool
) -> None:
    """同型号 DXGI 库存必须按 Vulkan 的有效 LUID 关联，缺失或冲突时不猜测。"""
    monkeypatch.setattr(llama_probe, "sys", SimpleNamespace(platform="win32"))
    adapters = (
        windows_gpu.GpuAdapter(0, "selected", "GPU", 0x10DE, 1, 0, kind="discrete", kind_source="d3d12"),
        windows_gpu.GpuAdapter(2, "other", "GPU", 0x10DE, 1, 3, kind="integrated", kind_source="d3d12"),
    )
    monkeypatch.setattr(windows_gpu, "discover_gpus", lambda: adapters[::-1] if reverse else adapters)
    result = llama_probe._classify_devices(
        (llama_probe.LlamaDevice("Vulkan0", "GPU", "vulkan"),),
        (vulkan.VulkanDevice(0x10DE, 1, "GPU", 0, luid=luid),),
    )
    expected = {"selected": "discrete", "other": "integrated"}.get(luid, "unknown")
    assert result[0].kind == expected
    assert result[0].identity.luid == luid
    assert result[0].vulkan_index == 0


def test_windows_native_query_failure_does_not_use_ggml_or_vulkan_type(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows 的分类查询失败时保留未知，不把 ggml/Vulkan 的 GPU 标签升级为独显。"""
    monkeypatch.setattr(llama_probe, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(windows_gpu, "discover_gpus", Mock(side_effect=OSError("D3D12 unavailable")))
    result = llama_probe._classify_devices(
        (llama_probe.LlamaDevice("Vulkan0", "GPU", "vulkan"),),
        (vulkan.VulkanDevice(0x10DE, 1, "GPU", 0, kind="discrete", kind_source="vulkan"),),
    )
    assert result[0].kind == "unknown" and result[0].vulkan_index == 0


@pytest.mark.parametrize("native_type", [1, 3])
def test_ggml_gpu_enum_only_controls_inventory(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, native_type: int) -> None:
    """实际 ctypes 探测即使收到 ggml GPU 枚举，也必须采用原生集显或虚拟 GPU 分类。"""
    library_path = tmp_path / "lib" / "libggml.so"
    library_path.parent.mkdir()
    library_path.touch()
    library = SimpleNamespace(
        ggml_backend_load_all_from_path=Mock(),
        ggml_backend_dev_count=Mock(return_value=1),
        ggml_backend_dev_get=Mock(return_value=1),
        ggml_backend_dev_type=Mock(return_value=1),
        ggml_backend_dev_name=Mock(return_value=b"Vulkan0"),
        ggml_backend_dev_description=Mock(return_value=b"GPU"),
        ggml_backend_dev_backend_reg=Mock(return_value=1),
        ggml_backend_reg_name=Mock(return_value=b"Vulkan"),
    )
    physical = with_gpu_classification(vulkan.VulkanDevice(0x1002, 1, "GPU", 0), classify_vulkan_gpu(native_type))
    monkeypatch.setattr(llama_probe, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(llama_probe.ct, "CDLL", lambda path: library)
    monkeypatch.setattr(llama_probe, "enumerate_vulkan_devices", lambda: (physical,))
    result = llama_probe._probe_library(tmp_path)
    assert result[0].kind == physical.kind and result[0].kind_source == "vulkan"
    assert not hasattr(result[0], "integrated")
    with pytest.raises(TypeError, match="integrated"):
        llama_probe.LlamaDevice("Vulkan0", "GPU", "vulkan", integrated=True)


@pytest.mark.parametrize("folder", ["bin", "lib"])
@pytest.mark.parametrize("prefix", ["", "lib"])
def test_windows_probe_loads_device_accessors_from_base_dll(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, folder: str, prefix: str
) -> None:
    """Windows DLL 导出不沿依赖解析；严格分离注册与设备接口以覆盖真实 wheel 布局。"""
    registry_path = tmp_path / folder / f"{prefix}ggml.dll"
    registry_path.parent.mkdir()
    registry_path.touch()
    base_path = registry_path.with_name(f"{prefix}ggml-base.dll")
    base_path.touch()
    registry_library = SimpleNamespace(
        ggml_backend_load_all_from_path=Mock(),
        ggml_backend_dev_count=Mock(return_value=1),
        ggml_backend_dev_get=Mock(return_value=1),
    )
    base_library = SimpleNamespace(
        ggml_backend_dev_type=Mock(return_value=1),
        ggml_backend_dev_name=Mock(return_value=b"Vulkan0"),
        ggml_backend_dev_description=Mock(return_value=b"GPU"),
        ggml_backend_dev_backend_reg=Mock(return_value=1),
        ggml_backend_reg_name=Mock(return_value=b"Vulkan"),
    )
    libraries = {str(registry_path): registry_library, str(base_path): base_library}
    monkeypatch.setattr(llama_probe, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(llama_probe.ct, "CDLL", libraries.__getitem__)
    monkeypatch.setattr(llama_probe, "enumerate_vulkan_devices", lambda: ())
    monkeypatch.setattr(windows_gpu, "discover_gpus", lambda: ())

    assert llama_probe._probe_library(tmp_path) == (llama_probe.LlamaDevice("Vulkan0", "GPU", "vulkan"),)


def test_apple_classification_preserves_unified_memory() -> None:
    """Apple Silicon 的集显分类保留统一内存事实，不改变 MPS 的可用性推荐。"""
    apple = classify_macos_gpu("Apple", "arm64", integrated=False)
    assert apple.kind == "integrated" and apple.unified_memory is True
    unknown = classify_macos_gpu("AMD", "x86_64", integrated=False)
    assert unknown.kind == "unknown" and unknown.unified_memory is None


def test_shared_policy_imports_without_inference_dependencies() -> None:
    """共享事实和排序层的导入不得触发 Torch、ORT 或 llama 原生运行时。"""
    code = """
import importlib.abc
import sys
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        # 阻止测试进程意外导入推理依赖。
        if fullname.split(".")[0] in {"torch", "onnxruntime", "mineru_llama_cpp"}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockRuntime())
from mineru.model.runtime.gpu import GPU_KIND_ORDER, GpuIdentity, match_gpu_identities
assert GPU_KIND_ORDER[0] == "discrete"
assert match_gpu_identities((GpuIdentity(1, 2),), (GpuIdentity(1, 2),)) == (0,)
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=15)


def test_windows_memory_query_without_torch_or_other_inference_dependencies() -> None:
    """基础安装的 Windows 原生查询和 LUID 匹配无需导入任何推理后端。"""
    code = """
import importlib.abc
import sys
from types import SimpleNamespace
class BlockRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'onnxruntime', 'mineru_llama_cpp'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, BlockRuntime())
from mineru.model.runtime import gpu_memory, windows_gpu
from mineru.model.runtime.gpu import GpuIdentity
gpu_memory.sys = SimpleNamespace(platform='win32')
rows = [
    {'vendor_id': 4098, 'device_id': 5510, 'name': '8060S', 'luid': luid,
     'kind': 'integrated', 'memory_budget_bytes': budget, 'pci_address': None}
    for luid, budget in [('other', 128 * 1024**3), ('selected', 8 * 1024**3)]
]
windows_gpu.inspect_gpus = lambda **kwargs: rows
memory = gpu_memory.native_gpu_memory(GpuIdentity(luid='selected'), device='dml:gpu')
assert memory.memory_bytes == 8 * 1024**3
assert memory.source == 'dxgi.local_budget' and memory.identity.luid == 'selected'
assert 'torch' not in sys.modules and 'onnxruntime' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, timeout=15)
