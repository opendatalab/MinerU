"""默认硬件报告突出运行时选择，完整设备库存仅保留在 JSON 中。"""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any

import pytest

from mineru.kit.commands import hardware as command
from mineru.model.runtime.hardware_types import RuntimeProbe


@pytest.fixture
def windows_runtimes() -> dict[str, Any]:
    """复现双显卡 Windows 报告中的重复适配器、Torch 后端矩阵和原生身份字段。"""
    adapters = [
        {"name": "NVIDIA GeForce RTX 3060 Ti", "luid": "00000000:0000f514", "index": 0, "kind": "unknown"},
        {"name": "Intel(R) UHD Graphics 770", "luid": "00000000:00010445", "index": 1, "kind": "unknown"},
        {"name": "NVIDIA GeForce RTX 3060 Ti", "luid": "00000000:0001e2fd", "index": 2, "kind": "unknown"},
    ]
    probes = [
        RuntimeProbe("selection", "ok", {"small_backend": "torch", "vlm_engine": "lmdeploy"}),
        RuntimeProbe(
            "torch",
            "ok",
            {
                "version": "2.12.1+cu126",
                "cuda_build": "12.6",
                "rocm_build": None,
                "backends": {
                    "cuda": {
                        "available": True,
                        "devices": [
                            {"index": 0, "name": "NVIDIA GeForce RTX 3060 Ti", "total_memory_bytes": 8 * 1024**3},
                        ],
                    },
                    "mps": {"available": False},
                    "xpu": {"available": False, "devices": []},
                },
                "selected_device": {"device": "cuda", "reason": "torch_accelerator", "warning": None},
            },
        ),
        RuntimeProbe(
            "onnx",
            "ok",
            {
                "version": "1.24.4",
                "providers": ["DmlExecutionProvider", "CPUExecutionProvider"],
                "backend": "dml",
                "adapters": adapters,
                "selected_adapter": adapters[0],
                "selected_device": {"device": "gpu", "backend": "dml", "reason": "configured_gpu", "adapter": adapters[0]},
            },
        ),
        RuntimeProbe(
            "llama-cpp",
            "ok",
            {
                "devices": [
                    {"name": "Vulkan0", "description": "NVIDIA GeForce RTX 3060 Ti", "backend": "vulkan", "kind": "discrete"},
                    {"name": "Vulkan1", "description": "Intel(R) UHD Graphics 770", "backend": "vulkan", "kind": "integrated"},
                ]
            },
        ),
        RuntimeProbe("lmdeploy", "ok", {"version": "0.17.0", "module_path": "/lmdeploy/__init__.py"}),
        RuntimeProbe("dxgi", "ok", {"devices": [{**adapter, "vendor_id": 4318, "pci_error": None} for adapter in adapters]}),
    ]
    return {"runtimes": [asdict(probe) for probe in probes]}


def test_windows_runtime_summary_preserves_raw_diagnostics(windows_runtimes: dict[str, Any]) -> None:
    """用户提供的繁杂报告缩为关键信息，JSON 原始字段和重复适配器不被改写。"""
    before = json.dumps(windows_runtimes)
    rendered = command._render_info_section("runtimes", windows_runtimes)
    assert rendered == "\n".join(
        [
            "运行时：",
            "  小模型后端：torch；VLM 引擎：lmdeploy",
            "  Torch：2.12.1+cu126，正常",
            "    当前设备：cuda:0（NVIDIA GeForce RTX 3060 Ti，8.000 GiB）",
            "    CUDA 构建版本：12.6",
            "  ONNX Runtime：1.24.4，正常",
            "    可用 Provider：DirectML, CPU",
            "    设备选择：dml:gpu（NVIDIA GeForce RTX 3060 Ti）",
            "  llama.cpp：正常",
            "    可见设备：",
            "      Vulkan0：NVIDIA GeForce RTX 3060 Ti",
            "      Vulkan1：Intel(R) UHD Graphics 770",
            "  LMDeploy：0.17.0，正常",
        ]
    )
    assert json.dumps(windows_runtimes) == before


@pytest.mark.parametrize("backend", ["cuda", "npu", "xpu", "gcu", "musa", "mlu", "sdaa"])
def test_torch_summary_selects_actual_device_index(backend: str) -> None:
    """多卡和第三方后端只显示选中索引，扩展设备不展示无关 CUDA 构建字段。"""
    probe = asdict(
        RuntimeProbe(
            "torch",
            "ok",
            {
                "cuda_build": "12.6",
                "rocm_build": None,
                "selected_device": {"device": f"{backend}:1"},
                "backends": {
                    backend: {
                        "available": True,
                        "devices": [
                            {"index": 0, "name": "other GPU", "total_memory_bytes": 4 * 1024**3},
                            {"index": 1, "name": "selected GPU", "total_memory_bytes": 16 * 1024**3},
                        ],
                    }
                },
            },
        )
    )
    rendered = "\n".join(command._render_runtime_probe(probe))
    assert f"当前设备：{backend}:1（selected GPU，16.000 GiB）" in rendered
    assert "other GPU" not in rendered
    assert ("CUDA 构建版本" in rendered) == (backend == "cuda")
    assert "ROCm" not in rendered


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda:2"])
def test_torch_summary_does_not_guess_missing_device(device: str) -> None:
    """CPU、MPS 和缺失索引不能误用第一张卡的型号或容量。"""
    probe = asdict(
        RuntimeProbe(
            "torch",
            "ok",
            {
                "selected_device": {"device": device},
                "backends": {"cuda": {"devices": [{"index": 0, "name": "wrong GPU", "total_memory_bytes": 8 * 1024**3}]}},
            },
        )
    )
    rendered = "\n".join(command._render_runtime_probe(probe))
    assert f"当前设备：{device}" in rendered
    assert "wrong GPU" not in rendered and "8.000 GiB" not in rendered


def test_torch_summary_displays_rocm_build_when_applicable() -> None:
    """ROCm 构建沿用实际 cuda 设备接口，只展示存在的 ROCm 构建版本。"""
    probe = asdict(
        RuntimeProbe(
            "torch",
            "ok",
            {
                "selected_device": {"device": "cuda"},
                "cuda_build": None,
                "rocm_build": "6.3",
            },
        )
    )
    rendered = "\n".join(command._render_runtime_probe(probe))
    assert "ROCm 构建版本：6.3" in rendered and "CUDA 构建版本" not in rendered


@pytest.mark.parametrize("device, reason", [("cpu", "configured_cpu"), (None, "calibration_required")])
def test_onnx_summary_distinguishes_candidate_from_selected_gpu(device: str | None, reason: str) -> None:
    """显式 CPU 不展示 GPU 为当前设备，待校准只将适配器标为候选。"""
    probe = asdict(
        RuntimeProbe(
            "onnx",
            "ok",
            {
                "selected_device": {"device": device, "reason": reason},
                "selected_adapter": {"name": "candidate GPU"},
            },
        )
    )
    rendered = "\n".join(command._render_runtime_probe(probe))
    if device == "cpu":
        assert "设备选择：cpu" in rendered and "candidate GPU" not in rendered
    else:
        assert "设备选择：待校准" in rendered and "候选设备：candidate GPU" in rendered


@pytest.mark.parametrize("name", ["dxgi", "vulkan"])
def test_native_inventory_does_not_create_empty_output(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    """原生库存不重复展示，分阶段渲染也不会额外写出空行。"""
    command._emit_info_section("runtimes", {"runtime_header": False, "runtimes": [asdict(RuntimeProbe(name, "ok"))]})
    assert capsys.readouterr().out == ""


def test_runtime_summary_keeps_actionable_errors() -> None:
    """简化展示仍保留运行时导入错误和局部设备探测错误。"""
    failed = asdict(RuntimeProbe("onnx", "error", error="driver unavailable"))
    torch = asdict(
        RuntimeProbe(
            "torch",
            "ok",
            {
                "selected_device": {"device": "cpu", "warning": "configured GPU unavailable"},
                "backends": {"cuda": {"available": None, "error": "driver failure"}},
            },
        )
    )
    rendered = command._render_info_section("runtimes", {"runtimes": [failed, torch]})
    assert "ONNX Runtime：失败" in rendered and "driver unavailable" in rendered
    assert "configured GPU unavailable" in rendered and "cuda 探测失败：driver failure" in rendered
