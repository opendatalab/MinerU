"""硬件推荐与 OCR 校准证据的回归，不依赖测试机器的 GPU 或模型。"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

from mineru.model.runtime import onnx_calibration_cache as cache
from mineru.model.runtime import tier_recommendation as recommendation
from mineru.model.runtime.llama_probe import LlamaDevice
from mineru.model.runtime.webgpu import WebGPUDevice
from mineru.model.runtime.windows_gpu import GpuAdapter
from mineru.types import TIERS


@pytest.mark.parametrize(
    ("cores", "gpu", "expected"),
    [
        (1, False, {"flash"}),
        (3, False, {"flash"}),
        (4, False, set(TIERS[:-1])),
        (32, False, set(TIERS[:-1])),
        (1, True, set(TIERS)),
        (4, True, set(TIERS)),
    ],
)
def test_recommendation_matrix(cores: int, gpu: bool, expected: set[str]) -> None:
    """验证 CPU 边界及 GPU 对全部档位的开放规则。"""
    assert recommendation.tiers_for_hardware(recommendation.TierHardware(cores, gpu)) == expected


@pytest.mark.parametrize(
    ("affinity", "quota", "expected"), [(2, "max 100000", 2), (8, "300000 100000", 3), (8, "400000 100000", 4)]
)
def test_recommendation_respects_affinity_and_quota(
    monkeypatch: pytest.MonkeyPatch, affinity: int, quota: str, expected: int
) -> None:
    """使用真实有效 CPU 计数逻辑，保证亲和性与容器配额限制进入推荐。"""
    from mineru.model.runtime import cpu_threads

    monkeypatch.setattr(cpu_threads.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(cpu_threads.os, "sched_getaffinity", lambda pid: set(range(affinity)), raising=False)
    monkeypatch.setattr(cpu_threads.Path, "read_text", lambda path: quota if path.name == "cpu.max" else "-1")
    monkeypatch.setattr(recommendation, "_advanced_gpu_reason", lambda: None)
    hardware = recommendation.detect_tier_hardware()
    assert hardware.cpu_count == expected
    assert recommendation.tiers_for_hardware(hardware) == frozenset(TIERS[:-1] if expected >= 4 else ("flash",))


@pytest.fixture
def evidence_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """每个测试使用独立证据文件，避免写入用户配置目录。"""
    path = tmp_path / "settings.yaml.onnx-calibration.json"
    monkeypatch.setattr(cache, "calibration_record_path", lambda: path)
    monkeypatch.setattr(cache, "effective_cpu_count", lambda: 8)
    monkeypatch.setattr(cache, "_runtime_version", lambda distribution: "1.0")
    monkeypatch.delenv("MINERU_INTRA_OP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MINERU_INTER_OP_NUM_THREADS", raising=False)
    return path


def _adapter() -> GpuAdapter:
    """构造具有稳定物理身份的集显。"""
    return GpuAdapter(0, "physical-id", "integrated", 0x8086, 0x7D51, True, 0)


def _measurement(**overrides: Any) -> dict[str, Any]:
    """构造校准 worker 已验证的最小持久化数据。"""
    return {
        "intra_threads": 4,
        "inter_threads": 1,
        "cpu_elapsed": 2.0,
        "gpu_elapsed": 1.0,
        "decision_threshold": 0.8,
        "outputs_match": True,
        "decision_valid": True,
        "model_sha256": "model-hash",
        **overrides,
    }


@pytest.mark.parametrize(("ratio", "valid", "expected"), [(0.8, True, True), (0.801, True, False), (0.5, False, False)])
def test_calibration_threshold_and_validity(evidence_path: Path, ratio: float, valid: bool, expected: bool) -> None:
    """读取实际保存的证据，保留既有阈值和稳定性判定。"""
    cache.save_calibration_record(_adapter(), _measurement(gpu_elapsed=2.0 * ratio, decision_valid=valid))
    assert cache.calibrated_gpu_is_faster(_adapter(), 0.8) is expected
    assert cache.calibrated_gpu_is_faster(_adapter(), 0.75) is False


@pytest.mark.parametrize("change", ["device", "runtime", "threads", "cpu", "backend"])
def test_stale_evidence_is_ignored(evidence_path: Path, monkeypatch: pytest.MonkeyPatch, change: str) -> None:
    """设备、运行时和 CPU 配置改变时，不复用旧的性能结论。"""
    adapter = _adapter()
    cache.save_calibration_record(adapter, _measurement())
    assert cache.calibrated_gpu_is_faster(adapter, 0.8)
    if change == "device":
        adapter = replace(adapter, device_id=2)
    elif change == "runtime":
        monkeypatch.setattr(cache, "_runtime_version", lambda distribution: "2.0")
    elif change == "threads":
        monkeypatch.setenv("MINERU_INTRA_OP_NUM_THREADS", "8")
    elif change == "cpu":
        monkeypatch.setattr(cache, "effective_cpu_count", lambda: 2)
    else:
        adapter = WebGPUDevice(adapter.vendor_id, adapter.device_id)
    assert not cache.calibrated_gpu_is_faster(adapter, 0.8)


def test_adapter_reordering_and_webgpu_metadata(evidence_path: Path) -> None:
    """设备排序不作为物理身份，WebGPU 元数据经过 JSON 后仍可匹配。"""
    cache.save_calibration_record(_adapter(), _measurement())
    assert cache.calibrated_gpu_is_faster(replace(_adapter(), index=7, preference=2), 0.8)
    adapter = WebGPUDevice(0x8086, 0x7D51, (("driver", "v1"),))
    cache.save_calibration_record(adapter, _measurement())
    assert cache.calibrated_gpu_is_faster(adapter, 0.8)
    assert not cache.calibrated_gpu_is_faster(replace(adapter, metadata=(("driver", "v2"),)), 0.8)


@pytest.mark.parametrize("content", [None, "broken", '{"device":"gpu"}', '{"schema_version":0}'])
def test_missing_and_invalid_evidence_is_read_only(evidence_path: Path, content: str | None) -> None:
    """缺失、旧偏好和损坏记录均不授予高级能力，也不修写文件。"""
    if content is not None:
        evidence_path.write_text(content, encoding="utf-8")
    assert not cache.calibrated_gpu_is_faster(_adapter(), 0.8)
    assert evidence_path.read_text(encoding="utf-8") == content if content is not None else not evidence_path.exists()
    assert list(evidence_path.parent.iterdir()) == ([evidence_path] if content is not None else [])


def test_unwritable_record_does_not_break_calibration(evidence_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """证据写入失败不能影响校准完成后的运行设备选择。"""
    monkeypatch.setattr(cache.os, "replace", Mock(side_effect=PermissionError("read only")))
    cache.save_calibration_record(_adapter(), _measurement())
    assert not list(evidence_path.parent.iterdir())


@pytest.mark.parametrize(("platform", "metal"), [("darwin", True), ("darwin", False), ("other", False)])
def test_apple_metal_recommendation(monkeypatch: pytest.MonkeyPatch, platform: str, metal: bool) -> None:
    """Apple Silicon 按 Metal 可用性推荐，不要求 Windows/Linux 校准记录。"""
    from mineru.model.runtime import llama_device

    monkeypatch.delenv("MINERU_LLAMA_CPP_BACKEND", raising=False)
    monkeypatch.setattr(recommendation, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: 4)
    monkeypatch.setattr(recommendation, "_apple_metal_available", lambda: metal)
    monkeypatch.setattr(llama_device, "discover_llama_devices", lambda: ())
    assert recommendation.recommend_tiers() == frozenset(TIERS if metal else TIERS[:-1])


@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize("kind", ["discrete", "calibrated", "slow", "unknown", "mismatch", "hidden", "failure"])
def test_gpu_discovery_uses_only_existing_evidence(
    evidence_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, kind: str
) -> None:
    """以真实证据读取验证跨平台 GPU 路由，并禁止推荐路径调用校准。"""
    from mineru.model.runtime import llama_device, onnx_device, webgpu, windows_gpu

    monkeypatch.delenv("MINERU_LLAMA_CPP_BACKEND", raising=False)
    monkeypatch.setattr(recommendation, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: 4)
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", Mock(side_effect=AssertionError("must not calibrate")))
    descriptor = LlamaDevice("Vulkan0", "test", "vulkan", kind != "discrete", 0x8086, 0x7D51)
    if kind == "mismatch":
        descriptor = replace(descriptor, device_id=1)
    inventory = () if kind == "hidden" else (descriptor,)
    monkeypatch.setattr(
        llama_device,
        "discover_llama_devices",
        Mock(
            return_value=inventory,
            side_effect=RuntimeError("driver") if kind == "failure" else None,
        ),
    )
    adapter = _adapter() if platform == "win32" else WebGPUDevice(0x8086, 0x7D51)
    monkeypatch.setattr(windows_gpu, "discover_gpus", lambda: (adapter,))
    monkeypatch.setattr(webgpu, "discover_webgpu_devices", lambda: (adapter,))
    monkeypatch.setattr(webgpu, "select_webgpu_device", lambda devices: devices[0])
    if kind != "unknown":
        cache.save_calibration_record(adapter, _measurement(gpu_elapsed=3.0 if kind == "slow" else 1.0))
    before = evidence_path.read_bytes() if evidence_path.exists() else None
    assert recommendation.recommend_tiers() == frozenset(TIERS if kind in {"discrete", "calibrated"} else TIERS[:-1])
    assert (evidence_path.read_bytes() if evidence_path.exists() else None) == before
    onnx_device.resolve_onnx_device.assert_not_called()


def test_explicit_cpu_and_discovery_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 CPU 禁止 GPU 推荐，CPU 探测失败时只推荐基础档位。"""
    monkeypatch.setenv("MINERU_LLAMA_CPP_BACKEND", "cpu")
    monkeypatch.setattr(recommendation, "effective_cpu_count", Mock(side_effect=OSError("affinity")))
    assert recommendation.recommend_tiers() == {"flash"}
