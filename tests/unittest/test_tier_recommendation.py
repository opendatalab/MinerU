"""硬件推荐与 OCR 校准证据的回归，不依赖测试机器的 GPU 或模型。"""

from __future__ import annotations

import importlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
import yaml

from mineru.model.runtime import onnx_calibration_cache as cache
from mineru.model.runtime import tier_recommendation as recommendation
from mineru.model.runtime.gpu import GpuIdentity, classify_d3d12_gpu, classify_vulkan_gpu, with_gpu_classification
from mineru.model.runtime.gpu_memory import GpuMemory
from mineru.model.runtime.llama_probe import LlamaDevice
from mineru.model.runtime.webgpu import WebGPUDevice
from mineru.model.runtime.windows_gpu import GpuAdapter
from mineru.types import TIERS


@pytest.fixture(autouse=True)
def onnx_recommendation_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """本文件验证 ONNX 既有推荐规则，隔离宿主可能自动选择的 Torch 后端。"""
    monkeypatch.setattr(recommendation, "resolve_small_model_backend", lambda: "onnx")
    from mineru.config import get_config

    monkeypatch.setattr(get_config().model, "onnx_device", "auto")
    monkeypatch.setattr(
        recommendation,
        "native_gpu_memory",
        lambda identity, *, device: GpuMemory(device, identity, 8 * 1024**3, "budget", "test"),
    )


@pytest.mark.parametrize(
    ("cores", "gpu", "expected"),
    [
        (1, False, {"flash"}),
        (3, False, {"flash"}),
        (4, False, set(TIERS[:2])),
        (32, False, set(TIERS[:2])),
        (1, True, set(TIERS)),
        (4, True, set(TIERS)),
    ],
)
def test_recommendation_matrix(cores: int, gpu: bool, expected: set[str]) -> None:
    """验证 CPU 边界及 GPU 对全部档位的开放规则。"""
    assert (
        recommendation.tiers_for_hardware(
            recommendation.TierHardware(cores, gpu, gpu_memory=GpuMemory("gpu", GpuIdentity(), 8 * 1024**3))
        )
        == expected
    )


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
    monkeypatch.setattr(recommendation, "_onnx_gpu_hardware", lambda count: recommendation.TierHardware(count))
    hardware = recommendation.detect_tier_hardware()
    assert hardware.cpu_count == expected
    assert recommendation.tiers_for_hardware(hardware) == frozenset(TIERS[:2] if expected >= 4 else ("flash",))


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
    return with_gpu_classification(GpuAdapter(0, "physical-id", "integrated", 0x8086, 0x7D51, 0), classify_d3d12_gpu(True))


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
    from mineru.model.runtime import llama_device, onnx_device

    monkeypatch.delenv("MINERU_LLAMA_CPP_BACKEND", raising=False)
    monkeypatch.setattr(recommendation, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: 4)
    monkeypatch.setattr(recommendation, "_apple_metal_available", lambda: metal)
    monkeypatch.setattr(
        recommendation, "torch_gpu_memory", lambda device: GpuMemory(device, GpuIdentity(), 8 * 1024**3, "budget", "test")
    )
    monkeypatch.setattr(llama_device, "discover_llama_devices", lambda: ())
    selection = Mock(side_effect=AssertionError("unsupported platform must not select ONNX device"))
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", selection)
    assert frozenset(recommendation.recommend_tiers().pdf_image_tiers) == frozenset(TIERS if metal else TIERS[:2])
    selection.assert_not_called()


@pytest.mark.parametrize("platform", ["win32", "linux"])
@pytest.mark.parametrize("kind", ["discrete", "calibrated", "slow", "unknown", "mismatch", "hidden", "failure"])
def test_gpu_discovery_uses_only_existing_evidence(
    evidence_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str, kind: str
) -> None:
    """硬件初选只读取真实证据，校准由达到门槛后的启动编排触发。"""
    from mineru.model.runtime import llama_device, onnx_device, webgpu, windows_gpu

    monkeypatch.delenv("MINERU_LLAMA_CPP_BACKEND", raising=False)
    monkeypatch.setattr(recommendation, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: 4)
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", Mock(side_effect=AssertionError("must not calibrate")))
    descriptor = LlamaDevice(
        "Vulkan0",
        "test",
        "vulkan",
        0x8086,
        0x7D51,
        kind="integrated" if kind != "discrete" else "discrete",
        kind_source="vulkan",
    )
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
    adapter = _adapter() if platform == "win32" else WebGPUDevice(0x8086, 0x7D51, vulkan_device=(0x8086, 0x7D51))
    monkeypatch.setattr(windows_gpu, "discover_gpus", lambda: (adapter,))
    monkeypatch.setattr(webgpu, "discover_webgpu_devices", lambda: (adapter,))
    monkeypatch.setattr(webgpu, "select_webgpu_device", lambda devices: devices[0])
    if kind != "unknown":
        cache.save_calibration_record(adapter, _measurement(gpu_elapsed=3.0 if kind == "slow" else 1.0))
    before = evidence_path.read_bytes() if evidence_path.exists() else None
    hardware = recommendation.detect_tier_hardware()
    assert recommendation.tiers_for_hardware(hardware) == frozenset(TIERS if kind in {"discrete", "calibrated"} else TIERS[:2])
    assert (evidence_path.read_bytes() if evidence_path.exists() else None) == before
    onnx_device.resolve_onnx_device.assert_not_called()


def test_explicit_cpu_and_discovery_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """显式 CPU 禁止 GPU 推荐，CPU 探测失败时只推荐基础档位。"""
    monkeypatch.setenv("MINERU_LLAMA_CPP_BACKEND", "cpu")
    monkeypatch.setattr(recommendation, "effective_cpu_count", Mock(side_effect=OSError("affinity")))
    assert recommendation.recommend_tiers().pdf_image_tiers == ("flash",)


@pytest.fixture(params=["win32", "linux"])
def startup_calibration(
    request: pytest.FixtureRequest, evidence_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Mock]:
    """模拟双平台物理设备和测速 worker，保留真实选型、配置落盘及证据读取链路。"""
    import onnxruntime

    from mineru.model.download import ModelPath
    from mineru.model.runtime import llama_device, onnx_calibration, onnx_device, vulkan, webgpu, windows_gpu

    settings = importlib.import_module("mineru.config")
    config_file = evidence_path.with_name("settings.yaml")
    config_file.write_text("model:\n  onnx_device: auto\n  source: local\n", encoding="utf-8")
    monkeypatch.setattr(settings.get_config().model, "onnx_device", "auto")
    monkeypatch.setattr(settings, "get_config_file_path", lambda: str(config_file))
    monkeypatch.setattr(settings, "get_config_source", lambda key: "file")
    monkeypatch.delenv("MINERU_LLAMA_CPP_BACKEND", raising=False)
    for module in (recommendation, onnx_device, webgpu):
        monkeypatch.setattr(module, "sys", SimpleNamespace(platform=request.param))
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        lambda: (with_gpu_classification(vulkan.VulkanDevice(0x8086, 0x7D51, "test"), classify_vulkan_gpu(1)),),
    )
    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: 8)
    monkeypatch.setattr(onnx_device, "_selections", {})
    monkeypatch.setattr(onnxruntime, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    for module in (onnx_device, windows_gpu):
        monkeypatch.setattr(module, "discover_gpus", lambda **kwargs: (_adapter(),))
    for module in (onnx_device, webgpu):
        monkeypatch.setattr(module, "discover_webgpu_devices", lambda: (WebGPUDevice(0x8086, 0x7D51),))
    monkeypatch.setattr(
        llama_device,
        "discover_llama_devices",
        lambda: (LlamaDevice("Vulkan0", "test", "vulkan", 0x8086, 0x7D51, kind="integrated", kind_source="vulkan"),),
    )
    # 只替换权重准备，避免服务惰性导入的模块永久持有伪造的仓库对象。
    monkeypatch.setattr(ModelPath, "ensure", Mock(return_value=Path("rec.onnx")))
    measurement = Mock(return_value=_measurement())

    def run_calibration(
        model_path: str,
        adapter: GpuAdapter | WebGPUDevice,
        *,
        intra_op_num_threads: int,
        inter_op_num_threads: int,
        threshold: float,
    ) -> dict[str, Any]:
        """只替换实际硬件测量，按真实 worker 生命周期撤销并保存性能证据。"""
        cache.invalidate_calibration_record()
        result = measurement()
        cache.save_calibration_record(adapter, result)
        return result

    monkeypatch.setattr(onnx_calibration, "run_calibration", run_calibration)
    return config_file, measurement


@pytest.fixture
def configured_gpu(startup_calibration: tuple[Path, Mock]) -> tuple[Path, Mock]:
    """复用双平台选型环境，仅将用户设备偏好设置为 GPU。"""
    from mineru.config import get_config

    config_file, measurement = startup_calibration
    get_config().model.onnx_device = "gpu"
    config_file.write_text("model:\n  onnx_device: gpu\n  source: local\n", encoding="utf-8")
    return config_file, measurement


@pytest.mark.parametrize("outcome", ["fast", "slow", "unstable", "timeout", "output_error"])
def test_auto_startup_calibrates_before_publishing(
    startup_calibration: tuple[Path, Mock], monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    """真实服务工厂在发布能力前读取本次校准，失败保持 Basic 上限且不伪造 CPU 胜出。"""
    from fastapi.testclient import TestClient

    from mineru.model.runtime import onnx_device
    from mineru.parser import api_server

    config_file, measurement = startup_calibration
    if outcome == "slow":
        measurement.return_value = _measurement(gpu_elapsed=2.0)
    elif outcome == "unstable":
        measurement.return_value = _measurement(decision_valid=False)
    elif outcome in {"timeout", "output_error"}:
        measurement.side_effect = TimeoutError("worker") if outcome == "timeout" else ValueError("outputs differ")
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", Mock())
    preload = Mock(side_effect=AssertionError("startup calibration must not require model preload"))
    monkeypatch.setattr(api_server, "_preload_server_models", preload)

    app = api_server.create_app(upload_dir=str(config_file.parent / "uploads"))
    expected = TIERS if outcome == "fast" else TIERS[:2]
    measurement.assert_called_once()
    assert app.state.server_tiers == expected
    assert app.state.default_tier == ("standard" if outcome == "fast" else "basic")
    with TestClient(app) as client:
        assert tuple(row["id"] for row in client.get("/v1/tiers").json()["data"]) == expected
    preload.assert_not_called()
    # 随后的模型初始化复用本进程选择，不再开启第二次校准。
    selected = onnx_device.resolve_onnx_device()
    assert selected.device == ("gpu" if outcome == "fast" else "cpu")
    measurement.assert_called_once()
    saved = "gpu" if outcome == "fast" else "cpu" if outcome == "slow" else "auto"
    assert yaml.safe_load(config_file.read_text(encoding="utf-8"))["model"] == {
        "onnx_device": saved,
        "source": "local",
    }


def test_flash_without_sufficient_gpu_skips_calibration(
    startup_calibration: tuple[Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """可用 CPU 不足四且无已有 GPU 证据时，只开放 Flash 并跳过启动选型。"""
    from mineru.model.runtime import onnx_device

    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: 3)
    monkeypatch.setattr(recommendation, "native_gpu_memory", lambda identity, *, device: GpuMemory(device, identity))
    assert recommendation.recommend_tiers().pdf_image_tiers == ("flash",)
    startup_calibration[1].assert_not_called()
    assert not onnx_device._selections


def test_low_cpu_advanced_recommendation_still_selects_device(
    startup_calibration: tuple[Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """门槛按初选档位判断，低 CPU 配额的独显设备仍按平台规则提前选型。"""
    from mineru.model.runtime import llama_device, onnx_device, vulkan

    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: 2)
    monkeypatch.setattr(
        llama_device,
        "discover_llama_devices",
        lambda: (LlamaDevice("Vulkan0", "test", "vulkan", 0x8086, 0x7D51, kind="discrete", kind_source="vulkan"),),
    )
    monkeypatch.setattr(
        onnx_device, "discover_gpus", lambda **kwargs: (with_gpu_classification(_adapter(), classify_d3d12_gpu(False)),)
    )
    monkeypatch.setattr(
        vulkan,
        "enumerate_vulkan_devices",
        lambda: (with_gpu_classification(vulkan.VulkanDevice(0x8086, 0x7D51, "test"), classify_vulkan_gpu(2)),),
    )
    report = recommendation.recommend_tiers()
    assert frozenset(report.pdf_image_tiers) == frozenset(TIERS)
    assert report.calibration.status == "skipped" and report.calibration.reason == "discrete_gpu"
    assert report.default_request_tier == "standard"
    assert onnx_device._selections
    startup_calibration[1].assert_not_called()
    config_file = startup_calibration[0]
    assert yaml.safe_load(config_file.read_text(encoding="utf-8"))["model"]["onnx_device"] == "auto"
    assert not cache.calibration_record_path().exists()


@pytest.mark.parametrize("tier", TIERS)
def test_explicit_ceiling_skips_startup_selection(startup_calibration: tuple[Path, Mock], tier: str) -> None:
    """显式启动上限和 Python 精确集合均不进入自动推荐的设备选择。"""
    from mineru.model.runtime import onnx_device
    from mineru.parser.server_tiers import parse_server_tier, resolve_server_tiers

    selected = parse_server_tier(tier)
    assert resolve_server_tiers(selected) == selected
    startup_calibration[1].assert_not_called()
    assert not onnx_device._selections


@pytest.mark.parametrize("requested", ["cpu", "gpu"])
def test_startup_preserves_explicit_device_without_performance_evidence(
    startup_calibration: tuple[Path, Mock], requested: str
) -> None:
    """显式设备不触发测速，实际选中的达标 GPU 无需校准证据即可开放四档。"""
    settings = importlib.import_module("mineru.config")
    config_file, measurement = startup_calibration
    settings.get_config().model.onnx_device = requested
    config_file.write_text(f"model:\n  onnx_device: {requested}\n", encoding="utf-8")
    report = recommendation.recommend_tiers()
    assert report.pdf_image_tiers == (TIERS if requested == "gpu" else TIERS[:2])
    assert report.reason == f"configured_{requested}"
    assert report.calibration.status == "skipped"
    measurement.assert_not_called()
    assert yaml.safe_load(config_file.read_text(encoding="utf-8"))["model"]["onnx_device"] == requested


@pytest.mark.parametrize("remove_evidence", [False, True])
def test_next_startup_reuses_saved_device(startup_calibration: tuple[Path, Mock], remove_evidence: bool) -> None:
    """自动校准保存 GPU 后，即使记录丢失也直接采用 GPU 与当前内存预算。"""
    from mineru.model.runtime import onnx_device

    assert frozenset(recommendation.recommend_tiers().pdf_image_tiers) == frozenset(TIERS)
    onnx_device._selections.clear()
    if remove_evidence:
        cache.calibration_record_path().unlink()
    report = recommendation.recommend_tiers()
    assert report.pdf_image_tiers == TIERS
    assert report.reason == "configured_gpu"
    assert report.calibration.status == "skipped" and report.calibration.reason == "configured_gpu"
    startup_calibration[1].assert_called_once()
    assert onnx_device.resolve_onnx_device().reason == "configured_gpu"


@pytest.mark.parametrize("record", ["missing", "invalid", "stale", "slow", "unstable", "valid", "read_error"])
def test_configured_gpu_does_not_require_performance_evidence(
    configured_gpu: tuple[Path, Mock], monkeypatch: pytest.MonkeyPatch, record: str
) -> None:
    """历史测速只供诊断，不能限制已选 GPU、触发补测或改写配置。"""
    from mineru.model.runtime import onnx_device

    settings = importlib.import_module("mineru.config")
    config_file, measurement = configured_gpu
    path = cache.calibration_record_path()
    if record == "invalid":
        path.write_text("broken", encoding="utf-8")
    elif record not in {"missing", "read_error"}:
        adapter = _adapter() if onnx_device.sys.platform == "win32" else WebGPUDevice(0x8086, 0x7D51)
        if record == "stale":
            adapter = replace(adapter, device_id=1)
        cache.save_calibration_record(
            adapter, _measurement(gpu_elapsed=3.0 if record == "slow" else 1.0, decision_valid=record != "unstable")
        )
    if record == "read_error":
        monkeypatch.setattr(recommendation, "inspect_calibration_record", Mock(side_effect=OSError("record unavailable")))
    evidence_before = path.read_bytes() if path.exists() else None
    config_before = config_file.read_bytes()
    writes = Mock(side_effect=AssertionError("configured GPU must not write configuration"))
    performance = Mock(side_effect=AssertionError("configured GPU must not require performance evidence"))
    monkeypatch.setattr(settings, "update_config_file", writes)
    monkeypatch.setattr(cache, "calibrated_gpu_is_faster", performance)
    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: 3)

    report = recommendation.recommend_tiers()

    assert report.pdf_image_tiers == TIERS and report.default_request_tier == "standard"
    assert report.reason == "configured_gpu" and report.device.endswith(":gpu")
    assert report.calibration.status == "skipped" and report.calibration.reason == "configured_gpu"
    if record == "read_error":
        assert report.calibration.evidence is None
        assert any("record unavailable" in warning for warning in report.warnings)
    else:
        assert report.calibration.evidence is not None
    assert config_file.read_bytes() == config_before
    assert (path.read_bytes() if path.exists() else None) == evidence_before
    measurement.assert_not_called()
    performance.assert_not_called()
    writes.assert_not_called()


@pytest.mark.parametrize("cores", [3, 4])
@pytest.mark.parametrize("amount", [None, 5_905_580_032 - 1, 5_905_580_032, 6 * 1024**3 - 1])
def test_configured_gpu_memory_threshold(
    configured_gpu: tuple[Path, Mock], monkeypatch: pytest.MonkeyPatch, cores: int, amount: int | None
) -> None:
    """配置 GPU 仍遵守精确内存门槛，内存不足时按 CPU 核数降级且不测速。"""
    config_file, measurement = configured_gpu
    before = config_file.read_bytes()
    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: cores)
    monkeypatch.setattr(
        recommendation, "native_gpu_memory", lambda identity, *, device: GpuMemory(device, identity, amount, "budget", "test")
    )

    report = recommendation.recommend_tiers()

    enough = amount is not None and amount >= 5_905_580_032
    expected = TIERS if enough else TIERS[:2] if cores >= 4 else ("flash",)
    assert report.pdf_image_tiers == expected
    assert report.default_request_tier == ("standard" if enough else expected[-1])
    assert report.gpu_memory.memory_bytes == amount
    if cores >= 4 or enough:
        assert report.device.endswith(":gpu")
        assert report.calibration.status == "skipped" and report.calibration.reason == "configured_gpu"
        assert report.reason == (
            "configured_gpu" if enough else "gpu_memory_unknown" if amount is None else "gpu_memory_insufficient"
        )
    measurement.assert_not_called()
    assert config_file.read_bytes() == before
    assert not cache.calibration_record_path().exists()


def test_configured_gpu_does_not_require_calibration_threshold(
    configured_gpu: tuple[Path, Mock], monkeypatch: pytest.MonkeyPatch
) -> None:
    """性能阈值仅约束自动测速，不能阻断已选 GPU 的内存判定。"""
    from mineru.model.runtime import onnx_device

    monkeypatch.setitem(onnx_device.ONNX_GPU_TIME_RATIO_THRESHOLDS, "dml", None)
    monkeypatch.setitem(onnx_device.ONNX_GPU_TIME_RATIO_THRESHOLDS, "webgpu", None)
    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: 3)
    assert recommendation.recommend_tiers().pdf_image_tiers == TIERS
    configured_gpu[1].assert_not_called()


@pytest.mark.parametrize("inventory", ["hidden", "mismatch", "ambiguous"])
def test_configured_gpu_still_requires_visible_matching_device(
    configured_gpu: tuple[Path, Mock], monkeypatch: pytest.MonkeyPatch, inventory: str
) -> None:
    """配置 GPU 不能借用不可见、另一张卡或身份不唯一的显存来开放高级档位。"""
    from mineru.model.runtime import llama_device

    device = LlamaDevice("Vulkan0", "test", "vulkan", 0x8086, 1 if inventory == "mismatch" else 0x7D51)
    devices = () if inventory == "hidden" else (device, device) if inventory == "ambiguous" else (device,)
    monkeypatch.setattr(llama_device, "discover_llama_devices", lambda: devices)
    report = recommendation.recommend_tiers()
    assert report.device.endswith(":gpu") and report.pdf_image_tiers == TIERS[:2]
    assert report.reason == ("no_visible_gpu" if inventory == "hidden" else "gpu_identity_unconfirmed")
    configured_gpu[1].assert_not_called()


def test_configured_gpu_cli_and_api_agree(configured_gpu: tuple[Path, Mock], monkeypatch: pytest.MonkeyPatch) -> None:
    """真实推荐同时驱动 CLI 报告和 API 能力，缺少校准记录仍默认 Standard。"""
    from fastapi.testclient import TestClient
    from typer.testing import CliRunner

    from mineru.kit.commands import hardware
    from mineru.kit.main import app as kit_app
    from mineru.parser import api_server

    monkeypatch.setattr(hardware, "_detect_data", lambda: asdict(recommendation.recommend_tiers()))
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", Mock())
    output = CliRunner().invoke(kit_app, ["hardware", "detect", "--json"])
    assert output.exit_code == 0, output.output
    data = json.loads(output.stdout)
    assert data["ceiling"] == "advanced" and data["default_request_tier"] == "standard"
    assert data["reason"] == "configured_gpu" and data["calibration"]["status"] == "skipped"
    assert data["calibration"]["reason"] == "configured_gpu" and data["calibration"]["evidence"]["status"] == "missing"
    assert "推荐启动上限：advanced" in hardware.render_detection(data)
    app = api_server.create_app(upload_dir=str(configured_gpu[0].parent / "uploads"))
    assert app.state.server_tiers == tuple(data["pdf_image_tiers"]) == TIERS
    assert app.state.default_tier == data["default_request_tier"]
    with TestClient(app) as client:
        assert tuple(row["id"] for row in client.get("/v1/tiers").json()["data"]) == TIERS
        for options in ({}, {"tier": "standard"}, {"tier": "advanced"}):
            response = client.post("/v1/tasks", json={"files": [{"file_name": "test.pdf", "bytes": 1}], **options})
            assert response.status_code == 202
            assert app.state.task_store._records[response.json()["task_id"]].options.tier == options.get("tier", "standard")
    configured_gpu[1].assert_not_called()
    assert not cache.calibration_record_path().exists()


def test_failed_recalibration_revokes_previous_fast_evidence(startup_calibration: tuple[Path, Mock]) -> None:
    """强制重新校准失败后，本次启动不能沿用已经撤销的旧 GPU 胜出证据。"""
    from mineru.model.runtime import onnx_device

    adapter = _adapter() if onnx_device.sys.platform == "win32" else WebGPUDevice(0x8086, 0x7D51)
    cache.save_calibration_record(adapter, _measurement())
    startup_calibration[1].side_effect = TimeoutError("worker")
    assert frozenset(recommendation.recommend_tiers().pdf_image_tiers) == frozenset(TIERS[:2])
    startup_calibration[1].assert_called_once()
    assert not cache.calibration_record_path().exists()


@pytest.mark.parametrize("cores", [3, 4])
@pytest.mark.parametrize("amount", [None, 5_905_580_032 - 1, 5_905_580_032, 6 * 1024**3 - 1, 8 * 1024**3])
def test_startup_gpu_memory_threshold_and_low_cpu_calibration(
    startup_calibration: tuple[Path, Mock], monkeypatch: pytest.MonkeyPatch, cores: int, amount: int | None
) -> None:
    """低核数的达标 GPU 仍能完成首次校准；四核小显存 GPU 只加速 Basic。"""
    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: cores)
    monkeypatch.setattr(
        recommendation, "native_gpu_memory", lambda identity, *, device: GpuMemory(device, identity, amount, "budget", "test")
    )
    enough = amount is not None and amount >= 5_905_580_032
    expected = TIERS if enough else ("flash", "basic") if cores >= 4 else ("flash",)
    report = recommendation.recommend_tiers()
    assert report.pdf_image_tiers == expected
    assert report.default_request_tier == ("standard" if enough else expected[-1])
    assert startup_calibration[1].call_count == int(cores >= 4 or enough)
    assert report.gpu_memory.memory_bytes == amount
    if cores >= 4 and not enough:
        assert report.device.endswith(":gpu") and report.calibration.status == "completed"
        assert report.reason == ("gpu_memory_unknown" if amount is None else "gpu_memory_insufficient")


@pytest.mark.parametrize(
    "failure", ["configured_cpu", "selection_failed", "no_hardware_gpu", "webgpu_architecture_policy", "raise"]
)
def test_final_cpu_result_cannot_be_upgraded_by_discrete_inventory(
    startup_calibration: tuple[Path, Mock], monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """即使初选发现达标独显，最终 CPU 或选型异常也必须撤销高级能力。"""
    from mineru.model.runtime import onnx_device

    candidate = recommendation.TierHardware(4, True, "discrete_gpu", GpuMemory("gpu", GpuIdentity(), 24 * 1024**3))
    monkeypatch.setattr(recommendation, "detect_tier_hardware", lambda: candidate)
    selection = Mock(
        return_value=onnx_device.OnnxSelection(reason=failure),
        side_effect=RuntimeError("driver") if failure == "raise" else None,
    )
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", selection)
    report = recommendation.recommend_tiers()
    assert report.pdf_image_tiers == ("flash", "basic") and report.default_request_tier == "basic"
    selection.assert_called_once()


@pytest.mark.parametrize("amount", [None, 5_905_580_032 - 1, 5_905_580_032, 6 * 1024**3 - 1])
def test_apple_onnx_path_also_requires_budget(monkeypatch: pytest.MonkeyPatch, amount: int | None) -> None:
    """ONNX 小模型配合 Apple Metal 的旧免测速路径同样受预算门槛约束。"""
    monkeypatch.delenv("MINERU_LLAMA_CPP_BACKEND", raising=False)
    monkeypatch.setattr(recommendation, "sys", SimpleNamespace(platform="darwin"))
    monkeypatch.setattr(recommendation, "effective_cpu_count", lambda: 4)
    monkeypatch.setattr(recommendation, "_apple_metal_available", lambda: True)
    monkeypatch.setattr(
        recommendation, "torch_gpu_memory", lambda device: GpuMemory(device, GpuIdentity(), amount, "budget", "metal")
    )
    report = recommendation.recommend_tiers()
    assert report.pdf_image_tiers == (TIERS if amount is not None and amount >= 5_905_580_032 else TIERS[:2])
    assert report.device == "mps" and report.calibration.reason == "apple_silicon_metal"


@pytest.mark.parametrize("backend", ["dml", "webgpu"])
def test_shared_classification_preserves_existing_calibration_fingerprint(evidence_path: Path, backend: str) -> None:
    """分类和诊断字段变化不废弃旧校准格式，真实设备或绑定变化仍使证据失效。"""
    adapter = _adapter() if backend == "dml" else WebGPUDevice(0x8086, 0x7D51)
    cache.save_calibration_record(adapter, _measurement())
    identity = json.loads(evidence_path.read_text(encoding="utf-8"))["fingerprint"]["adapter"]
    assert set(identity) == (
        {"vendor_id", "device_id", "name", "luid", "uma"}
        if backend == "dml"
        else {"vendor_id", "device_id", "metadata", "vulkan_device"}
    )
    assert cache.calibrated_gpu_is_faster(replace(adapter, kind="unknown", kind_source="new_probe"), 0.8)
    assert not cache.calibrated_gpu_is_faster(replace(adapter, device_id=1), 0.8)
    if backend == "webgpu":
        assert not cache.calibrated_gpu_is_faster(replace(adapter, vulkan_device=(0x8086, 0x7D51)), 0.8)
