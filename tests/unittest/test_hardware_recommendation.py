"""验证 Torch 免校准、实际设备可用性及命令/API/llama.cpp 的统一决策。"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest
from typer.testing import CliRunner

from mineru.config import VlmConfig, get_config
from mineru.kit.commands import hardware as hardware_cmd
from mineru.kit.main import app as kit_app
from mineru.model.runtime import device, llama_device, onnx_calibration_cache, onnx_device, tier_recommendation
from mineru.model.runtime.llama_probe import LlamaDevice
from mineru.parser import api_server
from mineru.types import TIERS


@pytest.fixture
def torch_environment(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """为 Torch 分支设置最小运行环境，并禁止任何 ONNX 选型或证据写入。"""
    monkeypatch.setattr(get_config().model, "small_backend", "torch")
    monkeypatch.setattr(get_config().model.vlm, "engine", "llama-cpp")
    monkeypatch.setattr(get_config().model.vlm, "server_url", None)
    monkeypatch.setattr(tier_recommendation, "effective_cpu_count", lambda: 8)
    monkeypatch.setattr(device, "module_available", lambda name: True)
    forbidden = Mock(side_effect=AssertionError("Torch must not calibrate or write ONNX settings"))
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", forbidden)
    monkeypatch.setattr(onnx_calibration_cache, "save_calibration_record", forbidden)
    monkeypatch.setattr(onnx_calibration_cache, "invalidate_calibration_record", forbidden)
    monkeypatch.setattr(tier_recommendation, "_onnx_gpu_hardware", forbidden)
    return forbidden


def _fake_torch(monkeypatch: pytest.MonkeyPatch, kind: str, available: bool, count: int = 1) -> None:
    """只提供可用性 API，调用模型、张量或设备分配接口都会因缺失而失败。"""
    api = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
        current_device=lambda: 0,
        get_device_properties=lambda index: SimpleNamespace(total_memory=8 * 1024**3, is_integrated_gpu=False),
    )
    unavailable = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    fake = SimpleNamespace(
        mps=SimpleNamespace(recommended_max_memory=lambda: 8 * 1024**3),
        cuda=api if kind == "cuda" else unavailable,
        xpu=api if kind == "xpu" else unavailable,
        backends=SimpleNamespace(mps=api if kind == "mps" else unavailable),
    )
    monkeypatch.setitem(sys.modules, "torch", fake)
    monkeypatch.setitem(sys.modules, "torch_npu", None)


@pytest.mark.parametrize("backend", ["torch", "auto"])
@pytest.mark.parametrize("accelerator", ["cuda", "mps", "xpu"])
@pytest.mark.parametrize("engine", ["llama-cpp", "vllm", "lmdeploy"])
def test_torch_accelerator_skips_onnx_for_every_engine(
    torch_environment: Mock, monkeypatch: pytest.MonkeyPatch, backend: str, accelerator: str, engine: str
) -> None:
    """显式和自动 Torch 都直接使用有效加速设备，且不依赖 llama 库存。"""
    monkeypatch.setattr(get_config().model, "small_backend", backend)
    monkeypatch.delenv("MINERU_DEVICE_MODE", raising=False)
    monkeypatch.setenv("MINERU_LLAMA_CPP_BACKEND", "cpu")
    _fake_torch(monkeypatch, accelerator, True)
    report = tier_recommendation.recommend_tiers(vlm_config=VlmConfig(engine=engine))
    assert report.ceiling == "advanced" and report.pdf_image_tiers == TIERS
    assert report.default_request_tier == "standard"
    assert report.small_backend == "torch" and report.device == accelerator
    assert report.vlm_engine == engine
    assert report.calibration.status == "skipped" and report.calibration.reason == "torch_backend"
    assert report.calibration.evidence is None and not report.warnings
    torch_environment.assert_not_called()


@pytest.mark.parametrize(
    ("requested", "available", "expected_warning"),
    [("cpu", True, False), ("cuda", False, True), ("cuda:2", True, True), ("xpu:-1", True, True)],
)
@pytest.mark.parametrize("cores", [3, 4])
def test_torch_cpu_and_invalid_explicit_device_are_conservative(
    torch_environment: Mock,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    available: bool,
    expected_warning: bool,
    cores: int,
) -> None:
    """显式字符串不能伪造 GPU 可用性，CPU 边界按当前进程可用核数判断。"""
    monkeypatch.setenv("MINERU_DEVICE_MODE", requested)
    monkeypatch.setattr(tier_recommendation, "effective_cpu_count", lambda: cores)
    _fake_torch(monkeypatch, requested.split(":")[0], available)
    report = tier_recommendation.recommend_tiers()
    assert report.device == "cpu" and report.ceiling == ("basic" if cores == 4 else "flash")
    assert bool(report.warnings) == expected_warning
    assert get_config().model.small_backend == "torch"
    assert os.environ["MINERU_DEVICE_MODE"] == requested
    torch_environment.assert_not_called()


def test_torch_probe_exception_keeps_warning(torch_environment: Mock, monkeypatch: pytest.MonkeyPatch) -> None:
    """运行时探测异常形成有解释的 CPU 推荐，而非伪造 GPU 或开始 OCR 测速。"""
    monkeypatch.setattr(device, "get_device", Mock(side_effect=RuntimeError("driver error")))
    report = tier_recommendation.recommend_tiers()
    assert report.device == "cpu" and report.ceiling == "basic"
    assert "driver error" in report.warnings[0]
    torch_environment.assert_not_called()


@pytest.mark.parametrize("cores", [3, 4])
@pytest.mark.parametrize("accelerator", ["cuda", "mps", "xpu"])
@pytest.mark.parametrize("amount", [None, 5_905_580_032 - 1, 5_905_580_032, 6 * 1024**3 - 1])
def test_torch_memory_gate_preserves_onnx_isolation(
    torch_environment: Mock, monkeypatch: pytest.MonkeyPatch, cores: int, accelerator: str, amount: int | None
) -> None:
    """真实推荐链路检查精确容量边界，内存不足和未知均不触发 ONNX。"""
    monkeypatch.setattr(tier_recommendation, "effective_cpu_count", lambda: cores)
    monkeypatch.setenv("MINERU_DEVICE_MODE", accelerator)
    _fake_torch(monkeypatch, accelerator, True)
    fake = sys.modules["torch"]
    if accelerator == "mps":
        fake.mps.recommended_max_memory = lambda: amount
    else:
        getattr(fake, accelerator).get_device_properties = lambda index: SimpleNamespace(
            total_memory=amount, is_integrated_gpu=False
        )
    report = tier_recommendation.recommend_tiers()
    expected = TIERS if amount is not None and amount >= 5_905_580_032 else TIERS[:2] if cores >= 4 else ("flash",)
    assert report.pdf_image_tiers == expected
    assert report.gpu_memory.memory_bytes == amount and report.calibration.reason == "torch_backend"
    assert bool(report.warnings) is (amount is None)
    torch_environment.assert_not_called()


def test_auto_basic_api_defaults_and_rejects_higher_tiers(
    torch_environment: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """经过真实推荐形成 Basic 服务，文本/JSON、能力列表和接单错误保持一致。"""
    monkeypatch.setenv("MINERU_DEVICE_MODE", "cuda")
    _fake_torch(monkeypatch, "cuda", True)
    sys.modules["torch"].cuda.get_device_properties = lambda index: SimpleNamespace(total_memory=5_905_580_032 - 1)
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", Mock())
    report = tier_recommendation.recommend_tiers()
    monkeypatch.setattr(hardware_cmd, "_detect_data", lambda: asdict(report))
    output = CliRunner().invoke(kit_app, ["hardware", "detect", "--json"])
    data = json.loads(output.stdout)
    assert data["schema_version"] == 1 and data["gpu_memory"]["memory_bytes"] == 5_905_580_032 - 1
    rendered = hardware_cmd.render_detection(data)
    assert "gpu_memory_insufficient" in rendered and "5905580031 bytes" in rendered and "torch.cuda.total_memory" in rendered
    app = api_server.create_app(upload_dir=str(tmp_path))
    with TestClient(app) as client:
        assert tuple(row["id"] for row in client.get("/v1/tiers").json()["data"]) == ("flash", "basic")
        for options in ({}, {"tier": None}):
            response = client.post("/v1/tasks", json={"files": [{"file_name": "test.pdf", "bytes": 1}], **options})
            assert response.status_code == 202
            assert app.state.task_store._records[response.json()["task_id"]].options.tier == "basic"
        for tier in ("standard", "advanced"):
            response = client.post("/v1/tasks", json={"files": [{"file_name": "test.pdf", "bytes": 1}], "tier": tier})
            assert response.status_code == 400 and response.json()["error"]["code"] == "invalid_request"
            assert response.json()["error"]["param"] == "tier"
    torch_environment.assert_not_called()


def test_remote_vlm_and_api_override_match_detect(
    torch_environment: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """API 显式 VLM 配置传入共享推荐，命令报告与首次服务能力保持一致。"""
    _fake_torch(monkeypatch, "cuda", True)
    monkeypatch.setenv("MINERU_DEVICE_MODE", "cuda")
    settings = VlmConfig(engine="mlx", server_url="http://localhost:9999", api_key="not-for-report")
    expected = tier_recommendation.recommend_tiers(vlm_config=settings)
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", Mock())
    recommend = Mock(wraps=tier_recommendation.recommend_tiers)
    monkeypatch.setattr(tier_recommendation, "recommend_tiers", recommend)
    api = api_server.create_app(upload_dir=str(tmp_path), vlm_config=settings)
    assert recommend.call_args.kwargs["vlm_config"] == settings
    assert expected.vlm_engine == "http-client"
    with TestClient(api) as client:
        assert tuple(row["id"] for row in client.get("/v1/tiers").json()["data"]) == expected.pdf_image_tiers
        assert api.state.default_tier == expected.default_request_tier
    monkeypatch.setattr(hardware_cmd, "_detect_data", lambda: asdict(expected))
    result = CliRunner().invoke(kit_app, ["hardware", "detect", "--json"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["ceiling"] == api.state.server_tiers[-1]
    assert "not-for-report" not in result.stdout
    torch_environment.assert_not_called()


@pytest.mark.parametrize("accelerated", [True, False])
def test_torch_llama_integrated_route_does_not_return_to_onnx(
    torch_environment: Mock, monkeypatch: pytest.MonkeyPatch, accelerated: bool
) -> None:
    """llama 自动集显复用 Torch 可用性，后续绑定及 Intel 兼容策略继续执行。"""
    monkeypatch.setattr(llama_device, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.delenv("MINERU_LLAMA_CPP_BACKEND", raising=False)
    monkeypatch.delenv("GGML_VK_SERIALIZE_SUBMISSIONS", raising=False)
    monkeypatch.setenv("MINERU_DEVICE_MODE", "xpu" if accelerated else "cpu")
    _fake_torch(monkeypatch, "xpu", accelerated)
    native = LlamaDevice("Vulkan0", "Arc", "vulkan", 0x8086, 0x7D51, 0, kind="integrated", kind_source="vulkan")
    monkeypatch.setattr(llama_device, "discover_llama_devices", lambda: (native,))
    bind = Mock(side_effect=lambda selection: selection)
    monkeypatch.setattr(llama_device, "_bind", bind)
    result = llama_device.resolve_llama_device()
    assert result.device == ("gpu" if accelerated else "cpu")
    if accelerated:
        assert result.devices == (native,)
        assert dict(result.environment)["GGML_VK_SERIALIZE_SUBMISSIONS"] == "1"
        bind.assert_called_once()
    else:
        bind.assert_not_called()
    torch_environment.assert_not_called()
