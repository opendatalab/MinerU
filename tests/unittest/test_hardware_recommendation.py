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
from mineru.model.runtime import device, gpu_memory, llama_device, onnx_calibration_cache, onnx_device, tier_recommendation
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
    monkeypatch.setattr(gpu_memory, "native_gpu_memory", forbidden)
    monkeypatch.setattr(gpu_memory, "torch_gpu_memory", forbidden)
    return forbidden


def _fake_torch(monkeypatch: pytest.MonkeyPatch, kind: str, available: bool, count: int = 1) -> None:
    """只提供可用性 API，调用模型、张量或设备分配接口都会因缺失而失败。"""
    api = SimpleNamespace(
        is_available=lambda: available,
        device_count=lambda: count,
    )
    unavailable = SimpleNamespace(is_available=lambda: False, device_count=lambda: 0)
    fake = SimpleNamespace(
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
@pytest.mark.parametrize("cores", [1, 3, 4, 9, 10, 11, 12, 32])
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
    assert report.device == "cpu"
    assert report.ceiling == ("standard" if cores >= 12 else "basic" if cores >= 4 else "flash")
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


@pytest.mark.parametrize("cores", [1, 10])
@pytest.mark.parametrize("accelerator", ["cuda", "mps", "xpu"])
@pytest.mark.parametrize("amount", [None, 1024**3, "error"])
def test_torch_memory_does_not_affect_recommendation(
    torch_environment: Mock, monkeypatch: pytest.MonkeyPatch, cores: int, accelerator: str, amount: int | str | None
) -> None:
    """容量偏小、未知或读取异常均不影响设备结论，且无需任何显存查询。"""
    monkeypatch.setattr(tier_recommendation, "effective_cpu_count", lambda: cores)
    monkeypatch.setenv("MINERU_DEVICE_MODE", accelerator)
    _fake_torch(monkeypatch, accelerator, True)
    memory = Mock(return_value=amount, side_effect=RuntimeError("memory unavailable") if amount == "error" else None)
    fake = sys.modules["torch"]
    if accelerator == "mps":
        fake.mps = SimpleNamespace(recommended_max_memory=memory)
    else:
        getattr(fake, accelerator).get_device_properties = memory
    report = tier_recommendation.recommend_tiers()
    assert report.pdf_image_tiers == TIERS and report.default_request_tier == "standard"
    assert report.gpu_memory is None and report.calibration.reason == "torch_backend"
    assert not report.warnings
    memory.assert_not_called()
    torch_environment.assert_not_called()


@pytest.mark.parametrize(("cores", "expected"), [(4, TIERS[:2]), (10, TIERS[:2]), (11, TIERS[:2]), (12, TIERS[:3])])
def test_auto_cpu_api_defaults_and_rejects_higher_tiers(
    torch_environment: Mock, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cores: int, expected: tuple[str, ...]
) -> None:
    """真实 CPU 推荐驱动文本/JSON、API 能力、默认请求以及显式请求的上下界。"""
    monkeypatch.setenv("MINERU_DEVICE_MODE", "cpu")
    monkeypatch.setattr(tier_recommendation, "effective_cpu_count", lambda: cores)
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", Mock())
    report = tier_recommendation.recommend_tiers()
    monkeypatch.setattr(hardware_cmd, "_detect_data", lambda: asdict(report))
    output = CliRunner().invoke(kit_app, ["hardware", "detect", "--json"])
    assert output.exit_code == 0, output.output
    data = json.loads(output.stdout)
    assert data["schema_version"] == 1 and data["gpu_memory"] is None
    rendered = hardware_cmd.render_detection(data)
    assert f"推荐启动上限：{expected[-1]}" in rendered
    assert "5.5" not in rendered and "GPU 内存" not in rendered
    app = api_server.create_app(upload_dir=str(tmp_path))
    with TestClient(app) as client:
        assert tuple(row["id"] for row in client.get("/v1/tiers").json()["data"]) == expected
        for options in ({}, {"tier": None}, {"tier": expected[-1]}):
            response = client.post("/v1/tasks", json={"files": [{"file_name": "test.pdf", "bytes": 1}], **options})
            assert response.status_code == 202
            assert app.state.task_store._records[response.json()["task_id"]].options.tier == expected[-1]
        for tier in TIERS[len(expected) :]:
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
    """llama 自动模式仅复用 Torch 可用性，不探测显卡或注入 Intel 兼容设置。"""
    monkeypatch.setattr(llama_device, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.delenv("MINERU_LLAMA_CPP_BACKEND", raising=False)
    monkeypatch.delenv("GGML_VK_SERIALIZE_SUBMISSIONS", raising=False)
    monkeypatch.setenv("MINERU_DEVICE_MODE", "xpu" if accelerated else "cpu")
    _fake_torch(monkeypatch, "xpu", accelerated)
    result = llama_device.resolve_llama_device()
    assert result.device == ("gpu" if accelerated else "cpu")
    assert result.environment == ()
    assert tier_recommendation.recommend_tiers().ceiling == ("advanced" if accelerated else "basic")
    torch_environment.assert_not_called()


@pytest.mark.parametrize("accelerated", [False, True])
def test_torch_cpu_count_failure_preserves_device_result(
    torch_environment: Mock, monkeypatch: pytest.MonkeyPatch, accelerated: bool
) -> None:
    """CPU 计数异常回退一核，已验证的 Torch 加速设备仍开放全部档位。"""
    monkeypatch.setattr(tier_recommendation, "effective_cpu_count", Mock(side_effect=OSError("affinity")))
    monkeypatch.setenv("MINERU_DEVICE_MODE", "cuda" if accelerated else "cpu")
    _fake_torch(monkeypatch, "cuda", accelerated)
    report = tier_recommendation.recommend_tiers()
    assert report.cpu_count == 1 and report.pdf_image_tiers == (TIERS if accelerated else TIERS[:1])
    assert any("affinity" in warning for warning in report.warnings)
    torch_environment.assert_not_called()
