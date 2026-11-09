"""验证 health 的设备字段只读、惰性出现及启动选型一致性。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from mineru.config import VlmConfig, get_config
from mineru.model.runtime import device, device_state, onnx_device
from mineru.model.runtime.device_types import OnnxSelection, TorchDeviceSelection
from mineru.parser import api_server


def test_lazy_health_does_not_initialize_device(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """未初始化时保持原 health 形状，重复请求也不能触发设备选型。"""
    forbidden = Mock(side_effect=AssertionError("health must not select devices"))
    monkeypatch.setattr(device_state, "resolve_runtime_device", forbidden)
    monkeypatch.setattr(onnx_device, "resolve_onnx_device", forbidden)
    monkeypatch.setattr(device, "resolve_torch_device", forbidden)
    app = api_server.create_app(upload_dir=str(tmp_path), tiers={"flash"})
    with TestClient(app) as client:
        for _ in range(3):
            response = client.get("/v1/health")
            assert response.status_code == 200
            assert set(response.json()) == {"status", "version", "features"}
    forbidden.assert_not_called()
    assert device_state.cached_runtime_device() is None


@pytest.mark.parametrize("selection", [OnnxSelection(), OnnxSelection("gpu"), TorchDeviceSelection("mps")])
def test_health_reports_published_mode_without_reselection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: OnnxSelection | TorchDeviceSelection
) -> None:
    """首次发布后 health 只读同一模式，配置及局部模型覆盖不改变响应。"""
    backend = "torch" if isinstance(selection, TorchDeviceSelection) else "onnx"
    monkeypatch.setattr(get_config().model, "small_backend", backend)
    probe = Mock(return_value=selection)
    monkeypatch.setattr(device if backend == "torch" else onnx_device, f"resolve_{backend}_device", probe)
    app = api_server.create_app(upload_dir=str(tmp_path), tiers={"flash"})
    with TestClient(app) as client:
        assert "runtime" not in client.get("/v1/health").json()
        runtime = device_state.resolve_runtime_device()
        monkeypatch.setattr(get_config().model, "onnx_device", "gpu")
        with device_state.model_device_scope(device_state.resolve_model_device("torch", "cpu")):
            response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json()["runtime"] == {"device": runtime.device}
    probe.assert_called_once()


def test_auto_tier_and_health_share_startup_decision(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """自动启动产生的设备决策在 health 中直接可见，无需模型预加载。"""
    monkeypatch.setattr(get_config().model, "small_backend", "torch")
    probe = Mock(return_value=TorchDeviceSelection("cuda:1"))
    monkeypatch.setattr(device, "resolve_torch_device", probe)
    monkeypatch.setattr(api_server, "ensure_tier_runtime_dependencies", Mock())
    app = api_server.create_app(upload_dir=str(tmp_path), vlm_config=VlmConfig(engine="llama-cpp"))
    with TestClient(app) as client:
        assert client.get("/v1/health").json()["runtime"] == {"device": "gpu"}
        assert len(client.get("/v1/tiers").json()["data"]) == 4
    probe.assert_called_once()


def test_health_model_accepts_legacy_payload_and_rejects_invalid_mode() -> None:
    """旧服务缺少设备字段仍合法，新字段仅接受 CPU/GPU 二值。"""
    assert api_server.HealthResponse.model_validate({"status": "ok", "version": "test"}).runtime is None
    with pytest.raises(ValidationError):
        api_server.HealthResponse.model_validate({"version": "test", "runtime": {"device": "mps"}})
