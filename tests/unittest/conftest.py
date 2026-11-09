"""Unit tests must not benchmark real GPUs or persist choices in the developer's config."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_onnx_device_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """隔离进程设备快照和 ONNX 偏好，策略测试显式注入独立配置与探测替身。"""
    from mineru.config import get_config
    from mineru.model.runtime import device_state

    monkeypatch.setattr(get_config().model, "onnx_device", "cpu")
    monkeypatch.setattr(device_state, "_states", {})
