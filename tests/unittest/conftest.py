"""Unit tests must not benchmark real GPUs or persist choices in the developer's config."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def isolate_onnx_device_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Individual policy tests explicitly override this CPU default with isolated files."""
    from mineru.config import get_config
    from mineru.model.runtime import onnx_device

    monkeypatch.setattr(get_config().model, "onnx_device", "cpu")
    monkeypatch.setattr(onnx_device, "_selections", {})
