"""验证设备探测顺序与 XPU 的缓存释放、显存探测分支。"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mineru.model.runtime.device import get_device
from mineru.model.runtime.memory import clean_memory, get_vram


def _accelerator_api(available: bool) -> SimpleNamespace:
    """构造只包含设备可用性查询的加速器 API 替身。"""
    return SimpleNamespace(is_available=lambda: available)


def _fake_torch(xpu_available: bool = True, generic_available: bool = True) -> SimpleNamespace:
    """构造只暴露探测与缓存释放所需 API 的 torch 替身。"""
    return SimpleNamespace(
        cuda=_accelerator_api(False),
        backends=SimpleNamespace(mps=_accelerator_api(False)),
        xpu=_accelerator_api(xpu_available),
        gcu=_accelerator_api(generic_available),
        musa=_accelerator_api(generic_available),
        mlu=_accelerator_api(generic_available),
        sdaa=_accelerator_api(generic_available),
    )


def _install_fake_torch(monkeypatch: pytest.MonkeyPatch, fake_torch: SimpleNamespace) -> None:
    """隔离设备环境变量与可选 NPU 扩展，安装测试用 torch 替身。"""
    monkeypatch.delenv("MINERU_DEVICE_MODE", raising=False)
    monkeypatch.delenv("MINERU_VIRTUAL_VRAM_SIZE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    # sys.modules 中置 None 使 import torch_npu 直接抛 ImportError，避免宿主环境干扰探测顺序。
    monkeypatch.setitem(sys.modules, "torch_npu", None)


def test_get_device_probes_xpu_before_generic_accelerators(monkeypatch: pytest.MonkeyPatch) -> None:
    """xpu 可用时优先于国产加速后端命中。"""
    _install_fake_torch(monkeypatch, _fake_torch())
    assert get_device() == "xpu"


def test_get_device_falls_back_to_cpu_without_accelerators(monkeypatch: pytest.MonkeyPatch) -> None:
    """xpu 与国产后端都不可用时回退 CPU。"""
    _install_fake_torch(monkeypatch, _fake_torch(xpu_available=False, generic_available=False))
    assert get_device() == "cpu"


def test_configured_device_overrides_autodetect(monkeypatch: pytest.MonkeyPatch) -> None:
    """MINERU_DEVICE_MODE 显式配置优先于任何探测结果。"""
    _install_fake_torch(monkeypatch, _fake_torch())
    monkeypatch.setenv("MINERU_DEVICE_MODE", "cpu")
    assert get_device() == "cpu"


def test_clean_memory_releases_xpu_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """xpu 设备调用 torch.xpu.empty_cache 释放框架缓存。"""
    empty_cache = Mock()
    fake_torch = _fake_torch()
    fake_torch.xpu = SimpleNamespace(is_available=lambda: True, empty_cache=empty_cache)
    _install_fake_torch(monkeypatch, fake_torch)
    clean_memory("xpu")
    empty_cache.assert_called_once_with()


def test_get_vram_reads_xpu_total_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """xpu 显存按 get_device_properties().total_memory 换算为整数 GB。"""
    fake_torch = _fake_torch()
    fake_torch.xpu = SimpleNamespace(
        is_available=lambda: True,
        get_device_properties=lambda device: SimpleNamespace(total_memory=12 * 1024**3 + 200 * 1024**2),
    )
    _install_fake_torch(monkeypatch, fake_torch)
    assert get_vram("xpu") == 12
