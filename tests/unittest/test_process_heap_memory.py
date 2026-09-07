"""覆盖可选 CPU 堆回收的能力检测、缓存、开关与故障隔离。"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from mineru.model.runtime import memory


@pytest.fixture(autouse=True)
def isolate_allocator(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """隔离每个用例的 libc 缓存与平台，确保不会调用真实系统分配器。"""
    memory._get_malloc_trim.cache_clear()
    monkeypatch.delenv("MINERU_MALLOC_TRIM", raising=False)
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform="linux"))
    yield
    memory._get_malloc_trim.cache_clear()


@pytest.mark.parametrize("value", [None, "", "0", "false", "no", "off", "disabled", "invalid", "2"])
def test_disabled_trim_does_not_load_allocator(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    """默认关闭及非法配置不得探测 libc，更不能调用设备清理或垃圾回收。"""
    if value is not None:
        monkeypatch.setenv("MINERU_MALLOC_TRIM", value)
    load = Mock(side_effect=AssertionError("must not load libc"))
    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setattr(memory.gc, "collect", Mock(side_effect=AssertionError("must not collect")))
    monkeypatch.setattr(memory, "_optional_torch", Mock(side_effect=AssertionError("must not load torch")))
    assert memory.trim_process_heap() is False
    load.assert_not_called()


@pytest.mark.parametrize("value", ["1", "true", "YES", " On "])
@pytest.mark.parametrize("released", [0, 1])
def test_enabled_trim_caches_signature_but_not_switch(monkeypatch: pytest.MonkeyPatch, value: str, released: int) -> None:
    """只缓存正确 ABI 的底层函数，每次调用仍检查开关并报告实际回收结果。"""
    malloc_trim = Mock(return_value=released)
    load = Mock(return_value=SimpleNamespace(malloc_trim=malloc_trim))
    monkeypatch.setattr(ctypes, "CDLL", load)
    monkeypatch.setenv("MINERU_MALLOC_TRIM", value)
    assert memory.trim_process_heap() is bool(released)
    assert memory.trim_process_heap() is bool(released)
    load.assert_called_once_with(None)
    assert malloc_trim.argtypes == [ctypes.c_size_t]
    assert malloc_trim.restype is ctypes.c_int
    assert malloc_trim.call_count == 2
    malloc_trim.assert_called_with(0)
    monkeypatch.setenv("MINERU_MALLOC_TRIM", "0")
    assert memory.trim_process_heap() is False
    assert malloc_trim.call_count == 2


@pytest.mark.parametrize("platform", ["darwin", "win32"])
def test_unsupported_platform_does_not_load_libc(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """显式开启也不会在不支持的平台加载 libc。"""
    monkeypatch.setenv("MINERU_MALLOC_TRIM", "1")
    monkeypatch.setattr(memory, "sys", SimpleNamespace(platform=platform))
    load = Mock(side_effect=AssertionError("must not load libc"))
    monkeypatch.setattr(ctypes, "CDLL", load)
    assert memory.trim_process_heap() is False
    load.assert_not_called()


@pytest.mark.parametrize("failure", ["load", "symbol", "call"])
def test_allocator_failures_are_optional(monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    """libc 加载、符号缺失和底层调用失败都返回空操作结果。"""
    monkeypatch.setenv("MINERU_MALLOC_TRIM", "1")
    library = SimpleNamespace()
    if failure == "call":
        library.malloc_trim = Mock(side_effect=RuntimeError("allocator failed"))
    load = Mock(side_effect=OSError("no libc")) if failure == "load" else Mock(return_value=library)
    monkeypatch.setattr(ctypes, "CDLL", load)
    assert memory.trim_process_heap() is False
    assert memory.trim_process_heap() is False
    load.assert_called_once_with(None)


def test_trim_import_and_call_do_not_load_torch() -> None:
    """在干净解释器中验证导入无配置副作用，关闭的 CPU 回收不加载 Torch。"""
    code = """
import os
import sys
before = dict(os.environ)
from mineru.model.runtime.memory import trim_process_heap
assert dict(os.environ) == before
assert trim_process_heap() is False
assert 'torch' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "MINERU_MALLOC_TRIM": "0"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
