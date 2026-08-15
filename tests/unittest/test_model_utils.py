from typing import Any

from mineru.utils import model_utils


def test_trim_process_memory_invokes_malloc_trim(monkeypatch: Any) -> None:
    calls = []

    def fake_malloc_trim(padding: int) -> int:
        calls.append(padding)
        return 1

    monkeypatch.setattr(model_utils, "_get_malloc_trim", lambda: fake_malloc_trim)

    model_utils.trim_process_memory()

    assert calls == [0]


def test_trim_process_memory_is_optional(monkeypatch: Any) -> None:
    monkeypatch.setattr(model_utils, "_get_malloc_trim", lambda: None)

    model_utils.trim_process_memory()


def test_clean_memory_runs_process_heap_trim(monkeypatch: Any) -> None:
    calls = []
    monkeypatch.setattr(model_utils, "trim_process_memory", lambda: calls.append(True))

    model_utils.clean_memory("cpu")

    assert calls == [True]


def test_get_malloc_trim_configures_available_symbol(monkeypatch: Any) -> None:
    class FakeTrim:
        argtypes = None
        restype = None

        def __call__(self, padding: int) -> int:
            return padding

    fake_trim = FakeTrim()
    fake_libc = type("FakeLibc", (), {"malloc_trim": fake_trim})()
    monkeypatch.setattr(model_utils.sys, "platform", "linux")
    monkeypatch.setattr(model_utils.ctypes, "CDLL", lambda _: fake_libc)
    model_utils._get_malloc_trim.cache_clear()

    try:
        assert model_utils._get_malloc_trim() is fake_trim
        assert fake_trim.argtypes == [model_utils.ctypes.c_size_t]
        assert fake_trim.restype is model_utils.ctypes.c_int
    finally:
        model_utils._get_malloc_trim.cache_clear()
