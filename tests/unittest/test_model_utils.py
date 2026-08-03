# Copyright (c) Opendatalab. All rights reserved.
import sys
from unittest import mock

import pytest

from mineru.utils.model_utils import (
    clean_memory,
    is_heap_trim_enabled,
    trim_process_heap,
)


class TestTrimProcessHeap:
    """Regression tests for the optional glibc heap-trimming hook."""

    def test_trim_process_heap_does_not_raise(self):
        """The helper must be safe to call on every supported platform."""
        trim_process_heap()

    def test_trim_process_heap_swallows_missing_symbol(self):
        """On allocators without ``malloc_trim`` the call should be a no-op."""
        with mock.patch("ctypes.CDLL") as mock_cdll:
            mock_cdll.return_value = mock.Mock(spec=[])
            trim_process_heap()

    def test_trim_process_heap_swallows_cdll_error(self):
        """Any failure to load the C library must not propagate."""
        with mock.patch("ctypes.CDLL", side_effect=OSError("no libc")):
            trim_process_heap()


class TestIsHeapTrimEnabled:
    def test_default_enabled_on_linux(self, monkeypatch):
        """When ``MINERU_MALLOC_TRIM`` is unset the feature defaults to on for Linux."""
        monkeypatch.delenv("MINERU_MALLOC_TRIM", raising=False)
        with mock.patch.object(sys, "platform", "linux"):
            assert is_heap_trim_enabled() is True

    def test_default_disabled_on_non_linux(self, monkeypatch):
        """When unset the feature defaults to off outside Linux."""
        monkeypatch.delenv("MINERU_MALLOC_TRIM", raising=False)
        with mock.patch.object(sys, "platform", "darwin"):
            assert is_heap_trim_enabled() is False

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("1", True),
            ("true", True),
            ("True", True),
            ("yes", True),
            ("on", True),
            ("0", False),
            ("false", False),
            ("False", False),
            ("no", False),
            ("off", False),
            ("disable", False),
            ("disabled", False),
        ],
    )
    def test_env_var_overrides_default(self, monkeypatch, value, expected):
        monkeypatch.setenv("MINERU_MALLOC_TRIM", value)
        assert is_heap_trim_enabled() is expected


class TestCleanMemoryHeapTrim:
    def test_clean_memory_trims_heap_when_enabled(self, monkeypatch):
        """``clean_memory`` should invoke heap trimming when explicitly enabled."""
        monkeypatch.setenv("MINERU_MALLOC_TRIM", "1")

        with mock.patch("mineru.utils.model_utils.gc.collect") as mock_gc:
            with mock.patch("mineru.utils.model_utils.trim_process_heap") as mock_trim:
                clean_memory("cpu", trim_heap=True)
                mock_gc.assert_called_once()
                mock_trim.assert_called_once()

    def test_clean_memory_skips_heap_trim_when_disabled(self, monkeypatch):
        """``clean_memory`` should not trim when explicitly disabled."""
        monkeypatch.setenv("MINERU_MALLOC_TRIM", "0")

        with mock.patch("mineru.utils.model_utils.gc.collect") as mock_gc:
            with mock.patch("mineru.utils.model_utils.trim_process_heap") as mock_trim:
                clean_memory("cpu", trim_heap=False)
                mock_gc.assert_called_once()
                mock_trim.assert_not_called()

    def test_clean_memory_uses_env_default_when_trim_heap_unset(self, monkeypatch):
        """When ``trim_heap`` is None the env var / platform default decides."""
        monkeypatch.setenv("MINERU_MALLOC_TRIM", "1")

        with mock.patch("mineru.utils.model_utils.gc.collect"):
            with mock.patch("mineru.utils.model_utils.trim_process_heap") as mock_trim:
                clean_memory("cpu", trim_heap=None)
                mock_trim.assert_called_once()
