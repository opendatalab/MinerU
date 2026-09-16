from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

import mineru.utils.stdio as stdio


def test_configure_standard_streams_uses_utf8(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Stream:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def reconfigure(self, **kwargs: str) -> None:
            self.calls.append(kwargs)

    stdout = _Stream()
    stderr = _Stream()
    monkeypatch.setattr(stdio.sys, "stdout", stdout)
    monkeypatch.setattr(stdio.sys, "stderr", stderr)

    stdio.configure_standard_streams()

    expected = [{"encoding": "utf-8", "errors": "backslashreplace"}]
    assert stdout.calls == expected
    assert stderr.calls == expected


def test_utf8_subprocess_env_overrides_inherited_stdio_encoding() -> None:
    env = stdio.utf8_subprocess_env(
        {
            "PYTHONIOENCODING": "gbk",
            "MINERU_TEST_VALUE": "preserved",
        }
    )

    assert env["PYTHONIOENCODING"] == "utf-8:backslashreplace"
    assert env["MINERU_TEST_VALUE"] == "preserved"


def test_console_scripts_use_standard_stream_configuring_entrypoints() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    scripts = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))["project"]["scripts"]

    assert scripts["mineru"] == "mineru.cli.main:main"
    assert scripts["mineru-kit"] == "mineru.kit.main:main"
