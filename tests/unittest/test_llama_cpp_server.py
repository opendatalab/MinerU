from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from mineru.kit.vlm_server import llama_cpp_server


@pytest.mark.parametrize(
    ("platform", "expected_name"),
    [("win32", "llama-server.exe"), ("darwin", "llama-server"), ("linux", "llama-server")],
)
def test_llama_server_binary_matches_platform_executable_name(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, expected_name: str
) -> None:
    """Windows wheel 分发 llama-server.exe，其余平台无后缀。"""
    fake_pkg = types.ModuleType("mineru_llama_cpp")
    fake_pkg.__file__ = str(tmp_path / "mineru_llama_cpp" / "__init__.py")
    monkeypatch.setitem(sys.modules, "mineru_llama_cpp", fake_pkg)
    monkeypatch.setattr(sys, "platform", platform)

    binary = llama_cpp_server.llama_server_binary()

    assert binary == tmp_path / "mineru_llama_cpp" / "bin" / expected_name
