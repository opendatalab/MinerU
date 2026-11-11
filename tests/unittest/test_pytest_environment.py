"""验证测试环境初始化的导入边界与 pytest 生命周期顺序。"""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("configured_value", [None, "0", ""])
def test_environment_initialization_waits_for_pytest_configure(
    monkeypatch: pytest.MonkeyPatch, configured_value: str | None
) -> None:
    """单独导入配置不修改环境，执行钩子后保留显式配置并固定测试语言。"""
    names = ("ORT_DISABLE_TELEMETRY", "MINERU_LANG", "_TYPER_FORCE_DISABLE_TERMINAL")
    for name in names:
        if configured_value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, configured_value)
    environment = dict(os.environ)

    configuration = runpy.run_path(str(Path(__file__).resolve().parents[1] / "conftest.py"))

    assert dict(os.environ) == environment
    configuration["pytest_configure"]()
    assert os.environ["MINERU_LANG"] == "en"
    expected_default = "1" if configured_value is None else configured_value
    assert os.environ["ORT_DISABLE_TELEMETRY"] == expected_default
    assert os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] == expected_default


def test_pytest_configures_environment_before_test_module_import(tmp_path: Path) -> None:
    """通过真实 pytest 收集入口确认配置先于测试模块导入生效。"""
    shutil.copyfile(Path(__file__).resolve().parents[1] / "conftest.py", tmp_path / "conftest.py")
    (tmp_path / "test_environment.py").write_text(
        '''import os

assert os.environ["ORT_DISABLE_TELEMETRY"] == "1"
assert os.environ["MINERU_LANG"] == "en"
assert os.environ["_TYPER_FORCE_DISABLE_TERMINAL"] == "1"

def test_environment() -> None:
    """提供一个可收集的测试，环境断言已在模块导入时执行。"""
''',
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.pop("ORT_DISABLE_TELEMETRY", None)
    environment.pop("_TYPER_FORCE_DISABLE_TERMINAL", None)
    environment.update(MINERU_LANG="zh", PYTEST_ADDOPTS="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1", GITHUB_ACTIONS="true")

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-o", "addopts="],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 test collected" in result.stdout
