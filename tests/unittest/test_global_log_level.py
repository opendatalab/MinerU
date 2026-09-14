"""验证全局 Loguru 日志级别配置和自定义 sink 兼容行为。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import pytest

_SCRIPT = r"""
import io
import json
import sys
from contextlib import redirect_stderr

from loguru import logger

from mineru.utils.logger import configure_global_log_level

mode = sys.argv[1]
if mode == "filter":
    level = sys.argv[2]
    with redirect_stderr(io.StringIO()) as stderr:
        configure_global_log_level(level)
        logger.debug("DEBUG_MARKER")
        logger.info("INFO_MARKER")
    print(json.dumps({"stderr": stderr.getvalue()}))
elif mode == "custom-sink":
    custom_stream = io.StringIO()
    custom_sink = logger.add(custom_stream, level="INFO", format="{message}")
    with redirect_stderr(io.StringIO()) as stderr:
        configure_global_log_level("ERROR")
        logger.info("CUSTOM_BEFORE")
        configure_global_log_level("ERROR")
        logger.info("CUSTOM_AFTER")
        logger.error("DEFAULT_ERROR")
    logger.remove(custom_sink)
    print(json.dumps({"custom": custom_stream.getvalue(), "stderr": stderr.getvalue()}))
else:
    with redirect_stderr(io.StringIO()) as stderr:
        configure_global_log_level("ERROR")
        configure_global_log_level()
        logger.info("INFO_SHOULD_NOT_APPEAR")
        logger.error("ERROR_SHOULD_APPEAR")
    print(json.dumps({"stderr": stderr.getvalue()}))
"""


def _run_script(mode: str, *args: str) -> dict[str, str]:
    """在独立进程中执行日志行为脚本，避免污染当前测试进程的全局 Loguru 状态。"""
    completed = subprocess.run(
        [sys.executable, "-c", _SCRIPT, mode, *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize(("level", "expected_debug"), [("info", False), ("debug", True)])
def test_global_log_level_filters_default_stderr_sink(level: str, expected_debug: bool) -> None:
    """验证显式级别会过滤 Loguru 默认 stderr sink。"""
    output = _run_script("filter", level)

    assert ("DEBUG_MARKER" in output["stderr"]) is expected_debug
    assert "INFO_MARKER" in output["stderr"]


def test_repeated_configuration_preserves_custom_sink() -> None:
    """验证幂等配置不会移除宿主显式添加的自定义 sink。"""
    output = _run_script("custom-sink")

    assert output["custom"].count("CUSTOM_BEFORE") == 1
    assert output["custom"].count("CUSTOM_AFTER") == 1
    assert "DEFAULT_ERROR" in output["stderr"]


def test_default_call_reads_global_environment_config() -> None:
    """验证未显式传级别时读取 MINERU_LOG_LEVEL 对应的全局配置。"""
    script = (
        "import io, json; from contextlib import redirect_stderr; "
        "from loguru import logger; "
        "from mineru.utils.logger import configure_global_log_level; "
        "stream = io.StringIO(); "
        "redirect = redirect_stderr(stream); redirect.__enter__(); "
        "configure_global_log_level(); logger.info('HIDDEN'); logger.error('VISIBLE'); "
        "redirect.__exit__(None, None, None); "
        "print(json.dumps({'stderr': stream.getvalue()}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
        env={**os.environ, "MINERU_LOG_LEVEL": "error"},
    )
    output = json.loads(completed.stdout)

    assert "HIDDEN" not in output["stderr"]
    assert "VISIBLE" in output["stderr"]


def test_explicit_level_persists_across_default_calls() -> None:
    """验证局部显式级别不会被后续默认配置回退。"""
    output = _run_script("explicit-persist")

    assert "INFO_SHOULD_NOT_APPEAR" not in output["stderr"]
    assert "ERROR_SHOULD_APPEAR" in output["stderr"]


def test_helper_module_has_no_runtime_side_effect() -> None:
    """验证导入日志工具模块不会修改 Loguru sink。"""
    script = (
        "from mineru.utils import logger as module; from loguru import logger; print(json.dumps(list(logger._core.handlers)))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", f"import json; {script}"],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )

    assert json.loads(completed.stdout) == [0]
