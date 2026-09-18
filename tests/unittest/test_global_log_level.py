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


def test_parser_entry_configures_before_docvortex(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """验证直接实例化 MinerUParser 时，日志配置先于 prepare 阶段的 docvortex 调用。"""
    import mineru.parser.mineru_parser as mineru_parser_module

    calls: list[str] = []
    monkeypatch.setattr(
        mineru_parser_module,
        "configure_global_log_level",
        lambda: calls.append("configure"),
    )

    def fake_read_source_properties(*args: object, **kwargs: object) -> None:
        calls.append("docvortex")

    monkeypatch.setattr(mineru_parser_module, "read_source_properties", fake_read_source_properties)

    def stop_after_prepare(*args: object, **kwargs: object) -> None:
        raise RuntimeError("stop-after-prepare")

    monkeypatch.setattr(mineru_parser_module, "doc_analyze", stop_after_prepare)

    source = tmp_path / "sample.csv"
    source.write_text("a,b\n1,2\n", encoding="utf-8")
    parser = mineru_parser_module.MinerUParser()

    with pytest.raises(RuntimeError, match="stop-after-prepare"):
        parser.parse(source)

    assert calls == ["configure", "docvortex"]


def test_render_entry_configures_global_log_level(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证 render 统一入口在任何渲染分发前完成日志配置。"""
    import mineru.render.api as render_api_module

    calls: list[str] = []
    monkeypatch.setattr(
        render_api_module,
        "configure_global_log_level",
        lambda: calls.append("configure"),
    )

    with pytest.raises(TypeError):
        render_api_module.render(object(), render_api_module.RenderFormat.MARKDOWN)

    assert calls == ["configure"]


@pytest.mark.parametrize(("level", "expected_debug"), [("info", False), ("debug", True)])
def test_configured_level_propagates_to_spawned_children(level: str, expected_debug: bool) -> None:
    """验证配置级别经 LOGURU_LEVEL 传播到之后 spawn 的子进程出厂 sink。"""
    script = (
        "import json, subprocess, sys; "
        "from mineru.utils.logger import configure_global_log_level; "
        f"configure_global_log_level({level!r}); "
        "import os; "
        "child = subprocess.run("
        "[sys.executable, '-c', "
        "\"from loguru import logger; logger.debug('CHILD_DEBUG'); logger.info('CHILD_INFO')\"], "
        "capture_output=True, text=True, check=True); "
        "print(json.dumps({'env': os.environ.get('LOGURU_LEVEL'), 'child_stderr': child.stderr}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    output = json.loads(completed.stdout)

    assert output["env"] == level.upper()
    assert ("CHILD_DEBUG" in output["child_stderr"]) is expected_debug
    assert "CHILD_INFO" in output["child_stderr"]
