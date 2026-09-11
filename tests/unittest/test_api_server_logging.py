"""验证 API 启动入口的日志级别、配置隔离和实际输出。"""

from __future__ import annotations

import json
import subprocess
import sys
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner
from typer.testing import CliRunner as TyperRunner

from mineru.kit.commands import api_server as kit_api
from mineru.kit.main import app as kit_app
from mineru.parser import api_server


def _invoke_entrypoint(monkeypatch: pytest.MonkeyPatch, entrypoint: str, args: list[str]) -> int:
    """调用三个正式入口的真实参数解析链，同时避免启动真实服务。"""
    if entrypoint == "module":
        return CliRunner().invoke(api_server.main, args).exit_code
    if entrypoint == "kit":
        return TyperRunner().invoke(kit_app, ["api-server", *args]).exit_code
    monkeypatch.setattr(sys, "argv", ["mineru-api", *args])
    with pytest.raises(SystemExit) as caught:
        kit_api.main()
    return int(caught.value.code)


@pytest.mark.parametrize("entrypoint", ["module", "kit", "alias"])
@pytest.mark.parametrize("level", [None, "critical", "ERROR", "Warning", "info", "debug", "TRACE"])
def test_log_level_reaches_server(monkeypatch: pytest.MonkeyPatch, entrypoint: str, level: str | None) -> None:
    """默认级别与大小写混合参数均传递到服务配置，且不进入应用业务参数。"""
    application = object()
    create_app = MagicMock(return_value=application)
    config = MagicMock(return_value=object())
    server = SimpleNamespace(run=MagicMock(), should_exit=False)
    monkeypatch.setattr(api_server, "create_app", create_app)
    monkeypatch.setattr(api_server.uvicorn, "Config", config)
    monkeypatch.setattr(api_server.uvicorn, "Server", MagicMock(return_value=server))
    monkeypatch.setattr(api_server.ManagedProcessControlWatcher, "from_environment", lambda callback: None)

    assert _invoke_entrypoint(monkeypatch, entrypoint, [] if level is None else ["--log-level", level]) == 0
    expected = (level or "info").lower()
    config.assert_called_once()
    assert config.call_args.args == (application,)
    assert config.call_args.kwargs["log_level"] == expected
    assert (
        config.call_args.kwargs["log_config"]["loggers"][api_server.logger.name]["level"]
        == (api_server.uvicorn.config.LOG_LEVELS[expected])
    )
    assert "log_level" not in create_app.call_args.kwargs
    server.run.assert_called_once()


@pytest.mark.parametrize("entrypoint", ["module", "kit", "alias"])
def test_invalid_log_level_fails_before_initialization(monkeypatch: pytest.MonkeyPatch, entrypoint: str) -> None:
    """非法级别在建立应用、加载模型或创建服务之前失败。"""
    create_app = MagicMock()
    config = MagicMock()
    monkeypatch.setattr(api_server, "create_app", create_app)
    monkeypatch.setattr(api_server.uvicorn, "Config", config)
    assert _invoke_entrypoint(monkeypatch, entrypoint, ["--log-level", "silent"]) != 0
    create_app.assert_not_called()
    config.assert_not_called()


@pytest.mark.parametrize("level", ["info", "warning", "error"])
def test_service_log_output_is_filtered_without_changing_model_logs(level: str) -> None:
    """在独立进程验证真实日志输出、重复配置以及 root、模型日志和 tqdm 的隔离。"""
    script = r"""
import io
import json
import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from copy import deepcopy

import uvicorn
from loguru import logger as model_logger
from tqdm import tqdm
from mineru.parser.api_server import _build_server_log_config

level = sys.argv[1]
root = logging.getLogger()
root_stream = io.StringIO()
root_handler = logging.StreamHandler(root_stream)
root.handlers = [root_handler]
root.setLevel(logging.DEBUG)
model_stream = io.StringIO()
model_sink = model_logger.add(model_stream, level="INFO", format="{message}")
before = deepcopy(uvicorn.config.LOGGING_CONFIG)
stdout, stderr, progress = io.StringIO(), io.StringIO(), io.StringIO()
with redirect_stdout(stdout), redirect_stderr(stderr):
    for _ in range(2):
        uvicorn.Config(object(), log_level=level, log_config=_build_server_log_config(level))
    for name in ("uvicorn", "uvicorn.error", "uvicorn.asgi", "mineru.parser.api_server"):
        service_logger = logging.getLogger(name)
        service_logger.info("%s INFO_MARKER", name)
        service_logger.warning("%s WARNING_MARKER", name)
        service_logger.error("%s ERROR_MARKER", name)
    access_logger = logging.getLogger("uvicorn.access")
    for severity in (logging.INFO, logging.WARNING, logging.ERROR):
        access_logger.log(severity, '%s - "%s %s HTTP/%s" %d', "127.0.0.1", "GET", f"/access-{severity}", "1.1", 200)
    logging.getLogger("mineru.model.probe").info("STDLIB_MODEL_INFO")
    root.debug("ROOT_DEBUG")
    model_logger.info("LOGURU_MODEL_INFO")
    with tqdm(total=1, desc="VLM Predict", file=progress) as bar:
        bar.update(1)
assert uvicorn.config.LOGGING_CONFIG == before
assert root.handlers == [root_handler] and root.level == logging.DEBUG
model_logger.remove(model_sink)
print(json.dumps({"stdout": stdout.getvalue(), "stderr": stderr.getvalue(),
                 "model": model_stream.getvalue(), "root": root_stream.getvalue(), "progress": progress.getvalue()}))
"""
    completed = subprocess.run([sys.executable, "-c", script, level], capture_output=True, text=True, check=True, timeout=30)
    output: dict[str, Any] = json.loads(completed.stdout)
    assert output["stderr"].count("INFO_MARKER") == (4 if level == "info" else 0)
    assert output["stderr"].count("WARNING_MARKER") == (0 if level == "error" else 4)
    assert output["stderr"].count("ERROR_MARKER") == 4
    assert ("/access-20" in output["stdout"]) is (level == "info")
    assert ("/access-30" in output["stdout"]) is (level != "error")
    assert output["stdout"].count("/access-40") == 1
    assert "STDLIB_MODEL_INFO" in output["root"]
    assert "ROOT_DEBUG" in output["root"]
    assert "LOGURU_MODEL_INFO" in output["model"]
    assert "VLM Predict" in output["progress"]


def test_log_configuration_is_independent() -> None:
    """修改一次生成的配置不会污染 Uvicorn 模板或后续服务配置。"""
    original = api_server.uvicorn.config.LOGGING_CONFIG
    config = api_server._build_server_log_config("warning")
    config["formatters"]["default"]["fmt"] = "changed"
    config["loggers"]["uvicorn"]["level"] = 1
    fresh = api_server._build_server_log_config("info")
    assert fresh["formatters"]["default"]["fmt"] == original["formatters"]["default"]["fmt"]
    assert fresh["loggers"]["uvicorn"]["level"] == 20
    assert "root" not in fresh
    assert fresh["disable_existing_loggers"] is False
