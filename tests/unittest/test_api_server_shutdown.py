"""验证 API 服务正常中断、异常传播和三个入口的真实信号退出。"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from mineru.parser import api_server
from mineru.parser.process_control import CONTROL_ENV


@pytest.mark.parametrize("error", [None, KeyboardInterrupt(), RuntimeError("server failed")])
def test_server_run_closes_control_watcher(monkeypatch: pytest.MonkeyPatch, error: BaseException | None) -> None:
    """正常中断应成功退出，真实服务异常继续传播，所有路径均关闭托管控制通道。"""
    watcher = MagicMock()
    server = SimpleNamespace(run=MagicMock(side_effect=error), should_exit=False)
    monkeypatch.setattr(api_server, "create_app", MagicMock(return_value=object()))
    monkeypatch.setattr(api_server.uvicorn, "Config", MagicMock())
    monkeypatch.setattr(api_server.uvicorn, "Server", MagicMock(return_value=server))
    monkeypatch.setattr(api_server.ManagedProcessControlWatcher, "from_environment", lambda callback: watcher)

    result = CliRunner().invoke(api_server.main, ["--tier", "flash"])

    watcher.start.assert_called_once_with()
    watcher.close.assert_called_once_with()
    server.run.assert_called_once_with()
    if isinstance(error, RuntimeError):
        assert result.exit_code != 0
        assert result.exception is error
    else:
        assert result.exit_code == 0, result.output
        assert result.exception is None


@pytest.mark.skipif(sys.platform == "win32", reason="Windows 不支持通过 send_signal 模拟终端 Ctrl+C")
@pytest.mark.parametrize(
    "entrypoint",
    [
        ["-m", "mineru.parser.api_server"],
        ["-c", "from mineru.kit.main import main; main()", "api-server"],
        ["-c", "from mineru.kit.commands.api_server import main; main()"],
    ],
    ids=["module", "kit", "alias"],
)
def test_api_server_sigint_exits_cleanly(tmp_path: Path, entrypoint: list[str]) -> None:
    """启动真实服务并发送 SIGINT，确保 lifespan 完整关闭且不出现退出异常堆栈。"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    env = os.environ.copy()
    env.pop(CONTROL_ENV, None)
    env.update(MINERU_CONFIG=str(config_path), PYTHONUNBUFFERED="1", CI="1")
    log_path = tmp_path / "server.log"
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                *entrypoint,
                "--host",
                "127.0.0.1",
                "--port",
                "0",
                "--tier",
                "flash",
                "--upload-dir",
                str(tmp_path / "uploads"),
            ],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                output = log_path.read_text(encoding="utf-8")
                if "Uvicorn running on" in output:
                    break
                assert process.poll() is None, output
                time.sleep(0.05)
            else:
                pytest.fail(f"API 服务未在预期时间内启动：\n{output}")

            process.send_signal(signal.SIGINT)
            process.wait(timeout=15)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)

    output = log_path.read_text(encoding="utf-8")
    assert "Application shutdown complete." in output
    assert "Finished server process" in output
    assert process.returncode == 0, output
    for marker in ("Traceback", "CancelledError", "KeyboardInterrupt", "Abort"):
        assert marker not in output, output
