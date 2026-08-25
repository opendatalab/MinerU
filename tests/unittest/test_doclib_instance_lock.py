from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from mineru.config import PatchedConfig
from mineru.doclib import app as doclib_app
from mineru.doclib import instance_lock
from mineru.doclib.endpoint import write_endpoint_file


def test_default_doclib_lock_path_is_fixed_under_mineru_home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(instance_lock, "_mineru_home", lambda: str(tmp_path))

    assert instance_lock.default_doclib_lock_path() == tmp_path / "doclib.lock"


def test_doclib_home_lock_blocks_another_holder_and_releases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(instance_lock, "_mineru_home", lambda: str(tmp_path))

    with instance_lock.doclib_home_lock() as lock_path:
        assert lock_path == tmp_path / "doclib.lock"
        assert lock_path.is_file()
        with pytest.raises(instance_lock.DoclibLockUnavailable) as exc_info:
            with instance_lock.doclib_home_lock():
                pass

    assert exc_info.value.args == ()
    with instance_lock.doclib_home_lock():
        pass


def test_doclib_home_lock_is_released_after_owner_is_killed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    script = """
from mineru.doclib.instance_lock import doclib_home_lock
with doclib_home_lock():
    print("locked", flush=True)
    input()
"""
    env = os.environ.copy()
    env["MINERU_HOME"] = str(tmp_path)
    monkeypatch.setenv("MINERU_HOME", str(tmp_path))
    proc = subprocess.Popen(
        [sys.executable, "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == "locked"
        with pytest.raises(instance_lock.DoclibLockUnavailable):
            with instance_lock.doclib_home_lock():
                pass
    finally:
        proc.kill()
        proc.wait(timeout=5)

    with instance_lock.doclib_home_lock():
        pass


def test_doclib_main_exits_before_startup_when_home_lock_is_held(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(instance_lock, "_mineru_home", lambda: str(tmp_path))
    endpoint_path = tmp_path / "doclib.endpoint.json"
    socket_path = tmp_path / "doclib.sock"
    write_endpoint_file(endpoint_path, pid=12345, server_id="server-123", transports=[])
    endpoint_contents = endpoint_path.read_text(encoding="utf-8")
    socket_path.write_text("owned socket", encoding="utf-8")
    cfg = PatchedConfig(
        doclib={
            "endpoint_path": str(endpoint_path),
            "uds": {"path": str(socket_path)},
        }
    )
    monkeypatch.setattr(doclib_app, "config", cfg)
    monkeypatch.setattr(instance_lock, "default_endpoint_path", lambda: str(endpoint_path))

    with instance_lock.doclib_home_lock():
        with pytest.raises(SystemExit) as exc_info:
            doclib_app.main()

    message = str(exc_info.value)
    assert message == f"MinerU home [{tmp_path}] is currently owned by another doclib server process (reported PID 12345)."
    assert "doclib.lock" not in message
    assert endpoint_path.read_text(encoding="utf-8") == endpoint_contents
    assert socket_path.read_text(encoding="utf-8") == "owned socket"


def test_run_server_writes_owner_endpoint_before_binding_transport(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    endpoint_path = tmp_path / "doclib.endpoint.json"
    cfg = PatchedConfig(
        doclib={
            "endpoint_path": str(endpoint_path),
            "uds": {"enabled": False, "path": str(tmp_path / "doclib.sock")},
            "tcp": {"enabled": True},
        }
    )
    app = SimpleNamespace(state=SimpleNamespace(doclib_state=SimpleNamespace(server_id="server-123")))
    writes: list[dict[str, object]] = []

    monkeypatch.setattr(doclib_app, "create_app", lambda cfg: app)
    monkeypatch.setattr(doclib_app.uvicorn, "Config", lambda *args, **kwargs: object())
    monkeypatch.setattr(doclib_app.uvicorn, "Server", lambda cfg: object())

    def _write_endpoint(path: str, *, pid: int, server_id: str, transports: list[object]) -> None:
        writes.append({"path": path, "pid": pid, "server_id": server_id, "transports": transports})

    def _fail_bind(*args: object, **kwargs: object) -> None:
        assert writes == [
            {
                "path": str(endpoint_path),
                "pid": os.getpid(),
                "server_id": "server-123",
                "transports": [],
            }
        ]
        raise RuntimeError("bind failed")

    monkeypatch.setattr(doclib_app, "write_endpoint_file", _write_endpoint)
    monkeypatch.setattr(doclib_app, "_bind_tcp_socket", _fail_bind)

    with pytest.raises(RuntimeError, match="bind failed"):
        doclib_app._run_server(cfg)
