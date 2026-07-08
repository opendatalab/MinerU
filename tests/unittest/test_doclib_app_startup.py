import asyncio
import logging
import subprocess
import tomllib
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from loguru import logger as loguru_logger

from mineru.config import LogConfig, PatchedConfig, config as mineru_config
from mineru.doclib import app as doclib_app
from mineru.doclib.app import _assert_required_schema
from mineru.doclib.core.db import DatabaseManager
from mineru.doclib.server import _tail_log, _write_temp_asset
from mineru.parser import tier as parser_tier
from mineru.version import __version__


def _clear_test_loggers() -> None:
    for name in ("mineru", "uvicorn", "uvicorn.error", "uvicorn.access", "py.warnings"):
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        logger.propagate = True


def test_required_schema_check_fails_before_migration_and_passes_after(tmp_path) -> None:
    async def _run() -> None:
        db = DatabaseManager(str(tmp_path / "doclib.db"))

        with pytest.raises(RuntimeError, match="missing tables"):
            await _assert_required_schema(db)

        await db.initialize()
        await _assert_required_schema(db)

        row = await db.fetchone("SELECT name FROM sqlite_master WHERE type='table' AND name='parses'")
        assert row == {"name": "parses"}

    asyncio.run(_run())


def test_doclib_runtime_dependencies_are_in_base_install() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    dependency_names = {dependency.split(">", 1)[0].split("=", 1)[0].lower() for dependency in dependencies}

    assert "aiosqlite" in dependency_names
    assert "packaging" in dependency_names
    assert "watchfiles" in dependency_names


def test_pdftext_dependency_is_capped_below_pagechars_api() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    pdftext_dependencies = [dependency for dependency in dependencies if dependency.lower().startswith("pdftext")]

    assert pdftext_dependencies == ["pdftext>=0.6.3,<0.7.0"]


def test_standard_extra_includes_preflight_runtime_dependencies() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    standard_dependencies = pyproject["project"]["optional-dependencies"]["standard"]
    dependency_names = {dependency.split(">", 1)[0].split("=", 1)[0].lower() for dependency in standard_dependencies}
    module_to_distribution = {
        "ftfy": "ftfy",
        "pyclipper": "pyclipper",
        "shapely": "shapely",
        "six": "six",
        "torch": "torch",
        "torchvision": "torchvision",
        "transformers": "transformers",
    }

    missing = [
        module_name
        for module_name in parser_tier.required_modules_for_tier("standard")
        if module_to_distribution[module_name] not in dependency_names
    ]

    assert missing == []


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("parse_server.local.mode", "totally-invalid-mode"),
        ("parse_server.local.managed_tier", "ultra"),
        ("parse_server.remote.url", "ftp://example.com/api"),
        ("parse_server.local.self_hosted_url", "not-a-url"),
    ],
)
def test_config_set_rejects_invalid_known_config_values(key: str, value: str, monkeypatch, tmp_path) -> None:
    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)

    cfg = PatchedConfig(doclib={"data_dir": str(tmp_path), "sqlite": {"path": str(tmp_path / "doclib.db")}})
    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.put(f"/api/v1/configs/{key}", json={"value": value})
        config_response = client.get(f"/api/v1/configs/{key}")

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "invalid_config_value"
    assert payload["error"]["param"] == "value"
    assert config_response.json()["source"] == "default"


def test_config_set_managed_mode_preflights_managed_tier_dependencies(monkeypatch, tmp_path) -> None:
    def _skip_background_task(*args, **kwargs):
        return None

    def _import_module(module_name: str):
        if module_name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return object()

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)
    monkeypatch.setattr(parser_tier.importlib, "import_module", _import_module)
    monkeypatch.setattr(parser_tier.importlib_metadata, "packages_distributions", lambda: {"mineru": ["mineru-next-dev"]})

    cfg = PatchedConfig(doclib={"data_dir": str(tmp_path), "sqlite": {"path": str(tmp_path / "doclib.db")}})
    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.put("/api/v1/configs/parse_server.local.mode", json={"value": "managed"})
        config_response = client.get("/api/v1/configs/parse_server.local.mode")

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "parse_server_dependency_missing"
    assert payload["error"]["param"] == "parse_server.local.mode"
    assert "torch" in payload["error"]["message"]
    assert "pip install 'mineru-next-dev[high]'" in payload["error"]["message"]
    assert config_response.json()["value"] == "disabled"


def test_config_set_managed_tier_preflights_even_when_mode_is_disabled(monkeypatch, tmp_path) -> None:
    def _skip_background_task(*args, **kwargs):
        return None

    def _import_module(module_name: str):
        if module_name == "mlx":
            raise ModuleNotFoundError("No module named 'mlx'")
        return object()

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)
    monkeypatch.setattr(parser_tier.importlib, "import_module", _import_module)
    monkeypatch.setattr(parser_tier.importlib_metadata, "packages_distributions", lambda: {"mineru": ["mineru-next-dev"]})
    monkeypatch.setattr(parser_tier.sys, "platform", "darwin")

    cfg = PatchedConfig(doclib={"data_dir": str(tmp_path), "sqlite": {"path": str(tmp_path / "doclib.db")}})
    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.put("/api/v1/configs/parse_server.local.managed_tier", json={"value": "extra_high"})
        config_response = client.get("/api/v1/configs/parse_server.local.managed_tier")

    assert response.status_code == 400
    payload = response.json()
    assert payload["error"]["code"] == "parse_server_dependency_missing"
    assert payload["error"]["param"] == "parse_server.local.managed_tier"
    assert "mlx" in payload["error"]["message"]
    assert "pip install 'mineru-next-dev[extra_high]'" in payload["error"]["message"]
    assert config_response.json()["value"] == "high"


def test_background_task_crash_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    async def _boom() -> None:
        raise RuntimeError("boom")

    async def _run() -> None:
        task = asyncio.create_task(_boom())
        await asyncio.sleep(0)
        doclib_app._log_background_task_result("test-task", task)

    with caplog.at_level(logging.ERROR):
        asyncio.run(_run())

    assert "Background task test-task crashed: boom" in caplog.text


def test_background_tasks_are_cancelled_and_awaited_on_shutdown() -> None:
    async def _run() -> None:
        state = doclib_app.AppState()
        state.db_ready = asyncio.Event()

        async def _blocked() -> None:
            await asyncio.Event().wait()

        task = doclib_app._create_background_task(state, "blocked", _blocked)
        await asyncio.sleep(0)

        await doclib_app._cancel_background_tasks(state)

        assert task.cancelled()
        assert task.done()
        assert state.background_tasks == []

    asyncio.run(_run())


def test_pending_loop_tasks_are_cancelled_before_loop_close() -> None:
    loop = asyncio.new_event_loop()
    try:
        task = loop.create_task(asyncio.Event().wait())

        doclib_app._cancel_pending_loop_tasks(loop)

        assert task.cancelled()
        assert task.done()
    finally:
        loop.close()


def test_bind_tcp_socket_tries_configured_port_range(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_calls: list[tuple[str, int]] = []

    class _Socket:
        def __init__(self) -> None:
            self.bound_port: int | None = None

        def setsockopt(self, *args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            bind_calls.append(address)
            host, port = address
            if port in (15980, 15981):
                raise OSError(doclib_app.errno.EADDRINUSE, "in use")
            self.bound_port = port

        def getsockname(self) -> tuple[str, int]:
            assert self.bound_port is not None
            return ("127.0.0.1", self.bound_port)

    monkeypatch.setattr(doclib_app.socket, "socket", lambda *args, **kwargs: _Socket())

    tcp_sock, port = doclib_app._bind_tcp_socket("127.0.0.1", 15980, strict_port=False, port_probe_count=3)

    assert isinstance(tcp_sock, _Socket)
    assert port == 15982
    assert bind_calls == [("127.0.0.1", 15980), ("127.0.0.1", 15981), ("127.0.0.1", 15982)]


def test_bind_tcp_socket_strict_port_does_not_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    bind_calls: list[tuple[str, int]] = []

    class _Socket:
        def setsockopt(self, *args: object) -> None:
            return None

        def bind(self, address: tuple[str, int]) -> None:
            bind_calls.append(address)
            raise OSError(doclib_app.errno.EADDRINUSE, "in use")

    monkeypatch.setattr(doclib_app.socket, "socket", lambda *args, **kwargs: _Socket())

    with pytest.raises(OSError):
        doclib_app._bind_tcp_socket("127.0.0.1", 15980, strict_port=True, port_probe_count=3)

    assert bind_calls == [("127.0.0.1", 15980)]


def test_setup_logging_routes_application_logs_to_rotating_file_without_stderr_duplication(tmp_path: Path) -> None:
    _clear_test_loggers()
    log_path = tmp_path / "doclib.log"
    access_log_path = tmp_path / "doclib.access.log"
    try:
        doclib_app._setup_logging(LogConfig(app_path=str(log_path), access_path=str(access_log_path)))

        mineru_logger = logging.getLogger("mineru")
        uvicorn_error_logger = logging.getLogger("uvicorn.error")
        uvicorn_access_logger = logging.getLogger("uvicorn.access")
        warnings_logger = logging.getLogger("py.warnings")

        mineru_rotating_handlers = [handler for handler in mineru_logger.handlers if isinstance(handler, RotatingFileHandler)]
        mineru_plain_stream_handlers = [
            handler
            for handler in mineru_logger.handlers
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, RotatingFileHandler)
        ]

        assert len(mineru_rotating_handlers) == 1
        assert mineru_rotating_handlers[0].baseFilename == str(log_path)
        assert mineru_plain_stream_handlers == []
        assert mineru_logger.propagate is False
        assert [handler.baseFilename for handler in uvicorn_error_logger.handlers if isinstance(handler, RotatingFileHandler)] == [
            str(log_path)
        ]
        assert [handler.baseFilename for handler in warnings_logger.handlers if isinstance(handler, RotatingFileHandler)] == [
            str(log_path)
        ]
        assert [handler.baseFilename for handler in uvicorn_access_logger.handlers if isinstance(handler, RotatingFileHandler)] == [
            str(access_log_path)
        ]
    finally:
        _clear_test_loggers()


def test_setup_logging_routes_loguru_records_to_application_log(tmp_path: Path) -> None:
    _clear_test_loggers()
    log_path = tmp_path / "doclib.log"
    access_log_path = tmp_path / "doclib.access.log"
    try:
        doclib_app._setup_logging(LogConfig(app_path=str(log_path), access_path=str(access_log_path)))

        patched_logger = loguru_logger.patch(lambda record: record.update(name="mineru.test_loguru"))
        patched_logger.error("loguru bridged message")

        for handler in logging.getLogger("mineru").handlers:
            handler.flush()

        content = log_path.read_text(encoding="utf-8")
        assert "mineru.test_loguru" in content
        assert "loguru bridged message" in content
    finally:
        _clear_test_loggers()


def test_server_status_reports_configured_socket_path(monkeypatch, tmp_path) -> None:
    app_log_path = tmp_path / "doclib.log"
    access_log_path = tmp_path / "doclib.access.log"
    stdout_log_path = tmp_path / "doclib.stdout.log"
    stderr_log_path = tmp_path / "doclib.stderr.log"
    parse_server_stdout_log_path = tmp_path / "doclib.parse-server.stdout.log"
    parse_server_stderr_log_path = tmp_path / "doclib.parse-server.stderr.log"
    for prefix, path, count in (
        ("app", app_log_path, 30),
        ("access", access_log_path, 12),
        ("stdout", stdout_log_path, 12),
        ("stderr", stderr_log_path, 12),
        ("parse-stdout", parse_server_stdout_log_path, 12),
        ("parse-stderr", parse_server_stderr_log_path, 12),
    ):
        path.write_text("".join(f"{prefix}-{idx}\n" for idx in range(count)), encoding="utf-8")

    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(app_log_path),
                "access_path": str(access_log_path),
                "stdout_path": str(stdout_log_path),
                "stderr_path": str(stderr_log_path),
                "parse_server_stdout_path": str(parse_server_stdout_log_path),
                "parse_server_stderr_path": str(parse_server_stderr_log_path),
            },
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)

    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.get("/api/v1/server/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["mineru_home"]
    assert payload["version"] == __version__
    assert isinstance(payload["python_version"], str)
    assert payload["socket_path"] == str(tmp_path / "doclib.sock")
    assert payload["sqlite_path"] == str(tmp_path / "doclib.db")
    assert payload["log_path"] == str(tmp_path / "doclib.log")
    assert payload["access_log_path"] == str(access_log_path)
    assert payload["stdout_log_path"] == str(stdout_log_path)
    assert payload["stderr_log_path"] == str(stderr_log_path)
    assert payload["tcp"] == {"enabled": cfg.doclib.resolved_tcp_enabled, "host": cfg.doclib.tcp.host, "port": None}
    assert payload["app_logs"] == [f"app-{idx}\n" for idx in range(5, 30)]
    assert payload["access_logs"] == [f"access-{idx}\n" for idx in range(2, 12)]
    assert payload["stdout_logs"] == [f"stdout-{idx}\n" for idx in range(2, 12)]
    assert payload["stderr_logs"] == [f"stderr-{idx}\n" for idx in range(2, 12)]
    assert payload["parse_server_stdout_logs"] == [f"parse-stdout-{idx}\n" for idx in range(2, 12)]
    assert payload["parse_server_stderr_logs"] == [f"parse-stderr-{idx}\n" for idx in range(2, 12)]


def test_doclib_app_uses_lifespan_instead_of_deprecated_event_handlers(monkeypatch, tmp_path) -> None:
    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(tmp_path / "doclib.log"),
                "access_path": str(tmp_path / "doclib.access.log"),
            },
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)

    app = doclib_app.create_app(cfg)

    assert app.router.on_startup == []
    assert app.router.on_shutdown == []
    with TestClient(app) as client:
        response = client.get("/api/v1/server/status")

    assert response.status_code == 200


def test_managed_parse_server_startup_writes_stdout_and_stderr_logs(monkeypatch, tmp_path) -> None:
    db = DatabaseManager(str(tmp_path / "doclib.db"))
    asyncio.run(db.initialize())
    asyncio.run(db.execute("INSERT INTO config (key, value) VALUES (?, ?)", ("parse_server.local.mode", "managed")))

    parse_stdout_log_path = tmp_path / "doclib.parse-server.stdout.log"
    parse_stderr_log_path = tmp_path / "doclib.parse-server.stderr.log"
    popen_calls: list[dict[str, object]] = []

    class _Proc:
        pid = 12345

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout: int) -> None:
            return None

    def _popen(*args: object, **kwargs: object) -> _Proc:
        popen_calls.append({"args": args, "kwargs": kwargs})
        cmd = args[0]
        assert "--host" in cmd
        assert cmd[cmd.index("--host") + 1] == "127.0.0.2"
        assert "--port" in cmd
        assert cmd[cmd.index("--port") + 1] == "16582"
        assert kwargs["stdout"] is not subprocess.DEVNULL
        assert kwargs["stderr"] is not subprocess.DEVNULL
        assert kwargs["stdout"] is not kwargs["stderr"]
        assert kwargs["stdin"] is subprocess.PIPE
        assert kwargs["env"]["MINERU_MANAGED_PARSE_SERVER"] == "1"
        kwargs["stdout"].write("parse stdout\n")
        kwargs["stdout"].flush()
        kwargs["stderr"].write("parse stderr\n")
        kwargs["stderr"].flush()
        return _Proc()

    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(tmp_path / "doclib.log"),
                "access_path": str(tmp_path / "doclib.access.log"),
                "parse_server_stdout_path": str(parse_stdout_log_path),
                "parse_server_stderr_path": str(parse_stderr_log_path),
            },
            "managed_parse_server": {
                "host": "127.0.0.2",
                "port": 16580,
                "port_probe_count": 3,
            },
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.select_available_managed_port", lambda *args, **kwargs: 16582)
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.subprocess.Popen", _popen)

    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.get("/api/v1/server/status")

    assert response.status_code == 200
    assert popen_calls
    assert parse_stdout_log_path.read_text(encoding="utf-8").endswith("parse stdout\n")
    assert parse_stderr_log_path.read_text(encoding="utf-8").endswith("parse stderr\n")


def test_managed_parse_server_startup_clears_invalid_tier_override(monkeypatch, tmp_path) -> None:
    db = DatabaseManager(str(tmp_path / "doclib.db"))
    asyncio.run(db.initialize())
    asyncio.run(db.execute("INSERT INTO config (key, value) VALUES (?, ?)", ("parse_server.local.mode", "managed")))
    asyncio.run(
        db.execute("INSERT INTO config (key, value) VALUES (?, ?)", ("parse_server.local.managed_tier", "standard"))
    )

    class _Proc:
        pid = 12345

        def poll(self) -> None:
            return None

        def wait(self, timeout: int) -> None:
            return None

    def _popen(*args: object, **kwargs: object) -> _Proc:
        cmd = args[0]
        assert "--tier" in cmd
        assert cmd[cmd.index("--tier") + 1] == "high"
        return _Proc()

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.select_available_managed_port", lambda *args, **kwargs: 16582)
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.subprocess.Popen", _popen)

    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(tmp_path / "doclib.log"),
                "access_path": str(tmp_path / "doclib.access.log"),
                "parse_server_stdout_path": str(tmp_path / "parse.stdout.log"),
                "parse_server_stderr_path": str(tmp_path / "parse.stderr.log"),
            },
            "managed_parse_server": {"port": 16580},
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    with TestClient(doclib_app.create_app(cfg)) as client:
        config_response = client.get("/api/v1/configs/parse_server.local.managed_tier")
        status_response = client.get("/api/v1/server/status")

    assert config_response.json()["value"] == "high"
    assert status_response.json()["parse_server"]["local"]["managed_tier"] == "high"


def test_managed_parse_server_shutdown_uses_health_proc(monkeypatch, tmp_path) -> None:
    db = DatabaseManager(str(tmp_path / "doclib.db"))
    asyncio.run(db.initialize())
    asyncio.run(db.execute("INSERT INTO config (key, value) VALUES (?, ?)", ("parse_server.local.mode", "managed")))

    events: list[str] = []

    class _Stdin:
        closed = False

        def close(self) -> None:
            events.append("stdin.close")
            self.closed = True

    class _Proc:
        pid = 12345
        stdin = _Stdin()

        def poll(self) -> None:
            return None

        def wait(self, timeout: int) -> None:
            events.append(f"wait:{timeout}")

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")

    def _popen(*args: object, **kwargs: object) -> _Proc:
        return _Proc()

    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(tmp_path / "doclib.log"),
                "access_path": str(tmp_path / "doclib.access.log"),
                "parse_server_stdout_path": str(tmp_path / "parse.stdout.log"),
                "parse_server_stderr_path": str(tmp_path / "parse.stderr.log"),
            },
            "managed_parse_server": {"port": 16580},
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.select_available_managed_port", lambda *args, **kwargs: 16582)
    monkeypatch.setattr("mineru.doclib.background.parse_server_health.subprocess.Popen", _popen)

    with TestClient(doclib_app.create_app(cfg)) as client:
        state = client.app.state.doclib_state
        assert not hasattr(state, "parse_server_proc")
        assert client.get("/api/v1/server/status").status_code == 200

    assert events == ["stdin.close", "wait:10"]


def test_startup_resets_running_scans_to_failed(monkeypatch, tmp_path) -> None:
    db = DatabaseManager(str(tmp_path / "doclib.db"))
    asyncio.run(db.initialize())
    asyncio.run(
        db.execute(
            "INSERT INTO scans (path, kind, source, status, locked_at, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (str(tmp_path / "watch-root"), "watch", "watch", "running", 123, 1000, 1000),
        )
    )

    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(tmp_path / "doclib.log"),
                "access_path": str(tmp_path / "doclib.access.log"),
            },
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)

    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.get("/api/v1/server/status")

    assert response.status_code == 200
    row = asyncio.run(db.fetchone("SELECT status, locked_at, error_code FROM scans"))
    assert row == {"status": "failed", "locked_at": None, "error_code": "scan_interrupted"}


def test_telemetry_observation_route_uses_context_headers(monkeypatch, tmp_path) -> None:
    cfg = PatchedConfig(
        doclib={
            "data_dir": str(tmp_path),
            "sqlite": {"path": str(tmp_path / "doclib.db")},
            "log": {
                "app_path": str(tmp_path / "doclib.log"),
                "access_path": str(tmp_path / "doclib.access.log"),
            },
            "uds": {"path": str(tmp_path / "doclib.sock")},
        }
    )

    def _skip_background_task(*args, **kwargs):
        return None

    monkeypatch.setattr(doclib_app, "_create_background_task", _skip_background_task)

    with TestClient(doclib_app.create_app(cfg)) as client:
        response = client.post(
            "/api/v1/observations",
            headers={"X-MinerU-Telemetry-Source": "cli", "X-MinerU-Telemetry-Caller": "agent"},
            json={"observations": [{"metric_name": "parse.request.count"}]},
        )
        preview = client.get("/api/v1/telemetry/preview")

    assert response.status_code == 200
    assert response.json() == {"accepted": 1}
    assert preview.status_code == 200
    metrics = preview.json()["body"]["metrics"]
    assert metrics == [{"name": "parse.request.count", "value": 1, "dimensions": {"caller": "agent", "source": "cli"}}]


def test_write_temp_asset_uses_temp_read_assets_directory(tmp_path: Path) -> None:
    asset = _write_temp_asset(
        str(tmp_path),
        "abc1234",
        "png",
        b"image-bytes",
        mime_type="image/png",
        width=10,
        height=10,
    )

    asset_path = Path(asset.path)
    assert asset_path.parent == tmp_path / "temp" / "read-assets"


def test_tail_log_uses_current_config_default_when_data_dir_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_path = tmp_path / "doclib.log"
    log_path.write_text("line-1\nline-2\n", encoding="utf-8")
    monkeypatch.setattr(mineru_config.doclib, "log", LogConfig(app_path=str(log_path)))

    assert _tail_log("") == ["line-1\n", "line-2\n"]
